"""Explicit context facade for the CIMA Demonstrator."""
from __future__ import annotations

import re
import uuid
import hashlib
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import inspect

from cima_demo.demo.contracts import ContextSnapshot
from cima_demo.demo.context.evidence_marker_registry import filter_citable_items, marker_resolution_status, split_citable_and_auxiliary_items
from cima_demo.demo.lineage.witness_resolver import WitnessLineageResolver
from cima_demo.demo.runtime.journal import DemoRunJournal
from cima_demo.domain.entities import ContextView, Plan, TaskMemory
from cima_demo.domain.ports import RelDBPort
from cima_demo.domain.value_objects import ContextBudget
from cima_demo.memory.service import MemoryService
from cima_demo.retrieval.context_builder import ContextBuilder

if TYPE_CHECKING:  # pragma: no cover
    from cima_demo.geometry.boundary import GeometryCommandsPort, GeometryHintsPort
    from cima_demo.demo.handoff import DemoHandoffService
from cima_demo.witness_backend.ephemeral_runtime import EphemeralRuntimeMirrorPort


@dataclass(slots=True)
class _BoundRun:
    run_id: str
    conversation_id: str
    turn_id: str
    query_text: str


_bound_run: ContextVar[_BoundRun | None] = ContextVar("cima_demo_bound_run", default=None)
_last_snapshot_id: ContextVar[str | None] = ContextVar("cima_demo_last_snapshot_id", default=None)


async def _maybe_call(obj: object, name: str, *args: object, default: Any = None, **kwargs: object) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result



def _norm_context_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


_INTERNAL_MARKER_LITERAL_RE = re.compile(r"\[(S|E|P)[1-9][0-9]*\]")


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _escape_internal_marker_literals(text: str) -> str:
    """Prevent source text from manufacturing CIMA citation markers."""
    return _INTERNAL_MARKER_LITERAL_RE.sub(lambda m: f"⟦{m.group(0)[1:-1]}⟧", text or "")


def _looks_like_current_turn_item(item: dict[str, Any], current_texts: set[str]) -> bool:
    if not current_texts:
        return False
    content = _norm_context_text(item.get("content") or "")
    if content in current_texts:
        return True
    # Guard the publication evidence path: the current OpenAI user message may
    # be persisted as a chat_user C-item before synthesis. It is task input, not
    # source evidence, and must not become a citable S# marker in the same turn.
    actor = str(item.get("actor") or "").lower()
    ref_kind = str(item.get("ref_kind") or "").lower()
    if actor == "user" and ref_kind == "citem" and content in current_texts:
        return True
    return False


def _render_marker_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["CONTEXT"]
    for item in items:
        marker = str(item.get("marker") or "").strip()
        content = _escape_internal_marker_literals(str(item.get("content") or "").strip())
        if marker and content:
            lines.append(f"[{marker}] {content}")
    return "\n\n".join(lines)


def _render_auxiliary_context(items: list[dict[str, Any]]) -> str:
    """Render non-citable context without S#/P# labels.

    Auxiliary material may help disambiguate a task, but it is not admissible as
    factual support.  Therefore we never expose bracketed citation markers here.
    """
    if not items:
        return ""
    lines = [
        "AUXILIARY CONTEXT (NOT CITABLE)",
        "Use only for orientation. Do not cite or use as evidence for factual claims.",
    ]
    for index, item in enumerate(items, start=1):
        content = _escape_internal_marker_literals(str(item.get("content") or "").strip())
        reason = str(item.get("reason_if_not_citable") or "uncitable").strip()
        if content:
            lines.append(f"AUX{index} | reason={reason}\n{content}")
    return "\n\n".join(lines)


def _retokenize_rough(text: str) -> int:
    # Conservative local approximation for evidence artifacts. The model-side
    # tokenizer remains the runtime source of truth when available; this keeps
    # snapshot accounting monotonic after deterministic item filtering.
    return max(0, len(re.findall(r"\w+|[^\w\s]", text or "")))


def _trim_marker_items_to_budget(items: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_tokens <= 0 or not items:
        return [], [dict(item) for item in items]
    kept: list[dict[str, Any]] = []
    trimmed: list[dict[str, Any]] = []
    for item in items:
        candidate = [*kept, dict(item)]
        if _retokenize_rough(_render_marker_context(candidate)) <= max_tokens:
            kept.append(dict(item))
        else:
            # Preserve at least a bounded evidence anchor when a single selected
            # item is larger than the remaining context budget.  This is a
            # deterministic truncation of item content, not a change of claim or
            # citation marker.
            if not kept:
                truncated = _truncate_marker_item_to_budget(dict(item), max_tokens)
                if truncated is not None:
                    kept.append(truncated)
                    continue
            trimmed.append(dict(item))
    return kept, trimmed


def _truncate_marker_item_to_budget(item: dict[str, Any], max_tokens: int) -> dict[str, Any] | None:
    content = str(item.get("content") or "")
    if not content.strip():
        return None
    lo = 0
    hi = len(content)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_text = content[:mid].rstrip()
        if mid < len(content) and candidate_text:
            candidate_text = candidate_text.rstrip(" .,:;-") + " ..."
        candidate = dict(item)
        candidate["content"] = candidate_text
        if candidate_text and _retokenize_rough(_render_marker_context([candidate])) <= max_tokens:
            best = candidate_text
            lo = mid + 1
        else:
            hi = mid - 1
    if not best:
        return None
    out = dict(item)
    full_content = str(item.get("content") or "")
    out["content"] = best
    out["context_truncated_for_budget"] = True
    out["visible_support_scope"] = "truncated_prompt"
    out["visible_char_start"] = 0
    out["visible_char_end"] = len(best)
    out["full_content_sha256"] = _sha256_text(full_content)
    out["visible_content_sha256"] = _sha256_text(best)
    return out


def _visible_marker_support_from_items(items: list[dict[str, Any]], *, rendered_text: str = "", namespace: str = "runtime_context") -> list[dict[str, Any]]:
    """Build prompt-visible support rows anchored to rendered context text.

    This is intentionally stronger than "marker exists in storage": each row records
    the rendered text hash plus char offsets for the exact block handed to the LLM.
    The offsets are relative to the rendered block (`context_pack`, `zoom_block`, etc.).
    """
    out: list[dict[str, Any]] = []
    cursor = 0
    for item in items or []:
        marker = str(item.get("marker") or "").strip()
        if not marker:
            continue
        visible_text = _escape_internal_marker_literals(str(item.get("content") or "").strip())
        if not visible_text:
            continue
        rendered_line = f"[{marker}] {visible_text}"
        start = rendered_text.find(rendered_line, cursor) if rendered_text else -1
        if start < 0 and rendered_text:
            start = rendered_text.find(f"[{marker}]", cursor)
        if start < 0:
            start = int(item.get("visible_char_start") or 0)
            end = int(item.get("visible_char_end") or len(visible_text))
        else:
            end = start + len(rendered_line)
            cursor = end
        truncated = bool(item.get("context_truncated_for_budget"))
        out.append({
            "marker": marker,
            "marker_namespace": namespace,
            "marker_uid": f"{namespace}:{marker}",
            "ref_kind": str(item.get("ref_kind") or ""),
            "ref_id": str(item.get("ref_id") or ""),
            "support_source": namespace,
            "rendered_text_sha256": _sha256_text(rendered_text),
            "visible_text_sha256": _sha256_text(visible_text),
            "visible_text_preview": visible_text[:240],
            "visible_char_count": len(visible_text),
            "context_truncated_for_budget": truncated,
            "visible_support_scope": str(item.get("visible_support_scope") or ("truncated_prompt" if truncated else "full_prompt_item")),
            "full_content_sha256": str(item.get("full_content_sha256") or _sha256_text(str(item.get("content") or ""))),
            "visible_content_sha256": str(item.get("visible_content_sha256") or _sha256_text(visible_text)),
            "prompt_char_start": start,
            "prompt_char_end": end,
            "visible_char_start": start,
            "visible_char_end": end,
            "escaped_marker_literals": bool(_INTERNAL_MARKER_LITERAL_RE.search(str(item.get("content") or ""))),
        })
    return out


def _drop_current_turn_items(context_view: ContextView, current_texts: set[str] | None) -> ContextView:
    normalized_current = {_norm_context_text(text) for text in (current_texts or set()) if _norm_context_text(text)}
    if not normalized_current or not getattr(context_view, "items", None):
        return context_view
    kept: list[dict[str, Any]] = []
    changed = False
    for item in context_view.items:
        materialized = dict(item)
        if _looks_like_current_turn_item(materialized, normalized_current):
            changed = True
            continue
        kept.append(materialized)
    if not changed:
        return context_view
    renumbered: list[dict[str, Any]] = []
    for index, item in enumerate(kept, start=1):
        materialized = dict(item)
        materialized["marker"] = f"S{index}"
        renumbered.append(materialized)
    context_view.items = renumbered
    context_view.citem_ids = [
        str(item.get("ref_id"))
        for item in renumbered
        if str(item.get("ref_kind") or "") == "citem" and item.get("ref_id")
    ]
    context_view.text = _render_marker_context(renumbered)
    context_view.tokens_used = _retokenize_rough(context_view.text)
    return context_view

def _merge_resolution_modes(modes: list[str]) -> str:
    witness = any(mode in {"witness_first", "mixed"} for mode in modes)
    legacy = any(mode in {"legacy_fallback", "mixed"} for mode in modes)
    if witness and legacy:
        return "mixed"
    if witness:
        return "witness_first"
    if legacy:
        return "legacy_fallback"
    return "empty"


class _NoOpRetrievalHandle:
    def set_rag_config(self, _config: object | None) -> None:
        return None


class DemoContextService:
    """Primary context façade for demo mode.

    Implements ContextBuilderPort.build so the existing runtime can keep using the
    same protocol while the demonstrator gains durable context snapshots and
    explicit zoom/zoom_out/apply_memory operations.
    """

    def __init__(
        self,
        base_builder: ContextBuilder,
        memory_service: MemoryService,
        rel_db: RelDBPort,
        run_journal: DemoRunJournal | None = None,
        geometry_reader: GeometryHintsPort | None = None,
        geometry_commands: GeometryCommandsPort | None = None,
        geometry_service: GeometryHintsPort | GeometryCommandsPort | None = None,
        handoff_service: DemoHandoffService | None = None,
        ephemeral_runtime: EphemeralRuntimeMirrorPort | None = None,
    ) -> None:
        if geometry_service is not None:
            geometry_reader = geometry_reader or geometry_service  # backward-compat for tests/demo wiring
            geometry_commands = geometry_commands or geometry_service

        self._base = base_builder
        self._memory = memory_service
        self._db = rel_db
        self._runs = run_journal
        self._lineage_resolver = WitnessLineageResolver(rel_db)
        self._geometry_reader = geometry_reader
        self._geometry_commands = geometry_commands
        self._handoff = handoff_service
        self._ephemeral_runtime = ephemeral_runtime
        # Legacy orchestrator compatibility: some cleanup paths still expect
        # a retrieval object with set_rag_config(). In demo mode this facade
        # owns context authority, so we expose a no-op shim instead of letting
        # the legacy runtime reach into a frontier retrieval engine.
        self._retrieval = _NoOpRetrievalHandle()

    def bind_run(self, *, run_id: str, conversation_id: str, turn_id: str, query_text: str) -> Token:
        return _bound_run.set(_BoundRun(
            run_id=run_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            query_text=query_text,
        ))

    def reset_run(self, token: Token) -> None:
        _bound_run.reset(token)
        _last_snapshot_id.set(None)

    def last_snapshot_id(self) -> str | None:
        return _last_snapshot_id.get()

    async def _marker_resolution_rows(
        self,
        *,
        conversation_id: str,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        detailed = await self._lineage_resolver.resolve_selected_items_detailed(
            conversation_id=conversation_id,
            selected_items=items,
        )
        rows: list[dict[str, Any]] = []
        for item, detail in zip(items, detailed, strict=False):
            row = {
                "marker": str(item.get("marker") or detail.get("marker") or ""),
                "ref_kind": str(item.get("ref_kind") or detail.get("ref_kind") or "citem"),
                "ref_id": str(item.get("ref_id") or detail.get("ref_id") or ""),
                "resolution_mode": str(detail.get("resolution_mode") or "empty"),
                "support_resolution_mode": str(detail.get("support_resolution_mode") or ""),
                "resolution_scope": str(detail.get("resolution_scope") or ""),
                "resolved_source_ids": list(detail.get("resolved_source_ids") or []),
                "resolved_span_ids": list(detail.get("resolved_span_ids") or []),
                "resolved_source_count": int(detail.get("resolved_source_count") or 0),
                "resolved_span_count": int(detail.get("resolved_span_count") or 0),
                "unresolved_ref_ids": list(detail.get("unresolved_ref_ids") or []),
                "unresolved_citem_ids": list(detail.get("unresolved_citem_ids") or []),
                "citem_witnesses": [dict(row) for row in list(detail.get("citem_witnesses") or []) if isinstance(row, dict)],
                # Direct lineage payload required for independent CIMA checks:
                # summary markers must expose the immediate L0 C-items they use;
                # citem markers expose themselves as citem_ids.
                "citem_ids": list(detail.get("citem_ids") or []),
                "summary_ids": list(detail.get("summary_ids") or []),
            }
            if row["marker"]:
                rows.append(row)
        return rows

    async def _normalize_snapshot_public(self, snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        normalized = dict(snapshot)
        items = [dict(item) for item in list(snapshot.get("items") or [])]
        conversation_id = str(snapshot.get("conversation_id") or "")
        marker_resolution = list(snapshot.get("marker_resolution") or [])
        if not marker_resolution:
            marker_resolution = await self._marker_resolution_rows(
                conversation_id=conversation_id,
                items=items,
            ) if conversation_id and items else []
        detail_by_key = {
            (str(row.get("marker") or ""), str(row.get("ref_kind") or ""), str(row.get("ref_id") or "")): row
            for row in marker_resolution
        }
        for item in items:
            key = (str(item.get("marker") or ""), str(item.get("ref_kind") or ""), str(item.get("ref_id") or ""))
            detail = detail_by_key.get(key)
            if detail is None:
                continue
            ref_kind = str(item.get("ref_kind") or "citem")
            if ref_kind in {"summary", "local_summary", "global_summary"}:
                item.setdefault("summary_resolution_mode", detail["resolution_mode"])
                if detail.get("resolution_scope"):
                    item.setdefault("summary_scope", detail["resolution_scope"])
            else:
                item.setdefault("item_resolution_mode", detail["resolution_mode"])
                if detail.get("resolution_scope"):
                    item.setdefault("item_resolution_scope", detail["resolution_scope"])

        item_modes = [
            str(item.get("item_resolution_mode") or "")
            for item in items
            if str(item.get("item_resolution_mode") or "")
        ]
        summary_modes = [
            str(item.get("summary_resolution_mode") or "")
            for item in items
            if str(item.get("ref_kind") or "") in {"summary", "local_summary", "global_summary"}
            and str(item.get("summary_resolution_mode") or "")
        ]
        marker_modes = [str(row.get("resolution_mode") or "") for row in marker_resolution if str(row.get("resolution_mode") or "")]
        marker_support_modes = [str(row.get("support_resolution_mode") or "") for row in marker_resolution if str(row.get("support_resolution_mode") or "")]
        if not list(normalized.get("evidence_marker_registry") or []):
            _, _, registry = filter_citable_items(items=items, marker_resolution=marker_resolution)
            normalized["evidence_marker_registry"] = registry
        normalized["items"] = items
        normalized["marker_resolution"] = marker_resolution
        normalized["resolution_mode"] = _merge_resolution_modes([
            str(snapshot.get("resolution_mode") or "empty"),
            *item_modes,
            *summary_modes,
            *marker_modes,
            *marker_support_modes,
        ])
        return normalized

    async def load_context_snapshot_public(self, context_id: str) -> dict[str, Any] | None:
        snapshot = await self._db.load_demo_context_snapshot(context_id)
        return await self._normalize_snapshot_public(snapshot)

    async def load_context_snapshots_for_run_public(self, run_id: str) -> list[dict[str, Any]]:
        snapshots = await _maybe_call(self._db, "load_demo_context_snapshots_for_run", run_id, default=[]) or []
        out: list[dict[str, Any]] = []
        for snapshot in snapshots:
            normalized = await self._normalize_snapshot_public(dict(snapshot))
            if normalized is not None:
                out.append(normalized)
        return out

    async def _schedule_geometry_recompute(self, conversation_id: str, *, reason: str) -> None:
        if self._geometry_commands is None:
            return
        maybe_awaitable = self._geometry_commands.schedule_recompute(conversation_id, reason=reason)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    async def build(
        self,
        phase: str,
        task_memory: TaskMemory,
        plan: Plan | None,
        query: str,
        conversation_id: str,
        budget: ContextBudget,
        history_contents: set[str] | None = None,
        global_objective: str = "",
        local_objective: str = "",
        exclude_ids: set[str] | None = None,
    ) -> ContextView:
        context_view = await self._base.build(
            phase=phase,
            task_memory=task_memory,
            plan=plan,
            query=query,
            conversation_id=conversation_id,
            budget=budget,
            history_contents=history_contents,
            global_objective=global_objective,
            local_objective=local_objective,
            exclude_ids=exclude_ids,
            disable_geometric_expand=True,
        )
        context_view = _drop_current_turn_items(context_view, history_contents)

        if self._geometry_reader is not None and context_view.items:
            ref_ids = [str(item.get("ref_id")) for item in context_view.items if item.get("ref_kind") == "citem" and item.get("ref_id")]
            if ref_ids:
                hints = await self._geometry_reader.get_item_hints(conversation_id=conversation_id, ref_ids=ref_ids)
                if hints:
                    enriched = []
                    for item in context_view.items:
                        ref_id = str(item.get("ref_id", ""))
                        hint = hints.get(ref_id)
                        if hint:
                            merged = dict(item)
                            merged["geom_cluster"] = hint.get("cluster_top1")
                            merged["geom_label"] = hint.get("label")
                            merged["geom_role"] = "CORE" if hint.get("is_core") else "BRIDGE" if hint.get("is_bridge_candidate") else "PERIPH"
                            enriched.append(merged)
                        else:
                            enriched.append(dict(item))
                    context_view.items = enriched
        marker_resolution = await self._marker_resolution_rows(
            conversation_id=conversation_id,
            items=[dict(item) for item in context_view.items],
        )
        detail_by_key = {
            (str(row.get("marker") or ""), str(row.get("ref_kind") or ""), str(row.get("ref_id") or "")): row
            for row in marker_resolution
        }
        normalized_items: list[dict[str, Any]] = []
        resolved_source_ids: set[str] = set()
        resolved_span_ids: set[str] = set()
        unresolved_ref_ids: set[str] = set()
        item_resolution_modes: list[str] = []
        summary_modes: list[str] = []
        for item in context_view.items:
            materialized = dict(item)
            key = (str(materialized.get("marker") or ""), str(materialized.get("ref_kind") or ""), str(materialized.get("ref_id") or ""))
            detail = detail_by_key.get(key)
            if detail is not None:
                ref_kind = str(materialized.get("ref_kind") or "citem")
                if ref_kind in {"summary", "local_summary", "global_summary"}:
                    materialized.setdefault("summary_resolution_mode", str(detail.get("resolution_mode") or "empty"))
                    if detail.get("resolution_scope"):
                        materialized.setdefault("summary_scope", str(detail.get("resolution_scope") or ""))
                    if str(materialized.get("summary_resolution_mode") or ""):
                        summary_modes.append(str(materialized.get("summary_resolution_mode") or ""))
                else:
                    materialized.setdefault("item_resolution_mode", str(detail.get("resolution_mode") or "empty"))
                    if detail.get("resolution_scope"):
                        materialized.setdefault("item_resolution_scope", str(detail.get("resolution_scope") or ""))
                    if str(materialized.get("item_resolution_mode") or ""):
                        item_resolution_modes.append(str(materialized.get("item_resolution_mode") or ""))
                resolved_source_ids.update(str(v) for v in detail.get("resolved_source_ids") or [] if str(v))
                resolved_span_ids.update(str(v) for v in detail.get("resolved_span_ids") or [] if str(v))
                unresolved_ref_ids.update(str(v) for v in detail.get("unresolved_ref_ids") or [] if str(v))
            normalized_items.append(materialized)
        filtered_items, auxiliary_items, filtered_marker_resolution, evidence_marker_registry = split_citable_and_auxiliary_items(
            items=normalized_items,
            marker_resolution=marker_resolution,
        )
        pre_budget_citable_item_count = len(filtered_items)
        filtered_items, budget_trimmed_items = _trim_marker_items_to_budget(filtered_items, int(budget.available_for_content or 0))
        kept_markers_after_budget = {str(item.get("marker") or "") for item in filtered_items if str(item.get("marker") or "")}
        if budget_trimmed_items:
            filtered_marker_resolution = [row for row in filtered_marker_resolution if str(row.get("marker") or "") in kept_markers_after_budget]
        context_view.items = filtered_items
        setattr(context_view, "auxiliary_items", auxiliary_items)
        main_context_text = _render_marker_context(filtered_items)
        # Do not inject auxiliary non-citable items into the prompt by default.
        # They are kept in the snapshot and metrics so we can diagnose upstream
        # lineage loss instead of silently hiding it, but they cannot influence a
        # factual answer as evidence.
        context_view.text = main_context_text
        context_view.citem_ids = [
            str(item.get("ref_id"))
            for item in filtered_items
            if str(item.get("ref_kind") or "") == "citem" and item.get("ref_id")
        ]
        context_view.tokens_used = _retokenize_rough(context_view.text)
        setattr(context_view, "evidence_marker_registry", evidence_marker_registry)
        visible_marker_support = _visible_marker_support_from_items(filtered_items, rendered_text=main_context_text, namespace="runtime_context")
        visible_markers = {str(row.get("marker") or "") for row in visible_marker_support if str(row.get("marker") or "")}
        truncated_visible_markers = sorted({
            str(row.get("marker") or "")
            for row in visible_marker_support
            if row.get("context_truncated_for_budget") is True and str(row.get("marker") or "")
        })
        visible_support_metrics = {
            "visible_marker_count": len(visible_markers),
            "visible_marker_support_count": len(visible_marker_support),
            "truncated_visible_marker_count": len(truncated_visible_markers),
            "truncated_visible_markers": truncated_visible_markers,
            "all_citable_markers_visible": all(str(item.get("marker") or "") in visible_markers for item in filtered_items if str(item.get("marker") or "")),
            "auxiliary_items_rendered_to_prompt": False,
        }
        setattr(context_view, "visible_marker_support", visible_marker_support)
        setattr(context_view, "visible_support_metrics", visible_support_metrics)
        dropped_uncitable_items = [
            {
                "uncitable_marker": str(item.get("uncitable_marker") or ""),
                "ref_kind": str(item.get("ref_kind") or ""),
                "ref_id": str(item.get("ref_id") or ""),
                "reason_if_not_citable": str(item.get("reason_if_not_citable") or ""),
            }
            for item in auxiliary_items
        ]
        input_item_count = len(normalized_items)
        post_budget_citable_item_count = len(filtered_items)
        context_drop_metrics = {
            "input_item_count": input_item_count,
            "pre_budget_citable_item_count": pre_budget_citable_item_count,
            "post_budget_citable_item_count": post_budget_citable_item_count,
            "citable_item_count": post_budget_citable_item_count,  # backward-compatible alias
            "auxiliary_item_count": len(auxiliary_items),
            "dropped_uncitable_item_count": len(dropped_uncitable_items),
            "dropped_uncitable_marker_count": len(dropped_uncitable_items),
            "budget_trimmed_marker_count": len(budget_trimmed_items),
            "citable_item_retention_rate": (pre_budget_citable_item_count / input_item_count) if input_item_count else 1.0,
            "context_item_retention_rate": (post_budget_citable_item_count / input_item_count) if input_item_count else 1.0,
            "budget_retention_rate": (post_budget_citable_item_count / pre_budget_citable_item_count) if pre_budget_citable_item_count else 1.0,
            "dropped_uncitable_reasons": sorted({str(item.get("reason_if_not_citable") or "") for item in dropped_uncitable_items if str(item.get("reason_if_not_citable") or "")}),
        }
        setattr(context_view, "dropped_uncitable_items", dropped_uncitable_items)
        setattr(context_view, "context_drop_metrics", context_drop_metrics)

        marker_resolution = filtered_marker_resolution
        resolved_source_ids = {
            str(v)
            for row in marker_resolution
            for v in list(row.get("resolved_source_ids") or [])
            if str(v)
        }
        resolved_span_ids = {
            str(v)
            for row in marker_resolution
            for v in list(row.get("resolved_span_ids") or [])
            if str(v)
        }
        unresolved_ref_ids = {
            str(v)
            for row in marker_resolution
            for v in list(row.get("unresolved_ref_ids") or [])
            if str(v)
        }

        marker_modes = [str(row.get("resolution_mode") or "") for row in marker_resolution if str(row.get("resolution_mode") or "")]
        marker_support_modes = [str(row.get("support_resolution_mode") or "") for row in marker_resolution if str(row.get("support_resolution_mode") or "")]
        effective_resolution_mode = _merge_resolution_modes([
            *item_resolution_modes,
            *summary_modes,
            *marker_modes,
            *marker_support_modes,
        ])

        bound = _bound_run.get()
        if bound is not None and bound.conversation_id == conversation_id:
            snapshot = ContextSnapshot(
                context_id=str(uuid.uuid4()),
                run_id=bound.run_id,
                conversation_id=conversation_id,
                turn_id=bound.turn_id,
                query_text=query,
                phase=phase,
                context_text=context_view.text,
                markers=[item.get("marker", "") for item in context_view.items],
                items=[dict(item) for item in context_view.items],
                auxiliary_items=[dict(item) for item in auxiliary_items],
                dropped_uncitable_items=dropped_uncitable_items,
                context_drop_metrics=context_drop_metrics,
                visible_marker_support=visible_marker_support,
                visible_support_metrics=visible_support_metrics,
                budget={
                    "max_tokens": budget.max_tokens,
                    "overhead_tokens": budget.overhead_tokens,
                    "available_for_content": budget.available_for_content,
                    "tokens_used": context_view.tokens_used,
                    "coverage_score": context_view.coverage_score,
                    "global_objective": global_objective,
                    "local_objective": local_objective,
                },
                resolved_source_ids=sorted(resolved_source_ids),
                resolved_span_ids=sorted(resolved_span_ids),
                resolved_source_count=len(resolved_source_ids),
                resolved_span_count=len(resolved_span_ids),
                unresolved_ref_ids=sorted(unresolved_ref_ids),
                marker_resolution=marker_resolution,
                evidence_marker_registry=evidence_marker_registry,
                resolution_mode=effective_resolution_mode,
            )
            await self._db.save_demo_context_snapshot(snapshot.to_dict())
            _last_snapshot_id.set(snapshot.context_id)
            if self._runs is not None:
                await self._runs.write_json_artifact(
                    conversation_id=conversation_id,
                    run_id=bound.run_id,
                    relative_path=f"context_snapshot_{snapshot.context_id}.json",
                    payload=snapshot.to_dict(),
                )
                await self._runs.write_text_artifact(
                    conversation_id=conversation_id,
                    run_id=bound.run_id,
                    relative_path=f"context_pack_{snapshot.context_id}.txt",
                    text=context_view.text,
                )
                await self._runs.write_json_artifact(
                    conversation_id=conversation_id,
                    run_id=bound.run_id,
                    relative_path=f"budget_trace_{snapshot.context_id}.json",
                    payload=snapshot.budget,
                )
        if self._ephemeral_runtime is not None and context_view.items:
            await self._ephemeral_runtime.mirror_context_items(
                conversation_id=conversation_id,
                items=[dict(item) for item in context_view.items],
            )
        await self._schedule_geometry_recompute(conversation_id, reason="context_snapshot")
        return context_view

    async def get_context(
        self,
        *,
        conversation_id: str,
        query: str,
        phase: str,
        task_memory: TaskMemory,
        plan: Plan | None,
        budget: ContextBudget,
        history_contents: set[str] | None = None,
        global_objective: str = "",
        local_objective: str = "",
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        context_view = await self.build(
            phase=phase,
            task_memory=task_memory,
            plan=plan,
            query=query,
            conversation_id=conversation_id,
            budget=budget,
            history_contents=history_contents,
            global_objective=global_objective,
            local_objective=local_objective,
            exclude_ids=exclude_ids,
        )
        snapshot_resolution: dict[str, Any] = {}
        snapshot_id = self.last_snapshot_id()
        if snapshot_id:
            snapshot = await self.load_context_snapshot_public(snapshot_id)
            if snapshot is not None:
                snapshot_resolution = {
                    "resolved_source_ids": list(snapshot.get("resolved_source_ids") or []),
                    "resolved_span_ids": list(snapshot.get("resolved_span_ids") or []),
                    "resolved_source_count": int(snapshot.get("resolved_source_count") or 0),
                    "resolved_span_count": int(snapshot.get("resolved_span_count") or 0),
                    "unresolved_ref_ids": list(snapshot.get("unresolved_ref_ids") or []),
                    "resolution_mode": str(snapshot.get("resolution_mode") or "empty"),
                    "marker_resolution": list(snapshot.get("marker_resolution") or []),
                    "evidence_marker_registry": list(snapshot.get("evidence_marker_registry") or []),
                    "visible_marker_support": list(snapshot.get("visible_marker_support") or []),
                    "visible_support_metrics": dict(snapshot.get("visible_support_metrics") or {}),
                }
        return {
            "context_id": snapshot_id,
            "context_pack": context_view.text,
            "markers": [item.get("marker") for item in context_view.items],
            "token_usage": {
                "context": context_view.tokens_used,
                "available_for_content": budget.available_for_content,
                "overhead": budget.overhead_tokens,
            },
            **snapshot_resolution,
        }

    async def zoom(self, *, context_id: str, zoom_targets: list[str], max_evidence_tokens: int = 800) -> dict[str, Any]:
        snapshot = await self.load_context_snapshot_public(context_id)
        if snapshot is None:
            return {
                "context_id": context_id,
                "evidence_block": "",
                "markers_added": [],
                "token_usage": {"evidence": 0},
                "resolved_source_ids": [],
                "resolved_span_ids": [],
                "resolved_source_count": 0,
                "resolved_span_count": 0,
                "unresolved_ref_ids": [],
                "resolution_mode": "empty",
                "marker_resolution": [],
                "visible_marker_support": [],
            }

        conversation_id = str(snapshot.get("conversation_id") or "")
        items = list(snapshot.get("items") or [])
        targets = {str(t) for t in zoom_targets if str(t)}
        lines: list[str] = []
        added: list[str] = []
        budget = 0
        resolved_source_ids: set[str] = set()
        resolved_span_ids: set[str] = set()
        unresolved_ref_ids: set[str] = set()
        resolution_modes: list[str] = []
        visible_marker_support: list[dict[str, Any]] = []

        for item in items:
            marker = str(item.get("marker") or "")
            if marker not in targets:
                continue

            detail_rows = await self._lineage_resolver.resolve_selected_items_detailed(
                conversation_id=conversation_id,
                selected_items=[dict(item)],
            )
            detail_row = detail_rows[0] if detail_rows else {}
            if marker_resolution_status(detail_row) not in {"source_span", "summary_witness"}:
                continue
            resolution_modes.append(str(detail_row.get("resolution_mode") or "empty"))
            detail_source_ids = [str(v) for v in list(detail_row.get("resolved_source_ids") or []) if str(v)]
            detail_span_ids = [str(v) for v in list(detail_row.get("resolved_span_ids") or []) if str(v)]
            detail_unresolved = [str(v) for v in list(detail_row.get("unresolved_ref_ids") or []) if str(v)]
            resolved_source_ids.update(detail_source_ids)
            resolved_span_ids.update(detail_span_ids)
            unresolved_ref_ids.update(detail_unresolved)

            span_rows = await _maybe_call(
                self._db,
                "load_demo_source_spans",
                conversation_id,
                detail_span_ids,
                default=[],
            ) or []
            source_rows = await _maybe_call(
                self._db,
                "load_demo_sources",
                conversation_id,
                detail_source_ids,
                default=[],
            ) or []
            source_by_id = {str(row.get("source_id")): dict(row) for row in source_rows if row.get("source_id")}

            evidence_lines: list[str] = []
            seen_previews: set[str] = set()
            for span in span_rows:
                preview = _escape_internal_marker_literals(str(span.get("preview_text") or "").strip())
                if not preview or preview in seen_previews:
                    continue
                seen_previews.add(preview)
                locator = dict(span.get("locator") or {})
                source = source_by_id.get(str(span.get("source_id") or ""), {})
                tags: list[str] = []
                source_kind = str(source.get("source_kind") or "").strip()
                role = str(source.get("role") or "").strip()
                if source_kind and role:
                    tags.append(f"{source_kind}:{role}")
                elif source_kind:
                    tags.append(source_kind)
                if source.get("origin_ref"):
                    tags.append(f"ref={source['origin_ref']}")
                if locator.get("filename"):
                    tags.append(f"file={locator['filename']}")
                if locator.get("page_num") is not None:
                    tags.append(f"page={locator['page_num']}")
                if locator.get("chunk_index") is not None:
                    tags.append(f"chunk={locator['chunk_index']}")
                suffix = f" ({', '.join(tags)})" if tags else ""
                evidence_lines.append(f"  - {preview}{suffix}")

            if not evidence_lines:
                for source in source_rows:
                    preview = _escape_internal_marker_literals(str(source.get("display_text") or source.get("process_text") or "").strip())
                    if not preview or preview in seen_previews:
                        continue
                    seen_previews.add(preview)
                    tags: list[str] = []
                    source_kind = str(source.get("source_kind") or "").strip()
                    role = str(source.get("role") or "").strip()
                    if source_kind and role:
                        tags.append(f"{source_kind}:{role}")
                    elif source_kind:
                        tags.append(source_kind)
                    if source.get("origin_ref"):
                        tags.append(f"ref={source['origin_ref']}")
                    suffix = f" ({', '.join(tags)})" if tags else ""
                    evidence_lines.append(f"  - {preview[:500]}{suffix}")

            header = f"[{marker}] {item.get('content', '')}"
            block = "\n".join([header, *evidence_lines]) if evidence_lines else header
            block_cost = max(1, len(block) // 4)
            if lines and budget + block_cost > max_evidence_tokens:
                break
            lines.append(block)
            block_start = len("\n\n".join(lines[:-1])) + (2 if len(lines) > 1 else 0)
            block_end = block_start + len(block)
            visible_marker_support.append({
                "marker": marker,
                "marker_namespace": "zoom",
                "marker_uid": f"zoom:{marker}",
                "ref_kind": str(item.get("ref_kind") or ""),
                "ref_id": str(item.get("ref_id") or ""),
                "support_source": "zoom_block",
                "rendered_text_sha256": _sha256_text(block),
                "visible_text_sha256": _sha256_text(block),
                "visible_text_preview": block[:240],
                "visible_char_count": len(block),
                "context_truncated_for_budget": False,
                "visible_support_scope": "zoom_block",
                "prompt_char_start": block_start,
                "prompt_char_end": block_end,
                "visible_char_start": block_start,
                "visible_char_end": block_end,
                "escaped_marker_literals": bool(_INTERNAL_MARKER_LITERAL_RE.search(str(item.get("content") or ""))),
            })
            added.append(marker)
            budget += block_cost
            if budget >= max_evidence_tokens:
                break

        item_resolution_modes = [
            str(item.get("item_resolution_mode") or "")
            for item in items
            if str(item.get("marker") or "") in targets and str(item.get("item_resolution_mode") or "")
        ]
        marker_resolution = [
            dict(row)
            for row in list(snapshot.get("marker_resolution") or [])
            if str(row.get("marker") or "") in targets
        ]
        return {
            "context_id": context_id,
            "evidence_block": "\n\n".join(lines),
            "markers_added": added,
            "token_usage": {"evidence": budget},
            "resolved_source_ids": sorted(resolved_source_ids),
            "resolved_span_ids": sorted(resolved_span_ids),
            "resolved_source_count": len(resolved_source_ids),
            "resolved_span_count": len(resolved_span_ids),
            "unresolved_ref_ids": sorted(unresolved_ref_ids),
            "resolution_mode": _merge_resolution_modes([*resolution_modes, *item_resolution_modes]),
            "marker_resolution": marker_resolution,
            "visible_marker_support": visible_marker_support,
        }

    async def zoom_out(self, *, context_id: str, targets: list[str], max_perspective_tokens: int = 800) -> dict[str, Any]:
        snapshot = await self.load_context_snapshot_public(context_id)
        if snapshot is None:
            return {
                "context_id": context_id,
                "perspective_block": "",
                "markers_added": [],
                "token_usage": {"perspective": 0},
                "focus_citem_ids": [],
                "resolution_mode": "empty",
                "marker_resolution": [],
            }
        conversation_id = snapshot["conversation_id"]
        focus_items = [
            dict(item)
            for item in list(snapshot.get("items") or [])
            if str(item.get("marker") or "") in {str(v) for v in targets if str(v)}
        ]
        focus_resolution = await self._lineage_resolver.resolve_selected_items(
            conversation_id=conversation_id,
            selected_items=focus_items,
        ) if focus_items else None

        summaries = await self._db.fetch_pyramid_tops(conversation_id, limit=8)
        summary_modes = [
            str(getattr(node, "summary_resolution_mode", "legacy_fallback") or "legacy_fallback")
            for node in summaries
        ]
        focus_item_modes = [
            str(item.get("item_resolution_mode") or "")
            for item in focus_items
            if str(item.get("item_resolution_mode") or "")
        ]
        if focus_resolution is not None and focus_resolution.citem_ids:
            focus_ids = set(str(v) for v in focus_resolution.citem_ids if str(v))
            filtered = [
                node for node in summaries
                if set(str(v) for v in (node.origin_citem_ids or [])) & focus_ids
            ]
            if filtered:
                summaries = filtered
        lines: list[str] = []
        added: list[str] = []
        added_items: list[dict[str, Any]] = []
        visible_marker_support: list[dict[str, Any]] = []
        budget = 0
        for idx, node in enumerate(summaries, start=1):
            marker = f"P{idx}"
            node_content = _escape_internal_marker_literals(str(node.content or ""))
            line = f"[{marker}] {node_content}"
            line_cost = max(1, len(node_content) // 4)
            if lines and budget + line_cost > max_perspective_tokens:
                break
            lines.append(line)
            block_start = len("\n\n".join(lines[:-1])) + (2 if len(lines) > 1 else 0)
            block_end = block_start + len(line)
            visible_marker_support.append({
                "marker": marker,
                "marker_namespace": "zoom_out",
                "marker_uid": f"zoom_out:{marker}",
                "ref_kind": str(getattr(node, "summary_ref_kind", "local_summary") or "local_summary"),
                "ref_id": str(getattr(node, "node_id", "") or ""),
                "support_source": "zoom_out_block",
                "rendered_text_sha256": _sha256_text(line),
                "visible_text_sha256": _sha256_text(node_content),
                "visible_text_preview": node_content[:240],
                "visible_char_count": len(node_content),
                "context_truncated_for_budget": False,
                "visible_support_scope": "zoom_out_block",
                "prompt_char_start": block_start,
                "prompt_char_end": block_end,
                "visible_char_start": block_start,
                "visible_char_end": block_end,
                "escaped_marker_literals": bool(_INTERNAL_MARKER_LITERAL_RE.search(str(node.content or ""))),
            })
            added.append(marker)
            added_items.append({
                "marker": marker,
                "ref_kind": str(getattr(node, "summary_ref_kind", "local_summary") or "local_summary"),
                "ref_id": str(getattr(node, "node_id", "") or ""),
                "summary_resolution_mode": str(getattr(node, "summary_resolution_mode", "") or ""),
                "summary_scope": str(getattr(node, "summary_scope", "") or ""),
            })
            budget += line_cost
            if budget >= max_perspective_tokens:
                break

        added_resolution = await self._lineage_resolver.resolve_selected_items_detailed(
            conversation_id=conversation_id,
            selected_items=added_items,
        ) if added_items else []
        added_source_ids = sorted({
            str(source_id)
            for detail in added_resolution
            for source_id in list(detail.get("resolved_source_ids") or [])
            if str(source_id)
        })
        added_span_ids = sorted({
            str(span_id)
            for detail in added_resolution
            for span_id in list(detail.get("resolved_span_ids") or [])
            if str(span_id)
        })
        added_unresolved = sorted({
            str(ref_id)
            for detail in added_resolution
            for ref_id in list(detail.get("unresolved_ref_ids") or [])
            if str(ref_id)
        })
        summary_lineage_valid = bool(
            added
            and len(added_resolution) == len(added)
            and not added_unresolved
            and all(marker_resolution_status(detail) == "summary_witness" for detail in added_resolution)
        )
        return {
            "context_id": context_id,
            "perspective_block": "\n\n".join(lines),
            "markers_added": added,
            "token_usage": {"perspective": budget},
            "focus_citem_ids": sorted(focus_resolution.citem_ids) if focus_resolution is not None else [],
            "resolution_mode": _merge_resolution_modes([
                focus_resolution.resolution_mode if focus_resolution is not None else "empty",
                *focus_item_modes,
                *summary_modes,
                *[str(detail.get("resolution_mode") or "") for detail in added_resolution],
            ]),
            "marker_resolution": [*list(snapshot.get("marker_resolution") or []), *added_resolution],
            "zoom_out_marker_resolution": added_resolution,
            "visible_marker_support": visible_marker_support,
            "summary_lineage_valid": summary_lineage_valid,
            "summary_lineage_policy": "direct_effective_inputs_only",
            "summary_used_refs_by_marker": {
                str(detail.get("marker") or ""): list(detail.get("citem_ids") or [])
                for detail in added_resolution
                if str(detail.get("marker") or "")
            },
            "resolved_source_ids": added_source_ids,
            "resolved_span_ids": added_span_ids,
            "resolved_source_count": len(added_source_ids),
            "resolved_span_count": len(added_span_ids),
            "unresolved_ref_ids": added_unresolved,
        }

    async def apply_memory(
        self,
        *,
        conversation_id: str,
        conclude: list[str],
        phase: str,
        turn_id: str,
    ) -> dict[str, Any]:
        accepted: list[str] = []
        rejected: list[str] = []
        conclusions: list[dict[str, Any]] = []
        for line in conclude:
            if ":" not in line:
                rejected.append(f"{line} | reason=missing_colon")
                continue
            kind, rest = line.split(":", 1)
            kind = kind.strip().upper()
            if kind not in {"AXIOM", "FACT", "HEDGED_FACT", "DECISION", "TODO", "NOTE"}:
                rejected.append(f"{line} | reason=unsupported_kind")
                continue
            text = rest.split("| prov=", 1)[0].strip()
            if not text:
                rejected.append(f"{line} | reason=empty_statement")
                continue
            conclusions.append({
                "type": kind,
                "content": text,
                "motivation": "demo_memory_apply",
                "confidence": 1.0 if kind in {"AXIOM", "FACT", "DECISION"} else 0.7,
            })
            accepted.append(line)
        if conclusions:
            await self._memory.ingest_batch(conclusions, phase, conversation_id, turn_id)
            await self._schedule_geometry_recompute(conversation_id, reason="memory_apply")
        return {
            "accepted": accepted,
            "rejected": rejected,
            "memory_delta_id": str(uuid.uuid4()),
        }

    async def create_handoff(self, **kwargs: Any) -> dict[str, Any]:
        if self._handoff is None:
            raise RuntimeError("handoff service not configured")
        manifest = await self._handoff.create_handoff(**kwargs)
        return manifest.to_dict()

    async def validate_handoff(self, **kwargs: Any) -> dict[str, Any]:
        if self._handoff is None:
            raise RuntimeError("handoff service not configured")
        validation = await self._handoff.validate_handoff(**kwargs)
        return validation.to_dict()

    async def restore_handoff(self, **kwargs: Any) -> dict[str, Any]:
        if self._handoff is None:
            raise RuntimeError("handoff service not configured")
        restore = await self._handoff.restore_handoff(**kwargs)
        return restore.to_dict()
