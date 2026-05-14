"""Structured control loop for the CIMA Demonstrator."""
from __future__ import annotations

import json
import logging
import re
import time
import hashlib
from dataclasses import asdict
from typing import Any

from cima_demo.application.stream_manager import StreamManager
from cima_demo.demo.runtime import c3_sanitize, prompt_trace
from cima_demo.cognitive.message_builder import _render_output_contract_block
from cima_demo.demo.contracts import MemoryProposal, NeedProposal
from cima_demo.demo.context.service import DemoContextService
from cima_demo.demo.context.evidence_marker_registry import build_registry, marker_resolution_status
from cima_demo.demo.runtime.journal import DemoRunJournal
from cima_demo.domain.entities import KimaDelta, LLMMessage, Plan, TaskMemory
from cima_demo.domain.ports import LLMPort
from cima_demo.domain.value_objects import ContextBudget, ExecutionStage, KimaDeltaType, TurnOutcome
from cima_demo.memory.service import MemoryService

log = logging.getLogger(__name__)

_MARKER_GROUP_RE = re.compile(r"\[((?:[A-Za-z]\d+)(?:\s*,\s*[A-Za-z]\d+)*)\]")
_MARKER_LITERAL_RE = re.compile(r"\[([SEP][1-9][0-9]*)\]")


_VISIBLE_DELTA_TARGET_CHARS = 1024
_VISIBLE_DELTA_MAX_SOURCE_CHUNKS = 64


def _visible_answer_chunks(
    answer_text: str,
    source_chunks: list[str] | None = None,
    *,
    max_chars: int = _VISIBLE_DELTA_TARGET_CHARS,
    max_source_chunks: int = _VISIBLE_DELTA_MAX_SOURCE_CHUNKS,
) -> list[str]:
    """Return visible chunks preserving the exact accepted answer text.

    Small streamed answers keep their original granularity for compatibility.
    Large post-validated answers are coalesced because they are emitted in a
    burst at turn end and can otherwise overflow the in-process SSE queue.
    """
    if not answer_text:
        return []
    if source_chunks and len(source_chunks) <= max_source_chunks:
        joined = "".join(source_chunks)
        if joined == answer_text:
            return [chunk for chunk in source_chunks if chunk]
    max_chars = max(128, int(max_chars))
    return [
        answer_text[i:i + max_chars]
        for i in range(0, len(answer_text), max_chars)
    ]


def _marker_sort_key(marker: str) -> tuple[int, str, int, str]:
    """Sort citation markers deterministically with S# evidence first, then P# perspective markers."""
    marker = str(marker or "")
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", marker)
    if not match:
        return (9, marker, 0, marker)
    prefix = match.group(1).upper()
    number = int(match.group(2))
    # S markers are primary ContextView evidence; P markers are zoom-out perspective.
    priority = 0 if prefix == "S" else 1 if prefix == "P" else 2
    return (priority, prefix, number, marker)


def _render_marker_list(markers: set[str] | list[str]) -> str:
    ordered = sorted({str(m) for m in markers if str(m)}, key=_marker_sort_key)
    return ", ".join(f"[{m}]" for m in ordered)


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _marker_uid_from_row(row: dict[str, Any]) -> str:
    marker = str(row.get("marker") or "").strip()
    namespace = str(row.get("marker_namespace") or row.get("support_source") or "runtime_context").strip()
    return str(row.get("marker_uid") or (f"{namespace}:{marker}" if marker else "")).strip()


def _visible_support_signature(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Return the semantic evidence identity of a visible marker row.

    A marker may legitimately appear in multiple prompt blocks (for example a
    ContextView marker and a focused-zoom block) as long as every occurrence
    points to the same evidence object.  Distinct evidence identities behind the
    same public label remain ambiguous and are rejected.
    """
    source_ids = tuple(str(v) for v in list(row.get("source_ids") or row.get("resolved_source_ids") or []) if str(v))
    span_ids = tuple(str(v) for v in list(row.get("span_ids") or row.get("resolved_span_ids") or []) if str(v))
    return (
        str(row.get("ref_kind") or "").strip(),
        str(row.get("ref_id") or "").strip(),
        source_ids,
        span_ids,
    )


def _row_has_verified_prompt_anchor(row: dict[str, Any]) -> bool:
    """True only for support rows verified against the rendered prompt slice."""
    return bool(
        row.get("prompt_offsets_exact") is True
        and row.get("visible_slice_verified") is True
        and str(row.get("prompt_sha256") or "")
        and str(row.get("visible_slice_sha256") or "")
    )


class DemoTurnController:
    """Primary turn authority for demo mode.

    Keeps control in the demonstrator slice: explicit context selection,
    structured control decisions, visible answer streaming, and structured
    memory application.
    """

    def __init__(
        self,
        *,
        llm_port: LLMPort,
        stream_manager: StreamManager,
        context_service: DemoContextService,
        memory_service: MemoryService,
        context_budget: ContextBudget,
        run_journal: DemoRunJournal | None = None,
        llm_temperature: float = 0.2,
        llm_top_p: float = 0.9,
        llm_repeat_penalty: float = 1.1,
        llm_max_tokens: int | None = None,
        answer_default_max_words: int = 220,
        answer_summary_max_words: int = 700,
        answer_specific_max_words: int = 160,
        debug_trace: bool = False,
        debug_trace_max_chars: int = 50000,
        llm_memory_pass: bool = True,
    ) -> None:
        self._llm = llm_port
        self._stream = stream_manager
        self._ctx = context_service
        self._memory = memory_service
        self._budget = context_budget
        self._runs = run_journal
        self._llm_temperature = llm_temperature
        self._llm_top_p = llm_top_p
        self._llm_repeat_penalty = llm_repeat_penalty
        self._llm_max_tokens = llm_max_tokens
        self._answer_default_max_words = max(80, int(answer_default_max_words))
        self._answer_summary_max_words = max(self._answer_default_max_words, int(answer_summary_max_words))
        self._answer_specific_max_words = max(60, int(answer_specific_max_words))
        self._debug_trace = bool(debug_trace)
        self._debug_trace_max_chars = max(1000, int(debug_trace_max_chars))
        self._llm_memory_pass = bool(llm_memory_pass)

    async def run_turn(
        self,
        rt: Any,
        task_memory: TaskMemory,
        plan: Plan | None,
        *,
        context_budget_override: ContextBudget | None = None,
        llm_max_tokens_override: int | None = None,
    ) -> None:
        conversation_id = rt.conversation_id
        turn_id = rt.turn_id
        run_id = rt.run_id
        query = rt.user_message
        phase = str(rt.phase)

        budget = context_budget_override or self._budget
        llm_max_tokens = (
            int(llm_max_tokens_override)
            if llm_max_tokens_override is not None and llm_max_tokens_override > 0
            else self._llm_max_tokens
        )

        rt.execution_stage = ExecutionStage.EVIDENCE_GATHERING
        # The current user request can be stored as a chat_user C-item before
        # synthesis. It must not become a citable S# evidence marker in the same
        # turn: otherwise external context probes (which do not include the
        # just-submitted chat_user item) and the actual answer prompt drift by
        # one marker. Filter exact current-turn text from the context pack while
        # keeping the source document lineage intact.
        current_turn_texts = {query.strip()} if query and query.strip() else set()
        context_view = await self._ctx.build(
            phase=phase,
            task_memory=task_memory,
            plan=plan,
            query=query,
            conversation_id=conversation_id,
            budget=budget,
            history_contents=current_turn_texts,
            global_objective=query,
            local_objective=query,
        )
        context_id = self._ctx.last_snapshot_id()
        rt.artifact_count = len(context_view.items)
        if self._runs is not None:
            await self._runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="CONTEXT_0",
                payload={
                    "context_id": context_id,
                    "item_count": len(context_view.items),
                    "tokens_used": context_view.tokens_used,
                },
            )

        need = await self._control_pass(rt=rt, query=query, context_view=context_view)
        rt.demo_need_proposal = need.to_dict()
        if self._runs is not None:
            await self._runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="CONTROL_PASS",
                payload=need.to_dict(),
            )

        extra_blocks: list[str] = []        # citable — included in citation contract
        extra_marker_resolution: list[dict[str, Any]] = []
        extra_visible_marker_support: list[dict[str, Any]] = []
        background_blocks: list[str] = []  # navigation/enrichment only — not citable
        if need.needs_zoom and context_id and need.zoom_markers:
            zoom_payload = await self._ctx.zoom(
                context_id=context_id,
                zoom_targets=need.zoom_markers[:3],
                max_evidence_tokens=min(800, budget.available_for_content),
            )
            rt.execution_stage = ExecutionStage.EVIDENCE_GATHERING
            zoom_block = zoom_payload.get("evidence_block") or ""
            extra_blocks.append("Focused evidence:\n" + zoom_block)
            zoom_rows = [dict(row) for row in list(zoom_payload.get("marker_resolution") or []) if isinstance(row, dict)]
            extra_marker_resolution.extend(zoom_rows)
            zoom_support = [dict(row) for row in list(zoom_payload.get("visible_marker_support") or []) if isinstance(row, dict)]
            if not zoom_support:
                zoom_support = self._visible_support_from_structured_rows(zoom_rows, zoom_block, namespace="zoom")
            extra_visible_marker_support.extend(zoom_support)
            if self._runs is not None:
                await self._runs.append_phase(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    phase_name="ENRICH_ZOOM",
                    payload=zoom_payload,
                )
        if need.needs_zoom_out and context_id:
            zoom_out_payload = await self._ctx.zoom_out(
                context_id=context_id,
                targets=need.zoom_markers[:3],
                max_perspective_tokens=min(800, budget.available_for_content),
            )
            rt.execution_stage = ExecutionStage.EVIDENCE_GATHERING
            # Zoom-out is citable only when CIMA has proved direct summary
            # lineage for the added perspective markers.  Otherwise it remains
            # navigational background and must not enter the citation contract.
            perspective_block = zoom_out_payload.get("perspective_block") or ""
            zoom_out_rows = [dict(row) for row in list(zoom_out_payload.get("zoom_out_marker_resolution") or []) if isinstance(row, dict)]
            zoom_out_markers = [str(v) for v in list(zoom_out_payload.get("markers_added") or []) if str(v)]
            zoom_out_citable = bool(
                zoom_out_payload.get("summary_lineage_valid") is True
                and zoom_out_rows
                and len(zoom_out_rows) >= len(zoom_out_markers)
                and all(marker_resolution_status(row) == "summary_witness" for row in zoom_out_rows)
            )
            if zoom_out_citable:
                extra_blocks.append("Perspective evidence:\n" + perspective_block)
                extra_marker_resolution.extend(zoom_out_rows)
                zoom_out_support = [dict(row) for row in list(zoom_out_payload.get("visible_marker_support") or []) if isinstance(row, dict)]
                if not zoom_out_support:
                    zoom_out_support = self._visible_support_from_structured_rows(zoom_out_rows, perspective_block, namespace="zoom_out")
                extra_visible_marker_support.extend(zoom_out_support)
            else:
                background_blocks.append("Perspective (background context, not citable):\n" + perspective_block)
            if self._runs is not None:
                await self._runs.append_phase(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    phase_name="ENRICH_ZOOM_OUT",
                    payload=zoom_out_payload,
                )

        rt.execution_stage = ExecutionStage.SYNTHESIS
        if self._runs is not None:
            await self._runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="ANSWER",
                payload={
                    "context_id": context_id,
                    "extra_blocks": len([b for b in extra_blocks if b.strip()]),
                },
            )
        setattr(context_view, "runtime_extra_visible_marker_support", extra_visible_marker_support)
        answer_text, citation_contract = await self._answer_pass(
            rt=rt,
            query=query,
            context_view=context_view,
            extra_blocks=extra_blocks,
            background_blocks=background_blocks,
            extra_marker_resolution=extra_marker_resolution,
            llm_max_tokens=llm_max_tokens,
        )
        rt.assistant_reply_buffer = answer_text
        rt.demo_citation_contract = citation_contract
        if self._runs is not None:
            await self._runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="CITATION_CONTRACT",
                payload=citation_contract,
            )

        memory = await self._memory_pass(
            rt=rt,
            query=query,
            answer_text=answer_text,
            context_view=context_view,
        )
        merged_markers = list(dict.fromkeys(memory.cited_markers + self._extract_markers(answer_text)))
        rt.cited_markers = merged_markers
        rt.demo_memory_proposal = memory.to_dict()
        if self._runs is not None:
            await self._runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="MEMORY_APPLY",
                payload=memory.to_dict(),
            )
        applied = await self._ctx.apply_memory(
            conversation_id=conversation_id,
            conclude=[f"{item.kind}: {item.content}" for item in memory.conclusions],
            phase=phase,
            turn_id=turn_id,
        )
        if self._runs is not None:
            await self._runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="COMMIT",
                payload={
                    "accepted": len(applied.get("accepted", [])),
                    "rejected": len(applied.get("rejected", [])),
                    "cited_markers": merged_markers,
                    "citation_contract_passed": citation_contract.get("passed"),
                },
            )
        rt.conclusions_types_seen = [item.kind for item in memory.conclusions]
        rt.compute_done = False
        rt.outcome = TurnOutcome.SUCCESS

    async def _control_pass(self, *, rt: Any, query: str, context_view: Any) -> NeedProposal:
        marker_catalog = [
            {
                "marker": item.get("marker"),
                "kind": item.get("ref_kind"),
                "preview": str(item.get("content", ""))[:240],
            }
            for item in context_view.items[:12]
        ]
        prompt = {
            "task": query,
            "phase": str(rt.phase),
            "markers": marker_catalog,
            "output_contract": self._output_contract_payload(rt),
            "instruction": {
                "return": {
                    "needs_zoom": "bool",
                    "zoom_markers": ["marker", "..."],
                    "needs_zoom_out": "bool",
                    "focus": "short string or null",
                    "reason": "short justification",
                },
                "rules": [
                    "Return JSON only.",
                    "Choose at most 3 zoom_markers and only from the provided markers.",
                    "Use zoom only when direct evidence is needed for the answer.",
                    "Use zoom_out only when broader perspective is needed.",
                ],
            },
        }
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are the control pass of CIMA Demonstrator. "
                    "Decide whether the current context is enough, whether a focused zoom is needed, "
                    "and whether a zoom-out perspective is needed. Return only a JSON object."
                ),
            ),
            LLMMessage(role="user", content=json.dumps(prompt, ensure_ascii=False, indent=2)),
        ]
        try:
            payload = await self._llm.complete_structured(
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            await self._record_llm_call(
                rt=rt,
                call_kind="control_pass",
                messages=messages,
                params={"temperature": 0.0, "max_tokens": 256, "response_format": "json_object"},
                context_id=self._ctx.last_snapshot_id(),
                response_json=payload,
            )
            need = NeedProposal.from_dict(payload)
        except Exception as exc:
            await self._record_llm_call(
                rt=rt,
                call_kind="control_pass",
                messages=messages,
                params={"temperature": 0.0, "max_tokens": 256, "response_format": "json_object"},
                context_id=self._ctx.last_snapshot_id(),
                error={"error_class": type(exc).__name__, "error": str(exc)},
            )
            log.warning("Demo control pass failed for %s — defaulting to no enrichment: %s", rt.conversation_id, exc)
            need = NeedProposal()
        need.zoom_markers = need.zoom_markers[:3]
        allowed = {str(item.get("marker", "")) for item in context_view.items}
        need.zoom_markers = [marker for marker in need.zoom_markers if marker in allowed]
        if not need.zoom_markers:
            need.needs_zoom = False
        return need

    def _debug_truncate(self, text: str) -> dict[str, Any]:
        text = text or ""
        limit = self._debug_trace_max_chars
        return {
            "chars": len(text),
            "truncated": len(text) > limit,
            "text": text[:limit],
        }

    async def _write_debug_json(self, *, rt: Any, relative_path: str, payload: dict[str, Any]) -> None:
        if not self._debug_trace or self._runs is None:
            return
        await self._write_json_artifact(rt=rt, relative_path=relative_path, payload=payload)

    async def _write_json_artifact(self, *, rt: Any, relative_path: str, payload: dict[str, Any]) -> None:
        if self._runs is None or not hasattr(self._runs, "write_json_artifact"):
            return
        try:
            await self._runs.write_json_artifact(
                conversation_id=rt.conversation_id,
                run_id=rt.run_id,
                relative_path=relative_path,
                payload=payload,
            )
        except Exception as exc:  # pragma: no cover - diagnostics must never break the turn
            log.warning("Failed to write JSON artifact %s for %s: %s", relative_path, getattr(rt, "conversation_id", "?"), exc)

    async def _append_jsonl_artifact(self, *, rt: Any, relative_path: str, payload: dict[str, Any]) -> None:
        if self._runs is None or not hasattr(self._runs, "append_jsonl_artifact"):
            return
        try:
            await self._runs.append_jsonl_artifact(
                conversation_id=rt.conversation_id,
                run_id=rt.run_id,
                relative_path=relative_path,
                payload=payload,
            )
        except Exception as exc:  # pragma: no cover - diagnostics must never break the turn
            log.warning("Failed to append JSONL artifact %s for %s: %s", relative_path, getattr(rt, "conversation_id", "?"), exc)

    def _llm_runtime_metadata(self) -> dict[str, Any]:
        try:
            metadata = self._llm.runtime_metadata()
            if isinstance(metadata, dict):
                return metadata
        except Exception:
            pass
        return {"provider": self._llm.__class__.__name__}

    async def _record_llm_call(
        self,
        *,
        rt: Any,
        call_kind: str,
        messages: list[LLMMessage],
        params: dict[str, Any] | None = None,
        allowed_markers: list[str] | set[str] | None = None,
        context_id: str | None = None,
        prompt_lint: dict[str, Any] | None = None,
        response_text: str | None = None,
        response_json: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        record = prompt_trace.build_llm_call_record(
            run_id=rt.run_id,
            conversation_id=rt.conversation_id,
            turn_id=rt.turn_id,
            call_kind=call_kind,
            messages=messages,
            params=params or {},
            runtime=self._llm_runtime_metadata(),
            allowed_markers=allowed_markers or [],
            context_id=context_id,
            prompt_lint=prompt_lint or {},
            response_text=response_text,
            response_json=response_json,
            error=error,
        )
        await self._append_jsonl_artifact(rt=rt, relative_path="llm_calls.jsonl", payload=record)

    async def _write_prompt_lint(
        self,
        *,
        rt: Any,
        answer_lint: dict[str, Any],
        repair_lints: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = {
            "schema_version": "cima_demo.prompt_lint.run.v1",
            "run_id": rt.run_id,
            "conversation_id": rt.conversation_id,
            "turn_id": rt.turn_id,
            "passed": bool(answer_lint.get("passed")) and all(bool(v.get("passed")) for v in (repair_lints or [])),
            "answer_generation": answer_lint,
            "repairs": repair_lints or [],
        }
        await self._write_json_artifact(rt=rt, relative_path="prompt_lint.json", payload=payload)

    async def _answer_pass(
        self,
        *,
        rt: Any,
        query: str,
        context_view: Any,
        extra_blocks: list[str],           # citable — enter citation contract
        background_blocks: list[str] | None = None,  # navigational only — not citable
        extra_marker_resolution: list[dict[str, Any]] | None = None,
        llm_max_tokens: int | None,
    ) -> tuple[str, dict[str, Any]]:
        contract_block = _render_output_contract_block(getattr(rt, "output_contract", None))
        answer_guidance = self._answer_guidance_block(query)
        # Compose the prompt and support rows by construction.  The support
        # offsets are shifted as parts are rendered; they are not recovered from
        # the finished prompt by regex/search.
        _marker_registry = self._evidence_marker_registry(context_view, extra_marker_resolution or [])
        setattr(context_view, "runtime_evidence_marker_registry", _marker_registry)

        prompt_parts: list[tuple[str, list[dict[str, Any]], int]] = []
        prompt_parts.append((f"User task:\n{query.strip()}", [], 0))
        if contract_block.strip():
            prompt_parts.append((contract_block.strip(), [], 0))

        context_prefix = "Context pack:\n"
        context_text = str(getattr(context_view, "text", "") or "").strip()
        prompt_parts.append((context_prefix + context_text, list(getattr(context_view, "visible_marker_support", []) or []), len(context_prefix)))

        extra_support_rows = [dict(row) for row in list(getattr(context_view, "runtime_extra_visible_marker_support", []) or []) if isinstance(row, dict)]
        zoom_rows = [row for row in extra_support_rows if str(row.get("marker_namespace") or "") == "zoom"]
        zoom_out_rows = [row for row in extra_support_rows if str(row.get("marker_namespace") or "") == "zoom_out"]
        other_extra_rows = [row for row in extra_support_rows if str(row.get("marker_namespace") or "") not in {"zoom", "zoom_out"}]

        for block in extra_blocks:
            block = str(block or "").strip()
            if not block:
                continue
            if block.startswith("Focused evidence:\n"):
                prompt_parts.append((block, zoom_rows, len("Focused evidence:\n")))
            elif block.startswith("Perspective evidence:\n"):
                prompt_parts.append((block, zoom_out_rows, len("Perspective evidence:\n")))
            else:
                # Unknown extra blocks are rendered, but they are not allowed to
                # create visible support unless explicit structured rows were
                # provided with matching non-standard namespace.
                prompt_parts.append((block, other_extra_rows, 0))

        # Background blocks are visible for navigation but excluded from the
        # citation contract — the LLM must not cite them.
        for block in (background_blocks or []):
            block = str(block or "").strip()
            if block:
                non_citable = _MARKER_GROUP_RE.sub("", block)
                prompt_parts.append((non_citable, [], 0))

        _pre_payload, _visible_support = self._render_prompt_parts_with_support(prompt_parts)
        _all_markers = sorted(
            self._available_markers(
                context_view,
                extra_blocks,
                marker_registry=_marker_registry,
                visible_marker_support=_visible_support,
            ),
            key=_marker_sort_key,
        )

        grounding_policy = (
            "## CIMA GROUNDING POLICY\n"
            "Use only the CIMA-provided context in this request as factual evidence.\n"
            "You may use general language ability to understand the task and write clearly, "
            "but you must not rely on training-time knowledge, memory, or external facts as factual support.\n"
            "If the provided context contains evidence that supports a factual answer, answer using that evidence.\n"
            "If the provided context does not contain enough evidence to answer factually, abstain with NOT ENOUGH INFO "
            "or a concise insufficient-evidence statement; do not guess and do not force a citation."
        )
        prompt_parts.append((grounding_policy, [], 0))
        if _all_markers:
            _marker_list = _render_marker_list(_all_markers)
            citation_constraint = (
                f"## CITATION CONSTRAINT\n"
                f"Allowed citation markers are exactly and only: {_marker_list}.\n"
                "No other citation markers exist.\n"
                "Use only these exact bracketed markers. Any other bracketed marker is invalid and must not appear.\n"
                "Do not infer additional marker numbers from the sequence; the list above is closed even if it has gaps.\n"
                "For every factual paragraph or bullet, cite at least one marker from the closed list.\n"
                "If none of the listed markers supports a claim, omit that claim rather than inventing or substituting a marker."
            )
            prompt_parts.append((citation_constraint, [], 0))
        if answer_guidance:
            prompt_parts.append((answer_guidance, [], 0))

        user_payload, _visible_support = self._render_prompt_parts_with_support(prompt_parts)
        setattr(context_view, "runtime_visible_marker_support", [dict(row) for row in _visible_support.values()])
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are CIMA Demonstrator. "
                    "Use only the CIMA-provided context in the current request as factual evidence; "
                    "do not use training-time knowledge, memory, or external facts as factual support. "
                    "Produce only the final user-visible answer. Every factual paragraph or bullet must include "
                    "one or more markers from the exact closed CITATION CONSTRAINT list. "
                    "If the context is insufficient, abstain honestly without forcing citations. "
                    "Do not use any marker not present in that list. "
                    "Do not emit XML, JSON, thoughts, tool calls or protocol tags."
                ),
            ),
            LLMMessage(role="user", content=user_payload),
        ]
        answer_prompt_lint = prompt_trace.build_prompt_lint(
            call_kind="answer_generation",
            messages=messages,
            allowed_markers=_all_markers,
            require_answer_grounding=True,
            marker_registry=_marker_registry,
            visible_marker_support=list(_visible_support.values()),
            task_family=self._prompt_lint_task_family(rt),
            output_format=self._prompt_lint_output_format(rt),
        )
        await self._write_prompt_lint(rt=rt, answer_lint=answer_prompt_lint)
        if not answer_prompt_lint.get("passed"):
            await self._record_llm_call(
                rt=rt,
                call_kind="answer_generation",
                messages=messages,
                params={
                    "temperature": self._llm_temperature,
                    "top_p": self._llm_top_p,
                    "repeat_penalty": self._llm_repeat_penalty,
                    "max_tokens": llm_max_tokens,
                },
                allowed_markers=_all_markers,
                context_id=self._ctx.last_snapshot_id(),
                prompt_lint=answer_prompt_lint,
                error={"error_class": "PromptLintError", "error": "; ".join(answer_prompt_lint.get("failures") or [])},
            )
            raise RuntimeError(f"Answer prompt failed CIMA prompt lint: {answer_prompt_lint.get('failures')}")

        await self._write_debug_json(
            rt=rt,
            relative_path="debug/answer_request.json",
            payload={
                "conversation_id": rt.conversation_id,
                "turn_id": rt.turn_id,
                "run_id": rt.run_id,
                "query": self._debug_truncate(query),
                "context_id": self._ctx.last_snapshot_id(),
                "context_items": len(getattr(context_view, "items", []) or []),
                "context_tokens_used": getattr(context_view, "tokens_used", None),
                "context_text": self._debug_truncate(getattr(context_view, "text", "") or ""),
                "extra_block_count": len([b for b in extra_blocks if str(b).strip()]),
                "llm_max_tokens": llm_max_tokens,
                "llm_temperature": self._llm_temperature,
                "llm_top_p": self._llm_top_p,
                "llm_repeat_penalty": self._llm_repeat_penalty,
                "messages": [
                    {
                        "role": m.role,
                        "content": self._debug_truncate(m.content or ""),
                    }
                    for m in messages
                ],
            },
        )

        # Generate first, validate, then publish. This avoids streaming an uncited
        # answer that would later need replacement in non-streaming evaluation runs.
        generation_started = time.monotonic()
        generation_error: dict[str, Any] | None = None
        try:
            answer, chunks = await self._generate_answer_text(messages, llm_max_tokens=llm_max_tokens)
        except Exception as exc:
            generation_error = {
                "error_class": type(exc).__name__,
                "error": str(exc),
            }
            await self._record_llm_call(
                rt=rt,
                call_kind="answer_generation",
                messages=messages,
                params={
                    "temperature": self._llm_temperature,
                    "top_p": self._llm_top_p,
                    "repeat_penalty": self._llm_repeat_penalty,
                    "max_tokens": llm_max_tokens,
                    "transport": "stream_text_or_complete_fallback",
                },
                allowed_markers=_all_markers,
                context_id=self._ctx.last_snapshot_id(),
                prompt_lint=answer_prompt_lint,
                error=generation_error,
            )
            await self._write_debug_json(
                rt=rt,
                relative_path="debug/answer_generation_result.json",
                payload={
                    "ok": False,
                    "elapsed_seconds": round(time.monotonic() - generation_started, 3),
                    **generation_error,
                },
            )
            raise
        await self._record_llm_call(
            rt=rt,
            call_kind="answer_generation",
            messages=messages,
            params={
                "temperature": self._llm_temperature,
                "top_p": self._llm_top_p,
                "repeat_penalty": self._llm_repeat_penalty,
                "max_tokens": llm_max_tokens,
                "transport": "stream_text_or_complete_fallback",
            },
            allowed_markers=_all_markers,
            context_id=self._ctx.last_snapshot_id(),
            prompt_lint=answer_prompt_lint,
            response_text=answer,
        )
        await self._write_debug_json(
            rt=rt,
            relative_path="debug/answer_generation_result.json",
            payload={
                "ok": True,
                "elapsed_seconds": round(time.monotonic() - generation_started, 3),
                "answer_chars": len(answer or ""),
                "chunk_count": len(chunks or []),
                "answer_preview": self._debug_truncate(answer or ""),
            },
        )
        raw_model_answer = answer
        cleaned_answer = self._sanitize_visible_answer(answer)
        if cleaned_answer != answer:
            answer = cleaned_answer
            chunks = [answer] if answer else []

        # C3-SAN-v1: run on the actual raw model output before visible/meta cleanup,
        # deterministic marker stripping, or repair changes the published answer.
        _c3_allowed = self._available_markers(
            context_view,
            extra_blocks,
            visible_marker_support=self._visible_marker_support(context_view, extra_blocks),
        )
        _c3_raw_model = c3_sanitize.sanitize(raw_model_answer, _c3_allowed).report

        validation = self._validate_citation_contract(answer, context_view=context_view, extra_blocks=extra_blocks)
        initial_validation = validation
        await self._write_debug_json(
            rt=rt,
            relative_path="debug/initial_citation_contract.json",
            payload={
                **initial_validation,
                "answer_chars": len(answer or ""),
                "answer_preview": self._debug_truncate(answer or ""),
            },
        )
        repair_attempted = False
        repaired_from = None
        sanitize_only = False
        citation_sanitization_reports: list[dict[str, Any]] = []
        sanitized_answer, sanitized_validation, sanitize_changed, sanitize_report = self._sanitize_answer_for_citation_contract(
            answer,
            validation=validation,
            context_view=context_view,
            extra_blocks=extra_blocks,
        )
        if sanitize_report:
            citation_sanitization_reports.append(sanitize_report)
        if sanitize_changed:
            answer = sanitized_answer
            chunks = [answer] if answer else []
            validation = sanitized_validation
            sanitize_only = True

        repair_decision = {
            "repair_attempted": False,
            "sanitize_only": sanitize_only,
            "reason": "citation_contract_passed_after_sanitize" if sanitize_only and validation.get("passed") else "citation_contract_passed" if validation.get("passed") else "pending_repair_decision",
            "validation": validation,
        }
        if self._should_repair_citation_contract(validation):
            repair_attempted = True
            repaired_from = validation
            repair_decision = {
                "repair_attempted": True,
                "sanitize_only": sanitize_only,
                "reason": "invalid_or_substantive_uncited_claims",
                "validation": validation,
            }
            await self._write_debug_json(
                rt=rt,
                relative_path="debug/repair_decision.json",
                payload=repair_decision,
            )
            repair = await self._repair_citation_contract(
                rt=rt,
                query=query,
                answer_text=answer,
                context_view=context_view,
                extra_blocks=extra_blocks,
                answer_guidance=answer_guidance,
                llm_max_tokens=llm_max_tokens,
                answer_prompt_lint=answer_prompt_lint,
                extra_marker_resolution=extra_marker_resolution or [],
            )
            repair = self._sanitize_visible_answer(repair.strip())
            if repair:
                answer = repair
                chunks = [repair]
                validation = self._validate_citation_contract(answer, context_view=context_view, extra_blocks=extra_blocks)
                sanitized_answer, sanitized_validation, sanitize_changed, sanitize_report = self._sanitize_answer_for_citation_contract(
                    answer,
                    validation=validation,
                    context_view=context_view,
                    extra_blocks=extra_blocks,
                )
                if sanitize_report:
                    citation_sanitization_reports.append(sanitize_report)
                if sanitize_changed:
                    answer = sanitized_answer
                    chunks = [answer] if answer else []
                    validation = sanitized_validation
                    sanitize_only = True
        else:
            await self._write_debug_json(
                rt=rt,
                relative_path="debug/repair_decision.json",
                payload=repair_decision,
            )

        if answer:
            # The answer is validated before it is published, so original LLM
            # stream-token granularity no longer buys real TTFT. Coalesce the
            # accepted visible text into bounded deltas to avoid overflowing the
            # in-process SSE queue when the whole answer is emitted at turn end.
            for token in _visible_answer_chunks(answer, chunks):
                await self._stream.publish(KimaDelta(
                    type=KimaDeltaType.TOKEN,
                    conversation_id=rt.conversation_id,
                    token=token,
                ))
        # C3-SAN-v1: run on the final published answer (after all passes).
        _c3_published = c3_sanitize.sanitize(answer, _c3_allowed).report
        _c3a_abstention = c3_sanitize.build_c3a_abstention_report(
            answer=answer,
            allowed_markers=_c3_allowed,
            context_view_id=self._ctx.last_snapshot_id(),
            inspected_markers=sorted(_c3_allowed, key=_marker_sort_key),
            zoom_attempted=bool(extra_blocks),
            zoom_out_attempted=any("[P" in (block or "") for block in extra_blocks),
            extra_trace={"context_item_count": len(getattr(context_view, "items", []) or [])},
        )
        _answer_type = str(_c3a_abstention.get("answer_type") or "factual_answer")
        _factual_citations_required = _answer_type != "insufficient_evidence"
        _normal_passed = bool(validation.get("passed")) if _factual_citations_required else False
        _published_integrity_passed = bool(_normal_passed or _c3a_abstention.get("passed") is True)
        _publication_gate = c3_sanitize.build_publication_gate(
            raw_answer=raw_model_answer,
            published_answer=answer,
            published_integrity_passed=_published_integrity_passed,
            c3_published_report=_c3_published,
            c3a_abstention_report=_c3a_abstention,
            generation_passed=bool(answer.strip()),
            generation_failure_kind="empty_generation" if not answer.strip() else None,
            factual_citations_required=_factual_citations_required,
            sanitization_applied=bool(citation_sanitization_reports),
        )
        citation_contract = {
            **validation,
            "schema_version": "cima_demo.citation_contract.v2",
            "answer_type": _answer_type,
            "factual_citations_required": _factual_citations_required,
            "legacy_marker_citation_passed": bool(validation.get("passed")),
            "published_integrity_passed": _published_integrity_passed,
            "passed": _publication_gate["publishable"],
            "publication_gate": _publication_gate,
            "publication_status": _publication_gate["publication_status"],
            "publishable": _publication_gate["publishable"],
            "blocked_by_cima": _publication_gate["blocked_by_cima"],
            "blocked_reason": _publication_gate["blocked_reason"],
            "invalid_published_as_valid": _publication_gate["invalid_published_as_valid"],
            "repair_attempted": repair_attempted,
            "sanitize_only": sanitize_only,
            "deterministic_sanitization_applied": bool(citation_sanitization_reports),
            "citation_sanitization_reports": citation_sanitization_reports,
            "repaired_from": repaired_from,
            "initial_validation": initial_validation,
            # C3-SAN-v1 independent contract — raw vs published split.
            # raw_model_answer_text is persisted in demo artifacts so an external
            # validator can verify that the sanitizer neither added citations nor
            # rewrote non-citation text.
            "allowed_markers": sorted(_c3_allowed, key=_marker_sort_key),
            "raw_model_answer_text": raw_model_answer,
            "published_answer_text": answer,
            "c3_raw_model": _c3_raw_model,
            "c3_published": _c3_published,
            "c3a_traceable_abstention": _c3a_abstention,
            "visible_marker_support": self._visible_marker_support_rows(context_view, extra_blocks),
        }
        return answer, citation_contract

    async def _generate_answer_text(self, messages: list[LLMMessage], *, llm_max_tokens: int | None) -> tuple[str, list[str]]:
        chunks: list[str] = []
        try:
            async for token in self._llm.stream_text(
                messages=messages,
                temperature=self._llm_temperature,
                top_p=self._llm_top_p,
                repeat_penalty=self._llm_repeat_penalty,
                max_tokens=llm_max_tokens,
            ):
                chunks.append(token)
        except Exception as exc:
            log.warning("Demo answer streaming failed — falling back to complete(): %s", exc)
            chunks = []
        answer = "".join(chunks).strip()
        if answer:
            return answer, chunks
        fallback = await self._llm.complete(
            messages=messages,
            temperature=self._llm_temperature,
            max_tokens=llm_max_tokens or 1024,
        )
        fallback = fallback.strip()
        return fallback, [fallback] if fallback else []

    def _sanitize_visible_answer(self, answer_text: str) -> str:
        """Remove internal repair scaffolding and diagnostic meta-notes.

        User-visible answers must not expose the control flow used to enforce
        citations. This sanitizer strips repair prefaces, Markdown separators,
        and evidence-gap notes produced by the repair prompt. It is deliberately
        conservative: unsupported substantive claims are handled by the citation
        validator/repair policy, while meta text about omitted claims is simply
        not part of the answer.
        """
        text = (answer_text or "").strip()
        if not text:
            return ""
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            normalized = self._normalize_answer_line(line)
            if not normalized:
                lines.append(raw_line)
                continue
            if self._is_visible_scaffolding_or_meta(normalized):
                continue
            lines.append(raw_line.rstrip())
        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _normalize_answer_line(self, line: str) -> str:
        normalized = re.sub(r"[*_`#]", "", line or "").strip()
        # Unwrap common Markdown parenthetical notes: *(Note: ...)* -> Note: ...
        normalized = re.sub(r"^\((note\s*:.*)\)\.?$", r"\1", normalized, flags=re.IGNORECASE).strip()
        return normalized

    def _is_visible_scaffolding_or_meta(self, normalized: str) -> bool:
        if not normalized:
            return False
        if re.fullmatch(r"[-—–_\s]{3,}", normalized):
            return True
        if re.match(
            r"^here\s+is\s+(the\s+)?(corrected|revised|updated)?\s*(summary|answer)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        if re.match(
            r"^corrected\s+(summary|answer)\s*(with\s+valid\s+citations)?\s*:?$",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        # Meta notes generated by the repair prompt, not answer content.
        if re.match(r"^note\s*:", normalized, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(claims?\s+about|lacked\s+direct\s+contextual\s+support|were\s+omitted|has\s+been\s+omitted|removed\s+because|available\s+evidence\s+is\s+limited|evidence\s+is\s+limited|context\s+is\s+limited|source\s+material\s+is\s+limited|broader\s+[^.!?]{0,80}\s+remain(?:s)?\s+(?:unspecified|unclear|unknown|not\s+covered))\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    def _sanitize_answer_for_citation_contract(
        self,
        answer_text: str,
        *,
        validation: dict[str, Any],
        context_view: Any,
        extra_blocks: list[str],
    ) -> tuple[str, dict[str, Any], bool, dict[str, Any] | None]:
        """Apply deterministic cleanups before spending a second LLM call.

        The local model often produces a good cited summary plus one uncited
        closing slogan/meta note. Such text is not needed for the answer and
        should be removed locally rather than triggering an expensive repair.
        """
        answer = self._sanitize_visible_answer(answer_text)
        changed = answer != (answer_text or "").strip()
        current = self._validate_citation_contract(answer, context_view=context_view, extra_blocks=extra_blocks)
        report: dict[str, Any] | None = None

        # C3-SAN-v1: deterministic citation sanitization — remove only markers
        # not in the closed allowed set. Never adds or substitutes. Invariants
        # B–F verified. This is syntactic enforcement, not semantic repair.
        allowed = self._available_markers(
            context_view,
            extra_blocks,
            visible_marker_support=self._visible_marker_support(context_view, extra_blocks),
        )
        c3_result = c3_sanitize.sanitize(answer, allowed)
        if c3_result.removed_invalid_markers and c3_result.sanitized_answer != answer:
            changed = True
            answer = c3_result.sanitized_answer
            current = self._validate_citation_contract(answer, context_view=context_view, extra_blocks=extra_blocks)
            current["citation_sanitization"] = c3_result.report
            report = c3_result.report

        # Drop final uncited framing/conclusion even when the validator would
        # otherwise ignore it as non-claim. The demo publication contract is
        # stricter for visible output: no uncited closing slogans.
        if current.get("requires_citations") and current.get("valid_cited_markers") and not current.get("invalid_cited_markers"):
            cleaned = self._drop_trailing_uncited_closers(answer, current, context_view=context_view, extra_blocks=extra_blocks)
            if cleaned != answer:
                changed = True
                answer = cleaned
                current = self._validate_citation_contract(answer, context_view=context_view, extra_blocks=extra_blocks)

        return answer, current, changed, report

    def _strip_invalid_citation_markers(self, answer_text: str, allowed: set[str]) -> tuple[str, dict[str, Any]]:
        """Remove citation markers that are not in the closed CIMA marker set.

        This function never adds or substitutes evidence. If a bracket contains
        both valid and invalid markers, it preserves only the valid subset; if it
        contains no valid markers, it removes the bracket entirely.
        """
        allowed = {str(marker) for marker in allowed}
        removed: list[str] = []
        preserved: list[str] = []

        def repl(match: re.Match[str]) -> str:
            parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
            keep = [part for part in parts if part in allowed]
            drop = [part for part in parts if part not in allowed]
            removed.extend(drop)
            preserved.extend(keep)
            if not keep:
                return ""
            return "[" + ", ".join(keep) + "]"

        stripped = _MARKER_GROUP_RE.sub(repl, answer_text or "")
        # Clean whitespace artifacts left by removing a trailing citation. Keep
        # line structure intact so block validation remains meaningful.
        cleaned_lines: list[str] = []
        for line in stripped.splitlines():
            line = re.sub(r"\s+([.,;:!?])", r"\1", line)
            line = re.sub(r" {2,}", " ", line)
            cleaned_lines.append(line.rstrip())
        stripped = "\n".join(cleaned_lines).strip()
        return stripped, {
            "applied": bool(removed),
            "method": "deterministic_strip_invalid_markers",
            "removed_invalid_markers": list(dict.fromkeys(removed)),
            "preserved_valid_markers_in_mixed_groups": list(dict.fromkeys(preserved)),
        }

    def _drop_trailing_uncited_closers(
        self,
        answer_text: str,
        validation: dict[str, Any],
        *,
        context_view: Any,
        extra_blocks: list[str],
    ) -> str:
        allowed = self._available_markers(
            context_view,
            extra_blocks,
            visible_marker_support=self._visible_marker_support(context_view, extra_blocks),
        )
        text = (answer_text or "").strip()
        if not text:
            return ""
        paragraphs = re.split(r"\n\s*\n", text)
        changed = False
        while paragraphs:
            last = paragraphs[-1].strip()
            if not last:
                paragraphs.pop()
                changed = True
                continue
            if any(marker in allowed for marker in self._extract_markers(last)):
                break
            normalized = self._normalize_answer_line(last)
            if self._is_trailing_uncited_closer(normalized):
                paragraphs.pop()
                changed = True
                continue
            break
        return "\n\n".join(p.strip() for p in paragraphs if p.strip()).strip() if changed else text

    def _is_trailing_uncited_closer(self, normalized: str) -> bool:
        normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", normalized or "").strip()
        normalized = re.sub(r"[*_`#]", "", normalized).strip()
        if not normalized:
            return True
        if len(normalized) > 520:
            return False
        if self._is_visible_scaffolding_or_meta(normalized):
            return True
        return bool(re.match(
            r"^(overall|in\s+summary|in\s+conclusion|to\s+summarize|the\s+meeting|this\s+meeting|the\s+discussion|this\s+discussion)\b",
            normalized,
            flags=re.IGNORECASE,
        ))

    def _should_repair_citation_contract(self, validation: dict[str, Any]) -> bool:
        if not validation.get("requires_citations"):
            return False
        if validation.get("passed") is True:
            return False
        # Invalid or missing marker references are real lineage failures and
        # need a repair attempt. Framing-only failures are handled by the local
        # validator/sanitizer to avoid a second expensive LLM call.
        if validation.get("invalid_cited_markers"):
            return True
        if not validation.get("valid_cited_markers"):
            return True
        return bool(validation.get("uncited_answer_block_count", 0))

    async def _repair_citation_contract(
        self,
        *,
        rt: Any,
        query: str,
        answer_text: str,
        context_view: Any,
        extra_blocks: list[str],
        answer_guidance: str,
        llm_max_tokens: int | None,
        answer_prompt_lint: dict[str, Any],
        extra_marker_resolution: list[dict[str, Any]] | None = None,
    ) -> str:
        context_parts = ["Context pack:\n" + context_view.text.strip()]
        for block in extra_blocks:
            block = block.strip()
            if block:
                context_parts.append(block)
        _repair_registry = self._evidence_marker_registry(context_view, extra_marker_resolution or [])
        setattr(context_view, "runtime_evidence_marker_registry", _repair_registry)
        _repair_visible_support = self._visible_marker_support(context_view, extra_blocks)
        _repair_markers = sorted(
            self._available_markers(
                context_view,
                extra_blocks,
                marker_registry=_repair_registry,
                visible_marker_support=_repair_visible_support,
            ),
            key=_marker_sort_key,
        )
        if _repair_markers:
            _marker_list = _render_marker_list(_repair_markers)
            _constraint = (
                f"CITATION CONSTRAINT — Allowed citation markers are exactly and only: {_marker_list}.\n"
                "No other citation markers exist. Use only markers listed exactly in that set. Any unlisted marker does not exist. "
                "For each citation in the previous answer that used an unlisted marker, remove the unsupported claim entirely — do not substitute another marker."
            )
        else:
            _constraint = ""
        grounding = (
            "## CIMA GROUNDING POLICY\n"
            "Use only the CIMA-provided context in this request as factual evidence.\n"
            "Do not rely on training-time knowledge, memory, or external facts as factual support.\n"
            "If the provided context does not contain enough evidence to answer factually, abstain with NOT ENOUGH INFO "
            "or a concise insufficient-evidence statement; do not guess and do not force a citation."
        )
        payload = "\n\n".join(
            part for part in [
                f"User task:\n{query.strip()}",
                grounding,
                *context_parts,
                "Previous answer that violated the citation contract:\n" + (answer_text or "").strip(),
                answer_guidance,
                _constraint,
                (
                    "Citation repair task:\n"
                    "- Rewrite the previous answer so it satisfies the citation contract.\n"
                    "- Preserve only claims supported by the CIMA-provided context.\n"
                    "- Every factual paragraph or bullet MUST include at least one valid inline citation from the list above.\n"
                    "- Use ONLY markers listed in the CITATION CONSTRAINT above — do not invent new ones.\n"
                    "- If the context does not support a factual answer, say NOT ENOUGH INFO or state that evidence is insufficient. Do not attach a citation as if it supported the absence of evidence.\n"
                    "- Return only the corrected final answer."
                ),
            ] if part
        )
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are CIMA Demonstrator's citation repair pass. "
                    "Use only the CIMA-provided context as factual evidence and do not use training-time knowledge as support. "
                    "Return only a corrected final answer. "
                    "Use ONLY the citation markers listed in the CITATION CONSTRAINT — no others."
                ),
            ),
            LLMMessage(role="user", content=payload),
        ]
        repair_lint = prompt_trace.build_prompt_lint(
            call_kind="citation_repair",
            messages=messages,
            allowed_markers=_repair_markers,
            require_answer_grounding=True,
            marker_registry=_repair_registry,
            visible_marker_support=list(_repair_visible_support.values()),
        )
        await self._write_prompt_lint(rt=rt, answer_lint=answer_prompt_lint, repair_lints=[repair_lint])
        if not repair_lint.get("passed"):
            await self._record_llm_call(
                rt=rt,
                call_kind="citation_repair",
                messages=messages,
                params={"temperature": 0.0, "max_tokens": llm_max_tokens or 1024},
                allowed_markers=_repair_markers,
                context_id=self._ctx.last_snapshot_id(),
                prompt_lint=repair_lint,
                error={"error_class": "PromptLintError", "error": "; ".join(repair_lint.get("failures") or [])},
            )
            raise RuntimeError(f"Citation repair prompt failed CIMA prompt lint: {repair_lint.get('failures')}")
        try:
            repair = await self._llm.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=llm_max_tokens or 1024,
            )
        except Exception as exc:
            await self._record_llm_call(
                rt=rt,
                call_kind="citation_repair",
                messages=messages,
                params={"temperature": 0.0, "max_tokens": llm_max_tokens or 1024},
                allowed_markers=_repair_markers,
                context_id=self._ctx.last_snapshot_id(),
                prompt_lint=repair_lint,
                error={"error_class": type(exc).__name__, "error": str(exc)},
            )
            raise
        await self._record_llm_call(
            rt=rt,
            call_kind="citation_repair",
            messages=messages,
            params={"temperature": 0.0, "max_tokens": llm_max_tokens or 1024},
            allowed_markers=_repair_markers,
            context_id=self._ctx.last_snapshot_id(),
            prompt_lint=repair_lint,
            response_text=repair,
        )
        return repair

    def _visible_support_from_structured_rows(self, rows: list[dict[str, Any]], rendered_block: str, *, namespace: str) -> list[dict[str, Any]]:
        """Construct visible support from already-authorized marker rows.

        This is a compatibility path for tests/legacy runtime components that do
        not yet return visible_marker_support. It does not discover markers from
        rendered text; it only anchors markers already present in structured
        resolution rows.
        """
        out: list[dict[str, Any]] = []
        cursor = 0
        for row in rows or []:
            marker = str(row.get("marker") or "")
            if not marker:
                continue
            token = f"[{marker}]"
            start = rendered_block.find(token, cursor)
            if start < 0:
                start = rendered_block.find(token)
            if start < 0:
                continue
            end = rendered_block.find("\n\n", start + len(token))
            if end < 0:
                end = len(rendered_block)
            block = rendered_block[start:end]
            cursor = end
            out.append({
                "marker": marker,
                "marker_namespace": namespace,
                "marker_uid": f"{namespace}:{marker}",
                "ref_kind": str(row.get("ref_kind") or ""),
                "ref_id": str(row.get("ref_id") or ""),
                "support_source": f"{namespace}_block",
                "rendered_text_sha256": _sha256_text(rendered_block),
                "visible_text_sha256": _sha256_text(block),
                "visible_text_preview": block[:240],
                "visible_char_count": len(block),
                "context_truncated_for_budget": False,
                "visible_support_scope": f"{namespace}_block",
                "prompt_char_start": start,
                "prompt_char_end": end,
                "visible_char_start": start,
                "visible_char_end": end,
                "prompt_offsets_exact": True,
            })
        return out

    def _support_rows_by_uid(self, rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            marker = str(row.get("marker") or "").strip()
            if not marker:
                continue
            materialized = dict(row)
            materialized.setdefault("marker_namespace", "runtime_context")
            uid = _marker_uid_from_row(materialized)
            if not uid:
                continue
            materialized["marker_uid"] = uid
            out[uid] = materialized
        return out

    def _visible_marker_support(self, context_view: Any, extra_blocks: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Return structured visible-support rows keyed by marker_uid.

        Marker support must originate from runtime metadata, never from regex
        scanning arbitrary rendered text. The public marker label (S1/E1/P1) is
        not the internal identity; marker_uid (namespace:marker) is.  This
        prevents collisions between runtime context, zoom evidence and zoom-out
        perspective blocks.
        """
        out: dict[str, dict[str, Any]] = {}
        runtime_rows = list(getattr(context_view, "runtime_visible_marker_support", []) or [])
        if runtime_rows:
            # After prompt composition, this is the authoritative visible-support
            # view because it contains shifted prompt offsets and prompt hashes.
            out.update(self._support_rows_by_uid(runtime_rows))
            return out
        out.update(self._support_rows_by_uid(list(getattr(context_view, "visible_marker_support", []) or [])))
        out.update(self._support_rows_by_uid(list(getattr(context_view, "runtime_extra_visible_marker_support", []) or [])))

        # Publication/strict runs must not manufacture visible support from
        # context_view.items.  The compatibility fallback exists only for older
        # unit tests or explicitly opted-in legacy artifacts.
        if bool(getattr(context_view, "allow_legacy_visible_support", False)):
            for item in list(getattr(context_view, "items", []) or []):
                if not isinstance(item, dict):
                    continue
                marker = str(item.get("marker") or "").strip()
                visible_text = str(item.get("content") or "").strip()
                if not marker or not visible_text:
                    continue
                row = {
                    "marker": marker,
                    "marker_namespace": "legacy_context_item",
                    "marker_uid": f"legacy_context_item:{marker}",
                    "ref_kind": str(item.get("ref_kind") or ""),
                    "ref_id": str(item.get("ref_id") or ""),
                    "support_source": "context_view_items_legacy",
                    "visible_text_sha256": _sha256_text(visible_text),
                    "visible_text_preview": visible_text[:240],
                    "visible_char_count": len(visible_text),
                    "context_truncated_for_budget": bool(item.get("context_truncated_for_budget")),
                    "visible_support_scope": "legacy_prompt_item",
                    "prompt_offsets_exact": False,
                    "prompt_anchor_method": "legacy_unanchored",
                }
                out[row["marker_uid"]] = row
        return out

    def _visible_support_label_collisions(self, support: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in support.values():
            marker = str(row.get("marker") or "").strip()
            if marker:
                by_label.setdefault(marker, []).append(row)
        collisions: dict[str, list[str]] = {}
        for marker, rows in by_label.items():
            # Strict publication rule: a visible marker label may designate only
            # one evidence identity. Multiple verified prompt occurrences are
            # allowed only when they point to the same ref/source/span identity.
            signatures = {_visible_support_signature(row) for row in rows}
            if len(signatures) > 1:
                collisions[marker] = sorted(str(row.get("marker_uid") or "") for row in rows if str(row.get("marker_uid") or ""))
        return collisions

    def _visible_support_labels(self, support: dict[str, dict[str, Any]]) -> set[str]:
        collisions = self._visible_support_label_collisions(support)
        return {
            str(row.get("marker") or "")
            for row in support.values()
            if str(row.get("marker") or "")
            and str(row.get("marker") or "") not in collisions
            and _row_has_verified_prompt_anchor(row)
        }

    def _render_prompt_parts_with_support(self, parts: list[tuple[str, list[dict[str, Any]], int]]) -> tuple[str, dict[str, dict[str, Any]]]:
        """Render prompt parts and shift structured support offsets by construction.

        This is the strict path used for new runs.  Offsets are not recovered
        by searching the finished prompt; they are shifted from support rows
        emitted by the renderer that created each block.
        """
        rendered_parts: list[str] = []
        support_by_uid: dict[str, dict[str, Any]] = {}
        cursor = 0
        for part, rows, row_shift in parts:
            part = str(part or "")
            if not part:
                continue
            if rendered_parts:
                cursor += 2  # the "\n\n" separator inserted by join
            part_start = cursor
            rendered_parts.append(part)
            part_end = part_start + len(part)
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                marker = str(row.get("marker") or "").strip()
                if not marker:
                    continue
                materialized = dict(row)
                materialized.setdefault("marker_namespace", "runtime_context")
                uid = _marker_uid_from_row(materialized)
                if not uid:
                    continue
                materialized["marker_uid"] = uid
                rel_start = int(materialized.get("prompt_char_start") if materialized.get("prompt_char_start") is not None else materialized.get("visible_char_start") or 0)
                rel_end = int(materialized.get("prompt_char_end") if materialized.get("prompt_char_end") is not None else materialized.get("visible_char_end") or (rel_start + int(materialized.get("visible_char_count") or 0)))
                abs_start = part_start + int(row_shift or 0) + rel_start
                abs_end = part_start + int(row_shift or 0) + rel_end
                exact = part_start <= abs_start <= abs_end <= part_end and abs_end > abs_start
                if not exact:
                    abs_start = max(part_start, min(part_end, abs_start))
                    abs_end = max(abs_start, min(part_end, abs_end))
                materialized["prompt_message_index"] = 1
                materialized["prompt_char_start"] = abs_start
                materialized["prompt_char_end"] = abs_end
                materialized["prompt_offsets_exact"] = bool(exact)
                materialized["prompt_anchor_method"] = "render_shift" if exact else "render_shift_clamped"
                materialized["prompt_part_sha256"] = _sha256_text(part)
                support_by_uid[uid] = materialized
            cursor += len(part)
        payload = "\n\n".join(rendered_parts)
        prompt_sha = _sha256_text(payload)
        for row in support_by_uid.values():
            start = int(row.get("prompt_char_start") or 0)
            end = int(row.get("prompt_char_end") or start)
            visible_slice = payload[start:end]
            marker = str(row.get("marker") or "").strip()
            marker_token = f"[{marker}]" if marker else ""
            preview = str(row.get("visible_text_preview") or "").strip()
            # Offsets are exact only if the rendered slice can be verified
            # against the expected marker token and visible preview.  Merely
            # being in bounds is not sufficient evidence.
            slice_verified = bool(visible_slice) and (not marker_token or marker_token in visible_slice) and (not preview or preview in visible_slice)
            row["prompt_sha256"] = prompt_sha
            row["visible_slice_sha256"] = _sha256_text(visible_slice)
            row["visible_slice_preview"] = visible_slice[:240]
            row["visible_slice_verified"] = bool(slice_verified)
            if row.get("prompt_offsets_exact") is True and slice_verified:
                row["prompt_offsets_exact"] = True
                row["prompt_anchor_method"] = "render_shift_verified"
            else:
                row["prompt_offsets_exact"] = False
                row["prompt_anchor_method"] = "render_shift_unverified" if row.get("prompt_anchor_method") == "render_shift" else row.get("prompt_anchor_method", "render_shift_unverified")
        return payload, support_by_uid

    def _anchor_visible_support_to_prompt(self, support: dict[str, dict[str, Any]], prompt_text: str) -> dict[str, dict[str, Any]]:
        """Compatibility anchoring path for legacy tests/artifacts.

        New runtime paths use `_render_prompt_parts_with_support`.  This method
        may locate rows post-hoc, so it deliberately reports anchored—not exact—
        offsets.
        """
        anchored: dict[str, dict[str, Any]] = {}
        cursor = 0
        for key, row in sorted(support.items(), key=lambda kv: _marker_sort_key(str(kv[1].get("marker") or kv[0]))):
            materialized = dict(row)
            marker = str(materialized.get("marker") or key).split(":")[-1]
            uid = _marker_uid_from_row(materialized) or str(key)
            marker_token = f"[{marker}]"
            preview = str(materialized.get("visible_text_preview") or "")[:80]
            pos = prompt_text.find(marker_token, cursor)
            chosen = -1
            while pos >= 0:
                window = prompt_text[pos:pos + max(500, len(preview) + 80)]
                if not preview or preview in window:
                    chosen = pos
                    break
                pos = prompt_text.find(marker_token, pos + len(marker_token))
            if chosen >= 0:
                next_boundary = prompt_text.find("\n\n", chosen + len(marker_token))
                if next_boundary < 0:
                    next_boundary = min(len(prompt_text), chosen + int(materialized.get("visible_char_count") or 0) + len(marker_token) + 16)
                materialized["prompt_message_index"] = 1
                materialized["prompt_char_start"] = chosen
                materialized["prompt_char_end"] = next_boundary
                materialized["prompt_offsets_exact"] = False
                materialized["prompt_anchor_method"] = "posthoc_marker_preview"
                materialized["prompt_sha256"] = _sha256_text(prompt_text)
                materialized["visible_slice_sha256"] = _sha256_text(prompt_text[chosen:next_boundary])
                cursor = next_boundary
            else:
                materialized.setdefault("prompt_message_index", 1)
                materialized["prompt_offsets_exact"] = False
                materialized["prompt_anchor_method"] = "not_found"
            materialized["marker_uid"] = uid
            anchored[uid] = materialized
        return anchored

    def _visible_marker_support_rows(self, context_view: Any, extra_blocks: list[str] | None = None) -> list[dict[str, Any]]:
        rows = list(self._visible_marker_support(context_view, extra_blocks).values())
        return [dict(row) for row in sorted(rows, key=lambda row: (_marker_sort_key(str(row.get("marker") or "")), str(row.get("marker_uid") or "")))]

    def _evidence_marker_registry(self, context_view: Any, extra_marker_resolution: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _append(row: dict[str, Any]) -> None:
            marker = str(row.get("marker") or "").strip()
            if not marker:
                return
            key = (
                marker,
                str(row.get("marker_namespace") or row.get("support_source") or "runtime_context"),
                str(row.get("ref_kind") or ""),
                str(row.get("ref_id") or ""),
            )
            if key in seen:
                return
            payload = dict(row)
            payload.setdefault("marker_namespace", key[1])
            payload.setdefault("marker_uid", f"{key[1]}:{marker}")
            rows.append(payload)
            seen.add(key)

        for row in list(getattr(context_view, "evidence_marker_registry", []) or []):
            if isinstance(row, dict):
                _append(row)
        extra_rows = [dict(row) for row in (extra_marker_resolution or []) if str(row.get("marker") or "")]
        if extra_rows:
            extra_items = [
                {
                    "marker": str(row.get("marker") or ""),
                    "ref_kind": str(row.get("ref_kind") or "citem"),
                    "ref_id": str(row.get("ref_id") or ""),
                    "marker_namespace": str(row.get("marker_namespace") or row.get("support_source") or "runtime_context"),
                }
                for row in extra_rows
            ]
            for entry in build_registry(items=extra_items, marker_resolution=extra_rows):
                _append(entry.to_dict())
        return rows

    def _available_markers(
        self,
        context_view: Any,
        extra_blocks: list[str],
        *,
        marker_registry: list[dict[str, Any]] | None = None,
        visible_marker_support: dict[str, dict[str, Any]] | None = None,
    ) -> set[str]:
        """Return the closed citation set from structured registry only.

        Do not scan arbitrary rendered text.  Source text and snippets can
        contain bracketed strings that look like S# markers but are not CIMA
        citation labels.  The registry is the authority.
        """
        rows = marker_registry
        if rows is None:
            rows = list(getattr(context_view, "runtime_evidence_marker_registry", []) or [])
        if not rows:
            rows = list(getattr(context_view, "evidence_marker_registry", []) or [])
        markers = {
            str(row.get("marker") or "")
            for row in rows or []
            if str(row.get("marker") or "")
            and row.get("citable") is True
            and str(row.get("resolution_status") or "") in {"source_span", "summary_witness"}
        }
        if visible_marker_support is not None:
            collisions = self._visible_support_label_collisions(visible_marker_support)
            if collisions:
                try:
                    setattr(context_view, "runtime_visible_marker_label_collisions", collisions)
                except Exception:
                    pass
            markers &= self._visible_support_labels(visible_marker_support)
            return markers
        if markers:
            return markers
        # Strict publication/runtime mode: do not infer allowed markers from
        # context_view.items.  A marker must be both registry-citable and visibly
        # rendered through structured support metadata.
        return set()

    def _validate_citation_contract(self, answer_text: str, *, context_view: Any, extra_blocks: list[str]) -> dict[str, Any]:
        visible_support = self._visible_marker_support(context_view, extra_blocks)
        candidate_allowed = self._available_markers(context_view, extra_blocks, visible_marker_support=None)
        allowed = self._available_markers(context_view, extra_blocks, visible_marker_support=visible_support)
        cited = self._extract_markers(answer_text)
        valid = [marker for marker in cited if marker in allowed]
        invalid = [marker for marker in cited if marker not in allowed]
        visible_labels = self._visible_support_labels(visible_support)
        visible_collisions = self._visible_support_label_collisions(visible_support)
        visible_cited = [marker for marker in cited if marker in visible_labels]
        cited_without_visible_support = [marker for marker in cited if marker not in visible_labels]
        visible_support_coverage = (len(visible_cited) / len(cited)) if cited else (1.0 if not valid else 0.0)
        answer_type = "insufficient_evidence" if c3_sanitize.is_insufficient_evidence_answer(answer_text) else "factual_answer"
        requires = bool(allowed) and answer_type != "insufficient_evidence"
        # Treat paragraphs and bullets as claim blocks. Headings without factual
        # content are ignored, but any substantive block in a context-grounded
        # answer must carry a valid marker. Pure insufficient-evidence abstentions
        # are not factual claims and are covered by C3A, not normal C3.
        blocks = self._answer_blocks(answer_text) if requires else []
        uncited_blocks = [block for block in blocks if not any(marker in allowed for marker in self._extract_markers(block))]
        passed = (not requires) or (bool(valid) and not invalid and not uncited_blocks)
        return {
            "requires_citations": requires,
            "factual_citations_required": requires,
            "answer_type": answer_type,
            "passed": passed,
            "cited_markers": cited,
            "valid_cited_markers": list(dict.fromkeys(valid)),
            "invalid_cited_markers": list(dict.fromkeys(invalid)),
            "cited_markers_with_visible_support": list(dict.fromkeys(visible_cited)),
            "cited_markers_without_visible_support": list(dict.fromkeys(cited_without_visible_support)),
            "visible_marker_anchor_coverage": round(visible_support_coverage, 4),
            "visible_prompt_support_coverage_deprecated_alias": round(visible_support_coverage, 4),
            "metric_notes": {
                "visible_prompt_support_coverage_deprecated_alias": "Deprecated alias for visible_marker_anchor_coverage; this is structural prompt-anchor coverage, not semantic entailment."
            },
            "visible_marker_support_count": len(visible_support),
            "visible_marker_label_collisions": visible_collisions,
            "available_marker_count": len(allowed),
            "candidate_allowed_marker_count": len(candidate_allowed),
            "dropped_allowed_marker_count": max(0, len(candidate_allowed) - len(allowed)),
            "answer_block_count": len(blocks),
            "uncited_answer_block_count": len(uncited_blocks),
            "uncited_answer_block_preview": [block[:180] for block in uncited_blocks[:5]],
        }

    def _is_non_claim_block(self, block: str, *, position: int) -> bool:
        normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", block or "").strip()
        normalized = re.sub(r"[*_`#]", "", normalized).strip()
        if not normalized:
            return True
        if self._is_visible_scaffolding_or_meta(normalized):
            return True
        if self._extract_markers(normalized):
            return False
        if len(normalized) < 24:
            return True
        if normalized.endswith(":") and len(normalized) < 160:
            return True
        if len(normalized) < 140 and not re.search(r"[.!?]", normalized):
            return True
        if position <= 2 and len(normalized) < 420 and re.match(
            r"^(this|the|overall,?\s+the)\s+(meeting|document|discussion|source|text)\s+",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        if len(normalized) < 420 and re.match(
            r"^(this|the|overall,?\s+the)\s+(meeting|document|discussion|source|text)\s+"
            r"(addressed|explored|discussed|highlighted|covered|focused|identified|summarized)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        if position <= 2 and len(normalized) < 160 and re.match(
            r"^here\s+is\s+(the\s+)?(corrected\s+)?(summary|answer)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        # Explicit evidence-gap notes are meta-reporting about the support set,
        # not substantive source claims. They may appear when the model is asked
        # to state what was not evidenced. Do not let them force an expensive
        # repair loop or fail an otherwise grounded answer.
        if re.search(
            r"\b(no direct evidence cited|evidence was insufficient|insufficient evidence|not enough evidence|no concrete examples|limited data|available evidence is limited|evidence is limited|context is limited|source material is limited|lacked direct contextual support|were omitted|broader\s+[^.!?]{0,80}\s+remain(?:s)?\s+(?:unspecified|unclear|unknown|not covered)|without accessible evidence|resolved.*evidence text is unavailable|cannot be classified as|claim cannot be|evidence text.*not bundled|only metadata)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    def _answer_blocks(self, answer_text: str) -> list[str]:
        text = (answer_text or "").strip()
        if not text:
            return []
        raw_blocks: list[str] = []
        paragraph_chunks = re.split(r"\n\s*\n", text)
        for paragraph in paragraph_chunks:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue
            if len(lines) == 1:
                raw_blocks.append(lines[0])
                continue
            for line in lines:
                raw_blocks.append(line)
        blocks: list[str] = []
        for index, block in enumerate(raw_blocks):
            normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", block).strip()
            # Citation enforcement is for factual answer content, not structural
            # headings, section labels, or short framing lead-ins.
            if self._is_non_claim_block(normalized, position=index):
                continue
            blocks.append(normalized)
        return blocks


    def _answer_guidance_block(self, query: str) -> str:
        q = (query or "").lower()
        is_summary = any(term in q for term in ("summarize", "summary", "whole meeting", "entire meeting", "overall", "resumen", "resume"))
        is_broad = is_summary or any(term in q for term in ("compare", "pros and cons", "advantages", "disadvantages", "explain", "describe", "list", "principales", "ventajas", "desventajas"))
        if is_summary:
            max_words = self._answer_summary_max_words
            shape = "Use 4-7 concise bullets or short paragraphs only when the task asks for a broad summary."
            citation_count = "Use the strongest 4-10 markers, not every available marker."
        elif is_broad:
            max_words = self._answer_default_max_words
            shape = "Use 2-5 concise bullets or short paragraphs."
            citation_count = "Use the strongest 2-6 markers."
        else:
            max_words = self._answer_specific_max_words
            shape = "Use 1-3 direct sentences unless the user explicitly asks for a list."
            citation_count = "Use the strongest 1-3 markers."
        return (
            "Answer rules:\n"
            "- Answer the user directly; do not restate the task.\n"
            f"- Target length: at most {max_words} words. {shape}\n"
            "- Every factual paragraph or bullet grounded in context MUST include at least one exact inline citation marker from the closed CITATION CONSTRAINT list.\n"
            f"- {citation_count} Cite only markers that directly support the claim.\n"
            "- Prefer the most specific evidence over broad meeting context.\n"
            "- Do not mention internal reasoning, tools, control passes or runtime details.\n"
            "- If the context is insufficient to answer factually, say NOT ENOUGH INFO or state the missing evidence concisely; do not invent support or force a citation. CIMA will record the inspected context separately.\n"
            "- Do not add an uncited final conclusion or policy slogan; every substantive sentence must be inside a cited bullet/paragraph.\n"
            "- Self-verify before submitting: every citation token in your answer must appear verbatim in the exact CITATION CONSTRAINT list; otherwise remove the unsupported claim rather than substituting another marker."
        )

    async def _memory_pass(self, *, rt: Any, query: str, answer_text: str, context_view: Any) -> MemoryProposal:
        deterministic_markers = self._extract_markers(answer_text)
        if not self._llm_memory_pass:
            return MemoryProposal(cited_markers=deterministic_markers, conclusions=[])
        prompt = {
            "task": query,
            "answer": answer_text,
            "available_markers": [item.get("marker") for item in context_view.items if item.get("marker")],
            "instruction": {
                "return": {
                    "cited_markers": ["marker", "..."],
                    "conclusions": [
                        {"kind": "FACT|HEDGED_FACT|DECISION|TODO|NOTE", "content": "text", "confidence": 0.0}
                    ],
                },
                "rules": [
                    "Return JSON only.",
                    "Use only markers that actually appear in the answer or clearly support it.",
                    "Return at most 5 conclusions.",
                    "Finish the JSON object completely within the token budget.",
                    "Do not invent unsupported facts.",
                ],
            },
        }
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are the memory/citation pass of CIMA Demonstrator. "
                    "Extract cited markers and a small set of durable conclusions from the final answer. "
                    "Return only a JSON object."
                ),
            ),
            LLMMessage(role="user", content=json.dumps(prompt, ensure_ascii=False, indent=2)),
        ]
        try:
            payload = await self._llm.complete_structured(
                messages=messages,
                temperature=0.0,
                max_tokens=768,
            )
            proposal = MemoryProposal.from_dict(payload)
        except Exception as exc:
            log.warning("Demo memory pass failed for %s — using marker-only fallback: %s", rt.conversation_id, exc)
            proposal = MemoryProposal(cited_markers=deterministic_markers, conclusions=[])
        proposal.cited_markers = list(dict.fromkeys(proposal.cited_markers + self._extract_markers(answer_text)))
        proposal.conclusions = proposal.conclusions[:5]
        return proposal

    def _prompt_lint_task_family(self, rt: Any) -> str:
        task_spec = getattr(rt, "task_spec", None)
        mode = getattr(task_spec, "mode", None)
        if mode is not None:
            return str(getattr(mode, "value", mode) or "")
        task_state = getattr(rt, "task_state", None)
        for attr in ("task_family", "mode", "execution_mode"):
            value = getattr(task_state, attr, None)
            if value is not None:
                return str(getattr(value, "value", value) or "")
        return ""

    def _prompt_lint_output_format(self, rt: Any) -> str:
        contract = getattr(rt, "output_contract", None)
        value = getattr(contract, "format", None) if contract is not None else None
        return str(value or "")

    def _output_contract_payload(self, rt: Any) -> dict[str, Any]:
        contract = getattr(rt, "output_contract", None)
        if contract is None:
            return {}
        payload = asdict(contract) if hasattr(contract, "__dataclass_fields__") else {}
        if not payload:
            for key in ("format", "representation", "base_unit", "display_scale", "rounding_rule", "precision"):
                value = getattr(contract, key, None)
                if value is not None:
                    payload[key] = value
        return payload

    def _extract_markers(self, text: str) -> list[str]:
        markers: list[str] = []
        for match in _MARKER_GROUP_RE.finditer(text or ""):
            for part in match.group(1).split(","):
                marker = part.strip()
                if marker:
                    markers.append(marker)
        return list(dict.fromkeys(markers))
