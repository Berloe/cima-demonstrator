from __future__ import annotations

"""Witness-first lineage resolution helpers.

These helpers let answer-lineage and evidence-book code resolve source/span
coverage directly from the witness plane whenever local/global evidence and
summary-origin rows already exist. Legacy demo lineage tables remain as a
fallback so the demonstrator can advance incrementally without forcing an
all-at-once migration.
"""

from dataclasses import dataclass, field
import inspect
from typing import Any, Iterable

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


@dataclass(slots=True)
class ResolvedLineageSupport:
    citem_ids: list[str] = field(default_factory=list)
    summary_ids: list[str] = field(default_factory=list)
    resolved_source_ids: list[str] = field(default_factory=list)
    resolved_span_ids: list[str] = field(default_factory=list)
    unresolved_ref_ids: list[str] = field(default_factory=list)
    unresolved_citem_ids: list[str] = field(default_factory=list)
    citem_witnesses: list[dict[str, Any]] = field(default_factory=list)
    resolution_mode: str = "empty"

    @property
    def resolved_source_count(self) -> int:
        return len(self.resolved_source_ids)

    @property
    def resolved_span_count(self) -> int:
        return len(self.resolved_span_ids)


async def _maybe_call(obj: object, name: str, *args: object, default: Any = None, **kwargs: object) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class WitnessLineageResolver:
    def __init__(self, db: object) -> None:
        self._db = db

    async def resolve_selected_items(
        self,
        *,
        conversation_id: str,
        selected_items: Iterable[dict[str, Any]],
    ) -> ResolvedLineageSupport:
        citem_refs: list[str] = []
        summary_refs: list[str] = []
        for item in selected_items:
            ref_id = str(item.get("ref_id") or "")
            if not ref_id:
                continue
            ref_kind = str(item.get("ref_kind") or "citem")
            if ref_kind in {"summary", "local_summary", "global_summary"}:
                summary_refs.append(ref_id)
            else:
                citem_refs.append(ref_id)
        return await self.resolve_refs(
            conversation_id=conversation_id,
            citem_refs=citem_refs,
            summary_refs=summary_refs,
        )

    async def resolve_selected_items_detailed(
        self,
        *,
        conversation_id: str,
        selected_items: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        detailed: list[dict[str, Any]] = []
        for item in selected_items:
            materialized = dict(item)
            support = await self.resolve_selected_items(
                conversation_id=conversation_id,
                selected_items=[materialized],
            )
            explicit_mode = ""
            scope = ""
            ref_kind = str(materialized.get("ref_kind") or "citem")
            if ref_kind in {"summary", "local_summary", "global_summary"}:
                explicit_mode = str(materialized.get("summary_resolution_mode") or "")
                scope = str(materialized.get("summary_scope") or materialized.get("summary_resolution_scope") or "")
            else:
                explicit_mode = str(materialized.get("item_resolution_mode") or materialized.get("citem_resolution_mode") or "")
                scope = str(materialized.get("item_resolution_scope") or materialized.get("citem_resolution_scope") or "")
            resolution_mode = explicit_mode or support.resolution_mode
            support_resolution_mode = support.resolution_mode
            if not scope and support.resolution_mode == "legacy_fallback":
                scope = "legacy"
            if not scope and explicit_mode == "legacy_fallback":
                scope = "legacy"
            detailed.append({
                "marker": str(materialized.get("marker") or ""),
                "ref_kind": ref_kind,
                "ref_id": str(materialized.get("ref_id") or ""),
                "resolution_mode": resolution_mode,
                "support_resolution_mode": support_resolution_mode,
                "resolution_scope": scope,
                "resolved_source_ids": list(support.resolved_source_ids),
                "resolved_span_ids": list(support.resolved_span_ids),
                "resolved_source_count": support.resolved_source_count,
                "resolved_span_count": support.resolved_span_count,
                "unresolved_ref_ids": list(support.unresolved_ref_ids),
                "citem_ids": list(support.citem_ids),
                "summary_ids": list(support.summary_ids),
                "unresolved_citem_ids": list(support.unresolved_citem_ids),
                "citem_witnesses": [dict(row) for row in support.citem_witnesses],
            })
        return detailed

    async def resolve_refs(
        self,
        *,
        conversation_id: str,
        citem_refs: Iterable[str],
        summary_refs: Iterable[str],
    ) -> ResolvedLineageSupport:
        citem_ids = list(dict.fromkeys(str(v) for v in citem_refs if str(v)))
        summary_ids = list(dict.fromkeys(str(v) for v in summary_refs if str(v)))
        source_ids: set[str] = set()
        span_ids: set[str] = set()
        unresolved_ref_ids: set[str] = set()
        unresolved_citem_ids: set[str] = set()
        citem_witnesses_by_id: dict[str, dict[str, Any]] = {}
        witness_hits = 0
        legacy_hits = 0

        local_summary_rows = {
            str(row["local_summary_id"]): row
            for row in await _maybe_call(self._db, "list_local_summary_records", conversation_id, summary_ids=summary_ids, default=[])
        }
        global_summary_rows = {
            str(row["global_summary_id"]): row
            for row in await _maybe_call(self._db, "list_global_summary_records", summary_ids=summary_ids, default=[])
        }

        resolved_citem_ids: set[str] = set(citem_ids)
        unresolved_summary_ids: list[str] = []
        for summary_id in summary_ids:
            if summary_id in local_summary_rows:
                witness_hits += 1
                origins = await _maybe_call(self._db, "list_local_summary_origins", summary_id, default=[])
                for row in origins:
                    origin_id = str(row.get("origin_id") or "")
                    if origin_id:
                        resolved_citem_ids.add(origin_id)
                continue
            if summary_id in global_summary_rows:
                witness_hits += 1
                origins = await _maybe_call(self._db, "list_global_summary_origins", summary_id, default=[])
                for row in origins:
                    origin_id = str(row.get("origin_id") or "")
                    if origin_id:
                        resolved_citem_ids.add(origin_id)
                continue
            unresolved_summary_ids.append(summary_id)

        if unresolved_summary_ids:
            resolutions = await _maybe_call(
                self._db,
                "load_demo_summary_resolutions",
                conversation_id,
                unresolved_summary_ids,
                default=[],
            )
            res_by_id = {str(row.get("summary_id")): row for row in resolutions}
            for summary_id in unresolved_summary_ids:
                row = res_by_id.get(summary_id)
                if row is not None:
                    # Legacy summary resolutions may describe either an L1 -> L0
                    # direct edge (origin_citem_ids populated) or, for newer CIMA
                    # hierarchy, an L2+ summary whose direct children are summaries
                    # represented in demo_lineage_edges. Do not treat an empty
                    # origin_citem_ids list as an empty terminal lineage; fall
                    # through to adjacent-level edges instead.
                    origin_ids = [str(v) for v in (row.get("origin_citem_ids", []) or []) if v]
                    if origin_ids:
                        legacy_hits += 1
                        for citem_id in origin_ids:
                            resolved_citem_ids.add(citem_id)
                        continue
                legacy_edges = await _maybe_call(
                    self._db,
                    "load_demo_lineage_edges",
                    conversation_id,
                    src_kind="summary",
                    src_ids=[summary_id],
                    default=[],
                )
                if legacy_edges:
                    legacy_hits += 1
                    nested_summary_ids: list[str] = []
                    for edge in legacy_edges:
                        dst_kind = str(edge.get("dst_kind") or "")
                        dst_id = str(edge.get("dst_id") or "")
                        if not dst_id:
                            continue
                        if dst_kind == "citem":
                            resolved_citem_ids.add(dst_id)
                        elif dst_kind in {"summary", "local_summary", "global_summary"}:
                            if dst_id != summary_id:
                                nested_summary_ids.append(dst_id)
                    if nested_summary_ids:
                        nested = await self.resolve_refs(
                            conversation_id=conversation_id,
                            citem_refs=[],
                            summary_refs=nested_summary_ids,
                        )
                        resolved_citem_ids.update(nested.citem_ids)
                        source_ids.update(nested.resolved_source_ids)
                        span_ids.update(nested.resolved_span_ids)
                        unresolved_ref_ids.update(nested.unresolved_ref_ids)
                        unresolved_citem_ids.update(nested.unresolved_citem_ids)
                        for witness in nested.citem_witnesses:
                            citem_id = str(witness.get("citem_id") or "")
                            if citem_id:
                                citem_witnesses_by_id[citem_id] = dict(witness)
                        if nested.resolution_mode in {"witness_first", "mixed"}:
                            witness_hits += 1
                        if nested.resolution_mode in {"legacy_fallback", "mixed"}:
                            legacy_hits += 1
                else:
                    unresolved_ref_ids.add(summary_id)

        local_citem_rows = {
            str(row["local_citem_id"]): row
            for row in await _maybe_call(self._db, "list_local_citem_records", conversation_id, citem_ids=list(resolved_citem_ids), default=[])
        }
        global_citem_rows = {
            str(row["global_citem_id"]): row
            for row in await _maybe_call(self._db, "list_global_citem_records", global_citem_ids=list(resolved_citem_ids), default=[])
        }

        for citem_id in list(resolved_citem_ids):
            citem_source_ids: set[str] = set()
            citem_span_ids: set[str] = set()
            if citem_id in local_citem_rows:
                witness_hits += 1
                rows = await _maybe_call(self._db, "list_local_citem_evidence", citem_id, default=[])
                self._collect_source_refs(rows, source_ids=source_ids, span_ids=span_ids)
                self._collect_source_refs(rows, source_ids=citem_source_ids, span_ids=citem_span_ids)
                if not citem_source_ids or not citem_span_ids:
                    unresolved_citem_ids.add(citem_id)
                citem_witnesses_by_id[citem_id] = {
                    "citem_id": citem_id,
                    "source_ids": sorted(citem_source_ids),
                    "span_ids": sorted(citem_span_ids),
                    "mode": "witness_first",
                }
                continue
            if citem_id in global_citem_rows:
                witness_hits += 1
                rows = await _maybe_call(self._db, "list_global_citem_evidence", citem_id, default=[])
                self._collect_source_refs(rows, source_ids=source_ids, span_ids=span_ids)
                self._collect_source_refs(rows, source_ids=citem_source_ids, span_ids=citem_span_ids)
                if not citem_source_ids or not citem_span_ids:
                    unresolved_citem_ids.add(citem_id)
                citem_witnesses_by_id[citem_id] = {
                    "citem_id": citem_id,
                    "source_ids": sorted(citem_source_ids),
                    "span_ids": sorted(citem_span_ids),
                    "mode": "witness_first",
                }
                continue
            legacy_edges = await _maybe_call(
                self._db,
                "load_demo_lineage_edges",
                conversation_id,
                src_kind="citem",
                src_ids=[citem_id],
                default=[],
            )
            if legacy_edges:
                legacy_hits += 1
                for edge in legacy_edges:
                    dst_kind = str(edge.get("dst_kind") or "")
                    dst_id = str(edge.get("dst_id") or "")
                    if not dst_id:
                        continue
                    if dst_kind == "source":
                        source_ids.add(dst_id)
                        citem_source_ids.add(dst_id)
                    elif dst_kind == "source_span":
                        span_ids.add(dst_id)
                        citem_span_ids.add(dst_id)
                if not citem_source_ids or not citem_span_ids:
                    unresolved_citem_ids.add(citem_id)
                citem_witnesses_by_id[citem_id] = {
                    "citem_id": citem_id,
                    "source_ids": sorted(citem_source_ids),
                    "span_ids": sorted(citem_span_ids),
                    "mode": "legacy_fallback",
                }
            else:
                unresolved_ref_ids.add(citem_id)
                unresolved_citem_ids.add(citem_id)

        if witness_hits and legacy_hits:
            resolution_mode = "mixed"
        elif witness_hits:
            resolution_mode = "witness_first"
        elif legacy_hits:
            resolution_mode = "legacy_fallback"
        else:
            resolution_mode = "empty"

        return ResolvedLineageSupport(
            citem_ids=sorted(resolved_citem_ids),
            summary_ids=summary_ids,
            resolved_source_ids=sorted(source_ids),
            resolved_span_ids=sorted(span_ids),
            unresolved_ref_ids=sorted(unresolved_ref_ids),
            unresolved_citem_ids=sorted(unresolved_citem_ids),
            citem_witnesses=[citem_witnesses_by_id[cid] for cid in sorted(citem_witnesses_by_id)],
            resolution_mode=resolution_mode,
        )

    def _collect_source_refs(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        source_ids: set[str],
        span_ids: set[str],
    ) -> None:
        for row in rows:
            locator = dict(row.get("locator_json") or {})
            source_id = row.get("source_id") or locator.get("source_id")
            if source_id:
                source_ids.add(str(source_id))
            span_id = row.get("source_span_id") or locator.get("source_span_id")
            if span_id:
                span_ids.add(str(span_id))
