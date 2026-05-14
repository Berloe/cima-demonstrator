"""Lineage and provenance services for the CIMA Demonstrator."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict
from typing import Any, Iterable

from cima_demo.demo.contracts import (
    AnswerLineage,
    DemoLineageEdge,
    DemoSourceRecord,
    DemoSourceSpan,
    SummaryResolution,
)
from cima_demo.demo.lineage.witness_resolver import WitnessLineageResolver
from cima_demo.domain.ports import RelDBPort


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


class DemoLineageService:
    """Persists source/span/lineage artifacts without polluting the visible transcript."""

    def __init__(self, rel_db: RelDBPort) -> None:
        self._db = rel_db

    async def register_text_source(
        self,
        *,
        conversation_id: str,
        source_kind: str,
        role: str | None,
        display_text: str | None,
        process_text: str | None,
        origin_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[DemoSourceRecord, DemoSourceSpan | None]:
        text = process_text if process_text is not None else display_text
        source = DemoSourceRecord(
            source_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            source_kind=source_kind,
            role=role,
            origin_ref=origin_ref,
            display_text=display_text,
            process_text=process_text,
            metadata={**(metadata or {}), "content_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest()},
        )
        await self._db.save_demo_source(source.to_dict())
        if not text:
            return source, None
        span = DemoSourceSpan(
            span_id=str(uuid.uuid4()),
            source_id=source.source_id,
            conversation_id=conversation_id,
            span_kind="full_text",
            char_start=0,
            char_end=len(text),
            locator={"kind": source_kind, "origin_ref": origin_ref},
            preview_text=text[:500],
        )
        await self._db.save_demo_source_span(span.to_dict())
        return source, span

    async def register_spans_from_chunks(
        self,
        *,
        conversation_id: str,
        source_id: str,
        process_text: str,
        chunks: Iterable[Any],
    ) -> dict[int, DemoSourceSpan]:
        """Create rehydratable spans for chunks.

        Uses monotonic substring search. When a chunk cannot be found exactly,
        falls back to a preview-only span anchored at the current cursor so the
        demonstrator never loses provenance metadata.
        """
        cursor = 0
        results: dict[int, DemoSourceSpan] = {}
        for chunk in chunks:
            text = getattr(chunk, "text", "") or ""
            idx = process_text.find(text, cursor)
            if idx < 0:
                idx = process_text.find(text)
            if idx < 0:
                idx = cursor
                end = min(len(process_text), cursor + len(text))
                preview = text[:500]
            else:
                end = idx + len(text)
                preview = process_text[idx:end][:500]
                cursor = end
            span = DemoSourceSpan(
                span_id=str(uuid.uuid4()),
                source_id=source_id,
                conversation_id=conversation_id,
                span_kind=getattr(chunk, "doc_type", None) or "chunk",
                char_start=idx,
                char_end=end,
                locator={
                    "chunk_index": getattr(chunk, "index", None),
                    "page_num": getattr(chunk, "page_num", None),
                    "section_hint": getattr(chunk, "section_hint", None),
                    "filename": getattr(chunk, "filename", None),
                },
                preview_text=preview,
            )
            await self._db.save_demo_source_span(span.to_dict())
            results[getattr(chunk, "index", len(results))] = span
        return results

    async def record_citem_lineage(
        self,
        *,
        conversation_id: str,
        citem_id: str,
        source_id: str | None = None,
        source_span_ids: list[str] | None = None,
        dependency_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = metadata or {}
        if source_id is not None:
            await self._db.save_demo_lineage_edge(
                DemoLineageEdge(
                    edge_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    src_kind="citem",
                    src_id=citem_id,
                    dst_kind="source",
                    dst_id=source_id,
                    relation="DERIVED_FROM_SOURCE",
                    metadata=payload,
                ).to_dict()
            )
        for span_id in source_span_ids or []:
            await self._db.save_demo_lineage_edge(
                DemoLineageEdge(
                    edge_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    src_kind="citem",
                    src_id=citem_id,
                    dst_kind="source_span",
                    dst_id=span_id,
                    relation="DERIVED_FROM_SPAN",
                    metadata=payload,
                ).to_dict()
            )
        for dep_id in dependency_ids or []:
            await self._db.save_demo_lineage_edge(
                DemoLineageEdge(
                    edge_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    src_kind="citem",
                    src_id=citem_id,
                    dst_kind="citem",
                    dst_id=dep_id,
                    relation="DEPENDS_ON",
                    metadata=payload,
                ).to_dict()
            )

    async def record_summary_resolution(
        self,
        *,
        conversation_id: str,
        summary_id: str,
        summary_text: str,
        origin_citem_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> SummaryResolution:
        resolution = SummaryResolution(
            summary_id=summary_id,
            conversation_id=conversation_id,
            summary_text=summary_text,
            origin_citem_ids=list(origin_citem_ids),
            metadata=metadata or {},
        )
        await self._db.save_demo_summary_resolution(resolution.to_dict())
        for citem_id in origin_citem_ids:
            await self._db.save_demo_lineage_edge(
                DemoLineageEdge(
                    edge_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    src_kind="summary",
                    src_id=summary_id,
                    dst_kind="citem",
                    dst_id=citem_id,
                    relation="SUMMARIZES",
                    metadata=metadata or {},
                ).to_dict()
            )
        return resolution

    async def record_answer_lineage(
        self,
        *,
        conversation_id: str,
        run_id: str,
        response_turn_id: str | None,
        context_id: str | None,
        answer_text: str,
        cited_markers: list[str] | None,
        selected_items: list[dict[str, Any]],
    ) -> AnswerLineage:
        answer = AnswerLineage(
            answer_lineage_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            run_id=run_id,
            response_turn_id=response_turn_id,
            context_id=context_id,
            answer_text=answer_text,
            cited_markers=list(cited_markers or []),
            lineage=[dict(item) for item in selected_items],
        )
        detailed = await WitnessLineageResolver(self._db).resolve_selected_items_detailed(
            conversation_id=conversation_id,
            selected_items=selected_items,
        )
        detail_by_key = {
            (str(row.get("marker") or ""), str(row.get("ref_kind") or ""), str(row.get("ref_id") or "")): row
            for row in detailed
        }
        materialized_lineage: list[dict[str, Any]] = []
        resolved_source_ids: set[str] = set()
        resolved_span_ids: set[str] = set()
        unresolved_ref_ids: set[str] = set()
        resolution_modes: list[str] = []
        marker_resolution: list[dict[str, Any]] = []
        for item in selected_items:
            materialized = dict(item)
            key = (str(materialized.get("marker") or ""), str(materialized.get("ref_kind") or ""), str(materialized.get("ref_id") or ""))
            detail = detail_by_key.get(key)
            if detail is not None:
                ref_kind = str(materialized.get("ref_kind") or "citem")
                resolution_mode = str(detail.get("resolution_mode") or "empty")
                resolution_scope = str(detail.get("resolution_scope") or "")
                support_resolution_mode = str(detail.get("support_resolution_mode") or "")
                if ref_kind in {"summary", "local_summary", "global_summary"}:
                    materialized.setdefault("summary_resolution_mode", resolution_mode)
                    if resolution_scope:
                        materialized.setdefault("summary_scope", resolution_scope)
                else:
                    materialized.setdefault("item_resolution_mode", resolution_mode)
                    if resolution_scope:
                        materialized.setdefault("item_resolution_scope", resolution_scope)
                resolved_source_ids.update(str(v) for v in detail.get("resolved_source_ids") or [] if str(v))
                resolved_span_ids.update(str(v) for v in detail.get("resolved_span_ids") or [] if str(v))
                unresolved_ref_ids.update(str(v) for v in detail.get("unresolved_ref_ids") or [] if str(v))
                if resolution_mode:
                    resolution_modes.append(resolution_mode)
                if support_resolution_mode:
                    resolution_modes.append(support_resolution_mode)
                row = {
                    "marker": str(materialized.get("marker") or detail.get("marker") or ""),
                    "ref_kind": ref_kind,
                    "ref_id": str(materialized.get("ref_id") or detail.get("ref_id") or ""),
                    "resolution_mode": resolution_mode,
                    "support_resolution_mode": support_resolution_mode,
                    "resolution_scope": resolution_scope,
                    "resolved_source_ids": list(detail.get("resolved_source_ids") or []),
                    "resolved_span_ids": list(detail.get("resolved_span_ids") or []),
                    "resolved_source_count": int(detail.get("resolved_source_count") or 0),
                    "resolved_span_count": int(detail.get("resolved_span_count") or 0),
                    "unresolved_ref_ids": list(detail.get("unresolved_ref_ids") or []),
                    "unresolved_citem_ids": list(detail.get("unresolved_citem_ids") or []),
                    "citem_witnesses": [dict(row) for row in list(detail.get("citem_witnesses") or []) if isinstance(row, dict)],
                    "citem_ids": list(detail.get("citem_ids") or []),
                    "summary_ids": list(detail.get("summary_ids") or []),
                }
                if row["marker"]:
                    marker_resolution.append(row)
            materialized_lineage.append(materialized)
        answer.lineage = materialized_lineage
        answer.resolved_source_ids = sorted(resolved_source_ids)
        answer.resolved_span_ids = sorted(resolved_span_ids)
        answer.resolved_source_count = len(resolved_source_ids)
        answer.resolved_span_count = len(resolved_span_ids)
        answer.unresolved_ref_ids = sorted(unresolved_ref_ids)
        answer.marker_resolution = marker_resolution
        answer.resolution_mode = _merge_resolution_modes(resolution_modes)
        await self._db.save_demo_answer_lineage(answer.to_dict())
        if context_id is not None:
            await self._db.save_demo_lineage_edge(
                DemoLineageEdge(
                    edge_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    src_kind="answer",
                    src_id=answer.answer_lineage_id,
                    dst_kind="context_snapshot",
                    dst_id=context_id,
                    relation="USES_CONTEXT",
                    metadata={"markers": list(cited_markers or [])},
                ).to_dict()
            )
        for item in selected_items:
            ref_kind = str(item.get("ref_kind", "citem"))
            ref_id = str(item.get("ref_id", ""))
            if not ref_id:
                continue
            await self._db.save_demo_lineage_edge(
                DemoLineageEdge(
                    edge_id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    src_kind="answer",
                    src_id=answer.answer_lineage_id,
                    dst_kind=ref_kind,
                    dst_id=ref_id,
                    relation="USES_ITEM",
                    metadata={"marker": item.get("marker")},
                ).to_dict()
            )
        return answer
