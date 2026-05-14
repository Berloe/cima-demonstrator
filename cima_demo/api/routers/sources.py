from __future__ import annotations

import logging
import re
import uuid

from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from cima_demo.demo.contracts import DemoSourceSpan, SummaryResolution

from cima_demo.api.auth import verify_api_key
from cima_demo.api.conversation_guard import ensure_active_conversation
from cima_demo.api.dependencies import get_db, get_memory_service, get_source_registration_service
from cima_demo.witness_backend.events import TraceContext

router = APIRouter(prefix="/cima/v1/sources", tags=["sources"])
log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _StandaloneSegment:
    text: str
    char_start: int
    char_end: int
    segment_index: int


def _collapse_segment_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _bounded_text_windows(text: str, *, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split a source block into monotonic bounded windows preserving offsets."""
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        while cursor < end and text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break
        hard_end = min(end, cursor + max_chars)
        chunk_end = hard_end
        if hard_end < end:
            # Prefer a word boundary inside the last 25% of the window.
            boundary_floor = cursor + max(1, int(max_chars * 0.75))
            boundary = max(text.rfind(" ", boundary_floor, hard_end), text.rfind("\n", boundary_floor, hard_end))
            if boundary > cursor:
                chunk_end = boundary
        while chunk_end > cursor and text[chunk_end - 1].isspace():
            chunk_end -= 1
        if chunk_end <= cursor:
            chunk_end = hard_end
        windows.append((cursor, chunk_end))
        cursor = max(chunk_end, hard_end if chunk_end == cursor else chunk_end)
    return windows


def _standalone_inline_segment_specs(text: str, *, max_chars: int = 1200, max_segments: int = 64) -> list[_StandaloneSegment]:
    """Split registered process text into bounded standalone memory items.

    Unlike the legacy helper, this returns source offsets.  Each generated
    C-item can therefore point to a granular span while still retaining the
    parent document source_id.
    """
    if not text or not text.strip():
        return []
    stripped_start = len(text) - len(text.lstrip())
    stripped_end = len(text.rstrip())
    block_matches = list(re.finditer(r"\S(?:[\s\S]*?\S)?(?=\n\s*\n+|\Z)", text[stripped_start:stripped_end]))
    windows: list[tuple[int, int]] = []
    if block_matches:
        for match in block_matches:
            block_start = stripped_start + match.start()
            block_end = stripped_start + match.end()
            if block_end - block_start > max_chars:
                windows.extend(_bounded_text_windows(text, start=block_start, end=block_end, max_chars=max_chars))
            else:
                windows.append((block_start, block_end))
            if len(windows) >= max_segments:
                break
    else:
        windows.extend(_bounded_text_windows(text, start=stripped_start, end=stripped_end, max_chars=max_chars))

    segments: list[_StandaloneSegment] = []
    for idx, (start, end) in enumerate(windows[:max_segments]):
        collapsed = _collapse_segment_text(text[start:end])
        if collapsed:
            segments.append(_StandaloneSegment(text=collapsed, char_start=start, char_end=end, segment_index=idx))
    return segments




def _summary_anchor_segments(segments: list[_StandaloneSegment], *, max_anchors: int = 5) -> list[_StandaloneSegment]:
    """Return the exact segment set that is allowed to support the L1 summary.

    CIMA lineage is direct and input-effective: an L1 summary may link only to
    the L0 C-items actually used to build the abstraction.  The full ContextView
    or full source may be captured as audit metadata, but it is not lineage.
    """
    return list(segments[:max_anchors])


def _summarize_inline_segments_for_l1(*, origin_ref: str | None, segments: list[_StandaloneSegment]) -> str:
    """Create a deterministic L1 extractive abstraction for publication evidence.

    This text is deliberately built only from the returned anchor segments.  It
    avoids claims about the whole source (for example total segment count), so
    the persisted lineage can be exactly the set of C-items whose segments were
    quoted into the summary.
    """
    label = f"source {origin_ref}" if origin_ref else "registered source"
    anchors: list[str] = []
    for segment in _summary_anchor_segments(segments):
        text = segment.text.strip()
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        anchors.append(f"segment {segment.segment_index}: {text}")
    if not anchors:
        return f"L1 extractive summary for {label}: no processable anchor segments were selected."
    return f"L1 extractive summary for {label}: " + " | ".join(anchors)


async def _create_standalone_l1_summary(
    *,
    db,
    conversation_id: str,
    source_id: str,
    source_span_id: str | None,
    external_message_id: str | None,
    segments: list[_StandaloneSegment],
) -> str | None:
    """Persist a minimal L1 summary with origin links to local C-items.

    The witness resolver already knows how to expand local summaries through
    local_summary_origin rows.  This function therefore gives the demonstrator a
    real multi-scale object: summary -> C-items -> source spans -> source.
    """
    required = ("list_local_citem_records", "save_local_summary_record", "save_local_summary_origin")
    if not all(hasattr(db, name) for name in required):
        return None
    rows = await db.list_local_citem_records(conversation_id)
    source_rows: list[dict[str, object]] = []
    for row in rows:
        prov = dict(row.get("provenance_json") or {})
        locator = dict(prov.get("locator_json") or {})
        if str(prov.get("source_id") or locator.get("source_id") or "") == source_id:
            source_rows.append(dict(row))

    if not source_rows and hasattr(db, "load_demo_lineage_edges"):
        # Full mode: C-items are in Qdrant, not in cima.local_citem.
        # Recover citem_id → source mapping from demo_lineage_edges.
        edges = await db.load_demo_lineage_edges(
            conversation_id,
            src_kind="citem",
            dst_kind="source",
            dst_ids=[source_id],
        )
        for edge in edges:
            meta = dict(edge.get("metadata") or {})
            source_rows.append({
                "local_citem_id": edge["src_id"],
                "provenance_json": {
                    "source_id": source_id,
                    "locator_json": meta,
                },
            })

    if not source_rows:
        return None

    def _ordinal(row: dict[str, object]) -> int:
        prov = dict(row.get("provenance_json") or {})
        locator = dict(prov.get("locator_json") or {})
        try:
            return int(locator.get("segment_index", 0))
        except Exception:
            return 0

    source_rows.sort(key=lambda row: (_ordinal(row), str(row.get("local_citem_id") or "")))

    # Lineage is established only to effective summary inputs.  The deterministic
    # L1 summary text is composed from the selected anchor segments; therefore
    # its direct children are exactly the C-items whose segment_index appears in
    # that anchor set.  The rest of the source is retained only as audit metadata.
    anchor_segments = _summary_anchor_segments(segments)
    anchor_indexes = {int(segment.segment_index) for segment in anchor_segments}
    used_rows = [row for row in source_rows if _ordinal(row) in anchor_indexes]
    if not used_rows:
        # Fallback for legacy rows without segment_index metadata: use the first
        # N source rows, matching the summary anchor count.
        used_rows = source_rows[: max(1, len(anchor_segments))]
    origin_ids = [str(row.get("local_citem_id")) for row in used_rows if str(row.get("local_citem_id") or "")]
    if not origin_ids:
        return None
    summary_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{conversation_id}:{source_id}:l1:source"))
    summary_text = _summarize_inline_segments_for_l1(origin_ref=external_message_id, segments=segments)
    now = datetime.now(UTC).isoformat()
    covers = {
        "kind": "source_l1_extract",
        "source_id": source_id,
        "source_span_id": source_span_id,
        "origin_ref": external_message_id,
        "origin_citem_ids": origin_ids,
        "summary_input_set": origin_ids,
        "summary_used_refs": origin_ids,
        "summary_available_refs_count": len(source_rows),
        "summary_anchor_segment_indexes": sorted(anchor_indexes),
        "available_segment_count": len(segments),
        "lineage_policy": "direct_effective_inputs_only",
        "summary_method": "deterministic_extractive_l1",
    }
    await db.save_local_summary_record({
        "local_summary_id": summary_id,
        "conversation_id": conversation_id,
        "level": "EPOCH",
        "cluster_id": f"source:{source_id}",
        "epoch_no": 0,
        "text": summary_text,
        "covers_json": covers,
        "created_at": now,
        "updated_at": now,
        "vector_state": "NONE",
        "is_pinned": True,
        "was_cited": False,
    })
    for ordinal, origin_id in enumerate(origin_ids):
        await db.save_local_summary_origin({
            "local_summary_id": summary_id,
            "conversation_id": conversation_id,
            "origin_kind": "local_citem",
            "origin_id": origin_id,
            "ordinal": ordinal,
        })
    if hasattr(db, "save_demo_summary_resolution"):
        await db.save_demo_summary_resolution(SummaryResolution(
            summary_id=summary_id,
            conversation_id=conversation_id,
            summary_text=summary_text,
            origin_citem_ids=origin_ids,
            metadata=covers,
        ).to_dict())
    if hasattr(db, "save_demo_lineage_edge"):
        for ordinal, origin_id in enumerate(origin_ids):
            await db.save_demo_lineage_edge({
                "edge_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "src_kind": "summary",
                "src_id": summary_id,
                "dst_kind": "citem",
                "dst_id": origin_id,
                "relation": "SUMMARIZES",
                "metadata": {"ordinal": ordinal, "source_id": source_id, "summary_method": "deterministic_extractive_l1"},
            })
    return summary_id


def _standalone_inline_segments(text: str, *, max_chars: int = 1200, max_segments: int = 64) -> list[str]:
    """Backward-compatible text-only view used by older tests."""
    return [segment.text for segment in _standalone_inline_segment_specs(text, max_chars=max_chars, max_segments=max_segments)]


async def _inline_ingest_standalone_memory(
    *,
    runtime_mode: str,
    memory_service,
    db,
    conversation_id: str,
    text: str,
    source_kind: str,
    source_id: str,
    source_span_id: str | None,
    external_message_id: str | None,
    processable: bool,
) -> int:
    if runtime_mode not in ("standalone", "full") or not processable:
        return 0
    segments = _standalone_inline_segment_specs(text)
    if not segments:
        return 0
    prefix = ""
    if source_kind == "file_text" and external_message_id:
        prefix = f"DOCUMENT {external_message_id}: "

    conclusions: list[dict[str, object]] = []
    for segment in segments:
        granular_span_id = str(uuid.uuid4())
        locator = {
            "kind": "standalone_inline_segment",
            "source_kind": source_kind,
            "origin_ref": external_message_id,
            "segment_index": segment.segment_index,
            "char_start": segment.char_start,
            "char_end": segment.char_end,
            "parent_span_id": source_span_id,
        }
        if hasattr(db, "save_demo_source_span"):
            await db.save_demo_source_span(DemoSourceSpan(
                span_id=granular_span_id,
                source_id=source_id,
                conversation_id=conversation_id,
                span_kind="inline_segment",
                char_start=segment.char_start,
                char_end=segment.char_end,
                locator=locator,
                preview_text=text[segment.char_start:segment.char_end][:500],
            ).to_dict())
        else:  # pragma: no cover - defensive fallback for non-demo stores
            granular_span_id = source_span_id or granular_span_id
        conclusions.append({
            "type": "CONTEXT",
            "content": prefix + segment.text,
            "confidence": 1.0,
            # Keep document-level origin and granular span simultaneously.
            "source_id": source_id,
            "source_span_ids": [granular_span_id] if granular_span_id else ([source_span_id] if source_span_id else []),
            "lineage_meta": locator,
            "locator_json": {"source_id": source_id, "source_span_id": granular_span_id, **locator},
        })
    try:
        await memory_service.ingest_batch(
            conclusions,
            phase="INGEST",
            conversation_id=conversation_id,
            turn_id=source_id,
        )
        summary_id = await _create_standalone_l1_summary(
            db=db,
            conversation_id=conversation_id,
            source_id=source_id,
            source_span_id=source_span_id,
            external_message_id=external_message_id,
            segments=segments,
        )
        if summary_id:
            log.info("Standalone L1 summary created source_id=%s summary_id=%s segments=%d", source_id, summary_id, len(segments))
    except Exception as exc:  # pragma: no cover - compatibility guard
        log.warning("Standalone inline ingest failed source_id=%s error=%s", source_id, type(exc).__name__)
        return 0
    return len(conclusions)


class RegisterTextRequest(BaseModel):
    conversation_id: str
    text: str = Field(min_length=1)
    source_kind: str = Field(default="chat_user")
    role: str | None = None
    external_provider: str | None = None
    external_conversation_id: str | None = None
    external_message_id: str | None = None
    displayable: bool = True
    processable: bool = True
    request_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    actor_kind: str | None = None


_SOURCE_KIND_ALIASES = {
    "file": "file_text",
    "dataset_document": "file_text",
    "document": "file_text",
    "doc": "file_text",
    "text": "chat_user",
    "chat": "chat_user",
    "assistant": "chat_assistant",
    "user": "chat_user",
}


def _canonical_source_kind(source_kind: str) -> str:
    return _SOURCE_KIND_ALIASES.get(source_kind, source_kind)


@router.post("/register_text", status_code=status.HTTP_202_ACCEPTED)
async def register_text(
    body: RegisterTextRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
    service=Depends(get_source_registration_service),
    memory_service=Depends(get_memory_service),
):
    ensure_active_conversation(await db.get_conversation(body.conversation_id))
    trace = None
    if body.request_id or body.correlation_id or body.causation_id or body.actor_kind:
        trace = TraceContext(
            request_id=body.request_id or body.correlation_id or body.conversation_id,
            correlation_id=body.correlation_id or body.request_id or body.conversation_id,
            causation_id=body.causation_id,
            actor_kind=body.actor_kind,
        )
    canonical_kind = _canonical_source_kind(body.source_kind)
    result = await service.register_text(
        conversation_id=body.conversation_id,
        text=body.text,
        role=body.role,
        source_kind=canonical_kind,
        external_provider=body.external_provider,
        external_conversation_id=body.external_conversation_id,
        external_message_id=body.external_message_id,
        displayable=body.displayable,
        processable=body.processable,
        trace=trace,
    )
    inline_ingested_count = await _inline_ingest_standalone_memory(
        runtime_mode=str(getattr(request.app.state, "runtime_mode", "")),
        memory_service=memory_service,
        db=db,
        conversation_id=body.conversation_id,
        text=body.text,
        source_kind=canonical_kind,
        source_id=result.source_id,
        source_span_id=result.span_id,
        external_message_id=body.external_message_id,
        processable=body.processable,
    )
    status_value = "indexed" if inline_ingested_count else "queued"
    return {
        "accepted": True,
        "conversation_id": body.conversation_id,
        "source_id": result.source_id,
        "source_span_id": result.span_id,
        "outbox_id": result.outbox_id,
        "status": status_value,
        "inline_ingested_count": inline_ingested_count,
    }
