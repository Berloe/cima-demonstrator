"""IngestionCore — embed + upsert C-Items with conflict detection (SPEC-5 split)."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from cima_demo.domain.entities import CItem, IngestRequest
from cima_demo.domain.errors import ConflictDetectionError
from cima_demo.domain.operations import compute_static_importance
from cima_demo.domain.ports import CItemStorePort, LLMPort
from cima_demo.domain.value_objects import ItemType
from cima_demo.memory.ingestion.conflict import CONFLICTABLE, ConflictDetector

log = logging.getLogger(__name__)


class IngestionCore:
    """Core C-Item ingestion: embed + upsert + conflict detection."""

    def __init__(
        self,
        citem_store: CItemStorePort,
        llm_port: LLMPort,
        conflict_detector: ConflictDetector,
        ingest_citem_fn: Any | None = None,
        lineage_recorder: Any | None = None,
    ) -> None:
        self._cstore = citem_store
        self._llm = llm_port
        self._conflict = conflict_detector
        self._ingest_citem_fn = ingest_citem_fn if ingest_citem_fn is not None else self.ingest_citem
        self._lineage = lineage_recorder

    async def ingest_citem(
        self,
        request: IngestRequest,
        skip_conflict_detection: bool = False,
    ) -> CItem | None:
        """Embed + upsert to Qdrant (APP-INV-08 — synchronous, no pending_qdrant_sync).

        Returns the persisted CItem, or None if the content was a duplicate.
        """
        if request.importance_override is not None:
            importance = max(0.0, min(1.0, request.importance_override))
        else:
            importance = compute_static_importance(
                item_type=request.item_type,
                confidence=request.confidence,
                validation_label=request.validation_label,
            )

        citem = CItem(
            conversation_id=request.conversation_id,
            content=request.content,
            item_type=request.item_type,
            scope=request.scope,
            scope_status="active",
            importance=importance,
            confidence=request.confidence,
            validation_label=request.validation_label,
            conflict_status="none",
            phase_ingested=request.phase_ingested,
            actor=request.actor,
            motivation=request.motivation,
            dependency_ids=list(request.dependency_ids),
            chunk_kind=request.chunk_kind,
        )

        citem.content_hash = hashlib.sha256(citem.content.encode()).hexdigest()

        async def _count_tokens_safe() -> int:
            try:
                return await self._llm.count_tokens(citem.content)
            except Exception:
                return max(1, len(citem.content) // 4)

        is_dup, token_count = await asyncio.gather(
            self._cstore.exists_by_hash(citem.content_hash, request.conversation_id),
            _count_tokens_safe(),
        )
        if is_dup:
            log.debug(
                "ingest_citem: duplicate skipped (hash=%s) for %s",
                citem.content_hash[:8], request.conversation_id,
            )
            return None
        citem.token_count = token_count

        await self._cstore.save(citem)

        if self._lineage is not None and (
            request.source_id is not None or request.source_span_ids or request.dependency_ids
        ):
            try:
                await self._lineage.record_citem_lineage(
                    conversation_id=request.conversation_id,
                    citem_id=citem.citem_id,
                    source_id=request.source_id,
                    source_span_ids=list(request.source_span_ids),
                    dependency_ids=list(request.dependency_ids),
                    metadata=dict(request.lineage_meta),
                )
            except Exception:
                log.exception("demo lineage record failed for citem=%s", citem.citem_id)

        if not skip_conflict_detection and request.item_type in CONFLICTABLE:
            try:
                await self._conflict.detect_conflict(citem, request)
            except ConflictDetectionError as e:
                log.warning("Conflict detection failed (non-fatal): %s", e)

        return citem

    async def ingest_batch(
        self,
        conclusions: list[dict[str, Any]],
        phase: str,
        conversation_id: str,
        turn_id: str,
    ) -> None:
        """Ingest multiple conclusions from LLM <conclusions> section."""
        valid: list[tuple[dict[str, Any], str]] = []
        for c in conclusions:
            content = c.get("content", "")
            if content:
                valid.append((c, hashlib.sha256(content.encode()).hexdigest()))

        if not valid:
            return

        dup_flags: list[bool] = list(await asyncio.gather(*[
            self._cstore.exists_by_hash(h, conversation_id)
            for _, h in valid
        ]))

        prev_id: str | None = None
        for (conclusion, _content_hash), already_exists in zip(valid, dup_flags):
            raw_source_span_ids = conclusion.get("source_span_ids") or conclusion.get("span_ids") or []
            if isinstance(raw_source_span_ids, str):
                source_span_ids = [raw_source_span_ids]
            else:
                source_span_ids = [str(v) for v in raw_source_span_ids if str(v)]
            explicit_deps = [str(v) for v in conclusion.get("dependency_ids", []) if str(v)]
            request = IngestRequest(
                content=conclusion["content"],
                item_type=conclusion.get("type", ItemType.OBSERVATION),
                phase_ingested=phase,
                actor=str(conclusion.get("actor", "agent")),
                conversation_id=conversation_id,
                motivation=conclusion.get("motivation", "LLM conclusion"),
                confidence=float(conclusion.get("confidence", 0.8)),
                dependency_ids=explicit_deps or ([prev_id] if prev_id else []),
                source_id=str(conclusion.get("source_id") or "") or None,
                source_span_ids=source_span_ids,
                lineage_meta=dict(conclusion.get("lineage_meta") or {}),
            )
            if already_exists:
                continue
            citem = await self._ingest_citem_fn(request)
            if citem is not None:
                prev_id = citem.citem_id
