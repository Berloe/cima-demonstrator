from __future__ import annotations

"""Async semantic plane for the witness backend.

This tranche completes the missing middle of the approved async pipeline:
chunk manifests become EDUs, EDUs become local C-items with evidence rows,
and chunk/C-item vectors are upserted into the witness Qdrant collections.
"""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cima_demo.demo.lineage import DemoLineageService
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.lifecycle_guard import complete_if_conversation_not_active
from cima_demo.witness_backend.events import (
    CItemCreatedData,
    ChunkCreatedData,
    CloudEventEnvelope,
    EduSegmentedData,
    EventType,
    Producer,
    SummaryChangedData,
    VectorMeta,
    VectorUpsertedData,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


def _one_line(text: str) -> str:
    return " ".join(text.split())[:280]


class SemanticStoreLike(Protocol):
    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...

    async def load_demo_sources(self, conversation_id: str, source_ids: list[str]) -> list[dict[str, Any]]: ...

    async def load_demo_source_spans(self, conversation_id: str, span_ids: list[str]) -> list[dict[str, Any]]: ...

    async def list_chunk_records(self, conversation_id: str, *, source_id: str | None = None) -> list[dict[str, Any]]: ...

    async def save_edu_record(self, edu_json: dict[str, Any]) -> None: ...

    async def list_edu_records(
        self,
        conversation_id: str,
        *,
        chunk_id: str | None = None,
        edu_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def save_local_citem_record(self, citem_json: dict[str, Any]) -> None: ...

    async def list_local_citem_records(self, conversation_id: str, *, citem_ids: list[str] | None = None) -> list[dict[str, Any]]: ...

    async def save_local_citem_evidence(self, evidence_json: dict[str, Any]) -> None: ...

    async def list_local_citem_evidence(self, local_citem_id: str) -> list[dict[str, Any]]: ...

    async def save_local_summary_record(self, summary_json: dict[str, Any]) -> None: ...

    async def list_local_summary_records(
        self,
        conversation_id: str,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        cluster_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def save_local_summary_origin(self, origin_json: dict[str, Any]) -> None: ...

    async def update_local_summary_vector_state(
        self,
        local_summary_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def update_chunk_vector_state(
        self,
        chunk_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def update_local_citem_vector_state(
        self,
        local_citem_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def get_file_record(self, file_id: str): ...

    async def update_file_record(
        self,
        file_id: str,
        *,
        status: str,
        chunk_count: int = 0,
        citem_ids: list[str] | None = None,
        error_message: str | None = None,
    ) -> None: ...


class EmbeddingLike(Protocol):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class TokenCounterLike(Protocol):
    def count_text_tokens_sync(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EduSegment:
    kind: str
    text: str
    start: int
    end: int
    features: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BuiltCItem:
    citem_id: str
    semantic_identity_id: str
    text: str
    embedding_text: str
    item_type: str
    edu_ids: list[str]
    spans: list[dict[str, int]]
    salience: float
    meta_json: dict[str, Any]


_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z\s\d\-_:]{3,80})$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_ROW_RE = re.compile(r"\|.*\|")
_CODE_RE = re.compile(r"```|\b(def|class|return|import|SELECT|FROM|WHERE|UPDATE|INSERT|DELETE)\b")
_SENTENCE_RE = re.compile(r".+?(?:[.!?;:](?:\s+|$)|$)", re.S)
_CONNECTORS = {
    "y", "e", "pero", "aunque", "porque", "sin", "sin embargo", "ademas", "además", "por tanto", "then", "but", "however", "therefore",
}
_DECISION_RE = re.compile(r"\b(decid|acord|must\b|shall\b|we will\b|se acuerda|debe(?:mos)?\b)", re.I)
_CONSTRAINT_RE = re.compile(r"\b(constraint|restricci[oó]n|solo\s+oss|open source only|must not|no debe|obligatorio)\b", re.I)
_DEFINITION_RE = re.compile(r"\b(es|son|means|defined as|se define|consiste en)\b", re.I)
_RISK_RE = re.compile(r"\b(risk|riesgo|blocker|bloqueo|problema|issue)\b", re.I)
_PLAN_RE = re.compile(r"\b(next step|siguiente paso|plan|implementar|hacer|build|create|add)\b", re.I)
_ATTRIBUTION_RE = re.compile(r"\b(seg[uú]n|according to|per\s+|docs?|documentaci[oó]n|source:)\b", re.I)
_HEDGED_RE = re.compile(r"\b(maybe|might|could|perhaps|probablemente|quiz[aá]s|podr[ií]a)\b", re.I)
_EVAL_RE = re.compile(r"\b(mejor|peor|correcto|incorrecto|recomend|good|bad|better|worse)\b", re.I)
_IDENTIFIER_RE = re.compile(r"(?:[A-Za-z]:\\|/[^\s]+|v?\d+\.\d+(?:\.\d+)?|[A-Z]{2,}-\d+|[A-Za-z_][\w.-]+\.[A-Za-z]{2,})")

_SALIENCE_BY_TYPE = {
    "DECISION": 0.95,
    "CONSTRAINT": 0.98,
    "DEFINITION": 0.85,
    "FACT": 0.72,
    "HEDGED_FACT": 0.55,
    "CONTEXT": 0.45,
    "QUESTION": 0.40,
    "PLAN_STEP": 0.68,
    "RISK": 0.78,
    "EVALUATION": 0.40,
    "ATTRIBUTION": 0.35,
    "CODE_ARTIFACT": 0.70,
}


def _deterministic_event_id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts))


class DeterministicEduSegmenter:
    def __init__(self, *, token_counter: TokenCounterLike, edu_segmenter_version: int = 1) -> None:
        self._counter = token_counter
        self._version = edu_segmenter_version

    @property
    def version(self) -> int:
        return self._version

    def segment(self, text: str, *, source_offset: int = 0) -> list[EduSegment]:
        raw_segments = self._split_structural(text)
        segments: list[EduSegment] = []
        cursor = 0
        for kind, part in raw_segments:
            part = part.strip()
            if not part:
                continue
            idx = text.find(part, cursor)
            if idx < 0:
                idx = text.find(part)
            if idx < 0:
                idx = cursor
            end = idx + len(part)
            cursor = end
            segments.append(
                EduSegment(
                    kind=kind,
                    text=part,
                    start=source_offset + idx,
                    end=source_offset + end,
                    features=_edu_features(part, kind=kind, token_count=self._counter.count_text_tokens_sync(part)),
                )
            )
        return segments

    def _split_structural(self, text: str) -> list[tuple[str, str]]:
        blocks = [block for block in re.split(r"\n\s*\n", text) if block and block.strip()]
        segments: list[tuple[str, str]] = []
        for block in blocks:
            stripped = block.strip()
            lines = [line.strip() for line in stripped.splitlines() if line.strip()]
            if not lines:
                continue
            if len(lines) == 1 and _HEADING_RE.match(lines[0]) and len(lines[0]) <= 100:
                segments.append(("heading", lines[0]))
                continue
            if all(_LIST_RE.match(line) for line in lines):
                for line in lines:
                    segments.append(("list_item", line))
                continue
            if _TABLE_ROW_RE.search(stripped) and len(lines) > 1:
                for line in lines:
                    if line:
                        segments.append(("table_row", line))
                continue
            if _CODE_RE.search(stripped):
                segments.append(("code_block", stripped))
                continue
            for sent in _split_sentences(stripped):
                segments.append(("prose_clause", sent))
        return segments or [("sentence", text.strip())]


class DeterministicCItemBuilder:
    def __init__(self, *, token_counter: TokenCounterLike, citem_builder_version: int = 1, max_tokens: int = 220) -> None:
        self._counter = token_counter
        self._version = citem_builder_version
        self._max_tokens = max_tokens

    @property
    def version(self) -> int:
        return self._version

    def build(
        self,
        *,
        conversation_id: str,
        source_row: dict[str, Any],
        chunk_row: dict[str, Any],
        edu_rows: list[dict[str, Any]],
    ) -> list[BuiltCItem]:
        source_text = str(source_row.get("process_text") or source_row.get("display_text") or "")
        ordered = sorted(([dict(row, _source_text=source_text) for row in edu_rows]), key=lambda row: _edu_start(row))
        built: list[BuiltCItem] = []
        pending_heading: dict[str, Any] | None = None
        current: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal current
            if not current:
                return
            built.append(self._materialize(conversation_id=conversation_id, source_row=source_row, chunk_row=chunk_row, edu_rows=current, heading_row=pending_heading if current and current[0] is not pending_heading else None))
            current = []

        for idx, edu in enumerate(ordered):
            kind = str(edu.get("edu_kind") or "prose_clause")
            if kind == "heading":
                pending_heading = edu
                continue
            if not current:
                if pending_heading is not None:
                    current.append(pending_heading)
                    pending_heading = None
                current.append(edu)
            else:
                candidate_rows = [row for row in current if row.get("edu_kind") != "heading"] + [edu]
                candidate_text = " ".join(_edu_text(row) for row in current + [edu]).strip()
                should_join = _should_join(previous=current[-1], current=edu)
                if should_join and self._counter.count_text_tokens_sync(candidate_text) <= self._max_tokens:
                    current.append(edu)
                else:
                    flush()
                    if pending_heading is not None:
                        current.append(pending_heading)
                        pending_heading = None
                    current.append(edu)
            next_row = ordered[idx + 1] if idx + 1 < len(ordered) else None
            if _is_complete(edu) and not _next_requires_join(next_row):
                flush()
        flush()
        if pending_heading is not None:
            built.append(self._materialize(conversation_id=conversation_id, source_row=source_row, chunk_row=chunk_row, edu_rows=[pending_heading], heading_row=None))
        return [item for item in built if item.text.strip()]

    def _materialize(
        self,
        *,
        conversation_id: str,
        source_row: dict[str, Any],
        chunk_row: dict[str, Any],
        edu_rows: list[dict[str, Any]],
        heading_row: dict[str, Any] | None,
    ) -> BuiltCItem:
        text_parts = [_edu_text(row) for row in edu_rows if _edu_text(row)]
        canonical_text = "\n".join(text_parts).strip()
        item_type = _classify_citem(canonical_text, [row.get("features_json") or {} for row in edu_rows], [row.get("edu_kind") or "" for row in edu_rows])
        meta_json = {
            "speaker": source_row.get("role"),
            "source_kind": source_row.get("source_kind"),
            "origin_ref": source_row.get("origin_ref"),
            "page_num": chunk_row.get("page_num"),
            "section_hint": chunk_row.get("section_hint"),
            "modality": _detect_modality(canonical_text),
        }
        spans = [
            {"char_start": _edu_start(row), "char_end": _edu_end(row)}
            for row in edu_rows
            if _edu_end(row) > _edu_start(row)
        ]
        embedding_text = _build_embedding_text(item_type=item_type, chunk_row=chunk_row, source_row=source_row, text=canonical_text)
        return BuiltCItem(
            citem_id=str(uuid.uuid4()),
            semantic_identity_id=str(uuid.uuid4()),
            text=canonical_text,
            embedding_text=embedding_text,
            item_type=item_type,
            edu_ids=[str(row["edu_id"]) for row in edu_rows],
            spans=spans,
            salience=_SALIENCE_BY_TYPE.get(item_type, 0.5),
            meta_json=meta_json,
        )


class MemorySemanticConsumer:
    def __init__(
        self,
        *,
        db: SemanticStoreLike,
        ledger: ConsumerEffectLedger,
        tokenizer: TokenCounterLike,
        embedder: EmbeddingLike,
        qdrant_plane: QdrantWitnessPlane,
        producer: Producer = Producer.CIMA_WORKER,
        edu_segmenter_version: int = 1,
        citem_builder_version: int = 1,
        embedding_model_id: str = "tei",
        embedding_schema_version: int = 1,
    ) -> None:
        self._db = db
        self._ledger = ledger
        self._tokenizer = tokenizer
        self._segmenter = DeterministicEduSegmenter(token_counter=tokenizer, edu_segmenter_version=edu_segmenter_version)
        self._builder = DeterministicCItemBuilder(token_counter=tokenizer, citem_builder_version=citem_builder_version)
        self._embedder = embedder
        self._plane = qdrant_plane
        self._producer = producer
        self._embedding_model_id = embedding_model_id
        self._embedding_schema_version = embedding_schema_version
        self._lineage = DemoLineageService(db)

    async def handle(self, payload_json: dict[str, Any]) -> None:
        envelope = CloudEventEnvelope.model_validate(payload_json)
        if envelope.type == EventType.MEMORY_CHUNK_CREATED:
            await self._handle_chunk_created(envelope)
        elif envelope.type == EventType.MEMORY_EDU_SEGMENTED:
            await self._handle_edu_segmented(envelope)
        elif envelope.type == EventType.MEMORY_CITEM_CREATED:
            await self._handle_citem_created(envelope)
        elif envelope.type in {EventType.MEMORY_SUMMARY_CREATED, EventType.MEMORY_SUMMARY_UPDATED}:
            await self._handle_summary_changed(envelope)

    async def _handle_chunk_created(self, envelope: CloudEventEnvelope) -> None:
        data = ChunkCreatedData.model_validate(envelope.data)
        effect_key = f"chunk-created:{','.join(sorted(str(v) for v in data.chunk_ids))}:edu-v{self._segmenter.version}:chunk-index-v{self._embedding_schema_version}"
        key = ConsumerEffectKey("memory-semantic-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        chunk_rows = await self._resolve_chunk_rows(envelope.subject, [str(v) for v in data.chunk_ids])
        if not chunk_rows:
            await self._ledger.complete(key, details_json={"status": "missing_chunks"})
            return
        source_rows = await self._db.load_demo_sources(envelope.subject, sorted({row["source_id"] for row in chunk_rows}))
        source_map = {row["source_id"]: row for row in source_rows}
        span_ids = [row["source_span_id"] for row in chunk_rows if row.get("source_span_id")]
        span_rows = await self._db.load_demo_source_spans(envelope.subject, span_ids)
        span_map = {row["span_id"]: row for row in span_rows}

        edu_ids: list[uuid.UUID] = []
        chunk_points: list[dict[str, Any]] = []
        for chunk_row in chunk_rows:
            source_row = source_map.get(chunk_row["source_id"])
            if source_row is None:
                continue
            chunk_text = _rehydrate_chunk_text(source_row=source_row, span_row=span_map.get(chunk_row.get("source_span_id")))
            if not chunk_text.strip():
                continue
            source_span = span_map.get(chunk_row.get("source_span_id"))
            source_offset = int(source_span.get("char_start", 0)) if source_span else 0
            for segment in self._segmenter.segment(chunk_text, source_offset=source_offset):
                edu_id = str(uuid.uuid4())
                edu_ids.append(uuid.UUID(edu_id))
                await self._db.save_edu_record(
                    {
                        "edu_id": edu_id,
                        "conversation_id": envelope.subject,
                        "source_id": chunk_row["source_id"],
                        "chunk_id": chunk_row["chunk_id"],
                        "edu_kind": segment.kind,
                        "span_refs_json": [{"char_start": segment.start, "char_end": segment.end}],
                        "features_json": segment.features,
                        "quality": 1.0,
                        "normalizer_version": chunk_row.get("normalizer_version", 1),
                        "edu_segmenter_version": self._segmenter.version,
                    }
                )
            chunk_points.append(
                {
                    "id": chunk_row["chunk_id"],
                    "payload": {
                        "conversation_id": envelope.subject,
                        "scope": "local",
                        "kind": "chunk",
                        "ref_id": chunk_row["chunk_id"],
                        "file_id": chunk_row.get("file_id"),
                        "created_at": chunk_row.get("created_at") or datetime.now(UTC).isoformat(),
                        "source_id": chunk_row["source_id"],
                        "preview_text": _one_line(chunk_text),
                    },
                    "text": chunk_text,
                }
            )

        if chunk_points:
            vectors = await self._embedder.embed_batch([point["text"] for point in chunk_points])
            upsert_rows = [
                {"id": point["id"], "payload": point["payload"], "vector": vector}
                for point, vector in zip(chunk_points, vectors, strict=False)
            ]
            await self._plane.upsert_points(collection_name=self._plane.catalog.chunks, points=upsert_rows)
            for point in chunk_points:
                await self._db.update_chunk_vector_state(
                    point["id"],
                    vector_state="INDEXED",
                    embedding_model_id=self._embedding_model_id,
                    embedding_schema_version=self._embedding_schema_version,
                )
                await self._emit_vector_upserted(
                    conversation_id=envelope.subject,
                    ref_kind="chunk",
                    ref_id=point["id"],
                    collection=self._plane.catalog.chunks,
                    eligible_for_geometry=False,
                    scope="local",
                    item_type=None,
                )

        if edu_ids:
            payload = EduSegmentedData(
                chunk_ids=data.chunk_ids,
                edu_ids=edu_ids,
                edu_segmenter_version=self._segmenter.version,
                normalizer_version=data.normalizer_version,
            )
            event_id = _deterministic_event_id(envelope.subject, EventType.MEMORY_EDU_SEGMENTED, ",".join(sorted(str(v) for v in edu_ids)))
            outbox = CloudEventEnvelope(
                id=event_id,
                type=EventType.MEMORY_EDU_SEGMENTED,
                source=self._producer,
                subject=envelope.subject,
                dataschema="schemas/cima.memory.edu.segmented.v1.json",
                data=payload.model_dump(mode="json"),
            )
            await self._db.append_outbox_event(
                topic=TOPICS.memory_events,
                message_key=conversation_key(envelope.subject),
                payload_json=outbox.model_dump(mode="json"),
            )
        await self._ledger.complete(key, details_json={"status": "segmented", "edu_count": len(edu_ids), "chunk_count": len(chunk_rows)})

    async def _handle_edu_segmented(self, envelope: CloudEventEnvelope) -> None:
        data = EduSegmentedData.model_validate(envelope.data)
        effect_key = f"edu-segmented:{','.join(sorted(str(v) for v in data.edu_ids))}:citem-v{self._builder.version}"
        key = ConsumerEffectKey("memory-semantic-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        chunk_ids = [str(v) for v in data.chunk_ids]
        chunk_rows = await self._resolve_chunk_rows(envelope.subject, chunk_ids)
        if not chunk_rows:
            await self._ledger.complete(key, details_json={"status": "missing_chunks"})
            return
        source_rows = await self._db.load_demo_sources(envelope.subject, sorted({row["source_id"] for row in chunk_rows}))
        source_map = {row["source_id"]: row for row in source_rows}
        all_edu_rows = await self._db.list_edu_records(envelope.subject, edu_ids=[str(v) for v in data.edu_ids])
        edu_by_chunk: dict[str, list[dict[str, Any]]] = {}
        for row in all_edu_rows:
            edu_by_chunk.setdefault(str(row["chunk_id"]), []).append(row)
        created_ids: list[uuid.UUID] = []
        file_updates: dict[str, list[str]] = {}
        for chunk_row in chunk_rows:
            edu_rows = edu_by_chunk.get(chunk_row["chunk_id"], [])
            if not edu_rows:
                continue
            source_row = source_map.get(chunk_row["source_id"])
            if source_row is None:
                continue
            built = self._builder.build(
                conversation_id=envelope.subject,
                source_row=source_row,
                chunk_row=chunk_row,
                edu_rows=edu_rows,
            )
            for ordinal, item in enumerate(built):
                created_ids.append(uuid.UUID(item.citem_id))
                await self._db.save_local_citem_record(
                    {
                        "local_citem_id": item.citem_id,
                        "semantic_identity_id": item.semantic_identity_id,
                        "conversation_id": envelope.subject,
                        "type": item.item_type,
                        "text": item.text,
                        "embedding_text": item.embedding_text,
                        "meta_json": item.meta_json,
                        "provenance_json": {
                            "source_id": chunk_row["source_id"],
                            "chunk_id": chunk_row["chunk_id"],
                            "edu_ids": item.edu_ids,
                            "span_refs": item.spans,
                            "source_span_id": chunk_row.get("source_span_id"),
                        },
                        "validity": "unknown",
                        "salience": item.salience,
                        "vector_state": "NONE",
                        "normalizer_version": chunk_row.get("normalizer_version", 1),
                        "citem_builder_version": self._builder.version,
                    }
                )
                for ev_ordinal, edu_id in enumerate(item.edu_ids):
                    evidence_locator = {
                        "source_span_id": chunk_row.get("source_span_id"),
                        "page_num": chunk_row.get("page_num"),
                        "section_hint": chunk_row.get("section_hint"),
                    }
                    await self._db.save_local_citem_evidence(
                        {
                            "local_citem_id": item.citem_id,
                            "source_id": chunk_row["source_id"],
                            "chunk_id": chunk_row["chunk_id"],
                            "edu_id": edu_id,
                            "ordinal": ev_ordinal,
                            "locator_json": evidence_locator,
                            "conversation_id": envelope.subject,
                        }
                    )
                await self._lineage.record_citem_lineage(
                    conversation_id=envelope.subject,
                    citem_id=item.citem_id,
                    source_id=chunk_row["source_id"],
                    source_span_ids=[chunk_row["source_span_id"]] if chunk_row.get("source_span_id") else None,
                    metadata={"chunk_id": chunk_row["chunk_id"], "type": item.item_type},
                )
                if chunk_row.get("file_id"):
                    file_updates.setdefault(str(chunk_row["file_id"]), []).append(item.citem_id)
        for file_id, new_ids in file_updates.items():
            record = await self._db.get_file_record(file_id)
            if record is None:
                continue
            current_ids = [str(v) for v in getattr(record, "citem_ids", [])]
            merged = list(dict.fromkeys(current_ids + new_ids))
            await self._db.update_file_record(file_id, status="READY", chunk_count=getattr(record, "chunk_count", 0), citem_ids=merged)
        if created_ids:
            payload = CItemCreatedData(
                citem_ids=created_ids,
                citem_builder_version=self._builder.version,
                normalizer_version=data.normalizer_version,
            )
            event_id = _deterministic_event_id(envelope.subject, EventType.MEMORY_CITEM_CREATED, ",".join(sorted(str(v) for v in created_ids)))
            outbox = CloudEventEnvelope(
                id=event_id,
                type=EventType.MEMORY_CITEM_CREATED,
                source=self._producer,
                subject=envelope.subject,
                dataschema="schemas/cima.memory.citem.created.v1.json",
                data=payload.model_dump(mode="json"),
            )
            await self._db.append_outbox_event(
                topic=TOPICS.memory_events,
                message_key=conversation_key(envelope.subject),
                payload_json=outbox.model_dump(mode="json"),
            )
        await self._ledger.complete(key, details_json={"status": "built", "citem_count": len(created_ids)})

    async def _handle_citem_created(self, envelope: CloudEventEnvelope) -> None:
        data = CItemCreatedData.model_validate(envelope.data)
        effect_key = f"citem-created:{','.join(sorted(str(v) for v in data.citem_ids))}:index-v{self._embedding_schema_version}"
        key = ConsumerEffectKey("memory-semantic-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        citem_rows = await self._db.list_local_citem_records(envelope.subject, citem_ids=[str(v) for v in data.citem_ids])
        if not citem_rows:
            await self._ledger.complete(key, details_json={"status": "missing_citems"})
            return
        texts = [row["embedding_text"] for row in citem_rows]
        vectors = await self._embedder.embed_batch(texts)
        upsert_rows: list[dict[str, Any]] = []
        for row, vector in zip(citem_rows, vectors, strict=False):
            created_iso = row.get("created_at") or datetime.now(UTC).isoformat()
            try:
                created_unix = datetime.fromisoformat(created_iso).timestamp()
            except Exception:
                created_unix = datetime.now(UTC).timestamp()
            meta = dict(row.get("meta_json") or {})
            token_count = self._tokenizer.count_text_tokens_sync(row.get("text") or "")
            payload = {
                "conversation_id": envelope.subject,
                "scope": "episodic",
                "scope_status": "active",
                "kind": "citem",
                "w_scope": "local",
                "ref_id": row["local_citem_id"],
                "citem_id": row["local_citem_id"],
                "type": row["type"],
                "item_type": row["type"],
                "content": row["text"],
                "created_at": created_iso,
                "created_at_unix": created_unix,
                "importance": float(row.get("salience", 0.5) or 0.5),
                "confidence": 1.0,
                "validation_label": row.get("validity") or "unknown",
                "conflict_status": "none",
                "phase_ingested": "IDLE",
                "actor": meta.get("speaker") or "agent",
                "motivation": meta.get("source_kind"),
                "dependency_ids": [],
                "token_count": token_count,
            }
            upsert_rows.append({"id": row["local_citem_id"], "payload": payload, "vector": vector})
        if upsert_rows:
            await self._plane.upsert_points(collection_name=self._plane.catalog.local_citems, points=upsert_rows)
        for row in citem_rows:
            await self._db.update_local_citem_vector_state(
                row["local_citem_id"],
                vector_state="INDEXED",
                embedding_model_id=self._embedding_model_id,
                embedding_schema_version=self._embedding_schema_version,
            )
            await self._emit_vector_upserted(
                conversation_id=envelope.subject,
                ref_kind="local_citem",
                ref_id=row["local_citem_id"],
                collection=self._plane.catalog.local_citems,
                eligible_for_geometry=True,
                scope="local",
                item_type=row["type"],
            )
        await self._ledger.complete(key, details_json={"status": "indexed", "citem_count": len(upsert_rows)})

    async def _handle_summary_changed(self, envelope: CloudEventEnvelope) -> None:
        data = SummaryChangedData.model_validate(envelope.data)
        effect_key = f"summary-changed:{data.summary_id}:index-v{self._embedding_schema_version}"
        key = ConsumerEffectKey("memory-semantic-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        summary_rows = await self._db.list_local_summary_records(envelope.subject, summary_ids=[str(data.summary_id)])
        if not summary_rows:
            await self._ledger.complete(key, details_json={"status": "missing_summaries"})
            return
        texts = [row["text"] for row in summary_rows]
        vectors = await self._embedder.embed_batch(texts)
        upsert_rows: list[dict[str, Any]] = []
        for row, vector in zip(summary_rows, vectors, strict=False):
            created_iso = row.get("created_at") or datetime.now(UTC).isoformat()
            try:
                created_unix = datetime.fromisoformat(created_iso).timestamp()
            except Exception:
                created_unix = datetime.now(UTC).timestamp()
            token_count = self._tokenizer.count_text_tokens_sync(row.get("text") or "")
            payload = {
                "conversation_id": envelope.subject,
                "scope": "episodic",
                "scope_status": "active",
                "kind": "summary",
                "w_scope": "local",
                "ref_id": row["local_summary_id"],
                "citem_id": row["local_summary_id"],
                "type": row["level"],
                "item_type": "OBSERVATION",
                "content": row["text"],
                "created_at": created_iso,
                "created_at_unix": created_unix,
                "importance": 0.6,
                "confidence": 1.0,
                "validation_label": "accepted",
                "conflict_status": "none",
                "phase_ingested": "IDLE",
                "actor": "agent",
                "motivation": f"summary:{row['level'].lower()}",
                "dependency_ids": [str(v) for v in dict(row.get("covers_json") or {}).get("origin_citem_ids", [])],
                "token_count": token_count,
            }
            upsert_rows.append({"id": row["local_summary_id"], "payload": payload, "vector": vector})
        if upsert_rows:
            await self._plane.upsert_points(collection_name=self._plane.catalog.local_summaries, points=upsert_rows)
        for row in summary_rows:
            await self._db.update_local_summary_vector_state(
                row["local_summary_id"],
                vector_state="INDEXED",
                embedding_model_id=self._embedding_model_id,
                embedding_schema_version=self._embedding_schema_version,
            )
            await self._emit_vector_upserted(
                conversation_id=envelope.subject,
                ref_kind="local_summary",
                ref_id=row["local_summary_id"],
                collection=self._plane.catalog.local_summaries,
                eligible_for_geometry=True,
                scope="local",
                item_type=row["level"],
            )
        await self._ledger.complete(key, details_json={"status": "indexed", "summary_count": len(upsert_rows)})

    async def _emit_vector_upserted(
        self,
        *,
        conversation_id: str,
        ref_kind: str,
        ref_id: str,
        collection: str,
        eligible_for_geometry: bool,
        scope: str,
        item_type: str | None,
    ) -> None:
        event_id = _deterministic_event_id(conversation_id, EventType.VECTOR_UPSERTED, ref_kind, ref_id, self._embedding_model_id, str(self._embedding_schema_version))
        payload = VectorUpsertedData(
            ref_kind=ref_kind,  # type: ignore[arg-type]
            ref_id=uuid.UUID(ref_id),
            qdrant_collection=collection,
            vector_state="INDEXED",
            embedding_model_id=self._embedding_model_id,
            embedding_schema_version=self._embedding_schema_version,
            eligible_for_geometry=eligible_for_geometry,
            meta=VectorMeta(scope=scope, type=item_type),
        )
        outbox = CloudEventEnvelope(
            id=event_id,
            type=EventType.VECTOR_UPSERTED,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.vector.upserted.v1.json",
            data=payload.model_dump(mode="json"),
        )
        await self._db.append_outbox_event(
            topic=TOPICS.vector_events,
            message_key=conversation_key(conversation_id),
            payload_json=outbox.model_dump(mode="json"),
        )

    async def _resolve_chunk_rows(self, conversation_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        all_rows = await self._db.list_chunk_records(conversation_id)
        wanted = {str(v) for v in chunk_ids}
        rows = [row for row in all_rows if str(row.get("chunk_id")) in wanted]
        rows.sort(key=lambda row: int(row.get("chunk_index", 0)))
        return rows


def _split_sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _edu_features(text: str, *, kind: str, token_count: int) -> dict[str, Any]:
    lower = text.lower().strip()
    first_two = " ".join(lower.split()[:2])
    return {
        "token_count": token_count,
        "ends_with_strong_punct": bool(re.search(r"[.!?]$", text.strip())),
        "starts_with_connector": lower.split()[:1][0] if lower.split()[:1] and (lower.split()[:1][0] in _CONNECTORS or first_two in _CONNECTORS) else None,
        "is_question": text.strip().endswith("?") or lower.startswith(("how ", "what ", "why ", "que ", "qué ", "como ", "cómo ")),
        "is_definition_pattern": bool(_DEFINITION_RE.search(text)),
        "is_decision_pattern": bool(_DECISION_RE.search(text)),
        "is_constraint_pattern": bool(_CONSTRAINT_RE.search(text)),
        "is_risk_pattern": bool(_RISK_RE.search(text)),
        "is_plan_pattern": bool(_PLAN_RE.search(text)),
        "is_attribution": bool(_ATTRIBUTION_RE.search(text)),
        "is_hedged": bool(_HEDGED_RE.search(text)),
        "is_evaluation": bool(_EVAL_RE.search(text)),
        "contains_identifiers": bool(_IDENTIFIER_RE.search(text)),
        "kind": kind,
    }


def _edu_text(row: dict[str, Any]) -> str:
    features = row.get("features_json") or {}
    text = features.get("text")
    if isinstance(text, str):
        return text
    if row.get("text"):
        return str(row["text"])
    span_refs = row.get("span_refs_json") or []
    if span_refs and isinstance(span_refs, list) and row.get("_source_text"):
        first = span_refs[0]
        return str(row["_source_text"])[int(first.get("char_start", 0)):int(first.get("char_end", 0))]
    return ""


def _edu_start(row: dict[str, Any]) -> int:
    span_refs = row.get("span_refs_json") or []
    if span_refs and isinstance(span_refs, list):
        return int(span_refs[0].get("char_start", 0))
    return 0


def _edu_end(row: dict[str, Any]) -> int:
    span_refs = row.get("span_refs_json") or []
    if span_refs and isinstance(span_refs, list):
        return int(span_refs[-1].get("char_end", 0))
    return 0


def _is_complete(row: dict[str, Any]) -> bool:
    features = row.get("features_json") or {}
    return bool(
        features.get("ends_with_strong_punct")
        or features.get("is_question")
        or features.get("is_decision_pattern")
        or features.get("is_constraint_pattern")
        or row.get("edu_kind") in {"list_item", "heading", "code_block", "table_row"}
    )


def _should_join(*, previous: dict[str, Any], current: dict[str, Any]) -> bool:
    prev_features = previous.get("features_json") or {}
    curr_features = current.get("features_json") or {}
    if current.get("edu_kind") == "heading":
        return False
    if curr_features.get("is_attribution") or curr_features.get("starts_with_connector"):
        return True
    if previous.get("edu_kind") == "heading":
        return True
    if not prev_features.get("ends_with_strong_punct"):
        return True
    return False


def _next_requires_join(next_row: dict[str, Any] | None) -> bool:
    if next_row is None:
        return False
    features = next_row.get("features_json") or {}
    return bool(features.get("starts_with_connector") or features.get("is_attribution"))


def _classify_citem(text: str, feature_rows: list[dict[str, Any]], kinds: list[str]) -> str:
    lower = text.lower()
    if any(features.get("is_question") for features in feature_rows):
        return "QUESTION"
    if _CONSTRAINT_RE.search(text):
        return "CONSTRAINT"
    if _DECISION_RE.search(text):
        return "DECISION"
    if _DEFINITION_RE.search(text):
        return "DEFINITION"
    if _RISK_RE.search(text):
        return "RISK"
    if _PLAN_RE.search(text):
        return "PLAN_STEP"
    if _ATTRIBUTION_RE.search(text):
        return "ATTRIBUTION"
    if any(kind == "code_block" for kind in kinds):
        return "CODE_ARTIFACT"
    if _HEDGED_RE.search(text):
        return "HEDGED_FACT"
    if _EVAL_RE.search(text):
        return "EVALUATION"
    if any(features.get("contains_identifiers") for features in feature_rows):
        return "FACT"
    if len(lower.split()) <= 6 and any(kind == "heading" for kind in kinds):
        return "CONTEXT"
    return "FACT"


def _detect_modality(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "question"
    if _CONSTRAINT_RE.search(stripped):
        return "constraint"
    if _DECISION_RE.search(stripped):
        return "decision"
    return "assertion"


def _build_embedding_text(*, item_type: str, chunk_row: dict[str, Any], source_row: dict[str, Any], text: str) -> str:
    tags = [f"type:{item_type}"]
    if chunk_row.get("section_hint"):
        tags.append(f"section:{chunk_row['section_hint']}")
    if source_row.get("source_kind"):
        tags.append(f"origin:{source_row['source_kind']}")
    return "\n".join(tags + [text])


def _rehydrate_chunk_text(*, source_row: dict[str, Any], span_row: dict[str, Any] | None) -> str:
    source_text = str(source_row.get("process_text") or source_row.get("display_text") or "")
    if span_row is None:
        return source_text
    start = int(span_row.get("char_start", 0))
    end = int(span_row.get("char_end", start))
    if 0 <= start <= end <= len(source_text):
        return source_text[start:end]
    return str(span_row.get("preview_text") or source_text)
