from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.demo.lineage import DemoLineageService
from cima_demo.infrastructure.files.chunker import SemanticChunkerAdapter
from cima_demo.infrastructure.files.processor import FileProcessingAdapter
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType
from cima_demo.witness_backend.semantic_pipeline import MemorySemanticConsumer
from cima_demo.witness_backend.source_ingest import MemorySourceConsumer, SourceRegistrationService
from cima_demo.witness_backend.topic_catalog import TOPICS


class _FakeTokenizer:
    def count_text_tokens_sync(self, text: str) -> int:
        return max(1, len((text or "").split()))


class _FakeEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text or "")), 0.5, float(len((text or "").split()))] for text in texts]


class _FakePoint:
    def __init__(self, *, id: str, payload: dict[str, Any] | None = None, vector: dict[str, list[float]] | None = None) -> None:
        self.id = id
        self.payload = payload or {}
        self.vector = vector or {}


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, _FakePoint]] = {}

    async def get_collections(self) -> Any:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    async def create_collection(self, *, collection_name: str, **_: Any) -> None:
        self.collections.setdefault(collection_name, {})

    async def create_payload_index(self, **_: Any) -> None:
        return None

    async def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        bucket = self.collections.setdefault(collection_name, {})
        for point in points:
            bucket[str(point.id)] = _FakePoint(id=str(point.id), payload=dict(point.payload), vector=dict(point.vector))

    async def delete(self, *, collection_name: str, points_selector: Any) -> None:
        return None

    async def set_payload(self, *, collection_name: str, points: list[str], payload: dict[str, Any]) -> None:
        bucket = self.collections.setdefault(collection_name, {})
        for pid in points:
            row = bucket.get(str(pid))
            if row is not None:
                row.payload = {**row.payload, **payload}


def _latest_outbox_event(db: InMemoryDemoDB, event_type: EventType, *, topic: str | None = None) -> dict[str, Any]:
    for row in reversed(db.outbox_rows):
        if topic is not None and row.get("topic") != topic:
            continue
        payload = row.get("payload_json")
        if payload is None:
            continue
        envelope = CloudEventEnvelope.model_validate(payload)
        if envelope.type == event_type:
            return payload
    raise AssertionError(f"event {event_type} not found")


async def _prepare_chunk_event(db: InMemoryDemoDB, tmp_path: Path, *, text: str) -> dict[str, Any]:
    conversation_id = "55555555-5555-5555-5555-555555555555"
    await db.create_conversation(conversation_id)
    registration = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )
    await registration.register_text(
        conversation_id=conversation_id,
        text=text,
        role="user",
        source_kind="chat_user",
        external_provider="librechat",
        external_message_id="msg-semantic-1",
    )
    source_event = _latest_outbox_event(db, EventType.MEMORY_SOURCE_REGISTERED, topic=TOPICS.memory_events)
    source_consumer = MemorySourceConsumer(
        db=db,
        chunker=SemanticChunkerAdapter(token_counter=lambda value: max(1, len(value.split()))),
        file_processor=FileProcessingAdapter(),
        ledger=ConsumerEffectLedger(db),
    )
    await source_consumer.handle(source_event)
    return _latest_outbox_event(db, EventType.MEMORY_CHUNK_CREATED, topic=TOPICS.memory_events)


@pytest.mark.asyncio
async def test_chunk_created_segments_edus_and_indexes_chunks(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    chunk_event = await _prepare_chunk_event(
        db,
        tmp_path,
        text="## Objetivo\n\nDebemos cerrar el contrato. Sin embargo, primero validamos restricciones. ¿Seguimos?",
    )
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = MemorySemanticConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=1,
    )

    await consumer.handle(chunk_event)

    edus = await db.list_edu_records("55555555-5555-5555-5555-555555555555")
    assert edus
    edu_event = CloudEventEnvelope.model_validate(
        _latest_outbox_event(db, EventType.MEMORY_EDU_SEGMENTED, topic=TOPICS.memory_events)
    )
    assert len(edu_event.data["edu_ids"]) == len(edus)
    assert client.collections[plane.catalog.chunks]
    for chunk in await db.list_chunk_records("55555555-5555-5555-5555-555555555555"):
        assert chunk["vector_state"] == "INDEXED"
    vector_events = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row.get("topic") == TOPICS.vector_events and row.get("payload_json") is not None
    ]
    assert any(evt.type == EventType.VECTOR_UPSERTED for evt in vector_events)


@pytest.mark.asyncio
async def test_edu_segmented_builds_local_citems_with_evidence_and_lineage(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    chunk_event = await _prepare_chunk_event(
        db,
        tmp_path,
        text="Decidimos usar Kafka. La restricción principal es Open Source only. El siguiente paso es cerrar el DDL.",
    )
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = MemorySemanticConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=1,
    )

    await consumer.handle(chunk_event)
    edu_event = _latest_outbox_event(db, EventType.MEMORY_EDU_SEGMENTED, topic=TOPICS.memory_events)
    await consumer.handle(edu_event)

    citems = await db.list_local_citem_records("55555555-5555-5555-5555-555555555555")
    assert citems
    assert {row["type"] for row in citems}.intersection({"DECISION", "CONSTRAINT", "PLAN_STEP", "FACT"})
    evidence_rows = await db.list_local_citem_evidence(citems[0]["local_citem_id"])
    assert evidence_rows
    assert any(edge.get("src_kind") == "citem" for edge in db.demo_lineage_edges)
    citem_event = CloudEventEnvelope.model_validate(
        _latest_outbox_event(db, EventType.MEMORY_CITEM_CREATED, topic=TOPICS.memory_events)
    )
    assert len(citem_event.data["citem_ids"]) == len(citems)


@pytest.mark.asyncio
async def test_citem_created_indexes_local_citems_and_emits_vector_upserted(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    chunk_event = await _prepare_chunk_event(
        db,
        tmp_path,
        text="Definition: a chunk is evidence. We should keep C-items as semantic memory.",
    )
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = MemorySemanticConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=2,
    )

    await consumer.handle(chunk_event)
    await consumer.handle(_latest_outbox_event(db, EventType.MEMORY_EDU_SEGMENTED, topic=TOPICS.memory_events))
    citem_event = _latest_outbox_event(db, EventType.MEMORY_CITEM_CREATED, topic=TOPICS.memory_events)
    await consumer.handle(citem_event)

    citems = await db.list_local_citem_records("55555555-5555-5555-5555-555555555555")
    assert citems
    assert client.collections[plane.catalog.local_citems]
    for row in citems:
        assert row["vector_state"] == "INDEXED"
        assert row["embedding_model_id"] == "tei-demo"
        assert row["embedding_schema_version"] == 2
    vector_events = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row.get("topic") == TOPICS.vector_events and row.get("payload_json") is not None
    ]
    citem_vector_events = [evt for evt in vector_events if evt.data.get("ref_kind") == "local_citem"]
    assert citem_vector_events


@pytest.mark.asyncio
async def test_semantic_consumer_skips_late_chunk_events_for_deleting_conversation(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    chunk_event = await _prepare_chunk_event(db, tmp_path, text="Alpha. Beta. Gamma.")
    conversation_id = CloudEventEnvelope.model_validate(chunk_event).subject
    db.conversations[conversation_id]["status"] = "DELETING"
    client = _FakeQdrantClient()
    catalog = QdrantCollectionCatalog()
    plane = QdrantWitnessPlane(client=client, catalog=catalog, dense_dim=3)
    await plane.ensure_ready()
    consumer = MemorySemanticConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=3,
    )

    await consumer.handle(chunk_event)

    assert await db.list_edu_records(conversation_id) == []
    assert client.collections[catalog.chunks] == {}
