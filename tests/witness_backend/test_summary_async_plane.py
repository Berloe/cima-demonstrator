from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import (
    CItemCreatedData,
    CloudEventEnvelope,
    EventType,
    Producer,
    SummaryRequestedData,
)
from cima_demo.witness_backend.semantic_pipeline import MemorySemanticConsumer
from cima_demo.witness_backend.summary_pipeline import MemorySummaryConsumer
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


async def _seed_local_citems(db: InMemoryDemoDB, conversation_id: str) -> list[str]:
    await db.create_conversation(conversation_id)
    rows = [
        {
            "local_citem_id": "11111111-1111-1111-1111-111111111111",
            "semantic_identity_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "conversation_id": conversation_id,
            "type": "DECISION",
            "text": "Decidimos usar Kafka como plano asíncrono principal.",
            "embedding_text": "DECISION Kafka plano asíncrono principal",
            "salience": 0.95,
            "meta_json": {"speaker": "user", "source_kind": "chat_user"},
            "provenance_json": {},
            "vector_state": "INDEXED",
            "created_at": "2026-04-29T20:00:00+00:00",
            "updated_at": "2026-04-29T20:00:00+00:00",
            "validity": "accepted",
            "normalizer_version": 1,
            "citem_builder_version": 1,
        },
        {
            "local_citem_id": "22222222-2222-2222-2222-222222222222",
            "semantic_identity_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "conversation_id": conversation_id,
            "type": "CONSTRAINT",
            "text": "La implementación debe usar exclusivamente Open Source.",
            "embedding_text": "CONSTRAINT Open Source only",
            "salience": 0.98,
            "meta_json": {"speaker": "user", "source_kind": "chat_user"},
            "provenance_json": {},
            "vector_state": "INDEXED",
            "created_at": "2026-04-29T20:01:00+00:00",
            "updated_at": "2026-04-29T20:01:00+00:00",
            "validity": "accepted",
            "normalizer_version": 1,
            "citem_builder_version": 1,
        },
        {
            "local_citem_id": "33333333-3333-3333-3333-333333333333",
            "semantic_identity_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "conversation_id": conversation_id,
            "type": "PLAN_STEP",
            "text": "El siguiente paso es cerrar el DDL y el plano Kafka end-to-end.",
            "embedding_text": "PLAN_STEP cerrar DDL y plano Kafka",
            "salience": 0.68,
            "meta_json": {"speaker": "assistant", "source_kind": "chat_assistant"},
            "provenance_json": {},
            "vector_state": "INDEXED",
            "created_at": "2026-04-29T20:02:00+00:00",
            "updated_at": "2026-04-29T20:02:00+00:00",
            "validity": "unknown",
            "normalizer_version": 1,
            "citem_builder_version": 1,
        },
    ]
    for row in rows:
        await db.save_local_citem_record(row)
    return [row["local_citem_id"] for row in rows]


@pytest.mark.asyncio
async def test_citem_created_schedules_epoch_and_master_summary_requests() -> None:
    db = InMemoryDemoDB()
    conversation_id = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
    ids = await _seed_local_citems(db, conversation_id)
    consumer = MemorySummaryConsumer(db=db, ledger=ConsumerEffectLedger(db), tokenizer=_FakeTokenizer())
    envelope = CloudEventEnvelope(
        type=EventType.MEMORY_CITEM_CREATED,
        source=Producer.CIMA_WORKER,
        subject=conversation_id,
        dataschema="schemas/cima.memory.citem.created.v1.json",
        data=CItemCreatedData(
            citem_ids=[UUID(v) for v in ids],
            citem_builder_version=1,
            normalizer_version=1,
        ).model_dump(mode="json"),
    )

    await consumer.handle(envelope.model_dump(mode="json"))

    requests = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row.get("topic") == TOPICS.summary_cmd and row.get("payload_json") is not None
    ]
    assert {evt.data["level"] for evt in requests} == {"EPOCH", "MASTER"}


@pytest.mark.asyncio
async def test_summary_requested_persists_local_summary_and_origins() -> None:
    db = InMemoryDemoDB()
    conversation_id = "bbbbbbbb-1111-2222-3333-cccccccccccc"
    ids = await _seed_local_citems(db, conversation_id)
    consumer = MemorySummaryConsumer(db=db, ledger=ConsumerEffectLedger(db), tokenizer=_FakeTokenizer())
    req = CloudEventEnvelope(
        type=EventType.SUMMARY_REQUESTED,
        source=Producer.CIMA_WORKER,
        subject=conversation_id,
        dataschema="schemas/cima.summary.requested.v1.json",
        data=SummaryRequestedData(
            level="EPOCH",
            epoch_no=1,
            reason="EPOCH_CLOSED",
            priority="NORMAL",
            target_citem_ids=[UUID(v) for v in ids],
        ).model_dump(mode="json"),
    )

    await consumer.handle(req.model_dump(mode="json"))

    summaries = await db.list_local_summary_records(conversation_id)
    assert summaries
    assert summaries[0]["level"] == "EPOCH"
    origins = await db.list_local_summary_origins(summaries[0]["local_summary_id"])
    assert origins
    created_env = CloudEventEnvelope.model_validate(
        _latest_outbox_event(db, EventType.MEMORY_SUMMARY_CREATED, topic=TOPICS.memory_events)
    )
    assert created_env.data["summary_id"] == summaries[0]["local_summary_id"]


@pytest.mark.asyncio
async def test_memory_summary_created_indexes_local_summary_and_emits_vector_upserted() -> None:
    db = InMemoryDemoDB()
    conversation_id = "cccccccc-1111-2222-3333-dddddddddddd"
    ids = await _seed_local_citems(db, conversation_id)
    summary_consumer = MemorySummaryConsumer(db=db, ledger=ConsumerEffectLedger(db), tokenizer=_FakeTokenizer())
    req = CloudEventEnvelope(
        type=EventType.SUMMARY_REQUESTED,
        source=Producer.CIMA_WORKER,
        subject=conversation_id,
        dataschema="schemas/cima.summary.requested.v1.json",
        data=SummaryRequestedData(
            level="MASTER",
            reason="PERIODIC",
            priority="NORMAL",
            target_citem_ids=[UUID(v) for v in ids],
        ).model_dump(mode="json"),
    )
    await summary_consumer.handle(req.model_dump(mode="json"))
    summary_event = _latest_outbox_event(db, EventType.MEMORY_SUMMARY_CREATED, topic=TOPICS.memory_events)

    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    semantic_consumer = MemorySemanticConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=3,
    )

    await semantic_consumer.handle(summary_event)

    summaries = await db.list_local_summary_records(conversation_id)
    assert summaries[0]["vector_state"] == "INDEXED"
    assert client.collections[plane.catalog.local_summaries]
    vector_events = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row.get("topic") == TOPICS.vector_events and row.get("payload_json") is not None
    ]
    assert any(evt.data.get("ref_kind") == "local_summary" for evt in vector_events)


@pytest.mark.asyncio
async def test_summary_consumer_skips_requests_for_deleting_conversation() -> None:
    db = InMemoryDemoDB()
    conversation_id = "dddddddd-1111-2222-3333-eeeeeeeeeeee"
    ids = await _seed_local_citems(db, conversation_id)
    db.conversations[conversation_id]["status"] = "DELETING"
    consumer = MemorySummaryConsumer(db=db, ledger=ConsumerEffectLedger(db), tokenizer=_FakeTokenizer())
    req = CloudEventEnvelope(
        type=EventType.SUMMARY_REQUESTED,
        source=Producer.CIMA_WORKER,
        subject=conversation_id,
        dataschema="schemas/cima.summary.requested.v1.json",
        data=SummaryRequestedData(
            level="EPOCH",
            epoch_no=1,
            reason="EPOCH_CLOSED",
            priority="NORMAL",
            target_citem_ids=[UUID(v) for v in ids],
        ).model_dump(mode="json"),
    )

    await consumer.handle(req.model_dump(mode="json"))

    summaries = await db.list_local_summary_records(conversation_id)
    assert summaries == []
