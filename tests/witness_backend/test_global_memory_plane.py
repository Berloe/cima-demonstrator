from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from cima_demo.domain.entities import SummaryNode
from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import (
    CItemCreatedData,
    CItemPromotedGlobalData,
    CloudEventEnvelope,
    EventType,
    Producer,
)
from cima_demo.witness_backend.global_memory import GlobalPromotionConsumer
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


def _latest_outbox_events(db: InMemoryDemoDB, event_type: EventType, *, topic: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in db.outbox_rows:
        if topic is not None and row.get("topic") != topic:
            continue
        payload = row.get("payload_json")
        if payload is None:
            continue
        envelope = CloudEventEnvelope.model_validate(payload)
        if envelope.type == event_type:
            rows.append(payload)
    return rows


async def _seed_local_citems(db: InMemoryDemoDB, conversation_id: str) -> list[str]:
    await db.create_conversation(conversation_id)
    rows = [
        {
            "local_citem_id": "11111111-1111-1111-1111-111111111111",
            "semantic_identity_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "conversation_id": conversation_id,
            "type": "DECISION",
            "text": "Decidimos usar PostgreSQL como source of truth.",
            "embedding_text": "DECISION PostgreSQL source of truth",
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
            "text": "El siguiente paso es cerrar el DDL.",
            "embedding_text": "PLAN_STEP cerrar DDL",
            "salience": 0.68,
            "meta_json": {"speaker": "assistant", "source_kind": "chat_assistant"},
            "provenance_json": {},
            "vector_state": "INDEXED",
            "created_at": "2026-04-29T20:02:00+00:00",
            "updated_at": "2026-04-29T20:02:00+00:00",
            "validity": "accepted",
            "normalizer_version": 1,
            "citem_builder_version": 1,
        },
    ]
    for row in rows:
        await db.save_local_citem_record(row)
        await db.save_local_citem_evidence(
            {
                "local_citem_id": row["local_citem_id"],
                "source_id": None,
                "chunk_id": "44444444-4444-4444-4444-444444444444",
                "edu_id": None,
                "ordinal": 0,
                "locator_json": {"span": [0, 42]},
            }
        )
    return [row["local_citem_id"] for row in rows]


@pytest.mark.asyncio
async def test_citem_created_schedules_promoted_global_events_for_eligible_items() -> None:
    db = InMemoryDemoDB()
    conversation_id = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
    ids = await _seed_local_citems(db, conversation_id)
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = GlobalPromotionConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=3,
    )
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

    promoted = _latest_outbox_events(db, EventType.MEMORY_CITEM_PROMOTED_GLOBAL, topic=TOPICS.memory_events)
    promoted_ids = {CloudEventEnvelope.model_validate(row).data["local_citem_id"] for row in promoted}
    assert promoted_ids == {
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    }


@pytest.mark.asyncio
async def test_promoted_global_materialises_global_memory_and_indexes_it() -> None:
    db = InMemoryDemoDB()
    conversation_id = "bbbbbbbb-1111-2222-3333-cccccccccccc"
    ids = await _seed_local_citems(db, conversation_id)
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = GlobalPromotionConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=3,
    )
    schedule = CloudEventEnvelope(
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
    await consumer.handle(schedule.model_dump(mode="json"))
    promoted = _latest_outbox_events(db, EventType.MEMORY_CITEM_PROMOTED_GLOBAL, topic=TOPICS.memory_events)

    for payload in promoted:
        await consumer.handle(payload)

    global_rows = await db.list_global_citem_records(origin_conversation_id=conversation_id)
    assert len(global_rows) == 2
    assert all(row["vector_state"] == "INDEXED" for row in global_rows)
    assert client.collections[plane.catalog.global_citems]
    vector_events = _latest_outbox_events(db, EventType.VECTOR_UPSERTED, topic=TOPICS.vector_events)
    assert any(CloudEventEnvelope.model_validate(row).data.get("ref_kind") == "global_citem" for row in vector_events)




@pytest.mark.asyncio
async def test_load_summaries_and_fetch_nodes_at_level_prefer_witness_rows_when_available() -> None:
    db = InMemoryDemoDB()
    conversation_id = "abababab-1111-2222-3333-cdefcdefcdef"
    await db.create_conversation(conversation_id)
    legacy_l1 = SummaryNode(
        node_id="legacy-l1",
        conversation_id=conversation_id,
        level=1,
        content="legacy l1 summary",
        token_count=3,
        parent_id=None,
    )
    await db.save_summary(legacy_l1)
    await db.save_local_summary_record({
        "local_summary_id": "11111111-2222-3333-4444-555555555555",
        "conversation_id": conversation_id,
        "level": "EPOCH",
        "text": "witness epoch summary",
        "covers_json": {"origin_citem_ids": []},
        "created_at": "2026-04-30T00:00:00+00:00",
        "updated_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    })
    await db.save_global_summary_record({
        "global_summary_id": "66666666-7777-8888-9999-000000000000",
        "level": "MASTER",
        "origin_conversation_id": conversation_id,
        "text": "witness global summary",
        "covers_json": {"origin_global_citem_ids": []},
        "created_at": "2026-04-30T00:00:00+00:00",
        "updated_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    })

    loaded = await db.load_summaries(conversation_id)
    assert {node.node_id for node in loaded} == {
        "11111111-2222-3333-4444-555555555555",
        "66666666-7777-8888-9999-000000000000",
    }
    assert {getattr(node, "summary_resolution_mode", None) for node in loaded} == {"witness_first"}

    tops = await db.fetch_nodes_at_level(3, conversation_id, parentless_only=True)
    assert [node.node_id for node in tops] == ["66666666-7777-8888-9999-000000000000"]

@pytest.mark.asyncio
async def test_fetch_pyramid_tops_surfaces_witness_global_summaries_for_origin_conversation() -> None:
    db = InMemoryDemoDB()
    conversation_id = "cccccccc-1111-2222-3333-dddddddddddd"
    ids = await _seed_local_citems(db, conversation_id)
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = GlobalPromotionConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=3,
    )
    schedule = CloudEventEnvelope(
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
    await consumer.handle(schedule.model_dump(mode="json"))
    promoted = _latest_outbox_events(db, EventType.MEMORY_CITEM_PROMOTED_GLOBAL, topic=TOPICS.memory_events)
    for payload in promoted:
        await consumer.handle(payload)

    summary_rows = await db.list_global_summary_records(origin_conversation_id=conversation_id)
    assert summary_rows
    tops = await db.fetch_pyramid_tops(conversation_id, limit=10)
    assert any(node.node_id == summary_rows[0]["global_summary_id"] for node in tops)
    assert plane.catalog.global_summaries in client.collections


@pytest.mark.asyncio
async def test_fetch_pyramid_tops_prefers_witness_summaries_over_legacy_nodes_when_available() -> None:
    db = InMemoryDemoDB()
    conversation_id = "eeeeeeee-1111-2222-3333-ffffffffffff"
    await db.create_conversation(conversation_id)
    legacy_node = SummaryNode(
        node_id="legacy-top",
        conversation_id=conversation_id,
        level=2,
        content="legacy top summary",
        token_count=3,
        parent_id=None,
    )
    await db.save_summary(legacy_node)
    await db.save_local_summary_record({
        "local_summary_id": "99999999-9999-9999-9999-999999999999",
        "conversation_id": conversation_id,
        "level": "MASTER",
        "text": "witness master summary",
        "covers_json": {"origin_citem_ids": []},
        "created_at": "2026-04-30T00:00:00+00:00",
        "updated_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    })

    tops = await db.fetch_pyramid_tops(conversation_id, limit=10)

    assert [node.node_id for node in tops] == ["99999999-9999-9999-9999-999999999999"]
    assert getattr(tops[0], "summary_resolution_mode", None) == "witness_first"


@pytest.mark.asyncio
async def test_fetch_pyramid_tops_marks_legacy_nodes_as_explicit_fallback() -> None:
    db = InMemoryDemoDB()
    conversation_id = "12121212-1111-2222-3333-343434343434"
    await db.create_conversation(conversation_id)
    legacy_node = SummaryNode(
        node_id="legacy-only",
        conversation_id=conversation_id,
        level=2,
        content="legacy only summary",
        token_count=4,
        parent_id=None,
    )
    await db.save_summary(legacy_node)

    tops = await db.fetch_pyramid_tops(conversation_id, limit=10)

    assert [node.node_id for node in tops] == ["legacy-only"]
    assert getattr(tops[0], "summary_resolution_mode", None) == "legacy_fallback"


@pytest.mark.asyncio
async def test_global_promotion_consumer_skips_late_events_for_deleting_conversation() -> None:
    db = InMemoryDemoDB()
    conversation_id = "dddddddd-1111-2222-3333-ffffffffffff"
    ids = await _seed_local_citems(db, conversation_id)
    db.conversations[conversation_id]["status"] = "DELETING"
    client = _FakeQdrantClient()
    plane = QdrantWitnessPlane(client=client, catalog=QdrantCollectionCatalog(), dense_dim=3)
    await plane.ensure_ready()
    consumer = GlobalPromotionConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        qdrant_plane=plane,
        embedding_model_id="tei-demo",
        embedding_schema_version=3,
    )
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

    promoted = _latest_outbox_events(db, EventType.MEMORY_CITEM_PROMOTED_GLOBAL, topic=TOPICS.memory_events)
    assert promoted == []
