from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from cima_demo.demo.context import DemoContextService
from cima_demo.demo.harness.fakes import HarnessContextBuilder, HarnessMemoryService, InMemoryCItemStore, InMemoryDemoDB
from cima_demo.domain.entities import CItem, TaskMemory
from cima_demo.domain.value_objects import ContextBudget
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.ephemeral import EphemeralVectorRegistry
from cima_demo.witness_backend.ephemeral_runtime import EphemeralRuntimeMirror
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType, GcRequestedData, Producer
from cima_demo.witness_backend.hard_delete import HardDeleteConsumer, HardDeleteScheduler
from cima_demo.witness_backend.maintenance import MaintenanceConsumer
from tests.witness_backend.test_qdrant_collection_split import _FakeQdrantClient


@dataclass
class _CaptureMirror:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def mirror_context_items(self, *, conversation_id: str, items: list[dict[str, Any]]):
        self.calls.append({
            "conversation_id": conversation_id,
            "items": [dict(item) for item in items],
        })
        return {"accepted_items": len(items)}


@pytest.mark.asyncio
async def test_demo_context_service_build_mirrors_selected_items_into_ephemeral_lane() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    memory = HarnessMemoryService(store=store, db=db)
    builder = HarnessContextBuilder(store=store, db=db, include_summaries=False)
    mirror = _CaptureMirror()
    service = DemoContextService(
        base_builder=builder,
        memory_service=memory,
        rel_db=db,
        ephemeral_runtime=mirror,
    )

    item = CItem(conversation_id="conv-1", content="runtime ephemeral candidate", item_type="FACT", scope="episodic")
    await store.save(item)
    await db.save_local_citem_record({"local_citem_id": item.citem_id, "conversation_id": "conv-1"})
    await db.save_local_citem_evidence({
        "local_citem_id": item.citem_id,
        "conversation_id": "conv-1",
        "source_id": "src-1",
        "source_span_id": "span-1",
        "locator_json": {"source_id": "src-1", "source_span_id": "span-1"},
    })

    view = await service.build(
        phase="IDLE",
        task_memory=TaskMemory(conversation_id="conv-1"),
        plan=None,
        query="runtime candidate",
        conversation_id="conv-1",
        budget=ContextBudget(max_tokens=256, overhead_tokens=32),
    )

    assert view.items
    assert mirror.calls
    assert mirror.calls[0]["conversation_id"] == "conv-1"
    assert mirror.calls[0]["items"][0]["ref_id"] == item.citem_id


class _EmbedBatchStub:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


@pytest.mark.asyncio
async def test_ephemeral_lane_reconcile_and_hard_delete_flow_is_end_to_end_safe() -> None:
    db = InMemoryDemoDB()
    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)

    client = _FakeQdrantClient()
    catalog = QdrantCollectionCatalog(
        local_citems="local-citems",
        local_summaries="local-summaries",
        chunks="chunks",
        global_citems="global-citems",
        global_summaries="global-summaries",
        ephemeral="ephemeral",
    )
    plane = QdrantWitnessPlane(client=client, catalog=catalog, dense_dim=3)
    registry = EphemeralVectorRegistry(db)
    mirror = EphemeralRuntimeMirror(
        plane=plane,
        embedder=_EmbedBatchStub(),
        registry=registry,
        conversation_reader=db,
        ttl_seconds=600,
        max_items=4,
    )

    stats = await mirror.mirror_context_items(
        conversation_id=conversation_id,
        items=[{"ref_id": str(uuid4()), "ref_kind": "citem", "content": "runtime mirrored item", "item_type": "FACT"}],
    )
    assert stats.accepted_items == 1
    assert db.ephemeral_vector_records
    ephemeral_id = next(iter(db.ephemeral_vector_records))
    assert ephemeral_id in client.collections[catalog.ephemeral]

    client.collections[catalog.ephemeral].pop(ephemeral_id, None)

    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="RECONCILE", conversation_id=conversation_id, maintenance_run_id=maintenance_run_id)
    reconcile = CloudEventEnvelope(
        type=EventType.GC_RECONCILE_REQUESTED,
        source=Producer.CIMA_API,
        subject=conversation_id,
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="RECONCILE").model_dump(mode="json"),
    )
    await MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ConsumerEffectLedger(db)).handle(
        payload_json=reconcile.model_dump(mode="json")
    )

    assert db.ephemeral_vector_records[ephemeral_id]["lifecycle_state"] == "PURGED"
    vector_events = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row["topic"] == "cima.vector.events.v1"
        and row["payload_json"].get("type") == EventType.VECTOR_DELETED.value
    ]
    assert any(evt.data["ref_kind"] == "ephemeral" and evt.data["reason"] == "RECONCILE" for evt in vector_events)

    scheduler = HardDeleteScheduler(db)
    request = await scheduler.request(conversation_id)
    consumer = HardDeleteConsumer(store=db, citem_store=InMemoryCItemStore(), ledger=ConsumerEffectLedger(db))
    await consumer.handle(payload_json=db.outbox_rows[-1]["payload_json"])

    assert request.accepted is True
    assert await db.get_conversation(conversation_id) is None
