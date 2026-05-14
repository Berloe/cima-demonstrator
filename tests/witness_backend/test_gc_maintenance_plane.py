from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType, GcRequestedData, Producer
from cima_demo.witness_backend.maintenance import MaintenanceConsumer, MaintenanceScheduler
from cima_demo.witness_backend.topic_catalog import TOPICS


class _FakeQdrantPlane:
    def __init__(self) -> None:
        self.catalog = QdrantCollectionCatalog(
            local_citems="local-citems",
            local_summaries="local-summaries",
            chunks="chunks",
            global_citems="global-citems",
            global_summaries="global-summaries",
            ephemeral="ephemeral",
        )
        self.delete_calls: list[tuple[str, list[str]]] = []
        self.ephemeral_sweeps = 0
        self.collections: dict[str, set[str]] = {
            self.catalog.local_citems: set(),
            self.catalog.local_summaries: set(),
            self.catalog.chunks: set(),
            self.catalog.global_citems: set(),
            self.catalog.global_summaries: set(),
            self.catalog.ephemeral: set(),
        }
        self.conversation_membership: dict[str, dict[str, set[str]]] = {}

    def add_point(self, *, collection_name: str, point_id: str, conversation_id: str | None = None) -> None:
        self.collections.setdefault(collection_name, set()).add(str(point_id))
        if conversation_id is not None:
            self.conversation_membership.setdefault(collection_name, {}).setdefault(str(conversation_id), set()).add(str(point_id))

    async def delete_point_ids(self, *, collection_name: str, point_ids: list[str]) -> int:
        ids = [str(v) for v in point_ids]
        self.delete_calls.append((collection_name, ids))
        bucket = self.collections.setdefault(collection_name, set())
        for point_id in ids:
            bucket.discard(point_id)
        members = self.conversation_membership.setdefault(collection_name, {})
        for rows in members.values():
            rows.difference_update(ids)
        return len(ids)

    async def list_point_ids_by_conversation(self, *, collection_name: str, conversation_id: str) -> list[str]:
        return sorted(self.conversation_membership.get(collection_name, {}).get(str(conversation_id), set()))

    async def list_all_point_ids(self, *, collection_name: str) -> list[str]:
        return sorted(self.collections.get(collection_name, set()))

    async def sweep_ephemeral_expired(self, *, now: datetime | None = None) -> int:
        self.ephemeral_sweeps += 1
        return 2


@pytest.mark.asyncio
async def test_maintenance_scheduler_writes_requested_run_and_outbox_event() -> None:
    db = InMemoryDemoDB()
    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)

    scheduler = MaintenanceScheduler(db)
    result = await scheduler.request_thinning(conversation_id)

    assert result.accepted is True
    run_row = db.maintenance_runs[result.maintenance_run_id]
    assert run_row["kind"] == "THINNING"
    assert run_row["status"] == "REQUESTED"
    assert run_row["conversation_id"] == conversation_id
    assert db.outbox_rows[-1]["topic"] == TOPICS.gc_events
    assert db.outbox_rows[-1]["payload_json"]["type"] == EventType.GC_THINNING_REQUESTED.value


@pytest.mark.asyncio
async def test_maintenance_scheduler_can_request_reconcile() -> None:
    db = InMemoryDemoDB()
    scheduler = MaintenanceScheduler(db)

    result = await scheduler.request_reconcile()

    assert result.accepted is True
    assert db.maintenance_runs[result.maintenance_run_id]["kind"] == "RECONCILE"
    assert db.outbox_rows[-1]["payload_json"]["type"] == EventType.GC_RECONCILE_REQUESTED.value


@pytest.mark.asyncio
async def test_maintenance_consumer_thins_only_stale_unpinned_local_rows() -> None:
    db = InMemoryDemoDB()
    ledger = ConsumerEffectLedger(db)
    plane = _FakeQdrantPlane()
    consumer = MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ledger, thinning_age_hours=24)

    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)
    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="THINNING", conversation_id=conversation_id, maintenance_run_id=maintenance_run_id)

    stale = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
    fresh = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

    chunk_id = str(uuid4())
    await db.save_chunk_record({
        "chunk_id": chunk_id,
        "conversation_id": conversation_id,
        "source_id": str(uuid4()),
        "chunk_kind": "doc_chunk",
        "chunk_index": 0,
        "vector_state": "INDEXED",
        "created_at": stale,
    })
    local_citem_id = str(uuid4())
    await db.save_local_citem_record({
        "local_citem_id": local_citem_id,
        "semantic_identity_id": str(uuid4()),
        "conversation_id": conversation_id,
        "type": "FACT",
        "text": "Old local fact",
        "embedding_text": "Old local fact",
        "vector_state": "INDEXED",
        "created_at": stale,
        "is_pinned": False,
        "was_cited": False,
    })
    pinned_id = str(uuid4())
    await db.save_local_citem_record({
        "local_citem_id": pinned_id,
        "semantic_identity_id": str(uuid4()),
        "conversation_id": conversation_id,
        "type": "DECISION",
        "text": "Pinned decision",
        "embedding_text": "Pinned decision",
        "vector_state": "INDEXED",
        "created_at": stale,
        "is_pinned": True,
        "was_cited": False,
    })
    fresh_id = str(uuid4())
    await db.save_local_citem_record({
        "local_citem_id": fresh_id,
        "semantic_identity_id": str(uuid4()),
        "conversation_id": conversation_id,
        "type": "FACT",
        "text": "Fresh local fact",
        "embedding_text": "Fresh local fact",
        "vector_state": "INDEXED",
        "created_at": fresh,
        "is_pinned": False,
        "was_cited": False,
    })
    local_summary_id = str(uuid4())
    await db.save_local_summary_record({
        "local_summary_id": local_summary_id,
        "conversation_id": conversation_id,
        "level": "MASTER",
        "text": "Old summary",
        "covers_json": {},
        "vector_state": "INDEXED",
        "created_at": stale,
        "updated_at": stale,
        "is_pinned": False,
        "was_cited": False,
    })
    global_id = str(uuid4())
    await db.save_global_citem_record({
        "global_citem_id": global_id,
        "semantic_identity_id": str(uuid4()),
        "origin_conversation_id": conversation_id,
        "promotion_origin_local_citem_id": local_citem_id,
        "type": "FACT",
        "text": "Global fact",
        "embedding_text": "Global fact",
        "vector_state": "INDEXED",
        "created_at": stale,
    })

    envelope = CloudEventEnvelope(
        type=EventType.GC_THINNING_REQUESTED,
        source=Producer.CIMA_API,
        subject=conversation_id,
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="THINNING").model_dump(mode="json"),
    )
    await consumer.handle(payload_json=envelope.model_dump(mode="json"))

    assert db.maintenance_runs[maintenance_run_id]["status"] == "SUCCEEDED"
    assert db.chunk_records[chunk_id]["vector_state"] == "THINNED"
    assert db.local_citem_records[local_citem_id]["vector_state"] == "THINNED"
    assert db.local_summary_records[local_summary_id]["vector_state"] == "THINNED"
    assert db.local_citem_records[pinned_id]["vector_state"] == "INDEXED"
    assert db.local_citem_records[fresh_id]["vector_state"] == "INDEXED"
    assert db.global_citem_records[global_id]["vector_state"] == "INDEXED"

    assert ("chunks", [chunk_id]) in plane.delete_calls
    assert ("local-citems", [local_citem_id]) in plane.delete_calls
    assert ("local-summaries", [local_summary_id]) in plane.delete_calls

    vector_deleted = [row for row in db.outbox_rows if row["topic"] == TOPICS.vector_events]
    assert len(vector_deleted) == 3
    assert {row["payload_json"]["data"]["ref_kind"] for row in vector_deleted} == {"chunk", "local_citem", "local_summary"}


@pytest.mark.asyncio
async def test_maintenance_consumer_reconcile_removes_orphan_points_and_emits_events() -> None:
    db = InMemoryDemoDB()
    ledger = ConsumerEffectLedger(db)
    plane = _FakeQdrantPlane()
    consumer = MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ledger, thinning_age_hours=24)

    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)
    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="RECONCILE", conversation_id=conversation_id, maintenance_run_id=maintenance_run_id)

    live_chunk_id = str(uuid4())
    orphan_chunk_id = str(uuid4())
    await db.save_chunk_record({
        "chunk_id": live_chunk_id,
        "conversation_id": conversation_id,
        "source_id": str(uuid4()),
        "chunk_kind": "doc_chunk",
        "chunk_index": 0,
        "vector_state": "INDEXED",
        "created_at": datetime.now(UTC).isoformat(),
    })
    await db.save_local_citem_record({
        "local_citem_id": str(uuid4()),
        "semantic_identity_id": str(uuid4()),
        "conversation_id": conversation_id,
        "type": "FACT",
        "text": "live",
        "embedding_text": "live",
        "vector_state": "THINNED",
        "created_at": datetime.now(UTC).isoformat(),
    })
    plane.add_point(collection_name=plane.catalog.chunks, point_id=live_chunk_id, conversation_id=conversation_id)
    plane.add_point(collection_name=plane.catalog.chunks, point_id=orphan_chunk_id, conversation_id=conversation_id)

    envelope = CloudEventEnvelope(
        type=EventType.GC_RECONCILE_REQUESTED,
        source=Producer.CIMA_API,
        subject=conversation_id,
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="RECONCILE").model_dump(mode="json"),
    )
    await consumer.handle(payload_json=envelope.model_dump(mode="json"))

    assert db.maintenance_runs[maintenance_run_id]["status"] == "SUCCEEDED"
    stats = db.maintenance_runs[maintenance_run_id]["stats_json"]
    assert stats["orphan_points_deleted"] == 1
    assert stats["orphan_points_by_collection"][plane.catalog.chunks] == 1
    assert (plane.catalog.chunks, [orphan_chunk_id]) in plane.delete_calls
    vector_deleted = [row for row in db.outbox_rows if row["topic"] == TOPICS.vector_events]
    assert any(row["payload_json"]["data"]["reason"] == "ORPHAN_CLEANUP" for row in vector_deleted)


@pytest.mark.asyncio
async def test_maintenance_consumer_handles_ephemeral_expiry() -> None:
    db = InMemoryDemoDB()
    ledger = ConsumerEffectLedger(db)
    plane = _FakeQdrantPlane()
    consumer = MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ledger, thinning_age_hours=24)

    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)
    now = datetime.now(UTC)
    due_ephemeral_id = str(uuid4())
    retry_ephemeral_id = str(uuid4())
    await db.save_ephemeral_vector_record({
        "ephemeral_id": due_ephemeral_id,
        "conversation_id": conversation_id,
        "origin_ref_kind": "chunk",
        "origin_ref_id": str(uuid4()),
        "qdrant_collection": plane.catalog.ephemeral,
        "lifecycle_state": "ACTIVE",
        "vector_state": "EPHEMERAL",
        "meta_json": {"scope": "local", "type": "FACT"},
        "created_at": (now - timedelta(hours=2)).isoformat(),
        "expires_at": (now - timedelta(minutes=5)).isoformat(),
    })
    await db.save_ephemeral_vector_record({
        "ephemeral_id": retry_ephemeral_id,
        "conversation_id": conversation_id,
        "origin_ref_kind": "local_citem",
        "origin_ref_id": str(uuid4()),
        "qdrant_collection": plane.catalog.ephemeral,
        "lifecycle_state": "EXPIRED",
        "vector_state": "EPHEMERAL",
        "meta_json": {"scope": "local", "type": "DECISION"},
        "created_at": (now - timedelta(hours=3)).isoformat(),
        "expires_at": (now - timedelta(hours=1)).isoformat(),
        "expired_at": (now - timedelta(minutes=30)).isoformat(),
    })
    plane.add_point(collection_name=plane.catalog.ephemeral, point_id=due_ephemeral_id, conversation_id=conversation_id)
    plane.add_point(collection_name=plane.catalog.ephemeral, point_id=retry_ephemeral_id, conversation_id=conversation_id)

    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="EPHEMERAL_EXPIRY", conversation_id=None, maintenance_run_id=maintenance_run_id)
    envelope = CloudEventEnvelope(
        type=EventType.GC_EPHEMERAL_EXPIRY_REQUESTED,
        source=Producer.CIMA_API,
        subject="*",
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="EPHEMERAL_EXPIRY").model_dump(mode="json"),
    )

    await consumer.handle(payload_json=envelope.model_dump(mode="json"))

    assert any(collection == plane.catalog.ephemeral and set(ids) == {due_ephemeral_id, retry_ephemeral_id} for collection, ids in plane.delete_calls)
    assert db.ephemeral_vector_records[due_ephemeral_id]["lifecycle_state"] == "PURGED"
    assert db.ephemeral_vector_records[due_ephemeral_id]["expired_at"] is not None
    assert db.ephemeral_vector_records[due_ephemeral_id]["purged_at"] is not None
    assert db.ephemeral_vector_records[retry_ephemeral_id]["lifecycle_state"] == "PURGED"
    assert db.ephemeral_vector_records[retry_ephemeral_id]["expired_at"] is not None
    assert db.ephemeral_vector_records[retry_ephemeral_id]["purged_at"] is not None
    assert db.maintenance_runs[maintenance_run_id]["status"] == "SUCCEEDED"
    stats = db.maintenance_runs[maintenance_run_id]["stats_json"]
    assert stats["ephemeral_due_records"] == 2
    assert stats["ephemeral_records_marked_expired"] == 1
    assert stats["ephemeral_records_marked_purged"] == 2
    assert stats["ephemeral_points_deleted"] == 2
    vector_deleted = [row for row in db.outbox_rows if row["topic"] == TOPICS.vector_events]
    assert len(vector_deleted) == 2
    assert {row["payload_json"]["data"]["ref_kind"] for row in vector_deleted} == {"ephemeral"}
    assert {row["payload_json"]["data"]["reason"] for row in vector_deleted} == {"EXPIRED"}


@pytest.mark.asyncio
async def test_maintenance_consumer_reconcile_marks_missing_indexed_rows_thinned_and_emits_reconcile_event() -> None:
    db = InMemoryDemoDB()
    ledger = ConsumerEffectLedger(db)
    plane = _FakeQdrantPlane()
    consumer = MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ledger, thinning_age_hours=24)

    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)
    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="RECONCILE", conversation_id=conversation_id, maintenance_run_id=maintenance_run_id)

    local_citem_id = str(uuid4())
    await db.save_local_citem_record({
        "local_citem_id": local_citem_id,
        "semantic_identity_id": str(uuid4()),
        "conversation_id": conversation_id,
        "type": "FACT",
        "text": "indexed but missing point",
        "embedding_text": "indexed but missing point",
        "vector_state": "INDEXED",
        "created_at": datetime.now(UTC).isoformat(),
    })

    envelope = CloudEventEnvelope(
        type=EventType.GC_RECONCILE_REQUESTED,
        source=Producer.CIMA_API,
        subject=conversation_id,
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="RECONCILE").model_dump(mode="json"),
    )
    await consumer.handle(payload_json=envelope.model_dump(mode="json"))

    stats = db.maintenance_runs[maintenance_run_id]["stats_json"]
    assert stats["missing_indexed_rows_marked"] == 1
    assert stats["missing_indexed_rows_by_collection"][plane.catalog.local_citems] == 1
    assert db.local_citem_records[local_citem_id]["vector_state"] == "THINNED"
    vector_deleted = [row for row in db.outbox_rows if row["topic"] == TOPICS.vector_events]
    assert any(
        row["payload_json"]["data"]["ref_kind"] == "local_citem"
        and row["payload_json"]["data"]["ref_id"] == local_citem_id
        and row["payload_json"]["data"]["reason"] == "RECONCILE"
        for row in vector_deleted
    )


@pytest.mark.asyncio
async def test_maintenance_consumer_reconcile_converges_ephemeral_missing_rows_and_orphans() -> None:
    db = InMemoryDemoDB()
    ledger = ConsumerEffectLedger(db)
    plane = _FakeQdrantPlane()
    consumer = MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ledger, thinning_age_hours=24)

    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)
    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="RECONCILE", conversation_id=conversation_id, maintenance_run_id=maintenance_run_id)

    missing_ephemeral_id = str(uuid4())
    await db.save_ephemeral_vector_record({
        "ephemeral_id": missing_ephemeral_id,
        "conversation_id": conversation_id,
        "origin_ref_kind": "local_citem",
        "origin_ref_id": str(uuid4()),
        "qdrant_collection": plane.catalog.ephemeral,
        "lifecycle_state": "ACTIVE",
        "vector_state": "EPHEMERAL",
        "meta_json": {"scope": "local", "type": "FACT"},
        "created_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    })
    orphan_ephemeral_id = str(uuid4())
    plane.add_point(collection_name=plane.catalog.ephemeral, point_id=orphan_ephemeral_id, conversation_id=conversation_id)

    envelope = CloudEventEnvelope(
        type=EventType.GC_RECONCILE_REQUESTED,
        source=Producer.CIMA_API,
        subject=conversation_id,
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="RECONCILE").model_dump(mode="json"),
    )
    await consumer.handle(payload_json=envelope.model_dump(mode="json"))

    stats = db.maintenance_runs[maintenance_run_id]["stats_json"]
    assert stats["ephemeral_rows_purged"] == 1


@pytest.mark.asyncio
async def test_maintenance_consumer_reconcile_all_scans_local_global_and_ephemeral_scopes() -> None:
    db = InMemoryDemoDB()
    ledger = ConsumerEffectLedger(db)
    plane = _FakeQdrantPlane()
    consumer = MaintenanceConsumer(store=db, qdrant_plane=plane, ledger=ledger, thinning_age_hours=24)

    conv_a = str(uuid4())
    conv_b = str(uuid4())
    await db.create_conversation(conv_a)
    await db.create_conversation(conv_b)
    maintenance_run_id = str(uuid4())
    await db.begin_maintenance_run(kind="RECONCILE", conversation_id=None, maintenance_run_id=maintenance_run_id)

    missing_local_id = str(uuid4())
    await db.save_local_citem_record({
        "local_citem_id": missing_local_id,
        "semantic_identity_id": str(uuid4()),
        "conversation_id": conv_a,
        "type": "FACT",
        "text": "missing local point",
        "embedding_text": "missing local point",
        "vector_state": "INDEXED",
        "created_at": datetime.now(UTC).isoformat(),
    })

    orphan_chunk_id = str(uuid4())
    plane.add_point(collection_name=plane.catalog.chunks, point_id=orphan_chunk_id, conversation_id=conv_b)

    orphan_global_summary_id = str(uuid4())
    plane.add_point(collection_name=plane.catalog.global_summaries, point_id=orphan_global_summary_id)

    missing_ephemeral_id = str(uuid4())
    await db.save_ephemeral_vector_record({
        "ephemeral_id": missing_ephemeral_id,
        "conversation_id": conv_a,
        "origin_ref_kind": "local_citem",
        "origin_ref_id": missing_local_id,
        "qdrant_collection": plane.catalog.ephemeral,
        "lifecycle_state": "ACTIVE",
        "vector_state": "EPHEMERAL",
        "meta_json": {"scope": "local", "type": "FACT"},
        "created_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    })

    envelope = CloudEventEnvelope(
        type=EventType.GC_RECONCILE_REQUESTED,
        source=Producer.CIMA_API,
        subject="*",
        dataschema="schemas/cima.gc.requested.v1.json",
        data=GcRequestedData(maintenance_run_id=maintenance_run_id, reason="RECONCILE").model_dump(mode="json"),
    )
    await consumer.handle(payload_json=envelope.model_dump(mode="json"))

    stats = db.maintenance_runs[maintenance_run_id]["stats_json"]
    assert stats["conversations_checked"] == 2
    assert stats["missing_indexed_rows_marked"] == 1
    assert stats["missing_indexed_rows_by_collection"][plane.catalog.local_citems] == 1
    assert stats["orphan_points_by_collection"][plane.catalog.chunks] == 1
    assert stats["orphan_points_by_collection"][plane.catalog.global_summaries] == 1
    assert stats["ephemeral_rows_purged"] == 1
    assert db.local_citem_records[missing_local_id]["vector_state"] == "THINNED"
    assert db.ephemeral_vector_records[missing_ephemeral_id]["lifecycle_state"] == "PURGED"
    assert db.ephemeral_vector_records[missing_ephemeral_id]["expired_at"] is not None
    assert db.ephemeral_vector_records[missing_ephemeral_id]["purged_at"] is not None
    vector_deleted = [row["payload_json"]["data"] for row in db.outbox_rows if row["topic"] == TOPICS.vector_events]
    assert any(row["ref_kind"] == "ephemeral" and row["ref_id"] == missing_ephemeral_id and row["reason"] == "RECONCILE" for row in vector_deleted)
