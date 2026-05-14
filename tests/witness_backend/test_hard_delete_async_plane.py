from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from cima_demo.api import settings as settings_module
from cima_demo.api.routers.conversations import delete_conversation
from cima_demo.demo.harness.fakes import InMemoryCItemStore, InMemoryDemoDB
from cima_demo.demo.lifecycle import DemoLifecycleAuditService
from cima_demo.geometry.boundary import GeometryCommandPublisher
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType
from cima_demo.witness_backend.hard_delete import HardDeleteConsumer, HardDeleteScheduler
from cima_demo.witness_backend.topic_catalog import TOPICS


class _GeometryStub:
    async def purge_conversation(self, conversation_id: str, *, delete_run_id: str | None = None) -> None:
        return None


class _MemoryStub:
    def __init__(self, store: InMemoryCItemStore) -> None:
        self._store = store

    async def fetch_by_conversation(self, conversation_id: str):
        return await self._store.fetch_by_conversation(conversation_id)


@pytest.mark.asyncio
async def test_hard_delete_scheduler_marks_conversation_deleting_and_enqueues_request() -> None:
    db = InMemoryDemoDB()
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)

    scheduler = HardDeleteScheduler(db)
    result = await scheduler.request(conversation_id, reason="USER_REQUEST")

    assert result.accepted is True
    row = await db.get_conversation(conversation_id)
    assert row is not None
    assert row["status"] == "DELETING"
    assert db.outbox_rows[-1]["topic"] == TOPICS.conversation_events
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert envelope.type == EventType.CONVERSATION_HARD_DELETE_REQUESTED
    assert envelope.subject == conversation_id


@pytest.mark.asyncio
async def test_hard_delete_consumer_purges_and_emits_completed_event() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    from cima_demo.domain.entities import CItem
    item = CItem(conversation_id=conversation_id, content="to-delete", item_type="FACT")
    await store.save(item)
    scheduler = HardDeleteScheduler(db)
    request = await scheduler.request(conversation_id)
    request_row = db.outbox_rows[-1]

    consumer = HardDeleteConsumer(store=db, citem_store=store, ledger=ConsumerEffectLedger(db))
    await consumer.handle(payload_json=request_row["payload_json"])

    assert await db.get_conversation(conversation_id) is None
    assert await store.fetch_by_conversation(conversation_id) == []
    delete_run = db.delete_runs[request.delete_run_id]
    assert delete_run["status"] == "SUCCEEDED"
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert envelope.type == EventType.CONVERSATION_HARD_DELETE_COMPLETED


@pytest.mark.asyncio
async def test_hard_delete_consumer_requests_geometry_purge_and_persists_auditable_stats() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    from cima_demo.domain.entities import CItem

    item = CItem(conversation_id=conversation_id, content="to-delete", item_type="FACT")
    await store.save(item)
    scheduler = HardDeleteScheduler(db)
    request = await scheduler.request(conversation_id)
    request_row = db.outbox_rows[-1]

    consumer = HardDeleteConsumer(
        store=db,
        citem_store=store,
        ledger=ConsumerEffectLedger(db),
        geometry_commands=GeometryCommandPublisher(db),
    )
    await consumer.handle(payload_json=request_row["payload_json"])

    geom_commands = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row["topic"] == TOPICS.geom_cmd
    ]
    assert geom_commands
    assert geom_commands[-1].type == EventType.GEOM_PURGE
    assert geom_commands[-1].subject == conversation_id
    assert geom_commands[-1].data["delete_run_id"] == request.delete_run_id

    stats = db.delete_runs[request.delete_run_id]["stats_json"]
    assert stats["geometry_purge_requested"] == 1
    assert stats["qdrant_points_deleted_by_collection"]["local_scoped"] == 1



@pytest.mark.asyncio
async def test_hard_delete_consumer_purges_ephemeral_leases_and_emits_vector_deleted() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    ephemeral_id = str(uuid.uuid4())
    await db.save_ephemeral_vector_record({
        "ephemeral_id": ephemeral_id,
        "conversation_id": conversation_id,
        "origin_ref_kind": "local_citem",
        "origin_ref_id": None,
        "qdrant_collection": "cima_ephemeral",
        "lifecycle_state": "ACTIVE",
        "vector_state": "EPHEMERAL",
        "embedding_model_id": "tei",
        "embedding_schema_version": 1,
        "eligible_for_geometry": False,
        "meta_json": {"scope": "local", "type": "FACT"},
        "expires_at": "2099-01-01T00:00:00+00:00",
    })
    scheduler = HardDeleteScheduler(db)
    await scheduler.request(conversation_id)
    request_row = db.outbox_rows[-1]

    consumer = HardDeleteConsumer(store=db, citem_store=store, ledger=ConsumerEffectLedger(db))
    await consumer.handle(payload_json=request_row["payload_json"])

    vector_deleted = [
        CloudEventEnvelope.model_validate(row["payload_json"])
        for row in db.outbox_rows
        if row["topic"] == TOPICS.vector_events
    ]
    assert vector_deleted
    assert vector_deleted[0].type == EventType.VECTOR_DELETED
    assert vector_deleted[0].data["ref_kind"] == "ephemeral"
    assert vector_deleted[0].data["reason"] == "HARD_DELETE"
    assert await db.get_conversation(conversation_id) is None
    stats = db.delete_runs[next(iter(db.delete_runs))]["stats_json"]
    assert stats["ephemeral_records_purged"] == 1


@pytest.mark.asyncio
async def test_hard_delete_consumer_destructively_purges_local_planes_and_preserves_promoted_global_state() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)

    # Local runtime / witness-plane state.
    db.task_memory[conversation_id] = {"conversation_id": conversation_id, "phase": "IDLE"}
    db.turns[conversation_id] = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    db.demo_sources["src-local"] = {"source_id": "src-local", "conversation_id": conversation_id}
    db.demo_source_spans["span-local"] = {"span_id": "span-local", "conversation_id": conversation_id}
    db.file_records["file-local"] = {"file_id": "file-local", "conversation_id": conversation_id}
    db.chunk_records["chunk-local"] = {"chunk_id": "chunk-local", "conversation_id": conversation_id}
    db.edu_records["edu-local"] = {"edu_id": "edu-local", "conversation_id": conversation_id}
    db.local_citem_records["lc-local"] = {"local_citem_id": "lc-local", "conversation_id": conversation_id}
    db.local_citem_evidence_rows.append({"conversation_id": conversation_id, "local_citem_id": "lc-local", "ordinal": 0})
    db.local_summary_records["ls-local"] = {"local_summary_id": "ls-local", "conversation_id": conversation_id}
    db.local_summary_origin_rows.append({"conversation_id": conversation_id, "local_summary_id": "ls-local", "origin_id": "lc-local"})
    db.demo_lineage_edges.append({"conversation_id": conversation_id, "src_id": "lc-local", "dst_id": "chunk-local"})
    db.demo_summary_resolutions.append({"conversation_id": conversation_id, "summary_id": "ls-local"})
    db.demo_context_snapshots["ctx-local"] = {"context_id": "ctx-local", "conversation_id": conversation_id}
    db.demo_answer_lineage.append({"conversation_id": conversation_id, "run_id": "run-local"})
    db.demo_runs["run-local"] = {"run_id": "run-local", "conversation_id": conversation_id}
    db.demo_run_phases["run-local"] = [{"phase": "IDLE"}]
    db.demo_checkpoints["run-local"] = [{"checkpoint_id": "cp-local"}]
    db.demo_handoff_manifests["handoff-local"] = {"handoff_id": "handoff-local", "conversation_id": conversation_id}
    db.demo_handoff_validations["handoff-local"] = {"handoff_id": "handoff-local"}
    db.demo_handoff_restores["restore-local"] = {"restore_id": "restore-local", "handoff_id": "handoff-local", "target_conversation_id": conversation_id}
    db.geometry_runs.append({"conversation_id": conversation_id, "run_id": "geom-run-local"})
    db.geometry_item_states.append({"conversation_id": conversation_id, "ref_id": "lc-local"})
    db.geometry_cluster_states.append({"conversation_id": conversation_id, "cluster_id": "cluster-local"})
    await db.save_ephemeral_vector_record({
        "ephemeral_id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "origin_ref_kind": "local_citem",
        "origin_ref_id": "lc-local",
        "qdrant_collection": "cima_ephemeral",
        "lifecycle_state": "ACTIVE",
        "vector_state": "EPHEMERAL",
        "embedding_model_id": "tei",
        "embedding_schema_version": 1,
        "eligible_for_geometry": False,
        "meta_json": {"scope": "local", "type": "FACT"},
        "expires_at": "2099-01-01T00:00:00+00:00",
    })

    from cima_demo.domain.entities import CItem

    await store.save(CItem(conversation_id=conversation_id, content="delete me", item_type="FACT"))

    # Global witness-plane state must survive the local destructive delete.
    db.global_citem_records["gc-global"] = {
        "global_citem_id": "gc-global",
        "origin_conversation_id": conversation_id,
        "semantic_identity_id": "sid-1",
    }
    db.global_citem_evidence_rows.append({"global_citem_id": "gc-global", "conversation_id": conversation_id})
    db.global_summary_records["gs-global"] = {"global_summary_id": "gs-global", "source_conversation_id": conversation_id}
    db.global_summary_origin_rows.append({"global_summary_id": "gs-global", "origin_id": "gc-global"})

    scheduler = HardDeleteScheduler(db)
    request = await scheduler.request(conversation_id)
    consumer = HardDeleteConsumer(store=db, citem_store=store, ledger=ConsumerEffectLedger(db))
    await consumer.handle(payload_json=db.outbox_rows[-1]["payload_json"])

    assert request.accepted is True
    assert await db.get_conversation(conversation_id) is None
    assert await store.fetch_by_conversation(conversation_id) == []

    # Local planes are gone.
    assert conversation_id not in db.task_memory
    assert conversation_id not in db.turns
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_sources.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_source_spans.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.file_records.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.chunk_records.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.edu_records.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.local_citem_records.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.local_citem_evidence_rows)
    assert not any(row.get("conversation_id") == conversation_id for row in db.local_summary_records.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.local_summary_origin_rows)
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_lineage_edges)
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_summary_resolutions)
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_context_snapshots.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_answer_lineage)
    assert not any(row.get("conversation_id") == conversation_id for row in db.demo_runs.values())
    assert not any(row.get("conversation_id") == conversation_id for row in db.geometry_runs)
    assert not any(row.get("conversation_id") == conversation_id for row in db.geometry_item_states)
    assert not any(row.get("conversation_id") == conversation_id for row in db.geometry_cluster_states)
    assert not any(row.get("conversation_id") == conversation_id for row in db.ephemeral_vector_records.values())
    assert "handoff-local" not in db.demo_handoff_manifests
    assert "handoff-local" not in db.demo_handoff_validations
    assert "restore-local" not in db.demo_handoff_restores

    # Promoted global witness-plane state survives.
    assert "gc-global" in db.global_citem_records
    assert any(row.get("global_citem_id") == "gc-global" for row in db.global_citem_evidence_rows)
    assert "gs-global" in db.global_summary_records
    assert any(row.get("global_summary_id") == "gs-global" for row in db.global_summary_origin_rows)


@pytest.mark.asyncio
async def test_hard_delete_consumer_destructive_delete_keeps_other_conversations_intact() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    victim_conversation_id = str(uuid.uuid4())
    survivor_conversation_id = str(uuid.uuid4())
    await db.create_conversation(victim_conversation_id)
    await db.create_conversation(survivor_conversation_id)

    db.demo_sources["src-victim"] = {"source_id": "src-victim", "conversation_id": victim_conversation_id}
    db.demo_sources["src-survivor"] = {"source_id": "src-survivor", "conversation_id": survivor_conversation_id}
    db.local_citem_records["lc-victim"] = {"local_citem_id": "lc-victim", "conversation_id": victim_conversation_id}
    db.local_citem_records["lc-survivor"] = {"local_citem_id": "lc-survivor", "conversation_id": survivor_conversation_id}

    from cima_demo.domain.entities import CItem

    await store.save(CItem(conversation_id=victim_conversation_id, content="delete me", item_type="FACT"))
    survivor_item = CItem(conversation_id=survivor_conversation_id, content="keep me", item_type="FACT")
    await store.save(survivor_item)

    scheduler = HardDeleteScheduler(db)
    await scheduler.request(victim_conversation_id)
    consumer = HardDeleteConsumer(store=db, citem_store=store, ledger=ConsumerEffectLedger(db))
    await consumer.handle(payload_json=db.outbox_rows[-1]["payload_json"])

    assert await db.get_conversation(victim_conversation_id) is None
    survivor_row = await db.get_conversation(survivor_conversation_id)
    assert survivor_row is not None
    assert survivor_row["status"] == "ACTIVE"
    assert any(row.get("conversation_id") == survivor_conversation_id for row in db.demo_sources.values())
    assert any(row.get("conversation_id") == survivor_conversation_id for row in db.local_citem_records.values())
    survivor_items = await store.fetch_by_conversation(survivor_conversation_id)
    assert [item.citem_id for item in survivor_items] == [survivor_item.citem_id]


@pytest.mark.asyncio
async def test_delete_route_in_full_mode_schedules_async_delete_and_keeps_standalone_path_clean(tmp_path: Path) -> None:
    previous = settings_module._settings
    settings_module._settings = settings_module.Settings(runtime_mode="full", api_key_required=False)
    try:
        db = InMemoryDemoDB()
        store = InMemoryCItemStore()
        conversation_id = str(uuid.uuid4())
        await db.create_conversation(conversation_id)
        await db.append_turn(conversation_id, "user", "assistant")
        audit = DemoLifecycleAuditService(
            rel_db=db,
            citem_store=store,
            memory_service=_MemoryStub(store),
            artifacts_root=tmp_path,
        )
        response = await delete_conversation(
            conversation_id=conversation_id,
            _auth=None,
            db=db,
            citem_store=store,
            geometry_commands=_GeometryStub(),
            lifecycle_audit_service=audit,
        )
        assert response.status_code == 202
        row = await db.get_conversation(conversation_id)
        assert row is not None and row["status"] == "DELETING"
        assert db.outbox_rows[-1]["topic"] == TOPICS.conversation_events
    finally:
        settings_module._settings = previous
