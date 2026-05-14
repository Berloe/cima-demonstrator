from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import uuid

import pytest

from cima_demo.api.routers.conversations import delete_conversation
from cima_demo.demo.lifecycle import DemoLifecycleAuditService
from cima_demo.demo.harness.fakes import InMemoryCItemStore, InMemoryDemoDB
from cima_demo.domain.entities import CItem
from cima_demo.memory.lifecycle import LifecycleService


class _FakeStream:
    async def emit(self, *_args, **_kwargs):  # pragma: no cover - not used
        return None


class _FakeMemoryForAudit:
    def __init__(self, store: InMemoryCItemStore) -> None:
        self._store = store

    async def check_promotions_detailed(self, conversation_id: str, chm_reference_counts: dict[str, int]) -> dict[str, object]:
        promoted_ids: list[str] = []
        for item in await self._store.fetch_by_conversation(conversation_id, scope_status="active"):
            if item.scope == "episodic" and chm_reference_counts.get(item.citem_id, 0) >= 2:
                await self._store.update_field(item.citem_id, "scope", "global")
                promoted_ids.append(item.citem_id)
        return {"conversation_id": conversation_id, "n_promoted": len(promoted_ids), "n_demoted": 0, "promoted_ids": promoted_ids, "demoted_ids": []}

    async def run_forget_cycle_detailed(self, conversation_id: str) -> dict[str, object]:
        archived_ids: list[str] = []
        purged_ids: list[str] = []
        items = await self._store.fetch_by_conversation(conversation_id, scope_status=None)
        for item in items:
            if item.scope_status == "active" and item.importance < 0.2:
                await self._store.update_field(item.citem_id, "scope_status", "archived")
                archived_ids.append(item.citem_id)
            elif item.scope_status == "archived":
                await self._store.delete(item.citem_id)
                purged_ids.append(item.citem_id)
        return {
            "conversation_id": conversation_id,
            "n_attenuated": len(archived_ids),
            "n_archived": len(archived_ids),
            "n_purged": len(purged_ids),
            "attenuated_ids": archived_ids,
            "archived_ids": archived_ids,
            "purged_ids": purged_ids,
        }

    async def run_dedup_cycle_detailed(self, conversation_id: str) -> dict[str, object]:
        return {"conversation_id": conversation_id, "n_archived": 0, "archived_duplicate_ids": []}

    async def trigger_l2_check(self, conversation_id: str) -> bool:
        return True


class _GeometryStub:
    def __init__(self, db: InMemoryDemoDB) -> None:
        self._db = db

    async def purge_conversation(self, conversation_id: str) -> None:
        await self._db.delete_geometry_conversation(conversation_id)


@pytest.mark.asyncio
async def test_lifecycle_service_emits_citem_audit_events() -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    service = LifecycleService(
        rel_db=db,
        citem_store=store,
        stream_manager=_FakeStream(),
    )
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    now = datetime.now(UTC)

    promote_item = CItem(conversation_id=conversation_id, content="fact promoted", item_type="FACT", importance=0.8)
    demote_item = CItem(conversation_id=conversation_id, content="fact demoted", item_type="FACT", importance=0.1, scope="global")
    attenuate_item = CItem(conversation_id=conversation_id, content="low importance old", item_type="NOTE", importance=0.1, created_at=now - timedelta(days=40))
    purged_item = CItem(conversation_id=conversation_id, content="archived old", item_type="NOTE", importance=0.1, scope_status="archived", created_at=now - timedelta(days=60), archived_at_unix=(now - timedelta(days=45)).timestamp())
    duplicate_keep = CItem(conversation_id=conversation_id, content="dup", item_type="FACT", content_hash="dup-hash")
    duplicate_drop = CItem(conversation_id=conversation_id, content="dup", item_type="FACT", content_hash="dup-hash", created_at=now + timedelta(seconds=1))

    for item in [promote_item, demote_item, attenuate_item, purged_item, duplicate_keep, duplicate_drop]:
        await store.save(item)

    await service.check_promotions_detailed(conversation_id, {promote_item.citem_id: 5})
    await service.run_dedup_cycle_detailed(conversation_id)
    await service.run_forget_cycle_detailed(conversation_id)

    events = await db.load_citem_audit_events(conversation_id)
    event_types = {row["event_type"] for row in events}
    assert {"PROMOTED", "DEMOTED", "ARCHIVED", "PURGED"}.issubset(event_types)


@pytest.mark.asyncio
async def test_demo_lifecycle_audit_service_records_gc_trace(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    memory = _FakeMemoryForAudit(store)
    audit = DemoLifecycleAuditService(
        rel_db=db,
        citem_store=store,
        memory_service=memory,
        artifacts_root=tmp_path,
    )
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    active = CItem(conversation_id=conversation_id, content="active", item_type="FACT", importance=0.9)
    low = CItem(conversation_id=conversation_id, content="old low", item_type="NOTE", importance=0.1)
    archived = CItem(conversation_id=conversation_id, content="archived", item_type="NOTE", importance=0.1, scope_status="archived")
    for item in [active, low, archived]:
        await store.save(item)

    rec1 = await audit.run_scope_transition_cycle(conversation_id, {active.citem_id: 3})
    rec2 = await audit.run_stale_maintenance_cycle(conversation_id)

    assert rec1.action == "scope_transition_cycle"
    assert rec2.action == "stale_maintenance_cycle"
    assert len(await db.load_demo_gc_audits(conversation_id)) == 2
    trace = (tmp_path / "gc" / f"gc_trace_{conversation_id}.json").read_text(encoding="utf-8")
    assert "scope_transition_cycle" in trace
    assert "stale_maintenance_cycle" in trace


@pytest.mark.asyncio
async def test_delete_conversation_records_gc_trace_and_reconciles(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    store = InMemoryCItemStore()
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    await db.append_turn(conversation_id, "user", "assistant")
    await db.save_turn_metadata(conversation_id, {"phase": "cleanup-regression"})
    item = CItem(conversation_id=conversation_id, content="to delete", item_type="FACT")
    await store.save(item)
    await db.save_demo_source({
        "source_id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "source_kind": "text",
        "display_text": "src",
    })
    await db.save_geometry_run({
        "run_id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "reason": "test",
        "algo_version": "geom_v1",
        "n_items": 1,
        "cluster_count": 1,
        "core_count": 1,
        "bridge_count": 0,
        "created_at": datetime.now(UTC).isoformat(),
    })
    audit = DemoLifecycleAuditService(
        rel_db=db,
        citem_store=store,
        memory_service=_FakeMemoryForAudit(store),
        artifacts_root=tmp_path,
    )
    response = await delete_conversation(
        conversation_id=conversation_id,
        _auth=None,
        db=db,
        citem_store=store,
        geometry_commands=_GeometryStub(db),
        lifecycle_audit_service=audit,
    )
    assert response.status_code == 204
    counts = await audit.collect_counts(conversation_id)
    assert counts["conversations"] == 0
    assert counts["task_metadata"] == 0
    assert counts["citems_total"] == 0
    audits = await db.load_demo_gc_audits(conversation_id)
    assert audits[-1]["action"] == "conversation_delete"
    assert audits[-1]["consistency"]["cleanup_ok"] is True
    assert (tmp_path / "gc" / f"gc_trace_{conversation_id}.json").exists()
