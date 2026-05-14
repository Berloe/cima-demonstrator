from __future__ import annotations

from uuid import uuid4

import pytest

from cima_demo.geometry.boundary import (
    DirectGeometryBoundary,
    GeometryCommandPublisher,
    GeometryReadModelService,
)
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType
from cima_demo.witness_backend.topic_catalog import TOPICS


class _FakeDB:
    def __init__(self) -> None:
        self.outbox: list[dict] = []
        self.rows = [
            {
                "conversation_id": "conv-1",
                "ref_kind": "local_citem",
                "ref_id": str(uuid4()),
                "run_id": str(uuid4()),
                "cluster_top1": "c_001",
                "cluster_top2": None,
                "w1": 0.8,
                "w2": None,
                "margin": 0.8,
                "is_core": True,
                "is_bridge_candidate": False,
                "centrality": 0.8,
                "label": None,
                "updated_at": "2026-04-29T00:00:00+00:00",
            }
        ]
        self.cluster_rows = [
            {
                "conversation_id": "conv-1",
                "cluster_id": "c_001",
                "run_id": str(uuid4()),
                "mass": 0.6,
                "medoid_ref_id": str(uuid4()),
                "summary_id": None,
                "label": None,
                "updated_at": "2026-04-29T00:00:00+00:00",
            }
        ]

    async def append_outbox_event(self, *, topic: str, message_key: str, payload_json: dict, headers_json: dict | None = None) -> int:
        self.outbox.append({
            "topic": topic,
            "message_key": message_key,
            "payload_json": payload_json,
            "headers_json": headers_json or {},
        })
        return len(self.outbox)

    async def load_geometry_item_states(self, conversation_id: str, ref_ids: list[str] | None = None) -> list[dict]:
        rows = [row for row in self.rows if row["conversation_id"] == conversation_id]
        if ref_ids is not None:
            rows = [row for row in rows if row["ref_id"] in ref_ids]
        return rows

    async def load_geometry_cluster_states(self, conversation_id: str) -> list[dict]:
        return [row for row in self.cluster_rows if row["conversation_id"] == conversation_id]


class _SyncGeometryService:
    def __init__(self) -> None:
        self.scheduled: list[tuple[str, str]] = []
        self.purged: list[str] = []

    def schedule_recompute(self, conversation_id: str, *, reason: str = "context_snapshot") -> None:
        self.scheduled.append((conversation_id, reason))

    async def purge_conversation(self, conversation_id: str) -> None:
        self.purged.append(conversation_id)

    async def get_item_hints(self, *, conversation_id: str, ref_ids: list[str]):
        return {}

    async def load_all_item_hints(self, *, conversation_id: str):
        return []

    async def get_cluster_hints(self, *, conversation_id: str):
        return []


@pytest.mark.asyncio
async def test_geometry_command_publisher_writes_cloudevents_to_outbox() -> None:
    db = _FakeDB()
    publisher = GeometryCommandPublisher(db)

    await publisher.schedule_recompute("conv-1", reason="context_snapshot")
    await publisher.purge_conversation("conv-1", delete_run_id="00000000-0000-0000-0000-000000000001")

    assert [row["topic"] for row in db.outbox] == [TOPICS.geom_cmd, TOPICS.geom_cmd]
    recompute = CloudEventEnvelope.model_validate(db.outbox[0]["payload_json"])
    purge = CloudEventEnvelope.model_validate(db.outbox[1]["payload_json"])
    assert recompute.type == EventType.GEOM_RECOMPUTE
    assert purge.type == EventType.GEOM_PURGE
    assert recompute.subject == "conv-1"
    assert purge.subject == "conv-1"


@pytest.mark.asyncio
async def test_geometry_read_model_service_loads_hints_from_cima_read_model() -> None:
    db = _FakeDB()
    service = GeometryReadModelService(db)
    ref_id = db.rows[0]["ref_id"]

    hints = await service.get_item_hints(conversation_id="conv-1", ref_ids=[ref_id])
    clusters = await service.get_cluster_hints(conversation_id="conv-1")

    assert ref_id in hints
    assert hints[ref_id]["cluster_top1"] == "c_001"
    assert clusters[0]["cluster_id"] == "c_001"


@pytest.mark.asyncio
async def test_direct_geometry_boundary_supports_sync_schedule_and_async_purge() -> None:
    service = _SyncGeometryService()
    boundary = DirectGeometryBoundary(service)

    await boundary.schedule_recompute("conv-1", reason="memory_apply")
    await boundary.purge_conversation("conv-1")

    assert service.scheduled == [("conv-1", "memory_apply")]
    assert service.purged == ["conv-1"]
