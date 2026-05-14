from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from cima_demo.geometry.worker import GeometryCommandProcessor
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType, GeometryPurgeData, Producer
from cima_demo.witness_backend.topic_catalog import TOPICS


class _FakeGeometryService:
    def __init__(self) -> None:
        self.purged: list[str] = []
        self.item_rows = [
            {
                "ref_kind": "local_citem",
                "ref_id": str(uuid4()),
            },
            {
                "ref_kind": "local_summary",
                "ref_id": str(uuid4()),
            },
        ]
        self.cluster_rows = [
            {"cluster_id": "c_001"},
            {"cluster_id": "c_002"},
        ]

    async def load_all_item_hints(self, *, conversation_id: str):
        return list(self.item_rows)

    async def get_cluster_hints(self, *, conversation_id: str):
        return list(self.cluster_rows)

    async def purge_conversation(self, conversation_id: str):
        self.purged.append(conversation_id)


@pytest.mark.asyncio
async def test_geometry_purge_emits_per_key_tombstones_instead_of_wildcard() -> None:
    service = _FakeGeometryService()
    processor = GeometryCommandProcessor(service)  # type: ignore[arg-type]
    envelope = CloudEventEnvelope(
        type=EventType.GEOM_PURGE,
        source=Producer.CIMA_API,
        subject="conv-1",
        dataschema="schemas/cima.geom.purge.requested.v1.json",
        time=datetime.now(UTC),
        data=GeometryPurgeData(delete_run_id=str(uuid4())).model_dump(mode="json"),
    )

    outputs = await processor.handle(envelope)

    assert service.purged == ["conv-1"]
    assert all(payload is None for _, _, payload in outputs)
    keys = [key for _, key, _ in outputs]
    assert "conv-1|*" not in keys
    assert any(topic == TOPICS.geom_item_state for topic, _, _ in outputs)
    assert any(topic == TOPICS.geom_cluster_state for topic, _, _ in outputs)
    assert len(outputs) == 4
