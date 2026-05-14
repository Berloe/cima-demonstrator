from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from cima_demo.geometry.projector import GeometryReadModelProjector


class _FakeRelDB:
    def __init__(self) -> None:
        self.saved_item: dict | None = None
        self.saved_cluster: dict | None = None
        self.deleted: str | None = None

    async def save_geometry_item_state(self, item_state_json: dict) -> None:
        self.saved_item = item_state_json

    async def save_geometry_cluster_state(self, cluster_state_json: dict) -> None:
        self.saved_cluster = cluster_state_json

    async def delete_geometry_conversation(self, conversation_id: str) -> None:
        self.deleted = conversation_id


@pytest.mark.asyncio
async def test_geometry_projector_materialises_item_and_cluster_state() -> None:
    db = _FakeRelDB()
    projector = GeometryReadModelProjector(db)  # type: ignore[arg-type]
    run_id = uuid4()
    now = datetime.now(UTC)
    await projector.apply_item_state(
        "conv-1",
        {
            "run_id": str(run_id),
            "algo_version": "geom_v1.0",
            "ref_kind": "local_citem",
            "ref_id": str(uuid4()),
            "cluster_top1": "c_001",
            "cluster_top2": None,
            "w1": 0.8,
            "w2": None,
            "margin": 0.8,
            "is_core": True,
            "is_bridge_candidate": False,
            "centrality": 0.9,
            "updated_at": now.isoformat(),
        },
    )
    assert db.saved_item is not None
    assert db.saved_item["conversation_id"] == "conv-1"

    await projector.apply_cluster_state(
        "conv-1",
        {
            "run_id": str(run_id),
            "algo_version": "geom_v1.0",
            "cluster_id": "c_001",
            "mass": 0.5,
            "medoid": {"ref_kind": "local_citem", "ref_id": str(uuid4())},
            "summary_id": None,
            "updated_at": now.isoformat(),
        },
    )
    assert db.saved_cluster is not None
    assert db.saved_cluster["cluster_id"] == "c_001"

    await projector.purge_conversation("conv-1")
    assert db.deleted == "conv-1"
