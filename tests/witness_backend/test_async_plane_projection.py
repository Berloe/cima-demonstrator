from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from cima_demo.geometry.boundary import GeometryReadModelService
from cima_demo.geometry.projector import GeometryReadModelProjector
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    EventType,
    GeometryClusterMedoid,
    GeometryClusterStateData,
    GeometryItemStateData,
    GeometryRunCompletedData,
    GeometryRunMetrics,
    Producer,
)
from cima_demo.witness_backend.projector_consumer import GeometryReadModelProjectorConsumer
from cima_demo.witness_backend.topic_catalog import TOPICS, geom_cluster_state_key, geom_item_state_key


class _FakeReadModelDB:
    def __init__(self) -> None:
        self.read_model_runs: list[dict] = []
        self.saved_items: list[dict] = []
        self.saved_clusters: list[dict] = []
        self.deleted_items: list[tuple[str, str, str]] = []
        self.deleted_clusters: list[tuple[str, str]] = []
        self.deleted_conversations: list[str] = []
        self.effects: set[tuple[str, str, str]] = set()

    async def save_geometry_read_model_run(self, run_json: dict) -> None:
        self.read_model_runs.append(run_json)

    async def save_geometry_read_model_item_state(self, item_state_json: dict) -> None:
        self.saved_items.append(item_state_json)

    async def save_geometry_read_model_cluster_state(self, cluster_state_json: dict) -> None:
        self.saved_clusters.append(cluster_state_json)

    async def delete_geometry_read_model_item_state(self, conversation_id: str, ref_kind: str, ref_id: str) -> None:
        self.deleted_items.append((conversation_id, ref_kind, ref_id))

    async def delete_geometry_read_model_cluster_state(self, conversation_id: str, cluster_id: str) -> None:
        self.deleted_clusters.append((conversation_id, cluster_id))

    async def delete_geometry_read_model_conversation(self, conversation_id: str) -> None:
        self.deleted_conversations.append(conversation_id)

    async def begin_consumer_effect(self, *, consumer_name: str, event_id: str, effect_key: str) -> bool:
        key = (consumer_name, event_id, effect_key)
        if key in self.effects:
            return False
        self.effects.add(key)
        return True

    async def complete_consumer_effect(self, *, consumer_name: str, event_id: str, effect_key: str, details_json: dict | None = None) -> None:
        return None


class _DualGeometryDB:
    def __init__(self) -> None:
        self.rm_items = [{"conversation_id": "conv-1", "ref_id": str(uuid4()), "cluster_top1": "c_001"}]
        self.legacy_items = [{"conversation_id": "conv-1", "ref_id": str(uuid4()), "cluster_top1": "legacy"}]
        self.rm_clusters = [{"conversation_id": "conv-1", "cluster_id": "c_001"}]
        self.legacy_clusters = [{"conversation_id": "conv-1", "cluster_id": "legacy"}]

    async def load_geometry_read_model_item_states(self, conversation_id: str, ref_ids: list[str] | None = None):
        return list(self.rm_items)

    async def load_geometry_item_states(self, conversation_id: str, ref_ids: list[str] | None = None):
        return list(self.legacy_items)

    async def load_geometry_read_model_cluster_states(self, conversation_id: str):
        return list(self.rm_clusters)

    async def load_geometry_cluster_states(self, conversation_id: str):
        return list(self.legacy_clusters)


@pytest.mark.asyncio
async def test_geometry_read_model_service_prefers_cima_rm_over_geom_internal() -> None:
    db = _DualGeometryDB()
    service = GeometryReadModelService(db)

    items = await service.load_all_item_hints(conversation_id="conv-1")
    clusters = await service.get_cluster_hints(conversation_id="conv-1")

    assert items[0]["cluster_top1"] == "c_001"
    assert clusters[0]["cluster_id"] == "c_001"


@pytest.mark.asyncio
async def test_projector_consumer_materialises_run_state_and_tombstones_idempotently() -> None:
    db = _FakeReadModelDB()
    projector = GeometryReadModelProjector(db)
    ledger = ConsumerEffectLedger(db)
    consumer = GeometryReadModelProjectorConsumer(projector=projector, ledger=ledger)

    run_env = CloudEventEnvelope(
        type=EventType.GEOM_RUN_COMPLETED,
        source=Producer.CIMA_GEOMETRY,
        subject="conv-1",
        dataschema="schemas/cima.geom.run.completed.v1.json",
        data=GeometryRunCompletedData(
            run_id=uuid4(),
            algo_version="geom_v1.0",
            universe_hash="u:1",
            params={"k_used": 2, "temp": 0.7, "core_q": 0.2, "bridge_percentile": 90},
            metrics=GeometryRunMetrics(n_vectors=10, core_size=2, bridge_count=1, core_mass_frac=0.5),
        ).model_dump(mode="json"),
    )
    item_ref = uuid4()
    item_env = CloudEventEnvelope(
        type=EventType.GEOM_ITEM_STATE,
        source=Producer.CIMA_GEOMETRY,
        subject="conv-1",
        dataschema="schemas/cima.geom.item_state.v1.json",
        data=GeometryItemStateData(
            run_id=uuid4(),
            algo_version="geom_v1.0",
            ref_kind="local_citem",
            ref_id=item_ref,
            cluster_top1="c_001",
            cluster_top2=None,
            w1=0.9,
            w2=None,
            margin=0.9,
            is_core=True,
            is_bridge_candidate=False,
            centrality=0.9,
            updated_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )
    cluster_env = CloudEventEnvelope(
        type=EventType.GEOM_CLUSTER_STATE,
        source=Producer.CIMA_GEOMETRY,
        subject="conv-1",
        dataschema="schemas/cima.geom.cluster_state.v1.json",
        data=GeometryClusterStateData(
            run_id=uuid4(),
            algo_version="geom_v1.0",
            cluster_id="c_001",
            mass=0.7,
            medoid=GeometryClusterMedoid(ref_kind="local_citem", ref_id=uuid4()),
            summary_id=None,
            updated_at=datetime.now(UTC),
        ).model_dump(mode="json"),
    )
    delete_env = CloudEventEnvelope(
        type=EventType.CONVERSATION_HARD_DELETE_REQUESTED,
        source=Producer.CIMA_WORKER,
        subject="conv-1",
        dataschema="schemas/cima.conversation.hard_delete.requested.v1.json",
        data={"delete_run_id": str(uuid4()), "mode": "HARD", "reason": "USER_REQUEST"},
    )

    await consumer.handle(topic=TOPICS.geom_run, message_key="conv-1", payload_json=run_env.model_dump(mode="json"))
    await consumer.handle(topic=TOPICS.geom_item_state, message_key=geom_item_state_key("conv-1", "local_citem", str(item_ref)), payload_json=item_env.model_dump(mode="json"))
    await consumer.handle(topic=TOPICS.geom_cluster_state, message_key=geom_cluster_state_key("conv-1", "c_001"), payload_json=cluster_env.model_dump(mode="json"))
    await consumer.handle(topic=TOPICS.geom_item_state, message_key=geom_item_state_key("conv-1", "local_citem", str(item_ref)), payload_json=None)
    await consumer.handle(topic=TOPICS.geom_cluster_state, message_key=geom_cluster_state_key("conv-1", "c_001"), payload_json=None)
    await consumer.handle(topic=TOPICS.conversation_events, message_key="conv-1", payload_json=delete_env.model_dump(mode="json"))
    # duplicate delivery should be ignored by ledger
    await consumer.handle(topic=TOPICS.conversation_events, message_key="conv-1", payload_json=delete_env.model_dump(mode="json"))

    assert len(db.read_model_runs) == 1
    assert len(db.saved_items) == 1
    assert len(db.saved_clusters) == 1
    assert db.deleted_items == [("conv-1", "local_citem", str(item_ref))]
    assert db.deleted_clusters == [("conv-1", "c_001")]
    assert db.deleted_conversations == ["conv-1"]
