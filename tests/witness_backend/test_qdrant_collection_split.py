from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from cima_demo.domain.entities import CItem
from cima_demo.domain.value_objects import CItemFilter
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.qdrant import QdrantCItemAdapter
from cima_demo.witness_backend.ephemeral import EphemeralVectorRegistry
from cima_demo.witness_backend.ephemeral_runtime import EphemeralRuntimeMirror
from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane


class _FakeEmbed:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text or "")), 0.5]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text or "")), 0.5] for text in texts]


class _FakeWitnessDB:
    def __init__(self) -> None:
        self.local_rows: dict[str, dict[str, Any]] = {}
        self.global_rows: dict[str, dict[str, Any]] = {}

    async def list_local_citem_records(self, conversation_id: str, *, citem_ids: list[str] | None = None) -> list[dict[str, Any]]:
        rows = [dict(row) for row in self.local_rows.values() if row.get("conversation_id") == conversation_id]
        if citem_ids is not None:
            wanted = {str(v) for v in citem_ids}
            rows = [row for row in rows if str(row.get("local_citem_id")) in wanted]
        return rows

    async def list_global_citem_records(
        self,
        *,
        global_citem_ids: list[str] | None = None,
        semantic_identity_ids: list[str] | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = [dict(row) for row in self.global_rows.values()]
        if global_citem_ids is not None:
            wanted = {str(v) for v in global_citem_ids}
            rows = [row for row in rows if str(row.get("global_citem_id")) in wanted]
        if semantic_identity_ids is not None:
            wanted = {str(v) for v in semantic_identity_ids}
            rows = [row for row in rows if str(row.get("semantic_identity_id")) in wanted]
        if origin_conversation_id is not None:
            rows = [row for row in rows if row.get("origin_conversation_id") == origin_conversation_id]
        return rows


@dataclass
class _FakePoint:
    id: str
    payload: dict[str, Any] | None = None
    score: float = 0.0
    vector: dict[str, list[float]] | None = None


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, _FakePoint]] = {}
        self.upsert_calls: list[str] = []
        self.delete_calls: list[tuple[str, Any]] = []
        self.query_calls: list[str] = []

    async def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        bucket = self.collections.setdefault(collection_name, {})
        self.upsert_calls.append(collection_name)
        for point in points:
            bucket[str(point.id)] = _FakePoint(id=str(point.id), payload=dict(point.payload), vector={"content": list(point.vector["content"])})

    async def retrieve(self, *, collection_name: str, ids: list[str], with_payload: bool, with_vectors: Any) -> list[_FakePoint]:
        bucket = self.collections.setdefault(collection_name, {})
        rows: list[_FakePoint] = []
        for cid in ids:
            if cid in bucket:
                row = bucket[cid]
                payload = row.payload if with_payload else None
                vector = row.vector if with_vectors else None
                rows.append(_FakePoint(id=row.id, payload=payload, vector=vector))
        return rows

    async def query_points(self, *, collection_name: str, **_: Any) -> Any:
        self.query_calls.append(collection_name)
        bucket = self.collections.setdefault(collection_name, {})
        points = [
            _FakePoint(id=row.id, payload=row.payload, score=1.0, vector=row.vector)
            for row in bucket.values()
        ]
        return SimpleNamespace(points=points)

    async def scroll(self, *, collection_name: str, scroll_filter: Any, limit: int, offset: Any = None, with_payload: bool = True, with_vectors: bool = False) -> tuple[list[_FakePoint], None]:
        bucket = self.collections.setdefault(collection_name, {})
        conversation_id = None
        content_hash = None
        expires_lt = None
        for cond in getattr(scroll_filter, "must", []):
            key = getattr(cond, "key", None)
            match = getattr(cond, "match", None)
            range_ = getattr(cond, "range", None)
            if key == "conversation_id" and match is not None:
                conversation_id = getattr(match, "value", None)
            if key == "content_hash" and match is not None:
                content_hash = getattr(match, "value", None)
            if key == "expires_at" and range_ is not None:
                expires_lt = getattr(range_, "lt", None)
        rows: list[_FakePoint] = []
        for row in bucket.values():
            payload = row.payload or {}
            if conversation_id is not None and payload.get("conversation_id") != conversation_id:
                continue
            if content_hash is not None and payload.get("content_hash") != content_hash:
                continue
            if expires_lt is not None and not (payload.get("expires_at") and payload.get("expires_at") < expires_lt):
                continue
            rows.append(_FakePoint(id=row.id, payload=row.payload if with_payload else None, vector=row.vector if with_vectors else None))
        return rows[:limit], None

    async def delete(self, *, collection_name: str, points_selector: Any) -> None:
        self.delete_calls.append((collection_name, points_selector))

    async def set_payload(self, *, collection_name: str, payload: dict[str, Any], points: list[str]) -> None:
        bucket = self.collections.setdefault(collection_name, {})
        for point_id in points:
            bucket[point_id].payload = {**(bucket[point_id].payload or {}), **payload}

    async def get_collection(self, collection_name: str) -> Any:
        if collection_name not in self.collections:
            raise RuntimeError("missing collection")
        return SimpleNamespace(name=collection_name)

    async def get_collections(self) -> Any:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    async def create_collection(self, *, collection_name: str, **_: Any) -> None:
        self.collections.setdefault(collection_name, {})

    async def create_payload_index(self, **_: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_qdrant_adapter_routes_local_and_global_items_to_different_collections() -> None:
    client = _FakeQdrantClient()
    adapter = QdrantCItemAdapter(
        client=client, embedding_port=_FakeEmbed(), collection="local-c", global_collection="global-c"
    )
    local = CItem(conversation_id="conv-1", content="local", item_type="FACT", scope="episodic")
    global_item = CItem(conversation_id="conv-1", content="global", item_type="FACT", scope="global")

    await adapter.save(local)
    await adapter.save(global_item)

    assert client.upsert_calls == ["local-c", "global-c"]
    fetched_local = await adapter.fetch(local.citem_id)
    fetched_global = await adapter.fetch(global_item.citem_id)
    assert fetched_local.scope == "episodic"
    assert fetched_global.scope == "global"


@pytest.mark.asyncio
async def test_qdrant_adapter_search_uses_scope_specific_collection() -> None:
    client = _FakeQdrantClient()
    adapter = QdrantCItemAdapter(
        client=client, embedding_port=_FakeEmbed(), collection="local-c", global_collection="global-c"
    )
    await adapter.save(CItem(conversation_id="conv-1", content="local", item_type="FACT", scope="episodic"))
    await adapter.save(CItem(conversation_id="conv-1", content="global", item_type="FACT", scope="global"))

    local_results = await adapter.search("query", CItemFilter(scope="episodic", conversation_id="conv-1"), top_k=3)
    global_results = await adapter.search("query", CItemFilter(scope="global", conversation_id="conv-1"), top_k=3)

    assert local_results and local_results[0].citem.scope == "episodic"
    assert global_results and global_results[0].citem.scope == "global"
    assert client.query_calls[0] == "local-c"
    assert client.query_calls[2] == "global-c"


@pytest.mark.asyncio
async def test_witness_plane_deletes_only_local_scoped_collections_and_sweeps_ephemeral() -> None:
    client = _FakeQdrantClient()
    catalog = QdrantCollectionCatalog(
        local_citems="local-c",
        local_summaries="local-s",
        chunks="chunks",
        global_citems="global-c",
        global_summaries="global-s",
        ephemeral="ephemeral",
    )
    plane = QdrantWitnessPlane(client=client, catalog=catalog, dense_dim=8)

    await plane.delete_by_conversation("conv-1")
    touched = [name for name, _selector in client.delete_calls]
    assert touched == ["local-c", "local-s", "chunks", "ephemeral"]

    client.delete_calls.clear()
    await plane.sweep_ephemeral_expired()
    assert client.delete_calls[0][0] == "ephemeral"


@pytest.mark.asyncio
async def test_qdrant_adapter_fetch_prefers_witness_local_record_over_payload() -> None:
    client = _FakeQdrantClient()
    db = _FakeWitnessDB()
    adapter = QdrantCItemAdapter(
        client=client,
        embedding_port=_FakeEmbed(),
        collection="local-c",
        global_collection="global-c",
        rel_db=db,
    )
    item = CItem(conversation_id="conv-1", content="legacy local payload", item_type="FACT", scope="episodic")
    await adapter.save(item)
    db.local_rows[item.citem_id] = {
        "local_citem_id": item.citem_id,
        "conversation_id": "conv-1",
        "type": "DECISION",
        "text": "witness local decision",
        "meta_json": {"speaker": "user", "source_kind": "chat_user", "confidence": 0.88},
        "provenance_json": {"dependency_ids": ["dep-1"]},
        "validity": "accepted",
        "salience": 0.97,
        "created_at": "2026-04-30T09:00:00+00:00",
    }

    fetched = await adapter.fetch(item.citem_id)

    assert fetched.content == "witness local decision"
    assert fetched.item_type == "DECISION"
    assert fetched.scope == "episodic"
    assert fetched.actor == "user"
    assert fetched.dependency_ids == ["dep-1"]


@pytest.mark.asyncio
async def test_qdrant_adapter_search_prefers_witness_rows_for_local_and_global_results() -> None:
    client = _FakeQdrantClient()
    db = _FakeWitnessDB()
    adapter = QdrantCItemAdapter(
        client=client,
        embedding_port=_FakeEmbed(),
        collection="local-c",
        global_collection="global-c",
        rel_db=db,
    )
    local = CItem(conversation_id="conv-1", content="legacy local", item_type="FACT", scope="episodic")
    global_item = CItem(conversation_id="conv-1", content="legacy global", item_type="FACT", scope="global")
    await adapter.save(local)
    await adapter.save(global_item)
    db.local_rows[local.citem_id] = {
        "local_citem_id": local.citem_id,
        "conversation_id": "conv-1",
        "type": "CONSTRAINT",
        "text": "witness local constraint",
        "meta_json": {},
        "provenance_json": {},
        "validity": "accepted",
        "salience": 0.99,
        "created_at": "2026-04-30T09:00:00+00:00",
    }
    db.global_rows[global_item.citem_id] = {
        "global_citem_id": global_item.citem_id,
        "semantic_identity_id": "11111111-1111-4111-8111-111111111111",
        "origin_conversation_id": "conv-1",
        "promotion_origin_local_citem_id": local.citem_id,
        "type": "DEFINITION",
        "text": "witness global definition",
        "meta_json": {"speaker": "agent"},
        "provenance_json": {},
        "validity": "accepted",
        "salience": 0.77,
        "created_at": "2026-04-30T09:05:00+00:00",
    }

    local_results = await adapter.search("query", CItemFilter(scope="episodic", conversation_id="conv-1"), top_k=3)
    global_results = await adapter.search("query", CItemFilter(scope="global", conversation_id="conv-1"), top_k=3)

    assert local_results[0].citem.content == "witness local constraint"
    assert local_results[0].citem.item_type == "CONSTRAINT"
    assert global_results[0].citem.content == "witness global definition"
    assert global_results[0].citem.scope == "global"


@pytest.mark.asyncio
async def test_qdrant_adapter_fetch_by_conversation_hydrates_witness_rows() -> None:
    client = _FakeQdrantClient()
    db = _FakeWitnessDB()
    adapter = QdrantCItemAdapter(
        client=client,
        embedding_port=_FakeEmbed(),
        collection="local-c",
        global_collection="global-c",
        rel_db=db,
    )
    item = CItem(conversation_id="conv-1", content="legacy fetch_by_conversation", item_type="FACT", scope="episodic")
    await adapter.save(item)
    db.local_rows[item.citem_id] = {
        "local_citem_id": item.citem_id,
        "conversation_id": "conv-1",
        "type": "PLAN_STEP",
        "text": "witness plan step",
        "meta_json": {"speaker": "agent"},
        "provenance_json": {},
        "validity": "unknown",
        "salience": 0.68,
        "created_at": "2026-04-30T09:10:00+00:00",
    }

    rows = await adapter.fetch_by_conversation("conv-1", scope_status="active")

    assert rows[0].content == "witness plan step"
    assert rows[0].item_type == "PLAN_STEP"


@pytest.mark.asyncio
async def test_qdrant_adapter_search_includes_ephemeral_collection_for_episodic_scope() -> None:
    client = _FakeQdrantClient()
    adapter = QdrantCItemAdapter(
        client=client,
        embedding_port=_FakeEmbed(),
        collection="local-c",
        global_collection="global-c",
        ephemeral_collection="ephemeral",
    )
    client.collections.setdefault("ephemeral", {})["ephemeral-1"] = _FakePoint(
        id="ephemeral-1",
        payload={
            "citem_id": "logical-summary-1",
            "conversation_id": "conv-1",
            "content": "ephemeral summary context",
            "item_type": "SUMMARY",
            "scope": "episodic",
            "scope_status": "active",
            "importance": 0.6,
            "confidence": 1.0,
            "validation_label": "accepted",
            "conflict_status": "none",
            "phase_ingested": "IDLE",
            "actor": "agent",
            "motivation": "ephemeral:local_summary",
            "created_at_unix": 0.0,
            "token_count": 3,
            "dependency_ids": [],
            "w_scope": "local",
            "ref_id": "logical-summary-1",
            "origin_ref_kind": "local_summary",
            "origin_ref_id": "logical-summary-1",
            "vector_state": "EPHEMERAL",
            "eligible_for_geometry": False,
        },
        vector={"content": [1.0, 0.5]},
    )

    episodic = await adapter.search("query", CItemFilter(scope="episodic", conversation_id="conv-1"), top_k=3)
    global_results = await adapter.search("query", CItemFilter(scope="global", conversation_id="conv-1"), top_k=3)

    assert episodic
    assert episodic[0].citem.citem_id == "logical-summary-1"
    assert "ephemeral" in client.query_calls
    assert all(result.citem.citem_id != "logical-summary-1" for result in global_results)


@pytest.mark.asyncio
async def test_ephemeral_runtime_mirror_upserts_points_and_persists_geometry_ineligible_leases() -> None:
    client = _FakeQdrantClient()
    catalog = QdrantCollectionCatalog(
        local_citems="local-c",
        local_summaries="local-s",
        chunks="chunks",
        global_citems="global-c",
        global_summaries="global-s",
        ephemeral="ephemeral",
    )
    plane = QdrantWitnessPlane(client=client, catalog=catalog, dense_dim=8)
    db = InMemoryDemoDB()
    mirror = EphemeralRuntimeMirror(
        plane=plane,
        embedder=_FakeEmbed(),
        registry=EphemeralVectorRegistry(db),
        ttl_seconds=600,
        max_items=4,
        embedding_model_id="tei-test",
        embedding_schema_version=2,
    )

    stats = await mirror.mirror_context_items(
        conversation_id="conv-1",
        items=[
            {
                "marker": "S1",
                "ref_kind": "citem",
                "ref_id": "citem-1",
                "content": "temporary decision context",
                "item_type": "DECISION",
                "item_resolution_scope": "local",
            },
            {
                "marker": "S2",
                "ref_kind": "summary",
                "ref_id": "summary-1",
                "content": "temporary cluster summary",
                "item_type": "SUMMARY",
                "summary_scope": "global",
            },
        ],
    )

    assert stats.accepted_items == 2
    assert stats.upserted_points == 2
    assert stats.registered_leases == 2
    assert len(db.ephemeral_vector_records) == 2
    assert all(row["eligible_for_geometry"] is False for row in db.ephemeral_vector_records.values())
    assert all(row["lifecycle_state"] == "ACTIVE" for row in db.ephemeral_vector_records.values())
    assert all(row["qdrant_collection"] == "ephemeral" for row in db.ephemeral_vector_records.values())
    assert set(client.collections.get("ephemeral", {}).keys()) == set(db.ephemeral_vector_records.keys())
    assert all((point.payload or {}).get("eligible_for_geometry") is False for point in client.collections["ephemeral"].values())


@pytest.mark.asyncio
async def test_ephemeral_runtime_mirror_skips_non_active_conversation() -> None:
    client = _FakeQdrantClient()
    catalog = QdrantCollectionCatalog(
        local_citems="local-c",
        local_summaries="local-s",
        chunks="chunks",
        global_citems="global-c",
        global_summaries="global-s",
        ephemeral="ephemeral",
    )
    plane = QdrantWitnessPlane(client=client, catalog=catalog, dense_dim=8)
    db = InMemoryDemoDB()
    await db.create_conversation("conv-1")
    db.conversations["conv-1"]["status"] = "DELETING"
    mirror = EphemeralRuntimeMirror(
        plane=plane,
        embedder=_FakeEmbed(),
        registry=EphemeralVectorRegistry(db),
        ttl_seconds=600,
        max_items=4,
        embedding_model_id="tei-test",
        embedding_schema_version=2,
        conversation_reader=db,
    )

    stats = await mirror.mirror_context_items(
        conversation_id="conv-1",
        items=[
            {
                "marker": "S1",
                "ref_kind": "citem",
                "ref_id": "citem-1",
                "content": "temporary decision context",
                "item_type": "DECISION",
                "item_resolution_scope": "local",
            }
        ],
    )

    assert stats.accepted_items == 0
    assert stats.upserted_points == 0
    assert stats.registered_leases == 0
    assert db.ephemeral_vector_records == {}
    assert client.collections.get("ephemeral", {}) == {}
