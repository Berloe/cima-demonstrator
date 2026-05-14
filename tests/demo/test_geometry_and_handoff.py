from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cima_demo.demo.context.service import DemoContextService
from cima_demo.demo.handoff.service import DemoHandoffService
from cima_demo.demo.runtime.journal import DemoRunJournal
from cima_demo.domain.entities import CItem, ContextView, SummaryNode, TaskMemory
from cima_demo.domain.value_objects import ContextBudget
from cima_demo.geometry import DemoGeometryService


class FakeGeometryStore:
    def __init__(self) -> None:
        self.items = [
            CItem(citem_id="a", conversation_id="conv", content="authorization decisions", dependency_ids=["b"]),
            CItem(citem_id="b", conversation_id="conv", content="rbac policy matrix"),
            CItem(citem_id="c", conversation_id="conv", content="librechat integration contract", dependency_ids=["d"]),
            CItem(citem_id="d", conversation_id="conv", content="custom endpoint compatibility"),
        ]
        self.vectors = {
            "a": [1.0, 0.0],
            "b": [0.95, 0.05],
            "c": [-1.0, 0.0],
            "d": [-0.95, 0.05],
        }
        self.updated: list[tuple[str, str, Any]] = []

    async def fetch_by_conversation(self, conversation_id: str, scope_status: str | None = None):
        return list(self.items)

    async def fetch_dense_vectors(self, citem_ids: list[str]):
        return {cid: self.vectors[cid] for cid in citem_ids if cid in self.vectors}

    async def update_field(self, citem_id: str, field: str, value: Any):
        self.updated.append((citem_id, field, value))


class FakeGeometryDB:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.item_states: list[dict[str, Any]] = []
        self.cluster_states: list[dict[str, Any]] = []

    async def fetch_pyramid_tops(self, conversation_id: str, limit: int | None = None):
        return [SummaryNode(node_id="sum-1", conversation_id=conversation_id, level=2, content="authorization summary", origin_citem_ids=["a", "b"])]

    async def save_geometry_run(self, run_json: dict[str, Any]) -> None:
        self.runs.append(run_json)

    async def save_geometry_item_state(self, item_state_json: dict[str, Any]) -> None:
        self.item_states = [row for row in self.item_states if not (row["conversation_id"] == item_state_json["conversation_id"] and row["ref_id"] == item_state_json["ref_id"])]
        self.item_states.append(item_state_json)

    async def save_geometry_cluster_state(self, cluster_state_json: dict[str, Any]) -> None:
        self.cluster_states = [row for row in self.cluster_states if not (row["conversation_id"] == cluster_state_json["conversation_id"] and row["cluster_id"] == cluster_state_json["cluster_id"])]
        self.cluster_states.append(cluster_state_json)

    async def load_geometry_item_states(self, conversation_id: str, ref_ids: list[str] | None = None):
        rows = [row for row in self.item_states if row["conversation_id"] == conversation_id]
        if ref_ids is not None:
            rows = [row for row in rows if row["ref_id"] in ref_ids]
        return rows

    async def load_geometry_cluster_states(self, conversation_id: str):
        return [row for row in self.cluster_states if row["conversation_id"] == conversation_id]

    async def delete_geometry_conversation(self, conversation_id: str) -> None:
        self.item_states = [row for row in self.item_states if row["conversation_id"] != conversation_id]
        self.cluster_states = [row for row in self.cluster_states if row["conversation_id"] != conversation_id]
        self.runs = [row for row in self.runs if row["conversation_id"] != conversation_id]


@pytest.mark.asyncio
async def test_geometry_service_recomputes_and_persists_flags() -> None:
    db = FakeGeometryDB()
    store = FakeGeometryStore()
    service = DemoGeometryService(rel_db=db, citem_store=store, k_max=2)

    report = await service.recompute(conversation_id="conv", reason="test")

    assert report.n_items == 4
    assert report.cluster_count == 2
    assert len(db.item_states) == 4
    assert len(db.cluster_states) == 2
    assert any(field == "geom_is_core" for _cid, field, _value in store.updated)

    hints = await service.get_item_hints(conversation_id="conv", ref_ids=["a", "b"])
    assert set(hints) == {"a", "b"}

    await service.purge_conversation("conv")
    assert db.item_states == []
    assert db.cluster_states == []


class FakeContextBuilder:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def build(self, **kwargs: Any) -> ContextView:
        self.last_kwargs = dict(kwargs)
        return ContextView(
            text="CTX",
            tokens_used=42,
            coverage_score=0.8,
            items=[{"marker": "S1", "ref_kind": "citem", "ref_id": "a", "content": "authorization decisions"}],
        )


class FakeContextDB:
    def __init__(self) -> None:
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.summary_nodes = [SummaryNode(node_id="sum-1", conversation_id="conv", level=2, content="master summary")]
        self.local_citem_records = {"a": {"local_citem_id": "a", "conversation_id": "conv"}}
        self.local_citem_evidence_rows = [{
            "local_citem_id": "a",
            "conversation_id": "conv",
            "source_id": "src-a",
            "source_span_id": "span-a",
            "locator_json": {"source_id": "src-a", "source_span_id": "span-a"},
        }]

    async def save_demo_context_snapshot(self, snapshot_json: dict[str, Any]) -> None:
        self.snapshots[snapshot_json["context_id"]] = snapshot_json

    async def load_demo_context_snapshot(self, context_id: str) -> dict[str, Any] | None:
        return self.snapshots.get(context_id)

    async def fetch_pyramid_tops(self, conversation_id: str, limit: int | None = None):
        return self.summary_nodes[:limit]

    async def list_local_citem_records(self, conversation_id: str, *, citem_ids: list[str] | None = None):
        rows = [dict(row) for row in self.local_citem_records.values() if row.get("conversation_id") == conversation_id]
        if citem_ids is not None:
            wanted = {str(v) for v in citem_ids}
            rows = [row for row in rows if str(row.get("local_citem_id")) in wanted]
        return rows

    async def list_local_citem_evidence(self, local_citem_id: str):
        return [dict(row) for row in self.local_citem_evidence_rows if row.get("local_citem_id") == local_citem_id]


class FakeRunJournal:
    def __init__(self) -> None:
        self.json_artifacts: dict[str, dict[str, Any]] = {}
        self.text_artifacts: dict[str, str] = {}

    async def write_json_artifact(self, *, conversation_id: str, run_id: str, relative_path: str, payload: dict[str, Any]) -> None:
        self.json_artifacts[relative_path] = payload

    async def write_text_artifact(self, *, conversation_id: str, run_id: str, relative_path: str, text: str) -> None:
        self.text_artifacts[relative_path] = text


class FakeGeometryHints:
    def __init__(self) -> None:
        self.scheduled: list[tuple[str, str]] = []

    async def get_item_hints(self, *, conversation_id: str, ref_ids: list[str]):
        return {"a": {"cluster_top1": "c_001", "label": "authorization", "is_core": True, "is_bridge_candidate": False}}

    def schedule_recompute(self, conversation_id: str, *, reason: str = "context_snapshot") -> None:
        self.scheduled.append((conversation_id, reason))


@pytest.mark.asyncio
async def test_context_service_enriches_with_geometry_hints() -> None:
    db = FakeContextDB()
    runs = FakeRunJournal()
    base_builder = FakeContextBuilder()
    service = DemoContextService(
        base_builder=base_builder,
        memory_service=object(),  # not used here
        rel_db=db,
        run_journal=runs,
        geometry_service=FakeGeometryHints(),
    )
    token = service.bind_run(run_id="run-1", conversation_id="conv", turn_id="turn-1", query_text="what happened")
    budget = ContextBudget(max_tokens=256, overhead_tokens=32)
    task_memory = TaskMemory(conversation_id="conv")

    try:
        context = await service.build(
            phase="RECALL",
            task_memory=task_memory,
            plan=None,
            query="what happened",
            conversation_id="conv",
            budget=budget,
        )
    finally:
        service.reset_run(token)

    assert base_builder.last_kwargs is not None
    assert base_builder.last_kwargs["disable_geometric_expand"] is True
    assert context.items[0]["geom_role"] == "CORE"
    assert context.items[0]["geom_cluster"] == "c_001"
    assert any(name.startswith("context_snapshot_") for name in runs.json_artifacts)
    assert any(name.startswith("context_pack_") for name in runs.text_artifacts)


class FakeHandoffRunJournal:
    def __init__(self, bundle: Any) -> None:
        self.bundle = bundle
        self.artifacts: dict[str, dict[str, Any]] = {}

    async def load_bundle(self, run_id: str):
        if run_id != self.bundle.manifest["run_id"]:
            return None
        return self.bundle

    async def write_json_artifact(self, *, conversation_id: str, run_id: str, relative_path: str, payload: dict[str, Any]) -> None:
        self.artifacts[relative_path] = payload


@dataclass
class FakeRunBundle:
    manifest: dict[str, Any]
    phases: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]


class FakeHandoffDB:
    def __init__(self) -> None:
        self.context_snapshots = {
            "run-1": {
                "context_id": "ctx-1",
                "run_id": "run-1",
                "conversation_id": "conv-1",
                "turn_id": "turn-1",
                "query_text": "query",
                "phase": "RECALL",
                "context_text": "ctx",
                "markers": ["S1", "P1"],
                "items": [
                    {"marker": "S1", "ref_kind": "citem", "ref_id": "c1", "content": "fact one"},
                    {"marker": "S2", "ref_kind": "citem", "ref_id": "c2", "content": "fact two"},
                    {"marker": "P1", "ref_kind": "summary", "ref_id": "s1", "content": "summary"},
                ],
                "budget": {},
            }
        }
        self.manifests: dict[str, dict[str, Any]] = {}
        self.validations: dict[str, dict[str, Any]] = {}
        self.restores: dict[str, dict[str, Any]] = {}
        self.created_conversations: set[str] = set()
        self.saved_sources: list[dict[str, Any]] = []
        self.saved_spans: list[dict[str, Any]] = []
        self.saved_edges: list[dict[str, Any]] = []
        self.saved_summary_resolutions: list[dict[str, Any]] = []
        self.saved_summaries: list[SummaryNode] = []
        self.saved_task_memory: TaskMemory | None = None
        self.saved_plan: Any = None
        self.sources = {
            "src-1": {"source_id": "src-1", "conversation_id": "conv-1", "source_kind": "chat_user", "role": "user", "origin_ref": None, "display_text": "u", "process_text": "u", "metadata": {}},
        }
        self.spans = {
            "sp-1": {"span_id": "sp-1", "source_id": "src-1", "conversation_id": "conv-1", "span_kind": "full_text", "char_start": 0, "char_end": 10, "locator": {}, "preview_text": "u"},
        }
        self.local_citem_records: dict[str, dict[str, Any]] = {}
        self.local_citem_evidence_rows: list[dict[str, Any]] = []
        self.global_citem_records: dict[str, dict[str, Any]] = {}
        self.global_citem_evidence_rows: list[dict[str, Any]] = []
        self.local_summary_records: dict[str, dict[str, Any]] = {}
        self.local_summary_origin_rows: list[dict[str, Any]] = []
        self.global_summary_records: dict[str, dict[str, Any]] = {}
        self.global_summary_origin_rows: list[dict[str, Any]] = []

    async def load_latest_demo_context_snapshot_for_run(self, run_id: str):
        return self.context_snapshots.get(run_id)

    async def load_summaries(self, conversation_id: str, level: int | None = None):
        return [SummaryNode(node_id="s1", conversation_id=conversation_id, level=2, content="summary content", origin_citem_ids=["c1", "c2"])]

    async def load_demo_summary_resolutions(self, conversation_id: str, summary_ids: list[str] | None = None):
        return [{"summary_id": "s1", "conversation_id": conversation_id, "summary_text": "summary content", "origin_citem_ids": ["c1", "c2"], "metadata": {}}]

    async def load_demo_lineage_edges(self, conversation_id: str, *, src_kind: str | None = None, src_ids: list[str] | None = None, dst_kind: str | None = None, dst_ids: list[str] | None = None):
        rows = [
            {"edge_id": "e1", "conversation_id": conversation_id, "src_kind": "citem", "src_id": "c1", "dst_kind": "source", "dst_id": "src-1", "relation": "DERIVED_FROM_SOURCE", "metadata": {}},
            {"edge_id": "e2", "conversation_id": conversation_id, "src_kind": "citem", "src_id": "c1", "dst_kind": "source_span", "dst_id": "sp-1", "relation": "DERIVED_FROM_SPAN", "metadata": {}},
            {"edge_id": "e3", "conversation_id": conversation_id, "src_kind": "citem", "src_id": "c2", "dst_kind": "source", "dst_id": "src-1", "relation": "DERIVED_FROM_SOURCE", "metadata": {}},
            {"edge_id": "e4", "conversation_id": conversation_id, "src_kind": "citem", "src_id": "c2", "dst_kind": "source_span", "dst_id": "sp-1", "relation": "DERIVED_FROM_SPAN", "metadata": {}},
            {"edge_id": "e5", "conversation_id": conversation_id, "src_kind": "summary", "src_id": "s1", "dst_kind": "citem", "dst_id": "c1", "relation": "SUMMARIZES", "metadata": {}},
        ]
        out = rows
        if src_kind is not None:
            out = [r for r in out if r["src_kind"] == src_kind]
        if src_ids is not None:
            out = [r for r in out if r["src_id"] in src_ids]
        if dst_kind is not None:
            out = [r for r in out if r["dst_kind"] == dst_kind]
        if dst_ids is not None:
            out = [r for r in out if r["dst_id"] in dst_ids]
        return out

    async def load_demo_sources(self, conversation_id: str, source_ids: list[str]):
        return [self.sources[sid] for sid in source_ids if sid in self.sources]

    async def load_demo_source_spans(self, conversation_id: str, span_ids: list[str]):
        return [self.spans[sid] for sid in span_ids if sid in self.spans]

    async def list_local_citem_records(self, conversation_id: str, *, citem_ids: list[str] | None = None):
        rows = [dict(row) for row in self.local_citem_records.values() if row.get("conversation_id") == conversation_id]
        if citem_ids is not None:
            wanted = {str(v) for v in citem_ids}
            rows = [row for row in rows if str(row.get("local_citem_id")) in wanted]
        return rows

    async def list_local_citem_evidence(self, local_citem_id: str):
        return [dict(row) for row in self.local_citem_evidence_rows if row.get("local_citem_id") == local_citem_id]

    async def list_global_citem_records(self, *, global_citem_ids: list[str] | None = None, semantic_identity_ids: list[str] | None = None, origin_conversation_id: str | None = None):
        rows = [dict(row) for row in self.global_citem_records.values()]
        if global_citem_ids is not None:
            wanted = {str(v) for v in global_citem_ids}
            rows = [row for row in rows if str(row.get("global_citem_id")) in wanted]
        if origin_conversation_id is not None:
            rows = [row for row in rows if str(row.get("origin_conversation_id")) == origin_conversation_id]
        return rows

    async def list_global_citem_evidence(self, global_citem_id: str):
        return [dict(row) for row in self.global_citem_evidence_rows if row.get("global_citem_id") == global_citem_id]

    async def list_local_summary_records(self, conversation_id: str, *, summary_ids: list[str] | None = None, level: str | None = None, cluster_id: str | None = None):
        rows = [dict(row) for row in self.local_summary_records.values() if row.get("conversation_id") == conversation_id]
        if summary_ids is not None:
            wanted = {str(v) for v in summary_ids}
            rows = [row for row in rows if str(row.get("local_summary_id")) in wanted]
        return rows

    async def list_local_summary_origins(self, local_summary_id: str):
        return [dict(row) for row in self.local_summary_origin_rows if row.get("local_summary_id") == local_summary_id]

    async def list_global_summary_records(self, *, summary_ids: list[str] | None = None, level: str | None = None, origin_conversation_id: str | None = None):
        rows = [dict(row) for row in self.global_summary_records.values()]
        if summary_ids is not None:
            wanted = {str(v) for v in summary_ids}
            rows = [row for row in rows if str(row.get("global_summary_id")) in wanted]
        if origin_conversation_id is not None:
            origin_global_ids = {str(row.get("global_citem_id")) for row in self.global_citem_records.values() if str(row.get("origin_conversation_id")) == origin_conversation_id}
            rows = [
                row for row in rows
                if any(str(origin.get("origin_id")) in origin_global_ids for origin in self.global_summary_origin_rows if origin.get("global_summary_id") == row.get("global_summary_id"))
            ]
        return rows

    async def list_global_summary_origins(self, global_summary_id: str):
        return [dict(row) for row in self.global_summary_origin_rows if row.get("global_summary_id") == global_summary_id]

    async def save_local_citem_record(self, citem_json: dict[str, Any]) -> None:
        self.local_citem_records[str(citem_json["local_citem_id"])] = dict(citem_json)

    async def save_local_citem_evidence(self, evidence_json: dict[str, Any]) -> None:
        row = dict(evidence_json)
        self.local_citem_evidence_rows = [
            existing
            for existing in self.local_citem_evidence_rows
            if not (
                existing.get("local_citem_id") == row.get("local_citem_id")
                and int(existing.get("ordinal", 0)) == int(row.get("ordinal", 0))
            )
        ]
        self.local_citem_evidence_rows.append(row)

    async def save_local_summary_record(self, summary_json: dict[str, Any]) -> None:
        self.local_summary_records[str(summary_json["local_summary_id"])] = dict(summary_json)

    async def save_local_summary_origin(self, origin_json: dict[str, Any]) -> None:
        row = dict(origin_json)
        self.local_summary_origin_rows = [
            existing
            for existing in self.local_summary_origin_rows
            if not (
                existing.get("local_summary_id") == row.get("local_summary_id")
                and existing.get("origin_kind") == row.get("origin_kind")
                and existing.get("origin_id") == row.get("origin_id")
            )
        ]
        self.local_summary_origin_rows.append(row)

    async def save_global_citem_record(self, citem_json: dict[str, Any]) -> None:
        self.global_citem_records[str(citem_json["global_citem_id"])] = dict(citem_json)

    async def save_global_citem_evidence(self, evidence_json: dict[str, Any]) -> None:
        row = dict(evidence_json)
        self.global_citem_evidence_rows = [
            existing
            for existing in self.global_citem_evidence_rows
            if not (
                existing.get("global_citem_id") == row.get("global_citem_id")
                and int(existing.get("ordinal", 0)) == int(row.get("ordinal", 0))
            )
        ]
        self.global_citem_evidence_rows.append(row)

    async def save_global_summary_record(self, summary_json: dict[str, Any]) -> None:
        self.global_summary_records[str(summary_json["global_summary_id"])] = dict(summary_json)

    async def save_global_summary_origin(self, origin_json: dict[str, Any]) -> None:
        row = dict(origin_json)
        self.global_summary_origin_rows = [
            existing
            for existing in self.global_summary_origin_rows
            if not (
                existing.get("global_summary_id") == row.get("global_summary_id")
                and existing.get("origin_kind") == row.get("origin_kind")
                and existing.get("origin_id") == row.get("origin_id")
            )
        ]
        self.global_summary_origin_rows.append(row)

    async def save_demo_handoff_manifest(self, manifest_json: dict[str, Any]) -> None:
        self.manifests[manifest_json["handoff_id"]] = manifest_json

    async def load_demo_handoff_manifest(self, handoff_id: str):
        return self.manifests.get(handoff_id)

    async def save_demo_handoff_validation(self, validation_json: dict[str, Any]) -> None:
        self.validations[validation_json["handoff_id"]] = validation_json

    async def load_demo_handoff_validation(self, handoff_id: str):
        return self.validations.get(handoff_id)

    async def save_demo_handoff_restore(self, restore_json: dict[str, Any]) -> None:
        self.restores[restore_json["restore_id"]] = restore_json

    async def get_conversation(self, conversation_id: str):
        return {"conversation_id": conversation_id} if conversation_id in self.created_conversations else None

    async def create_conversation(self, conversation_id: str) -> None:
        self.created_conversations.add(conversation_id)

    async def save_demo_source(self, source_json: dict[str, Any]) -> None:
        self.saved_sources.append(source_json)

    async def save_demo_source_span(self, span_json: dict[str, Any]) -> None:
        self.saved_spans.append(span_json)

    async def save_demo_lineage_edge(self, edge_json: dict[str, Any]) -> None:
        self.saved_edges.append(edge_json)

    async def save_summary(self, node: SummaryNode) -> None:
        self.saved_summaries.append(node)

    async def save_demo_summary_resolution(self, resolution_json: dict[str, Any]) -> None:
        self.saved_summary_resolutions.append(resolution_json)

    async def save_task_memory(self, task_memory: TaskMemory) -> None:
        self.saved_task_memory = task_memory

    async def save_plan_with_task_memory(self, plan, task_memory):
        self.saved_plan = plan
        self.saved_task_memory = task_memory


class FakeHandoffStore:
    def __init__(self) -> None:
        self.fetched = {
            "c1": CItem(citem_id="c1", conversation_id="conv-1", content="fact one", token_count=5),
            "c2": CItem(citem_id="c2", conversation_id="conv-1", content="fact two", dependency_ids=["c1"], token_count=5),
        }
        self.saved: list[CItem] = []
        self.fetch_calls: list[list[str]] = []

    async def fetch_batch(self, citem_ids: list[str]):
        self.fetch_calls.append(list(citem_ids))
        return [self.fetched[cid] for cid in citem_ids if cid in self.fetched]

    async def save(self, citem: CItem) -> None:
        self.saved.append(citem)


@pytest.mark.asyncio
async def test_handoff_service_creates_validates_and_restores_portable_manifest(tmp_path: Path) -> None:
    bundle = FakeRunBundle(
        manifest={
            "run_id": "run-1",
            "conversation_id": "conv-1",
            "turn_id": "turn-1",
            "task_memory": {"turn_count": 3, "phase": "RECALL", "active_plan_id": None},
            "assistant_reply": "partial answer",
        },
        phases=[],
        checkpoints=[],
    )
    runs = FakeHandoffRunJournal(bundle)
    db = FakeHandoffDB()
    store = FakeHandoffStore()
    service = DemoHandoffService(rel_db=db, citem_store=store, run_journal=runs, artifacts_root=tmp_path)

    manifest = await service.create_handoff(conversation_id="conv-1", source_run_id="run-1", rationale="pause and resume")
    assert manifest.context_id == "ctx-1"
    assert len(manifest.bundled_citems) == 2
    assert len(manifest.bundled_sources) == 1
    assert len(manifest.bundled_spans) == 1

    validation = await service.validate_handoff(handoff_id=manifest.handoff_id)
    assert validation.valid is True
    assert validation.evidence_coverage == pytest.approx(1.0)

    restore = await service.restore_handoff(
        handoff_id=manifest.handoff_id,
        target_conversation_id="conv-2",
        target_run_id="run-2",
    )
    assert restore.valid is True
    assert restore.diff["restored_citems"] == 2
    assert restore.diff["restored_summaries"] == 1
    assert restore.diff["restored_local_citems_witness"] == 2
    assert restore.diff["restored_local_citem_evidence"] == 2
    assert restore.diff["restored_local_summaries_witness"] == 1
    assert restore.diff["restored_local_summary_origins"] == 2
    assert len(store.saved) == 2
    assert len(db.saved_summaries) == 1
    assert len(db.local_citem_records) == 2
    assert all(row["conversation_id"] == "conv-2" for row in db.local_citem_records.values())
    assert all(row["vector_state"] == "INDEXED" for row in db.local_citem_records.values())
    assert len(db.local_citem_evidence_rows) == 2
    assert len(db.local_summary_records) == 1
    assert len(db.local_summary_origin_rows) == 2
    assert db.saved_task_memory is not None


@pytest.mark.asyncio
async def test_handoff_service_prefers_witness_records_when_available(tmp_path: Path) -> None:
    bundle = FakeRunBundle(
        manifest={
            "run_id": "run-1",
            "conversation_id": "conv-1",
            "turn_id": "turn-1",
            "task_memory": {"turn_count": 2, "phase": "RECALL", "active_plan_id": None},
            "assistant_reply": "witness reply",
        },
        phases=[],
        checkpoints=[],
    )
    runs = FakeHandoffRunJournal(bundle)
    db = FakeHandoffDB()
    db.context_snapshots["run-1"]["items"] = [
        {"marker": "S1", "ref_kind": "citem", "ref_id": "lc-1", "content": "witness local fact"},
        {"marker": "P1", "ref_kind": "summary", "ref_id": "ls-1", "content": "witness summary"},
    ]
    db.local_citem_records["lc-1"] = {
        "local_citem_id": "lc-1",
        "semantic_identity_id": "sid-1",
        "conversation_id": "conv-1",
        "type": "FACT",
        "text": "witness local fact",
        "meta_json": {},
        "provenance_json": {},
        "validity": "confirmed",
        "salience": 0.8,
        "created_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    }
    db.local_citem_evidence_rows.append({
        "local_citem_id": "lc-1",
        "source_id": "src-1",
        "chunk_id": "ch-1",
        "edu_id": "edu-1",
        "ordinal": 0,
        "locator_json": {"source_span_id": "sp-1"},
    })
    db.local_summary_records["ls-1"] = {
        "local_summary_id": "ls-1",
        "conversation_id": "conv-1",
        "level": "MASTER",
        "text": "witness summary",
        "covers_json": {},
        "updated_at": "2026-04-30T00:00:00+00:00",
    }
    db.local_summary_origin_rows.append({
        "local_summary_id": "ls-1",
        "origin_kind": "local_citem",
        "origin_id": "lc-1",
        "ordinal": 0,
    })
    store = FakeHandoffStore()
    store.fetched = {}
    service = DemoHandoffService(rel_db=db, citem_store=store, run_journal=runs, artifacts_root=tmp_path)

    manifest = await service.create_handoff(conversation_id="conv-1", source_run_id="run-1", rationale="witness first")

    assert store.fetch_calls == []
    assert [row["citem_id"] for row in manifest.bundled_citems] == ["lc-1"]
    assert [row["summary_id"] for row in manifest.bundled_summaries] == ["ls-1"]
    assert manifest.bundled_summaries[0]["metadata"]["summary_scope"] == "local"
    validation = await service.validate_handoff(handoff_id=manifest.handoff_id)
    assert validation.valid is True
    assert validation.evidence_coverage == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_handoff_restore_rehydrates_global_witness_records(tmp_path: Path) -> None:
    bundle = FakeRunBundle(
        manifest={
            "run_id": "run-1",
            "conversation_id": "conv-1",
            "turn_id": "turn-1",
            "task_memory": {"turn_count": 4, "phase": "RECALL", "active_plan_id": None},
            "assistant_reply": "global witness reply",
        },
        phases=[],
        checkpoints=[],
    )
    runs = FakeHandoffRunJournal(bundle)
    db = FakeHandoffDB()
    db.context_snapshots["run-1"]["items"] = [
        {"marker": "S1", "ref_kind": "citem", "ref_id": "lc-1", "content": "local fact"},
        {"marker": "S2", "ref_kind": "citem", "ref_id": "gc-1", "content": "global fact"},
        {"marker": "P1", "ref_kind": "summary", "ref_id": "gs-1", "content": "global summary"},
    ]
    db.local_citem_records["lc-1"] = {
        "local_citem_id": "lc-1",
        "semantic_identity_id": "11111111-1111-1111-1111-111111111111",
        "conversation_id": "conv-1",
        "type": "FACT",
        "text": "local fact",
        "meta_json": {},
        "provenance_json": {},
        "validity": "confirmed",
        "salience": 0.7,
        "created_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    }
    db.local_citem_evidence_rows.append({
        "local_citem_id": "lc-1",
        "source_id": "src-1",
        "chunk_id": None,
        "edu_id": None,
        "ordinal": 0,
        "locator_json": {"source_span_id": "sp-1"},
    })
    db.global_citem_records["gc-1"] = {
        "global_citem_id": "gc-1",
        "semantic_identity_id": "22222222-2222-2222-2222-222222222222",
        "origin_conversation_id": "conv-1",
        "promotion_origin_local_citem_id": "lc-1",
        "type": "FACT",
        "text": "global fact",
        "meta_json": {},
        "provenance_json": {},
        "validity": "confirmed",
        "salience": 0.9,
        "created_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    }
    db.global_citem_evidence_rows.append({
        "global_citem_id": "gc-1",
        "ordinal": 0,
        "evidence_kind": "chunk_snippet",
        "source_text_snapshot": "u",
        "locator_json": {"source_id": "src-1", "source_span_id": "sp-1"},
    })
    db.global_summary_records["gs-1"] = {
        "global_summary_id": "gs-1",
        "level": "MASTER",
        "text": "global summary",
        "covers_json": {"origin_global_citem_ids": ["gc-1"]},
        "created_at": "2026-04-30T00:00:00+00:00",
        "updated_at": "2026-04-30T00:00:00+00:00",
        "vector_state": "INDEXED",
    }
    db.global_summary_origin_rows.append({
        "global_summary_id": "gs-1",
        "origin_kind": "global_citem",
        "origin_id": "gc-1",
        "ordinal": 0,
    })
    store = FakeHandoffStore()
    store.fetched = {}
    service = DemoHandoffService(rel_db=db, citem_store=store, run_journal=runs, artifacts_root=tmp_path)

    manifest = await service.create_handoff(conversation_id="conv-1", source_run_id="run-1", rationale="restore globals")
    restore = await service.restore_handoff(
        handoff_id=manifest.handoff_id,
        target_conversation_id="conv-3",
        target_run_id="run-3",
    )

    assert restore.valid is True
    assert restore.diff["restored_local_citems_witness"] == 1
    assert restore.diff["restored_global_citems_witness"] == 1
    assert restore.diff["restored_global_citem_evidence"] == 1
    assert restore.diff["restored_global_summaries_witness"] == 1
    assert restore.diff["restored_global_summary_origins"] == 1
    assert len(db.global_citem_records) == 2
    restored_global_rows = [row for row in db.global_citem_records.values() if row.get("origin_conversation_id") == "conv-3"]
    assert len(restored_global_rows) == 1
    restored_global_row = restored_global_rows[0]
    assert restored_global_row["promotion_origin_local_citem_id"] in db.local_citem_records
    assert restored_global_row["vector_state"] == "INDEXED"
    restored_global_evidence = [row for row in db.global_citem_evidence_rows if row.get("global_citem_id") == restored_global_row["global_citem_id"]]
    assert len(restored_global_evidence) == 1
    assert restored_global_evidence[0]["locator_json"]["source_span_id"] in {row["span_id"] for row in db.saved_spans}
    restored_global_summaries = [row for row in db.global_summary_records.values() if row.get("global_summary_id") != "gs-1"]
    assert len(restored_global_summaries) == 1
    restored_global_summary = restored_global_summaries[0]
    assert restored_global_summary["covers_json"]["origin_global_citem_ids"] == [restored_global_row["global_citem_id"]]
    restored_global_origins = [row for row in db.global_summary_origin_rows if row.get("global_summary_id") == restored_global_summary["global_summary_id"]]
    assert len(restored_global_origins) == 1
    assert restored_global_origins[0]["origin_id"] == restored_global_row["global_citem_id"]
