from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cima_demo.demo.context import DemoContextService
from cima_demo.demo.lineage import DemoLineageService
from cima_demo.domain.entities import ContextView, SummaryNode, TaskMemory
from cima_demo.domain.value_objects import ContextBudget


class _FakeLineageDB:
    def __init__(self) -> None:
        self.sources: list[dict] = []
        self.spans: list[dict] = []
        self.edges: list[dict] = []
        self.resolutions: list[dict] = []
        self.snapshots: dict[str, dict] = {}
        self.answers: list[dict] = []
        self.summaries: list[SummaryNode] = []
        self.local_citem_records: dict[str, dict] = {}
        self.local_citem_evidence_rows: list[dict] = []
        self.local_summary_records: dict[str, dict] = {}
        self.local_summary_origin_rows: list[dict] = []
        self.global_citem_records: dict[str, dict] = {}
        self.global_citem_evidence_rows: list[dict] = []
        self.global_summary_records: dict[str, dict] = {}
        self.global_summary_origin_rows: list[dict] = []

    async def save_demo_source(self, source_json: dict):
        self.sources.append(dict(source_json))

    async def save_demo_source_span(self, span_json: dict):
        self.spans.append(dict(span_json))

    async def save_demo_lineage_edge(self, edge_json: dict):
        self.edges.append(dict(edge_json))

    async def save_demo_summary_resolution(self, resolution_json: dict):
        self.resolutions.append(dict(resolution_json))

    async def save_demo_context_snapshot(self, snapshot_json: dict):
        self.snapshots[snapshot_json["context_id"]] = dict(snapshot_json)

    async def load_demo_context_snapshot(self, context_id: str):
        snap = self.snapshots.get(context_id)
        return dict(snap) if snap is not None else None

    async def load_demo_context_snapshots_for_run(self, run_id: str):
        rows = [dict(row) for row in self.snapshots.values() if row.get("run_id") == run_id]
        rows.sort(key=lambda row: row.get("created_at", ""))
        return rows

    async def load_demo_sources(self, conversation_id: str, source_ids: list[str]):
        wanted = {str(v) for v in source_ids}
        return [dict(row) for row in self.sources if row.get("conversation_id") == conversation_id and str(row.get("source_id")) in wanted]

    async def load_demo_source_spans(self, conversation_id: str, span_ids: list[str]):
        wanted = {str(v) for v in span_ids}
        return [dict(row) for row in self.spans if row.get("conversation_id") == conversation_id and str(row.get("span_id")) in wanted]

    async def save_demo_answer_lineage(self, answer_json: dict):
        self.answers.append(dict(answer_json))

    async def load_demo_lineage_edges(self, conversation_id: str, *, src_kind: str | None = None, src_ids: list[str] | None = None, dst_kind: str | None = None, dst_ids: list[str] | None = None):
        rows = [dict(row) for row in self.edges if row.get("conversation_id") == conversation_id]
        if src_kind is not None:
            rows = [row for row in rows if row.get("src_kind") == src_kind]
        if src_ids is not None:
            src_id_set = {str(v) for v in src_ids}
            rows = [row for row in rows if str(row.get("src_id")) in src_id_set]
        if dst_kind is not None:
            rows = [row for row in rows if row.get("dst_kind") == dst_kind]
        if dst_ids is not None:
            dst_id_set = {str(v) for v in dst_ids}
            rows = [row for row in rows if str(row.get("dst_id")) in dst_id_set]
        return rows

    async def load_demo_summary_resolutions(self, conversation_id: str, summary_ids: list[str] | None = None):
        rows = [dict(row) for row in self.resolutions if row.get("conversation_id") == conversation_id]
        if summary_ids is not None:
            wanted = {str(v) for v in summary_ids}
            rows = [row for row in rows if str(row.get("summary_id")) in wanted]
        return rows

    async def list_local_summary_records(self, conversation_id: str, summary_ids: list[str] | None = None):
        rows = [dict(row) for row in self.local_summary_records.values() if row.get("conversation_id") == conversation_id]
        if summary_ids is not None:
            wanted = {str(v) for v in summary_ids}
            rows = [row for row in rows if str(row.get("local_summary_id")) in wanted]
        return rows

    async def list_global_summary_records(self, *, summary_ids: list[str] | None = None, origin_conversation_id: str | None = None):
        rows = [dict(row) for row in self.global_summary_records.values()]
        if summary_ids is not None:
            wanted = {str(v) for v in summary_ids}
            rows = [row for row in rows if str(row.get("global_summary_id")) in wanted]
        if origin_conversation_id is not None:
            rows = [row for row in rows if row.get("origin_conversation_id") == origin_conversation_id]
        return rows

    async def list_local_summary_origins(self, local_summary_id: str):
        return [dict(row) for row in self.local_summary_origin_rows if row.get("local_summary_id") == local_summary_id]

    async def list_global_summary_origins(self, global_summary_id: str):
        return [dict(row) for row in self.global_summary_origin_rows if row.get("global_summary_id") == global_summary_id]

    async def list_local_citem_records(self, conversation_id: str, *, citem_ids: list[str] | None = None):
        rows = [dict(row) for row in self.local_citem_records.values() if row.get("conversation_id") == conversation_id]
        if citem_ids is not None:
            wanted = {str(v) for v in citem_ids}
            rows = [row for row in rows if str(row.get("local_citem_id")) in wanted]
        return rows

    async def list_global_citem_records(self, *, global_citem_ids: list[str] | None = None, semantic_identity_ids: list[str] | None = None, origin_conversation_id: str | None = None):
        rows = [dict(row) for row in self.global_citem_records.values()]
        if global_citem_ids is not None:
            wanted = {str(v) for v in global_citem_ids}
            rows = [row for row in rows if str(row.get("global_citem_id")) in wanted]
        if semantic_identity_ids is not None:
            wanted = {str(v) for v in semantic_identity_ids}
            rows = [row for row in rows if str(row.get("semantic_identity_id")) in wanted]
        if origin_conversation_id is not None:
            rows = [row for row in rows if row.get("origin_conversation_id") == origin_conversation_id]
        return rows

    async def list_local_citem_evidence(self, local_citem_id: str):
        return [dict(row) for row in self.local_citem_evidence_rows if row.get("local_citem_id") == local_citem_id]

    async def list_global_citem_evidence(self, global_citem_id: str):
        return [dict(row) for row in self.global_citem_evidence_rows if row.get("global_citem_id") == global_citem_id]

    async def fetch_pyramid_tops(self, conversation_id: str, limit: int | None = None):
        items = [s for s in self.summaries if s.conversation_id == conversation_id]
        return items[: limit or len(items)]


@dataclass
class _Chunk:
    index: int
    text: str
    page_num: int | None = None
    section_hint: str | None = None
    filename: str | None = None
    doc_type: str | None = None


def _seed_citable_builder_refs(db: _FakeLineageDB, conversation_id: str) -> None:
    db.local_citem_records["c1"] = {"local_citem_id": "c1", "conversation_id": conversation_id}
    db.local_citem_records["c2"] = {"local_citem_id": "c2", "conversation_id": conversation_id}
    db.local_citem_evidence_rows.extend([
        {
            "local_citem_id": "c1",
            "conversation_id": conversation_id,
            "source_id": "src-1",
            "source_span_id": "span-1",
            "locator_json": {"source_id": "src-1", "source_span_id": "span-1"},
        },
        {
            "local_citem_id": "c2",
            "conversation_id": conversation_id,
            "source_id": "src-1",
            "source_span_id": "span-2",
            "locator_json": {"source_id": "src-1", "source_span_id": "span-2"},
        },
    ])
    db.local_summary_records["s1"] = {"local_summary_id": "s1", "conversation_id": conversation_id}
    db.local_summary_origin_rows.append({"local_summary_id": "s1", "origin_kind": "citem", "origin_id": "c2"})
    db.sources.append({
        "conversation_id": conversation_id,
        "source_id": "src-1",
        "source_kind": "chat",
        "role": "user",
        "display_text": "source text",
    })
    db.spans.extend([
        {
            "conversation_id": conversation_id,
            "span_id": "span-1",
            "source_id": "src-1",
            "preview_text": "first fact",
            "locator": {"source_span_id": "span-1"},
        },
        {
            "conversation_id": conversation_id,
            "span_id": "span-2",
            "source_id": "src-1",
            "preview_text": "second fact",
            "locator": {"source_span_id": "span-2"},
        },
    ])


class _FakeBuilder:
    async def build(self, **_: object) -> ContextView:
        return ContextView(
            text="CONTEXT\n\n[S1] first fact\n\n[S2] second fact",
            tokens_used=42,
            coverage_score=0.75,
            citem_ids=["c1", "c2"],
            items=[
                {
                    "marker": "S1",
                    "ref_kind": "citem",
                    "ref_id": "c1",
                    "content": "first fact",
                    "section": "protected",
                    "item_type": "FACT",
                },
                {
                    "marker": "S2",
                    "ref_kind": "summary",
                    "ref_id": "s1",
                    "content": "second fact",
                    "section": "global_summary",
                    "item_type": "SUMMARY",
                },
            ],
        )


class _FakeMemory:
    def __init__(self) -> None:
        self.batches: list[tuple[list[dict], str, str, str]] = []

    async def ingest_batch(self, conclusions, phase, conversation_id, turn_id):
        self.batches.append((list(conclusions), phase, conversation_id, turn_id))


class _FakeJournal:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.written_json: list[tuple[str, dict]] = []
        self.written_text: list[tuple[str, str]] = []

    async def write_json_artifact(self, *, conversation_id: str, run_id: str, relative_path: str, payload: dict):
        path = self.root / conversation_id / run_id / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(payload), encoding="utf-8")
        self.written_json.append((relative_path, dict(payload)))

    async def write_text_artifact(self, *, conversation_id: str, run_id: str, relative_path: str, text: str):
        path = self.root / conversation_id / run_id / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.written_text.append((relative_path, text))


@pytest.mark.asyncio
async def test_lineage_service_records_sources_spans_summaries_and_answers():
    db = _FakeLineageDB()
    lineage = DemoLineageService(rel_db=db)

    source, full_span = await lineage.register_text_source(
        conversation_id="11111111-1111-4111-8111-111111111111",
        source_kind="chat_user",
        role="user",
        display_text="hola mundo",
        process_text="hola mundo",
        origin_ref="turn-1",
        metadata={"run_id": "run-1"},
    )
    assert source.source_id
    assert full_span is not None
    assert db.sources[0]["source_kind"] == "chat_user"
    assert db.spans[0]["preview_text"] == "hola mundo"

    chunks = [
        _Chunk(index=0, text="hola", page_num=1, section_hint="intro", filename="a.txt", doc_type="doc_chunk"),
        _Chunk(index=1, text="mundo", page_num=1, section_hint="intro", filename="a.txt", doc_type="doc_chunk"),
    ]
    spans = await lineage.register_spans_from_chunks(
        conversation_id=source.conversation_id,
        source_id=source.source_id,
        process_text="hola mundo",
        chunks=chunks,
    )
    assert set(spans.keys()) == {0, 1}

    await lineage.record_citem_lineage(
        conversation_id=source.conversation_id,
        citem_id="c1",
        source_id=source.source_id,
        source_span_ids=[full_span.span_id],
        dependency_ids=["c0"],
        metadata={"kind": "chat_user"},
    )
    assert {e["relation"] for e in db.edges} >= {"DERIVED_FROM_SOURCE", "DERIVED_FROM_SPAN", "DEPENDS_ON"}

    resolution = await lineage.record_summary_resolution(
        conversation_id=source.conversation_id,
        summary_id="22222222-2222-4222-8222-222222222222",
        summary_text="compressed summary",
        origin_citem_ids=["c1", "c2"],
        metadata={"level": 1},
    )
    assert resolution.summary_id == "22222222-2222-4222-8222-222222222222"
    assert db.resolutions[0]["origin_citem_ids"] == ["c1", "c2"]
    assert any(e["relation"] == "SUMMARIZES" for e in db.edges)

    answer = await lineage.record_answer_lineage(
        conversation_id=source.conversation_id,
        run_id="33333333-3333-4333-8333-333333333333",
        response_turn_id="44444444-4444-4444-8444-444444444444",
        context_id="55555555-5555-4555-8555-555555555555",
        answer_text="respuesta final",
        cited_markers=["S1"],
        selected_items=[{"marker": "S1", "ref_kind": "citem", "ref_id": "c1", "content": "fact"}],
    )
    assert answer.cited_markers == ["S1"]
    assert answer.resolved_source_count == 1
    assert answer.resolved_span_count == 1
    assert answer.resolution_mode == "legacy_fallback"
    assert db.answers[0]["answer_text"] == "respuesta final"
    assert db.answers[0]["resolved_source_count"] == 1
    assert db.answers[0]["resolved_span_count"] == 1
    assert any(e["relation"] == "USES_CONTEXT" for e in db.edges)
    assert any(e["relation"] == "USES_ITEM" for e in db.edges)


@pytest.mark.asyncio
async def test_context_service_persists_snapshots_and_supports_zoom(tmp_path: Path):
    db = _FakeLineageDB()
    db.summaries = [
        SummaryNode(conversation_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", level=2, content="master perspective"),
        SummaryNode(conversation_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", level=1, content="topic perspective"),
    ]
    _seed_citable_builder_refs(db, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    journal = _FakeJournal(tmp_path)
    memory = _FakeMemory()
    svc = DemoContextService(
        base_builder=_FakeBuilder(),
        memory_service=memory,
        rel_db=db,
        run_journal=journal,
    )
    token = svc.bind_run(
        run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        conversation_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        query_text="what happened?",
    )
    budget = ContextBudget(max_tokens=1024, overhead_tokens=128)
    try:
        result = await svc.get_context(
            conversation_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            query="what happened?",
            phase="recall",
            task_memory=TaskMemory(conversation_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            plan=None,
            budget=budget,
        )
        assert result["context_id"]
        assert result["markers"] == ["S1", "S2"]
        assert svc.last_snapshot_id() == result["context_id"]
        assert result["context_id"] in db.snapshots
        assert any(path.startswith("context_snapshot_") for path, _ in journal.written_json)
        assert any(path.startswith("context_pack_") for path, _ in journal.written_text)

        zoom = await svc.zoom(context_id=result["context_id"], zoom_targets=["S2"], max_evidence_tokens=200)
        assert zoom["markers_added"] == ["S2"]
        assert "second fact" in zoom["evidence_block"]

        zoom_out = await svc.zoom_out(context_id=result["context_id"], targets=["MASTER"], max_perspective_tokens=200)
        assert zoom_out["markers_added"]
        assert "perspective" in zoom_out["perspective_block"]

        apply = await svc.apply_memory(
            conversation_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            conclude=[
                "FACT: There is enough evidence | prov=S1",
                "NOTE: Follow up later | prov=NONE",
                "BROKEN without colon",
            ],
            phase="synthesis",
            turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        )
        assert len(apply["accepted"]) == 2
        assert len(apply["rejected"]) == 1
        assert memory.batches and memory.batches[0][0][0]["type"] == "FACT"
    finally:
        svc.reset_run(token)
        assert svc.last_snapshot_id() is None


@pytest.mark.asyncio
async def test_context_zoom_prefers_witness_span_previews_and_reports_resolution(tmp_path: Path):
    db = _FakeLineageDB()
    conversation_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )
    db.local_summary_records["s1"] = {
        "local_summary_id": "s1",
        "conversation_id": conversation_id,
        "level": "MASTER",
        "text": "summary from witness",
        "covers_json": {"origin_citem_ids": ["c1"]},
        "created_at": "2026-04-30T00:00:00+00:00",
    }
    db.local_summary_origin_rows.append(
        {
            "local_summary_id": "s1",
            "origin_kind": "local_citem",
            "origin_id": "c1",
        }
    )
    db.sources.append(
        {
            "source_id": "src-1",
            "conversation_id": conversation_id,
            "source_kind": "chat_user",
            "role": "user",
            "origin_ref": "turn-1",
            "display_text": "source text",
            "process_text": "source text",
            "metadata": {},
        }
    )
    db.spans.append(
        {
            "span_id": "span-1",
            "source_id": "src-1",
            "conversation_id": conversation_id,
            "span_kind": "chunk",
            "char_start": 0,
            "char_end": 24,
            "locator": {"page_num": 2, "chunk_index": 5},
            "preview_text": "precise witness preview",
        }
    )

    svc = DemoContextService(
        base_builder=_FakeBuilder(),
        memory_service=_FakeMemory(),
        rel_db=db,
        run_journal=_FakeJournal(tmp_path),
    )
    token = svc.bind_run(
        run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        conversation_id=conversation_id,
        turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        query_text="what happened?",
    )
    budget = ContextBudget(max_tokens=1024, overhead_tokens=128)
    try:
        result = await svc.get_context(
            conversation_id=conversation_id,
            query="what happened?",
            phase="recall",
            task_memory=TaskMemory(conversation_id=conversation_id),
            plan=None,
            budget=budget,
        )
        zoom = await svc.zoom(context_id=result["context_id"], zoom_targets=["S2"], max_evidence_tokens=200)
    finally:
        svc.reset_run(token)

    assert zoom["markers_added"] == ["S2"]
    assert zoom["resolution_mode"] == "witness_first"
    assert zoom["resolved_source_ids"] == ["src-1"]
    assert zoom["resolved_span_ids"] == ["span-1"]
    assert "precise witness preview" in zoom["evidence_block"]
    assert "page=2" in zoom["evidence_block"]
    assert "chunk=5" in zoom["evidence_block"]


@pytest.mark.asyncio
async def test_zoom_out_prefers_related_summary_perspective(tmp_path: Path):
    db = _FakeLineageDB()
    conversation_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )
    db.local_summary_records["s1"] = {
        "local_summary_id": "s1",
        "conversation_id": conversation_id,
        "level": "MASTER",
        "text": "summary from witness",
        "covers_json": {"origin_citem_ids": ["c1"]},
        "created_at": "2026-04-30T00:00:00+00:00",
    }
    db.local_summary_origin_rows.append(
        {
            "local_summary_id": "s1",
            "origin_kind": "local_citem",
            "origin_id": "c1",
        }
    )
    related = SummaryNode(conversation_id=conversation_id, level=3, content="related perspective", origin_citem_ids=["c1"])
    unrelated = SummaryNode(conversation_id=conversation_id, level=3, content="unrelated perspective", origin_citem_ids=["c9"])
    for node in (related, unrelated):
        setattr(node, "summary_resolution_mode", "witness_first")
        setattr(node, "summary_ref_kind", "local_summary")
        setattr(node, "summary_scope", "local")
    db.summaries = [related, unrelated]

    svc = DemoContextService(
        base_builder=_FakeBuilder(),
        memory_service=_FakeMemory(),
        rel_db=db,
        run_journal=_FakeJournal(tmp_path),
    )
    token = svc.bind_run(
        run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        conversation_id=conversation_id,
        turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        query_text="what happened?",
    )
    budget = ContextBudget(max_tokens=1024, overhead_tokens=128)
    try:
        result = await svc.get_context(
            conversation_id=conversation_id,
            query="what happened?",
            phase="recall",
            task_memory=TaskMemory(conversation_id=conversation_id),
            plan=None,
            budget=budget,
        )
        zoom_out = await svc.zoom_out(context_id=result["context_id"], targets=["S2"], max_perspective_tokens=200)
    finally:
        svc.reset_run(token)

    assert zoom_out["resolution_mode"] == "witness_first"
    assert zoom_out["focus_citem_ids"] == ["c1"]
    assert "related perspective" in zoom_out["perspective_block"]
    assert "unrelated perspective" not in zoom_out["perspective_block"]


@pytest.mark.asyncio
async def test_context_service_persists_witness_first_resolution_metadata(tmp_path: Path):
    db = _FakeLineageDB()
    conversation_id = "99999999-9999-4999-8999-999999999999"
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )
    db.local_summary_records["s1"] = {
        "local_summary_id": "s1",
        "conversation_id": conversation_id,
        "level": "MASTER",
        "text": "summary from witness",
        "covers_json": {"origin_citem_ids": ["c1"]},
        "created_at": "2026-04-30T00:00:00+00:00",
    }
    db.local_summary_origin_rows.append(
        {
            "local_summary_id": "s1",
            "origin_kind": "local_citem",
            "origin_id": "c1",
        }
    )

    journal = _FakeJournal(tmp_path)
    svc = DemoContextService(
        base_builder=_FakeBuilder(),
        memory_service=_FakeMemory(),
        rel_db=db,
        run_journal=journal,
    )
    token = svc.bind_run(
        run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        conversation_id=conversation_id,
        turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        query_text="what happened?",
    )
    budget = ContextBudget(max_tokens=1024, overhead_tokens=128)
    try:
        result = await svc.get_context(
            conversation_id=conversation_id,
            query="what happened?",
            phase="recall",
            task_memory=TaskMemory(conversation_id=conversation_id),
            plan=None,
            budget=budget,
        )
    finally:
        svc.reset_run(token)

    snapshot = db.snapshots[result["context_id"]]
    assert result["resolution_mode"] == "witness_first"
    assert result["marker_resolution"][0]["marker"] == "S1"
    assert result["marker_resolution"][0]["resolution_mode"] == "witness_first"
    assert result["resolved_source_ids"] == ["src-1"]
    assert result["resolved_span_ids"] == ["span-1"]
    assert snapshot["resolution_mode"] == "witness_first"
    assert snapshot["resolved_source_count"] == 1
    assert snapshot["resolved_span_count"] == 1
    assert snapshot["unresolved_ref_ids"] == []
    assert snapshot["marker_resolution"][0]["marker"] == "S1"
    assert snapshot["marker_resolution"][0]["resolution_mode"] == "witness_first"
    context_artifacts = [payload for rel, payload in journal.written_json if rel.startswith("context_snapshot_")]
    assert context_artifacts
    assert context_artifacts[0]["marker_resolution"][0]["marker"] == "S1"

@pytest.mark.asyncio
async def test_answer_lineage_prefers_witness_evidence_and_summary_origins() -> None:
    db = _FakeLineageDB()
    conversation_id = "11111111-1111-4111-8111-111111111111"
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )
    db.local_summary_records["sum-1"] = {
        "local_summary_id": "sum-1",
        "conversation_id": conversation_id,
    }
    db.local_summary_origin_rows.append(
        {
            "local_summary_id": "sum-1",
            "origin_kind": "local_citem",
            "origin_id": "c1",
        }
    )

    lineage = DemoLineageService(rel_db=db)
    answer = await lineage.record_answer_lineage(
        conversation_id=conversation_id,
        run_id="33333333-3333-4333-8333-333333333333",
        response_turn_id="44444444-4444-4444-8444-444444444444",
        context_id="55555555-5555-4555-8555-555555555555",
        answer_text="respuesta con witness",
        cited_markers=["S2"],
        selected_items=[{"marker": "S2", "ref_kind": "summary", "ref_id": "sum-1", "content": "summary"}],
    )

    assert answer.resolution_mode == "witness_first"
    assert answer.resolved_source_ids == ["src-1"]
    assert answer.resolved_span_ids == ["span-1"]
    assert answer.resolved_source_count == 1
    assert answer.resolved_span_count == 1
    assert answer.marker_resolution[0]["marker"] == "S2"
    assert answer.marker_resolution[0]["resolution_mode"] == "witness_first"
    assert answer.lineage[0]["summary_resolution_mode"] == "witness_first"
    assert db.answers[-1]["resolution_mode"] == "witness_first"
    assert db.answers[-1]["marker_resolution"][0]["marker"] == "S2"


@pytest.mark.asyncio
async def test_context_service_marks_legacy_summary_fallback_as_mixed_resolution(tmp_path: Path):
    db = _FakeLineageDB()
    conversation_id = "abababab-abab-4bab-8bab-abababababab"
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )
    db.local_citem_records["legacy-citem"] = {
        "local_citem_id": "legacy-citem",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append({
        "local_citem_id": "legacy-citem",
        "conversation_id": conversation_id,
        "source_id": "src-legacy",
        "source_span_id": "span-legacy",
        "locator_json": {"source_id": "src-legacy", "source_span_id": "span-legacy"},
    })
    db.local_summary_records["legacy-summary"] = {
        "local_summary_id": "legacy-summary",
        "conversation_id": conversation_id,
    }
    db.local_summary_origin_rows.append({
        "local_summary_id": "legacy-summary",
        "origin_kind": "citem",
        "origin_id": "legacy-citem",
    })

    class _BuilderWithLegacySummary:
        async def build(self, **_: object) -> ContextView:
            return ContextView(
                text="CONTEXT\n\n[S1] witness fact\n\n[S2] legacy summary",
                tokens_used=21,
                coverage_score=0.8,
                citem_ids=["c1", "legacy-summary"],
                items=[
                    {
                        "marker": "S1",
                        "ref_kind": "citem",
                        "ref_id": "c1",
                        "content": "witness fact",
                        "section": "protected",
                        "item_type": "FACT",
                    },
                    {
                        "marker": "S2",
                        "ref_kind": "summary",
                        "ref_id": "legacy-summary",
                        "content": "legacy summary",
                        "section": "global_summary",
                        "item_type": "SUMMARY",
                        "summary_resolution_mode": "legacy_fallback",
                        "summary_ref_kind": "legacy_summary",
                    },
                ],
            )

    svc = DemoContextService(
        base_builder=_BuilderWithLegacySummary(),
        memory_service=_FakeMemory(),
        rel_db=db,
        run_journal=_FakeJournal(tmp_path),
    )
    token = svc.bind_run(
        run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        conversation_id=conversation_id,
        turn_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        query_text="what changed?",
    )
    try:
        result = await svc.get_context(
            conversation_id=conversation_id,
            query="what changed?",
            phase="recall",
            task_memory=TaskMemory(conversation_id=conversation_id),
            plan=None,
            budget=ContextBudget(max_tokens=1024, overhead_tokens=128),
        )
    finally:
        svc.reset_run(token)

    snapshot = db.snapshots[result["context_id"]]
    assert result["resolution_mode"] == "mixed"
    assert result["marker_resolution"][0]["resolution_mode"] == "witness_first"
    assert result["marker_resolution"][1]["resolution_mode"] == "legacy_fallback"
    assert snapshot["resolution_mode"] == "mixed"
    assert sorted(snapshot["resolved_source_ids"]) == ["src-1", "src-legacy"]
    assert snapshot["marker_resolution"][1]["resolution_mode"] == "legacy_fallback"


@pytest.mark.asyncio
async def test_context_service_marks_legacy_citem_fallback_as_mixed_resolution(tmp_path: Path):
    db = _FakeLineageDB()
    conversation_id = "cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd"
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )

    class _BuilderWithLegacyCItem:
        async def build(self, **_: object) -> ContextView:
            return ContextView(
                text="CONTEXT\n\n[S1] legacy payload fact",
                tokens_used=11,
                coverage_score=0.7,
                citem_ids=["c1"],
                items=[
                    {
                        "marker": "S1",
                        "ref_kind": "citem",
                        "ref_id": "c1",
                        "content": "legacy payload fact",
                        "section": "direct_evidence",
                        "item_type": "FACT",
                        "item_resolution_mode": "legacy_fallback",
                        "item_resolution_scope": "local",
                    },
                ],
            )

    svc = DemoContextService(
        base_builder=_BuilderWithLegacyCItem(),
        memory_service=_FakeMemory(),
        rel_db=db,
        run_journal=_FakeJournal(tmp_path),
    )
    token = svc.bind_run(
        run_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        conversation_id=conversation_id,
        turn_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        query_text="show me the fact",
    )
    try:
        result = await svc.get_context(
            conversation_id=conversation_id,
            query="show me the fact",
            phase="recall",
            task_memory=TaskMemory(conversation_id=conversation_id),
            plan=None,
            budget=ContextBudget(max_tokens=1024, overhead_tokens=128),
        )
    finally:
        svc.reset_run(token)

    snapshot = db.snapshots[result["context_id"]]
    assert result["resolution_mode"] == "mixed"
    assert result["marker_resolution"][0]["resolution_mode"] == "legacy_fallback"
    assert snapshot["resolution_mode"] == "mixed"
    assert snapshot["items"][0]["item_resolution_mode"] == "legacy_fallback"
    assert snapshot["marker_resolution"][0]["resolution_mode"] == "legacy_fallback"


@pytest.mark.asyncio
async def test_context_service_public_snapshot_normalizes_marker_resolution(tmp_path: Path):
    db = _FakeLineageDB()
    conversation_id = "efefefef-efef-4fef-8fef-efefefefefef"
    _seed_citable_builder_refs(db, conversation_id)
    db.local_citem_records["c1"] = {
        "local_citem_id": "c1",
        "conversation_id": conversation_id,
    }
    db.local_citem_evidence_rows.append(
        {
            "local_citem_id": "c1",
            "source_id": "src-1",
            "locator_json": {"source_span_id": "span-1"},
        }
    )
    db.sources.append({
        "conversation_id": conversation_id,
        "source_id": "src-1",
        "source_kind": "chat",
        "role": "user",
        "display_text": "source text",
    })
    db.spans.append({
        "conversation_id": conversation_id,
        "span_id": "span-1",
        "source_id": "src-1",
        "preview_text": "source preview",
        "locator": {"source_span_id": "span-1"},
    })

    svc = DemoContextService(
        base_builder=_FakeBuilder(),
        memory_service=_FakeMemory(),
        rel_db=db,
        run_journal=_FakeJournal(tmp_path),
    )
    token = svc.bind_run(
        run_id="12121212-1212-4212-8212-121212121212",
        conversation_id=conversation_id,
        turn_id="34343434-3434-4434-8434-343434343434",
        query_text="public snapshot",
    )
    try:
        result = await svc.get_context(
            conversation_id=conversation_id,
            query="public snapshot",
            phase="recall",
            task_memory=TaskMemory(conversation_id=conversation_id),
            plan=None,
            budget=ContextBudget(max_tokens=1024, overhead_tokens=128),
        )
    finally:
        svc.reset_run(token)

    public_snapshot = await svc.load_context_snapshot_public(result["context_id"])
    assert public_snapshot is not None
    assert public_snapshot["resolution_mode"] == "witness_first"
    assert public_snapshot["marker_resolution"][0]["marker"] == "S1"
    assert public_snapshot["marker_resolution"][0]["resolution_mode"] == "witness_first"

    run_snapshots = await svc.load_context_snapshots_for_run_public("12121212-1212-4212-8212-121212121212")
    assert len(run_snapshots) == 1
    assert run_snapshots[0]["marker_resolution"][1]["marker"] == "S2"
