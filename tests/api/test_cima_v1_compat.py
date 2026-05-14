from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.test_standalone_runtime import _load_app


def test_cima_v1_conversation_context_and_delete_aliases(monkeypatch, tmp_path: Path):
    app = _load_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        conv = client.post(
            "/cima/v1/conversations/upsert",
            json={
                "external_system": "open_scenarios",
                "external_conversation_id": "ext-v1",
                "metadata": {"case_id": "case-1"},
            },
        )
        assert conv.status_code == 200, conv.text
        conversation_id = conv.json()["conversation_id"]

        source = client.post(
            "/cima/v1/sources/register_text",
            json={
                "conversation_id": conversation_id,
                "text": "Document body",
                "source_kind": "file",
                "displayable": False,
                "processable": True,
            },
        )
        assert source.status_code == 202, source.text

        context = client.post(
            "/cima/v1/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-v1",
                "user_text": "summarize",
                "mode": "chat",
                "max_context_tokens": 600,
                "reserve_output_tokens": 128,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 200, context.text
        assert context.json()["context_id"]

        delete = client.delete(f"/cima/v1/conversations/{conversation_id}?purge=true")
        assert delete.status_code in {200, 202, 204}, delete.text


def test_v1_register_text_file_resolves_marker_lineage(monkeypatch, tmp_path: Path):
    app = _load_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        conv = client.post(
            "/cima/v1/conversations/upsert",
            json={"external_system": "open_scenarios", "external_conversation_id": "lineage-v1"},
        )
        assert conv.status_code == 200, conv.text
        conversation_id = conv.json()["conversation_id"]

        source = client.post(
            "/cima/v1/sources/register_text",
            json={
                "conversation_id": conversation_id,
                "text": "Alpha evidence. Beta evidence.",
                "source_kind": "file",
                "displayable": False,
                "processable": True,
                "external_message_id": "doc-1",
            },
        )
        assert source.status_code == 202, source.text
        source_body = source.json()
        assert source_body["status"] == "indexed"
        assert source_body["source_span_id"]

        context = client.post(
            "/cima/v1/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-lineage",
                "user_text": "Alpha",
                "mode": "chat",
                "max_context_tokens": 600,
                "reserve_output_tokens": 128,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 200, context.text
        payload = context.json()
        assert payload["markers"]
        assert payload["resolved_source_count"] > 0
        assert payload["resolved_span_count"] > 0
        assert payload["resolution_mode"] in {"witness_first", "legacy_fallback", "mixed"}
        assert payload["marker_resolution"][0]["resolved_source_ids"]
        assert payload["marker_resolution"][0]["resolved_span_ids"]


def test_v1_register_text_file_uses_granular_spans_without_losing_source(monkeypatch, tmp_path: Path):
    app = _load_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        conv = client.post(
            "/cima/v1/conversations/upsert",
            json={"external_system": "open_scenarios", "external_conversation_id": "granular-lineage-v1"},
        )
        assert conv.status_code == 200, conv.text
        conversation_id = conv.json()["conversation_id"]

        text = "\n\n".join([
            "Alpha evidence block explains the first issue and contains alpha-key.",
            "Beta evidence block explains the second issue and contains beta-key.",
            "Gamma evidence block explains the third issue and contains gamma-key.",
        ])
        source = client.post(
            "/cima/v1/sources/register_text",
            json={
                "conversation_id": conversation_id,
                "text": text,
                "source_kind": "file",
                "displayable": False,
                "processable": True,
                "external_message_id": "doc-granular",
            },
        )
        assert source.status_code == 202, source.text
        source_id = source.json()["source_id"]

        context = client.post(
            "/cima/v1/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-granular-lineage",
                "user_text": "alpha-key beta-key gamma-key",
                "mode": "chat",
                "max_context_tokens": 2000,
                "reserve_output_tokens": 128,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 200, context.text
        payload = context.json()
        assert payload["resolved_source_ids"] == [source_id]
        assert payload["resolved_source_count"] == 1
        assert payload["resolved_span_count"] >= 2
        marker_span_sets = [tuple(row.get("resolved_span_ids") or []) for row in payload["marker_resolution"]]
        marker_span_ids = {span_id for spans in marker_span_sets for span_id in spans}
        assert len(marker_span_ids) >= 2
        assert all(row.get("resolved_source_ids") == [source_id] for row in payload["marker_resolution"] if row.get("resolved_source_ids"))

        saved_segment_spans = [
            row for row in app.state.db.demo_source_spans.values()
            if row.get("conversation_id") == conversation_id and row.get("span_kind") == "inline_segment"
        ]
        assert len(saved_segment_spans) >= 2
        assert {row["source_id"] for row in saved_segment_spans} == {source_id}
        assert all("char_start" in row and "char_end" in row for row in saved_segment_spans)


def test_context_get_selects_under_small_budget_without_loading_full_document(monkeypatch, tmp_path: Path):
    app = _load_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        conv = client.post(
            "/cima/v1/conversations/upsert",
            json={"external_system": "open_scenarios", "external_conversation_id": "small-budget-v1"},
        )
        assert conv.status_code == 200, conv.text
        conversation_id = conv.json()["conversation_id"]
        blocks = [
            f"Alpha budget evidence block {idx} contains alpha-key and detail {idx}. " + ("filler " * 20)
            for idx in range(20)
        ]
        source = client.post(
            "/cima/v1/sources/register_text",
            json={
                "conversation_id": conversation_id,
                "text": "\n\n".join(blocks),
                "source_kind": "file",
                "displayable": False,
                "processable": True,
                "external_message_id": "doc-small-budget",
            },
        )
        assert source.status_code == 202, source.text
        assert source.json()["inline_ingested_count"] > 1

        context = client.post(
            "/cima/v1/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-small-budget",
                "user_text": "alpha-key detail 1",
                "mode": "chat",
                "max_context_tokens": 120,
                "reserve_output_tokens": 40,
                "overhead_tokens": 0,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 200, context.text
        payload = context.json()
        assert 0 < payload["token_usage"]["context"] <= 80
        assert len(payload["markers"]) < source.json()["inline_ingested_count"]
        assert payload["resolved_source_count"] == 1
        assert payload["resolved_span_count"] == len(payload["markers"])


def test_standalone_creates_l1_summary_and_zoom_out_preserves_lineage(monkeypatch, tmp_path: Path):
    app = _load_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        conv = client.post(
            "/cima/v1/conversations/upsert",
            json={"external_system": "open_scenarios", "external_conversation_id": "summary-l1-v1"},
        )
        assert conv.status_code == 200, conv.text
        conversation_id = conv.json()["conversation_id"]
        text = "\n\n".join([
            "Alpha source segment explains core problem and cites alpha-key.",
            "Beta source segment explains second problem and cites beta-key.",
            "Gamma source segment explains proposed solution and cites gamma-key.",
        ])
        source = client.post(
            "/cima/v1/sources/register_text",
            json={
                "conversation_id": conversation_id,
                "text": text,
                "source_kind": "file_text",
                "displayable": False,
                "processable": True,
                "external_message_id": "doc-summary-l1",
            },
        )
        assert source.status_code == 202, source.text
        assert source.json()["inline_ingested_count"] == 3

        summaries = app.state.db.local_summary_records
        assert len(summaries) == 1
        summary = next(iter(summaries.values()))
        assert summary["level"] == "EPOCH"
        assert summary["covers_json"]["source_id"] == source.json()["source_id"]
        assert len(summary["covers_json"]["origin_citem_ids"]) == 3
        assert len(app.state.db.local_summary_origin_rows) == 3

        context = client.post(
            "/cima/v1/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-summary-l1",
                "user_text": "alpha-key",
                "mode": "chat",
                "max_context_tokens": 800,
                "reserve_output_tokens": 128,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 200, context.text
        context_body = context.json()
        assert context_body["markers"]

        zoom = client.post(
            "/cima/v1/context/zoom",
            json={"context_id": context_body["context_id"], "zoom_targets": [context_body["markers"][0]], "max_evidence_tokens": 800},
        )
        assert zoom.status_code == 200, zoom.text
        assert zoom.json()["resolved_source_count"] == 1
        assert zoom.json()["resolved_span_count"] >= 1

        zoom_out = client.post(
            "/cima/v1/context/zoom_out",
            json={"context_id": context_body["context_id"], "targets": [context_body["markers"][0]], "max_perspective_tokens": 800},
        )
        assert zoom_out.status_code == 200, zoom_out.text
        zoom_out_body = zoom_out.json()
        assert zoom_out_body["perspective_block"]
        assert zoom_out_body["markers_added"] == ["P1"]
        assert zoom_out_body["resolution_mode"] in {"witness_first", "mixed"}
        assert set(zoom_out_body["focus_citem_ids"]).issubset(set(summary["covers_json"]["origin_citem_ids"]))
