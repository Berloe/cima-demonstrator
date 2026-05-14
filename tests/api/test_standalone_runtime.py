from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def _load_app(monkeypatch,tmp_path):
    monkeypatch.setenv("CIMA_DEMO_RUNTIME_MODE", "standalone")
    monkeypatch.setenv("CIMA_DEMO_DEMO_MODE", "true")
    monkeypatch.setenv("CIMA_DEMO_API_KEY_REQUIRED", "false")
    monkeypatch.setenv("CIMA_DEMO_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("CIMA_DEMO_DEMO_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    import cima_demo.api.settings as settings_mod
    settings_mod._settings = None
    import cima_demo.api.app as app_mod
    app_mod = importlib.reload(app_mod)
    return app_mod.app


def test_standalone_http_flow(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CIMA_DEMO_DEMO_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    app = _load_app(monkeypatch,tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/readyz").status_code == 200

        conv = client.post("/cima_demo/conversations", json={"external_conversation_id": "ext-1", "title": "demo"})
        assert conv.status_code == 201
        conversation_id = conv.json()["conversation_id"]

        completion = client.post(
            "/v1/chat/completions",
            json={
                "model": "cima_demo",
                "stream": False,
                "conversation_id": conversation_id,
                "messages": [{"role": "user", "content": "What is the current context?"}],
            },
        )
        assert completion.status_code == 200, completion.text
        body = completion.json()
        assert body["model"] == "cima_demo"
        answer = body["choices"][0]["message"]["content"]
        assert "CIMA context" in answer

        context = client.post(
            "/cima_demo/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-1",
                "user_text": "give me an overview",
                "mode": "chat",
                "selected_artifact_ids": [],
                "selected_scope": "conversation",
                "max_context_tokens": 600,
                "reserve_output_tokens": 128,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 200, context.text
        payload = context.json()
        assert payload["context_id"]
        assert payload["run_id"]
        assert isinstance(payload["markers"], list)
        assert "marker_resolution" in payload

        snapshot = client.get(f"/cima_demo/context/{payload['context_id']}")
        assert snapshot.status_code == 200
        snap = snapshot.json()
        assert snap["context_id"] == payload["context_id"]
        assert "marker_resolution" in snap

        replay = client.get(f"/cima_demo/runs/{payload['run_id']}/replay")
        assert replay.status_code == 200, replay.text
        replay_json = replay.json()
        assert replay_json["run"]["run_id"] == payload["run_id"]
        assert isinstance(replay_json["checkpoints"], list)
        assert replay_json["context_snapshots"][0]["context_id"] == payload["context_id"]
        assert "marker_resolution" in replay_json["context_snapshots"][0]

        legacy = client.post("/kima/chat")
        assert legacy.status_code == 404



def test_context_route_rejects_deleting_conversation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CIMA_DEMO_DEMO_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    app = _load_app(monkeypatch,tmp_path)
    with TestClient(app) as client:
        conv = client.post("/cima_demo/conversations", json={"external_conversation_id": "ext-2", "title": "demo"})
        assert conv.status_code == 201
        conversation_id = conv.json()["conversation_id"]
        app.state.db.conversations[conversation_id]["status"] = "DELETING"

        context = client.post(
            "/cima_demo/context/get",
            json={
                "conversation_id": conversation_id,
                "request_id": "req-del",
                "user_text": "give me an overview",
                "mode": "chat",
                "selected_artifact_ids": [],
                "selected_scope": "conversation",
                "max_context_tokens": 600,
                "reserve_output_tokens": 128,
                "tokenizer_id": "standalone",
                "model_id": "standalone",
            },
        )
        assert context.status_code == 409
        body = context.json()
        assert body["detail"]["code"] == "CONVERSATION_DELETING"


def test_run_demo_local_script_exists_and_is_executable():
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_demo_local.sh"
    assert script.exists()
    assert script.stat().st_mode & 0o111


def test_standalone_runtime_can_use_llamacpp_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CIMA_DEMO_STANDALONE_LLM_BACKEND", "llamacpp")
    monkeypatch.setenv("CIMA_DEMO_LLM_URL", "http://127.0.0.1:18080")
    app = _load_app(monkeypatch, tmp_path)
    with TestClient(app):
        assert app.state.llm.__class__.__name__ == "LlamaCppAdapter"
