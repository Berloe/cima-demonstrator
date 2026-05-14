from __future__ import annotations

from fastapi.testclient import TestClient

from cima_demo.api.app import app


def test_rag_query_returns_minimal_empty_context() -> None:
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "What is CIMA?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"] == ""
    assert payload["text"] == ""
    assert payload["sources"] == []
    assert payload["source_documents"] == []
    assert payload["sourceDocuments"] == []
