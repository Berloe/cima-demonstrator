"""SPLADEAdapter → SparseEmbeddingPort (Phase 2, APP-D-09).

Replaces the fastembed BM25 internal to QdrantCItemAdapter with a
SPLADE model served over HTTP (TEI-compatible sparse endpoint).

Activation:
    Set KIMA_SPLADE_URL to the base URL of the SPLADE server.
    When empty (default), QdrantCItemAdapter falls back to BM25.

Expected API contract (TEI sparse endpoint):
    POST /embed_sparse
    Body:  {"inputs": "<text>"}
    Reply: [{"index": int, "value": float}, ...]

If the server returns a batch format (list of lists), only the first
element is used — call embed_sparse once per text, not in batch.
"""
from __future__ import annotations

import logging

import httpx

from cima_demo.domain.errors import SparseEmbeddingError
from cima_demo.domain.ports import SparseEmbeddingPort

log = logging.getLogger(__name__)


class SPLADEAdapter(SparseEmbeddingPort):
    """HTTP client for a SPLADE-serving endpoint (TEI sparse format).

    Keeps a single persistent httpx.AsyncClient (connection-pooled) for the
    lifetime of the adapter.  Call close() on shutdown if you want a clean teardown;
    otherwise the client is collected at process exit.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._url = base_url.rstrip("/") + "/embed_sparse"
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed_sparse(self, text: str) -> dict[int, float]:
        """Return sparse {token_id: weight} vector via SPLADE server."""
        try:
            resp = await self._client.post(self._url, json={"inputs": text})
            resp.raise_for_status()
            data = resp.json()

            # Normalise: TEI may return a list-of-dicts or list-of-list-of-dicts
            if data and isinstance(data[0], list):
                data = data[0]

            return {int(item["index"]): float(item["value"]) for item in data}

        except httpx.HTTPStatusError as exc:
            raise SparseEmbeddingError(
                f"SPLADE server returned HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise SparseEmbeddingError(f"SPLADE embed_sparse failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()
