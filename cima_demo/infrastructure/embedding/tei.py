"""TEIAdapter → EmbeddingPort (KIMA_Infrastructure_Layer_v0.6 §3.4)."""
from __future__ import annotations

import logging
from typing import cast

import httpx

from cima_demo.domain.errors import EmbeddingUnavailableError
from cima_demo.domain.ports import EmbeddingPort

log = logging.getLogger(__name__)


class TEIAdapter(EmbeddingPort):
    """Text Embeddings Inference (TEI) adapter.

    POST /embed → list[float]  (single text)
    POST /embed → list[list[float]]  (batch)
    GET  /info  → dim detection
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def embed(self, text: str) -> list[float]:
        try:
            resp = await self._client.post(
                "/embed", json={"inputs": text, "truncate": True}
            )
            if resp.status_code >= 500:
                raise EmbeddingUnavailableError(f"TEI returned {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            # TEI returns [[vec]] for single input
            if isinstance(data, list) and data and isinstance(data[0], list):
                return cast(list[float], data[0])
            return cast(list[float], data)
        except EmbeddingUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(f"TEI embed error: {exc}") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.post(
                "/embed", json={"inputs": texts, "truncate": True}
            )
            if resp.status_code >= 500:
                raise EmbeddingUnavailableError(f"TEI returned {resp.status_code}")
            resp.raise_for_status()
            return cast(list[list[float]], resp.json())
        except EmbeddingUnavailableError:
            raise
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(f"TEI embed_batch error: {exc}") from exc

    async def get_dim(self) -> int:
        """Return embedding dimension from /info endpoint."""
        try:
            resp = await self._client.get("/info")
            resp.raise_for_status()
            info = resp.json()
            return info.get("dim") or 768
        except Exception:
            return 768  # Default for nomic-embed-text

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
