"""TEIRerankerAdapter → RerankerPort (KIMA_Infrastructure_Layer_v0.6 §3.9)."""
from __future__ import annotations

import logging
import time as _time

import httpx

from cima_demo.api.settings import Settings, get_settings
from cima_demo.domain.errors import RerankerUnavailableError
from cima_demo.domain.ports import RerankerPort
from cima_demo.domain.value_objects import RerankResult

log = logging.getLogger(__name__)


class TEIRerankerAdapter(RerankerPort):
    """Cross-encoder reranking via separate TEI endpoint.

    POST /rerank → [{"index": i, "score": f}, ...]
    Sorted by score DESC, sliced to top_n.
    Raises RerankerUnavailableError → triggers graceful degradation in RetrievalOrchestrator.

    Resilience features:
    - Batch cap (max_batch): prevents OOM on the TEI pod by never sending more than
      max_batch texts in a single call.  Texts beyond the cap are silently dropped —
      the caller (RetrievalOrchestrator) already slices to rerank_top_n before calling.
    - Circuit breaker: after circuit_threshold consecutive 5xx / timeout failures the
      adapter stops calling TEI for circuit_open_secs seconds and immediately raises
      RerankerUnavailableError.  This prevents cascading load on a restarting pod and
      lets OKD complete the CrashLoopBackOff recovery undisturbed.
    - Connection limit: httpx pool capped at max_connections to avoid socket exhaustion
      when many parallel retrieval calls stack up during GAIA benchmarks.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 20.0,
        max_batch: int = 32,
        circuit_threshold: int = 3,
        circuit_open_secs: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._max_batch = max_batch
        self._circuit_threshold = circuit_threshold
        self._circuit_open_secs = circuit_open_secs
        # Mutable circuit-breaker state — all mutations happen in the same event loop.
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None

    # ── Circuit breaker helpers ────────────────────────────────────────────────

    def _is_circuit_open(self) -> bool:
        if self._circuit_opened_at is None:
            return False
        age = _time.monotonic() - self._circuit_opened_at
        if age > self._circuit_open_secs:
            # Transition to half-open: allow one probe request through.
            log.info(
                "TEIReranker circuit breaker: half-open after %.0fs — probing",
                age,
            )
            self._circuit_opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        if self._consecutive_failures > 0 or self._circuit_opened_at is not None:
            log.info("TEIReranker circuit breaker: closed (successful response)")
        self._consecutive_failures = 0
        self._circuit_opened_at = None

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_threshold and self._circuit_opened_at is None:
            self._circuit_opened_at = _time.monotonic()
            log.warning(
                "TEIReranker circuit breaker OPENED after %d consecutive failures — "
                "bypassing reranker for %.0fs",
                self._consecutive_failures,
                self._circuit_open_secs,
            )

    # ── RerankerPort ──────────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        texts: list[str],
        top_n: int,
        truncate: bool = True,
    ) -> list[RerankResult]:
        if not texts:
            return []

        # Fast-fail when circuit is open (reranker pod is recovering).
        if self._is_circuit_open():
            raise RerankerUnavailableError(
                f"TEI reranker circuit breaker open ({self._circuit_open_secs:.0f}s cooldown)"
            )

        # Batch cap: cross-encoder VRAM scales with N×seq_len² — hard cap prevents OOM.
        if len(texts) > self._max_batch:
            log.debug(
                "TEIReranker: capping batch from %d → %d texts",
                len(texts), self._max_batch,
            )
            texts = texts[: self._max_batch]

        try:
            resp = await self._client.post(
                "/rerank",
                json={"query": query, "texts": texts, "truncate": truncate},
                timeout=get_settings().tei_reranker_timeout
            )
            if resp.status_code >= 500:
                self._record_failure()
                raise RerankerUnavailableError(f"TEI reranker returned {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            results = sorted(data, key=lambda x: x["score"], reverse=True)[:top_n]
            self._record_success()
            return [RerankResult(index=r["index"], score=r["score"]) for r in results]
        except RerankerUnavailableError as e:
            log.error(f"TEI reranker: {e}")
            raise
        except httpx.TimeoutException as exc:
            self._record_failure()
            log.error(f"TEI reranker timeout: {exc}")
            raise RerankerUnavailableError(f"TEI reranker timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            self._record_failure()
            log.error(f"TEI reranker error: {exc}")
            raise RerankerUnavailableError(f"TEI reranker error: {exc}") from exc

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
