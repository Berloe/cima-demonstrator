"""TEINLIAdapter → NLIPort — sequence classification via TEI /predict endpoint."""
from __future__ import annotations

import logging

import httpx

from cima_demo.domain.errors import NLIUnavailableError
from cima_demo.domain.ports import NLIPort

log = logging.getLogger(__name__)

# Labels that TEI NLI models may return (case-insensitive match)
_LABEL_CONTRADICTION = "contradiction"
_LABEL_ENTAILMENT = "entailment"
_LABEL_NEUTRAL = "neutral"

# Canonical output strings
CONTRADICTION = "CONTRADICTION"
ENTAILMENT = "ENTAILMENT"
NEUTRAL = "NEUTRAL"


class TEINLIAdapter(NLIPort):
    """Natural Language Inference via TEI sequence-classification endpoint.

    Calls POST /predict with a premise/hypothesis pair and returns the
    highest-scoring label normalised to ENTAILMENT | NEUTRAL | CONTRADICTION.

    Compatible with cross-encoder NLI models served by text-embeddings-inference
    (e.g. cross-encoder/nli-deberta-v3-small, MoritzLaurer/deberta-v3-base-zeroshot-v1).

    Request:  {"inputs": [["premise", "hypothesis"]]}
    Response: [[{"label": "CONTRADICTION", "score": 0.9}, ...]]
    """

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def classify(self, text_a: str, text_b: str) -> str:
        """Return 'ENTAILMENT' | 'NEUTRAL' | 'CONTRADICTION'.

        text_a is treated as premise, text_b as hypothesis.
        Raises NLIUnavailableError on HTTP/network errors.
        """
        try:
            resp = await self._client.post(
                "/predict",
                json={"inputs": [[text_a, text_b]]},
            )
            if resp.status_code >= 500:
                raise NLIUnavailableError(f"TEI NLI returned {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except NLIUnavailableError:
            raise
        except httpx.TimeoutException as exc:
            raise NLIUnavailableError(f"TEI NLI timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise NLIUnavailableError(f"TEI NLI HTTP error: {exc}") from exc

        # data shape: [[{"label": "...", "score": f}, ...]]
        try:
            scores: list[dict] = data[0]
            best = max(scores, key=lambda x: x["score"])
            label: str = best["label"].upper()
        except (KeyError, IndexError, TypeError) as exc:
            raise NLIUnavailableError(f"TEI NLI unexpected response shape: {data}") from exc

        if _LABEL_CONTRADICTION in label.lower():
            return CONTRADICTION
        if _LABEL_ENTAILMENT in label.lower():
            return ENTAILMENT
        return NEUTRAL

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
