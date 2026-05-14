"""DependencyIdsGeometricExpander → GeometricExpansionPort (T-07, §3.10)."""
from __future__ import annotations

import asyncio
import math

from cima_demo.domain.entities import CItem
from cima_demo.domain.ports import CItemStorePort, EmbeddingPort, GeometricExpansionPort

# APP-D-08: minimum cosine similarity between a neighbor and any seed to keep it.
# Overridable via constructor; also settable via KIMA_BRIDGE_THRESHOLD.
_DEFAULT_BRIDGE_THRESHOLD = 0.15


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0.0 and nb > 0.0 else 0.0


class DependencyIdsGeometricExpander(GeometricExpansionPort):
    """Phase 1 geometric expansion via dependency_ids (APP-INV-27).

    Delegates structural neighbor fetch to CItemStorePort.fetch_neighbors().

    APP-D-08 — semantic bridge score (optional):
        When *embedding_port* is provided, the expanded neighbors are filtered
        by cosine similarity against the seed embeddings.  Any neighbor whose
        maximum similarity to any seed falls below *bridge_threshold* is
        dropped, and the survivors are returned sorted by bridge score
        descending (most relevant first).

        When *embedding_port* is None the behaviour is identical to Phase 1:
        all structural neighbors are returned in fetch order.

    Swappable for Phase 2 semantic expander without touching RetrievalOrchestrator.
    """

    def __init__(
        self,
        citem_store: CItemStorePort,
        embedding_port: EmbeddingPort | None = None,
        bridge_threshold: float = _DEFAULT_BRIDGE_THRESHOLD,
        backward_max: int = 500,
    ) -> None:
        self._store = citem_store
        self._embed = embedding_port
        self._threshold = bridge_threshold
        self._backward_max = backward_max

    async def expand(
        self,
        seeds: list[CItem],
        conversation_id: str,
        exclude_ids: set[str],
    ) -> list[CItem]:
        if not seeds:
            return []
        seed_ids = [c.citem_id for c in seeds]
        all_excluded = exclude_ids | set(seed_ids)
        neighbors = await self._store.fetch_neighbors(
            seed_ids=seed_ids,
            conversation_id=conversation_id,
            exclude_ids=all_excluded,
            backward_max=self._backward_max,
        )

        if not neighbors or self._embed is None:
            return neighbors

        # ── APP-D-08: semantic bridge filter ──────────────────────────────────
        seed_texts = [s.content for s in seeds]
        neighbor_texts = [n.content for n in neighbors]
        all_vecs = await self._embed.embed_batch(seed_texts + neighbor_texts)
        seed_vecs = all_vecs[: len(seed_texts)]
        neighbor_vecs = all_vecs[len(seed_texts) :]

        scored: list[tuple[float, CItem]] = []
        for neighbor, nvec in zip(neighbors, neighbor_vecs, strict=False):
            bridge = max(_cosine(nvec, svec) for svec in seed_vecs)
            if bridge >= self._threshold:
                scored.append((bridge, neighbor))

        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored]
