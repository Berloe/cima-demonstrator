"""Detached geometry service for the CIMA Demonstrator.

The service has no authority over the turn runtime. It computes structural hints
(core / bridge / cluster labels) asynchronously and persists them in its own
bounded context storage. The demonstrator may consume the latest hints, but the
turn loop never waits for geometry completion.
"""
from __future__ import annotations

import asyncio
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cima_demo.demo.contracts import GeometryClusterState, GeometryItemState, GeometryRunReport
from cima_demo.domain.entities import CItem, SummaryNode
from cima_demo.domain.ports import CItemStorePort, EmbeddingPort, GeometricExpansionPort, RelDBPort


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dim = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(dim))
    na = math.sqrt(sum(float(v) * float(v) for v in a[:dim]))
    nb = math.sqrt(sum(float(v) * float(v) for v in b[:dim]))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = min(len(v) for v in vectors if v)
    if dim <= 0:
        return []
    acc = [0.0] * dim
    for vec in vectors:
        for idx in range(dim):
            acc[idx] += float(vec[idx])
    out = [value / float(len(vectors)) for value in acc]
    norm = math.sqrt(sum(v * v for v in out))
    if norm > 0.0:
        out = [v / norm for v in out]
    return out


def _safe_label(text: str, fallback: str) -> str:
    words = [w.strip(" ,.;:()[]{}\n\t") for w in text.split() if w.strip(" ,.;:()[]{}\n\t")]
    label = " ".join(words[:10]).strip()
    return label or fallback


@dataclass(slots=True)
class _Cluster:
    cluster_id: str
    item_ids: list[str]
    centroid: list[float]


class NoOpGeometricExpander(GeometricExpansionPort):
    """Demo-mode expander: geometry must not have turn authority."""

    async def expand(self, seeds: list[CItem], conversation_id: str, exclude_ids: set[str]) -> list[CItem]:
        return []


class DemoGeometryService:
    """Asynchronous geometry service with explicit, limited outputs."""

    def __init__(
        self,
        *,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        embedding_port: EmbeddingPort | None = None,
        algo_version: str = "geom_v1",
        k_max: int = 4,
    ) -> None:
        self._db = rel_db
        self._store = citem_store
        self._embed = embedding_port
        self._algo_version = algo_version
        self._k_max = max(1, k_max)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def schedule_recompute(self, conversation_id: str, *, reason: str = "context_refresh") -> None:
        if not conversation_id:
            return
        existing = self._tasks.get(conversation_id)
        if existing is not None and not existing.done():
            return
        self._tasks[conversation_id] = asyncio.create_task(self._guarded_recompute(conversation_id, reason))

    async def _guarded_recompute(self, conversation_id: str, reason: str) -> None:
        lock = self._locks[conversation_id]
        async with lock:
            await self.recompute(conversation_id=conversation_id, reason=reason)

    async def recompute(self, *, conversation_id: str, reason: str = "manual") -> GeometryRunReport:
        items = await self._store.fetch_by_conversation(conversation_id, scope_status="active")
        item_ids = [item.citem_id for item in items]
        vectors = await self._store.fetch_dense_vectors(item_ids) if item_ids else {}
        usable = [item for item in items if item.citem_id in vectors and vectors[item.citem_id]]
        if not usable:
            run = GeometryRunReport(
                run_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                reason=reason,
                algo_version=self._algo_version,
                n_items=0,
                cluster_count=0,
                core_count=0,
                bridge_count=0,
            )
            await self._db.save_geometry_run(run.to_dict())
            return run

        vector_map = {item.citem_id: vectors[item.citem_id] for item in usable}
        k = min(max(1, round(math.sqrt(len(usable)))), self._k_max, len(usable))
        centroids = self._initial_centroids(usable, vector_map, k)
        assignments = self._assign_items(usable, vector_map, centroids)
        for _ in range(3):
            centroids = self._recompute_centroids(assignments, vector_map)
            assignments = self._assign_items(usable, vector_map, centroids)

        clusters = self._materialise_clusters(assignments, centroids)
        summaries = await self._db.fetch_pyramid_tops(conversation_id, limit=8)
        summary_by_cluster = self._match_summaries(clusters, summaries)
        run_id = str(uuid.uuid4())
        bridge_count = 0
        core_count = 0

        for cluster in clusters:
            ordered = sorted(
                cluster.item_ids,
                key=lambda ref_id: _cosine(vector_map[ref_id], cluster.centroid),
                reverse=True,
            )
            medoid_id = ordered[0]
            core_cut = max(1, math.ceil(len(ordered) * 0.2))
            core_ids = set(ordered[:core_cut])
            medoid_item = next((item for item in usable if item.citem_id == medoid_id), None)
            label = _safe_label(medoid_item.content if medoid_item is not None else "", cluster.cluster_id)
            summary_id = summary_by_cluster.get(cluster.cluster_id)
            await self._db.save_geometry_cluster_state(
                GeometryClusterState(
                    conversation_id=conversation_id,
                    cluster_id=cluster.cluster_id,
                    run_id=run_id,
                    mass=float(len(cluster.item_ids)) / float(len(usable)),
                    medoid_ref_id=medoid_id,
                    summary_id=summary_id,
                    label=label,
                ).to_dict()
            )
            for ref_id in cluster.item_ids:
                sims = [(_cosine(vector_map[ref_id], c.centroid), c.cluster_id) for c in clusters]
                sims.sort(key=lambda pair: pair[0], reverse=True)
                top1_score, top1_id = sims[0]
                top2_score, top2_id = sims[1] if len(sims) > 1 else (0.0, None)
                margin = top1_score - top2_score
                ref_item = next(item for item in usable if item.citem_id == ref_id)
                cross_dep = any(
                    dep in vector_map and self._cluster_for_ref(clusters, dep) != top1_id
                    for dep in ref_item.dependency_ids
                )
                is_bridge = bool(cross_dep or (top2_id is not None and margin <= 0.10))
                if is_bridge:
                    bridge_count += 1
                is_core = ref_id in core_ids
                if is_core:
                    core_count += 1
                state = GeometryItemState(
                    conversation_id=conversation_id,
                    ref_kind="citem",
                    ref_id=ref_id,
                    run_id=run_id,
                    cluster_top1=top1_id,
                    cluster_top2=top2_id,
                    w1=max(0.0, min(1.0, top1_score)),
                    w2=max(0.0, min(1.0, top2_score)) if top2_id is not None else None,
                    margin=margin,
                    is_core=is_core,
                    is_bridge_candidate=is_bridge,
                    centrality=max(0.0, min(1.0, top1_score)),
                    label=label,
                )
                await self._db.save_geometry_item_state(state.to_dict())
                await self._store.update_field(ref_id, "geom_cluster_top1", top1_id)
                await self._store.update_field(ref_id, "geom_cluster_top2", top2_id)
                await self._store.update_field(ref_id, "geom_is_core", is_core)
                await self._store.update_field(ref_id, "geom_is_bridge_candidate", is_bridge)
                await self._store.update_field(ref_id, "geom_label", label)

        run = GeometryRunReport(
            run_id=run_id,
            conversation_id=conversation_id,
            reason=reason,
            algo_version=self._algo_version,
            n_items=len(usable),
            cluster_count=len(clusters),
            core_count=core_count,
            bridge_count=bridge_count,
        )
        await self._db.save_geometry_run(run.to_dict())
        return run

    async def get_item_hints(self, *, conversation_id: str, ref_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = await self._db.load_geometry_item_states(conversation_id, ref_ids)
        return {str(row["ref_id"]): row for row in rows}


    async def load_all_item_hints(self, *, conversation_id: str) -> list[dict[str, Any]]:
        return await self._db.load_geometry_item_states(conversation_id)

    async def get_cluster_hints(self, *, conversation_id: str) -> list[dict[str, Any]]:
        return await self._db.load_geometry_cluster_states(conversation_id)

    async def purge_conversation(self, conversation_id: str) -> None:
        await self._db.delete_geometry_conversation(conversation_id)

    def _cluster_for_ref(self, clusters: list[_Cluster], ref_id: str) -> str | None:
        for cluster in clusters:
            if ref_id in cluster.item_ids:
                return cluster.cluster_id
        return None

    def _initial_centroids(self, items: list[CItem], vector_map: dict[str, list[float]], k: int) -> list[list[float]]:
        ordered = sorted(items, key=lambda item: item.citem_id)
        chosen: list[list[float]] = [vector_map[ordered[0].citem_id]]
        while len(chosen) < k:
            best_item: list[float] | None = None
            best_score = -1.0
            for item in ordered:
                vec = vector_map[item.citem_id]
                score = min(1.0 - _cosine(vec, centroid) for centroid in chosen)
                if score > best_score:
                    best_item = vec
                    best_score = score
            if best_item is None:
                break
            chosen.append(best_item)
        return chosen

    def _assign_items(self, items: list[CItem], vector_map: dict[str, list[float]], centroids: list[list[float]]) -> dict[str, list[str]]:
        assignments: dict[str, list[str]] = {f"c_{idx+1:03d}": [] for idx in range(len(centroids))}
        for item in items:
            sims = [(_cosine(vector_map[item.citem_id], centroid), idx) for idx, centroid in enumerate(centroids)]
            _, best_idx = max(sims, key=lambda pair: pair[0])
            assignments[f"c_{best_idx+1:03d}"].append(item.citem_id)
        return assignments

    def _recompute_centroids(self, assignments: dict[str, list[str]], vector_map: dict[str, list[float]]) -> list[list[float]]:
        centroids: list[list[float]] = []
        for cluster_id in sorted(assignments.keys()):
            vectors = [vector_map[ref_id] for ref_id in assignments[cluster_id] if ref_id in vector_map]
            if not vectors:
                continue
            centroids.append(_mean(vectors))
        return centroids

    def _materialise_clusters(self, assignments: dict[str, list[str]], centroids: list[list[float]]) -> list[_Cluster]:
        clusters: list[_Cluster] = []
        for idx, cluster_id in enumerate(sorted(assignments.keys())):
            item_ids = assignments[cluster_id]
            if not item_ids:
                continue
            centroid = centroids[idx] if idx < len(centroids) else []
            clusters.append(_Cluster(cluster_id=cluster_id, item_ids=item_ids, centroid=centroid))
        return clusters

    def _match_summaries(self, clusters: list[_Cluster], summaries: list[SummaryNode]) -> dict[str, str | None]:
        matched: dict[str, str | None] = {}
        available = list(summaries)
        for cluster in clusters:
            summary_id: str | None = None
            members = set(cluster.item_ids)
            best_overlap = -1
            for summary in available:
                overlap = len(members.intersection(set(summary.origin_citem_ids)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    summary_id = summary.node_id
            matched[cluster.cluster_id] = summary_id
        return matched
