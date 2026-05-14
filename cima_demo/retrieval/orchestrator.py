"""RetrievalOrchestrator — 4-phase RAG pipeline (APP-INV-22, INV-22)."""
from __future__ import annotations

import asyncio
import logging
import time

from cima_demo.domain.entities import CItem, ContextPack
from cima_demo.domain.errors import RerankerUnavailableError
from cima_demo.domain.operations import (
    adaptive_pre_rerank_k,
    adaptive_rerank_k,
    adaptive_retry_k,
    apply_cvs_density_scoring,
    compute_traceability_density,
    episodic_active_filter,
    episodic_flagged_filter,
    evaluate_coverage,
    filter_q3_by_local_relevance,
    global_active_filter,
    greedy_select,
    annotate_bridge_scores, compute_mean_sim_to_pool,
    percentile, is_conflict_candidate, is_geometric_candidate,
    extract_query_facets, slot_caps_for_query_type,
)
from cima_demo.domain.ports import (
    CItemStorePort,
    GeometricExpansionPort,
    RelDBPort,
    RerankerPort,
)
from cima_demo.domain.value_objects import (
    BudgetReport,
    CognitivePhase,
    ContextBudget,
    CoverageReport,
    CVSWeights,
    PhasePolicy,
    RecallSource,
    RetrievalPlan,
    ScoredCItem, BridgePolicy, DirectEvidenceStrategy,
)

from cima_demo.observability import emit_retrieval_span

log = logging.getLogger(__name__)


def _schedule_background(coro: object, /, *args: object, **kwargs: object) -> object:
    """Schedule best-effort background work without leaking test coroutines.

    Some tests patch ``asyncio.create_task`` with a plain mock to assert that the
    scheduling path was exercised. When that happens, the coroutine passed to the
    mock would normally remain un-awaited and trigger noisy RuntimeWarnings during
    garbage collection. We treat those patched calls as a non-runtime scheduler
    and explicitly close the coroutine after handing it to the mock. Real
    ``asyncio.Task`` instances are left untouched.
    """
    task = asyncio.create_task(coro, *args, **kwargs)
    if asyncio.iscoroutine(coro) and not isinstance(task, asyncio.Task):
        coro.close()
    return task


class RetrievalOrchestrator:
    """4-phase RAG pipeline:
    1. Multi-source recall (3 parallel hybrid queries — episodic + global + flagged)
    2. RRF merge across all three ranked lists + CrossEncoder rerank
    3. Geometric expansion via GeometricExpansionPort (APP-INV-27)
    4. Slot assembly + coverage evaluation + adaptive retry

    APP-INV-27: ONLY expands via GeometricExpansionPort — never calls fetch_neighbors directly.
    """

    def __init__(
        self,
        citem_store: CItemStorePort,
        geometric_expander: GeometricExpansionPort,
        reranker_port: RerankerPort,
        rel_db: RelDBPort,
        cvs_weights: CVSWeights | None = None,
        phase_policy: PhasePolicy | None = None,
        global_min_score: float = 0.0,
        colbert_port: RerankerPort | None = None,
    ) -> None:
        self._cstore = citem_store
        self._expander = geometric_expander
        self._reranker = reranker_port
        self._db = rel_db
        self._cvs_weights = cvs_weights or CVSWeights.default()
        self._phase_policy = phase_policy or PhasePolicy.default()
        # Minimum similarity score for global recall results (Q2).
        # Items below this threshold are dropped before RRF — prevents unrelated
        # knowledge domains from polluting the current conversation context.
        # 0.0 = disabled (all global items pass through).
        self._global_min_score = global_min_score
        # Optional ColBERT late-interaction reranker. When set, applied as a
        # second reranking pass after the CrossEncoder on the top-K candidates.
        # None = disabled (CrossEncoder-only pipeline).
        self._colbert: RerankerPort | None = colbert_port
        self._rag_config: object | None = None

    def set_rag_config(self, config: object | None) -> None:
        self._rag_config = config

    @staticmethod
    def ensure_dense_signal(items: list[ScoredCItem]) -> list[ScoredCItem]:
        for sc in items:
            if sc.dense_score is None:
                sc.dense_score = sc.score
        return items

    @staticmethod
    def reciprocal_rank_fusion_scored(
            pools: list[list[ScoredCItem]],
            k: int = 60,
    ) -> list[ScoredCItem]:
        fused: dict[str, ScoredCItem] = {}
        acc: dict[str, float] = {}

        for pool in pools:
            for rank, sc in enumerate(pool, start=1):
                cid = sc.citem.citem_id
                if cid not in fused:
                    fused[cid] = sc
                    acc[cid] = 0.0
                acc[cid] += 1.0 / (k + rank)

        out: list[ScoredCItem] = []
        for cid, sc in fused.items():
            sc.rrf_score = acc[cid]
            out.append(sc)

        out.sort(
            key=lambda x: (
                x.rrf_score if x.rrf_score is not None else float("-inf"),
                x.dense_score if x.dense_score is not None else float("-inf"),
                x.citem.citem_id,
            ),
            reverse=True,
        )
        return out

    @staticmethod
    def derive_bridge_policy(
            candidates: list[ScoredCItem],
            *,
            alpha: float = 0.5,
            relax: bool = False,
    ) -> BridgePolicy:
        rerank_scores = [sc.rerank_score for sc in candidates if sc.rerank_score is not None]
        bridge_scores = [sc.bridge_score for sc in candidates if sc.bridge_score is not None]

        if not rerank_scores or not bridge_scores:
            return BridgePolicy.disabled(alpha=alpha)

        anchor_floor = percentile(rerank_scores, 0.25)
        bridge_floor = percentile(bridge_scores, 0.25)

        if relax:
            anchor_floor = percentile(rerank_scores, 0.10)
            bridge_floor = percentile(bridge_scores, 0.10)

        return BridgePolicy(
            enabled=True,
            alpha=alpha,
            anchor_floor=anchor_floor,
            bridge_floor=bridge_floor,
            max_bridge_redundancy=0.92,
            max_consecutive_low_bridge=3,
        )


    async def retrieve(
            self,
            query: str,
            plan: RetrievalPlan,
            conversation_id: str,
            budget: ContextBudget,
            active_procedure_id: str | None = None,
            retry_count: int = 0,
            phase: str = CognitivePhase.IDLE,
            exclude_ids: set[str] | None = None,
    ) -> tuple[ContextPack, BudgetReport, CoverageReport]:
        """Execute full RAG pipeline.

        Reglas:
        - anchor por rerank_score;
        - bridge por bridge_score = alpha * dense_norm + (1 - alpha) * rerank_norm;
        - sin reranker, no hay bridge lane;
        - q3/conflicts y geometric expansion quedan fuera del bridge lane.
        """
        t0 = time.monotonic()
        _exclude = exclude_ids or set()
        bridge_alpha = getattr(plan, "bridge_alpha", 0.5)

        # ── Phase 1: Multi-source recall ──────────────────────────────────────
        q1, q2, q3 = await self._multi_source_recall(query, plan, conversation_id)
        before_rerank = len(q1) + len(q2) + len(q3)

        q1 = self.ensure_dense_signal(q1)
        q2 = self.ensure_dense_signal(q2)
        q3 = self.ensure_dense_signal(q3)

        # ── Phase 2: RRF merge (Q1+Q2 only) + rerank ──────────────────────────
        merged = self.reciprocal_rank_fusion_scored([q1, q2])

        reranker_available = True
        # Pre-rerank adaptive-k: trim low-RRF candidates before the expensive
        # CrossEncoder call.  Saves ~40-60% rerank latency on iGPU when many
        # recall items have negligible RRF scores.
        rrf_scores = [sc.rrf_score for sc in merged if sc.rrf_score is not None]
        pre_k = adaptive_pre_rerank_k(rrf_scores, plan.rerank_top_n)
        if pre_k < len(merged):
            log.debug(
                "adaptive_pre_rerank_k: %d → %d (plan.rerank_top_n=%d)",
                len(merged), pre_k, plan.rerank_top_n,
            )
            merged = merged[:pre_k]
        try:
            merged = await self._rerank(query, merged, plan.rerank_top_n)
            for sc in merged:
                setattr(sc, "rerank_verified", True)
            # Adaptive-k: trim to natural score cliff instead of fixed rerank_top_n.
            rerank_scores = [sc.rerank_score for sc in merged if sc.rerank_score is not None]
            effective_k = adaptive_rerank_k(rerank_scores, plan.rerank_top_n)
            if effective_k < len(merged):
                log.debug(
                    "adaptive_rerank_k: %d → %d (plan.rerank_top_n=%d)",
                    len(merged), effective_k, plan.rerank_top_n,
                )
                merged = merged[:effective_k]
        except RerankerUnavailableError:
            reranker_available = False
            log.warning("Reranker unavailable — bridge lane disabled; using RRF order")
            merged = merged[: plan.rerank_top_n]
            for sc in merged:
                if sc.rerank_score is None:
                    sc.rerank_score = (
                        sc.rrf_score
                        if sc.rrf_score is not None
                        else sc.dense_score
                        if sc.dense_score is not None
                        else sc.score
                    )
                setattr(sc, "rerank_verified", False)

        after_rerank = len(merged)

        # ── Phase 2b: ColBERT late-interaction reranking (optional) ──────────
        # Applied AFTER CrossEncoder adaptive trim so ColBERT sees only the
        # high-quality shortlist, not the full RRF-merged pool.
        if self._colbert is not None and merged:
            try:
                merged = await self._late_interaction_rerank(query, merged, len(merged))
                log.debug("ColBERT rerank: %d candidates re-ordered", len(merged))
            except RerankerUnavailableError:
                log.warning("ColBERT unavailable — keeping CrossEncoder order")

        # ── Phase 3: Geometric expansion ─────────────────────────────────────
        expanded: list[CItem] = []
        if plan.geometric_expand:
            seeds = [sc.citem for sc in merged[: plan.geometric_seeds_k]]
            seed_ids = {c.citem_id for c in seeds}
            merged_ids = {sc.citem.citem_id for sc in merged}
            expanded = await self._expander.expand(
                seeds=seeds,
                conversation_id=conversation_id,
                exclude_ids=seed_ids | merged_ids,
            )

        # ── Phase 3b: Candidate assembly ─────────────────────────────────────
        # Facets are extracted early so the q3 gate can use them.
        query_facets = extract_query_facets(query)

        # Area A: q3 gate — flagged items enter only if locally relevant.
        # They still route to conflicts slot in greedy_select; the gate prevents
        # unrelated flagged items from consuming the 10% conflicts budget.
        _q3_total = len(q3)
        q3_relevant, _ = filter_q3_by_local_relevance(q3, query_facets, merged[:3])
        _q3_relevant = len(q3_relevant)
        if _q3_total > 0:
            log.debug(
                "q3_gate: total=%d relevant=%d dropped=%d",
                _q3_total, _q3_relevant, _q3_total - _q3_relevant,
            )

        all_candidates: list[ScoredCItem] = []
        seen_ids: set[str] = set()

        def _append(sc: ScoredCItem) -> None:
            cid = sc.citem.citem_id
            if cid in seen_ids or cid in _exclude:
                return
            all_candidates.append(sc)
            seen_ids.add(cid)

        for sc in merged:
            _append(sc)

        for sc in sorted(
                q3_relevant,
                key=lambda x: (
                        x.score if x.score is not None else float("-inf"),
                        x.citem.citem_id,
                ),
                reverse=True,
        ):
            _append(sc)

        for item in expanded:
            _append(
                ScoredCItem(
                    citem=item,
                    score=0.1,
                    provenance=RecallSource.GEOMETRIC,
                    dense_score=None,
                    rrf_score=None,
                    rerank_score=None,
                    cvs_score=None,
                    value_density=None,
                    bridge_score=None,
                )
            )

        after_expand = len(all_candidates)

        # ── Phase 3c: CVS + density scoring ───────────────────────────────────
        _cvs_w = self._cvs_weights
        if self._rag_config is not None:
            _cvs_w = CVSWeights(
                alpha=getattr(self._rag_config, "cvs_alpha", _cvs_w.alpha),
                beta=getattr(self._rag_config, "cvs_beta", _cvs_w.beta),
                gamma=getattr(self._rag_config, "cvs_gamma", _cvs_w.gamma),
            )
        all_candidates = apply_cvs_density_scoring(
            all_candidates,
            phase=phase,
            cvs_weights=_cvs_w,
            phase_policy=self._phase_policy,
        )

        # ── Phase 3d: Bridge policy ───────────────────────────────────────────
        bridge_eligible = [
            sc for sc in all_candidates
            if not is_conflict_candidate(sc)
               and not is_geometric_candidate(sc)
               and sc.dense_score is not None
               and sc.rerank_score is not None
               and bool(getattr(sc, "rerank_verified", False))
        ]

        if reranker_available and bridge_eligible:
            # Fetch dense vectors for mean_sim_to_pool (pressure benchmark 3-signal formula)
            _bridge_ids = [sc.citem.citem_id for sc in bridge_eligible]
            try:
                _vecs = await self._citem_store.fetch_dense_vectors(_bridge_ids)
                _mean_sim_map = compute_mean_sim_to_pool(_vecs, _bridge_ids)
            except Exception:
                log.debug("fetch_dense_vectors failed — falling back to 2-signal bridge")
                _mean_sim_map = {}
            _bw = {}
            if self._rag_config is not None:
                _bw = dict(
                    w_dense=getattr(self._rag_config, "bridge_w_dense", 0.4),
                    w_rerank=getattr(self._rag_config, "bridge_w_rerank", 0.4),
                    w_centrality=getattr(self._rag_config, "bridge_w_centrality", 0.2),
                )
            annotate_bridge_scores(bridge_eligible, mean_sim_map=_mean_sim_map, **_bw)
            bridge_policy = self.derive_bridge_policy(
                bridge_eligible,
                alpha=bridge_alpha,
                relax=(retry_count > 0),
            )
            direct_strategy = DirectEvidenceStrategy.ANCHOR_BRIDGE_INTERLEAVED
        else:
            bridge_policy = BridgePolicy.disabled(alpha=bridge_alpha)
            direct_strategy = DirectEvidenceStrategy.CVS_DENSITY

        # Area C: packing profile — narrower slot caps for closed/precise queries.
        _slot_caps = slot_caps_for_query_type(plan.query_type)

        # ── Phase 4: Greedy slot assembly ─────────────────────────────────────
        pack, budget_report = greedy_select(
            candidates=all_candidates,
            budget=budget,
            active_procedure_id=active_procedure_id,
            query_facets=query_facets,
            direct_strategy=direct_strategy,
            bridge_policy=bridge_policy,
            slot_caps_override=_slot_caps,
        )

        # ── Traceability density Tᵈ ───────────────────────────────────────────
        td = compute_traceability_density(pack)
        _td_pool = (
                len(pack.protected_items)
                + len(pack.direct_evidence)
                + len(pack.bridge_evidence)
                + len(pack.global_summaries)
                + len(pack.conflicts)
        )
        if td < 0.85 and _td_pool >= 5:
            log.debug(
                "Tᵈ=%.3f < 0.85 for %s — low traceability density",
                td,
                conversation_id,
            )

        # ── Coverage evaluation ───────────────────────────────────────────────
        _cov_thr = plan.coverage_threshold
        if self._rag_config is not None:
            _cov_thr = getattr(self._rag_config, "coverage_threshold", _cov_thr)
        coverage = evaluate_coverage(query, pack, threshold=_cov_thr)
        pack.coverage_score = coverage.coverage_score

        # ── Adaptive retry ────────────────────────────────────────────────────
        if coverage.retry_recommended and retry_count == 0:
            retry_recall_k = adaptive_retry_k(
                plan.recall_top_k, coverage.coverage_score, _cov_thr,
            )
            log.debug(
                "Coverage %.2f < %.2f — adaptive retry with top_k=%d (base=%d)",
                coverage.coverage_score,
                _cov_thr,
                retry_recall_k,
                plan.recall_top_k,
            )
            retry_plan = RetrievalPlan(
                query_type=plan.query_type,
                recall_top_k=retry_recall_k,
                rerank_top_n=plan.rerank_top_n,
                geometric_expand=plan.geometric_expand,
                geometric_seeds_k=plan.geometric_seeds_k,
                coverage_threshold=plan.coverage_threshold,
            )
            return await self.retrieve(
                query=query,
                plan=retry_plan,
                conversation_id=conversation_id,
                budget=budget,
                active_procedure_id=active_procedure_id,
                retry_count=1,
                phase=phase,
                exclude_ids=exclude_ids,
            )

        # ── Telemetry ─────────────────────────────────────────────────────────
        latency_ms = int((time.monotonic() - t0) * 1000)
        _schedule_background(
            self._db.save_retrieval_telemetry(
                conversation_id=conversation_id,
                query_type=plan.query_type,
                recall_top_k=plan.recall_top_k,
                rerank_top_n=plan.rerank_top_n,
                items_selected=budget_report.items_selected,
                coverage_score=coverage.coverage_score,
                retry_count=retry_count,
                latency_ms=latency_ms,
                candidates_before_rerank=before_rerank,
                candidates_after_rerank=after_rerank,
                candidates_after_expand=after_expand,
                pack_total_tokens=budget_report.tokens_used,
                geometric_expand=plan.geometric_expand,
                reranker_available=reranker_available,
                traceability_density=td,
                q3_relevant_count=_q3_relevant,
                direct_strategy=direct_strategy.value,
                bridge_enabled=bridge_policy.enabled,
                bridge_alpha=bridge_policy.alpha,
                bridge_floor=bridge_policy.bridge_floor,
                bridge_candidates_eligible=len(bridge_eligible),
            ),
            name=f"rag_telemetry_{conversation_id}",
        )

        # H-17 (SPEC-7): emit completed RAG span with final stage sizes.
        emit_retrieval_span(
            query_type=plan.query_type,
            recall_top_k=plan.recall_top_k,
            rerank_top_n=plan.rerank_top_n,
            latency_ms=latency_ms,
            q1=len(q1),
            q2=len(q2),
            q3=_q3_total,
            after_rrf=after_rerank,  # merged size before geometric
            after_rerank=after_rerank,
            after_expand=after_expand,
            items_selected=budget_report.items_selected,
            coverage_score=coverage.coverage_score,
            reranker_available=reranker_available,
            bridge_enabled=bridge_policy.enabled,
            bridge_alpha=bridge_policy.alpha,
            retry_count=retry_count,
            conversation_id=conversation_id,
        )

        return pack, budget_report, coverage

    async def retrieve_multihop(
            self,
            query: str,
            sub_questions: list[str],
            plan: RetrievalPlan,
            conversation_id: str,
            budget: ContextBudget,
            active_procedure_id: str | None = None,
            retry_count: int = 0,
            phase: str = CognitivePhase.IDLE,
            exclude_ids: set[str] | None = None,
    ) -> tuple[ContextPack, BudgetReport, CoverageReport]:
        """Iterative multi-hop retrieval with signal-preserving scoring.

        Rules:
        - anchor lane: rerank_score
        - bridge lane: bridge_score = alpha*dense_norm + (1-alpha)*rerank_norm
        - only rerank-verified items are bridge-eligible
        - q3/conflicts and geometric-only items never enter bridge lane
        - one global greedy_select() applies slot caps over accumulated candidates
        """
        t0 = time.monotonic()
        _turn_exclude = exclude_ids or set()
        bridge_alpha = getattr(plan, "bridge_alpha", 0.5)

        # Empty decomposition: fallback to single-hop path
        if not sub_questions:
            return await self.retrieve(
                query=query,
                plan=plan,
                conversation_id=conversation_id,
                budget=budget,
                active_procedure_id=active_procedure_id,
                retry_count=retry_count,
                phase=phase,
                exclude_ids=exclude_ids,
            )

        n = max(1, len(sub_questions))
        iter_rerank_top_n = max(1, min(plan.rerank_top_n, max(10, plan.rerank_top_n // n)))

        all_candidates: list[ScoredCItem] = []
        seen_ids: set[str] = set(_turn_exclude)
        prev_seed_items: list[CItem] = []

        total_before_rerank = 0
        total_after_rerank = 0
        _q3_relevant_total = 0
        reranker_available = True

        def _append(sc: ScoredCItem) -> None:
            cid = sc.citem.citem_id
            if cid in seen_ids:
                return
            all_candidates.append(sc)
            seen_ids.add(cid)

        for sub_q in sub_questions:
            # ── hop 1: recall ──────────────────────────────────────────────────
            q1, q2, q3 = await self._multi_source_recall(sub_q, plan, conversation_id)
            total_before_rerank += len(q1) + len(q2) + len(q3)

            q1 = self.ensure_dense_signal(q1)
            q2 = self.ensure_dense_signal(q2)
            q3 = self.ensure_dense_signal(q3)

            # ── RRF (Q1+Q2 only) + rerank ─────────────────────────────────────
            merged = self.reciprocal_rank_fusion_scored([q1, q2])

            sub_rrf = [sc.rrf_score for sc in merged if sc.rrf_score is not None]
            sub_pre_k = adaptive_pre_rerank_k(sub_rrf, iter_rerank_top_n)
            if sub_pre_k < len(merged):
                merged = merged[:sub_pre_k]
            try:
                merged = await self._rerank(sub_q, merged, iter_rerank_top_n)
                for sc in merged:
                    setattr(sc, "rerank_verified", True)
                # Adaptive-k: trim to score cliff within each sub-question.
                sub_scores = [sc.rerank_score for sc in merged if sc.rerank_score is not None]
                eff_k = adaptive_rerank_k(sub_scores, iter_rerank_top_n)
                if eff_k < len(merged):
                    merged = merged[:eff_k]
            except RerankerUnavailableError:
                reranker_available = False
                log.warning("Reranker unavailable in multi-hop iteration — bridge lane disabled for fallback items")
                merged = merged[:iter_rerank_top_n]
                for sc in merged:
                    if sc.rerank_score is None:
                        sc.rerank_score = (
                            sc.rrf_score
                            if sc.rrf_score is not None
                            else sc.dense_score
                            if sc.dense_score is not None
                            else sc.score
                        )
                    setattr(sc, "rerank_verified", False)

            total_after_rerank += len(merged)

            # ── hop 2: geometric expansion with cross-iteration anchoring ─────
            expanded: list[CItem] = []
            if plan.geometric_expand:
                seeds = [sc.citem for sc in merged[: plan.geometric_seeds_k]]
                seed_ids = {c.citem_id for c in seeds}

                # Carry previous anchors into current expansion to bridge iterations
                anchor = [c for c in prev_seed_items if c.citem_id not in seed_ids][:5]
                combined_seeds = seeds + anchor

                expanded = await self._expander.expand(
                    seeds=combined_seeds,
                    conversation_id=conversation_id,
                    exclude_ids=seen_ids | {c.citem_id for c in combined_seeds},
                )

            # ── accumulate, preserving semantics ───────────────────────────────
            for sc in merged:
                _append(sc)

            # Q3: apply local relevance gate before accumulating.
            # Use per-hop sub_q facets so irrelevant conflicts don't accumulate.
            _hop_facets = extract_query_facets(sub_q)
            _q3_total_hop = len(q3)
            q3_relevant_hop, _ = filter_q3_by_local_relevance(q3, _hop_facets, merged[:3])
            _q3_relevant_total += len(q3_relevant_hop)
            if _q3_total_hop > 0:
                log.debug(
                    "q3_gate[hop=%s]: total=%d relevant=%d",
                    sub_q[:40], _q3_total_hop, len(q3_relevant_hop),
                )
            for sc in sorted(
                    q3_relevant_hop,
                    key=lambda x: (
                            x.score if x.score is not None else float("-inf"),
                            x.citem.citem_id,
                    ),
                    reverse=True,
            ):
                _append(sc)

            # Geometric-only items are support/fallback, never bridge-eligible
            for item in expanded:
                _append(
                    ScoredCItem(
                        citem=item,
                        score=0.1,
                        provenance=RecallSource.GEOMETRIC,
                        dense_score=None,
                        rrf_score=None,
                        rerank_score=None,
                        cvs_score=None,
                        value_density=None,
                        bridge_score=None,
                    )
                )

            # Next iteration anchoring uses top reranked items from this iteration
            prev_seed_items = [sc.citem for sc in merged[:5]]

        after_expand = len(all_candidates)

        # ── CVS rescoring + value density (signal-preserving) ────────────────
        _mh_cvs_w = self._cvs_weights
        if self._rag_config is not None:
            _mh_cvs_w = CVSWeights(
                alpha=getattr(self._rag_config, "cvs_alpha", _mh_cvs_w.alpha),
                beta=getattr(self._rag_config, "cvs_beta", _mh_cvs_w.beta),
                gamma=getattr(self._rag_config, "cvs_gamma", _mh_cvs_w.gamma),
            )
        all_candidates = apply_cvs_density_scoring(
            all_candidates,
            phase=phase,
            cvs_weights=_mh_cvs_w,
            phase_policy=self._phase_policy,
        )

        # ── Bridge policy over accumulated eligible pool ──────────────────────
        bridge_eligible = [
            sc
            for sc in all_candidates
            if not is_conflict_candidate(sc)
               and not is_geometric_candidate(sc)
               and sc.dense_score is not None
               and sc.rerank_score is not None
               and bool(getattr(sc, "rerank_verified", False))
        ]

        if bridge_eligible:
            _mh_ids = [sc.citem.citem_id for sc in bridge_eligible]
            try:
                _mh_vecs = await self._citem_store.fetch_dense_vectors(_mh_ids)
                _mh_sim_map = compute_mean_sim_to_pool(_mh_vecs, _mh_ids)
            except Exception:
                _mh_sim_map = {}
            _mh_bw = {}
            if self._rag_config is not None:
                _mh_bw = dict(
                    w_dense=getattr(self._rag_config, "bridge_w_dense", 0.4),
                    w_rerank=getattr(self._rag_config, "bridge_w_rerank", 0.4),
                    w_centrality=getattr(self._rag_config, "bridge_w_centrality", 0.2),
                )
            annotate_bridge_scores(
                bridge_eligible,
                mean_sim_map=_mh_sim_map,
                **_mh_bw,
            )
            bridge_policy = self.derive_bridge_policy(
                bridge_eligible,
                alpha=bridge_alpha,
                relax=(retry_count > 0),
            )
            direct_strategy = DirectEvidenceStrategy.ANCHOR_BRIDGE_INTERLEAVED
        else:
            bridge_policy = BridgePolicy.disabled(alpha=bridge_alpha)
            direct_strategy = DirectEvidenceStrategy.CVS_DENSITY

        query_facets = extract_query_facets(query)

        # Area C: packing profile for multi-hop
        _slot_caps = slot_caps_for_query_type(plan.query_type)

        # ── Single global slot assembly ───────────────────────────────────────
        pack, budget_report = greedy_select(
            candidates=all_candidates,
            budget=budget,
            active_procedure_id=active_procedure_id,
            query_facets=query_facets,
            direct_strategy=direct_strategy,
            bridge_policy=bridge_policy,
            slot_caps_override=_slot_caps,
        )

        # ── Traceability density Tᵈ ───────────────────────────────────────────
        td = compute_traceability_density(pack)
        _td_pool = (
                len(pack.protected_items)
                + len(pack.direct_evidence)
                + len(pack.bridge_evidence)
                + len(pack.global_summaries)
                + len(pack.conflicts)
        )
        if td < 0.85 and _td_pool >= 5:
            log.warning(
                "Tᵈ=%.3f < 0.85 for %s (multi-hop path)",
                td,
                conversation_id,
            )

        # ── Coverage evaluation ───────────────────────────────────────────────
        _mh_cov_thr = plan.coverage_threshold
        if self._rag_config is not None:
            _mh_cov_thr = getattr(self._rag_config, "coverage_threshold", _mh_cov_thr)
        coverage = evaluate_coverage(query, pack, threshold=_mh_cov_thr)
        pack.coverage_score = coverage.coverage_score

        # ── Adaptive retry (single retry only) ────────────────────────────────
        if coverage.retry_recommended and retry_count == 0:
            retry_recall_k = adaptive_retry_k(
                plan.recall_top_k, coverage.coverage_score, _mh_cov_thr,
            )
            log.debug(
                "MultiHop coverage %.2f < %.2f — adaptive retry with top_k=%d (base=%d)",
                coverage.coverage_score,
                _mh_cov_thr,
                retry_recall_k,
                plan.recall_top_k,
            )
            retry_plan = RetrievalPlan(
                query_type=plan.query_type,
                recall_top_k=retry_recall_k,
                rerank_top_n=plan.rerank_top_n,
                geometric_expand=plan.geometric_expand,
                geometric_seeds_k=plan.geometric_seeds_k,
                coverage_threshold=plan.coverage_threshold,
            )
            return await self.retrieve_multihop(
                query=query,
                sub_questions=sub_questions,
                plan=retry_plan,
                conversation_id=conversation_id,
                budget=budget,
                active_procedure_id=active_procedure_id,
                retry_count=1,
                phase=phase,
                exclude_ids=exclude_ids,
            )

        # ── Telemetry ─────────────────────────────────────────────────────────
        latency_ms = int((time.monotonic() - t0) * 1000)
        _schedule_background(
            self._db.save_retrieval_telemetry(
                conversation_id=conversation_id,
                query_type=plan.query_type,
                recall_top_k=plan.recall_top_k,
                rerank_top_n=plan.rerank_top_n,
                items_selected=budget_report.items_selected,
                coverage_score=coverage.coverage_score,
                retry_count=retry_count,
                latency_ms=latency_ms,
                candidates_before_rerank=total_before_rerank,
                candidates_after_rerank=total_after_rerank,
                candidates_after_expand=after_expand,
                pack_total_tokens=budget_report.tokens_used,
                geometric_expand=plan.geometric_expand,
                reranker_available=reranker_available,
                traceability_density=td,
                q3_relevant_count=_q3_relevant_total,
                direct_strategy=direct_strategy.value,
                bridge_enabled=bridge_policy.enabled,
                bridge_alpha=bridge_policy.alpha,
                bridge_floor=bridge_policy.bridge_floor,
                bridge_candidates_eligible=len(bridge_eligible),
            ),
            name=f"rag_telemetry_mh_{conversation_id}",
        )

        return pack, budget_report, coverage

    async def _multi_source_recall(
        self,
        query: str,
        plan: RetrievalPlan,
        conversation_id: str,
    ) -> tuple[list[ScoredCItem], list[ScoredCItem], list[ScoredCItem]]:
        """3 parallel queries: episodic (non-flagged) + global + flagged — T-04.

        Returns three separate ranked lists so the caller can pass them individually
        to reciprocal_rank_fusion() for proper multi-source boosting.
        Q3 (flagged) is returned separately so flagged items receive their own RRF
        score and still route to the conflicts slot via greedy_select().
        """
        _rc = self._rag_config
        _ep_ratio = getattr(_rc, "episodic_ratio", 0.8)
        _gl_ratio = getattr(_rc, "global_ratio", 0.4)
        _fl_cap = getattr(_rc, "flagged_cap", 10)
        episodic_k = int(plan.recall_top_k * _ep_ratio)
        global_k   = int(plan.recall_top_k * _gl_ratio)
        flagged_k  = int(_fl_cap)

        results = await asyncio.gather(
            self._cstore.search(query, episodic_active_filter(conversation_id), episodic_k),
            self._cstore.search(query, global_active_filter(), global_k),
            self._cstore.search(query, episodic_flagged_filter(conversation_id), flagged_k),
            return_exceptions=True,
        )

        lists: list[list[ScoredCItem]] = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                log.warning("Recall query %d failed: %s", i, res)
                lists.append([])
            else:
                lists.append(res)

        # Global relevance gate: drop Q2 items below minimum score before RRF.
        # Q1 (episodic) and Q3 (flagged conflicts) are never filtered — they are
        # always relevant to the current conversation by construction.
        q2 = lists[1]
        if self._global_min_score > 0.0 and q2:
            before = len(q2)
            q2 = [sc for sc in q2 if sc.score >= self._global_min_score]
            dropped = before - len(q2)
            if dropped:
                log.debug(
                    "Global relevance gate: dropped %d/%d global items (score < %.2f)",
                    dropped, before, self._global_min_score,
                )
            lists[1] = q2

        return lists[0], lists[1], lists[2]

    async def _rerank(
        self,
        query: str,
        candidates: list[ScoredCItem],
        top_n: int,
    ) -> list[ScoredCItem]:
        """CrossEncoder rerank — raises RerankerUnavailableError on failure."""
        if not candidates:
            return []
        texts = [sc.citem.content for sc in candidates]
        rerank_results = await self._reranker.rerank(query, texts, top_n)
        reranked = []
        for rr in rerank_results:
            if rr.index < len(candidates):
                sc = candidates[rr.index]
                reranked.append(ScoredCItem(
                    citem=sc.citem,
                    score=rr.score,
                    provenance=sc.provenance,
                    dense_score=sc.dense_score,   # cosine from Qdrant; preserved for ABS
                    rerank_score=rr.score,         # CrossEncoder score; fixed from here on
                    rerank_verified = True
                ))
        return reranked

    async def _late_interaction_rerank(
        self,
        query: str,
        candidates: list[ScoredCItem],
        top_n: int,
    ) -> list[ScoredCItem]:
        """ColBERT late-interaction rerank — raises RerankerUnavailableError on failure.

        Preserves the CrossEncoder rerank_score on each candidate; sets score to
        the ColBERT MaxSim score so downstream ABS/bridge logic uses the best
        available signal. Does NOT filter candidates — top_n == len(candidates)
        so the CrossEncoder shortlist length is preserved.
        """
        if self._colbert is None or not candidates:
            return candidates
        texts = [sc.citem.content for sc in candidates]
        colbert_results = await self._colbert.rerank(query, texts, top_n)
        if not colbert_results:
            return candidates
        # Build a reordered list anchored on ColBERT scores.
        # Candidates not returned by ColBERT (e.g. batch cap) keep their position.
        colbert_by_idx = {rr.index: rr.score for rr in colbert_results}
        reranked: list[ScoredCItem] = []
        for rr in colbert_results:
            if rr.index < len(candidates):
                sc = candidates[rr.index]
                reranked.append(ScoredCItem(
                    citem=sc.citem,
                    score=rr.score,
                    provenance=sc.provenance,
                    dense_score=sc.dense_score,
                    rerank_score=sc.rerank_score,   # CrossEncoder score preserved
                    rerank_verified=True,
                ))
                setattr(reranked[-1], "colbert_score", rr.score)
        # Append any candidates that ColBERT did not score (batch cap exceeded).
        scored_indices = set(colbert_by_idx.keys())
        for i, sc in enumerate(candidates):
            if i not in scored_indices:
                reranked.append(sc)
        return reranked
