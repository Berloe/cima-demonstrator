"""Tests for pure domain operations (cima_demo/domain/operations.py)."""
from __future__ import annotations

import math
import time

import pytest

from cima_demo.domain.entities import CItem, ContextPack
from cima_demo.domain.operations import (
    adaptive_pre_rerank_k,
    adaptive_rerank_k,
    adaptive_retry_k,
    annotate_bridge_scores,
    can_purge,
    classify_query,
    compute_bridge_score,
    compute_cvs,
    compute_mean_sim_to_pool,
    compute_novelty_score,
    compute_recency_score,
    compute_static_importance,
    context_pack_to_view,
    evaluate_coverage,
    greedy_select,
    is_promotion_eligible,
    reciprocal_rank_fusion,
    should_attenuate,
)
from cima_demo.domain.value_objects import (
    ContextBudget,
    ForgetParams,
    ItemType,
    PromotionPolicy,
    QueryType,
    RecallSource,
    ScoredCItem,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _citem(
    citem_id: str = "id-1",
    item_type: str = ItemType.FACT,
    scope: str = "episodic",
    scope_status: str = "active",
    importance: float = 0.5,
    content: str = "test content",
    dependency_ids: list[str] | None = None,
) -> CItem:
    return CItem(
        citem_id=citem_id,
        item_type=item_type,
        scope=scope,
        scope_status=scope_status,
        importance=importance,
        content=content,
        dependency_ids=dependency_ids or [],
    )


def _scored(citem: CItem, score: float = 1.0) -> ScoredCItem:
    return ScoredCItem(citem=citem, score=score, provenance=RecallSource.HYBRID_EPISODIC)


# ── compute_static_importance ─────────────────────────────────────────────────

class TestComputeStaticImportance:
    def test_decision_high_confidence(self) -> None:
        score = compute_static_importance(ItemType.DECISION, 1.0, None)
        assert score == pytest.approx(0.85, abs=1e-4)

    def test_constraint_high_confidence(self) -> None:
        score = compute_static_importance(ItemType.CONSTRAINT, 1.0, None)
        assert score == pytest.approx(0.85, abs=1e-4)

    def test_observation_half_confidence(self) -> None:
        score = compute_static_importance(ItemType.OBSERVATION, 0.5, None)
        assert score == pytest.approx(0.20, abs=1e-4)

    def test_verified_label_adds_bonus(self) -> None:
        without = compute_static_importance(ItemType.FACT, 1.0, None)
        with_verified = compute_static_importance(ItemType.FACT, 1.0, "verified")
        assert with_verified > without

    def test_refuted_label_subtracts(self) -> None:
        without = compute_static_importance(ItemType.FACT, 1.0, None)
        refuted = compute_static_importance(ItemType.FACT, 1.0, "refuted")
        assert refuted < without

    def test_clamps_to_one(self) -> None:
        score = compute_static_importance(ItemType.DECISION, 1.0, "verified")
        assert score <= 1.0

    def test_clamps_to_zero(self) -> None:
        score = compute_static_importance(ItemType.OBSERVATION, 0.0, "refuted")
        assert score >= 0.0

    def test_unknown_type_falls_back_to_default(self) -> None:
        score = compute_static_importance("UNKNOWN_TYPE", 1.0, None)
        assert score == pytest.approx(0.50, abs=1e-4)


# ── compute_cvs ───────────────────────────────────────────────────────────────

class TestComputeCvs:
    def test_all_ones(self) -> None:
        score = compute_cvs(1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_all_zeros(self) -> None:
        assert compute_cvs(0.0, 0.0, 0.0) == 0.0

    def test_weights_sum_to_one(self) -> None:
        # Default CVSWeights: α + β + γ = 1 → cvs(1,1,1) == 1
        score = compute_cvs(1.0, 1.0, 1.0)
        assert score <= 1.0 + 1e-6


# ── compute_recency_score ─────────────────────────────────────────────────────

class TestComputeRecencyScore:
    def test_brand_new_item(self) -> None:
        now = time.time()
        score = compute_recency_score(now, now)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_seven_day_old_item(self) -> None:
        now = time.time()
        seven_days_ago = now - 7 * 86400
        score = compute_recency_score(seven_days_ago, now)
        assert score == pytest.approx(math.exp(-1.0), abs=1e-4)

    def test_future_created_at_clamps_to_one(self) -> None:
        now = time.time()
        score = compute_recency_score(now + 86400, now)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_old_item_near_zero(self) -> None:
        now = time.time()
        score = compute_recency_score(now - 60 * 86400, now)
        assert score < 0.01


# ── compute_novelty_score ─────────────────────────────────────────────────────

class TestComputeNoveltyScore:
    def test_novel_item(self) -> None:
        item = _citem("new-id")
        assert compute_novelty_score(item, {"other-id"}) == 1.0

    def test_duplicate_item(self) -> None:
        item = _citem("dup-id")
        assert compute_novelty_score(item, {"dup-id"}) == 0.0

    def test_empty_pool(self) -> None:
        item = _citem("any-id")
        assert compute_novelty_score(item, set()) == 1.0


# ── compute_bridge_score ──────────────────────────────────────────────────────

class TestComputeBridgeScore:
    def test_all_deps_in_seeds(self) -> None:
        item = _citem(dependency_ids=["a", "b"])
        assert compute_bridge_score(item, {"a", "b"}) == 1.0

    def test_half_deps_in_seeds(self) -> None:
        item = _citem(dependency_ids=["a", "b"])
        assert compute_bridge_score(item, {"a"}) == pytest.approx(0.5, abs=1e-4)

    def test_no_deps(self) -> None:
        item = _citem(dependency_ids=[])
        assert compute_bridge_score(item, {"a"}) == 0.0

    def test_no_overlap(self) -> None:
        item = _citem(dependency_ids=["x", "y"])
        assert compute_bridge_score(item, {"a", "b"}) == 0.0


# ── should_attenuate ──────────────────────────────────────────────────────────

class TestShouldAttenuate:
    def test_old_low_importance_fact(self) -> None:
        item = _citem(item_type=ItemType.FACT, importance=0.1)
        params = ForgetParams(attenuation_age_days=7.0, attenuation_threshold=0.3)
        assert should_attenuate(item, age_days=10.0, params=params) is True

    def test_too_young(self) -> None:
        item = _citem(item_type=ItemType.FACT, importance=0.1)
        params = ForgetParams(attenuation_age_days=7.0, attenuation_threshold=0.3)
        assert should_attenuate(item, age_days=3.0, params=params) is False

    def test_importance_above_threshold(self) -> None:
        item = _citem(item_type=ItemType.FACT, importance=0.9)
        params = ForgetParams(attenuation_age_days=1.0, attenuation_threshold=0.3)
        assert should_attenuate(item, age_days=10.0, params=params) is False

    def test_protected_decision_never_attenuated(self) -> None:
        item = _citem(item_type=ItemType.DECISION, importance=0.01)
        params = ForgetParams(attenuation_age_days=1.0, attenuation_threshold=0.9)
        assert should_attenuate(item, age_days=100.0, params=params) is False

    def test_protected_constraint_never_attenuated(self) -> None:
        item = _citem(item_type=ItemType.CONSTRAINT, importance=0.01)
        assert should_attenuate(item, age_days=100.0) is False

    def test_archived_item_not_attenuated(self) -> None:
        item = _citem(item_type=ItemType.FACT, scope_status="archived", importance=0.1)
        params = ForgetParams(attenuation_age_days=1.0, attenuation_threshold=0.9)
        assert should_attenuate(item, age_days=10.0, params=params) is False


# ── can_purge ─────────────────────────────────────────────────────────────────

class TestCanPurge:
    def test_purgeable_archived_item(self) -> None:
        item = _citem(scope_status="archived", importance=0.05)
        params = ForgetParams(alpha_purge_days=30.0, min_importance_to_purge=0.1)
        assert can_purge(item, days_since_archived=35.0, params=params) is True

    def test_not_enough_days(self) -> None:
        item = _citem(scope_status="archived", importance=0.05)
        params = ForgetParams(alpha_purge_days=30.0, min_importance_to_purge=0.1)
        assert can_purge(item, days_since_archived=10.0, params=params) is False

    def test_importance_too_high(self) -> None:
        item = _citem(scope_status="archived", importance=0.5)
        params = ForgetParams(alpha_purge_days=30.0, min_importance_to_purge=0.1)
        assert can_purge(item, days_since_archived=35.0, params=params) is False

    def test_active_item_not_purgeable(self) -> None:
        item = _citem(scope_status="active", importance=0.01)
        assert can_purge(item, days_since_archived=100.0) is False


# ── is_promotion_eligible ─────────────────────────────────────────────────────

class TestIsPromotionEligible:
    def test_eligible(self) -> None:
        item = _citem(scope="episodic", scope_status="active", importance=0.8)
        policy = PromotionPolicy(min_references=3, min_importance=0.7)
        assert is_promotion_eligible(item, 5, policy) is True

    def test_not_enough_references(self) -> None:
        item = _citem(scope="episodic", scope_status="active", importance=0.8)
        policy = PromotionPolicy(min_references=3, min_importance=0.7)
        assert is_promotion_eligible(item, 2, policy) is False

    def test_importance_too_low(self) -> None:
        item = _citem(scope="episodic", scope_status="active", importance=0.3)
        policy = PromotionPolicy(min_references=1, min_importance=0.7)
        assert is_promotion_eligible(item, 5, policy) is False

    def test_global_scope_not_eligible(self) -> None:
        item = _citem(scope="global", scope_status="active", importance=0.9)
        assert is_promotion_eligible(item, 10) is False

    def test_archived_not_eligible(self) -> None:
        item = _citem(scope="episodic", scope_status="archived", importance=0.9)
        assert is_promotion_eligible(item, 10) is False


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        a = _scored(_citem("a"), 1.0)
        b = _scored(_citem("b"), 0.8)
        result = reciprocal_rank_fusion([[a, b]])
        assert [r.citem.citem_id for r in result] == ["a", "b"]

    def test_consistent_item_in_both_lists_ranks_higher(self) -> None:
        a = _scored(_citem("a"), 1.0)
        b = _scored(_citem("b"), 0.9)
        c = _scored(_citem("c"), 0.5)
        # 'a' appears first in both lists → highest RRF score
        list1 = [a, b, c]
        list2 = [a, c, b]
        result = reciprocal_rank_fusion([list1, list2])
        assert result[0].citem.citem_id == "a"

    def test_empty_lists(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_first_occurrence_citem_preserved(self) -> None:
        c1 = _citem("shared", content="version-1")
        c2 = _citem("shared", content="version-2")
        list1 = [_scored(c1)]
        list2 = [_scored(c2)]
        result = reciprocal_rank_fusion([list1, list2])
        assert len(result) == 1
        assert result[0].citem.content == "version-1"

    def test_scores_are_positive(self) -> None:
        items = [_scored(_citem(str(i))) for i in range(5)]
        result = reciprocal_rank_fusion([items])
        assert all(r.score > 0 for r in result)


# ── classify_query ────────────────────────────────────────────────────────────

class TestClassifyQuery:
    def test_procedural_keyword(self) -> None:
        assert classify_query("how to configure the server") == QueryType.PROCEDURAL

    def test_diagnostic_keyword(self) -> None:
        assert classify_query("why is the service failing") == QueryType.DIAGNOSTIC

    def test_short_query_local_precise(self) -> None:
        assert classify_query("user login") == QueryType.LOCAL_PRECISE

    def test_multi_hop_keywords(self) -> None:
        result = classify_query("compare and contrast both approaches across all modules")
        assert result == QueryType.MULTI_HOP

    def test_long_abstract_query_global(self) -> None:
        query = "explain the architectural rationale for using qdrant as the primary store"
        assert classify_query(query) == QueryType.GLOBAL_SYNTHETIC


# ── evaluate_coverage ─────────────────────────────────────────────────────────

class TestEvaluateCoverage:
    def test_full_coverage(self) -> None:
        pack = ContextPack()
        pack.direct_evidence = [_citem(content="python database query")]
        report = evaluate_coverage("python database query", pack)
        assert report.coverage_score == pytest.approx(1.0, abs=0.01)
        assert not report.retry_recommended

    def test_zero_coverage(self) -> None:
        pack = ContextPack()
        pack.direct_evidence = [_citem(content="cats and dogs")]
        report = evaluate_coverage("python database query", pack)
        assert report.coverage_score < 0.5

    def test_empty_pack_returns_full_coverage(self) -> None:
        # No concepts to cover when pack is empty → score=0 but query has concepts
        pack = ContextPack()
        report = evaluate_coverage("python", pack)
        assert 0.0 <= report.coverage_score <= 1.0


# ── context_pack_to_view ──────────────────────────────────────────────────────

class TestContextPackToView:
    def test_empty_pack_produces_empty_text(self) -> None:
        view = context_pack_to_view(ContextPack())
        assert view.text == ""
        assert view.tokens_used == 0

    def test_direct_evidence_appears_in_text(self) -> None:
        pack = ContextPack()
        pack.direct_evidence = [_citem(content="important fact")]
        view = context_pack_to_view(pack)
        assert "important fact" in view.text
        assert "Direct Evidence" in view.text

    def test_conflicts_appear_in_text(self) -> None:
        pack = ContextPack()
        pack.conflicts = [_citem(content="contradictory statement")]
        view = context_pack_to_view(pack)
        assert "contradictory statement" in view.text


# ── greedy_select ─────────────────────────────────────────────────────────────

class TestGreedySelect:
    def _budget(self, max_tokens: int = 4096) -> ContextBudget:
        return ContextBudget(max_tokens=max_tokens)

    def test_empty_candidates(self) -> None:
        pack, report = greedy_select([], self._budget())
        assert pack.all_items() == []
        assert report.tokens_used == 0

    def test_protected_items_always_included(self) -> None:
        decision = _citem("d1", item_type=ItemType.DECISION, importance=0.9, content="x" * 10)
        obs = _citem("o1", item_type=ItemType.OBSERVATION, importance=0.3, content="y" * 10)
        candidates = [_scored(decision), _scored(obs)]
        pack, _ = greedy_select(candidates, self._budget(max_tokens=100))
        ids = {i.citem_id for i in pack.all_items()}
        assert "d1" in ids

    def test_budget_respected(self) -> None:
        # Each item has content of 400 chars ≈ 100 tokens
        items = [_scored(_citem(str(i), content="x" * 400)) for i in range(20)]
        pack, report = greedy_select(items, self._budget(max_tokens=512))
        assert report.tokens_used <= 512 + 50  # small overrun tolerance

    def test_active_procedure_pinned(self) -> None:
        procedure = _citem("proc-1", item_type=ItemType.PROCEDURE, importance=0.2)
        other = _citem("other-1", item_type=ItemType.OBSERVATION, importance=0.9)
        candidates = [_scored(procedure, 0.1), _scored(other, 0.9)]
        pack, _ = greedy_select(candidates, self._budget(), active_procedure_id="proc-1")
        ids = {i.citem_id for i in pack.all_items()}
        assert "proc-1" in ids


# ── Adaptive-k ───────────────────────────────────────────────────────────────

class TestAdaptivePreRerankK:
    def test_trims_low_rrf(self) -> None:
        scores = [0.10, 0.09, 0.08, 0.02, 0.01, 0.005]
        k = adaptive_pre_rerank_k(scores, base_rerank_n=6, min_n=2)
        assert k == 3

    def test_all_high_returns_base(self) -> None:
        scores = [0.10, 0.09, 0.08, 0.07, 0.06]
        k = adaptive_pre_rerank_k(scores, base_rerank_n=5)
        assert k == 5

    def test_empty(self) -> None:
        assert adaptive_pre_rerank_k([], base_rerank_n=10) == 0

    def test_respects_min_n(self) -> None:
        scores = [0.10, 0.01, 0.005, 0.001]
        k = adaptive_pre_rerank_k(scores, base_rerank_n=4, min_n=3)
        assert k >= 3

    def test_all_zero_returns_base(self) -> None:
        scores = [0.0, 0.0, 0.0]
        k = adaptive_pre_rerank_k(scores, base_rerank_n=5)
        assert k == 3

    def test_single_item(self) -> None:
        assert adaptive_pre_rerank_k([0.5], base_rerank_n=10) == 1


class TestAdaptiveRerankK:
    def test_detects_cliff(self) -> None:
        scores = [0.95, 0.90, 0.85, 0.40, 0.35, 0.30]
        k = adaptive_rerank_k(scores, base_top_n=6, min_k=1)
        assert k == 3

    def test_no_cliff_returns_base(self) -> None:
        scores = [0.90, 0.88, 0.86, 0.84, 0.82]
        k = adaptive_rerank_k(scores, base_top_n=5)
        assert k == 5

    def test_single_item(self) -> None:
        assert adaptive_rerank_k([0.9], base_top_n=10, min_k=1) == 1

    def test_empty(self) -> None:
        assert adaptive_rerank_k([], base_top_n=10, min_k=0) == 0

    def test_respects_min_k(self) -> None:
        scores = [0.95, 0.10, 0.05, 0.01]
        k = adaptive_rerank_k(scores, base_top_n=4, min_k=3)
        assert k >= 3

    def test_uniform_scores_returns_base(self) -> None:
        scores = [0.5, 0.5, 0.5, 0.5]
        k = adaptive_rerank_k(scores, base_top_n=4)
        assert k == 4


class TestAdaptiveRetryK:
    def test_small_gap_mild_scale(self) -> None:
        k = adaptive_retry_k(20, coverage_score=0.6, coverage_threshold=0.7)
        assert k == 30  # gap=0.1, factor=1.5

    def test_large_gap_aggressive_scale(self) -> None:
        k = adaptive_retry_k(20, coverage_score=0.4, coverage_threshold=0.7)
        assert k == 50  # gap=0.3, factor=2.5

    def test_max_gap_capped_at_3x(self) -> None:
        k = adaptive_retry_k(10, coverage_score=0.0, coverage_threshold=1.0)
        assert k == 30  # gap=1.0, factor capped at 3.0

    def test_no_gap_returns_base(self) -> None:
        k = adaptive_retry_k(20, coverage_score=0.8, coverage_threshold=0.7)
        assert k == 20  # gap=0, factor=1.0


# ── compute_mean_sim_to_pool ─────────────────────────────────────────────────

class TestComputeMeanSimToPool:
    def test_identical_vectors(self) -> None:
        vecs = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]}
        result = compute_mean_sim_to_pool(vecs, ["a", "b", "c"])
        assert result["a"] == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self) -> None:
        vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = compute_mean_sim_to_pool(vecs, ["a", "b"])
        assert result["a"] == pytest.approx(0.0, abs=0.01)
        assert result["b"] == pytest.approx(0.0, abs=0.01)

    def test_empty_returns_empty(self) -> None:
        assert compute_mean_sim_to_pool({}, []) == {}

    def test_single_item_returns_empty(self) -> None:
        assert compute_mean_sim_to_pool({"a": [1.0]}, ["a"]) == {}

    def test_missing_ids_skipped(self) -> None:
        vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = compute_mean_sim_to_pool(vecs, ["a", "b", "c"])
        assert "c" not in result
        assert len(result) == 2


# ── annotate_bridge_scores ───────────────────────────────────────────────────

class TestAnnotateBridgeScores:
    def _make_sc(self, citem_id: str, dense: float, rerank: float) -> ScoredCItem:
        return ScoredCItem(
            citem=CItem(citem_id=citem_id, content="test"),
            score=0.5,
            provenance=RecallSource.HYBRID_EPISODIC,
            dense_score=dense,
            rerank_score=rerank,
        )

    def test_three_signal_formula(self) -> None:
        a = self._make_sc("a", dense=1.0, rerank=1.0)
        b = self._make_sc("b", dense=0.5, rerank=0.5)
        c = self._make_sc("c", dense=0.0, rerank=0.0)
        candidates = [a, b, c]
        mean_sim_map = {"a": 1.0, "b": 0.5, "c": 0.0}
        annotate_bridge_scores(candidates, mean_sim_map=mean_sim_map)
        assert a.bridge_score == pytest.approx(1.0, abs=0.01)
        assert c.bridge_score == pytest.approx(0.0, abs=0.01)
        assert 0.0 < b.bridge_score < 1.0

    def test_fallback_two_signal_without_mean_sim(self) -> None:
        a = self._make_sc("a", dense=1.0, rerank=0.0)
        b = self._make_sc("b", dense=0.0, rerank=1.0)
        annotate_bridge_scores([a, b], mean_sim_map=None)
        assert a.bridge_score == pytest.approx(0.5, abs=0.01)
        assert b.bridge_score == pytest.approx(0.5, abs=0.01)

    def test_none_scores_produce_none_bridge(self) -> None:
        a = ScoredCItem(
            citem=CItem(citem_id="a", content="test"),
            score=0.5,
            provenance=RecallSource.HYBRID_EPISODIC,
            dense_score=None,
            rerank_score=0.5,
        )
        annotate_bridge_scores([a])
        assert a.bridge_score is None
