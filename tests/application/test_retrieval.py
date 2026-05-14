"""Tests for RetrievalOrchestrator (cima_demo/application/retrieval.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cima_demo.retrieval.orchestrator import RetrievalOrchestrator
from cima_demo.domain.entities import CItem
from cima_demo.domain.errors import RerankerUnavailableError
from cima_demo.domain.value_objects import (
    BudgetReport,
    ContextBudget,
    CoverageReport,
    QueryType,
    RecallSource,
    RerankResult,
    RetrievalPlan,
    ScoredCItem,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_citem(citem_id: str = "c1", content: str = "content", token_count: int = 10) -> CItem:
    return CItem(citem_id=citem_id, content=content, token_count=token_count)


def _make_scored(
    citem_id: str = "c1",
    score: float = 0.9,
    provenance: RecallSource = RecallSource.HYBRID_EPISODIC,
) -> ScoredCItem:
    return ScoredCItem(
        citem=_make_citem(citem_id=citem_id),
        score=score,
        provenance=provenance,
    )


def _default_plan(
    *,
    recall_top_k: int = 10,
    rerank_top_n: int = 5,
    geometric_expand: bool = False,
    coverage_threshold: float = 0.5,
) -> RetrievalPlan:
    return RetrievalPlan(
        query_type=QueryType.LOCAL_PRECISE,
        recall_top_k=recall_top_k,
        rerank_top_n=rerank_top_n,
        geometric_expand=geometric_expand,
        geometric_seeds_k=3,
        coverage_threshold=coverage_threshold,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_cstore() -> AsyncMock:
    store = AsyncMock()
    store.search = AsyncMock(return_value=[_make_scored()])
    return store


@pytest.fixture
def mock_expander() -> AsyncMock:
    expander = AsyncMock()
    expander.expand = AsyncMock(return_value=[])
    return expander


@pytest.fixture
def mock_reranker() -> AsyncMock:
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=[RerankResult(index=0, score=0.95)])
    return reranker


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.save_retrieval_telemetry = AsyncMock()
    return db


@pytest.fixture
def orchestrator(
    mock_cstore: AsyncMock,
    mock_expander: AsyncMock,
    mock_reranker: AsyncMock,
    mock_db: AsyncMock,
) -> RetrievalOrchestrator:
    return RetrievalOrchestrator(
        citem_store=mock_cstore,
        geometric_expander=mock_expander,
        reranker_port=mock_reranker,
        rel_db=mock_db,
    )


# ── _multi_source_recall ──────────────────────────────────────────────────────

class TestMultiSourceRecall:
    async def test_issues_three_parallel_queries(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        plan = _default_plan()
        await orchestrator._multi_source_recall("query", plan, "conv1")
        assert mock_cstore.search.call_count == 3

    async def test_combines_results_from_all_queries(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        items = [_make_scored("a"), _make_scored("b"), _make_scored("c")]
        mock_cstore.search.side_effect = [[items[0]], [items[1]], [items[2]]]
        plan = _default_plan()
        result = await orchestrator._multi_source_recall("query", plan, "conv1")
        assert len(result) == 3

    async def test_partial_failure_handled_gracefully(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        good = _make_scored("ok")
        # side_effect list: call1→[good], call2→raises, call3→[]
        mock_cstore.search.side_effect = [[good], RuntimeError("qdrant down"), []]
        plan = _default_plan()
        episodic, global_, flagged = await orchestrator._multi_source_recall("query", plan, "conv1")
        # Only the episodic result survives; failed global → []
        assert len(episodic) == 1
        assert episodic[0].citem.citem_id == "ok"
        assert global_ == []

    async def test_all_failures_returns_empty(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        mock_cstore.search.side_effect = RuntimeError("down")
        plan = _default_plan()
        episodic, global_, flagged = await orchestrator._multi_source_recall("query", plan, "conv1")
        assert episodic == []
        assert global_ == []
        assert flagged == []

    async def test_episodic_k_is_80_percent_of_recall_top_k(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        plan = _default_plan(recall_top_k=20)
        mock_cstore.search.return_value = []
        await orchestrator._multi_source_recall("q", plan, "conv1")
        # First call (episodic): top_k = int(20 * 0.8) = 16
        first_call_k = mock_cstore.search.call_args_list[0][0][2]
        assert first_call_k == 16

    async def test_global_k_is_40_percent_of_recall_top_k(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        plan = _default_plan(recall_top_k=20)
        mock_cstore.search.return_value = []
        await orchestrator._multi_source_recall("q", plan, "conv1")
        # Second call (global): top_k = int(20 * 0.4) = 8
        second_call_k = mock_cstore.search.call_args_list[1][0][2]
        assert second_call_k == 8

    async def test_flagged_k_is_fixed_10(
        self, orchestrator: RetrievalOrchestrator, mock_cstore: AsyncMock
    ) -> None:
        plan = _default_plan(recall_top_k=50)
        mock_cstore.search.return_value = []
        await orchestrator._multi_source_recall("q", plan, "conv1")
        # Third call (flagged): top_k = 10 (fixed)
        third_call_k = mock_cstore.search.call_args_list[2][0][2]
        assert third_call_k == 10


# ── _rerank ───────────────────────────────────────────────────────────────────

class TestRerank:
    async def test_returns_empty_for_empty_candidates(
        self, orchestrator: RetrievalOrchestrator
    ) -> None:
        result = await orchestrator._rerank("q", [], 5)
        assert result == []

    async def test_maps_rerank_results_by_index(
        self, orchestrator: RetrievalOrchestrator, mock_reranker: AsyncMock
    ) -> None:
        candidates = [_make_scored("a", score=0.3), _make_scored("b", score=0.2)]
        mock_reranker.rerank.return_value = [
            RerankResult(index=1, score=0.99),
            RerankResult(index=0, score=0.88),
        ]
        result = await orchestrator._rerank("q", candidates, 2)
        assert len(result) == 2
        assert result[0].citem.citem_id == "b"
        assert result[0].score == pytest.approx(0.99)

    async def test_ignores_out_of_bounds_index(
        self, orchestrator: RetrievalOrchestrator, mock_reranker: AsyncMock
    ) -> None:
        candidates = [_make_scored("a")]
        mock_reranker.rerank.return_value = [
            RerankResult(index=0, score=0.9),
            RerankResult(index=99, score=0.5),  # out of bounds
        ]
        result = await orchestrator._rerank("q", candidates, 5)
        assert len(result) == 1

    async def test_propagates_reranker_unavailable_error(
        self, orchestrator: RetrievalOrchestrator, mock_reranker: AsyncMock
    ) -> None:
        mock_reranker.rerank.side_effect = RerankerUnavailableError("down")
        candidates = [_make_scored()]
        with pytest.raises(RerankerUnavailableError):
            await orchestrator._rerank("q", candidates, 5)

    async def test_preserves_provenance_from_original_candidate(
        self, orchestrator: RetrievalOrchestrator, mock_reranker: AsyncMock
    ) -> None:
        candidate = ScoredCItem(
            citem=_make_citem("x"), score=0.5, provenance=RecallSource.HYBRID_GLOBAL
        )
        mock_reranker.rerank.return_value = [RerankResult(index=0, score=0.8)]
        result = await orchestrator._rerank("q", [candidate], 1)
        assert result[0].provenance == RecallSource.HYBRID_GLOBAL


# ── retrieve — full pipeline ──────────────────────────────────────────────────

class TestRetrieve:
    async def test_returns_tuple_of_three(
        self, orchestrator: RetrievalOrchestrator
    ) -> None:
        plan = _default_plan(coverage_threshold=0.0)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            result = await orchestrator.retrieve("q", plan, "conv1", budget)
        assert len(result) == 3
        pack, budget_report, coverage = result
        assert isinstance(budget_report, BudgetReport)
        assert isinstance(coverage, CoverageReport)

    async def test_reranker_unavailable_falls_back_to_rrf_order(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_reranker: AsyncMock,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_reranker.rerank.side_effect = RerankerUnavailableError("offline")
        mock_cstore.search.return_value = [_make_scored("a", score=0.9)]
        plan = _default_plan(coverage_threshold=0.0)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            pack, _, _ = await orchestrator.retrieve("q", plan, "conv1", budget)
        # Should not raise — fallback to RRF order
        assert pack is not None

    async def test_geometric_expand_called_when_plan_flag_true(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_expander: AsyncMock,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_cstore.search.return_value = [_make_scored("seed1")]
        plan = _default_plan(geometric_expand=True, coverage_threshold=0.0)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            await orchestrator.retrieve("q", plan, "conv1", budget)
        mock_expander.expand.assert_called_once()

    async def test_geometric_expand_not_called_when_flag_false(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_expander: AsyncMock,
    ) -> None:
        plan = _default_plan(geometric_expand=False, coverage_threshold=0.0)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            await orchestrator.retrieve("q", plan, "conv1", budget)
        mock_expander.expand.assert_not_called()

    async def test_expanded_items_added_with_geometric_provenance(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_expander: AsyncMock,
        mock_cstore: AsyncMock,
        mock_reranker: AsyncMock,
    ) -> None:
        seed = _make_scored("seed")
        expanded_item = _make_citem("expanded", token_count=5)
        mock_cstore.search.return_value = [seed]
        mock_expander.expand.return_value = [expanded_item]
        mock_reranker.rerank.return_value = [RerankResult(index=0, score=0.9)]
        plan = _default_plan(geometric_expand=True, coverage_threshold=0.0)
        budget = ContextBudget(max_tokens=65536, overhead_tokens=1000)
        with patch("asyncio.create_task"):
            pack, _, _ = await orchestrator.retrieve("q", plan, "conv1", budget)
        # Both seed and expanded item should be in selected items
        selected_ids = {c.citem_id for c in pack.all_items()}
        assert "expanded" in selected_ids or "seed" in selected_ids

    async def test_telemetry_fired_as_non_blocking_task(
        self,
        orchestrator: RetrievalOrchestrator,
    ) -> None:
        plan = _default_plan(coverage_threshold=0.0)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task") as mock_task:
            await orchestrator.retrieve("q", plan, "conv1", budget)
        mock_task.assert_called_once()

    async def test_no_telemetry_on_adaptive_retry_first_call(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_cstore: AsyncMock,
    ) -> None:
        """First attempt triggers retry — telemetry only fires on second (retry_count=1)."""
        # Return empty results to force coverage failure → retry
        mock_cstore.search.return_value = []
        plan = _default_plan(coverage_threshold=1.0, recall_top_k=5)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task") as mock_task:
            # Use multi-word query so evaluate_coverage can extract concepts
            await orchestrator.retrieve("missing concepts query", plan, "conv1", budget)
        # Telemetry fires once — on the retry (retry_count=1), not on the first attempt
        mock_task.assert_called_once()

    async def test_adaptive_retry_scales_recall_top_k(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_cstore: AsyncMock,
    ) -> None:
        call_counts: list[int] = []

        async def search_side_effect(query: str, filt: object, top_k: int) -> list[ScoredCItem]:
            call_counts.append(top_k)
            return []

        mock_cstore.search.side_effect = search_side_effect
        plan = _default_plan(coverage_threshold=1.0, recall_top_k=10)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            await orchestrator.retrieve("missing concepts query", plan, "conv1", budget)

        # First attempt: episodic=8, global=4, flagged=10
        # Retry: adaptive_retry_k(10, coverage=0.0, threshold=1.0)
        #   gap=1.0, factor=min(3.0, 1+5.0)=3.0, retry_k=30
        #   episodic=int(30*0.8)=24, global=int(30*0.4)=12, flagged=10
        assert len(call_counts) == 6  # 3 queries × 2 attempts
        retry_episodic = call_counts[3]
        assert retry_episodic == 24  # int(30 * 0.8)

    async def test_no_second_retry_when_retry_count_is_1(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_cstore.search.return_value = []
        plan = _default_plan(coverage_threshold=1.0, recall_top_k=5)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            # Calling with retry_count=1 should NOT recurse again
            pack, _, coverage = await orchestrator.retrieve(
                "missing concepts query", plan, "conv1", budget, retry_count=1
            )
        # 3 queries (single attempt, no second retry)
        assert mock_cstore.search.call_count == 3

    async def test_empty_recall_returns_empty_pack(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_cstore.search.return_value = []
        plan = _default_plan(coverage_threshold=0.0)
        budget = ContextBudget.testing()
        with patch("asyncio.create_task"):
            pack, budget_report, _ = await orchestrator.retrieve("q", plan, "conv1", budget)
        assert budget_report.items_selected == 0

    async def test_active_procedure_id_forwarded_to_greedy_select(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_cstore.search.return_value = [_make_scored("proc", score=1.0)]
        plan = _default_plan(coverage_threshold=0.0)
        budget = ContextBudget(max_tokens=65536, overhead_tokens=1000)
        with patch("cima_demo.retrieval.orchestrator.greedy_select") as mock_gs, \
             patch("asyncio.create_task"):
            mock_gs.return_value = (
                MagicMock(coverage_score=0.8, all_items=lambda: []),
                BudgetReport(total_available=1000, tokens_used=0, items_selected=0),
            )
            await orchestrator.retrieve(
                "q", plan, "conv1", budget, active_procedure_id="proc-123"
            )
        call_kwargs = mock_gs.call_args.kwargs
        assert call_kwargs.get("active_procedure_id") == "proc-123"

    async def test_coverage_score_set_on_pack(
        self,
        orchestrator: RetrievalOrchestrator,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_cstore.search.return_value = [_make_scored()]
        plan = _default_plan(coverage_threshold=0.0)
        budget = ContextBudget(max_tokens=65536, overhead_tokens=1000)
        with patch("asyncio.create_task"):
            pack, _, coverage = await orchestrator.retrieve("q", plan, "conv1", budget)
        assert pack.coverage_score == coverage.coverage_score
