from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cima_demo.domain.entities import CItem, ContextPack, SummaryNode, TaskMemory
from cima_demo.domain.value_objects import BudgetReport, ContextBudget, CoverageReport, QueryType, RetrievalPlan
from cima_demo.retrieval.context_builder import ContextBuilder


class _Planner:
    def plan(self, query: str) -> RetrievalPlan:
        return RetrievalPlan(
            query_type=QueryType.LOCAL_PRECISE,
            recall_top_k=5,
            rerank_top_n=3,
            geometric_expand=False,
            geometric_seeds_k=0,
            coverage_threshold=0.5,
        )


class _Retrieval:
    async def retrieve(self, *, query: str, plan: RetrievalPlan, conversation_id: str, budget: ContextBudget, active_procedure_id: str | None, phase: str, exclude_ids: set[str]) -> tuple[ContextPack, BudgetReport, CoverageReport]:
        return ContextPack(tokens_used=0, coverage_score=1.0), BudgetReport(total_available=budget.available_for_content, tokens_used=0, items_selected=0), CoverageReport(coverage_score=1.0)


class _DB:
    async def fetch_pyramid_tops(self, conversation_id: str, limit: int | None = None):
        node = SummaryNode(
            node_id="sum-1",
            conversation_id=conversation_id,
            level=3,
            content="witness master summary",
            token_count=4,
            created_at=datetime(2026, 4, 30, tzinfo=UTC),
            origin_citem_ids=[],
        )
        setattr(node, "summary_resolution_mode", "witness_first")
        setattr(node, "summary_ref_kind", "local_summary")
        setattr(node, "summary_scope", "local")
        return [node]


@pytest.mark.asyncio
async def test_context_builder_marks_summary_items_with_resolution_metadata() -> None:
    builder = ContextBuilder(
        query_planner=_Planner(),
        retrieval_orchestrator=_Retrieval(),
        rel_db=_DB(),
        multi_hop_analyzer=None,
    )

    view = await builder.build(
        phase="RECALL",
        task_memory=TaskMemory(conversation_id="conv-1"),
        plan=None,
        query="witness master overview",
        conversation_id="conv-1",
        budget=ContextBudget(max_tokens=512, overhead_tokens=64),
    )

    assert len(view.items) == 1
    item = view.items[0]
    assert item["ref_kind"] == "summary"
    assert item["ref_id"] == "sum-1"
    assert item["summary_resolution_mode"] == "witness_first"
    assert item["summary_ref_kind"] == "local_summary"
    assert item["summary_scope"] == "local"


class _RetrievalWithWitnessCItem:
    async def retrieve(self, *, query: str, plan: RetrievalPlan, conversation_id: str, budget: ContextBudget, active_procedure_id: str | None, phase: str, exclude_ids: set[str]) -> tuple[ContextPack, BudgetReport, CoverageReport]:
        item = CItem(
            citem_id="c1",
            conversation_id=conversation_id,
            content="witness-backed fact",
            item_type="FACT",
            scope="episodic",
            token_count=5,
        )
        setattr(item, "citem_resolution_mode", "witness_first")
        setattr(item, "citem_resolution_scope", "local")
        pack = ContextPack(direct_evidence=[item], tokens_used=5, coverage_score=1.0)
        return pack, BudgetReport(total_available=budget.available_for_content, tokens_used=5, items_selected=1), CoverageReport(coverage_score=1.0)


class _DBNoSummaries:
    async def fetch_pyramid_tops(self, conversation_id: str, limit: int | None = None):
        return []


@pytest.mark.asyncio
async def test_context_builder_marks_citem_items_with_witness_resolution_metadata() -> None:
    builder = ContextBuilder(
        query_planner=_Planner(),
        retrieval_orchestrator=_RetrievalWithWitnessCItem(),
        rel_db=_DBNoSummaries(),
        multi_hop_analyzer=None,
    )

    view = await builder.build(
        phase="RECALL",
        task_memory=TaskMemory(conversation_id="conv-1"),
        plan=None,
        query="witness fact",
        conversation_id="conv-1",
        budget=ContextBudget(max_tokens=512, overhead_tokens=64),
    )

    assert len(view.items) == 1
    item = view.items[0]
    assert item["ref_kind"] == "citem"
    assert item["ref_id"] == "c1"
    assert item["item_resolution_mode"] == "witness_first"
    assert item["item_resolution_scope"] == "local"
