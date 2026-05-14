"""ContextBuilder — delegates to QueryPlanner + RetrievalOrchestrator (APP-INV-22)."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from cima_demo.retrieval.multi_hop import MultiHopAnalyzer
from cima_demo.retrieval.query_planner import QueryPlanner
from cima_demo.retrieval.orchestrator import RetrievalOrchestrator
from cima_demo.domain.entities import CItem, ContextView, Plan, TaskMemory
from cima_demo.domain.operations import (
    context_pack_to_view,
    extract_query_facets,
    filter_pack_by_objectives,
    is_relevant_global_obs_content,
)
from cima_demo.domain.ports import RelDBPort
from cima_demo.domain.value_objects import ContextBudget, ItemType, QueryType

log = logging.getLogger(__name__)

# ── MultiHop LLM-analysis gate (cheap-first principle) ───────────────────────
# The heuristic classifier already flags MULTI_HOP; the LLM analyze() call is an
# additional (expensive) hop-depth estimator. Only invoke it when the query clearly
# warrants N>2 hop decomposition — not for every vaguely multi-hop query.
_CHAIN_CONNECTIVES = frozenset({
    "and then", "after that", "which then", "as a result",
    "subsequently", "that led to", "following which", "because of which",
})


def _needs_multihop_analysis(query: str) -> bool:
    """True only when the query warrants LLM hop-depth estimation (cheap gate).

    Signals (any one is sufficient):
    1. Explicit chaining connective — semantically multi-hop by construction.
    2. ≥2 explicit question marks (multiple independent sub-questions).

    Word-count gate removed: long queries are not reliably multi-hop and the
    extra LLM analyze() call on iGPU costs ~600s for marginal benefit.
    """
    q_lower = query.lower()
    if any(c in q_lower for c in _CHAIN_CONNECTIVES):
        return True
    if query.count("?") >= 2:
        return True
    return False


class ContextBuilder:
    """Builds ContextView for LLM prompt injection.

    Delegates entirely to QueryPlanner + RetrievalOrchestrator.
    Also loads SummaryNode tops from PostgreSQL for pyramid context (N-06).

    Multi-hop routing:
    - QueryPlanner (pure) classifies the query and flags potential MULTI_HOP.
    - When MULTI_HOP, MultiHopAnalyzer (LLM) estimates the true hop depth and
      decomposes N-hop > 2 queries into ≤2-hop sub-questions.
    - Each sub-question runs a full 2-hop retrieve() (recall + geometric expand)
      with cross-iteration anchor seeding to propagate the reasoning chain.
    """

    def __init__(
        self,
        query_planner: QueryPlanner,
        retrieval_orchestrator: RetrievalOrchestrator,
        rel_db: RelDBPort,
        multi_hop_analyzer: MultiHopAnalyzer | None = None,
    ) -> None:
        self._planner = query_planner
        self._retrieval = retrieval_orchestrator
        self._db = rel_db
        self._analyzer = multi_hop_analyzer

    async def build(
        self,
        phase: str,
        task_memory: TaskMemory,
        plan: Plan | None,
        query: str,
        conversation_id: str,
        budget: ContextBudget,
        history_contents: set[str] | None = None,
        global_objective: str = "",
        local_objective: str = "",
        exclude_ids: set[str] | None = None,
        disable_geometric_expand: bool = False,
    ) -> ContextView:
        """Build ContextView for current turn iteration.

        1. QueryPlanner.plan() → RetrievalPlan  (pure, no LLM)
        2a. If MULTI_HOP: MultiHopAnalyzer.analyze() → MultiHopAnalysis  (LLM)
            - N > 2: RetrievalOrchestrator.retrieve_multihop() (iterative 2-hop)
            - N ≤ 2: RetrievalOrchestrator.retrieve() (standard 2-hop)
        2b. Otherwise: RetrievalOrchestrator.retrieve() (standard 2-hop)
        3. Load SummaryNode tops (N-06: fetch_pyramid_tops)
        4. context_pack_to_view() → ContextView
        """
        # Determine active procedure for greedy_select D-08
        active_procedure_id: str | None = None
        if plan is not None and plan.active_step is not None:
            active_procedure_id = plan.active_step.procedure_citem_id

        retrieval_plan = self._planner.plan(query)
        if disable_geometric_expand and retrieval_plan.geometric_expand:
            retrieval_plan = replace(retrieval_plan, geometric_expand=False, geometric_seeds_k=0)

        # ── Multi-hop routing ─────────────────────────────────────────────────
        # Cheap-first: LLM analyze() only fires when the heuristic gate confirms
        # the query likely requires N>2 hops. Single/dual-hop MULTI_HOP queries
        # skip the LLM call and run the standard 2-hop retrieve() path.
        use_multihop = (
            retrieval_plan.query_type == QueryType.MULTI_HOP
            and self._analyzer is not None
            and _needs_multihop_analysis(query)
        )

        _exclude = exclude_ids or set()

        if use_multihop:
            analysis = await self._analyzer.analyze(query)  # type: ignore[union-attr]
            log.debug(
                "MultiHopAnalysis: hop_depth=%d sub_questions=%d iterative=%s",
                analysis.hop_depth,
                len(analysis.sub_questions),
                analysis.needs_iterative_retrieval,
            )

            if analysis.needs_iterative_retrieval:
                pack, budget_report, coverage = await self._retrieval.retrieve_multihop(
                    query=query,
                    sub_questions=analysis.sub_questions,
                    plan=retrieval_plan,
                    conversation_id=conversation_id,
                    budget=budget,
                    active_procedure_id=active_procedure_id,
                    phase=phase,
                    exclude_ids=_exclude,
                )
            else:
                pack, budget_report, coverage = await self._retrieval.retrieve(
                    query=query,
                    plan=retrieval_plan,
                    conversation_id=conversation_id,
                    budget=budget,
                    active_procedure_id=active_procedure_id,
                    phase=phase,
                    exclude_ids=_exclude,
                )
        else:
            pack, budget_report, coverage = await self._retrieval.retrieve(
                query=query,
                plan=retrieval_plan,
                conversation_id=conversation_id,
                budget=budget,
                active_procedure_id=active_procedure_id,
                phase=phase,
                exclude_ids=_exclude,
            )

        # Load pyramid tops and add as global_summaries (N-06, L-02).
        # tokens_used starts from what greedy_select already placed in this slot
        # so SummaryNodes honour the 25% cap together with any global-scope CItems (Gap E fix).
        summary_budget = int(0.25 * budget.available_for_content)
        summary_nodes = await self._db.fetch_pyramid_tops(
            conversation_id=conversation_id,
            limit=10,
        )
        summary_modes_by_id: dict[str, dict[str, Any]] = {
            str(node.node_id): {
                "summary_resolution_mode": str(getattr(node, "summary_resolution_mode", "legacy_fallback") or "legacy_fallback"),
                "summary_ref_kind": str(getattr(node, "summary_ref_kind", "summary") or "summary"),
                "summary_scope": str(getattr(node, "summary_scope", "legacy") or "legacy"),
            }
            for node in summary_nodes
        }
        tokens_used = sum(item.token_count or 0 for item in pack.global_summaries)

        # Area D: gate pyramid tops by local relevance.
        # SummaryNode tops are fetched without per-query scoring; word-overlap
        # guards against cross-domain tops consuming summary budget in unrelated turns.
        # Pass-through when no query facets are extractable (short/opaque queries).
        _qf_pyramid = extract_query_facets(query)
        _pyramid_candidates = 0
        _pyramid_admitted = 0

        for node in summary_nodes:
            _pyramid_candidates += 1
            if not is_relevant_global_obs_content(node.content, _qf_pyramid):
                log.debug("pyramid_gate: skipping irrelevant top node_id=%s", node.node_id)
                continue
            if tokens_used + node.token_count > summary_budget:
                break
            # Wrap SummaryNode as CItem for context inclusion
            summary_citem = CItem(
                citem_id=node.node_id,
                conversation_id=node.conversation_id,
                content=node.content,
                item_type=ItemType.OBSERVATION,
                scope="global",
                scope_status="active",
                importance=0.6,
                token_count=node.token_count,
            )
            pack.global_summaries.append(summary_citem)
            tokens_used += node.token_count
            _pyramid_admitted += 1

        if _pyramid_candidates > 0:
            log.debug(
                "pyramid_gate: candidates=%d admitted=%d dropped=%d",
                _pyramid_candidates, _pyramid_admitted,
                _pyramid_candidates - _pyramid_admitted,
            )

        # Filter out C-Items already covered by conversation history to avoid
        # surfacing verbatim duplicates in the Memory Context section.
        if history_contents:
            def _not_in_history(item: CItem) -> bool:
                return item.content.strip() not in history_contents

            pack.direct_evidence = [c for c in pack.direct_evidence if _not_in_history(c)]
            pack.bridge_evidence  = [c for c in pack.bridge_evidence  if _not_in_history(c)]

        # Dual-objective relevance filter: drop items that contribute to neither
        # the global objective (original user request) nor the local objective
        # (active plan step). Protected items and conflicts are always kept.
        # Only fires when at least one objective is provided.
        if global_objective or local_objective:
            pack = filter_pack_by_objectives(
                pack,
                global_objective=global_objective or local_objective,
                local_objective=local_objective or global_objective,
            )

        view = context_pack_to_view(pack)
        if summary_modes_by_id:
            for item in view.items:
                if str(item.get("ref_kind") or "") != "summary":
                    continue
                meta = summary_modes_by_id.get(str(item.get("ref_id") or ""))
                if meta:
                    item.update(meta)
        return view
