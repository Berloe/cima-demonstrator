"""OrchestrationPolicy — ADR-001 v3.4 §Phase 4.

BudgetPolicy: implements BCM tier escalation logic.
  - Encapsulates BCM thresholds: synthesis_token_floor, max_continuations, max_subtasks.
  - budget_action(tier, continuations, impasse) -> BudgetAction.

OrchestrationPolicyPort: protocol port for pluggable policy (ADR-001 §D-11).
"""
from __future__ import annotations

from dataclasses import dataclass

from cima_demo.cognitive.kernel.state import BudgetAction


# ── OrchestrationPolicyPort ────────────────────────────────────────────────────

class OrchestrationPolicyPort:
    """Protocol port for orchestration policy (ADR-001 §D-11).

    Contract:
      - budget_action() is pure and deterministic.
      - Implementations perform no I/O.
    """

    def budget_action(
        self,
        tier_reached: int,
        synthesis_continuation_count: int,
        impasse_raised: bool,
    ) -> BudgetAction:
        return BudgetAction.NONE


# ── BudgetPolicy ──────────────────────────────────────────────────────────────

@dataclass
class BudgetPolicy(OrchestrationPolicyPort):
    """Budget Continuity Model policy (ADR-001 §BCM).

    Centralizes the three BCM thresholds that were previously module constants
    in engine.py — migrating threshold decisions to a domain object.

    budget_action(tier, continuations, impasse) -> BudgetAction:
      Computes the recommended action from the current BCM state.
    """
    synthesis_token_floor: int = 1_024
    max_synthesis_continuations: int = 2
    max_subtasks: int = 5

    def budget_action(
        self,
        tier_reached: int = 0,
        synthesis_continuation_count: int = 0,
        impasse_raised: bool = False,
    ) -> BudgetAction:
        """Determine appropriate BudgetAction for current BCM state.

        Priority descending (highest tier wins):
          T6 / impasse_raised → IMPASSE
          continuations >= max → IMPASSE (escalation from T5)
          tier >= 4            → EVICT_CONTEXT (T4)
          tier >= 2            → L1_FORCED     (T2)
          tier >= 1            → L1_EVICTABLE  (T1)
          otherwise            → NONE
        """
        if impasse_raised:
            return BudgetAction.IMPASSE
        if synthesis_continuation_count >= self.max_synthesis_continuations:
            return BudgetAction.IMPASSE
        if tier_reached >= 4:
            return BudgetAction.EVICT_CONTEXT
        if tier_reached >= 2:
            return BudgetAction.L1_FORCED
        if tier_reached >= 1:
            return BudgetAction.L1_EVICTABLE
        return BudgetAction.NONE

    # ── Helpers for the engine (access thresholds without coupling) ───────────

    def synthesis_budget(self, available_for_content: int) -> int:
        """Content budget reserving synthesis_token_floor for output."""
        return available_for_content - self.synthesis_token_floor

    def continuations_exhausted(self, count: int) -> bool:
        """True when T5 continuation attempts are exhausted."""
        return count >= self.max_synthesis_continuations
