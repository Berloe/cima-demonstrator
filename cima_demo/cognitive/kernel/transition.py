"""TransitionPolicy — single source of truth for turn-level state transitions.

All synthesis readiness logic, stall conditions, and replan decisions live
here.  The engine drives its cognitive loop via TurnRuntime.loop_mode
(LoopMode enum) which is set by each TransitionDecision.

Pure: no IO, no side effects, no mutable state. Fully unit-testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from cima_demo.cognitive.kernel.state import COMPUTE_GATE_PATIENCE, LoopMode
from cima_demo.cognitive.kernel.readiness import check_synthesis_readiness
from cima_demo.domain.value_objects import ExecutionMode

if TYPE_CHECKING:
    from cima_demo.domain.value_objects import TaskPlan, TurnContract, TurnProgress

log = logging.getLogger(__name__)

_MAX_REPLAN_PER_TURN: int = 2   # matches engine._MAX_REPLAN_PER_TURN

__all__ = ["LoopMode", "Transition", "TransitionDecision", "TransitionPolicy"]


class Transition(str, Enum):
    """Possible turn-level transitions returned by TransitionPolicy."""
    ACT               = "act"               # Mode A: gather evidence via tools
    SYNTHESIZE        = "synthesize"        # Mode B: produce final answer
    SYNTHESIZE_PARTIAL = "synthesize_partial"  # Mode B despite gaps (replan cap or forced)
    REPLAN            = "replan"            # request replan, then continue ACT
    STALL             = "stall"             # terminal: no progress
    NO_EVIDENCE       = "no_evidence"       # terminal: no evidence at all
    NO_COMPUTE        = "no_compute"        # terminal: compute required, impossible


@dataclass(frozen=True)
class TransitionDecision:
    """Result of a TransitionPolicy call."""
    transition: Transition
    reason: str = ""
    gaps: tuple[str, ...] = ()   # strategy gaps, populated for REPLAN decisions


@dataclass(frozen=True)
class TransitionPolicy:
    """Stateless policy that derives the next turn-level transition.

    All synthesis readiness logic, stall conditions, and replan decisions
    live here — not scattered across the cognitive loop.

    Usage:
        policy = TransitionPolicy()

        # Before LLM call — decide mode:
        decision = policy.pre_llm(contract, progress, task_plan,
                                  synthesis_requested=(ts.loop_mode == LoopMode.SYNTHESIZE))

        # When Mode A produced no tool calls and no text:
        decision = policy.after_mode_a_empty(contract, progress)

        # After tool batch dispatch:
        decision = policy.after_dispatch(contract, progress, task_plan,
                                         compute_just_done=..., stall_triggered=...)

        # When stall detected:
        decision = policy.on_stall(contract, progress, slot_pending=...)

        # When iteration limit reached:
        decision = policy.on_iteration_limit(has_tool_results=..., has_reply=...)

        # When synthesis validation fails:
        decision = policy.on_synthesis_invalid(error_class=..., retry_count=...)
    """
    compute_gate_patience: int = COMPUTE_GATE_PATIENCE
    max_replan_per_turn: int = _MAX_REPLAN_PER_TURN
    max_synthesis_leak_retries: int = 2

    # ── Public API ────────────────────────────────────────────────────────────

    def pre_llm(
        self,
        contract: "TurnContract",
        progress: "TurnProgress",
        task_plan: "TaskPlan | None",
        *,
        synthesis_requested: bool,
        source_requirements: list | None = None,
    ) -> TransitionDecision:
        """Decide LLM mode before the next call.

        Returns SYNTHESIZE (Mode B, no tools) or ACT (Mode A, with tools).
        Also returns terminal decisions (NO_EVIDENCE, NO_COMPUTE) when synthesis
        is requested but blocked by a hard guard.
        """
        if not synthesis_requested:
            return TransitionDecision(Transition.ACT, "default: gather evidence")
        return self._validate_synthesis(contract, progress, source_requirements)

    def after_mode_a_empty(
        self,
        contract: "TurnContract",
        progress: "TurnProgress",
    ) -> TransitionDecision:
        """Mode A LLM call produced no tool calls and no assistant text.

        Returns ACT (loop back — slot contract still pending) or
        SYNTHESIZE (try synthesis).
        """
        if self._compute_missing(contract, progress):
            return TransitionDecision(
                Transition.ACT,
                "slot_contract pending — loop back so model can call compute",
            )
        return TransitionDecision(
            Transition.SYNTHESIZE,
            "Mode A empty — no pending contract, escalate to synthesis",
        )

    def after_dispatch(
        self,
        contract: "TurnContract",
        progress: "TurnProgress",
        task_plan: "TaskPlan | None",
        *,
        compute_just_done: bool,
        stall_triggered: bool,
    ) -> TransitionDecision:
        """After a tool batch has been dispatched and results recorded in progress.

        Returns ACT, SYNTHESIZE, SYNTHESIZE_PARTIAL, REPLAN, or STALL.
        """
        if stall_triggered:
            return TransitionDecision(Transition.STALL, "stall detected by stall_tracker")

        # Compute just finished → go straight to synthesis; skip the next Mode A
        # round-trip (which the model would use only to emit prose, wasting tokens).
        if compute_just_done:
            return TransitionDecision(
                Transition.SYNTHESIZE,
                "compute done — skip redundant Mode A, synthesize immediately",
            )

        if task_plan is None or progress.artifact_count == 0:
            return TransitionDecision(Transition.ACT, "no plan or no evidence yet — keep gathering")

        readiness = check_synthesis_readiness(
            task_plan,
            progress.evidence_register,
            progress.has_final_compute_result,
            progress.artifact_count,
            compute_gate_relaxed=(
                progress.compute_gate_no_compute_iters >= self.compute_gate_patience
            ),
        )

        if readiness.ready:
            compute_bypassed = progress.compute_gate_no_compute_iters >= self.compute_gate_patience
            reason = "synthesis ready" + (" (compute gate bypassed)" if compute_bypassed else "")
            return TransitionDecision(Transition.SYNTHESIZE, reason)

        # Separate STRATEGY gaps (missing evidence → replan) from
        # PROGRESS gaps (compute pending → model should call compute, no replan).
        strategy_gaps = [
            g for g in readiness.gaps
            if not g.startswith("Computation required")
        ]

        if not strategy_gaps:
            return TransitionDecision(Transition.ACT, "progress gaps only — keep going")

        gaps_changed = set(strategy_gaps) != set(progress.last_validation_gaps)
        if not gaps_changed:
            return TransitionDecision(
                Transition.ACT,
                "strategy gaps unchanged — stall detection will handle if stuck",
            )

        if progress.replan_count >= self.max_replan_per_turn:
            return TransitionDecision(
                Transition.SYNTHESIZE_PARTIAL,
                f"replan cap ({progress.replan_count}/{self.max_replan_per_turn}) — force synthesis",
                gaps=tuple(strategy_gaps),
            )

        return TransitionDecision(
            Transition.REPLAN,
            f"new strategy gaps (replan #{progress.replan_count + 1}/{self.max_replan_per_turn})",
            gaps=tuple(strategy_gaps),
        )

    def on_stall(
        self,
        contract: "TurnContract",
        progress: "TurnProgress",
        *,
        slot_pending: bool,
    ) -> TransitionDecision:
        """When stall_tracker fires — decide whether to force synthesis or keep trying.

        Default: synthesize unless a compute slot contract is still pending
        (SOURCE_BOUND_QUANT without a verified result).
        """
        if slot_pending:
            return TransitionDecision(
                Transition.ACT,
                "stall with slot_contract pending — loop back for tools",
            )
        return TransitionDecision(
            Transition.SYNTHESIZE,
            "stall detected — force synthesis",
        )

    def on_iteration_limit(
        self,
        *,
        has_tool_results: bool,
        has_reply: bool,
        escape_used: bool,
    ) -> TransitionDecision:
        """When iteration count exceeds max_iterations.

        Default: allow one final synthesis escape if tool results exist
        but no reply has been generated yet, then STALL.
        """
        if not escape_used and has_tool_results and not has_reply:
            return TransitionDecision(
                Transition.SYNTHESIZE,
                "iteration limit — final synthesis escape",
            )
        return TransitionDecision(
            Transition.STALL,
            "iteration limit reached — terminal stall",
        )

    def on_synthesis_invalid(
        self,
        error_class: str,
        retry_count: int,
    ) -> TransitionDecision:
        """When synthesis validation fails (tool_call_leak, protocol_tag).

        Default: retry up to max_synthesis_leak_retries, then emit
        TOOL_PROTOCOL_ERROR outcome.
        """
        if retry_count < self.max_synthesis_leak_retries:
            return TransitionDecision(
                Transition.SYNTHESIZE,
                f"synthesis invalid ({error_class}) — retry {retry_count + 1}/{self.max_synthesis_leak_retries}",
            )
        return TransitionDecision(
            Transition.STALL,
            f"synthesis leak retry limit reached ({retry_count}/{self.max_synthesis_leak_retries})",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_synthesis(
        self,
        contract: "TurnContract",
        progress: "TurnProgress",
        source_requirements: list | None = None,
    ) -> TransitionDecision:
        """Check all synthesis guards. Called when synthesis has been requested."""
        # Guard 1: source requirements exist but nothing was fetched.
        unsatisfied = [r for r in (source_requirements or []) if not r.satisfied]
        evidence_absent = (
            bool(unsatisfied)
            and progress.artifact_count == 0
            and progress.resolved_slot_count == 0
        )
        if evidence_absent:
            return TransitionDecision(
                Transition.NO_EVIDENCE,
                f"synthesis blocked: {len(unsatisfied)} unsatisfied source req(s), no artifacts",
            )

        # Guard 2: compute contract is active but compute hasn't completed.
        if self._compute_missing(contract, progress):
            return TransitionDecision(
                Transition.ACT,
                "synthesis blocked: compute required but not yet done — loop back",
            )

        # Guard 3 (CR-6): global drift with zero evidence → hallucination risk.
        if progress.global_drift_detected and progress.artifact_count == 0:
            return TransitionDecision(
                Transition.NO_EVIDENCE,
                "synthesis blocked: global drift + no evidence gathered",
            )

        return TransitionDecision(Transition.SYNTHESIZE, "synthesis guards passed")

    def _compute_missing(
        self,
        contract: "TurnContract",
        progress: "TurnProgress",
    ) -> bool:
        """True when compute is contractually required but not yet completed."""
        gate_relaxed = progress.compute_gate_no_compute_iters >= self.compute_gate_patience
        if gate_relaxed:
            return False
        needs = (
            contract.mode == ExecutionMode.SOURCE_BOUND_QUANT
            or contract.needs_compute
        )
        return needs and not progress.has_final_compute_result
