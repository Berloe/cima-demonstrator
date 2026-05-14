"""Pre-synthesis readiness helpers kept by the demonstrator runtime.

This module preserves the small amount of planning-related logic that remains
useful for transition policy evaluation after the legacy task planner has been
removed from the active codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cima_demo.domain.value_objects import TaskPlan

if TYPE_CHECKING:  # pragma: no cover
    from cima_demo.domain.value_objects import TurnEvidenceRegister


@dataclass
class SynthesisReadiness:
    """Result of a synthesis-readiness evaluation."""

    ready: bool
    gaps: list[str] = field(default_factory=list)


def check_synthesis_readiness(
    task_plan: TaskPlan,
    evidence_register: "TurnEvidenceRegister | None",
    compute_done: bool,
    artifact_count: int,
    compute_gate_relaxed: bool = False,
) -> SynthesisReadiness:
    """Determine whether the current evidence state is sufficient for synthesis."""
    gaps: list[str] = []

    has_web_evidence = artifact_count > 0 or (
        evidence_register is not None
        and any(e.status in ("indexed", "retrieved") for e in evidence_register.entries)
    )
    has_computed_evidence = compute_done and not task_plan.needs_search
    has_evidence = has_web_evidence or has_computed_evidence

    if task_plan.needs_search and not has_evidence:
        gaps.append(f"No evidence retrieved yet for: {task_plan.objective}")

    if task_plan.needs_compute and not compute_done and not compute_gate_relaxed:
        gaps.append("Computation required but not yet completed")

    if task_plan.plan_steps and not has_evidence:
        gaps.append(f"First plan step not started: {task_plan.plan_steps[0]}")

    return SynthesisReadiness(ready=len(gaps) == 0, gaps=gaps)
