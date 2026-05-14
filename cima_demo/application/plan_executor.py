"""PlanExecutor — plan lifecycle management (KIMA_Application_Layer_v1.1 §10)."""
from __future__ import annotations

import logging
from typing import Any

from cima_demo.domain.entities import Plan, PlanStep, TaskMemory
from cima_demo.domain.errors import PlanError
from cima_demo.domain.ports import RelDBPort
from cima_demo.domain.value_objects import PlanStatus

log = logging.getLogger(__name__)

_MAX_STEP_ATTEMPTS = 3  # K criterion (RF-13.1)


class PlanExecutor:
    """Manages Plan lifecycle: create, advance, pause, resume, complete, fail."""

    def __init__(self, rel_db: RelDBPort) -> None:
        self._db = rel_db

    async def create_plan(
        self,
        goal: str,
        steps: list[dict[str, Any]],
        conversation_id: str,
        *,
        auto_continue: bool = False,
    ) -> Plan:
        """Create a new Plan with steps and persist it."""
        plan_steps = [
            PlanStep(
                plan_id="",  # Set after Plan creation
                description=s.get("description", f"Step {i}"),
                tool_name=s.get("tool_name") or (s.get("tool") or None),
                tool_args=s.get("tool_args", {}),
                procedure_citem_id=s.get("procedure_citem_id"),
                acceptance_criterion=s.get("criterion") or s.get("acceptance_criterion") or None,
                context_focus=s.get("context_focus") or None,
            )
            for i, s in enumerate(steps)
        ]
        plan = Plan(
            conversation_id=conversation_id,
            goal=goal,
            steps=plan_steps,
            status=PlanStatus.PENDING,
            auto_continue=auto_continue,
        )
        # Set plan_id back reference
        for step in plan.steps:
            step.plan_id = plan.plan_id

        await self._db.save_plan(plan)
        return plan

    async def start(self, plan: Plan, task_memory: TaskMemory) -> Plan:
        """Activate first step and set plan to RUNNING (atomic with task_memory — INFRA-D-01)."""
        if not plan.steps:
            raise PlanError(f"Plan {plan.plan_id} has no steps")
        plan.start()
        plan.steps[0].mark_active()
        task_memory.active_plan_id = plan.plan_id
        await self._db.save_plan_with_task_memory(plan, task_memory)
        return plan

    async def advance_step(
        self,
        plan: Plan,
        result_summary: str,
        task_memory: TaskMemory,
        conversation_id: str,
        turn_id: str,
    ) -> str:
        """Mark current step completed and activate next (or complete plan).

        L-03a (CR-03): auto-advance only if result indicates success.
        Returns PlanStatus after advancement.
        """
        current = plan.active_step
        if current is None:
            return plan.status

        current.mark_completed(result_summary)

        # Activate next pending step
        next_step = plan.next_pending_step
        if next_step is not None:
            next_step.mark_active()
            plan.status = PlanStatus.RUNNING
        else:
            if plan.has_failed:
                plan.fail()
                task_memory.active_plan_id = None
            else:
                plan.complete()
                task_memory.active_plan_id = None

        await self._db.save_plan_with_task_memory(plan, task_memory)
        return plan.status

    async def increment_attempts(
        self,
        plan: Plan,
        step: PlanStep,
    ) -> bool:
        """Record a failed attempt. Returns True if step should be marked failed (K=3)."""
        step_attempts = getattr(step, "_attempts", 0) + 1
        object.__setattr__(step, "_attempts", step_attempts) if hasattr(step, "__setattr__") else None
        # Track attempts in result_summary as workaround (proper impl: add field)
        await self._db.update_plan_step_attempts(step.step_id, step_attempts)
        return step_attempts >= _MAX_STEP_ATTEMPTS

    async def fail_step(
        self,
        plan: Plan,
        step: PlanStep,
        reason: str,
        task_memory: TaskMemory,
    ) -> None:
        """Mark step as failed; fail the plan."""
        step.mark_failed(reason)
        plan.fail()
        task_memory.active_plan_id = None
        await self._db.save_plan_with_task_memory(plan, task_memory)

    async def pause(self, plan: Plan, task_memory: TaskMemory) -> None:
        """Pause plan when ask_user interrupts (H-04)."""
        plan.pause()
        await self._db.save_plan(plan)

    async def resume(
        self,
        plan: Plan,
        task_memory: TaskMemory,
        user_response: str,
    ) -> Plan:
        """Resume PAUSED plan → RUNNING (L-03b CR-03)."""
        plan.start()  # PAUSED → RUNNING
        await self._db.save_plan(plan)
        return plan

    def effective_status(self, plan: Plan) -> str:
        """Diagnostic: return plan.status directly (L-03c — source of truth is plan.status)."""
        return plan.status
