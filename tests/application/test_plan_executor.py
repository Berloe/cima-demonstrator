"""Tests for PlanExecutor (cima_demo/application/plan_executor.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cima_demo.application.plan_executor import PlanExecutor
from cima_demo.domain.entities import TaskMemory
from cima_demo.domain.errors import PlanError
from cima_demo.domain.value_objects import PlanStatus, StepStatus

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.save_plan = AsyncMock()
    db.save_plan_with_task_memory = AsyncMock()
    db.update_plan_step_attempts = AsyncMock()
    return db


@pytest.fixture
def executor(mock_db: AsyncMock) -> PlanExecutor:
    return PlanExecutor(rel_db=mock_db)


@pytest.fixture
def task_memory() -> TaskMemory:
    return TaskMemory(conversation_id="conv-1")


def _steps() -> list[dict[str, object]]:
    return [
        {"description": "Step 1", "tool_name": "tool_a", "tool_args": {}},
        {"description": "Step 2", "tool_name": "tool_b", "tool_args": {}},
        {"description": "Step 3", "tool_name": "tool_c", "tool_args": {}},
    ]


# ── create_plan ───────────────────────────────────────────────────────────────

class TestCreatePlan:
    async def test_creates_plan_with_correct_goal(
        self, executor: PlanExecutor, mock_db: AsyncMock,
    ) -> None:
        plan = await executor.create_plan("test goal", _steps(), "conv-1")
        assert plan.goal == "test goal"
        assert plan.conversation_id == "conv-1"
        mock_db.save_plan.assert_awaited_once()

    async def test_step_count_matches_input(self, executor: PlanExecutor) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        assert len(plan.steps) == 3

    async def test_plan_id_back_reference_set_on_steps(
        self, executor: PlanExecutor,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        for step in plan.steps:
            assert step.plan_id == plan.plan_id

    async def test_initial_status_is_pending(self, executor: PlanExecutor) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        assert plan.status == PlanStatus.PENDING

    async def test_step_descriptions_preserved(self, executor: PlanExecutor) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        assert plan.steps[0].description == "Step 1"
        assert plan.steps[1].description == "Step 2"


# ── start ──────────────────────────────────────────────────────────────────────

class TestStart:
    async def test_plan_becomes_running(
        self, executor: PlanExecutor, task_memory: TaskMemory, mock_db: AsyncMock,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        assert plan.status == PlanStatus.RUNNING

    async def test_first_step_becomes_active(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        assert plan.steps[0].status == StepStatus.ACTIVE

    async def test_task_memory_active_plan_id_set(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        assert task_memory.active_plan_id == plan.plan_id

    async def test_raises_on_empty_steps(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", [], "conv-1")
        with pytest.raises(PlanError):
            await executor.start(plan, task_memory)

    async def test_save_plan_with_task_memory_called(
        self, executor: PlanExecutor, task_memory: TaskMemory, mock_db: AsyncMock,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        mock_db.save_plan_with_task_memory.assert_awaited_once_with(plan, task_memory)


# ── advance_step ──────────────────────────────────────────────────────────────

class TestAdvanceStep:
    async def test_advances_to_next_step(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        await executor.advance_step(plan, "done", task_memory, "conv-1", "turn-1")
        assert plan.steps[1].status == StepStatus.ACTIVE

    async def test_first_step_marked_completed(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        await executor.advance_step(plan, "result", task_memory, "conv-1", "turn-1")
        assert plan.steps[0].status == StepStatus.COMPLETED

    async def test_all_steps_done_completes_plan(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", [_steps()[0]], "conv-1")
        await executor.start(plan, task_memory)
        status = await executor.advance_step(plan, "done", task_memory, "conv-1", "t")
        assert status == PlanStatus.COMPLETED
        assert task_memory.active_plan_id is None

    async def test_no_active_step_returns_current_status(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        # Don't start — no active step
        status = await executor.advance_step(plan, "done", task_memory, "conv-1", "t")
        assert status == plan.status


# ── fail_step ─────────────────────────────────────────────────────────────────

class TestFailStep:
    async def test_step_marked_failed(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        step = plan.active_step
        assert step is not None
        await executor.fail_step(plan, step, "error occurred", task_memory)
        assert step.status == StepStatus.FAILED

    async def test_plan_marked_failed(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        step = plan.active_step
        assert step is not None
        await executor.fail_step(plan, step, "error", task_memory)
        assert plan.status == PlanStatus.FAILED

    async def test_active_plan_id_cleared(
        self, executor: PlanExecutor, task_memory: TaskMemory,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        step = plan.active_step
        assert step is not None
        await executor.fail_step(plan, step, "error", task_memory)
        assert task_memory.active_plan_id is None


# ── increment_attempts ────────────────────────────────────────────────────────

class TestIncrementAttempts:
    async def test_returns_false_before_max(
        self, executor: PlanExecutor,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        step = plan.steps[0]
        result = await executor.increment_attempts(plan, step)
        assert result is False  # 1 < 3

    async def test_returns_true_at_max(
        self, executor: PlanExecutor,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        step = plan.steps[0]
        for _ in range(2):
            await executor.increment_attempts(plan, step)
        result = await executor.increment_attempts(plan, step)
        assert result is True  # reached K=3


# ── pause / resume ────────────────────────────────────────────────────────────

class TestPauseResume:
    async def test_pause_sets_paused_status(
        self, executor: PlanExecutor, task_memory: TaskMemory, mock_db: AsyncMock,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        await executor.pause(plan, task_memory)
        assert plan.status == PlanStatus.PAUSED
        mock_db.save_plan.assert_awaited()

    async def test_resume_sets_running_status(
        self, executor: PlanExecutor, task_memory: TaskMemory, mock_db: AsyncMock,
    ) -> None:
        plan = await executor.create_plan("goal", _steps(), "conv-1")
        await executor.start(plan, task_memory)
        await executor.pause(plan, task_memory)
        await executor.resume(plan, task_memory, "user response")
        assert plan.status == PlanStatus.RUNNING
