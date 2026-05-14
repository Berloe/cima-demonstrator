"""BudgetAction + TurnRuntime — ADR-001 v3.4 §state (Phase 7).

TurnRuntime is the sole mutable working state for one agent turn.
BudgetAction is the type-safe enum for BCM tier actions.

Domain events are emitted to the event bus for audit trail / Kafka Phase 2
but are NOT reduced into a snapshot — TurnRuntime is the single source of truth.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

# TurnProgress must be imported at runtime so it can be used as default_factory.
from cima_demo.domain.value_objects import TurnProgress

if TYPE_CHECKING:
    from cima_demo.cognitive.tool_state_guard import ToolStateGuard
    from cima_demo.domain.entities import StallTracker
    from cima_demo.domain.value_objects import (
        ComputeTrace,
        ExecutionStage,
        FinalComputeResult,
        OutputContract,
        SourceRequirement,
        TaskPlan,
        TaskSpec,
        TaskState,
        TurnContract,
        TurnEvidenceRegister,
        TurnOutcome,
    )
    from cima_demo.tools.registry import ToolCall, ToolResult


# ── BudgetAction ──────────────────────────────────────────────────────────────

class LoopMode(str, Enum):
    """Explicit cognitive loop mode — replaces the implicit force_synthesis flag."""
    GATHER    = "gather"      # Mode A: tool calls enabled
    SYNTHESIZE = "synthesize"  # Mode B: no tools, produce final answer


class LoopSignal(str, Enum):
    """Typed return value from cognitive loop phase methods."""
    PROCEED = "proceed"   # continue to next phase within the iteration
    RESTART = "restart"   # restart from top of loop (was "continue")
    EXIT    = "exit"      # exit the loop entirely (was "return")


class BudgetAction(str, Enum):
    """Acción de presupuesto derivada por BudgetPolicy.

    Used by BudgetManager to determine BCM tier escalation actions.
    """
    NONE           = "none"         # dentro de presupuesto
    L1_EVICTABLE   = "l1_evictable" # T1: L1 sobre ítems evictables
    L1_FORCED      = "l1_forced"    # T2: L1 forzado sobre evidencia del turno
    EVICT_CONTEXT  = "evict_context"# T4: evicción greedy de context_view
    IMPASSE        = "impasse"      # T6: descomposición agresiva


# CR-1: consecutive Mode A iterations without compute() before the gate relaxes.
COMPUTE_GATE_PATIENCE: int = 2


# ── TurnRuntime ───────────────────────────────────────────────────────────────

@dataclass
class TurnRuntime:
    """Coordination shell for one agent turn.

    Holds asyncio tasks, LLM protocol buffers, caches, and control-flow
    flags. The typed containers are authoritative:
      - progress (TurnProgress): all evidence/compute/iteration state
      - contract (TurnContract): frozen production specification

    External code accesses progress fields via backward-compat properties
    (ts.artifact_count → ts.progress.artifact_count, etc.).
    """

    # ── Identity — set at turn start by AgentOrchestrator ────────────────────
    conversation_id: str = ""
    turn_id: str = ""
    run_id: str = ""
    user_message: str = ""
    phase: str = ""            # initial cognitive phase (str | CognitivePhase)
    checkpoint_seq: int = 0

    # ── asyncio Tasks — awaited before mutex release ──────────────────────────
    pending_ingest_tasks: list[asyncio.Task[None]] = field(default_factory=list)

    # ── Caches ────────────────────────────────────────────────────────────────
    web_cache: dict[str, tuple[str, datetime]] = field(default_factory=dict)
    fetched_evidence: dict[str, str] = field(default_factory=dict)

    # ── LLM protocol buffers ──────────────────────────────────────────────────
    reasoning_buffer: str = ""
    synthesis_fragment: str = ""

    # ── Stall detection ───────────────────────────────────────────────────────
    stall_tracker: StallTracker | None = None

    # ── Tool tracking ─────────────────────────────────────────────────────────
    tool_results_accumulated: list[ToolResult] = field(default_factory=list)
    tool_call_requests: list[ToolCall] = field(default_factory=list)
    tool_batch_sizes: list[int] = field(default_factory=list)
    state_guard: ToolStateGuard | None = None

    # ── Conclusions / memory tracking ─────────────────────────────────────────
    conclusions_types_seen: list[str] = field(default_factory=list)
    chm_citem_ids: set[str] = field(default_factory=set)
    current_turn_ingested_ids: set[str] = field(default_factory=set)

    # ── Control flow ──────────────────────────────────────────────────────────
    explicit_phase_declared: str | None = None
    assistant_reply_buffer: str = ""
    cited_markers: list[str] = field(default_factory=list)
    demo_need_proposal: dict[str, Any] = field(default_factory=dict)
    demo_memory_proposal: dict[str, Any] = field(default_factory=dict)
    plan_created_emitted: bool = False
    final_synthesis_escape_used: bool = False
    loop_mode: LoopMode = LoopMode.GATHER

    # ── Strategy context ──────────────────────────────────────────────────────
    strategy_ctx: Any | None = None
    strategy_retry_count: int = 0
    strategy_fail_type: str | None = None
    strategy_fail_reason: str | None = None

    # ── RAG evolution genome ──────────────────────────────────────────────────
    rag_genome: Any | None = None

    # ── Timing (INFRA-D-05) ───────────────────────────────────────────────────
    turn_started_at: float = field(default_factory=time.monotonic)
    ttft_at: float | None = None

    # ── Source / compute tracking ─────────────────────────────────────────────
    source_requirements: list[SourceRequirement] = field(default_factory=list)
    compute_outputs: list[tuple[str, int]] = field(default_factory=list)   # (value[:200], result_index)
    output_contract: OutputContract | None = None
    compute_requires_new_evidence: bool = False
    compute_requires_evidence_since: int = 0
    task_state: TaskState | None = None
    task_spec: TaskSpec | None = None

    # ── Execution stage / progress ────────────────────────────────────────────
    execution_stage: ExecutionStage | None = None   # initialized to ExecutionStage.INIT by engine.run_turn()

    # ── BCM state — source of truth for budget tier tracking ────────────────
    budget_tier_reached: int = 0
    synthesis_continuation_count: int = 0
    synthesis_leak_retry_count: int = 0   # incremented each time validator rejects synthesis for tool_call_leak/protocol_tag
    budget_impasse: bool = False

    # ── Task plan — pre-loop planning output + adaptive replanning ───────────
    task_plan: TaskPlan | None = None

    # ── Mode A prose violation tracking ──────────────────────────────────────
    # Counts consecutive iterations where the model emitted text in tool-call
    # mode.  Used by MessageBuilder to inject a correction reminder so the
    # model does not keep wasting generation budget on discarded prose.
    prose_violation_iters: int = 0

    # ── Web rendering ─────────────────────────────────────────────────────────
    render_escalation_urls: set[str] = field(default_factory=set)

    # ── Config — injected at turn start from engine settings ─────────────────
    max_iterations: int = 8
    max_stall_count: int = 5
    max_strategy_retries: int = 1

    # ── Typed turn containers ──────────────────────────────────────────────────
    # contract: frozen production specification (set after task_spec is known)
    # progress: all evidence/compute/iteration state — initialized by default_factory
    contract: "TurnContract | None" = None
    progress: TurnProgress = field(default_factory=TurnProgress)

    # ── Terminal outcome — set once at the canonical exit point ───────────────
    # Persisted in TurnTrace.outcome_code for benchmark / postmortem.
    # Never inferred from text; always set by engine exit logic.
    outcome: "TurnOutcome | None" = None

    # ── Benchmark isolation flag ──────────────────────────────────────────────
    # When True: web_cache is NOT pre-loaded from previous turns, pending_ingest_tasks
    # are not carried over, and the flag is logged in TurnTrace.extra so every
    # isolated run is distinguishable from production runs.
    benchmark_mode: bool = False

    # ── TurnProgress property delegates ──────────────────────────────────────
    # These provide backward-compatible access so all existing code using
    # ts.artifact_count, ts.evidence_register, etc. continues to work unchanged.
    # The canonical storage is TurnProgress; TurnRuntime is the coordination shell.

    @property
    def artifact_count(self) -> int:
        return self.progress.artifact_count

    @artifact_count.setter
    def artifact_count(self, v: int) -> None:
        self.progress.artifact_count = v

    @property
    def resolved_slot_count(self) -> int:
        return self.progress.resolved_slot_count

    @resolved_slot_count.setter
    def resolved_slot_count(self, v: int) -> None:
        self.progress.resolved_slot_count = v

    @property
    def evidence_register(self) -> "TurnEvidenceRegister | None":
        return self.progress.evidence_register

    @evidence_register.setter
    def evidence_register(self, v: "TurnEvidenceRegister | None") -> None:
        self.progress.evidence_register = v

    @property
    def compute_traces(self) -> "list[ComputeTrace]":
        return self.progress.compute_traces

    @compute_traces.setter
    def compute_traces(self, v: "list[ComputeTrace]") -> None:
        self.progress.compute_traces = v

    @property
    def final_compute_result(self) -> "FinalComputeResult | None":
        return self.progress.final_compute_result

    @final_compute_result.setter
    def final_compute_result(self, v: "FinalComputeResult | None") -> None:
        self.progress.final_compute_result = v

    @property
    def has_final_compute_result(self) -> bool:
        return self.progress.has_final_compute_result

    @property
    def compute_done(self) -> bool:
        return self.progress.compute_done

    @compute_done.setter
    def compute_done(self, v: bool) -> None:
        self.progress.compute_done = v

    @property
    def compute_gate_no_compute_iters(self) -> int:
        return self.progress.compute_gate_no_compute_iters

    @compute_gate_no_compute_iters.setter
    def compute_gate_no_compute_iters(self, v: int) -> None:
        self.progress.compute_gate_no_compute_iters = v

    @property
    def stall_hashes(self) -> list[str]:
        return self.progress.stall_hashes

    @stall_hashes.setter
    def stall_hashes(self, v: list[str]) -> None:
        self.progress.stall_hashes = v

    @property
    def stall_occurred(self) -> bool:
        return self.progress.stall_occurred

    @stall_occurred.setter
    def stall_occurred(self, v: bool) -> None:
        self.progress.stall_occurred = v

    @property
    def global_drift_detected(self) -> bool:
        return self.progress.global_drift_detected

    @global_drift_detected.setter
    def global_drift_detected(self, v: bool) -> None:
        self.progress.global_drift_detected = v

    @property
    def last_validation_gaps(self) -> list[str]:
        return self.progress.last_validation_gaps

    @last_validation_gaps.setter
    def last_validation_gaps(self, v: list[str]) -> None:
        self.progress.last_validation_gaps = v

    @property
    def replan_count(self) -> int:
        return self.progress.replan_count

    @replan_count.setter
    def replan_count(self, v: int) -> None:
        self.progress.replan_count = v

    @property
    def search_result_locators(self) -> list[str]:
        return self.progress.search_result_locators

    @search_result_locators.setter
    def search_result_locators(self, v: list[str]) -> None:
        self.progress.search_result_locators = v

    @property
    def iteration_count(self) -> int:
        return self.progress.iteration_count

    @iteration_count.setter
    def iteration_count(self, v: int) -> None:
        self.progress.iteration_count = v

    @property
    def tool_calls_emitted(self) -> list[str]:
        return self.progress.tool_calls_emitted

    @tool_calls_emitted.setter
    def tool_calls_emitted(self, v: list[str]) -> None:
        self.progress.tool_calls_emitted = v

    @property
    def iteration_limit_reached(self) -> bool:
        """True when iteration_count has reached the configured max."""
        return self.progress.iteration_count >= self.max_iterations

