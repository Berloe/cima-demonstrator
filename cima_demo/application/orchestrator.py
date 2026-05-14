"""AgentOrchestrator — request handler + turn lifecycle for CIMA Demonstrator.

The active demonstrator path is governed by DemoTurnController. The orchestrator
remains the boundary for turn setup, durable run bookkeeping and transcript
persistence, but it no longer owns nor instantiates the legacy frontier engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any

from cima_demo.retrieval.context_builder import ContextBuilder
from cima_demo.memory.service import MemoryService, strip_images_from_text
from cima_demo.application.plan_executor import PlanExecutor
from cima_demo.retrieval.query_planner import TaskSpecBuilder
from cima_demo.application.stream_manager import StreamManager
from cima_demo.domain.entities import (
    IngestRequest,
    KimaDelta,
    Plan,
    TaskMemory,
)
from cima_demo.domain.ports import LLMPort, RelDBPort
from cima_demo.domain.errors import LLMContextOverflowError
from cima_demo.domain.value_objects import (
    CognitivePhase,
    ContextBudget,
    ExecutionMode,
    ItemType,
    KimaDeltaType,
    PlanStatus,
    TaskSlot,
    TaskState,
    TurnContract,
    TurnProgress,
)
from cima_demo.observability import (
    TurnTrace,
    active_turn_dec,
    active_turn_inc,
    emit_turn_trace,
    record_turn_metrics,
)
from cima_demo.cognitive.kernel.ports import DomainEventPublisherPort
from cima_demo.cognitive.kernel.state import COMPUTE_GATE_PATIENCE as _COMPUTE_GATE_PATIENCE, TurnRuntime
from cima_demo.domain.entities import TurnMetadata
from cima_demo.cognitive.source_lock import _detect_source_lock
from cima_demo.cognitive.slot_resolver import _auto_create_task_state_from_locks
from cima_demo.cognitive.final_validator import validate_final_answer
from cima_demo.cognitive.messages import DECOMPOSITION_PLAN_ACK, DECOMPOSITION_FAILED_ACK, TURN_TIMEOUT, ANSWER_FILE_REQUIRED

log = logging.getLogger(__name__)

_MAX_ITERATIONS = 8           # iGPU: ~600s/call → 8 iterations ≈ 80 min safety cap
_MAX_STRATEGY_RETRIES = 1   # RM-INV-07
# Cumulative stall threshold across turns: when stall_count reaches this value the agent
# emits a user-visible message and resets the counter so the next turn starts fresh.
_MAX_STALL_COUNT = 5

# ── Budget Continuity Model constants ─────────────────────────────────────────
# T3: minimum tokens reserved for synthesis output (prevents truncation mid-response).
_SYNTHESIS_TOKEN_FLOOR = 1_024
# T5: maximum continuation attempts before escalating to T6.
_MAX_SYNTHESIS_CONTINUATIONS = 2
# T6: maximum atomic subtasks in an aggressive decomposition plan.
_T6_MAX_SUBTASKS = 5

# NOTE: _PSEUDO_TOOLS, TurnMetadata, ToolStateGuard, module-level helpers and
# BCM constants live in cima_demo application runtime (legacy engine removed) (Phase 6).
# The active working state is TurnRuntime (cima_demo.cognitive.kernel.state).


class AgentOrchestrator:
    """Main cognitive loop (RF-03, DD-08, KIMA_Application_Layer_v1.1 §4).

    One instance per deployment; conversation state lives in TaskMemory/TurnRuntime.
    """

    def __init__(
        self,
        llm_port: LLMPort,
        rel_db: RelDBPort,
        memory_service: MemoryService,
        context_builder: ContextBuilder,
        stream_manager: StreamManager,
        plan_executor: PlanExecutor,
        context_budget: ContextBudget,
        system_prompt_factory: Callable[..., str],
        turn_timeout_secs: int = 7200,
        llm_max_tokens: int = 0,
        llm_max_tokens_tool: int = 0,
        llm_temperature: float = 0.2,
        llm_temperature_tool: float = 0.0,
        llm_repeat_penalty: float = 1.0,
        llm_top_p: float = 1.0,
        llm_vision: bool = False,
        max_iterations: int = _MAX_ITERATIONS,
        max_stall_count: int = _MAX_STALL_COUNT,
        max_strategy_retries: int = _MAX_STRATEGY_RETRIES,
        domain_event_publisher: DomainEventPublisherPort | None = None,
        demo_run_journal: Any | None = None,
        demo_turn_controller: Any | None = None,
    ) -> None:
        self._domain_events = domain_event_publisher
        self._alias_provider = None
        self._llm = llm_port
        self._db = rel_db
        self._memory = memory_service
        self._ctx = context_builder
        self._stream = stream_manager
        self._planner = plan_executor
        self._budget = context_budget
        self._prompt_factory = system_prompt_factory
        self._task_spec_builder = TaskSpecBuilder()
        self._turn_timeout_secs = turn_timeout_secs
        self._llm_max_tokens: int | None = llm_max_tokens if llm_max_tokens > 0 else None
        self._llm_max_tokens_tool: int | None = llm_max_tokens_tool if llm_max_tokens_tool > 0 else None
        self._llm_temperature: float = llm_temperature
        self._llm_temperature_tool: float = llm_temperature_tool
        self._llm_repeat_penalty: float = llm_repeat_penalty
        self._llm_top_p: float = llm_top_p
        self._llm_vision: bool = llm_vision
        self._max_iterations:       int = max(1, max_iterations)
        self._max_stall_count:      int = max(1, max_stall_count)
        self._max_strategy_retries: int = max(0, max_strategy_retries)
        self._demo_runs = demo_run_journal
        self._demo_controller = demo_turn_controller
        self._demo_lineage = getattr(memory_service, "_lineage", None)

    async def handle_turn(
        self,
        conversation_id: str,
        user_message: str,
        attached_files: list[tuple[bytes, str, str]] | None = None,
        context_budget_override: ContextBudget | None = None,
        llm_max_tokens_override: int | None = None,
    ) -> None:
        """Execute a full agent turn.

        Pre:  turn_in_progress=False (enforced by API layer mutex)
        Post: TaskMemory + TurnMetadata persisted; turn_in_progress=False
        """
        turn_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        run_manifest: Any | None = None
        _context_bind_token: Any | None = None
        _user_source: Any | None = None
        _user_span: Any | None = None
        _attached_meta = [
            {
                "filename": fname,
                "mime_type": mime,
                "size_bytes": len(data),
            }
            for data, fname, mime in (attached_files or [])
        ]

        # Emit an immediate REASONING delta so the SSE client (e.g. LibreChat) does not
        # time out waiting for content while we run DB queries and the strategy classifier.
        # The queue subscriber is guaranteed to exist before this task is scheduled (API-INV-01).
        try:
            await self._stream.publish(KimaDelta(
                type=KimaDeltaType.REASONING,
                conversation_id=conversation_id,
                token="⟳ Procesando...\n",
            ))
        except Exception:
            pass  # never block the turn over a stream emit

        # Load persistent state — parallel to save one round-trip (Step 6.4)
        _task_mem_raw, _tm_raw = await asyncio.gather(
            self._db.load_task_memory(conversation_id),
            self._db.load_turn_metadata(conversation_id),
        )
        task_memory = _task_mem_raw or self._default_task_memory(conversation_id)
        turn_metadata = TurnMetadata.from_dict(_tm_raw) if _tm_raw else None

        if self._demo_runs is not None:
            run_manifest = await self._demo_runs.open_skeleton_run(
                run_id=run_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                user_message=user_message,
                attached_files=_attached_meta,
            )
        if hasattr(self._ctx, "bind_run"):
            try:
                _context_bind_token = self._ctx.bind_run(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    query_text=user_message,
                )
            except Exception:
                log.exception("demo context bind failed for %s", conversation_id)

        # Resolve awaiting_user_input
        _was_awaiting = task_memory.awaiting_user_input
        if task_memory.awaiting_user_input:
            task_memory.awaiting_user_input = False
            await self._db.save_task_memory(task_memory)

        # ── File ingestion — visible in thinking panel ────────────────────────
        # HCR-1: log the exact file list received at the orchestrator boundary.
        # This distinguishes "file lost before orchestrator" from "file lost inside
        # the cognitive loop". If input_has_file=false appears in TURN_TRACE but
        # this log shows files present, the loss is downstream of this point.
        if attached_files:
            log.info(
                "orchestrator: attached_files received conv=%s count=%d files=%s",
                conversation_id,
                len(attached_files),
                [(fname, len(data)) for data, fname, _mime in attached_files],
            )
        else:
            log.debug("orchestrator: no attached_files conv=%s", conversation_id)

        # Emits THOUGHT + awaits ingestion + emits TOOL_RESULT before the
        # cognitive loop so chunks are queryable from Qdrant during retrieval.
        if attached_files:
            await self._ingest_files_with_emit(
                attached_files, conversation_id, user_message, turn_id
            )

        # Ingest user message as OBSERVATION — awaited before cognitive loop
        # (no UI representation; internal bookkeeping only).
        if self._demo_lineage is not None:
            try:
                _user_source, _user_span = await self._demo_lineage.register_text_source(
                    conversation_id=conversation_id,
                    source_kind="chat_user",
                    role="user",
                    display_text=user_message,
                    process_text=user_message,
                    origin_ref=turn_id,
                    metadata={"run_id": run_id},
                )
            except Exception:
                log.exception("demo user source registration failed for %s", conversation_id)
        try:
            await self._memory.ingest_citem(IngestRequest(
                content=user_message,
                item_type=ItemType.OBSERVATION,
                phase_ingested=CognitivePhase.IDLE,
                actor="user",
                conversation_id=conversation_id,
                motivation="Incoming user message",
                confidence=1.0,
                source_id=_user_source.source_id if _user_source is not None else None,
                source_span_ids=[_user_span.span_id] if _user_span is not None else [],
                lineage_meta={"kind": "chat_user"},
            ))
        except Exception:
            log.exception("user message ingest failed for %s", conversation_id)

        async def _load_plan_maybe() -> Plan | None:
            if not task_memory.active_plan_id:
                return None
            return await self._db.load_plan(task_memory.active_plan_id)

        # Load active plan only; the demonstrator no longer depends on strategy
        # selection or RAG genome branches from the legacy frontier runtime.
        plan = await _load_plan_maybe()
        strategy_ctx = None
        rag_genome = None
        if _was_awaiting and plan is not None and plan.status == PlanStatus.PAUSED:
            plan = await self._planner.resume(plan, task_memory, user_message)

        # Detect cognitive phase (constant for the turn — A-8.3)
        phase = self._detect_phase(task_memory, plan, turn_metadata)

        rt = TurnRuntime(
            conversation_id=conversation_id,
            turn_id=turn_id,
            run_id=run_id,
            user_message=user_message,
            phase=str(phase),
            strategy_ctx=strategy_ctx,
            rag_genome=rag_genome,
            max_iterations=self._max_iterations,
            max_stall_count=self._max_stall_count,
            max_strategy_retries=self._max_strategy_retries,
        )
        # Derive task spec — drives the entire turn (tools, slots, compute gate, drift)
        rt.task_spec = self._task_spec_builder.build(
            user_message=user_message,
            has_attachment=bool(attached_files),
        )
        log.debug(
            "TaskSpec conv=%s mode=%s schema=%s",
            conversation_id, rt.task_spec.mode, rt.task_spec.answer_schema,
        )
        rt.output_contract = rt.task_spec.output_contract

        if run_manifest is not None:
            self._sync_demo_manifest(
                run_manifest,
                task_memory=task_memory,
                rt=rt,
                plan=plan,
                status="running",
            )
            await self._demo_runs.update_manifest(run_manifest)

        # CR-7: ATTACHMENT_REQUIRED mode but no file actually attached.
        # The regex in _ATTACH_EXTS_RE can trigger this mode from text alone
        # (e.g. "analyze the CSV").  Without an actual file the workspace is
        # empty and the agent would iterate in vacuum.  Ask for the file early
        # instead of consuming the full iteration budget.
        if rt.task_spec.mode == ExecutionMode.ATTACHMENT_REQUIRED and not attached_files:
            log.warning(
                "CR-7: ATTACHMENT_REQUIRED but no file attached — asking user conv=%s",
                conversation_id,
            )
            await self._stream.publish(KimaDelta(
                type=KimaDeltaType.TOKEN,
                conversation_id=conversation_id,
                token=ANSWER_FILE_REQUIRED,
            ))
            if run_manifest is not None:
                self._sync_demo_manifest(
                    run_manifest,
                    task_memory=task_memory,
                    rt=rt,
                    plan=plan,
                    status="blocked",
                    assistant_reply=ANSWER_FILE_REQUIRED,
                )
                _early_cp = await self._demo_runs.checkpoint(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    checkpoint_kind="EARLY_EXIT",
                    state=self._build_demo_checkpoint_state(
                        stage="EARLY_EXIT",
                        task_memory=task_memory,
                        rt=rt,
                        plan=plan,
                    ),
                )
                run_manifest.checkpoint_count = _early_cp.sequence
                run_manifest.finished_at = datetime.now(UTC)
                await self._demo_runs.finalize_run(run_manifest)
            return

        _aliases = self._alias_provider.get_aliases() if self._alias_provider is not None else None
        rt.source_requirements = _detect_source_lock(user_message, aliases=_aliases)
        if rt.source_requirements:
            log.debug(
                "source_requirements detected conv=%s: %s",
                conversation_id, [(r.kind, r.value) for r in rt.source_requirements],
            )
            _auto_create_task_state_from_locks(rt)

        # Source-bound quantitative turns: slot contract stays opt-in (False) for
        # Mistral tool-call mode compatibility — finish_reason=tool_calls produces no
        # text content, so <conclusions>/SLOT declarations are impossible in Mode A.
        #
        # Compensation mechanism (equivalent guarantees, no deadlock):
        #   • Pre-populated slots from task_spec.slot_names give ready_to_compute=False
        #     even with slot_contract_required=False, because:
        #       ready_to_compute = (not slots) or all(s.resolved) → False when unresolved
        #   • Pre-evidence gate (artifact_count==0) blocks compute before any fetch.
        #   • _requires_final_compute includes SOURCE_BOUND_QUANT → synthesis blocked
        #     until has_final_compute_result.
        #   • _try_update_slots_from_evidence promotes slots via backend keyword scan
        #     after fetch, so the model doesn't need text declarations in Mode A.
        # DIRECT_ARITHMETIC: compute allowed immediately — no slot pre-population.
        if (
            rt.task_spec.mode == ExecutionMode.SOURCE_BOUND_QUANT
            and rt.task_state is None
        ):
            _pre_slots = [
                TaskSlot(
                    name=n,
                    description=n.replace("_", " "),
                    unit=rt.task_spec.answer_schema.unit,
                )
                for n in rt.task_spec.slot_names
            ]
            assert _pre_slots, (
                f"SOURCE_BOUND_QUANT invariant violated: task_spec.slot_names is empty "
                f"(conv={conversation_id}). TaskSpecBuilder._derive_sbq_slot_names "
                f"must always return at least one slot name for SOURCE_BOUND_QUANT mode."
            )
            rt.task_state = TaskState(
                objective=user_message,
                slot_contract_required=False,  # Mistral tool-call compat; see above
                slots=_pre_slots,
                output_contract=rt.output_contract,
            )
            log.debug(
                "TaskState shell created (SOURCE_BOUND_QUANT) conv=%s "
                "slots=%s source_lock=%s",
                conversation_id,
                [s.name for s in _pre_slots],
                bool(rt.source_requirements),
            )
        elif (
            rt.task_spec.mode == ExecutionMode.PROMPT_CONTAINED_QUANT
            and rt.task_state is None
        ):
            # All inputs are in the prompt — compute allowed immediately, no fetch required.
            rt.task_state = TaskState(
                objective=user_message,
                slot_contract_required=False,
                output_contract=rt.output_contract,
            )
            log.debug(
                "TaskState created (PROMPT_CONTAINED_QUANT, free compute) conv=%s", conversation_id,
            )
        elif (
            rt.task_spec.mode == ExecutionMode.DIRECT_ARITHMETIC
            and rt.task_state is None
        ):
            # Free compute — no slot contract, no fetch required
            rt.task_state = TaskState(
                objective=user_message,
                slot_contract_required=False,
                output_contract=rt.output_contract,
            )
            log.debug(
                "TaskState created (DIRECT_ARITHMETIC, free compute) conv=%s", conversation_id,
            )

        if rt.task_state is not None:
            rt.task_state.apply_output_contract(rt.output_contract)

        if run_manifest is not None:
            self._sync_demo_manifest(
                run_manifest,
                task_memory=task_memory,
                rt=rt,
                plan=plan,
                status="running",
            )
            _phase_seq = await self._demo_runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name="BOOTSTRAPPED",
                payload={
                    "cognitive_phase": str(phase),
                    "execution_mode": rt.task_spec.mode.value if rt.task_spec else None,
                    "has_task_state": rt.task_state is not None,
                    "has_source_requirements": bool(rt.source_requirements),
                },
            )
            run_manifest.phase_count = _phase_seq
            _cp = await self._demo_runs.checkpoint(
                run_id=run_id,
                conversation_id=conversation_id,
                checkpoint_kind="BOOTSTRAP",
                state=self._build_demo_checkpoint_state(
                    stage="BOOTSTRAP",
                    task_memory=task_memory,
                    rt=rt,
                    plan=plan,
                ),
            )
            rt.checkpoint_seq = _cp.sequence
            run_manifest.checkpoint_count = _cp.sequence
            await self._demo_runs.update_manifest(run_manifest)

        # rt.pending_ingest_tasks collects only conclusions tasks added during
        # the cognitive loop; gathered in finally before mutex release.

        # Activate turn mutex (INV-15)
        task_memory.begin_turn(phase)
        await self._db.save_task_memory(task_memory)

        if run_manifest is not None:
            _phase_seq = await self._demo_runs.append_phase(
                run_id=run_id,
                conversation_id=conversation_id,
                phase_name=("DEMO_RUNTIME_PREPARED" if self._demo_controller is not None else "ENGINE_RUNNING"),
                payload={
                    "iteration_budget": rt.max_iterations,
                    "cognitive_phase": str(phase),
                    "runtime_driver": ("demo_controller" if self._demo_controller is not None else "engine"),
                },
            )
            run_manifest.phase_count = _phase_seq

        _turn_start = time.monotonic()
        turn_exception: Exception | None = None
        active_turn_inc()
        try:
            if self._demo_controller is None:
                raise RuntimeError("DemoTurnController is required for the active demonstrator runtime")
            if run_manifest is not None:
                _phase_seq = await self._demo_runs.append_phase(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    phase_name="DEMO_CONTROLLER_RUNNING",
                    payload={
                        "execution_mode": rt.task_spec.mode.value if rt.task_spec else None,
                        "controller": type(self._demo_controller).__name__,
                    },
                )
                run_manifest.phase_count = _phase_seq
            try:
                _demo_turn_coro = self._demo_controller.run_turn(
                    rt,
                    task_memory,
                    plan,
                    context_budget_override=context_budget_override,
                    llm_max_tokens_override=llm_max_tokens_override,
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                # Compatibility for tests and older injected demo controllers.
                _demo_turn_coro = self._demo_controller.run_turn(rt, task_memory, plan)
            await asyncio.wait_for(
                _demo_turn_coro,
                timeout=self._turn_timeout_secs or None,
            )
        except asyncio.TimeoutError as exc:
            turn_exception = exc
            log.warning(
                "handle_turn TIMEOUT after %.0fs for %s",
                self._turn_timeout_secs, conversation_id,
            )
            await self._publish_error_delta(
                conversation_id=conversation_id,
                code="TURN_TIMEOUT",
                message=TURN_TIMEOUT.format(timeout=self._turn_timeout_secs),
            )
        except LLMContextOverflowError as exc:
            turn_exception = exc
            await self._publish_error_delta(conversation_id=conversation_id, code="CONTEXT_OVERFLOW", message=str(exc))
        except Exception as exc:
            turn_exception = exc
            log.exception("handle_turn failed for %s", conversation_id)
            await self._publish_error_delta(conversation_id=conversation_id, code="INTERNAL_ERROR", message=str(exc))
        finally:
            active_turn_dec()
            # Stop llama.cpp inference immediately when the turn is cancelled
            # (e.g. user pressed the stop button — SSE disconnect → CancelledError).
            # abort() is synchronous and fire-and-forget: safe during CancelledError.
            self._llm.abort()

            # INV-18: persist metadata → release mutex.
            # Each step is wrapped individually so that a failure in any one step
            # cannot prevent save_task_memory from running (mutex release).
            meta = TurnMetadata(
                conversation_id=conversation_id,
                tool_calls_emitted=rt.tool_calls_emitted,
                conclusions_item_types=rt.conclusions_types_seen,
                explicit_phase_declared=rt.explicit_phase_declared,
            )
            try:
                await self._db.save_turn_metadata(conversation_id, meta.to_json())
            except Exception:
                log.exception("save_turn_metadata failed for %s", conversation_id)

            # Always save a turn record so history has no gaps.
            # If synthesis was never reached (timeout, stall, cancellation) but
            # tools were called, save a compact trace so the next turn knows what
            # was attempted and can avoid repeating identical work.
            _reply = rt.assistant_reply_buffer
            if not _reply and rt.tool_calls_emitted:
                _tools_summary = ", ".join(
                    f"{name}×{rt.tool_calls_emitted.count(name)}"
                    for name in dict.fromkeys(rt.tool_calls_emitted)  # preserve order, dedupe
                )
                _reply = f"*(turn ended without response — tools called: {_tools_summary})*"
            _assistant_source = None
            _assistant_span = None
            if _reply:
                try:
                    await self._db.append_turn(
                        conversation_id=conversation_id,
                        user_message=user_message,
                        assistant_message=_reply,
                    )
                except Exception:
                    log.exception("append_turn failed for %s", conversation_id)
                if self._demo_lineage is not None:
                    try:
                        _assistant_source, _assistant_span = await self._demo_lineage.register_text_source(
                            conversation_id=conversation_id,
                            source_kind="chat_assistant",
                            role="assistant",
                            display_text=_reply,
                            process_text=_reply,
                            origin_ref=turn_id,
                            metadata={"run_id": run_id},
                        )
                    except Exception:
                        log.exception("demo assistant source registration failed for %s", conversation_id)
                if hasattr(self._ctx, "last_snapshot_id") and self._demo_lineage is not None:
                    try:
                        _context_id = self._ctx.last_snapshot_id()
                        if _context_id:
                            _snapshot = await self._db.load_demo_context_snapshot(_context_id)
                            if _snapshot is not None:
                                await self._demo_lineage.record_answer_lineage(
                                    conversation_id=conversation_id,
                                    run_id=run_id,
                                    response_turn_id=turn_id,
                                    context_id=_context_id,
                                    answer_text=_reply,
                                    cited_markers=list(getattr(rt, "cited_markers", []) or []),
                                    selected_items=list(_snapshot.get("items", [])),
                                )
                    except Exception:
                        log.exception("demo answer lineage failed for %s", conversation_id)

            # Await pending ingestion tasks BEFORE releasing the mutex.
            # Conclusions ingestion (ingest_batch) runs concurrently with streaming;
            # awaiting here guarantees C-Items are in Qdrant before the next turn's
            # retrieval runs, eliminating the inter-turn race condition.
            if rt.pending_ingest_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*rt.pending_ingest_tasks, return_exceptions=True),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "pending_ingest_tasks timed out (>10s) for %s — "
                        "%d tasks may be incomplete; mutex released anyway",
                        conversation_id, len(rt.pending_ingest_tasks),
                    )
                except Exception:
                    log.exception("pending_ingest_tasks gather failed for %s", conversation_id)

            # M-02: fallback OBSERVATION if no <conclusions> emitted — awaited directly
            # (not fire-and-forget) so it also commits before mutex release.
            if not rt.conclusions_types_seen and rt.assistant_reply_buffer.strip():
                try:
                    await self._memory.ingest_citem(IngestRequest(
                        content=rt.assistant_reply_buffer,
                        item_type=ItemType.OBSERVATION,
                        phase_ingested=rt.phase,
                        actor="agent",
                        conversation_id=conversation_id,
                        motivation="Agent response without <conclusions>",
                        confidence=0.7,
                        source_id=_assistant_source.source_id if _assistant_source is not None else None,
                        source_span_ids=[_assistant_span.span_id] if _assistant_span is not None else [],
                        lineage_meta={"kind": "chat_assistant_fallback"},
                    ))
                except Exception:
                    log.exception("M-02 fallback ingest failed for %s", conversation_id)

            # Evidence trace — store evidence register as a retrievable C-Item so
            # the user can request source citations for this turn via memory.
            # Stored only when evidence was actually gathered this turn.
            _ev_reg = getattr(rt, "evidence_register", None)
            if _ev_reg is not None and not _ev_reg.is_empty and rt.assistant_reply_buffer.strip():
                _task_obj = (
                    rt.task_plan.objective
                    if getattr(rt, "task_plan", None) is not None
                    else rt.user_message
                )
                _trace_content = (
                    f"Evidence trace — {_task_obj}\n\n"
                    f"{_ev_reg.context_summary()}\n\n"
                    f"Answer: {rt.assistant_reply_buffer.strip()[:500]}"
                )
                try:
                    await self._memory.ingest_citem(IngestRequest(
                        content=_trace_content,
                        item_type=ItemType.OBSERVATION,
                        phase_ingested=rt.phase,
                        actor="agent",
                        conversation_id=conversation_id,
                        motivation="evidence_trace",
                        confidence=1.0,
                    ))
                except Exception:
                    log.exception("Evidence trace ingest failed for %s", conversation_id)

            # INFRA-D-05: compute and emit TurnMetrics before mutex release
            try:
                _n_tools      = len(rt.tool_calls_emitted)
                _diversity    = len(set(rt.tool_calls_emitted))
                _n_success    = sum(1 for r in rt.tool_results_accumulated if r.success)
                _n_results    = len(rt.tool_results_accumulated)
                _success_rate = round(_n_success / _n_results, 3) if _n_results > 0 else 1.0
                _depth        = min(_n_tools / 2, 3.0)
                _width        = min(_diversity, 3.0)
                _complexity   = round((_depth * _width) / 2, 2)
                _now          = time.monotonic()
                _e2e_ms       = round((_now - rt.turn_started_at) * 1000)
                _ttft_ms      = (
                    round((rt.ttft_at - rt.turn_started_at) * 1000)
                    if rt.ttft_at is not None else None
                )
                _metrics = {
                    "action_count":      _n_tools,
                    "tool_diversity":    _diversity,
                    "tools_used":        list(set(rt.tool_calls_emitted)),
                    "step_success_rate": _success_rate,
                    "complexity_score":  _complexity,
                    "e2e_ms":            _e2e_ms,
                    "ttft_ms":           _ttft_ms,
                }
                log.info("TurnMetrics %s: %s", conversation_id, json.dumps(_metrics))
                record_turn_metrics(_metrics)
                await self._stream.publish(KimaDelta(
                    type=KimaDeltaType.THOUGHT,
                    conversation_id=conversation_id,
                    tool_name="turn_metrics",
                    thought=json.dumps(_metrics),
                ))

                # TurnTrace — structured per-turn audit record
                _final_answer = rt.assistant_reply_buffer.strip()
                # CR-1: relax compute requirement when patience threshold reached
                _compute_gate_relaxed = (
                    rt.compute_gate_no_compute_iters >= _COMPUTE_GATE_PATIENCE
                )
                _vr = validate_final_answer(
                    _final_answer,
                    requires_evidence=(
                        rt.task_spec is not None
                        and rt.task_spec.answer_schema.required_evidence
                    ),
                    resolved_slot_count=rt.resolved_slot_count,
                    artifact_count=rt.artifact_count,
                    slot_contract_required=(
                        (
                            (rt.task_state is not None and rt.task_state.slot_contract_required)
                            or bool(rt.task_plan is not None and rt.task_plan.needs_compute)
                        )
                        and not _compute_gate_relaxed
                    ),
                    compute_done=rt.has_final_compute_result,
                    final_compute_result=rt.final_compute_result,
                ) if _final_answer else None
                _exec_stage_val = (
                    rt.execution_stage.value
                    if hasattr(rt.execution_stage, "value")
                    else str(rt.execution_stage) if rt.execution_stage is not None else "INIT"
                )
                emit_turn_trace(TurnTrace(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    mode=rt.task_spec.mode.value if rt.task_spec else None,
                    execution_stage=_exec_stage_val,
                    input_has_file=bool(attached_files),
                    iteration_count=rt.iteration_count,
                    artifact_count=rt.artifact_count,
                    resolved_slot_count=rt.resolved_slot_count,
                    compute_done=rt.compute_done,
                    answer_valid=_vr.valid if _vr else bool(_final_answer),
                    answer_error_class=_vr.error_class if _vr else None,
                    e2e_ms=_e2e_ms,
                    ttft_ms=_ttft_ms,
                    tools_used=list(dict.fromkeys(rt.tool_calls_emitted)),
                    stage_transitions=[_exec_stage_val],
                    final_error_class=(
                        type(turn_exception).__name__ if turn_exception else None
                    ),
                    protocol_valid=_vr.valid if _vr else True,
                    outcome_code=rt.outcome.value if rt.outcome is not None else None,
                    extra={"benchmark_mode": True} if rt.benchmark_mode else {},
                ))
            except Exception:
                log.debug("TurnMetrics emit failed (non-fatal) for %s", conversation_id)

            # CRITICAL: finish_turn() sets turn_in_progress=False; save_task_memory
            # persists that — this is the mutex release. Must always run.
            task_memory.finish_turn()
            try:
                await self._db.save_task_memory(task_memory)
            except Exception:
                log.exception(
                    "save_task_memory failed for %s — mutex may be stuck; "
                    "attempting explicit release",
                    conversation_id,
                )
                try:
                    await self._db.release_turn_in_progress(conversation_id)
                except Exception:
                    log.exception(
                        "release_turn_in_progress also failed for %s — "
                        "conversation will remain locked until PG recovers",
                        conversation_id,
                    )

            if run_manifest is not None:
                _phase_seq = await self._demo_runs.append_phase(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    phase_name="FINALIZING",
                    payload={
                        "error_class": type(turn_exception).__name__ if turn_exception else None,
                        "assistant_reply_chars": len(rt.assistant_reply_buffer),
                    },
                )
                run_manifest.phase_count = _phase_seq
                _final_cp = await self._demo_runs.checkpoint(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    checkpoint_kind="FINAL_STATE",
                    state=self._build_demo_checkpoint_state(
                        stage="FINAL_STATE",
                        task_memory=task_memory,
                        rt=rt,
                        plan=plan,
                    ),
                )
                rt.checkpoint_seq = _final_cp.sequence
                _current_task = asyncio.current_task()
                _was_cancelled = bool(_current_task is not None and _current_task.cancelling())
                _run_status = (
                    "cancelled" if _was_cancelled else
                    "failed" if turn_exception is not None else
                    "completed"
                )
                self._sync_demo_manifest(
                    run_manifest,
                    task_memory=task_memory,
                    rt=rt,
                    plan=plan,
                    status=_run_status,
                    assistant_reply=rt.assistant_reply_buffer,
                    error_class=type(turn_exception).__name__ if turn_exception else None,
                )
                run_manifest.finished_at = datetime.now(UTC)
                run_manifest.checkpoint_count = _final_cp.sequence
                await self._demo_runs.finalize_run(run_manifest)

            if _context_bind_token is not None and hasattr(self._ctx, "reset_run"):
                try:
                    self._ctx.reset_run(_context_bind_token)
                except Exception:
                    log.exception("demo context reset failed for %s", conversation_id)

            # Persist CHM refs for this turn, then check promotions.
            # save_chm_refs must complete before _run_promotions reads them.
            if rt.chm_citem_ids:
                try:
                    await self._db.save_chm_refs(conversation_id, list(rt.chm_citem_ids))
                except Exception:
                    log.exception("save_chm_refs failed for %s", conversation_id)
            asyncio.create_task(self._run_promotions(conversation_id))

            log.info(
                "handle_turn DONE — conv=%s, total_turn=%.1fs, reply_chars=%d, error=%s",
                conversation_id,
                time.monotonic() - _turn_start,
                len(rt.assistant_reply_buffer),
                type(turn_exception).__name__ if turn_exception else "none",
            )

            # Emit DONE delta
            try:
                await self._stream.publish(KimaDelta(
                    type=KimaDeltaType.DONE,
                    conversation_id=conversation_id,
                ))
            except Exception:
                log.exception("publish DONE failed for %s", conversation_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _publish_error_delta(self, *, conversation_id: str, code: str, message: str) -> None:
        try:
            await self._stream.publish(KimaDelta(
                type=KimaDeltaType.ERROR,
                conversation_id=conversation_id,
                error_code=code,
                error_message=message,
            ))
        except Exception:
            log.exception("publish ERROR failed for %s", conversation_id)


    async def _run_promotions(self, conversation_id: str) -> None:
        """Fire-and-forget per-turn promotion check (runs for active conversations)."""
        try:
            chm_counts = await self._db.load_chm_refs(conversation_id)
            await self._memory.check_promotions(conversation_id, chm_counts)
        except Exception:
            log.debug("_run_promotions failed (non-fatal) for %s", conversation_id)

    async def _ingest_files_with_emit(
        self,
        attached_files: list[tuple[bytes, str, str]],
        conversation_id: str,
        user_message: str,
        turn_id: str,
    ) -> None:
        """Ingest uploaded files with THOUGHT/TOOL_RESULT deltas in the thinking panel.

        Runs synchronously (awaited) before the cognitive loop so chunks are
        indexed in Qdrant and available for retrieval in the current turn.
        """
        file_names = [fname for _, fname, _ in attached_files]
        total_kb = sum(len(b) for b, _, _ in attached_files) / 1024

        await self._stream.publish(KimaDelta(
            type=KimaDeltaType.THOUGHT,
            conversation_id=conversation_id,
            tool_name="ingest_files",
            thought=json.dumps(
                {"files": file_names, "total_kb": round(total_kb, 1)},
                ensure_ascii=False,
            ),
        ))

        # Progress callback: emit each status message as a REASONING delta
        # so the user sees real-time feedback during extraction and chunking.
        async def _progress(msg: str) -> None:
            try:
                await self._stream.publish(KimaDelta(
                    type=KimaDeltaType.REASONING,
                    conversation_id=conversation_id,
                    token=msg + "\n",
                ))
            except Exception:
                pass

        error_msg: str | None = None
        try:
            await self._memory.ingest_files(
                attached_files, conversation_id, user_message, turn_id,
                progress_cb=_progress,
            )
        except Exception as exc:
            log.exception("ingest_files failed for %s", conversation_id)
            error_msg = str(exc)[:200]

        n = len(attached_files)
        summary = (
            error_msg
            if error_msg
            else f"{n} file{'s' if n != 1 else ''} indexed — content available for retrieval."
        )
        await self._stream.publish(KimaDelta(
            type=KimaDeltaType.TOOL_RESULT,
            conversation_id=conversation_id,
            tool_name="ingest_files",
            tool_result=None if error_msg else summary,
            error_message=error_msg,
        ))

    def _default_task_memory(self, conversation_id: str) -> TaskMemory:
        return TaskMemory(conversation_id=conversation_id)

    def _detect_phase(
        self,
        task_memory: TaskMemory,
        plan: Plan | None,
        turn_metadata: TurnMetadata | None,
    ) -> str:
        if turn_metadata and turn_metadata.explicit_phase_declared:
            return turn_metadata.explicit_phase_declared
        if plan is not None and plan.status == PlanStatus.RUNNING:
            return CognitivePhase.EXECUTION
        if task_memory.active_plan_id:
            return CognitivePhase.PLANNING
        return CognitivePhase.IDLE

    def _demo_jsonable(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if is_dataclass(value):
            return {k: self._demo_jsonable(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): self._demo_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._demo_jsonable(v) for v in value]
        if hasattr(value, "value") and hasattr(value, "name"):
            return getattr(value, "value")
        return value

    def _snapshot_task_memory(self, task_memory: TaskMemory) -> dict[str, Any]:
        return {
            "conversation_id": task_memory.conversation_id,
            "turn_count": task_memory.turn_count,
            "phase": task_memory.phase,
            "active_plan_id": task_memory.active_plan_id,
            "awaiting_user_input": task_memory.awaiting_user_input,
            "turn_in_progress": task_memory.turn_in_progress,
            "stall_count": task_memory.stall_count,
            "last_turn_at": task_memory.last_turn_at.isoformat() if task_memory.last_turn_at else None,
            "created_at": task_memory.created_at.isoformat() if task_memory.created_at else None,
        }

    def _snapshot_plan(self, plan: Plan | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "status": plan.status,
            "auto_continue": plan.auto_continue,
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "status": step.status,
                    "tool_name": step.tool_name,
                    "result_summary": step.result_summary,
                }
                for step in plan.steps
            ],
        }

    def _snapshot_task_state(self, task_state: TaskState | None) -> dict[str, Any] | None:
        if task_state is None:
            return None
        return {
            "objective": task_state.objective,
            "slot_contract_required": task_state.slot_contract_required,
            "output_unit": task_state.output_unit,
            "output_format": task_state.output_format,
            "ready_to_compute": task_state.ready_to_compute,
            "should_compute": task_state.should_compute,
            "slots": [
                {
                    "name": slot.name,
                    "description": slot.description,
                    "state": getattr(slot.state, "value", str(slot.state)),
                    "resolved": slot.resolved,
                    "value": slot.value,
                    "unit": slot.unit,
                    "locator_refs": list(slot.locator_refs),
                    "evidence_refs": list(slot.evidence_refs),
                }
                for slot in task_state.slots
            ],
        }

    def _snapshot_task_spec(self, task_spec: Any | None) -> dict[str, Any]:
        if task_spec is None:
            return {}
        return {
            "mode": getattr(getattr(task_spec, "mode", None), "value", getattr(task_spec, "mode", None)),
            "answer_schema": self._demo_jsonable(getattr(task_spec, "answer_schema", None)),
            "slot_names": list(getattr(task_spec, "slot_names", []) or []),
            "output_contract": self._demo_jsonable(getattr(task_spec, "output_contract", None)),
        }

    def _sync_demo_manifest(
        self,
        manifest: Any,
        *,
        task_memory: TaskMemory,
        rt: TurnRuntime,
        plan: Plan | None,
        status: str,
        assistant_reply: str | None = None,
        error_class: str | None = None,
    ) -> None:
        manifest.status = status
        manifest.cognitive_phase = str(rt.phase) if rt.phase else None
        manifest.execution_mode = (
            rt.task_spec.mode.value if rt.task_spec is not None else manifest.execution_mode
        )
        manifest.active_plan_id = plan.plan_id if plan is not None else task_memory.active_plan_id
        manifest.task_memory = self._snapshot_task_memory(task_memory)
        manifest.task_spec = self._snapshot_task_spec(rt.task_spec)
        manifest.task_state = self._snapshot_task_state(rt.task_state)
        manifest.output_contract = self._demo_jsonable(rt.output_contract)
        if assistant_reply is not None:
            manifest.assistant_reply = assistant_reply
        elif rt.assistant_reply_buffer:
            manifest.assistant_reply = rt.assistant_reply_buffer
        manifest.error_class = error_class
        manifest.updated_at = datetime.now(UTC)

    def _build_demo_checkpoint_state(
        self,
        *,
        stage: str,
        task_memory: TaskMemory,
        rt: TurnRuntime,
        plan: Plan | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "cima_demo.checkpoint_state.v1",
            "stage": stage,
            "run_id": rt.run_id,
            "conversation_id": rt.conversation_id,
            "turn_id": rt.turn_id,
            "cognitive_phase": str(rt.phase) if rt.phase else None,
            "execution_mode": rt.task_spec.mode.value if rt.task_spec is not None else None,
            "assistant_reply": rt.assistant_reply_buffer,
            "cited_markers": list(getattr(rt, "cited_markers", []) or []),
            "demo_need_proposal": self._demo_jsonable(getattr(rt, "demo_need_proposal", {})),
            "demo_memory_proposal": self._demo_jsonable(getattr(rt, "demo_memory_proposal", {})),
            "tool_calls_emitted": list(rt.tool_calls_emitted),
            "conclusions_types_seen": list(rt.conclusions_types_seen),
            "source_requirements": self._demo_jsonable(rt.source_requirements),
            "task_memory": self._snapshot_task_memory(task_memory),
            "task_spec": self._snapshot_task_spec(rt.task_spec),
            "task_state": self._snapshot_task_state(rt.task_state),
            "plan": self._snapshot_plan(plan),
            "progress": {
                "artifact_count": rt.artifact_count,
                "resolved_slot_count": rt.resolved_slot_count,
                "compute_done": rt.compute_done,
                "has_final_compute_result": rt.has_final_compute_result,
                "iteration_count": rt.iteration_count,
            },
        }

    async def load_turn_metadata(self, conversation_id: str) -> dict[str, Any] | None:
        """Public accessor for API layer health/status checks."""
        return await self._db.load_turn_metadata(conversation_id)

    # ── Cross-turn web cache ──────────────────────────────────────────────────


