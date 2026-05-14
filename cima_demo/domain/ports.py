"""Abstract port interfaces (KIMA_Domain_CIMA_v0.10 §10, KIMA_Application_Layer_v1.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from cima_demo.domain.entities import (
    CItem,
    ConflictLogEntry,
    FileRecord,
    KimaDelta,
    LLMEvent,
    LLMMessage,
    Plan,
    PlanStep,
    SummaryNode,
    TaskMemory,
)
from cima_demo.domain.value_objects import (
    ChunkResult,
    CItemFilter,
    RerankResult,
    ScoredCItem,
)

# ── LLMPort ───────────────────────────────────────────────────────────────────


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Best-effort JSON object extraction for structured LLM calls."""
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Structured completion did not return a JSON object")


class LLMPort(ABC):
    """Streaming LLM interface (KIMA_Domain_CIMA_v0.10 §10.1)."""

    @abstractmethod
    def stream_chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int | None = None,
        prefill_response: bool = False,
    ) -> AsyncGenerator[LLMEvent, None]:
        """Stream LLM response events.

        Yields LLMEvent(type=TOKEN|TOOL_CALL|CONCLUSIONS|PHASE_DECL|DONE).
        Raises LLMUnavailableError, LLMContextOverflowError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming single completion for scoring / NLI calls.

        response_format — OpenAI-compatible format directive forwarded to the
        backend.  Pass {"type": "json_object"} to request guaranteed-valid JSON
        output (H-16).  When the backend ignores this hint, callers remain
        responsible for robust parsing.
        """
        ...  # pragma: no cover

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from a structured completion."""
        raw = await self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format or {"type": "json_object"},
        )
        return _extract_json_object(raw)

    async def stream_text(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream user-visible text tokens only."""
        from cima_demo.domain.value_objects import LLMEventType

        async for event in self.stream_chat(
            messages=messages,
            tools=None,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            prefill_response=True,
        ):
            if event.type == LLMEventType.TOKEN and event.token:
                yield event.token

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for *text* using the model's tokeniser."""
        ...  # pragma: no cover

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if LLM service is reachable."""
        ...  # pragma: no cover

    def abort(self) -> None:
        """Abort the active streaming request immediately (fire-and-forget).

        Default is a no-op; adapters that hold a live HTTP stream should
        override to close it so the model stops generating immediately when
        the user presses stop.
        """

    def runtime_metadata(self) -> dict[str, Any]:
        """Return non-secret LLM runtime metadata for prompt trace artifacts.

        Adapters may override this to expose provider/model/config fields.  The
        default intentionally avoids transport headers, API keys, cookies, and
        other secrets.
        """
        return {"provider": self.__class__.__name__}


# ── EmbeddingPort ─────────────────────────────────────────────────────────────

class EmbeddingPort(ABC):
    """Dense embedding interface (TEI)."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return dense vector for *text*.

        Raises EmbeddingUnavailableError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return dense vectors for a batch of texts."""
        ...  # pragma: no cover

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if embedding service is reachable."""
        ...  # pragma: no cover


# ── SparseEmbeddingPort ───────────────────────────────────────────────────────

class SparseEmbeddingPort(ABC):
    """Sparse embedding interface (SPLADE — Phase 2, APP-D-09).

    Phase 1: fastembed BM25 is handled internally by QdrantCItemAdapter.
    This port is reserved for Phase 2 SPLADE swap.
    """

    @abstractmethod
    async def embed_sparse(self, text: str) -> dict[int, float]:
        """Return sparse {token_id: weight} vector for *text*."""
        ...  # pragma: no cover


# ── CItemStorePort ────────────────────────────────────────────────────────────

class CItemStorePort(ABC):
    """Unified C-Item storage port — Qdrant is the exclusive store (v0.10 D-05).

    All mutation goes through this port. No parallel PostgreSQL citems table.
    """

    @abstractmethod
    async def save(self, citem: CItem) -> None:
        """Upsert C-Item with dense+sparse vectors and full payload.

        Raises CItemStoreError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def fetch(self, citem_id: str) -> CItem:
        """Retrieve a single C-Item by ID.

        Raises CItemNotFoundError if not present.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def fetch_batch(self, citem_ids: list[str]) -> list[CItem]:
        """Retrieve multiple C-Items by ID list (preserves order, skips missing)."""
        ...  # pragma: no cover

    @abstractmethod
    async def search(
        self,
        query_text: str,
        filter: CItemFilter,
        top_k: int,
    ) -> list[ScoredCItem]:
        """Hybrid dense+sparse search with RRF fusion.

        Returns up to *top_k* scored C-Items matching *filter*.
        Raises CItemStoreError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def fetch_neighbors(
        self,
        seed_ids: list[str],
        conversation_id: str,
        exclude_ids: set[str] | None = None,
        backward_max: int = 500,
    ) -> list[CItem]:
        """Bidirectional 1-hop graph expansion via dependency_ids.

        Forward:  seed → dep_ids → fetch
        Backward: scroll conversation (newest-first, capped at *backward_max*)
                  + post-filter by dep_ids containing seed.
        *backward_max* prevents O(n) memory usage on long conversations.
        Raises GeometricExpansionError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def update_field(
        self,
        citem_id: str,
        field: str,
        value: Any,
    ) -> None:
        """Update a single payload field for a C-Item.

        When field='scope_status' and value='archived', also writes archived_at_unix
        (partial fix for APP-D-04).
        Raises CItemNotFoundError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def delete(self, citem_id: str) -> None:
        """Delete a single C-Item.

        Raises CItemNotFoundError.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all C-Items for a conversation. Returns count deleted."""
        ...  # pragma: no cover

    @abstractmethod
    async def fetch_by_conversation(
        self,
        conversation_id: str,
        scope_status: str | None = None,
    ) -> list[CItem]:
        """Scroll all C-Items for a conversation, optionally filtered by scope_status."""
        ...  # pragma: no cover

    @abstractmethod
    async def exists_by_hash(self, content_hash: str, conversation_id: str) -> bool:
        """Return True if a C-Item with this content_hash exists for the conversation."""
        ...  # pragma: no cover

    @abstractmethod
    async def fetch_dense_vectors(self, citem_ids: list[str]) -> dict[str, list[float]]:
        """Return {citem_id: dense_vector} for a batch of IDs.

        Used by annotate_bridge_scores to compute mean_sim_to_pool.
        Skips missing IDs silently. Raises CItemStoreError on transport failure.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if Qdrant is reachable and collection exists."""
        ...  # pragma: no cover


# ── RelDBPort ─────────────────────────────────────────────────────────────────

class RelDBPort(ABC):
    """PostgreSQL relational data port (KIMA_Application_Layer_v1.1 APP-D-03).

    Handles: conversations, task_memory, summary_pyramid, plans/steps,
             conflict_log, retrieval_telemetry.
    Does NOT handle: citems (→ CItemStorePort).
    """

    # ── Conversation ──────────────────────────────────────────────────────────

    @abstractmethod
    async def create_conversation(self, conversation_id: str) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        ...  # pragma: no cover

    @abstractmethod
    async def list_conversations(self) -> list[dict[str, Any]]:
        ...  # pragma: no cover

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> None:
        """Cascade-deletes task_memory; C-Items deleted via CItemStorePort."""
        ...  # pragma: no cover

    @abstractmethod
    async def begin_hard_delete(self, conversation_id: str, *, delete_run_id: str) -> bool:
        """Mark a conversation as DELETING and create a durable delete run."""
        ...  # pragma: no cover

    @abstractmethod
    async def mark_hard_delete_completed(self, *, delete_run_id: str, stats_json: dict[str, Any] | None = None) -> None:
        """Mark a durable delete run as completed."""
        ...  # pragma: no cover

    @abstractmethod
    async def mark_hard_delete_failed(self, *, delete_run_id: str, stats_json: dict[str, Any] | None = None) -> None:
        """Mark a durable delete run as failed."""
        ...  # pragma: no cover

    # ── Turn mutex ────────────────────────────────────────────────────────────

    @abstractmethod
    async def try_set_turn_in_progress(self, conversation_id: str) -> bool:
        """Atomic CAS: set turn_in_progress=True if currently False.

        Returns True on success, False if already held.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def set_turn_finished(self, conversation_id: str) -> None:
        """Release turn mutex; increment turn_count; update last_turn_at."""
        ...  # pragma: no cover

    @abstractmethod
    async def release_turn_in_progress(self, conversation_id: str) -> None:
        """Force-release turn mutex (error path)."""
        ...  # pragma: no cover

    # ── TaskMemory ────────────────────────────────────────────────────────────

    @abstractmethod
    async def load_task_memory(self, conversation_id: str) -> TaskMemory | None:
        ...  # pragma: no cover

    @abstractmethod
    async def save_task_memory(self, task_memory: TaskMemory) -> None:
        ...  # pragma: no cover

    # ── History ───────────────────────────────────────────────────────────────

    @abstractmethod
    async def append_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        created_at: str | None = None,
    ) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def load_recent_history(
        self,
        conversation_id: str,
        max_turns: int = 10,
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return last *max_turns* turns as [{role, content, timestamp}].

        When *token_budget* is set, trims oldest messages until the total
        estimated token count fits within budget (rough estimate: len//4).
        """
        ...  # pragma: no cover

    # ── Summary pyramid ───────────────────────────────────────────────────────

    @abstractmethod
    async def save_summary(self, node: SummaryNode) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def load_summaries(
        self,
        conversation_id: str,
        level: int | None = None,
    ) -> list[SummaryNode]:
        ...  # pragma: no cover

    @abstractmethod
    async def set_summary_parent(self, node_id: str, parent_id: str) -> None:
        """Link a summary node to its parent (A-10 L2 AutoPromote).

        Updates parent_ids JSONB to [parent_id] and sets updated_at.
        """
        ...  # pragma: no cover

    # ── Plans ─────────────────────────────────────────────────────────────────

    @abstractmethod
    async def save_plan(self, plan: Plan) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def save_plan_with_task_memory(
        self,
        plan: Plan,
        task_memory: TaskMemory,
    ) -> None:
        """Atomically persist Plan + TaskMemory in a single transaction (INFRA-D-01).

        Use instead of separate save_plan + save_task_memory calls whenever the
        two must stay consistent (plan.start, plan.advance_step, plan completion).
        """
        ...  # pragma: no cover

    @abstractmethod
    async def load_plan(self, plan_id: str) -> Plan | None:
        ...  # pragma: no cover

    @abstractmethod
    async def save_plan_step(self, step: PlanStep) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def update_plan_step_attempts(self, step_id: str, attempts: int) -> None:
        """Increment attempts counter for a plan step (called after each retry)."""
        ...  # pragma: no cover

    # ── Conflict log ──────────────────────────────────────────────────────────

    @abstractmethod
    async def save_conflict(self, entry: ConflictLogEntry) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def load_conflicts(
        self,
        conversation_id: str,
        resolved: bool | None = None,
    ) -> list[ConflictLogEntry]:
        ...  # pragma: no cover

    # ── Telemetry ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def save_retrieval_telemetry(
        self,
        conversation_id: str,
        query_type: str,
        recall_top_k: int,
        rerank_top_n: int,
        items_selected: int,
        coverage_score: float,
        retry_count: int,
        latency_ms: int,
        candidates_before_rerank: int = 0,
        candidates_after_rerank: int = 0,
        candidates_after_expand: int = 0,
        pack_total_tokens: int = 0,
        geometric_expand: bool = False,
        reranker_available: bool = True,
        traceability_density: float = 1.0,
        # Bridge / strategy fields (Retrieval Instrumentation D)
        q3_relevant_count: int = 0,
        bridge_enabled: bool = False,
        bridge_alpha: float = 0.5,
        bridge_floor: float = 0.0,
        bridge_candidates_eligible: int = 0,
        direct_strategy: str | None = None,
    ) -> None:
        """Non-blocking telemetry insert (called via asyncio.create_task)."""
        ...  # pragma: no cover

    # ── Turn metadata ─────────────────────────────────────────────────────────

    @abstractmethod
    async def load_turn_metadata(self, conversation_id: str) -> dict[str, Any] | None:
        """Load previous turn's TurnMetadata as raw dict (from task_metadata JSONB)."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_turn_metadata(self, conversation_id: str, json_data: str) -> None:
        """Persist TurnMetadata as JSONB (upsert by conversation_id)."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_chm_refs(self, conversation_id: str) -> dict[str, int]:
        """Return {citem_id: reference_count} for all CHMs in this conversation."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_chm_refs(self, conversation_id: str, citem_ids: list[str]) -> None:
        """Upsert CHM reference rows, incrementing reference_count on conflict."""
        ...  # pragma: no cover

    # ── Demonstrator run journal ─────────────────────────────────────────────

    @abstractmethod
    async def create_demo_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        turn_id: str,
        status: str,
        user_message: str,
        manifest_json: dict[str, Any],
    ) -> None:
        """Create a durable demonstrator run shell."""
        ...  # pragma: no cover

    @abstractmethod
    async def append_demo_run_phase(
        self,
        *,
        run_id: str,
        phase_name: str,
        payload_json: dict[str, Any],
    ) -> int:
        """Append a phase record and return its monotonically increasing sequence."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_checkpoint(
        self,
        *,
        run_id: str,
        checkpoint_id: str,
        checkpoint_kind: str,
        state_json: dict[str, Any],
    ) -> int:
        """Persist a run checkpoint and return its monotonically increasing sequence."""
        ...  # pragma: no cover

    @abstractmethod
    async def touch_demo_run_counters(
        self,
        *,
        run_id: str,
        checkpoint_count: int | None = None,
        phase_count: int | None = None,
    ) -> None:
        """Update cached phase/checkpoint counters on the run manifest row."""
        ...  # pragma: no cover

    @abstractmethod
    async def update_demo_run_manifest(
        self,
        *,
        run_id: str,
        status: str,
        cognitive_phase: str | None,
        execution_mode: str | None,
        active_plan_id: str | None,
        assistant_reply: str,
        error_class: str | None,
        manifest_json: dict[str, Any],
        finished_at: str | None = None,
    ) -> None:
        """Persist the latest manifest view for a demonstrator run."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the persisted run manifest row as a JSON-like dict."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_run_phases(self, run_id: str) -> list[dict[str, Any]]:
        """Return phase records for a demonstrator run, ordered by sequence."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """Return checkpoints for a demonstrator run, ordered by sequence."""
        ...  # pragma: no cover

    # ── Extended pyramid queries ──────────────────────────────────────────────

    @abstractmethod
    async def fetch_nodes_at_level(
        self,
        level: int,
        conversation_id: str,
        parentless_only: bool = False,
        limit: int | None = None,
    ) -> list[SummaryNode]:
        """Return SummaryNodes at *level*, optionally filtered to parentless ones (N-01)."""
        ...  # pragma: no cover

    @abstractmethod
    async def fetch_pyramid_tops(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[SummaryNode]:
        """Return SummaryNodes without parents (top of pyramid), highest level first."""
        ...  # pragma: no cover

    # ── File registry ─────────────────────────────────────────────────────────

    @abstractmethod
    async def save_file_record(self, record: FileRecord) -> None:
        """Insert a new file registry row (status=QUEUED)."""
        ...  # pragma: no cover

    @abstractmethod
    async def update_file_record(
        self,
        file_id: str,
        *,
        status: str,
        chunk_count: int = 0,
        citem_ids: list[str] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update status, chunk_count, citem_ids and error_message for an existing record."""
        ...  # pragma: no cover

    @abstractmethod
    async def list_file_records(self, conversation_id: str) -> list[FileRecord]:
        """Return all file records for a conversation, newest first."""
        ...  # pragma: no cover

    @abstractmethod
    async def get_file_record(self, file_id: str) -> FileRecord | None:
        """Return one file record by id, or None if absent."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_chunk_record(self, chunk_json: dict[str, Any]) -> None:
        """Persist one witness chunk manifest row."""
        ...  # pragma: no cover

    @abstractmethod
    async def list_chunk_records(
        self,
        conversation_id: str,
        *,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return persisted chunk manifest rows, optionally filtered by source."""
        ...  # pragma: no cover

    # ── DEBT-01: Autonomous Plan Executor ────────────────────────────────────

    @abstractmethod
    async def list_auto_plans(self) -> list[tuple[str, str, str, int, int, str | None]]:
        """Return auto-continue candidates with context for the worker.

        Each tuple: (conversation_id, plan_id, active_step_description,
                      active_step_index, total_steps, prev_step_result).
        Only returns plans where auto_continue=True, status=RUNNING,
        turn_in_progress=FALSE, and awaiting_user_input=FALSE.
        """
        ...  # pragma: no cover


    # ── Demonstrator lineage / context artifacts ────────────────────────────

    @abstractmethod
    async def save_demo_source(self, source_json: dict[str, Any]) -> None:
        """Persist a demonstrator source record."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_source_span(self, span_json: dict[str, Any]) -> None:
        """Persist a rehydratable demonstrator source span."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_lineage_edge(self, edge_json: dict[str, Any]) -> None:
        """Persist a directed lineage edge between demonstrator artifacts."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_summary_resolution(self, resolution_json: dict[str, Any]) -> None:
        """Persist a summary->origin mapping for the demonstrator."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_context_snapshot(self, snapshot_json: dict[str, Any]) -> None:
        """Persist a durable context snapshot."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_context_snapshot(self, context_id: str) -> dict[str, Any] | None:
        """Return a persisted context snapshot."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_answer_lineage(self, answer_json: dict[str, Any]) -> None:
        """Persist answer lineage rooted at a context snapshot."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_latest_demo_context_snapshot_for_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the latest context snapshot for a run."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_sources(self, conversation_id: str, source_ids: list[str]) -> list[dict[str, Any]]:
        """Load demonstrator sources by ID."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_source_spans(self, conversation_id: str, span_ids: list[str]) -> list[dict[str, Any]]:
        """Load demonstrator source spans by ID."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_lineage_edges(
        self,
        conversation_id: str,
        *,
        src_kind: str | None = None,
        src_ids: list[str] | None = None,
        dst_kind: str | None = None,
        dst_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load demonstrator lineage edges with optional endpoint filtering."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_summary_resolutions(
        self,
        conversation_id: str,
        summary_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load demonstrator summary resolution rows."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_geometry_run(self, run_json: dict[str, Any]) -> None:
        """Persist a detached geometry run."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_geometry_item_state(self, item_state_json: dict[str, Any]) -> None:
        """Persist geometry item state for one ref."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_geometry_cluster_state(self, cluster_state_json: dict[str, Any]) -> None:
        """Persist geometry cluster state for one cluster."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_geometry_item_states(self, conversation_id: str, ref_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Load geometry item states for a conversation."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_geometry_cluster_states(self, conversation_id: str) -> list[dict[str, Any]]:
        """Load geometry cluster states for a conversation."""
        ...  # pragma: no cover

    @abstractmethod
    async def delete_geometry_conversation(self, conversation_id: str) -> None:
        """Delete all geometry rows for a conversation."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_handoff_manifest(self, manifest_json: dict[str, Any]) -> None:
        """Persist a portable handoff manifest."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_handoff_manifest(self, handoff_id: str) -> dict[str, Any] | None:
        """Load a portable handoff manifest by ID."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_handoff_validation(self, validation_json: dict[str, Any]) -> None:
        """Persist handoff validation results."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_handoff_validation(self, handoff_id: str) -> dict[str, Any] | None:
        """Load handoff validation results."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_handoff_restore(self, restore_json: dict[str, Any]) -> None:
        """Persist a handoff restore attempt and reconstruction diff."""
        ...  # pragma: no cover

    @abstractmethod
    async def save_demo_gc_audit(self, audit_json: dict[str, Any]) -> None:
        """Persist a durable GC/lifecycle audit event."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_gc_audits(self, conversation_id: str) -> list[dict[str, Any]]:
        """Load GC/lifecycle audit events for a conversation."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_demo_conversation_counts(self, conversation_id: str) -> dict[str, Any]:
        """Return relational artifact counts for a conversation, even if the conversation row is gone."""
        ...  # pragma: no cover

    @abstractmethod
    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any],
        headers_json: dict[str, Any] | None = None,
    ) -> int:
        """Persist one outbox row and return its numeric identifier."""
        ...  # pragma: no cover

    @abstractmethod
    async def claim_outbox_batch(self, limit: int = 100) -> list[dict[str, Any]]:
        """Claim a batch of NEW outbox rows for publication."""
        ...  # pragma: no cover

    @abstractmethod
    async def mark_outbox_sent(self, outbox_ids: list[int]) -> None:
        """Mark a batch of outbox rows as sent."""
        ...  # pragma: no cover

    @abstractmethod
    async def mark_outbox_error(self, outbox_id: int, error: str) -> None:
        """Record a publication error for one outbox row."""
        ...  # pragma: no cover

    @abstractmethod
    async def begin_consumer_effect(
        self,
        *,
        consumer_name: str,
        event_id: str,
        effect_key: str,
    ) -> bool:
        """Try to register a consumer effect execution. Returns False on duplicate."""
        ...  # pragma: no cover

    @abstractmethod
    async def complete_consumer_effect(
        self,
        *,
        consumer_name: str,
        event_id: str,
        effect_key: str,
        details_json: dict[str, Any] | None = None,
    ) -> None:
        """Mark a consumer effect as completed."""
        ...  # pragma: no cover

    @abstractmethod
    async def append_citem_audit_event(
        self,
        *,
        conversation_id: str,
        citem_id: str,
        event_type: str,
        old_value: str | None = None,
        new_value: str | None = None,
    ) -> None:
        """Persist one C-Item lifecycle audit event."""
        ...  # pragma: no cover

    @abstractmethod
    async def load_citem_audit_events(self, conversation_id: str) -> list[dict[str, Any]]:
        """Load lifecycle audit events for a conversation."""
        ...  # pragma: no cover

    # ── Health ────────────────────────────────────────────────────────────────

    @abstractmethod
    async def ping(self) -> bool:
        ...  # pragma: no cover


# ── RerankerPort ──────────────────────────────────────────────────────────────

class RerankerPort(ABC):
    """Cross-encoder reranker interface (TEI reranker)."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        texts: list[str],
        top_n: int,
        truncate: bool = True,
    ) -> list[RerankResult]:
        """Return top_n results sorted by score descending.

        Raises RerankerUnavailableError (triggers graceful degradation to RRF order).
        """
        ...  # pragma: no cover

    @abstractmethod
    async def ping(self) -> bool:
        ...  # pragma: no cover


# ── NLIPort ───────────────────────────────────────────────────────────────────

class NLIPort(ABC):
    """Natural Language Inference classification interface."""

    @abstractmethod
    async def classify(self, text_a: str, text_b: str) -> str:
        """Classify the relationship between two texts.

        Returns one of: 'ENTAILMENT' | 'NEUTRAL' | 'CONTRADICTION'.
        Raises NLIUnavailableError.
        """
        ...  # pragma: no cover


# ── GeometricExpansionPort ────────────────────────────────────────────────────

class GeometricExpansionPort(ABC):
    """Graph expansion strategy port (APP-INV-27).

    Phase 1: DependencyIdsGeometricExpander (structural, 2 Qdrant ops).
    Phase 2: semantic expansion via embeddings.
    """

    @abstractmethod
    async def expand(
        self,
        seeds: list[CItem],
        conversation_id: str,
        exclude_ids: set[str],
    ) -> list[CItem]:
        """Return neighbor C-Items not in *exclude_ids*."""
        ...  # pragma: no cover


# ── ChunkingPort ──────────────────────────────────────────────────────────────

class ChunkingPort(ABC):
    """Document chunking interface (KIMA_Infrastructure_Layer_v0.6 §3.11)."""

    @abstractmethod
    async def chunk(
        self,
        text: str,
        filename: str,
        doc_type: str,
    ) -> list[ChunkResult]:
        """Split *text* into chunks; returns ChunkResult list with dependency_ids chain."""
        ...  # pragma: no cover


# ── EventBusPort ──────────────────────────────────────────────────────────────

class EventBusPort(ABC):
    """Async event bus for SSE deltas (Phase 1: DirectSSEEventBus; Phase 2: Kafka)."""

    @abstractmethod
    async def publish(self, delta: KimaDelta) -> None:
        """Publish a KimaDelta to all subscribers of conversation_id."""
        ...  # pragma: no cover

    @abstractmethod
    def subscribe(self, conversation_id: str) -> AsyncGenerator[KimaDelta, None]:
        """Yield KimaDeltas for conversation_id until DONE sentinel."""
        ...  # pragma: no cover


# ── WebSearchPort ─────────────────────────────────────────────────────────────

class WebSearchPort(ABC):
    """Web search interface (SearXNG)."""

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Return list of {title, url, snippet} dicts.

        Raises WebSearchError.
        """
        ...  # pragma: no cover


# ── FileProcessingPort ────────────────────────────────────────────────────────

class FileProcessingPort(ABC):
    """File text extraction interface (PDF, DOCX, plain text)."""

    @abstractmethod
    def extract_text(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
    ) -> str:
        """Return extracted text from file bytes.

        Raises FileProcessingError.
        """
        ...  # pragma: no cover

    @abstractmethod
    def supported_mime_types(self) -> list[str]:
        """List of supported MIME types."""
        ...  # pragma: no cover


# ── DomainAliasPort ───────────────────────────────────────────────────────────

class DomainAliasPort(ABC):
    """Resolves named-source keywords to canonical domain strings (DEBT-02).

    Replaces the hardcoded _DOMAIN_ALIASES dict in source_lock.py.
    Implementations may load entries from environment variables, DB tables,
    or ConfigMaps — all without touching source-lock detection logic.
    """

    @abstractmethod
    def get_aliases(self) -> dict[str, str]:
        """Return {keyword: domain} mapping used by source-lock detection.

        Example: {"wikipedia": "wikipedia.org", "arxiv": "arxiv.org"}
        """
        ...  # pragma: no cover

