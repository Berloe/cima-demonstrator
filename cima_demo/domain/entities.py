"""Domain entities (KIMA_Domain_CIMA_v0.10 §3, §4, §7, §8)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cima_demo.domain.value_objects import (
    CognitivePhase,
    ItemType,
    PlanStatus,
    StepStatus,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── CItem ─────────────────────────────────────────────────────────────────────

@dataclass
class CItem:
    """Contextual Item — atomic unit of memory (CIMA §3.1, CIMA A-3).

    scope:        "episodic" | "global"
    scope_status: "active" | "archived"
    conflict_status: "none" | "flagged" | "resolved"  (default "none" per INV-14)
    """
    citem_id:         str          = field(default_factory=_uuid)
    conversation_id:  str          = ""
    content:          str          = ""
    item_type:        str          = ItemType.FACT
    scope:            str          = "episodic"
    scope_status:     str          = "active"
    importance:       float        = 0.5
    confidence:       float        = 1.0
    validation_label: str | None   = None    # "verified" | "unverified" | "refuted"
    conflict_status:  str          = "none"  # INV-14: default "none", NOT "clean"
    phase_ingested:   str          = CognitivePhase.IDLE
    actor:            str          = "agent"  # "agent" | "user"
    motivation:       str | None   = None
    created_at:       datetime     = field(default_factory=_now)
    dependency_ids:   list[str]    = field(default_factory=list)
    token_count:      int          = 0
    archived_at_unix:        float | None = None   # APP-D-04: set by update_field on archive
    summarized_by_node_id:   str | None   = None   # node_id of SummaryNode that archived this item
    content_hash:            str | None   = None   # SHA-256 of content for deduplication
    # Epistemic kind — set for web-content chunks; None for agent-authored C-Items.
    # Controls prompt_eligible and evidence_eligible gating (ChunkKind).
    chunk_kind:              str | None   = None

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def attenuate(self, threshold: float = 0.2) -> None:
        """Lower importance by 20 %; archive if below threshold (CIMA A-7)."""
        self.importance = round(self.importance * 0.8, 6)
        if self.importance < threshold:
            self.scope_status = "archived"

    def promote_to_global(self) -> None:
        """Elevate to global scope (CIMA §3.3, PromotionPolicy)."""
        self.scope = "global"


# ── SummaryNode ───────────────────────────────────────────────────────────────

@dataclass
class SummaryNode:
    """Pyramid DAG node (CIMA §4.2).

    level: 1=turn, 2=session, 3=global
    origin_citem_ids: C-Item IDs that were archived into this summary (APP-D-04).
    """
    node_id:            str        = field(default_factory=_uuid)
    conversation_id:    str        = ""
    level:              int        = 1
    content:            str        = ""
    token_count:        int        = 0
    created_at:         datetime   = field(default_factory=_now)
    parent_id:          str | None = None
    origin_citem_ids:   list[str]  = field(default_factory=list)


# ── Plan / PlanStep ───────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """Single step within a Plan (CIMA §5.2).

    Extended fields (backward-compat: all default None):
    - acceptance_criterion: observable condition that marks the step done.
      Shown in context so the model knows when to move on.
    - context_focus: optional retrieval-query override narrower than description.
      When set, replaces description in derive_query / refresh_query so the
      embedding search targets a tighter semantic cluster.
    """
    step_id:              str        = field(default_factory=_uuid)
    plan_id:              str        = ""
    description:          str        = ""
    status:               str        = StepStatus.PENDING
    tool_name:            str | None = None
    tool_args:            dict[str, Any] = field(default_factory=dict)
    result_summary:       str | None = None
    procedure_citem_id:   str | None = None   # D-02: links to PROCEDURE C-Item
    created_at:           datetime   = field(default_factory=_now)
    completed_at:         datetime | None = None
    # ── Semantic extensions (proactive decomposition — backward-compat) ─────────
    acceptance_criterion: str | None = None   # testable completion condition
    context_focus:        str | None = None   # retrieval query override

    def mark_active(self) -> None:
        self.status = StepStatus.ACTIVE

    def mark_completed(self, result_summary: str | None = None) -> None:
        self.status = StepStatus.COMPLETED
        self.completed_at = _now()
        if result_summary is not None:
            self.result_summary = result_summary

    def mark_failed(self, reason: str | None = None) -> None:
        self.status = StepStatus.FAILED
        self.completed_at = _now()
        if reason is not None:
            self.result_summary = reason


@dataclass
class Plan:
    """Execution plan for a multi-step task (CIMA §5.1)."""
    plan_id:         str            = field(default_factory=_uuid)
    conversation_id: str            = ""
    goal:            str            = ""
    status:          str            = PlanStatus.PENDING
    steps:           list[PlanStep] = field(default_factory=list)
    created_at:      datetime       = field(default_factory=_now)
    updated_at:      datetime       = field(default_factory=_now)
    auto_continue:   bool           = False  # DEBT-01: autonomous execution flag

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def active_step(self) -> PlanStep | None:
        for s in self.steps:
            if s.status == StepStatus.ACTIVE:
                return s
        return None

    @property
    def next_pending_step(self) -> PlanStep | None:
        for s in self.steps:
            if s.status == StepStatus.PENDING:
                return s
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    @property
    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def start(self) -> None:
        self.status = PlanStatus.RUNNING
        self.updated_at = _now()

    def pause(self) -> None:
        self.status = PlanStatus.PAUSED
        self.updated_at = _now()

    def complete(self) -> None:
        self.status = PlanStatus.COMPLETED
        self.updated_at = _now()

    def fail(self) -> None:
        self.status = PlanStatus.FAILED
        self.updated_at = _now()

    def replan(self, new_steps: list[PlanStep]) -> None:
        self.steps = new_steps
        self.status = PlanStatus.REPLANNED
        self.updated_at = _now()


# ── TaskMemory ────────────────────────────────────────────────────────────────

@dataclass
class TaskMemory:
    """Per-conversation persistent state (CIMA §6.1).

    Persisted in PostgreSQL (source of truth for session state).
    """
    conversation_id:      str       = ""
    turn_count:           int       = 0
    phase:                str       = CognitivePhase.IDLE
    active_plan_id:       str | None = None
    awaiting_user_input:  bool      = False
    turn_in_progress:     bool      = False
    stall_count:          int       = 0
    last_turn_at:         datetime | None = None
    created_at:           datetime  = field(default_factory=_now)

    # ── Turn management ───────────────────────────────────────────────────────

    def begin_turn(self, phase: str = CognitivePhase.IDLE) -> None:
        self.turn_in_progress = True
        self.phase = phase
        self.awaiting_user_input = False

    def finish_turn(self) -> None:
        self.turn_count += 1
        self.turn_in_progress = False
        self.last_turn_at = _now()

    def set_awaiting_user(self) -> None:
        self.awaiting_user_input = True
        self.turn_in_progress = False

    def record_stall(self) -> None:
        self.stall_count += 1


# ── ContextPack ───────────────────────────────────────────────────────────────

@dataclass
class ContextPack:
    """5-slot context assembly output (CIMA A-5, APP-INV-22).

    Slots and budget caps:
      protected_items  — no cap (PROCEDURE + DECISION + CONSTRAINT)
      direct_evidence  — ≤50 % of available tokens
      bridge_evidence  — ≤15 %
      global_summaries — ≤25 %
      conflicts        — ≤10 %
    """
    protected_items:  list[CItem] = field(default_factory=list)
    direct_evidence:  list[CItem] = field(default_factory=list)
    bridge_evidence:  list[CItem] = field(default_factory=list)
    global_summaries: list[CItem] = field(default_factory=list)
    conflicts:        list[CItem] = field(default_factory=list)
    tokens_used:      int         = 0
    coverage_score:   float       = 0.0

    def all_items(self) -> list[CItem]:
        return (
            self.protected_items
            + self.direct_evidence
            + self.bridge_evidence
            + self.global_summaries
            + self.conflicts
        )


# ── ContextView ───────────────────────────────────────────────────────────────

@dataclass
class ContextView:
    """Serialised context injected into the LLM prompt.

    items preserves a structured, marker-based view of what was selected so the
    demonstrator can persist snapshots and build traceable answer lineage.
    """
    text:          str        = ""
    tokens_used:   int        = 0
    coverage_score: float     = 0.0
    citem_ids:     list[str]  = field(default_factory=list)
    items:         list[dict[str, Any]] = field(default_factory=list)


# ── CHM ───────────────────────────────────────────────────────────────────────

@dataclass
class CHM:
    """Contextual History Metadata for the current turn (CIMA §6.2).

    Tracks conversation-level reference counts for promotion eligibility.
    """
    conversation_id:   str             = ""
    reference_counts:  dict[str, int]  = field(default_factory=dict)   # citem_id → count
    created_at:        datetime        = field(default_factory=_now)
    updated_at:        datetime        = field(default_factory=_now)

    def increment(self, citem_id: str) -> None:
        self.reference_counts[citem_id] = self.reference_counts.get(citem_id, 0) + 1
        self.updated_at = _now()

    def get_count(self, citem_id: str) -> int:
        return self.reference_counts.get(citem_id, 0)


# ── ConflictLogEntry ──────────────────────────────────────────────────────────

@dataclass
class ConflictLogEntry:
    """Record of a detected conflict between two C-Items (CIMA §7.1)."""
    entry_id:        str      = field(default_factory=_uuid)
    conversation_id: str      = ""
    item_a_id:       str      = ""
    item_b_id:       str      = ""
    conflict_type:   str      = ""    # e.g. "CONTRADICTION", "INCONSISTENCY"
    resolution:      str | None = None
    resolved:        bool     = False
    created_at:      datetime = field(default_factory=_now)
    resolved_at:     datetime | None = None

    def resolve(self, resolution: str) -> None:
        self.resolved = True
        self.resolution = resolution
        self.resolved_at = _now()


# ── StallTracker ──────────────────────────────────────────────────────────────

@dataclass
class StallTracker:
    """Detects agent stall conditions (repeated tool calls without progress)."""
    conversation_id:      str            = ""
    tool_call_hashes:     list[str]      = field(default_factory=list)  # last N hashes
    consecutive_repeats:  int            = 0
    stall_threshold:      int            = 2

    def record(self, call_hash: str) -> bool:
        """Record a tool call hash; return True if stall detected."""
        if self.tool_call_hashes and self.tool_call_hashes[-1] == call_hash:
            self.consecutive_repeats += 1
        else:
            self.consecutive_repeats = 1
        self.tool_call_hashes.append(call_hash)
        if len(self.tool_call_hashes) > 20:
            self.tool_call_hashes = self.tool_call_hashes[-20:]
        return self.consecutive_repeats >= self.stall_threshold

    def reset(self) -> None:
        self.consecutive_repeats = 0


# ── LLM Message / Event ───────────────────────────────────────────────────────

@dataclass
class LLMMessage:
    """Single message in an LLM conversation (role + content)."""
    role:         str                        # "system" | "user" | "assistant" | "tool"
    content:      str
    name:         str | None       = None   # tool name when role="tool"
    tool_call_id: str | None       = None   # when role="tool": id of the call being answered
    tool_calls:   list[Any] | None = None   # when role="assistant": list of tool call dicts
    # Multimodal content (vision): when set, the adapter uses this list instead of `content`.
    # Format: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
    # `content` must still hold the plain-text portion for token estimation and memory persistence.
    content_parts: list[Any] | None = None


@dataclass
class LLMEvent:
    """Streaming event from LLMPort.stream_chat() (KIMA_SystemPrompt_v0.1 §3).

    strategy_fail_type/reason added in domain v0.10.3 (appendix A1, A3).
    Both are None for all existing event types — fully backwards compatible.
    """
    type:                 str              # LLMEventType
    token:                str | None       = None
    tool_name:            str | None       = None
    tool_args:            str | None       = None   # JSON string (partial during streaming)
    tool_call_id:         str | None       = None
    conclusions:          list[dict[str, Any]] = field(default_factory=list)
    phase_decl:           str | None       = None
    strategy_fail_type:   str | None       = None   # "convergence" | "misclassification"
    strategy_fail_reason: str | None       = None   # text from <strategy_fail> tag
    truncated:            bool             = False   # True when finish_reason == "length" (T5)


# ── KimaDelta / KimaEvent ─────────────────────────────────────────────────────

@dataclass
class KimaDelta:
    """SSE delta emitted by AgentOrchestrator → SSEBroker (KIMA API §3.2)."""
    type:            str                       # KimaDeltaType
    conversation_id: str          = ""
    token:           str | None   = None
    thought:         str | None   = None
    tool_name:       str | None   = None
    tool_result:     str | None   = None
    plan_id:         str | None   = None
    step_index:      int | None   = None
    step_description: str | None  = None
    step_status:     str | None   = None
    total_steps:     int | None   = None
    conflict_summary: str | None  = None
    stall_message:   str | None   = None
    context_summary: str | None   = None
    success:         bool | None  = None
    error_code:      str | None   = None
    error_message:   str | None   = None


@dataclass
class KimaEvent:
    """Wrapper for KimaDelta published on EventBusPort."""
    delta:      KimaDelta
    created_at: datetime = field(default_factory=_now)


# ── IngestRequest ─────────────────────────────────────────────────────────────

@dataclass
class IngestRequest:
    """Request to ingest one or more C-Items (used by MemoryService)."""
    conversation_id: str
    content:         str
    item_type:       str  = ItemType.OBSERVATION
    scope:           str  = "episodic"
    importance:      float = 0.5
    confidence:      float = 1.0
    actor:           str   = "agent"
    motivation:      str | None = None
    phase_ingested:  str   = CognitivePhase.IDLE
    dependency_ids:  list[str] = field(default_factory=list)
    validation_label: str | None = None
    # CCP: when set, overrides compute_static_importance(). Use for tool results
    # whose importance is determined by caller heuristic, not by type+confidence alone.
    importance_override: float | None = None
    # ChunkKind value for web-content chunks (see domain/value_objects.py).
    # None for agent-authored items.
    chunk_kind: str | None = None
    # Demonstrator provenance hooks.
    source_id: str | None = None
    source_span_ids: list[str] = field(default_factory=list)
    lineage_meta: dict[str, Any] = field(default_factory=dict)


# ── TurnContext ───────────────────────────────────────────────────────────────

# ── FileRecord ────────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    """Registry entry for a file ingested into conversation memory.

    Persisted in PostgreSQL file_registry table.
    Lifecycle: QUEUED → PROCESSING → READY | FAILED
    """
    file_id:         str        = field(default_factory=_uuid)
    conversation_id: str        = ""
    filename:        str        = ""
    mime_type:       str        = "application/octet-stream"
    size_bytes:      int        = 0
    content_hash:    str        = ""
    status:          str        = "QUEUED"   # QUEUED | PROCESSING | READY | FAILED
    chunk_count:     int        = 0
    citem_ids:       list[str]  = field(default_factory=list)
    blob_path:       str | None = None
    ingested_at:     datetime   = field(default_factory=_now)
    error_message:   str | None = None


@dataclass
class TurnContext:
    """Ephemeral per-turn state (not persisted; rebuilt each turn).

    A-8.3: phase is constant within a turn.
    tool_calls_previous_round comes from prior turn's TaskMemory.
    """
    conversation_id:          str            = ""
    phase:                    str            = CognitivePhase.IDLE
    user_message:             str            = ""
    file_contents:            list[str]      = field(default_factory=list)
    tool_calls_previous_round: list[str]     = field(default_factory=list)
    context_view:             ContextView    = field(default_factory=ContextView)
    messages:                 list[LLMMessage] = field(default_factory=list)
    turn_number:              int            = 0


# ── Turn Metadata ─────────────────────────────────────────────────────────────

@dataclass
class TurnMetadata:
    """Inter-turn metadata persisted in task_metadata JSONB (APP-D-02 permanent)."""
    conversation_id: str
    tool_calls_emitted: list[str] = field(default_factory=list)
    conclusions_item_types: list[str] = field(default_factory=list)
    explicit_phase_declared: str | None = None

    def to_json(self) -> str:
        import json
        return json.dumps({
            "conversation_id":         self.conversation_id,
            "tool_calls_emitted":      self.tool_calls_emitted,
            "conclusions_item_types":  self.conclusions_item_types,
            "explicit_phase_declared": self.explicit_phase_declared,
        }, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnMetadata:
        return cls(
            conversation_id=data["conversation_id"],
            tool_calls_emitted=data.get("tool_calls_emitted", []),
            conclusions_item_types=data.get("conclusions_item_types", []),
            explicit_phase_declared=data.get("explicit_phase_declared"),
        )
