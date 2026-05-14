"""Domain value objects (KIMA_Domain_CIMA_v0.10 §5, §6, SystemPrompt v0.1 §4)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cima_demo.domain.entities import CItem


# ── Enumerations ──────────────────────────────────────────────────────────────

class ItemType(str, Enum):
    FACT        = "FACT"        # observed directly in cited evidence
    DERIVED     = "DERIVED"     # computed/inferred from cited evidence (traceable)
    HYPOTHESIS  = "HYPOTHESIS"  # plausible but not yet verified
    ASSUMPTION  = "ASSUMPTION"  # temporary working assumption, explicitly marked
    DECISION    = "DECISION"
    CONSTRAINT  = "CONSTRAINT"
    OBSERVATION = "OBSERVATION"
    PROCEDURE   = "PROCEDURE"


class PlanStatus(str, Enum):
    PENDING    = "PENDING"
    RUNNING    = "RUNNING"
    PAUSED     = "PAUSED"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"
    REPLANNED  = "REPLANNED"


class StepStatus(str, Enum):
    PENDING   = "PENDING"
    ACTIVE    = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


class CognitivePhase(str, Enum):
    RECALL    = "RECALL"
    PLANNING  = "PLANNING"
    EXECUTION = "EXECUTION"
    SYNTHESIS = "SYNTHESIS"
    IDLE      = "IDLE"


class PerformanceMode(str, Enum):
    STANDARD = "standard"  # current behaviour — balanced capabilities
    FULL     = "full"      # no limitations (reserved for future use)


class QueryType(str, Enum):
    LOCAL_PRECISE    = "LOCAL_PRECISE"
    GLOBAL_SYNTHETIC = "GLOBAL_SYNTHETIC"
    MULTI_HOP        = "MULTI_HOP"
    PROCEDURAL       = "PROCEDURAL"
    DIAGNOSTIC       = "DIAGNOSTIC"


class RecallSource(str, Enum):
    HYBRID_EPISODIC = "HYBRID_EPISODIC"
    HYBRID_GLOBAL   = "HYBRID_GLOBAL"
    PROTECTED       = "PROTECTED"
    GEOMETRIC       = "GEOMETRIC"


class ChunkKind(str, Enum):
    """Epistemic kind of a web-content chunk.

    Drives two gates:
      prompt_eligible   — whether the chunk may appear in the LLM context window
      evidence_eligible — whether the chunk may satisfy source_lock / promote to FACT
    """
    BODY_PARAGRAPH  = "body_paragraph"   # main article text
    HEADING         = "heading"          # section title
    INFOBOX_FIELD   = "infobox_field"    # key: value structured data (high evidential value)
    TABLE_ROW       = "table_row"        # tabular content
    CAPTION         = "caption"          # figure/table caption
    NAV_BOILERPLATE = "nav_boilerplate"  # menus, page footer, jump links
    CATEGORY        = "category"         # category/tag listings
    REFERENCE_LIST  = "reference_list"   # citation / reference entries
    RAW_FALLBACK    = "raw_fallback"     # unclassifiable content

    @property
    def prompt_eligible(self) -> bool:
        """True when this chunk kind may be injected into the LLM prompt."""
        return self.value in {
            "body_paragraph", "heading", "infobox_field", "table_row", "caption",
        }

    @property
    def evidence_eligible(self) -> bool:
        """True when this chunk kind may serve as evidence (FACT, source_lock)."""
        return self.value in {
            "body_paragraph", "infobox_field", "table_row",
        }


# ── Execution Model ───────────────────────────────────────────────────────────

class ExecutionMode(str, Enum):
    """Execution mode derived deterministically from the user request.

    Controls: which tools are offered, whether slot contract is active,
    whether compute is immediately available, and how the answer is validated.

    DIRECT_ANSWER:          Factual/riddle — answerable from training knowledge.
                            No tools required; no slot contract.
    DIRECT_ARITHMETIC:      Math with constants known at query time (4 dozen = 48).
                            compute allowed immediately; no fetch required.
    PROMPT_CONTAINED_QUANT: Quantitative where all numeric inputs are explicitly
                            stated in the user prompt (e.g. "42.195 km in 2:01:39").
                            compute allowed immediately; no fetch or slot contract.
    SOURCE_BOUND_QUANT:     Quantitative requiring web evidence (Kipchoge pace).
                            Slot contract active; fetch required before compute.
    ATTACHMENT_REQUIRED: Task involves file/spreadsheet/image/code.
                         workspace tool required; files must be present.
    BROWSE_LOOKUP:       General external-web factual lookup.
                         web(fetch/render) required; no slot contract.
    MEMORY_RAG:          Internal knowledge retrieval only.
                         memory tool; no web.
    """
    DIRECT_ANSWER          = "DIRECT_ANSWER"
    DIRECT_ARITHMETIC      = "DIRECT_ARITHMETIC"
    PROMPT_CONTAINED_QUANT = "PROMPT_CONTAINED_QUANT"
    SOURCE_BOUND_QUANT     = "SOURCE_BOUND_QUANT"
    ATTACHMENT_REQUIRED    = "ATTACHMENT_REQUIRED"
    BROWSE_LOOKUP          = "BROWSE_LOOKUP"
    MEMORY_RAG             = "MEMORY_RAG"

    @property
    def requires_web(self) -> bool:
        return self in (ExecutionMode.SOURCE_BOUND_QUANT, ExecutionMode.BROWSE_LOOKUP)

    @property
    def slot_contract(self) -> bool:
        """True when compute is blocked until all declared slots are resolved."""
        return self == ExecutionMode.SOURCE_BOUND_QUANT

    @property
    def compute_immediate(self) -> bool:
        """True when compute is allowed without prior fetches."""
        return self in (
            ExecutionMode.DIRECT_ARITHMETIC,
            ExecutionMode.DIRECT_ANSWER,
            ExecutionMode.PROMPT_CONTAINED_QUANT,
        )


class ExecutionStage(str, Enum):
    """Stage within a turn's execution pipeline.

    Transitions are one-directional. Progress invariant: at least one of
    artifact_count, resolved_slot_count, compute_done, or answer_valid must
    grow at each stage boundary; otherwise the turn is stalled.
    """
    INIT              = "INIT"              # before any tool calls
    EVIDENCE_GATHERING = "EVIDENCE_GATHERING"  # fetching web/memory artifacts
    SLOT_RESOLUTION   = "SLOT_RESOLUTION"  # resolving declared slots from artifacts
    COMPUTATION       = "COMPUTATION"      # running compute() with resolved slots
    SYNTHESIS         = "SYNTHESIS"        # building final answer from all evidence


class TurnOutcome(str, Enum):
    """Canonical terminal outcome code for every agent turn.

    Assigned at the single canonical exit point of _cognitive_loop /
    handle_turn and persisted in TurnTrace.  Enables benchmark and postmortem
    tools to classify turns without parsing free-text logs.

    SUCCESS             — turn produced a valid, published answer
    STALL               — repeated identical tool calls / iteration limit reached
    TIMEOUT             — asyncio.TimeoutError from handle_turn wrapper
    CANCELLED           — CancelledError (user pressed stop / SSE disconnect)
    NO_EVIDENCE         — synthesis blocked: unsatisfied source reqs + no artifacts
    COMPUTE_BLOCKED     — compute contract pending and gate never relaxed
    TOOL_PROTOCOL_ERROR — synthesis leak retry limit exceeded (tool tags in reply)
    SYNTHESIS_INVALID   — final validator rejected answer (other class)
    INTERNAL_ERROR      — unhandled exception
    """
    SUCCESS             = "success"
    STALL               = "stall"
    TIMEOUT             = "timeout"
    CANCELLED           = "cancelled"
    NO_EVIDENCE         = "no_evidence"
    COMPUTE_BLOCKED     = "compute_blocked"
    TOOL_PROTOCOL_ERROR = "tool_protocol_error"
    SYNTHESIS_INVALID   = "synthesis_invalid"
    INTERNAL_ERROR      = "internal_error"


@dataclass(frozen=True)
class OutputContract:
    """Canonical answer contract used across planning, computation, and synthesis.

    Keeps the final representation requirements in one place while preserving
    the legacy format/unit fields that existing components still consume.
    """
    format: str = "text"
    representation: str | None = None
    base_unit: str | None = None
    display_scale: str | None = None
    rounding_rule: str | None = None
    precision: int | None = None
    required_evidence: bool = False

    @classmethod
    def from_legacy(
        cls,
        *,
        format: str = "text",
        unit: str | None = None,
        precision: int | None = None,
        required_evidence: bool = False,
    ) -> OutputContract:
        return cls(
            format=format,
            base_unit=unit,
            precision=precision,
            required_evidence=required_evidence,
        )

    @classmethod
    def merge(
        cls,
        existing: "OutputContract | None",
        *,
        format: str = "text",
        unit: str | None = None,
        precision: int | None = None,
        required_evidence: bool = False,
    ) -> "OutputContract":
        """Merge legacy scalar fields into an OutputContract.

        When *existing* is None a new contract is created from the scalars.
        When *existing* is already set its rich fields (representation,
        display_scale, rounding_rule) are preserved; scalars only fill
        gaps that existing leaves as None.
        """
        if existing is None:
            return cls.from_legacy(
                format=format,
                unit=unit,
                precision=precision,
                required_evidence=required_evidence,
            )
        merged_format = existing.format
        if format != "text" and merged_format == "text":
            merged_format = format
        return cls(
            format=merged_format,
            representation=existing.representation,
            base_unit=existing.base_unit if existing.base_unit is not None else unit,
            display_scale=existing.display_scale,
            rounding_rule=existing.rounding_rule,
            precision=existing.precision if existing.precision is not None else precision,
            required_evidence=existing.required_evidence or required_evidence,
        )


@dataclass(frozen=True)
class AnswerSchema:
    """Describes the expected shape of the final answer.

    format:      "text" | "number" | "table" | "list" | "json"
    unit:        SI or natural unit (km, min/km, °C, …) — for number answers
    precision:   significant figures or decimal places — for number answers
    required_evidence: whether at least one evidence citation is required
    """
    format:            str = "text"
    unit:              str | None = None
    precision:         int | None = None
    required_evidence: bool = False
    output_contract:   OutputContract | None = None

    def __post_init__(self) -> None:
        contract = OutputContract.merge(
            self.output_contract,
            format=self.format,
            unit=self.unit,
            precision=self.precision,
            required_evidence=self.required_evidence,
        )
        object.__setattr__(self, "output_contract", contract)
        object.__setattr__(self, "format", contract.format)
        object.__setattr__(self, "unit", contract.base_unit)
        object.__setattr__(self, "precision", contract.precision)
        object.__setattr__(self, "required_evidence", contract.required_evidence)


@dataclass(frozen=True)
class ComputeTrace:
    """One compute execution result.

    It may be an intermediate calculation or the final computation already in the
    target representation. The engine can keep several traces and later select a
    single FinalComputeResult without losing prior steps.

    ComputeTrace records what happened. It is intentionally weaker than
    FinalComputeResult and must not be treated as a verified final answer unless
    it has been promoted explicitly.
    """
    value: str
    expression: str | None = None
    unit: str | None = None
    iteration: int | None = None
    is_final: bool = False
    output_contract: OutputContract | None = None
    evidence_ids: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def rendered_value(self) -> str:
        if self.unit:
            return f"{self.value} {self.unit}".strip()
        return self.value


@dataclass(frozen=True)
class FinalComputeResult:
    """Canonical final compute output promoted from a compute trace.

    Unlike ComputeTrace, this object represents the compute result that already
    satisfies the frozen OutputContract and is therefore safe to expose in the
    answer turn as a verified result.
    """
    value: str
    output_contract: OutputContract = field(default_factory=OutputContract)
    unit: str | None = None
    source_expression: str | None = None
    source_iteration: int | None = None
    evidence_ids: tuple[str, ...] = ()
    notes: str | None = None

    @classmethod
    def from_trace(
        cls,
        trace: ComputeTrace,
        *,
        output_contract: OutputContract | None = None,
    ) -> FinalComputeResult:
        contract = output_contract or trace.output_contract or OutputContract.from_legacy(
            format="number",
            unit=trace.unit,
        )
        representation = " ".join((contract.representation or "").lower().replace("_", " ").replace("-", " ").split())
        rendered_unit = None if representation == "bare number" else (trace.unit or contract.base_unit)
        return cls(
            value=trace.value,
            output_contract=contract,
            unit=rendered_unit,
            source_expression=trace.expression,
            source_iteration=trace.iteration,
            evidence_ids=trace.evidence_ids,
            notes=trace.notes,
        )

    @property
    def rendered_value(self) -> str:
        representation = " ".join((self.output_contract.representation or "").lower().replace("_", " ").replace("-", " ").split())
        if representation == "bare number":
            return self.value
        unit = self.unit or self.output_contract.base_unit
        if unit:
            return f"{self.value} {unit}".strip()
        return self.value


@dataclass(frozen=True)
class TaskSpec:
    """Immutable per-turn task specification derived by TaskSpecBuilder.

    Drives the entire execution: tool selection, slot contract, compute gate,
    drift monitors, and final answer validation.

    evidence_requirements: list of (kind, hint) tuples.
      kind: "url" | "domain" | "attachment" | "memory"
      hint: URL fragment, domain name, filename, or keyword
    """
    mode:                  ExecutionMode
    answer_schema:         AnswerSchema
    evidence_requirements: tuple[tuple[str, str], ...] = ()
    slot_names:            tuple[str, ...] = ()  # canonical slot names for SOURCE_BOUND_QUANT
    has_attachment:        bool = False
    output_contract:       OutputContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_contract",
            OutputContract.merge(
                self.output_contract or self.answer_schema.output_contract,
                format=self.answer_schema.format,
                unit=self.answer_schema.unit,
                precision=self.answer_schema.precision,
                required_evidence=self.answer_schema.required_evidence,
            ),
        )

    def context_summary(self) -> str:
        """Cognitive scaffold injected into every iteration so the model knows what it must do."""
        lines = [f"Type: {self.mode.value}"]
        contract = self.output_contract or self.answer_schema.output_contract
        if contract.format != "text":
            fmt = contract.format + (f" in {contract.base_unit}" if contract.base_unit else "")
            lines.append(f"Answer: {fmt}")
        extras: list[str] = []
        if contract.representation:
            extras.append(f"representation={contract.representation}")
        if contract.display_scale:
            extras.append(f"display_scale={contract.display_scale}")
        if contract.rounding_rule:
            extras.append(f"rounding_rule={contract.rounding_rule}")
        if extras:
            lines.append("Output contract: " + ", ".join(extras))
        if self.mode.requires_web:
            lines.append("Constraint: fetch web evidence BEFORE calling compute")
        if self.mode.compute_immediate:
            lines.append("Constraint: compute allowed immediately — no fetch required")
        return "\n".join(lines)


# ── Turn Evidence Register ────────────────────────────────────────────────────

@dataclass
class TurnEvidenceEntry:
    """One piece of evidence gathered during the current turn."""
    label:       str            # E1, E2… for web/memory; C1, C2… for compute
    tool:        str            # "web" | "memory" | "compute"
    source:      str            # URL, query fragment, or expression
    status:      str            # "indexed" | "retrieved" | "computed" | "failed" | "empty"
    summary:     str            # what was found or computed
    reliability: str = "UNKNOWN"  # AUTHORITATIVE | HIGH | MEDIUM | LOW | UNKNOWN


@dataclass
class TurnEvidenceRegister:
    """Structured record of evidence accumulated within a turn.

    The engine populates this after each tool batch. The message builder
    injects it into memory_context so the model always sees exactly what
    it has and what it still needs — no blind tool calls.

    Each entry carries a reliability tier (AUTHORITATIVE → UNKNOWN) so the
    model and constraint verifier can weight evidence by epistemic quality.
    """
    _entries:      list[TurnEvidenceEntry] = field(default_factory=list)
    _evidence_seq: int = 0   # E1, E2 …
    _compute_seq:  int = 0   # C1, C2 …

    def add_web(self, url: str, status: str, summary: str, reliability: str = "UNKNOWN") -> str:
        self._evidence_seq += 1
        label = f"E{self._evidence_seq}"
        self._entries.append(TurnEvidenceEntry(
            label=label, tool="web", source=url,
            status=status, summary=summary, reliability=reliability,
        ))
        return label

    def add_memory(self, query: str, summary: str) -> str:
        self._evidence_seq += 1
        label = f"E{self._evidence_seq}"
        self._entries.append(TurnEvidenceEntry(
            label=label, tool="memory", source=f"memory:{query}",
            status="retrieved", summary=summary, reliability="HIGH",
        ))
        return label

    def add_compute(self, expression: str, result: str) -> str:
        self._compute_seq += 1
        label = f"C{self._compute_seq}"
        self._entries.append(TurnEvidenceEntry(
            label=label, tool="compute", source=expression,
            status="computed", summary=result, reliability="AUTHORITATIVE",
        ))
        return label

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def context_summary(self) -> str:
        if not self._entries:
            return ""
        lines: list[str] = []
        for e in self._entries:
            rel = f" [{e.reliability}]" if e.reliability not in ("UNKNOWN",) else ""
            if e.status == "indexed":
                lines.append(f"[{e.label}] {e.source}{rel} — {e.summary} ✓")
            elif e.status == "retrieved":
                lines.append(f"[{e.label}] {e.source}{rel} — {e.summary} ✓")
            elif e.status == "computed":
                lines.append(f"[{e.label}] compute({e.source}) = {e.summary} [AUTHORITATIVE]")
            elif e.status == "failed":
                lines.append(f"[{e.label}] {e.source} — FAILED")
            else:
                lines.append(f"[{e.label}] {e.source} — empty (no extractable content)")
        return "\n".join(lines)

    @property
    def entries(self):
        return self._entries


class KimaDeltaType(str, Enum):
    TOKEN             = "TOKEN"
    REASONING         = "REASONING"   # accumulated <think> block — single event per turn
    THOUGHT           = "THOUGHT"
    TOOL_RESULT       = "TOOL_RESULT"
    PLAN_STEP         = "PLAN_STEP"
    CONFLICT_DETECTED = "CONFLICT_DETECTED"
    STALL             = "STALL"
    CONTEXT_REFRESH   = "CONTEXT_REFRESH"
    ERROR             = "ERROR"
    DONE              = "DONE"
    STRATEGY_SWITCH   = "STRATEGY_SWITCH"  # domain appendix A1 — autonomous strategy retry


class LLMEventType(str, Enum):
    TOKEN         = "TOKEN"
    REASONING     = "REASONING"   # accumulated <think> block — single event per turn
    TOOL_CALL     = "TOOL_CALL"
    CONCLUSIONS   = "CONCLUSIONS"
    PHASE_DECL    = "PHASE_DECL"
    DONE          = "DONE"
    STRATEGY_FAIL = "STRATEGY_FAIL"   # domain appendix A1 — emitted by LlamaCppAdapter


# ── CVS Weights ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CVSWeights:
    """Weights for CVS score computation (CIMA A-5).

    alpha: content-relevance weight
    beta:  recency weight
    gamma: novelty weight

    Canonical defaults: α=0.5, β=0.3, γ=0.2 (sum=1.0).
    """
    alpha: float = 0.5
    beta:  float = 0.3
    gamma: float = 0.2

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"CVSWeights must sum to 1.0, got {total}")

    @classmethod
    def default(cls) -> CVSWeights:
        return cls()


# ── Forget Parameters ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ForgetParams:
    """Parameters for the forget cycle (CIMA A-7).

    attenuation_threshold: importance below which active items get archived
    attenuation_age_days:  minimum age in days before attenuation is evaluated
    alpha_purge_days:      days since archiving before purge is considered
    min_importance_to_purge: purge only if importance < this threshold
    """
    attenuation_threshold:  float = 0.2
    attenuation_age_days:   float = 7.0
    alpha_purge_days:       float = 30.0
    min_importance_to_purge: float = 0.3

    @classmethod
    def default(cls) -> ForgetParams:
        return cls()


# ── Promotion Policy ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromotionPolicy:
    """Policy governing promotion of episodic C-Items to global scope.

    min_references: times item must be referenced before promotion
    min_importance:  minimum importance score for promotion eligibility
    """
    min_references: int   = 3
    min_importance: float = 0.7

    @classmethod
    def default(cls) -> PromotionPolicy:
        return cls()


# ── Phase Policy ─────────────────────────────────────────────────────────────

# Contextuality weights Xᵢ(phase) per (phase, item_type) — CIMA A-2.
# Defined in cima_demo/config/phase_policy.py to allow tuning without touching domain logic.
from cima_demo.config.phase_policy import DEFAULT_CONTEXTUALITY as _DEFAULT_CONTEXTUALITY


@dataclass(frozen=True)
class PhasePolicy:
    """Policy for phase detection heuristics and phase-dependent contextuality (CIMA A-2, A-8).

    plan_keyword_threshold: fraction of plan-signaling words to trigger PLANNING
    contextuality: tuple of (phase, item_type, weight) triples encoding Xᵢ(phase)
    """
    plan_keyword_threshold: float = 0.3
    contextuality: tuple[tuple[str, str, float], ...] = _DEFAULT_CONTEXTUALITY

    def get_contextuality(self, phase: str, item_type: str) -> float:
        """Return Xᵢ(phase) for the given phase and item type. Falls back to 0.5."""
        for p, t, w in self.contextuality:
            if p == phase and t == item_type:
                return w
        return 0.5

    @classmethod
    def default(cls) -> PhasePolicy:
        return cls()


# ── Context Budget ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContextBudget:
    """Token budget for a single LLM turn.

    max_tokens:           model ctx_size (65536 for Mistral-3-14B)
    overhead_tokens:      system prompt + history tokens (estimated)
    available_for_content: max_tokens - overhead_tokens
    """
    max_tokens:    int = 32_768
    overhead_tokens: int = 4_096

    @property
    def available_for_content(self) -> int:
        return max(0, self.max_tokens - self.overhead_tokens)

    @classmethod
    def production(cls) -> ContextBudget:
        return cls(max_tokens=32_768, overhead_tokens=4_096)

    @classmethod
    def testing(cls) -> ContextBudget:
        return cls(max_tokens=16_384, overhead_tokens=2_048)


# ── CItem Filter ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CItemFilter:
    """Unified filter for CItemStorePort.search() (domain v0.10 D-05).

    Replaces VectorFilter + FTSFilter from prior versions.
    actor field added in domain v0.10.3 (appendix A4) — None means no filter.
    """
    scope:           str | None       = None   # "episodic" | "global"
    scope_status:    str | None       = None   # "active" | "archived"
    conversation_id: str | None       = None
    item_types:      tuple[str, ...]  = ()
    conflict_status_in: tuple[str, ...] = ("none", "resolved")
    exclude_ids:     tuple[str, ...]  = ()
    actor:           str | None       = None   # "agent" | "user" | "leap_extractor" | None


# ── Scored CItem ─────────────────────────────────────────────────────────────

@dataclass
class ScoredCItem:
    """A retrieved C-Item with its recall score and provenance (domain v0.10 D-05).

    Eliminates PG round-trip after recall — citem has full content from Qdrant payload.

    Score lifecycle (pipeline stages):
      Qdrant recall  → score = cosine_similarity = dense_score
      RRF merge      → score = rrf_score;         dense_score preserved
      CrossEncoder   → score = rerank_score;       dense_score preserved; rerank_score set
      CVS rescoring  → score = cvs_density;        dense_score + rerank_score preserved

    dense_score and rerank_score are fixed once set and never overwritten, so ABS
    can always distinguish anchor (max rerank) from bridge (high dense, medium rerank).
    """
    citem:        CItem
    score:        float       # current pipeline score (changes through stages)
    provenance:   RecallSource
    dense_score:  float | None = 0.0  # cosine similarity from Qdrant; preserved through pipeline
    rerank_score: float | None = 0.0  # CrossEncoder score; set in _rerank, preserved thereafter
    rrf_score: float | None = None
    cvs_score: float | None = None
    value_density: float | None = None
    bridge_score: float | None = None
    rerank_verified: bool = False

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with",
    "de", "la", "el", "los", "las", "y", "o", "en", "con", "por", "para", "del", "al",
}

@dataclass(slots=True)
class QueryFacets:
    terms: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    phrases: tuple[str, ...] = ()


@dataclass(slots=True)
class BridgePolicy:
    enabled: bool
    alpha: float
    anchor_floor: float
    bridge_floor: float
    max_bridge_redundancy: float
    max_consecutive_low_bridge: int

    @classmethod
    def disabled(cls, alpha: float = 0.5) -> "BridgePolicy":
        return cls(
            enabled=False,
            alpha=alpha,
            anchor_floor=float("-inf"),
            bridge_floor=float("-inf"),
            max_bridge_redundancy=1.0,
            max_consecutive_low_bridge=999999,
        )


class DirectEvidenceStrategy(str, Enum):
    CVS_DENSITY = "cvs_density"
    ANCHOR_BRIDGE_INTERLEAVED = "anchor_bridge_interleaved"
# ── Chunk Result ─────────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """Output of ChunkingPort.chunk() (domain v0.10 D-03)."""
    text:         str
    index:        int
    filename:     str
    doc_type:     str
    page_num:     int | None = None   # source page (PDF/DOCX); None for unstructured text
    section_hint: str | None = None   # nearest heading preceding this chunk


# ── Rerank Result ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RerankResult:
    """Output of RerankerPort.rerank()."""
    index: int
    score: float


# ── Retrieval Plan ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalPlan:
    """Immutable retrieval plan produced by QueryPlanner (domain v0.9)."""
    query_type:        QueryType
    recall_top_k:      int
    rerank_top_n:      int
    geometric_expand:  bool
    geometric_seeds_k: int
    coverage_threshold: float
    estimated_hops:    int = 1   # coarse estimate from query type; refined by MultiHopAnalyzer at runtime


# ── Multi-hop Analysis ────────────────────────────────────────────────────────

@dataclass
class MultiHopAnalysis:
    """Result of MultiHopAnalyzer.analyze() — LLM-driven hop estimation + decomposition.

    hop_depth:     estimated number of reasoning hops (LLM-determined, semantically accurate).
    sub_questions: decomposed sub-questions covering ≤2 hops each; empty when hop_depth ≤ 2.
    """
    hop_depth:     int
    sub_questions: list[str] = field(default_factory=list)

    @property
    def needs_iterative_retrieval(self) -> bool:
        """True when the query requires more than one 2-hop retrieval iteration."""
        return self.hop_depth > 2 and len(self.sub_questions) > 1


# ── Budget Report ─────────────────────────────────────────────────────────────

@dataclass
class BudgetReport:
    """Token accounting after greedy_select."""
    total_available: int
    tokens_used:     int
    items_selected:  int

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.total_available - self.tokens_used)


# ── Coverage Report ──────────────────────────────────────────────────────────

@dataclass
class CoverageReport:
    """Coverage evaluation after slot assembly (RAG-06, INV-22)."""
    coverage_score:   float   # 0.0 – 1.0
    missing_concepts: list[str] = field(default_factory=list)
    retry_recommended: bool = False


# ── InputState ────────────────────────────────────────────────────────────────

class InputState(str, Enum):
    """Lifecycle state of a required task input (PR2).

    MISSING  — declared but no source located yet; model must search.
    LOCATED  — a potential source URL has been identified; model must fetch.
    VERIFIED — value extracted, evidence stored; slot is ready for compute.
    """
    MISSING  = "MISSING"
    LOCATED  = "LOCATED"
    VERIFIED = "VERIFIED"


# ── Task Slot / Task State ────────────────────────────────────────────────────

@dataclass
class TaskSlot:
    """A required variable for a task.

    Declared by the model as a CONSTRAINT conclusion with prefix 'SLOT name: description'.
    Resolved by a FACT conclusion with evidence and prefix 'SLOT_VALUE:name:value[:unit]'.

    PR2 lifecycle:
      MISSING  (default) → mark_located()  → LOCATED
      LOCATED  → mark_verified() / resolve() → VERIFIED
      MISSING  → mark_verified() / resolve() → VERIFIED  (shortcut)

    name:            canonical snake_case identifier (e.g. moon_min_perigee_km)
    description:     human-readable description of what this slot holds
    required_source: hostname/URL fragment that must be the evidence source
    value:           resolved value as string (set when VERIFIED)
    unit:            SI/natural unit (km, min/km, hours, …)
    evidence_id:     primary citem_id that provided the value (backward compat)
    evidence_refs:   all citem_ids that contributed evidence (PR2)
    locator_refs:    candidate source URLs identified but not yet fetched (PR2)
    state:           InputState lifecycle (PR2)
    resolved:        True when state == VERIFIED (backward compat mirror)
    """
    name: str
    description: str
    required_source: str | None = None
    value: str | None = None
    unit: str | None = None
    evidence_id: str | None = None
    resolved: bool = False
    # PR2 additions
    state: InputState = field(default_factory=lambda: InputState.MISSING)
    locator_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def resolve(
        self,
        value: str,
        unit: str | None = None,
        evidence_id: str | None = None,
    ) -> None:
        """Legacy resolve path — promotes directly to VERIFIED.

        Kept for backward compatibility with all existing callers.
        New code should prefer mark_verified().
        """
        self.value = value
        self.unit = unit or self.unit
        self.evidence_id = evidence_id
        self.resolved = True
        self.state = InputState.VERIFIED

    def mark_located(self, locator_refs: list[str]) -> None:
        """Promote slot to LOCATED: a candidate source was found, not yet fetched."""
        if self.state == InputState.VERIFIED:
            return  # never downgrade
        self.state = InputState.LOCATED
        for ref in locator_refs:
            if ref not in self.locator_refs:
                self.locator_refs.append(ref)

    def mark_verified(
        self,
        value: str,
        unit: str | None = None,
        evidence_id: str | None = None,
        evidence_refs: list[str] | None = None,
    ) -> None:
        """Promote slot to VERIFIED: value extracted and evidence stored."""
        self.value = value
        self.unit = unit or self.unit
        self.evidence_id = evidence_id
        self.resolved = True
        self.state = InputState.VERIFIED
        for ref in (evidence_refs or []):
            if ref and ref not in self.evidence_refs:
                self.evidence_refs.append(ref)
        if evidence_id and evidence_id not in self.evidence_refs:
            self.evidence_refs.insert(0, evidence_id)


@dataclass
class TaskState:
    """Typed task state — tracks slots, resolution, and compute/answer readiness.

    Lifecycle:
      CONSTRAINT conclusion 'SLOT name: …'    → slot declared (resolved=False)
      FACT conclusion with evidence            → slot resolved (resolved=True)
      all slots resolved                       → ready_to_compute = True

    slot_contract_required:
      When True, compute is ALWAYS blocked unless at least one slot has been
      declared AND all declared slots are resolved.  This is stricter than the
      default (which passes through when no slots are declared at all).
      Set by the orchestrator when source_requirements exist or the query is
      quantitative — ensures the model cannot short-circuit slot resolution.

    Injected into every system prompt iteration so the model knows what's pending.
    """
    objective: str
    slots: list[TaskSlot] = field(default_factory=list)
    output_unit: str | None = None      # legacy compatibility mirror of OutputContract.base_unit
    output_format: str | None = None    # legacy compatibility mirror of OutputContract.format
    slot_contract_required: bool = False
    output_contract: OutputContract | None = None
    compute_traces: list[ComputeTrace] = field(default_factory=list)
    final_compute_result: FinalComputeResult | None = None

    def __post_init__(self) -> None:
        self.output_contract = OutputContract.merge(
            self.output_contract,
            format=self.output_format or "text",
            unit=self.output_unit,
        )
        self.output_format = self.output_contract.format
        self.output_unit = self.output_contract.base_unit

    @property
    def unresolved_slots(self) -> list[TaskSlot]:
        """Slots not yet VERIFIED (MISSING or LOCATED). Backward compat."""
        return [s for s in self.slots if not s.resolved]

    # ── PR2: per-state slot views ─────────────────────────────────────────
    @property
    def missing_slots(self) -> list[TaskSlot]:
        """Slots with no source located yet."""
        return [s for s in self.slots if s.state == InputState.MISSING]

    @property
    def located_slots(self) -> list[TaskSlot]:
        """Slots with a candidate source URL identified, not yet fetched."""
        return [s for s in self.slots if s.state == InputState.LOCATED]

    @property
    def verified_slots(self) -> list[TaskSlot]:
        """Slots with extracted value and stored evidence."""
        return [s for s in self.slots if s.state == InputState.VERIFIED]

    @property
    def should_compute(self) -> bool:
        """True when all declared slots are VERIFIED — compute should fire.

        Unlike ready_to_compute, this does NOT require slot_contract_required=True.
        Use to harden the compute gate when enough slots are resolved, without
        enabling global slot_contract_required (which deadlocks Mistral tool-call mode).
        Returns False when no slots are declared (cannot infer readiness from nothing).
        """
        return bool(self.slots) and all(s.state == InputState.VERIFIED for s in self.slots)

    @property
    def ready_to_compute(self) -> bool:
        """True when compute is allowed.

        Standard mode (slot_contract_required=False):
          passes through when no slots are declared (model may compute freely
          once slots would not block it).
        Contract mode (slot_contract_required=True):
          requires at least one slot declared AND all declared slots resolved.
          An empty slot list → NOT ready (model must declare slots first).
        """
        if self.slot_contract_required:
            return bool(self.slots) and all(s.resolved for s in self.slots)
        return (not self.slots) or all(s.resolved for s in self.slots)

    @property
    def has_final_compute_result(self) -> bool:
        return self.final_compute_result is not None

    def get_slot(self, name: str) -> TaskSlot | None:
        return next((s for s in self.slots if s.name == name), None)

    def apply_output_contract(self, output_contract: OutputContract | None) -> None:
        self.output_contract = OutputContract.merge(
            output_contract,
            format=self.output_format or "text",
            unit=self.output_unit,
        )
        self.output_format = self.output_contract.format
        self.output_unit = self.output_contract.base_unit

    def add_compute_trace(self, trace: ComputeTrace) -> None:
        self.compute_traces.append(trace)
        # H-21: idempotent promotion — first contract-satisfying result wins.
        # A later is_final trace cannot overwrite an already-promoted result,
        # preventing accidental regression when the model re-runs compute with
        # a less precise expression after the contract was already satisfied.
        if trace.is_final and self.final_compute_result is None:
            self.final_compute_result = FinalComputeResult.from_trace(
                trace,
                output_contract=self.output_contract,
            )

    def context_summary(self) -> str:
        """Short string injected into system prompt so the model sees current slot state."""
        lines = [f"Objective: {self.objective}"]
        if self.output_unit:
            lines.append(f"Output unit: {self.output_unit}")
        if self.output_format:
            lines.append(f"Output format: {self.output_format}")
        contract = self.output_contract
        if contract is not None:
            extras: list[str] = []
            if contract.representation:
                extras.append(f"representation={contract.representation}")
            if contract.display_scale:
                extras.append(f"display_scale={contract.display_scale}")
            if contract.rounding_rule:
                extras.append(f"rounding_rule={contract.rounding_rule}")
            if extras:
                lines.append("Output contract: " + ", ".join(extras))
        if self.final_compute_result is not None:
            lines.append(f"Final compute result: {self.final_compute_result.rendered_value}")
        if self.slots:
            lines.append("Slots:")
            for s in self.slots:
                state_tag = s.state.value if hasattr(s.state, "value") else str(s.state)
                if s.state == InputState.VERIFIED:
                    val = f"{s.value}" + (f" {s.unit}" if s.unit else "")
                    ev = f" | evidence={s.evidence_refs[0]}" if s.evidence_refs else (
                        f" | evidence={s.evidence_id}" if s.evidence_id else "")
                    lines.append(f"  [{state_tag}] {s.name} = {val.strip()}{ev}")
                elif s.state == InputState.LOCATED:
                    src = f" | source={s.required_source}" if s.required_source else ""
                    locs = f" | locators={','.join(s.locator_refs[:2])}" if s.locator_refs else ""
                    lines.append(f"  [{state_tag}] {s.name}: {s.description}{src}{locs}")
                else:  # MISSING
                    src = f" | source={s.required_source}" if s.required_source else ""
                    lines.append(f"  [{state_tag}] {s.name}: {s.description}{src}")
        return "\n".join(lines)


# ── Task Plan ─────────────────────────────────────────────────────────────────

@dataclass
class TaskPlan:
    """Structured execution plan produced by TaskPlanner before the cognitive loop.

    One LLM.complete() call at turn start — no streaming, temperature=0.
    Injected into memory_context as ## Task at every loop iteration.

    Separates what the model SEES (objective + constraints + steps) from
    what the engine uses internally (TaskSpec modes and gates).
    """
    objective:    str
    constraints:  list[str] = field(default_factory=list)
    needs_search: bool = True
    needs_compute: bool = False
    strategies:   list[str] = field(default_factory=list)
    plan_steps:   list[str] = field(default_factory=list)
    answer_format: str = "text"
    answer_unit:  str | None = None
    output_contract: OutputContract | None = None
    # ── Deterministic compute (DP-3) ─────────────────────────────────────────
    # When present, the engine auto-creates slots from compute_variables,
    # waits until all are VERIFIED, substitutes values into compute_formula,
    # and evaluates mechanically — no LLM expression construction at runtime.
    compute_formula:    str | None = None
    compute_variables:  list[str] = field(default_factory=list)
    # ── Rich step semantics (proactive decomposition bridge) ──────────────────
    plan_steps_detail: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_contract = OutputContract.merge(
            self.output_contract,
            format=self.answer_format,
            unit=self.answer_unit,
        )
        self.answer_format = self.output_contract.format
        self.answer_unit = self.output_contract.base_unit

    def context_summary(self) -> str:
        lines = [f"Objective: {self.objective}"]
        if self.constraints:
            lines.append("Constraints:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        if self.answer_format != "text":
            fmt = self.answer_format + (f" ({self.answer_unit})" if self.answer_unit else "")
            lines.append(f"Answer: {fmt}")
        if self.output_contract is not None:
            extras: list[str] = []
            if self.output_contract.representation:
                extras.append(f"representation={self.output_contract.representation}")
            if self.output_contract.display_scale:
                extras.append(f"display_scale={self.output_contract.display_scale}")
            if self.output_contract.rounding_rule:
                extras.append(f"rounding_rule={self.output_contract.rounding_rule}")
            if extras:
                lines.append("Output contract: " + ", ".join(extras))
        if self.strategies:
            lines.append("Strategies (try in order, cheapest first):")
            for i, s in enumerate(self.strategies, 1):
                lines.append(f"  {i}. {s}")
        if self.plan_steps:
            lines.append("Steps:")
            for i, step in enumerate(self.plan_steps, 1):
                lines.append(f"  {i}. {step}")
        return "\n".join(lines)


# ── Reasoning Config ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReasoningConfig:
    """Feature flags for test-time compute scaling (KIMA_SystemPrompt_v0.1 §4).

    All multi-call flags default to n=1 (disabled).
    graph_context_depth is zero-cost — enabled by default.

    SP-INV-01: any failure in scoring/NLI degrades silently to n=1.
    SP-INV-03: immutable — does not mutate at runtime.
    """
    graph_context_depth:        int = 1   # 0=off, 1=1-hop dep annotations
    plan_best_of_n:             int = 1   # 1=off, 3=best-of-3
    conflict_self_consistency_n: int = 1  # 1=off, 3=self-consistency NLI

    @classmethod
    def default(cls) -> ReasoningConfig:
        """Production initial config: only graph annotations enabled."""
        return cls(graph_context_depth=1, plan_best_of_n=1, conflict_self_consistency_n=1)

    @classmethod
    def from_env(cls) -> ReasoningConfig:
        from cima_demo.api.settings import get_settings
        s = get_settings()
        return cls(
            graph_context_depth         = s.graph_context_depth,
            plan_best_of_n              = s.plan_best_of_n,
            conflict_self_consistency_n = s.conflict_sc_n,
        )


# ── Source Requirement ────────────────────────────────────────────────────────

@dataclass
class SourceRequirement:
    """A source explicitly named in the user message that must be fetched before compute.

    kind: "url"    — an explicit https:// URL was mentioned
          "domain" — a known keyword maps to a domain (via _DOMAIN_ALIASES)
    normalized: canonical form used for matching (see normalize_url in source_lock.py)
    satisfied: set True after successful fetch with evidence_atoms
    """
    kind: str        # "url" | "domain"
    value: str       # original token from user message
    normalized: str  # for comparison
    satisfied: bool = False


# ── Turn Contract — frozen production specification ────────────────────────────

@dataclass(frozen=True)
class TurnContract:
    """Immutable specification of what this turn must produce.

    Built once after TaskSpec is known. Never mutated. All components read
    from here; the OutputContract is the canonical answer format for the
    entire turn lifecycle — replan cannot change it.
    """
    # Identity
    conversation_id: str
    turn_id: str
    user_message: str
    phase: str

    # Task classification (drives tool selection and validation)
    mode: ExecutionMode
    output_contract: OutputContract
    task_spec: TaskSpec                # full spec for components that need it

    # What must be gathered (drives TransitionPolicy decisions)
    needs_evidence: bool = False       # web or memory retrieval required
    needs_compute: bool = False        # compute() call required
    slot_names: tuple[str, ...] = ()  # for SOURCE_BOUND_QUANT slot tracking

    # Execution bounds (injected from engine settings at turn start)
    max_iterations: int = 8
    max_stall_count: int = 5
    max_strategy_retries: int = 1

    @classmethod
    def from_task_spec(
        cls,
        *,
        conversation_id: str,
        turn_id: str,
        user_message: str,
        phase: str,
        task_spec: "TaskSpec",
        task_plan: "TaskPlan | None" = None,
        max_iterations: int = 8,
        max_stall_count: int = 5,
        max_strategy_retries: int = 1,
    ) -> "TurnContract":
        """Build a frozen contract from a TaskSpec (and optional initial TaskPlan)."""
        needs_evidence = task_spec.mode.requires_web or bool(
            task_plan and task_plan.needs_search
        )
        needs_compute = (
            task_spec.mode == ExecutionMode.SOURCE_BOUND_QUANT
            or bool(task_plan and task_plan.needs_compute)
        )
        return cls(
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message=user_message,
            phase=phase,
            mode=task_spec.mode,
            output_contract=task_spec.output_contract or OutputContract(),
            task_spec=task_spec,
            needs_evidence=needs_evidence,
            needs_compute=needs_compute,
            slot_names=task_spec.slot_names,
            max_iterations=max_iterations,
            max_stall_count=max_stall_count,
            max_strategy_retries=max_strategy_retries,
        )


# ── Turn Progress — append-only evidence accumulator ─────────────────────────

@dataclass
class TurnProgress:
    """Mutable accumulator of evidence gathered during one agent turn.

    Fields only grow or transition in one direction (False → True, 0 → N).
    TransitionPolicy reads this (together with TurnContract) to derive the
    next turn-level transition without touching any mutable engine state.
    """
    # Evidence counts
    artifact_count: int = 0
    resolved_slot_count: int = 0

    # Structured evidence
    evidence_register: "TurnEvidenceRegister | None" = None

    # Compute results
    compute_traces: list["ComputeTrace"] = field(default_factory=list)
    final_compute_result: "FinalComputeResult | None" = None
    compute_done: bool = False

    # Compute gate patience (CR-1)
    compute_gate_no_compute_iters: int = 0

    # Iteration / stall
    iteration_count: int = 0
    stall_hashes: list[str] = field(default_factory=list)
    stall_occurred: bool = False

    # Drift state
    global_drift_detected: bool = False

    # Tool tracking (dedup)
    tool_calls_emitted: list[str] = field(default_factory=list)

    # Replan tracking
    last_validation_gaps: list[str] = field(default_factory=list)
    replan_count: int = 0

    # Web search locators (shown in context header on each iteration)
    search_result_locators: list[str] = field(default_factory=list)

    @property
    def has_final_compute_result(self) -> bool:
        """True when a canonical verified compute result has been recorded."""
        return self.final_compute_result is not None
