"""Typed contracts for CIMA Demonstrator runtime artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "name"):
        return getattr(value, "value")
    return value


@dataclass(slots=True)
class RunManifest:
    schema_version: str = "cima_demo.run_manifest.v1"
    run_id: str = ""
    conversation_id: str = ""
    turn_id: str = ""
    status: str = "running"
    user_message: str = ""
    attached_files: list[dict[str, Any]] = field(default_factory=list)
    cognitive_phase: str | None = None
    execution_mode: str | None = None
    active_plan_id: str | None = None
    task_memory: dict[str, Any] = field(default_factory=dict)
    task_spec: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] | None = None
    output_contract: dict[str, Any] | None = None
    assistant_reply: str = ""
    error_class: str | None = None
    checkpoint_count: int = 0
    phase_count: int = 0
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class RunPhaseRecord:
    schema_version: str = "cima_demo.run_phase.v1"
    run_id: str = ""
    sequence: int = 0
    phase_name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class RunCheckpoint:
    schema_version: str = "cima_demo.run_checkpoint.v1"
    checkpoint_id: str = ""
    run_id: str = ""
    sequence: int = 0
    checkpoint_kind: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class RunBundle:
    manifest: dict[str, Any]
    phases: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": _jsonable(self.manifest),
            "phases": _jsonable(self.phases),
            "checkpoints": _jsonable(self.checkpoints),
        }


@dataclass(slots=True)
class DemoSourceRecord:
    schema_version: str = "cima_demo.source.v1"
    source_id: str = ""
    conversation_id: str = ""
    source_kind: str = ""
    role: str | None = None
    origin_ref: str | None = None
    display_text: str | None = None
    process_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class DemoSourceSpan:
    schema_version: str = "cima_demo.source_span.v1"
    span_id: str = ""
    source_id: str = ""
    conversation_id: str = ""
    span_kind: str = ""
    char_start: int = 0
    char_end: int = 0
    locator: dict[str, Any] = field(default_factory=dict)
    preview_text: str = ""
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class DemoLineageEdge:
    schema_version: str = "cima_demo.lineage_edge.v1"
    edge_id: str = ""
    conversation_id: str = ""
    src_kind: str = ""
    src_id: str = ""
    dst_kind: str = ""
    dst_id: str = ""
    relation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class SummaryResolution:
    schema_version: str = "cima_demo.summary_resolution.v1"
    summary_id: str = ""
    conversation_id: str = ""
    summary_text: str = ""
    origin_citem_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ContextSnapshot:
    schema_version: str = "cima_demo.context_snapshot.v1"
    context_id: str = ""
    run_id: str = ""
    conversation_id: str = ""
    turn_id: str = ""
    query_text: str = ""
    phase: str | None = None
    context_text: str = ""
    markers: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    auxiliary_items: list[dict[str, Any]] = field(default_factory=list)
    dropped_uncitable_items: list[dict[str, Any]] = field(default_factory=list)
    context_drop_metrics: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    resolved_source_ids: list[str] = field(default_factory=list)
    resolved_span_ids: list[str] = field(default_factory=list)
    resolved_source_count: int = 0
    resolved_span_count: int = 0
    unresolved_ref_ids: list[str] = field(default_factory=list)
    marker_resolution: list[dict[str, Any]] = field(default_factory=list)
    evidence_marker_registry: list[dict[str, Any]] = field(default_factory=list)
    # Prompt-visible support is separate from stored lineage. A marker is usable
    # by the LLM only if its supporting text was actually rendered into the prompt.
    visible_marker_support: list[dict[str, Any]] = field(default_factory=list)
    visible_support_metrics: dict[str, Any] = field(default_factory=dict)
    resolution_mode: str = "empty"
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class AnswerLineage:
    schema_version: str = "cima_demo.answer_lineage.v1"
    answer_lineage_id: str = ""
    conversation_id: str = ""
    run_id: str = ""
    response_turn_id: str | None = None
    context_id: str | None = None
    answer_text: str = ""
    cited_markers: list[str] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    resolved_source_ids: list[str] = field(default_factory=list)
    resolved_span_ids: list[str] = field(default_factory=list)
    resolved_source_count: int = 0
    resolved_span_count: int = 0
    unresolved_ref_ids: list[str] = field(default_factory=list)
    marker_resolution: list[dict[str, Any]] = field(default_factory=list)
    resolution_mode: str = "empty"
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class NeedProposal:
    schema_version: str = "cima_demo.need_proposal.v1"
    needs_zoom: bool = False
    zoom_markers: list[str] = field(default_factory=list)
    needs_zoom_out: bool = False
    focus: str | None = None
    reason: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "NeedProposal":
        payload = payload or {}
        return cls(
            needs_zoom=bool(payload.get("needs_zoom", False)),
            zoom_markers=[str(v) for v in payload.get("zoom_markers", []) if str(v)],
            needs_zoom_out=bool(payload.get("needs_zoom_out", False)),
            focus=(str(payload.get("focus")).strip() or None) if payload.get("focus") is not None else None,
            reason=(str(payload.get("reason")).strip() or None) if payload.get("reason") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class MemoryConclusion:
    kind: str = "NOTE"
    content: str = ""
    confidence: float = 0.7

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryConclusion | None":
        payload = payload or {}
        content = str(payload.get("content", "")).strip()
        if not content:
            return None
        kind = str(payload.get("kind", "NOTE")).strip().upper() or "NOTE"
        confidence = payload.get("confidence", 0.7)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.7
        return cls(kind=kind, content=content, confidence=max(0.0, min(1.0, confidence)))


@dataclass(slots=True)
class MemoryProposal:
    schema_version: str = "cima_demo.memory_proposal.v1"
    cited_markers: list[str] = field(default_factory=list)
    conclusions: list[MemoryConclusion] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryProposal":
        payload = payload or {}
        conclusions: list[MemoryConclusion] = []
        for item in payload.get("conclusions", []) or []:
            if not isinstance(item, dict):
                continue
            parsed = MemoryConclusion.from_dict(item)
            if parsed is not None:
                conclusions.append(parsed)
        return cls(
            cited_markers=[str(v) for v in payload.get("cited_markers", []) if str(v)],
            conclusions=conclusions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cited_markers": list(self.cited_markers),
            "conclusions": [_jsonable(asdict(item)) for item in self.conclusions],
        }


@dataclass(slots=True)
class GeometryItemState:
    schema_version: str = "cima_demo.geometry_item_state.v1"
    conversation_id: str = ""
    ref_kind: str = "citem"
    ref_id: str = ""
    run_id: str = ""
    cluster_top1: str = ""
    cluster_top2: str | None = None
    w1: float = 1.0
    w2: float | None = None
    margin: float = 1.0
    is_core: bool = False
    is_bridge_candidate: bool = False
    centrality: float = 0.0
    label: str | None = None
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class GeometryClusterState:
    schema_version: str = "cima_demo.geometry_cluster_state.v1"
    conversation_id: str = ""
    cluster_id: str = ""
    run_id: str = ""
    mass: float = 0.0
    medoid_ref_id: str = ""
    summary_id: str | None = None
    label: str | None = None
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class GeometryRunReport:
    schema_version: str = "cima_demo.geometry_run.v1"
    run_id: str = ""
    conversation_id: str = ""
    reason: str = ""
    algo_version: str = "geom_v1"
    n_items: int = 0
    cluster_count: int = 0
    core_count: int = 0
    bridge_count: int = 0
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class HandoffManifest:
    schema_version: str = "cima_demo.handoff_manifest.v1"
    handoff_id: str = ""
    conversation_id: str = ""
    source_run_id: str = ""
    context_id: str | None = None
    citem_refs: list[str] = field(default_factory=list)
    pyramid_refs: list[str] = field(default_factory=list)
    task_state: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None
    bundled_citems: list[dict[str, Any]] = field(default_factory=list)
    bundled_summaries: list[dict[str, Any]] = field(default_factory=list)
    bundled_sources: list[dict[str, Any]] = field(default_factory=list)
    bundled_spans: list[dict[str, Any]] = field(default_factory=list)
    bundled_lineage: list[dict[str, Any]] = field(default_factory=list)
    checksum: str | None = None
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class HandoffValidation:
    schema_version: str = "cima_demo.handoff_validation.v1"
    handoff_id: str = ""
    valid: bool = False
    issues: list[str] = field(default_factory=list)
    evidence_coverage: float | None = None
    validated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class HandoffRestore:
    schema_version: str = "cima_demo.handoff_restore.v1"
    restore_id: str = ""
    handoff_id: str = ""
    target_conversation_id: str = ""
    target_run_id: str | None = None
    valid: bool = False
    reconstructed_task_state: dict[str, Any] = field(default_factory=dict)
    diff: dict[str, Any] = field(default_factory=dict)
    restored_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class GCAuditRecord:
    schema_version: str = "cima_demo.gc_audit.v1"
    audit_id: str = ""
    conversation_id: str = ""
    action: str = ""
    status: str = "ok"
    run_id: str | None = None
    phase: str | None = None
    before_counts: dict[str, Any] = field(default_factory=dict)
    after_counts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error_class: str | None = None
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))
