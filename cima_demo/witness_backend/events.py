from __future__ import annotations

"""CloudEvents models for the CIMA witness backend v1.1.

This module intentionally models only envelope + canonical payload contracts.
It follows the approved witness-backend decisions:
- CloudEvents 1.0 JSON as the only event envelope.
- Kafka carries ids, metadata and metrics, never raw user text.
- Local/global memory are physically separate in the witness backend.
- Geometry is a separate bounded context with compacted state topics.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(StrEnum):
    MEMORY_SOURCE_REGISTERED = "cima.memory.source.registered.v1"
    MEMORY_FILE_UPLOADED = "cima.memory.file.uploaded.v1"
    MEMORY_CHUNK_CREATED = "cima.memory.chunk.created.v1"
    MEMORY_EDU_SEGMENTED = "cima.memory.edu.segmented.v1"
    MEMORY_CITEM_CREATED = "cima.memory.citem.created.v1"
    MEMORY_SUMMARY_CREATED = "cima.memory.summary.created.v1"
    MEMORY_SUMMARY_UPDATED = "cima.memory.summary.updated.v1"
    MEMORY_CITEM_PROMOTED_GLOBAL = "cima.memory.citem.promoted_global.v1"
    VECTOR_UPSERTED = "cima.vector.upserted.v1"
    VECTOR_DELETED = "cima.vector.deleted.v1"
    CONVERSATION_HARD_DELETE_REQUESTED = "cima.conversation.hard_delete.requested.v1"
    CONVERSATION_HARD_DELETE_COMPLETED = "cima.conversation.hard_delete.completed.v1"
    PIN_SET = "cima.pin.set.v1"
    PIN_UNSET = "cima.pin.unset.v1"
    GEOM_RECOMPUTE = "cima.geom.recompute.requested.v1"
    GEOM_PURGE = "cima.geom.purge.requested.v1"
    GEOM_SET_PARAMS = "cima.geom.params.set.v1"
    GEOM_RUN_COMPLETED = "cima.geom.run.completed.v1"
    GEOM_ITEM_STATE = "cima.geom.item_state.v1"
    GEOM_CLUSTER_STATE = "cima.geom.cluster_state.v1"
    SUMMARY_REQUESTED = "cima.summary.requested.v1"
    HANDOFF_CREATED = "cima.handoff.created.v1"
    HANDOFF_VALIDATED = "cima.handoff.validated.v1"
    HANDOFF_RESTORED = "cima.handoff.restored.v1"
    GC_THINNING_REQUESTED = "cima.gc.thinning.requested.v1"
    GC_RECONCILE_REQUESTED = "cima.gc.reconcile.requested.v1"
    GC_EPHEMERAL_EXPIRY_REQUESTED = "cima.gc.ephemeral_expiry.requested.v1"


class Producer(StrEnum):
    CIMA_API = "/cima/api"
    CIMA_WORKER = "/cima/worker"
    CIMA_GEOMETRY = "/cima/geometry"


class TraceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    correlation_id: str
    causation_id: str | None = None
    actor_kind: Literal["user", "system"] | None = None


class CloudEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specversion: Literal["1.0"] = "1.0"
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    source: Producer
    subject: str
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    datacontenttype: Literal["application/json"] = "application/json"
    dataschema: str
    data: dict[str, Any]


class SourceRegisteredData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: UUID
    kind: Literal["chat_user", "chat_assistant", "file_text", "feedback", "system"]
    external_provider: str | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def _canonicalize_kind(cls, value: Any) -> Any:
        aliases = {
            "file": "file_text",
            "dataset_document": "file_text",
            "document": "file_text",
            "doc": "file_text",
            "text": "chat_user",
            "chat": "chat_user",
            "assistant": "chat_assistant",
            "user": "chat_user",
        }
        if isinstance(value, str):
            return aliases.get(value, value)
        return value

    external_conversation_id: str | None = None
    external_message_id: str | None = None
    revision_no: int = 0
    displayable: bool = True
    processable: bool = True


class FileUploadedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: UUID
    filename: str
    mime_type: str | None = None
    sha256: str | None = None
    size_bytes: int = 0


class ChunkCreatedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[UUID]
    chunker_version: int
    normalizer_version: int
    origin_kind: Literal["chat", "file_text", "summary"]


class EduSegmentedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[UUID]
    edu_ids: list[UUID]
    edu_segmenter_version: int
    normalizer_version: int


class CItemCreatedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citem_ids: list[UUID]
    citem_builder_version: int
    normalizer_version: int


class SummaryChangedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_id: UUID
    level: Literal["EPOCH", "CLUSTER", "MASTER"]
    cluster_id: str | None = None
    epoch_no: int | None = None


class CItemPromotedGlobalData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_citem_id: UUID
    global_citem_id: UUID
    semantic_identity_id: UUID
    origin_conversation_id: str


class VectorMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["local", "global"]
    type: str | None = None


class VectorUpsertedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_kind: Literal["local_citem", "local_summary", "global_citem", "global_summary", "chunk", "ephemeral"]
    ref_id: UUID
    qdrant_collection: str
    vector_state: Literal["INDEXED", "EPHEMERAL"]
    embedding_model_id: str
    embedding_schema_version: int
    eligible_for_geometry: bool
    meta: VectorMeta


class VectorDeletedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_kind: Literal["local_citem", "local_summary", "global_citem", "global_summary", "chunk", "ephemeral"]
    ref_id: UUID
    qdrant_collection: str
    reason: Literal["THINNING", "HARD_DELETE", "EXPIRED", "ORPHAN_CLEANUP", "RECONCILE"]
    meta: VectorMeta


class ConversationHardDeleteRequestedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delete_run_id: UUID
    mode: Literal["HARD"] = "HARD"
    reason: Literal["USER_REQUEST", "RETENTION_POLICY"]


class ConversationHardDeleteCompletedStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    postgres_rows_deleted: int
    qdrant_points_deleted: int
    blob_bytes_deleted: int
    ephemeral_records_purged: int = 0
    geometry_purge_requested: int = 0
    qdrant_points_deleted_by_collection: dict[str, int] = Field(default_factory=dict)


class ConversationHardDeleteCompletedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delete_run_id: UUID
    stats: ConversationHardDeleteCompletedStats


class PinChangedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_kind: Literal["local_citem", "local_summary", "global_citem", "global_summary", "file", "chunk"]
    ref_id: str
    note: str | None = None


class GeometryOverrideParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temp: float | None = None
    core_q: float | None = None
    k_max: int | None = None
    bridge_percentile: int | None = None


class GeometryRecomputeData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd_id: UUID = Field(default_factory=uuid4)
    reason: Literal["DELTA_THRESHOLD", "EPOCH_CLOSED", "MANUAL", "RECOVERY"]
    override_params: GeometryOverrideParams | None = None


class GeometryPurgeData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd_id: UUID = Field(default_factory=uuid4)
    delete_run_id: UUID


class GeometrySetParamsData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd_id: UUID = Field(default_factory=uuid4)
    params: GeometryOverrideParams


class GeometryRunMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_vectors: int
    core_size: int
    bridge_count: int
    core_mass_frac: float
    stability_core_jaccard: float | None = None


class GeometryRunCompletedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    algo_version: str
    universe_hash: str
    params: dict[str, Any]
    metrics: GeometryRunMetrics


class GeometryItemStateData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    algo_version: str
    ref_kind: Literal["local_citem", "local_summary"]
    ref_id: UUID
    cluster_top1: str
    cluster_top2: str | None = None
    w1: float
    w2: float | None = None
    margin: float
    is_core: bool
    is_bridge_candidate: bool
    centrality: float | None = None
    updated_at: datetime


class GeometryClusterMedoid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_kind: Literal["local_citem", "local_summary"]
    ref_id: UUID


class GeometryClusterStateData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    algo_version: str
    cluster_id: str
    mass: float
    medoid: GeometryClusterMedoid
    summary_id: UUID | None = None
    updated_at: datetime


class SummaryRequestedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd_id: UUID = Field(default_factory=uuid4)
    level: Literal["EPOCH", "CLUSTER", "MASTER"]
    cluster_id: str | None = None
    epoch_no: int | None = None
    reason: Literal["EPOCH_CLOSED", "GEOM_CLUSTER_CHANGED", "PERIODIC", "MANUAL"]
    priority: Literal["NORMAL", "HIGH"] = "NORMAL"
    target_citem_ids: list[UUID] | None = None
    target_summary_ids: list[UUID] | None = None


class HandoffCreatedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_id: UUID
    schema_version: int
    checksum: str


class HandoffValidatedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_id: UUID
    manifest_id: UUID
    is_valid: bool


class HandoffRestoredData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    restore_run_id: UUID
    manifest_id: UUID
    target_conversation_id: str


class GcRequestedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maintenance_run_id: UUID
    reason: Literal["THINNING", "RECONCILE", "EPHEMERAL_EXPIRY"]


# Backward-compatible alias used by the detached geometry worker.
EventEnvelope = CloudEventEnvelope
