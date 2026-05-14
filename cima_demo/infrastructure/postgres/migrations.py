"""PostgreSQL DDL — run once on first deployment (KIMA_Infrastructure_Layer_v0.6 §1)."""
from __future__ import annotations

DDL = """

CREATE SCHEMA IF NOT EXISTS cima;
CREATE SCHEMA IF NOT EXISTS cima_rm;
CREATE SCHEMA IF NOT EXISTS geom;


-- ── witness-backend async plane foundation ─────────────────────────────────
CREATE TABLE IF NOT EXISTS cima.outbox (
    outbox_id        BIGSERIAL   PRIMARY KEY,
    topic            TEXT        NOT NULL,
    message_key      TEXT        NOT NULL,
    headers_json     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    payload_json     JSONB,
    status           TEXT        NOT NULL DEFAULT 'NEW'
                    CHECK (status IN ('NEW','CLAIMED','SENT','ERROR')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at       TIMESTAMPTZ,
    sent_at          TIMESTAMPTZ,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS cima_outbox_status_created_idx
    ON cima.outbox (status, created_at);
ALTER TABLE cima.outbox ALTER COLUMN payload_json DROP NOT NULL;

CREATE TABLE IF NOT EXISTS cima.consumer_effect (
    consumer_name    TEXT        NOT NULL,
    event_id         TEXT        NOT NULL,
    effect_key       TEXT        NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'STARTED'
                    CHECK (status IN ('STARTED','SUCCEEDED','FAILED','CANCELLED')),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    details_json     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (consumer_name, event_id, effect_key)
);

CREATE INDEX IF NOT EXISTS cima_consumer_effect_status_idx
    ON cima.consumer_effect (status, started_at DESC);
-- ── conversations ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID        PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          TEXT        NOT NULL DEFAULT 'ACTIVE'
                    CHECK (status IN ('ACTIVE','DELETING','DELETED')),
    delete_run_id   UUID,
    delete_requested_at TIMESTAMPTZ,
    delete_completed_at TIMESTAMPTZ
);

ALTER TABLE conversations ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ACTIVE';
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS delete_run_id UUID;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS delete_requested_at TIMESTAMPTZ;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS delete_completed_at TIMESTAMPTZ;

-- ── task_memory ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS task_memory (
    conversation_id     UUID        PRIMARY KEY
                        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_count          INT         NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    phase               TEXT        NOT NULL DEFAULT 'IDLE',
    active_plan_id      UUID,
    awaiting_user_input BOOLEAN     NOT NULL DEFAULT FALSE,
    awaiting_question   TEXT,
    turn_in_progress    BOOLEAN     NOT NULL DEFAULT FALSE,
    stall_count         INT         NOT NULL DEFAULT 0 CHECK (stall_count >= 0),
    last_turn_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_memory_turn_in_progress
    ON task_memory (conversation_id) WHERE turn_in_progress = FALSE;


-- ── hard delete lifecycle (async plane foundation) ───────────────────────
CREATE TABLE IF NOT EXISTS cima.delete_run (
    delete_run_id    UUID        PRIMARY KEY,
    conversation_id  UUID        NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'REQUESTED'
                    CHECK (status IN ('REQUESTED','RUNNING','SUCCEEDED','FAILED')),
    requested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    stats_json       JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cima_delete_run_conversation
    ON cima.delete_run (conversation_id, requested_at DESC);

CREATE TABLE IF NOT EXISTS cima.maintenance_run (
    maintenance_run_id UUID PRIMARY KEY,
    conversation_id    UUID REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    kind               TEXT NOT NULL CHECK (kind IN ('THINNING','RECONCILE','EPHEMERAL_EXPIRY','ORPHAN_CLEANUP')),
    status             TEXT NOT NULL CHECK (status IN ('REQUESTED','RUNNING','SUCCEEDED','FAILED')),
    requested_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at       TIMESTAMPTZ,
    stats_json         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cima_maintenance_run_requested
    ON cima.maintenance_run (kind, status, requested_at DESC);

CREATE TABLE IF NOT EXISTS cima.ephemeral_vector (
    ephemeral_id         UUID PRIMARY KEY,
    conversation_id      UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    origin_ref_kind      TEXT        NOT NULL CHECK (origin_ref_kind IN ('local_citem','local_summary','global_citem','global_summary','chunk')),
    origin_ref_id        UUID,
    qdrant_collection    TEXT        NOT NULL,
    lifecycle_state      TEXT        NOT NULL DEFAULT 'ACTIVE'
                       CHECK (lifecycle_state IN ('ACTIVE','EXPIRED','PURGED')),
    vector_state         TEXT        NOT NULL DEFAULT 'EPHEMERAL'
                       CHECK (vector_state = 'EPHEMERAL'),
    embedding_model_id   TEXT,
    embedding_schema_version INT,
    eligible_for_geometry BOOLEAN    NOT NULL DEFAULT FALSE,
    meta_json            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at           TIMESTAMPTZ NOT NULL,
    expired_at           TIMESTAMPTZ,
    purged_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cima_ephemeral_vector_due
    ON cima.ephemeral_vector (lifecycle_state, expires_at);
CREATE INDEX IF NOT EXISTS idx_cima_ephemeral_vector_conversation
    ON cima.ephemeral_vector (conversation_id, lifecycle_state, expires_at);

-- ── conversation_turns ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID        NOT NULL
                    REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    sequence        INT         NOT NULL,
    user_message    TEXT        NOT NULL,
    assistant_reply TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (conversation_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation_seq
    ON conversation_turns (conversation_id, sequence DESC);

-- ── summary_nodes ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS summary_nodes (
    node_id         UUID        PRIMARY KEY,
    conversation_id UUID        NOT NULL,
    level           INT         NOT NULL CHECK (level >= 1),
    text            TEXT        NOT NULL,
    origin_ids      UUID[]      NOT NULL DEFAULT '{}',
    parent_ids      JSONB       NOT NULL DEFAULT '[]',
    tags            TEXT[]      NOT NULL DEFAULT '{}',
    confidence      FLOAT4      NOT NULL DEFAULT 1.0
                    CHECK (confidence BETWEEN 0.0 AND 1.0),
    token_count     INT         NOT NULL DEFAULT 0 CHECK (token_count >= 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_conversation_level
    ON summary_nodes (conversation_id, level);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_updated_at
    ON summary_nodes (updated_at DESC);

-- ── task_metadata ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS task_metadata (
    conversation_id UUID        PRIMARY KEY,
    data            JSONB       NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── plans ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plans (
    plan_id         UUID        PRIMARY KEY,
    conversation_id UUID        NOT NULL,
    goal            TEXT        NOT NULL,
    current_seq     INT         NOT NULL DEFAULT 0,
    status          TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','running','paused','completed','failed','replanned')),
    auto_continue   BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── plan_steps ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plan_steps (
    step_id              UUID        PRIMARY KEY,
    plan_id              UUID        NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    sequence             INT         NOT NULL,
    description          TEXT        NOT NULL,
    tool_name            TEXT,
    tool_params          JSONB       NOT NULL DEFAULT '{}',
    status               TEXT        NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending','active','completed','failed','skipped')),
    result_summary       TEXT,
    procedure_citem_id   UUID,
    attempts             INT         NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    UNIQUE (plan_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_plan_steps_plan
    ON plan_steps (plan_id, sequence);

-- ── conflict_log ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conflict_log (
    entry_id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID        NOT NULL,
    citem_a_id          UUID        NOT NULL,
    citem_b_id          UUID        NOT NULL,
    conflict_type       TEXT        NOT NULL,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolution_citem_id UUID,
    resolved_at         TIMESTAMPTZ,
    resolver_actor      TEXT        CHECK (resolver_actor IN ('agent','user')),
    notes               TEXT        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_conflict_log_conversation
    ON conflict_log (conversation_id, detected_at DESC);

-- ── citem_audit ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS citem_audit (
    audit_id        BIGSERIAL   PRIMARY KEY,
    citem_id        UUID        NOT NULL,
    conversation_id UUID        NOT NULL,
    event_type      TEXT        NOT NULL
                    CHECK (event_type IN (
                        'CREATED','ARCHIVED','RESTORED','PROMOTED',
                        'DEMOTED','PURGED','CONFLICT_FLAGGED','CONFLICT_RESOLVED'
                    )),
    old_value       TEXT,
    new_value       TEXT,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_citem
    ON citem_audit (citem_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event
    ON citem_audit (event_type, occurred_at DESC);

-- ── chm_refs ──────────────────────────────────────────────────────────────
-- Tracks which C-Items are referenced by active Contextual History Markers (CHMs).
CREATE TABLE IF NOT EXISTS chm_refs (
    id              BIGSERIAL   PRIMARY KEY,
    conversation_id UUID        NOT NULL,
    citem_id        UUID        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (conversation_id, citem_id)
);

CREATE INDEX IF NOT EXISTS idx_chm_refs_conversation
    ON chm_refs (conversation_id);

-- Idempotent column addition: reference_count tracks how many turns referenced each C-Item.
-- PostgreSQL supports ADD COLUMN IF NOT EXISTS since 9.0.
ALTER TABLE chm_refs ADD COLUMN IF NOT EXISTS reference_count INT NOT NULL DEFAULT 1;

-- ── demonstrator run journal ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS demo_runs (
    run_id            UUID        PRIMARY KEY,
    conversation_id   UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_id           UUID        NOT NULL,
    status            TEXT        NOT NULL
                     CHECK (status IN ('running','completed','failed','cancelled','blocked','context_only')),
    user_message      TEXT        NOT NULL,
    cognitive_phase   TEXT,
    execution_mode    TEXT,
    active_plan_id    UUID,
    assistant_reply   TEXT        NOT NULL DEFAULT '',
    error_class       TEXT,
    manifest          JSONB       NOT NULL DEFAULT '{}',
    checkpoint_count  INT         NOT NULL DEFAULT 0 CHECK (checkpoint_count >= 0),
    phase_count       INT         NOT NULL DEFAULT 0 CHECK (phase_count >= 0),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_demo_runs_conversation_created
    ON demo_runs (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_run_phases (
    phase_id          BIGSERIAL   PRIMARY KEY,
    run_id            UUID        NOT NULL REFERENCES demo_runs(run_id) ON DELETE CASCADE,
    sequence          INT         NOT NULL CHECK (sequence >= 1),
    phase_name        TEXT        NOT NULL,
    payload           JSONB       NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_demo_run_phases_run_seq
    ON demo_run_phases (run_id, sequence);

CREATE TABLE IF NOT EXISTS demo_checkpoints (
    checkpoint_id     UUID        PRIMARY KEY,
    run_id            UUID        NOT NULL REFERENCES demo_runs(run_id) ON DELETE CASCADE,
    sequence          INT         NOT NULL CHECK (sequence >= 1),
    checkpoint_kind   TEXT        NOT NULL,
    state             JSONB       NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_demo_checkpoints_run_seq
    ON demo_checkpoints (run_id, sequence);

-- ── demonstrator lineage / context artifacts ─────────────────────────────
CREATE TABLE IF NOT EXISTS demo_sources (
    source_id          UUID        PRIMARY KEY,
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    source_kind        TEXT        NOT NULL,
    role               TEXT,
    origin_ref         TEXT,
    display_text       TEXT,
    process_text       TEXT,
    metadata           JSONB       NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_sources_conversation_created
    ON demo_sources (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_source_spans (
    span_id            UUID        PRIMARY KEY,
    source_id          UUID        NOT NULL REFERENCES demo_sources(source_id) ON DELETE CASCADE,
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    span_kind          TEXT        NOT NULL,
    char_start         INT         NOT NULL DEFAULT 0,
    char_end           INT         NOT NULL DEFAULT 0,
    locator            JSONB       NOT NULL DEFAULT '{}',
    preview_text       TEXT        NOT NULL DEFAULT '',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_source_spans_source
    ON demo_source_spans (source_id, char_start, char_end);

CREATE TABLE IF NOT EXISTS demo_lineage_edges (
    edge_id            UUID        PRIMARY KEY,
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    src_kind           TEXT        NOT NULL,
    src_id             TEXT        NOT NULL,
    dst_kind           TEXT        NOT NULL,
    dst_id             TEXT        NOT NULL,
    relation           TEXT        NOT NULL,
    metadata           JSONB       NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_lineage_edges_src
    ON demo_lineage_edges (conversation_id, src_kind, src_id);
CREATE INDEX IF NOT EXISTS idx_demo_lineage_edges_dst
    ON demo_lineage_edges (conversation_id, dst_kind, dst_id);

CREATE TABLE IF NOT EXISTS demo_summary_resolutions (
    summary_id         UUID        PRIMARY KEY,
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    summary_text       TEXT        NOT NULL,
    origin_citem_ids   UUID[]      NOT NULL DEFAULT '{}',
    metadata           JSONB       NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_summary_resolutions_conversation
    ON demo_summary_resolutions (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_context_snapshots (
    context_id             UUID        PRIMARY KEY,
    run_id                 UUID        NOT NULL REFERENCES demo_runs(run_id) ON DELETE CASCADE,
    conversation_id        UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_id                UUID        NOT NULL,
    query_text             TEXT        NOT NULL,
    phase                  TEXT,
    context_text           TEXT        NOT NULL,
    markers                JSONB       NOT NULL DEFAULT '[]',
    items                  JSONB       NOT NULL DEFAULT '[]',
    budget                 JSONB       NOT NULL DEFAULT '{}',
    resolved_source_ids    JSONB       NOT NULL DEFAULT '[]',
    resolved_span_ids      JSONB       NOT NULL DEFAULT '[]',
    resolved_source_count  INT         NOT NULL DEFAULT 0,
    resolved_span_count    INT         NOT NULL DEFAULT 0,
    unresolved_ref_ids     JSONB       NOT NULL DEFAULT '[]',
    marker_resolution      JSONB       NOT NULL DEFAULT '[]',
    resolution_mode        TEXT        NOT NULL DEFAULT 'empty',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_context_snapshots_run_created
    ON demo_context_snapshots (run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_answer_lineage (
    answer_lineage_id      UUID        PRIMARY KEY,
    conversation_id        UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    run_id                 UUID        NOT NULL REFERENCES demo_runs(run_id) ON DELETE CASCADE,
    response_turn_id       UUID,
    context_id             UUID REFERENCES demo_context_snapshots(context_id) ON DELETE SET NULL,
    answer_text            TEXT        NOT NULL,
    cited_markers          JSONB       NOT NULL DEFAULT '[]',
    lineage                JSONB       NOT NULL DEFAULT '[]',
    resolved_source_ids    JSONB       NOT NULL DEFAULT '[]',
    resolved_span_ids      JSONB       NOT NULL DEFAULT '[]',
    resolved_source_count  INT         NOT NULL DEFAULT 0,
    resolved_span_count    INT         NOT NULL DEFAULT 0,
    unresolved_ref_ids     JSONB       NOT NULL DEFAULT '[]',
    marker_resolution      JSONB       NOT NULL DEFAULT '[]',
    resolution_mode        TEXT        NOT NULL DEFAULT 'empty',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS resolved_source_ids JSONB NOT NULL DEFAULT '[]';
ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS resolved_span_ids JSONB NOT NULL DEFAULT '[]';
ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS resolved_source_count INT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS resolved_span_count INT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS unresolved_ref_ids JSONB NOT NULL DEFAULT '[]';
ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS resolution_mode TEXT NOT NULL DEFAULT 'empty';
ALTER TABLE IF EXISTS demo_context_snapshots ADD COLUMN IF NOT EXISTS marker_resolution JSONB NOT NULL DEFAULT '[]';

ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS resolved_source_ids JSONB NOT NULL DEFAULT '[]';
ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS resolved_span_ids JSONB NOT NULL DEFAULT '[]';
ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS resolved_source_count INT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS resolved_span_count INT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS resolution_mode TEXT NOT NULL DEFAULT 'empty';
ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS unresolved_ref_ids JSONB NOT NULL DEFAULT '[]';
ALTER TABLE IF EXISTS demo_answer_lineage ADD COLUMN IF NOT EXISTS marker_resolution JSONB NOT NULL DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_demo_answer_lineage_run
    ON demo_answer_lineage (run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_handoff_manifests (
    handoff_id         UUID        PRIMARY KEY,
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    source_run_id      UUID        NOT NULL REFERENCES demo_runs(run_id) ON DELETE CASCADE,
    context_id         UUID REFERENCES demo_context_snapshots(context_id) ON DELETE SET NULL,
    checksum           TEXT        NOT NULL,
    manifest           JSONB       NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_handoff_manifests_conversation_created
    ON demo_handoff_manifests (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_handoff_validations (
    handoff_id         UUID        PRIMARY KEY REFERENCES demo_handoff_manifests(handoff_id) ON DELETE CASCADE,
    valid              BOOLEAN     NOT NULL DEFAULT FALSE,
    issues             JSONB       NOT NULL DEFAULT '[]',
    evidence_coverage  FLOAT4      NOT NULL DEFAULT 0.0,
    validation         JSONB       NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS demo_handoff_restores (
    restore_id             UUID        PRIMARY KEY,
    handoff_id             UUID        NOT NULL REFERENCES demo_handoff_manifests(handoff_id) ON DELETE CASCADE,
    target_conversation_id UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    target_run_id          UUID REFERENCES demo_runs(run_id) ON DELETE SET NULL,
    valid                  BOOLEAN     NOT NULL DEFAULT FALSE,
    reconstructed_task_state JSONB     NOT NULL DEFAULT '{}',
    diff                   JSONB       NOT NULL DEFAULT '{}',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_handoff_restores_target
    ON demo_handoff_restores (target_conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS demo_gc_audits (
    audit_id            UUID        PRIMARY KEY,
    conversation_id     UUID        NOT NULL,
    run_id              UUID,
    action              TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'ok',
    phase               TEXT,
    before_counts       JSONB       NOT NULL DEFAULT '{}',
    after_counts        JSONB       NOT NULL DEFAULT '{}',
    metrics             JSONB       NOT NULL DEFAULT '{}',
    consistency         JSONB       NOT NULL DEFAULT '{}',
    notes               JSONB       NOT NULL DEFAULT '[]',
    error_class         TEXT,
    audit               JSONB       NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_gc_audits_conversation_created
    ON demo_gc_audits (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS geom.runs (
    run_id            UUID        PRIMARY KEY,
    conversation_id   UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    reason            TEXT        NOT NULL DEFAULT 'manual',
    algo_version      TEXT        NOT NULL DEFAULT 'geom_v1',
    n_items           INT         NOT NULL DEFAULT 0,
    cluster_count     INT         NOT NULL DEFAULT 0,
    core_count        INT         NOT NULL DEFAULT 0,
    bridge_count      INT         NOT NULL DEFAULT 0,
    metrics           JSONB       NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_geom_runs_conversation_created
    ON geom.runs (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS geom.item_state (
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    ref_kind           TEXT        NOT NULL,
    ref_id             UUID        NOT NULL,
    run_id             UUID        NOT NULL REFERENCES geom.runs(run_id) ON DELETE CASCADE,
    cluster_top1       TEXT        NOT NULL,
    cluster_top2       TEXT,
    w1                 FLOAT4      NOT NULL DEFAULT 1.0,
    w2                 FLOAT4,
    margin             FLOAT4      NOT NULL DEFAULT 1.0,
    is_core            BOOLEAN     NOT NULL DEFAULT FALSE,
    is_bridge_candidate BOOLEAN    NOT NULL DEFAULT FALSE,
    centrality         FLOAT4      NOT NULL DEFAULT 0.0,
    label              TEXT,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, ref_kind, ref_id)
);

CREATE INDEX IF NOT EXISTS idx_geom_item_state_conversation_cluster
    ON geom.item_state (conversation_id, cluster_top1);

CREATE TABLE IF NOT EXISTS geom.cluster_state (
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    cluster_id         TEXT        NOT NULL,
    run_id             UUID        NOT NULL REFERENCES geom.runs(run_id) ON DELETE CASCADE,
    mass               FLOAT4      NOT NULL DEFAULT 0.0,
    medoid_ref_id      UUID        NOT NULL,
    summary_id         UUID,
    label              TEXT,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_geom_cluster_state_conversation
    ON geom.cluster_state (conversation_id, updated_at DESC);

-- ── read-model geometry state consumed by CIMA runtime ─────────────────────
CREATE TABLE IF NOT EXISTS cima_rm.geom_run (
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    run_id             UUID        NOT NULL,
    algo_version       TEXT        NOT NULL,
    universe_hash      TEXT        NOT NULL,
    k_used             INT         NOT NULL DEFAULT 0,
    temp               FLOAT4      NOT NULL DEFAULT 0.0,
    core_q             FLOAT4      NOT NULL DEFAULT 0.0,
    bridge_percentile  INT         NOT NULL DEFAULT 0,
    metrics_json       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    completed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_cima_rm_geom_run_conversation
    ON cima_rm.geom_run (conversation_id, completed_at DESC);

CREATE TABLE IF NOT EXISTS cima_rm.geom_item_state (
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    ref_kind           TEXT        NOT NULL CHECK (ref_kind IN ('local_citem','local_summary')),
    ref_id             UUID        NOT NULL,
    run_id             UUID        NOT NULL,
    cluster_top1       TEXT        NOT NULL,
    cluster_top2       TEXT,
    w1                 FLOAT4      NOT NULL DEFAULT 1.0,
    w2                 FLOAT4,
    margin             FLOAT4      NOT NULL DEFAULT 1.0,
    is_core            BOOLEAN     NOT NULL DEFAULT FALSE,
    is_bridge_candidate BOOLEAN    NOT NULL DEFAULT FALSE,
    centrality         FLOAT4,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, ref_kind, ref_id)
);

CREATE INDEX IF NOT EXISTS idx_cima_rm_geom_item_state_conversation_cluster
    ON cima_rm.geom_item_state (conversation_id, cluster_top1);

CREATE TABLE IF NOT EXISTS cima_rm.geom_cluster_state (
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    cluster_id         TEXT        NOT NULL,
    run_id             UUID        NOT NULL,
    mass               FLOAT4      NOT NULL DEFAULT 0.0,
    medoid_ref_kind    TEXT        NOT NULL CHECK (medoid_ref_kind IN ('local_citem','local_summary')),
    medoid_ref_id      UUID        NOT NULL,
    summary_id         UUID,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_cima_rm_geom_cluster_state_conversation
    ON cima_rm.geom_cluster_state (conversation_id, updated_at DESC);

-- ── geometry service outbox (hard-boundary publisher) ──────────────────────
CREATE TABLE IF NOT EXISTS geom.outbox (
    outbox_id        BIGSERIAL   PRIMARY KEY,
    topic            TEXT        NOT NULL,
    message_key      TEXT        NOT NULL,
    headers_json     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    payload_json     JSONB,
    status           TEXT        NOT NULL DEFAULT 'NEW'
                    CHECK (status IN ('NEW','CLAIMED','SENT','ERROR')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at       TIMESTAMPTZ,
    sent_at          TIMESTAMPTZ,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS geom_outbox_status_created_idx
    ON geom.outbox (status, created_at);

ALTER TABLE geom.outbox ALTER COLUMN payload_json DROP NOT NULL;

-- ── retrieval_telemetry ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrieval_telemetry (
    id                        BIGSERIAL   PRIMARY KEY,
    conversation_id           UUID        NOT NULL,
    turn_id                   UUID        NOT NULL,
    query_type                TEXT        NOT NULL,
    recall_top_k              INT         NOT NULL,
    rerank_top_n              INT         NOT NULL,
    geometric_expand          BOOLEAN     NOT NULL DEFAULT FALSE,
    candidates_before_rerank  INT         NOT NULL DEFAULT 0,
    candidates_after_rerank   INT         NOT NULL DEFAULT 0,
    candidates_after_expand   INT         NOT NULL DEFAULT 0,
    pack_total_tokens         INT         NOT NULL DEFAULT 0,
    coverage_score            FLOAT4      NOT NULL CHECK (coverage_score BETWEEN 0.0 AND 1.0),
    retry_count               INT         NOT NULL DEFAULT 0,
    reranker_available        BOOLEAN     NOT NULL DEFAULT TRUE,
    latency_ms                INT         NOT NULL,
    traceability_density      FLOAT4      NOT NULL DEFAULT 1.0 CHECK (traceability_density BETWEEN 0.0 AND 1.0),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_conversation
    ON retrieval_telemetry (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_query_type
    ON retrieval_telemetry (query_type, created_at DESC);

-- Migration: add traceability_density to existing deployments
ALTER TABLE retrieval_telemetry
    ADD COLUMN IF NOT EXISTS traceability_density FLOAT4 NOT NULL DEFAULT 1.0
    CHECK (traceability_density BETWEEN 0.0 AND 1.0);

-- Migration: add bridge/strategy telemetry columns (Retrieval Instrumentation D)
ALTER TABLE retrieval_telemetry ADD COLUMN IF NOT EXISTS q3_relevant_count     INT      NOT NULL DEFAULT 0;
ALTER TABLE retrieval_telemetry ADD COLUMN IF NOT EXISTS bridge_enabled         BOOLEAN  NOT NULL DEFAULT FALSE;
ALTER TABLE retrieval_telemetry ADD COLUMN IF NOT EXISTS bridge_alpha           FLOAT4   NOT NULL DEFAULT 0.5;
ALTER TABLE retrieval_telemetry ADD COLUMN IF NOT EXISTS bridge_floor           FLOAT4   NOT NULL DEFAULT 0.0;
ALTER TABLE retrieval_telemetry ADD COLUMN IF NOT EXISTS bridge_candidates      INT      NOT NULL DEFAULT 0;
ALTER TABLE retrieval_telemetry ADD COLUMN IF NOT EXISTS direct_strategy        TEXT;

-- DEBT-01: auto_continue for PlanExecutorWorker
ALTER TABLE plans ADD COLUMN IF NOT EXISTS auto_continue BOOLEAN NOT NULL DEFAULT FALSE;

-- ── file_registry ──────────────────────────────────────────────────────────
-- Tracks every file ingested per conversation: status lifecycle, chunk count,
-- and the list of Qdrant C-Item IDs produced.  Enables list_files tool and
-- cross-session deduplication audit.
CREATE TABLE IF NOT EXISTS file_registry (
    file_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID        NOT NULL
                    REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    filename        TEXT        NOT NULL,
    mime_type       TEXT        NOT NULL DEFAULT 'application/octet-stream',
    size_bytes      INT         NOT NULL DEFAULT 0,
    content_hash    TEXT        NOT NULL DEFAULT '',
    status          TEXT        NOT NULL DEFAULT 'QUEUED'
                    CHECK (status IN ('QUEUED','PROCESSING','READY','FAILED')),
    chunk_count     INT         NOT NULL DEFAULT 0,
    citem_ids       UUID[]      NOT NULL DEFAULT '{}',
    blob_path       TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_file_registry_conversation
    ON file_registry (conversation_id, ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_file_registry_content_hash
    ON file_registry (content_hash);

-- ── witness chunk manifests (async source->chunk plane) ───────────────────
CREATE TABLE IF NOT EXISTS cima.chunk_manifest (
    chunk_id           UUID        PRIMARY KEY,
    conversation_id    UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    source_id          UUID        NOT NULL REFERENCES demo_sources(source_id) ON DELETE CASCADE,
    file_id            UUID,
    source_span_id     UUID        REFERENCES demo_source_spans(span_id) ON DELETE CASCADE,
    chunk_kind         TEXT        NOT NULL CHECK (chunk_kind IN ('doc_chunk','chat_chunk','summary_chunk')),
    chunk_index        INT         NOT NULL CHECK (chunk_index >= 0),
    page_num           INT,
    section_hint       TEXT,
    normalizer_version INT         NOT NULL DEFAULT 1,
    chunker_version    INT         NOT NULL DEFAULT 1,
    vector_state       TEXT        NOT NULL DEFAULT 'NONE'
                      CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
    embedding_model_id TEXT,
    embedding_schema_version INT,
    expires_at         TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_id, chunk_index, chunker_version)
);

CREATE INDEX IF NOT EXISTS idx_cima_chunk_manifest_conversation
    ON cima.chunk_manifest (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cima_chunk_manifest_source
    ON cima.chunk_manifest (source_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_cima_chunk_manifest_file
    ON cima.chunk_manifest (file_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_cima_chunk_manifest_vector_state
    ON cima.chunk_manifest (conversation_id, vector_state);


-- ── witness EDU manifests (async semantic plane) ──────────────────────────
CREATE TABLE IF NOT EXISTS cima.edu_manifest (
    edu_id              UUID        PRIMARY KEY,
    conversation_id     UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    source_id           UUID        NOT NULL REFERENCES demo_sources(source_id) ON DELETE CASCADE,
    chunk_id            UUID        NOT NULL REFERENCES cima.chunk_manifest(chunk_id) ON DELETE CASCADE,
    edu_kind            TEXT        NOT NULL,
    span_refs_json      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    features_json       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    quality             FLOAT4      NOT NULL DEFAULT 1.0,
    normalizer_version  INT         NOT NULL DEFAULT 1,
    edu_segmenter_version INT       NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cima_edu_manifest_conversation
    ON cima.edu_manifest (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cima_edu_manifest_chunk
    ON cima.edu_manifest (chunk_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cima_edu_manifest_source
    ON cima.edu_manifest (source_id, created_at);

-- ── witness local semantic memory (async semantic plane) ──────────────────
CREATE TABLE IF NOT EXISTS cima.local_citem (
    local_citem_id         UUID        PRIMARY KEY,
    semantic_identity_id   UUID        NOT NULL,
    conversation_id        UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    type                   TEXT        NOT NULL,
    text                   TEXT        NOT NULL,
    embedding_text         TEXT        NOT NULL,
    meta_json              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    provenance_json        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    validity               TEXT        NOT NULL DEFAULT 'unknown'
                       CHECK (validity IN ('unknown','accepted','rejected')),
    salience               FLOAT4      NOT NULL DEFAULT 0.0,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    vector_state           TEXT        NOT NULL DEFAULT 'NONE'
                       CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
    embedding_model_id     TEXT,
    embedding_schema_version INT,
    expires_at             TIMESTAMPTZ,
    is_pinned              BOOLEAN     NOT NULL DEFAULT FALSE,
    was_cited              BOOLEAN     NOT NULL DEFAULT FALSE,
    last_used_at           TIMESTAMPTZ,
    normalizer_version     INT         NOT NULL DEFAULT 1,
    citem_builder_version  INT         NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_cima_local_citem_conversation
    ON cima.local_citem (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cima_local_citem_type
    ON cima.local_citem (conversation_id, type);
CREATE INDEX IF NOT EXISTS idx_cima_local_citem_vector_state
    ON cima.local_citem (conversation_id, vector_state);
CREATE INDEX IF NOT EXISTS idx_cima_local_citem_semantic_identity
    ON cima.local_citem (semantic_identity_id);

CREATE TABLE IF NOT EXISTS cima.local_citem_evidence (
    local_citem_id         UUID        NOT NULL REFERENCES cima.local_citem(local_citem_id) ON DELETE CASCADE,
    source_id              UUID        REFERENCES demo_sources(source_id) ON DELETE CASCADE,
    chunk_id               UUID        REFERENCES cima.chunk_manifest(chunk_id) ON DELETE CASCADE,
    edu_id                 UUID        REFERENCES cima.edu_manifest(edu_id) ON DELETE CASCADE,
    ordinal                INT         NOT NULL DEFAULT 0,
    locator_json           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (local_citem_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_cima_local_citem_evidence_chunk
    ON cima.local_citem_evidence (chunk_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_cima_local_citem_evidence_edu
    ON cima.local_citem_evidence (edu_id, ordinal);

CREATE TABLE IF NOT EXISTS cima.local_summary (
    local_summary_id            UUID        PRIMARY KEY,
    conversation_id             UUID        NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    level                       TEXT        NOT NULL CHECK (level IN ('EPOCH','CLUSTER','MASTER')),
    cluster_id                  TEXT,
    epoch_no                    INT,
    text                        TEXT        NOT NULL,
    covers_json                 JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    vector_state                TEXT        NOT NULL DEFAULT 'NONE'
                          CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
    embedding_model_id          TEXT,
    embedding_schema_version    INT,
    is_pinned                   BOOLEAN     NOT NULL DEFAULT FALSE,
    was_cited                   BOOLEAN     NOT NULL DEFAULT FALSE,
    last_used_at                TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cima_local_summary_conversation
    ON cima.local_summary (conversation_id, level, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cima_local_summary_cluster
    ON cima.local_summary (conversation_id, cluster_id);

CREATE TABLE IF NOT EXISTS cima.local_summary_origin (
    local_summary_id            UUID        NOT NULL REFERENCES cima.local_summary(local_summary_id) ON DELETE CASCADE,
    origin_kind                 TEXT        NOT NULL CHECK (origin_kind IN ('local_citem','chunk','source')),
    origin_id                   UUID        NOT NULL,
    ordinal                     INT         NOT NULL DEFAULT 0,
    PRIMARY KEY (local_summary_id, origin_kind, origin_id)
);

CREATE INDEX IF NOT EXISTS idx_cima_local_summary_origin_summary
    ON cima.local_summary_origin (local_summary_id, ordinal);

CREATE TABLE IF NOT EXISTS cima.global_citem (
    global_citem_id             UUID        PRIMARY KEY,
    semantic_identity_id        UUID        NOT NULL,
    origin_conversation_id      UUID        NOT NULL,
    promotion_origin_local_citem_id UUID    NOT NULL REFERENCES cima.local_citem(local_citem_id),
    type                        TEXT        NOT NULL,
    text                        TEXT        NOT NULL,
    embedding_text              TEXT        NOT NULL,
    meta_json                   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    provenance_json             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    validity                    TEXT        NOT NULL DEFAULT 'unknown'
                          CHECK (validity IN ('unknown','accepted','rejected')),
    salience                    FLOAT4      NOT NULL DEFAULT 0.0,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    vector_state                TEXT        NOT NULL DEFAULT 'NONE'
                          CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
    embedding_model_id          TEXT,
    embedding_schema_version    INT,
    expires_at                  TIMESTAMPTZ,
    is_pinned                   BOOLEAN     NOT NULL DEFAULT FALSE,
    was_cited                   BOOLEAN     NOT NULL DEFAULT FALSE,
    last_used_at                TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cima_global_citem_semantic_identity
    ON cima.global_citem (semantic_identity_id);
CREATE INDEX IF NOT EXISTS idx_cima_global_citem_type
    ON cima.global_citem (type);
CREATE INDEX IF NOT EXISTS idx_cima_global_citem_vector_state
    ON cima.global_citem (vector_state);

CREATE TABLE IF NOT EXISTS cima.global_citem_evidence (
    global_citem_id             UUID        NOT NULL REFERENCES cima.global_citem(global_citem_id) ON DELETE CASCADE,
    ordinal                     INT         NOT NULL DEFAULT 0,
    evidence_kind               TEXT        NOT NULL CHECK (evidence_kind IN ('source_snippet','chunk_snippet','external_ref')),
    source_text_snapshot        TEXT,
    locator_json                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (global_citem_id, ordinal)
);

CREATE TABLE IF NOT EXISTS cima.global_summary (
    global_summary_id           UUID        PRIMARY KEY,
    level                       TEXT        NOT NULL CHECK (level IN ('MASTER','CLUSTER')),
    cluster_id                  TEXT,
    text                        TEXT        NOT NULL,
    covers_json                 JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    vector_state                TEXT        NOT NULL DEFAULT 'NONE'
                          CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
    embedding_model_id          TEXT,
    embedding_schema_version    INT
);

CREATE TABLE IF NOT EXISTS cima.global_summary_origin (
    global_summary_id           UUID        NOT NULL REFERENCES cima.global_summary(global_summary_id) ON DELETE CASCADE,
    origin_kind                 TEXT        NOT NULL CHECK (origin_kind IN ('global_citem','global_summary')),
    origin_id                   UUID        NOT NULL,
    ordinal                     INT         NOT NULL DEFAULT 0,
    PRIMARY KEY (global_summary_id, origin_kind, origin_id)
);

-- ── retrieval_genomes (RAG evolution) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrieval_genomes (
    genome_id       UUID        PRIMARY KEY,
    genes           JSONB       NOT NULL,
    score_medio     FLOAT4      NOT NULL DEFAULT 0.0,
    max_fitness     FLOAT4      NOT NULL DEFAULT 0.0,
    n_evaluations   INT         NOT NULL DEFAULT 0 CHECK (n_evaluations >= 0),
    ucb_score       FLOAT4      NOT NULL DEFAULT 0.0,
    temporal_weight FLOAT4      NOT NULL DEFAULT 1.0,
    generation      INT         NOT NULL DEFAULT 0,
    parent_ids      UUID[]      NOT NULL DEFAULT '{}',
    origin          TEXT        NOT NULL DEFAULT 'seed',
    is_seed         BOOLEAN     NOT NULL DEFAULT FALSE,
    last_evaluated  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrieval_genomes_ucb
    ON retrieval_genomes (ucb_score DESC);

CREATE TABLE IF NOT EXISTS retrieval_genome_counter (
    id      INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    n_turns INT NOT NULL DEFAULT 0
);
INSERT INTO retrieval_genome_counter (id, n_turns)
    VALUES (1, 0) ON CONFLICT DO NOTHING;
"""


async def run_migrations(pool: object) -> None:
    """Execute DDL against the pool. Idempotent (IF NOT EXISTS)."""
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        await conn.execute(DDL)
