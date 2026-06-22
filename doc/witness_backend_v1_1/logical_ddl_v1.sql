-- CIMA Witness Backend v1.1 — Logical DDL baseline
-- Implementation-facing logical schema. This is not a migration history.
-- Canonical decisions baked into this DDL:
--   * PostgreSQL is the source of truth.
--   * Local and global memory are physically separate.
--   * Transcript fidelity requires display_text (frontend witness) and content_text
--     (processable normalized text) to be modeled separately.
--   * Canonical FTS exists on source.content_text and *_citem.text, never on chunk.
--   * Generic job-queue tables are intentionally absent; the canonical async plane is
--     write + outbox + publisher + idempotent consumers.

CREATE SCHEMA IF NOT EXISTS cima;
CREATE SCHEMA IF NOT EXISTS cima_rm;
CREATE SCHEMA IF NOT EXISTS geom;
CREATE SCHEMA IF NOT EXISTS audit;

-- Conversations -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.conversation (
  conversation_id      text PRIMARY KEY,
  user_id              text NOT NULL,
  created_at           timestamptz NOT NULL DEFAULT now(),
  status               text NOT NULL CHECK (status IN ('ACTIVE','DELETING','DELETED')),
  delete_run_id        uuid,
  delete_requested_at  timestamptz,
  delete_completed_at  timestamptz,
  settings_json        jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS conversation_user_created_idx
  ON cima.conversation (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS conversation_status_created_idx
  ON cima.conversation (status, created_at DESC);

-- Sources ------------------------------------------------------------------
-- display_* is the immutable witness of what the frontend showed.
-- content_text is the normalized/processable text used by CIMA.
CREATE TABLE IF NOT EXISTS cima.source (
  source_id                 uuid PRIMARY KEY,
  conversation_id           text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  kind                      text NOT NULL CHECK (kind IN ('chat_user','chat_assistant','file_text','feedback','system')),
  role                      text,
  turn_index                int,
  external_provider         text,
  external_conversation_id  text,
  external_message_id       text,
  revision_no               int NOT NULL DEFAULT 0,
  supersedes_source_id      uuid REFERENCES cima.source(source_id) ON DELETE SET NULL,
  language                  text DEFAULT 'und',
  mime_type                 text,
  display_text              text,
  display_format            text,
  display_sha256            text,
  content_text              text,
  content_sha256            text,
  size_bytes                bigint,
  created_at                timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS source_conv_created_idx
  ON cima.source (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS source_conv_kind_created_idx
  ON cima.source (conversation_id, kind, created_at);
CREATE INDEX IF NOT EXISTS source_external_msg_idx
  ON cima.source (external_provider, external_message_id);
CREATE INDEX IF NOT EXISTS source_content_sha_idx
  ON cima.source (content_sha256);

-- Files / blobs -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.file (
  file_id             uuid PRIMARY KEY,
  conversation_id     text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  source_id           uuid NOT NULL REFERENCES cima.source(source_id) ON DELETE CASCADE,
  filename            text NOT NULL,
  mime_type           text,
  size_bytes          bigint,
  sha256              text,
  storage_uri         text NOT NULL,
  status              text NOT NULL CHECK (status IN ('UPLOADED','PARSED','FAILED','DELETING','DELETED')),
  parse_meta_json     jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS file_conv_created_idx
  ON cima.file (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS file_conv_status_idx
  ON cima.file (conversation_id, status);

-- Chunks / EDUs -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.chunk (
  chunk_id                 uuid PRIMARY KEY,
  conversation_id          text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  source_id                uuid NOT NULL REFERENCES cima.source(source_id) ON DELETE CASCADE,
  file_id                  uuid REFERENCES cima.file(file_id) ON DELETE CASCADE,
  kind                     text NOT NULL CHECK (kind IN ('doc_chunk','chat_chunk','summary_chunk')),
  char_start               int NOT NULL,
  char_end                 int NOT NULL,
  locator_json             jsonb NOT NULL DEFAULT '{}'::jsonb,
  normalizer_version       int NOT NULL,
  chunker_version          int NOT NULL,
  text_hash                text,
  created_at               timestamptz NOT NULL DEFAULT now(),
  vector_state             text NOT NULL DEFAULT 'NONE' CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
  embedding_model_id       text,
  embedding_schema_version int,
  expires_at               timestamptz,
  is_pinned                boolean NOT NULL DEFAULT false,
  was_cited                boolean NOT NULL DEFAULT false,
  last_used_at             timestamptz
);

CREATE INDEX IF NOT EXISTS chunk_conv_created_idx
  ON cima.chunk (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS chunk_conv_file_idx
  ON cima.chunk (conversation_id, file_id);
CREATE INDEX IF NOT EXISTS chunk_conv_vector_state_idx
  ON cima.chunk (conversation_id, vector_state);

CREATE TABLE IF NOT EXISTS cima.edu (
  edu_id                   uuid PRIMARY KEY,
  conversation_id          text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  source_id                uuid NOT NULL REFERENCES cima.source(source_id) ON DELETE CASCADE,
  chunk_id                 uuid REFERENCES cima.chunk(chunk_id) ON DELETE SET NULL,
  kind                     text NOT NULL,
  span_refs_json           jsonb NOT NULL,
  features_json            jsonb NOT NULL DEFAULT '{}'::jsonb,
  quality                  real NOT NULL DEFAULT 1.0,
  normalizer_version       int NOT NULL,
  edu_segmenter_version    int NOT NULL,
  created_at               timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS edu_conv_created_idx
  ON cima.edu (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS edu_conv_source_idx
  ON cima.edu (conversation_id, source_id);

-- Local memory --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.local_citem (
  local_citem_id             uuid PRIMARY KEY,
  semantic_identity_id       uuid NOT NULL,
  conversation_id            text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  type                       text NOT NULL,
  text                       text NOT NULL,
  embedding_text             text NOT NULL,
  meta_json                  jsonb NOT NULL DEFAULT '{}'::jsonb,
  provenance_json            jsonb NOT NULL DEFAULT '{}'::jsonb,
  validity                   text NOT NULL DEFAULT 'unknown' CHECK (validity IN ('unknown','accepted','rejected')),
  salience                   real NOT NULL DEFAULT 0.0,
  created_at                 timestamptz NOT NULL DEFAULT now(),
  updated_at                 timestamptz NOT NULL DEFAULT now(),
  vector_state               text NOT NULL DEFAULT 'NONE' CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
  embedding_model_id         text,
  embedding_schema_version   int,
  expires_at                 timestamptz,
  is_pinned                  boolean NOT NULL DEFAULT false,
  was_cited                  boolean NOT NULL DEFAULT false,
  last_used_at               timestamptz,
  normalizer_version         int NOT NULL,
  citem_builder_version      int NOT NULL
);

CREATE INDEX IF NOT EXISTS local_citem_conv_created_idx
  ON cima.local_citem (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS local_citem_conv_type_idx
  ON cima.local_citem (conversation_id, type);
CREATE INDEX IF NOT EXISTS local_citem_semantic_identity_idx
  ON cima.local_citem (semantic_identity_id);
CREATE INDEX IF NOT EXISTS local_citem_conv_vector_state_idx
  ON cima.local_citem (conversation_id, vector_state);

CREATE TABLE IF NOT EXISTS cima.local_citem_evidence (
  local_citem_id            uuid NOT NULL REFERENCES cima.local_citem(local_citem_id) ON DELETE CASCADE,
  source_id                 uuid REFERENCES cima.source(source_id) ON DELETE CASCADE,
  chunk_id                  uuid REFERENCES cima.chunk(chunk_id) ON DELETE CASCADE,
  edu_id                    uuid REFERENCES cima.edu(edu_id) ON DELETE CASCADE,
  ordinal                   int NOT NULL DEFAULT 0,
  locator_json              jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (local_citem_id, ordinal)
);

CREATE TABLE IF NOT EXISTS cima.local_summary (
  local_summary_id            uuid PRIMARY KEY,
  conversation_id             text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  level                       text NOT NULL CHECK (level IN ('EPOCH','CLUSTER','MASTER')),
  cluster_id                  text,
  epoch_no                    int,
  text                        text NOT NULL,
  covers_json                 jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at                  timestamptz NOT NULL DEFAULT now(),
  updated_at                  timestamptz NOT NULL DEFAULT now(),
  vector_state                text NOT NULL DEFAULT 'NONE' CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
  embedding_model_id          text,
  embedding_schema_version    int
);

CREATE INDEX IF NOT EXISTS local_summary_conv_level_idx
  ON cima.local_summary (conversation_id, level, updated_at DESC);
CREATE INDEX IF NOT EXISTS local_summary_conv_cluster_idx
  ON cima.local_summary (conversation_id, cluster_id);

CREATE TABLE IF NOT EXISTS cima.local_summary_origin (
  local_summary_id          uuid NOT NULL REFERENCES cima.local_summary(local_summary_id) ON DELETE CASCADE,
  origin_kind               text NOT NULL CHECK (origin_kind IN ('local_citem','chunk','source')),
  origin_id                 uuid NOT NULL,
  ordinal                   int NOT NULL DEFAULT 0,
  PRIMARY KEY (local_summary_id, origin_kind, origin_id)
);

-- Global memory -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.global_citem (
  global_citem_id              uuid PRIMARY KEY,
  semantic_identity_id         uuid NOT NULL,
  origin_conversation_id       text NOT NULL,
  promotion_origin_local_citem_id uuid NOT NULL REFERENCES cima.local_citem(local_citem_id),
  type                         text NOT NULL,
  text                         text NOT NULL,
  embedding_text               text NOT NULL,
  meta_json                    jsonb NOT NULL DEFAULT '{}'::jsonb,
  provenance_json              jsonb NOT NULL DEFAULT '{}'::jsonb,
  validity                     text NOT NULL DEFAULT 'unknown' CHECK (validity IN ('unknown','accepted','rejected')),
  salience                     real NOT NULL DEFAULT 0.0,
  created_at                   timestamptz NOT NULL DEFAULT now(),
  updated_at                   timestamptz NOT NULL DEFAULT now(),
  vector_state                 text NOT NULL DEFAULT 'NONE' CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
  embedding_model_id           text,
  embedding_schema_version     int,
  expires_at                   timestamptz,
  is_pinned                    boolean NOT NULL DEFAULT false,
  was_cited                    boolean NOT NULL DEFAULT false,
  last_used_at                 timestamptz
);

CREATE INDEX IF NOT EXISTS global_citem_semantic_identity_idx
  ON cima.global_citem (semantic_identity_id);
CREATE INDEX IF NOT EXISTS global_citem_type_idx
  ON cima.global_citem (type);
CREATE INDEX IF NOT EXISTS global_citem_vector_state_idx
  ON cima.global_citem (vector_state);

CREATE TABLE IF NOT EXISTS cima.global_citem_evidence (
  global_citem_id              uuid NOT NULL REFERENCES cima.global_citem(global_citem_id) ON DELETE CASCADE,
  ordinal                      int NOT NULL DEFAULT 0,
  evidence_kind                text NOT NULL CHECK (evidence_kind IN ('source_snippet','chunk_snippet','external_ref')),
  source_text_snapshot         text,
  locator_json                 jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (global_citem_id, ordinal)
);

CREATE TABLE IF NOT EXISTS cima.global_summary (
  global_summary_id            uuid PRIMARY KEY,
  level                        text NOT NULL CHECK (level IN ('MASTER','CLUSTER')),
  cluster_id                   text,
  text                         text NOT NULL,
  covers_json                  jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at                   timestamptz NOT NULL DEFAULT now(),
  updated_at                   timestamptz NOT NULL DEFAULT now(),
  vector_state                 text NOT NULL DEFAULT 'NONE' CHECK (vector_state IN ('INDEXED','THINNED','NONE','EPHEMERAL')),
  embedding_model_id           text,
  embedding_schema_version     int
);

CREATE TABLE IF NOT EXISTS cima.global_summary_origin (
  global_summary_id           uuid NOT NULL REFERENCES cima.global_summary(global_summary_id) ON DELETE CASCADE,
  origin_kind                 text NOT NULL CHECK (origin_kind IN ('global_citem','global_summary')),
  origin_id                   uuid NOT NULL,
  ordinal                     int NOT NULL DEFAULT 0,
  PRIMARY KEY (global_summary_id, origin_kind, origin_id)
);

-- Handoff / continuity ------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.handoff_manifest (
  handoff_manifest_id         uuid PRIMARY KEY,
  source_conversation_id      text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  schema_version              int NOT NULL,
  checksum                    text NOT NULL,
  payload_json                jsonb,
  created_at                  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cima.handoff_validation (
  handoff_validation_id       uuid PRIMARY KEY,
  handoff_manifest_id         uuid NOT NULL REFERENCES cima.handoff_manifest(handoff_manifest_id) ON DELETE CASCADE,
  is_valid                    boolean NOT NULL,
  validation_report_json      jsonb NOT NULL,
  created_at                  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cima.handoff_restore_run (
  handoff_restore_run_id      uuid PRIMARY KEY,
  handoff_manifest_id         uuid NOT NULL REFERENCES cima.handoff_manifest(handoff_manifest_id) ON DELETE CASCADE,
  target_conversation_id      text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  delta_report_json           jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at                  timestamptz NOT NULL DEFAULT now()
);

-- Context / answer lineage --------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.context_view (
  context_view_id             uuid PRIMARY KEY,
  conversation_id             text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  request_source_id           uuid NOT NULL REFERENCES cima.source(source_id) ON DELETE CASCADE,
  created_at                  timestamptz NOT NULL DEFAULT now(),
  selection_json              jsonb NOT NULL,
  budget_json                 jsonb NOT NULL,
  model_id                    text NOT NULL,
  tokenizer_id                text NOT NULL,
  prompt_tokens               int NOT NULL,
  completion_tokens           int,
  total_tokens                int
);

CREATE INDEX IF NOT EXISTS context_view_conv_created_idx
  ON cima.context_view (conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS cima.answer_source_map (
  assistant_source_id         uuid NOT NULL REFERENCES cima.source(source_id) ON DELETE CASCADE,
  marker                      text NOT NULL,
  ref_kind                    text NOT NULL CHECK (ref_kind IN ('local_citem','local_summary','chunk','source','file','global_citem','global_summary')),
  ref_id                      text NOT NULL,
  locator_json                jsonb NOT NULL DEFAULT '{}'::jsonb,
  snippet_text                text,
  score                       real,
  PRIMARY KEY (assistant_source_id, marker)
);

CREATE TABLE IF NOT EXISTS cima.pin (
  conversation_id             text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  ref_kind                    text NOT NULL CHECK (ref_kind IN ('local_citem','local_summary','global_citem','global_summary','file','chunk')),
  ref_id                      text NOT NULL,
  note                        text,
  created_at                  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (conversation_id, ref_kind, ref_id)
);

-- Lifecycle / async plane ---------------------------------------------------
CREATE TABLE IF NOT EXISTS cima.delete_run (
  delete_run_id               uuid PRIMARY KEY,
  conversation_id             text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  status                      text NOT NULL CHECK (status IN ('REQUESTED','RUNNING','SUCCEEDED','FAILED')),
  requested_at                timestamptz NOT NULL DEFAULT now(),
  completed_at                timestamptz,
  stats_json                  jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS cima.maintenance_run (
  maintenance_run_id          uuid PRIMARY KEY,
  conversation_id             text,
  kind                        text NOT NULL CHECK (kind IN ('THINNING','RECONCILE','EPHEMERAL_EXPIRY','ORPHAN_CLEANUP')),
  status                      text NOT NULL CHECK (status IN ('REQUESTED','RUNNING','SUCCEEDED','FAILED')),
  requested_at                timestamptz NOT NULL DEFAULT now(),
  completed_at                timestamptz,
  stats_json                  jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS cima.ephemeral_vector (
  ephemeral_id                uuid PRIMARY KEY,
  conversation_id             text NOT NULL REFERENCES cima.conversation(conversation_id) ON DELETE CASCADE,
  origin_ref_kind             text NOT NULL CHECK (origin_ref_kind IN ('local_citem','local_summary','global_citem','global_summary','chunk')),
  origin_ref_id               uuid,
  qdrant_collection           text NOT NULL,
  lifecycle_state             text NOT NULL DEFAULT 'ACTIVE' CHECK (lifecycle_state IN ('ACTIVE','EXPIRED','PURGED')),
  vector_state                text NOT NULL DEFAULT 'EPHEMERAL' CHECK (vector_state = 'EPHEMERAL'),
  embedding_model_id          text,
  embedding_schema_version    int,
  eligible_for_geometry       boolean NOT NULL DEFAULT false,
  meta_json                   jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at                  timestamptz NOT NULL DEFAULT now(),
  expires_at                  timestamptz NOT NULL,
  expired_at                  timestamptz,
  purged_at                   timestamptz
);

CREATE INDEX IF NOT EXISTS ephemeral_vector_due_idx
  ON cima.ephemeral_vector (lifecycle_state, expires_at);
CREATE INDEX IF NOT EXISTS ephemeral_vector_conversation_idx
  ON cima.ephemeral_vector (conversation_id, lifecycle_state, expires_at);

CREATE TABLE IF NOT EXISTS cima.stage_state (
  stage_state_id              bigserial PRIMARY KEY,
  conversation_id             text,
  stage_name                  text NOT NULL,
  ref_kind                    text,
  ref_id                      text,
  status                      text NOT NULL CHECK (status IN ('STARTED','SUCCEEDED','FAILED','CANCELLED')),
  version_tuple               text NOT NULL,
  created_at                  timestamptz NOT NULL DEFAULT now(),
  updated_at                  timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS stage_state_unique_idx
  ON cima.stage_state (stage_name, ref_kind, ref_id, version_tuple);

CREATE TABLE IF NOT EXISTS cima.consumer_effect (
  consumer_name               text NOT NULL,
  event_id                    uuid NOT NULL,
  effect_key                  text NOT NULL,
  created_at                  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (consumer_name, event_id, effect_key)
);

CREATE TABLE IF NOT EXISTS cima.outbox (
  outbox_id                   bigserial PRIMARY KEY,
  topic                       text NOT NULL,
  message_key                 text NOT NULL,
  headers_json                jsonb NOT NULL DEFAULT '{}'::jsonb,
  payload_json                jsonb,
  status                      text NOT NULL DEFAULT 'NEW' CHECK (status IN ('NEW','SENT','ERROR')),
  created_at                  timestamptz NOT NULL DEFAULT now(),
  sent_at                     timestamptz,
  error                       text
);

CREATE INDEX IF NOT EXISTS outbox_status_created_idx
  ON cima.outbox (status, created_at);

-- Read models ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cima_rm.geom_run (
  conversation_id             text NOT NULL,
  run_id                      uuid NOT NULL,
  algo_version                text NOT NULL,
  universe_hash               text NOT NULL,
  k_used                      int NOT NULL,
  temp                        real NOT NULL,
  core_q                      real NOT NULL,
  bridge_percentile           int NOT NULL,
  metrics_json                jsonb NOT NULL,
  completed_at                timestamptz NOT NULL,
  PRIMARY KEY (conversation_id, run_id)
);

CREATE TABLE IF NOT EXISTS cima_rm.geom_item_state (
  conversation_id             text NOT NULL,
  ref_kind                    text NOT NULL CHECK (ref_kind IN ('local_citem','local_summary')),
  ref_id                      uuid NOT NULL,
  run_id                      uuid NOT NULL,
  cluster_top1                text NOT NULL,
  cluster_top2                text,
  w1                          real NOT NULL,
  w2                          real,
  margin                      real NOT NULL,
  is_core                     boolean NOT NULL,
  is_bridge_candidate         boolean NOT NULL,
  centrality                  real,
  updated_at                  timestamptz NOT NULL,
  PRIMARY KEY (conversation_id, ref_kind, ref_id)
);

CREATE TABLE IF NOT EXISTS cima_rm.geom_cluster_state (
  conversation_id             text NOT NULL,
  cluster_id                  text NOT NULL,
  run_id                      uuid NOT NULL,
  mass                        real NOT NULL,
  medoid_ref_kind             text NOT NULL,
  medoid_ref_id               uuid NOT NULL,
  summary_id                  uuid,
  updated_at                  timestamptz NOT NULL,
  PRIMARY KEY (conversation_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS cima_rm.conversation_overview (
  conversation_id             text PRIMARY KEY,
  visible_message_count       int NOT NULL DEFAULT 0,
  local_citem_count           int NOT NULL DEFAULT 0,
  local_summary_count         int NOT NULL DEFAULT 0,
  last_activity_at            timestamptz
);

-- Geometry bounded context --------------------------------------------------
CREATE TABLE IF NOT EXISTS geom.conversation_params (
  conversation_id             text PRIMARY KEY,
  algo_version                text NOT NULL DEFAULT 'geom_v1.0',
  k_max                       int NOT NULL DEFAULT 8,
  temp                        real NOT NULL DEFAULT 0.7,
  core_q                      real NOT NULL DEFAULT 0.2,
  bridge_percentile           int NOT NULL DEFAULT 90,
  centroid_match_tau          real NOT NULL DEFAULT 0.8,
  updated_at                  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS geom.run (
  run_id                      uuid PRIMARY KEY,
  conversation_id             text NOT NULL,
  universe_hash               text NOT NULL,
  k_used                      int NOT NULL,
  temp                        real NOT NULL,
  core_q                      real NOT NULL,
  bridge_percentile           int NOT NULL,
  n_vectors                   int NOT NULL,
  started_at                  timestamptz NOT NULL,
  completed_at                timestamptz,
  status                      text NOT NULL CHECK (status IN ('RUNNING','SUCCEEDED','FAILED')),
  metrics_json                jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS geom.cluster (
  conversation_id             text NOT NULL,
  cluster_id                  text NOT NULL,
  last_run_id                 uuid NOT NULL,
  mass                        real NOT NULL,
  medoid_ref_kind             text NOT NULL,
  medoid_ref_id               uuid NOT NULL,
  summary_id                  uuid,
  centroid_bytes              bytea,
  active                      boolean NOT NULL DEFAULT true,
  updated_at                  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (conversation_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS geom.item_state (
  conversation_id             text NOT NULL,
  ref_kind                    text NOT NULL CHECK (ref_kind IN ('local_citem','local_summary')),
  ref_id                      uuid NOT NULL,
  run_id                      uuid NOT NULL,
  cluster_top1                text NOT NULL,
  cluster_top2                text,
  w1                          real NOT NULL,
  w2                          real,
  margin                      real NOT NULL,
  is_core                     boolean NOT NULL,
  is_bridge_candidate         boolean NOT NULL,
  centrality                  real,
  updated_at                  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (conversation_id, ref_kind, ref_id)
);

CREATE TABLE IF NOT EXISTS geom.outbox (
  outbox_id                   bigserial PRIMARY KEY,
  topic                       text NOT NULL,
  message_key                 text NOT NULL,
  headers_json                jsonb NOT NULL DEFAULT '{}'::jsonb,
  payload_json                jsonb,
  status                      text NOT NULL DEFAULT 'NEW' CHECK (status IN ('NEW','SENT','ERROR')),
  created_at                  timestamptz NOT NULL DEFAULT now(),
  sent_at                     timestamptz,
  error                       text
);

CREATE INDEX IF NOT EXISTS geom_outbox_status_created_idx
  ON geom.outbox (status, created_at);

-- Canonical FTS -------------------------------------------------------------
ALTER TABLE cima.source
  ADD COLUMN IF NOT EXISTS search_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content_text, ''))) STORED;
CREATE INDEX IF NOT EXISTS source_search_gin
  ON cima.source USING gin (search_tsv);

ALTER TABLE cima.local_citem
  ADD COLUMN IF NOT EXISTS search_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED;
CREATE INDEX IF NOT EXISTS local_citem_search_gin
  ON cima.local_citem USING gin (search_tsv);

ALTER TABLE cima.global_citem
  ADD COLUMN IF NOT EXISTS search_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED;
CREATE INDEX IF NOT EXISTS global_citem_search_gin
  ON cima.global_citem USING gin (search_tsv);
