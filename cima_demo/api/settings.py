"""Application settings for CIMA Demonstrator."""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _bootstrap_env_prefix_compat(primary: str = "CIMA_DEMO_", legacy: str = "KIMA_") -> None:
    """Mirror legacy KIMA_* variables into the new CIMA_DEMO_* namespace.

    The demonstrator now reads the CIMA_DEMO_* prefix as the canonical public
    surface. Legacy KIMA_* variables remain accepted as a compatibility path
    during the migration.
    """
    for key, value in list(os.environ.items()):
        if not key.startswith(legacy):
            continue
        mapped = f"{primary}{key[len(legacy):]}"
        os.environ.setdefault(mapped, value)


_bootstrap_env_prefix_compat()

class Settings(BaseSettings):
    def __init__(self, **values):
        _bootstrap_env_prefix_compat()
        super().__init__(**values)

    model_config = SettingsConfigDict(
        env_prefix="CIMA_DEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str = "postgresql://cima_demo:cima_demo@localhost:5432/cima_demo"
    db_pool_min: int = 5
    db_pool_max: int = 20

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    # Witness-backend physical vector layout. The demonstrator still routes most
    # live traffic through local_citems, but the collection split is explicit so
    # GC and lifecycle jobs can operate on the approved bounded-context surface.
    qdrant_local_citems_collection: str = "cima_local_citems"
    qdrant_local_summaries_collection: str = "cima_local_summaries"
    qdrant_chunks_collection: str = "cima_chunks"
    qdrant_global_citems_collection: str = "cima_global_citems"
    qdrant_global_summaries_collection: str = "cima_global_summaries"
    qdrant_ephemeral_collection: str = "cima_ephemeral"
    # Backward-compat alias for legacy wiring that still expects a single field.
    qdrant_collection: str = "cima_local_citems"

    # ── LLM ──────────────────────────────────────────────────────────────────
    # llm_provider: "llamacpp" for a local OpenAI-compatible llama.cpp server,
    # "openai" for the hosted OpenAI API.  standalone_llm_backend can still
    # select the deterministic "rule" backend for CI.
    llm_provider: str = "llamacpp"
    llm_url: str = "http://localhost:8080"
    llm_model: str = "mistral"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com"
    openai_organization: str = ""
    openai_project: str = ""
    # Local GGUF inference can be slow on limited GPUs. This timeout is the
    # per-HTTP-call read budget for llama.cpp generation/tokenization calls.
    # Keep it intentionally larger than ordinary API latency; override with
    # CIMA_DEMO_LLM_TIMEOUT when running faster backends.
    llm_timeout: float = 3600.0
    # Retry policy: exponential backoff — delay = min(base * 2^attempt, max).
    # max_retries=12, base=5, max=120 → total wait budget ~10 min in addition
    # to individual request read budgets.
    llm_max_retries: int = 12
    llm_retry_delay_base: float = 5.0    # seconds — first retry wait
    llm_retry_delay_max: float = 120.0   # seconds — cap for exponential growth
    # (DeepSeek-R1, QwQ-32B, Llama-3.3-thinking served via llama.cpp ≥ b4450).
    # Set to True when the LLM runtime supports multimodal image inputs.
    # When False, screenshot images are stripped from tool messages before sending.
    llm_vision: bool = False
    # Inject assistant prefill "<response>\n" on synthesis passes to reduce
    # pre-response boilerplate and stabilize answer formatting.
    llm_response_prefill: bool = True

    # ── TEI Embedding ─────────────────────────────────────────────────────────
    tei_url: str = "http://localhost:8081"
    tei_timeout: float = 120.0
    tei_embed_dim: int = 768

    # ── TEI Reranker ──────────────────────────────────────────────────────────
    tei_reranker_url: str = "http://localhost:8082"
    tei_reranker_timeout: float = 120.0
    # Batch cap: cross-encoder VRAM ∝ N × seq_len² — cap prevents OOMKilled restarts.
    tei_reranker_max_batch: int = 32
    # Circuit breaker: open after N consecutive failures, stay open for M seconds.
    tei_reranker_circuit_threshold: int = 3
    tei_reranker_circuit_open_secs: float = 30.0

    # ── TEI NLI (conflict detection) ─────────────────────────────────────────
    # Empty string = disabled; LLM-only fallback used for conflict NLI.
    tei_nli_url: str = ""
    tei_nli_timeout: float = 10.0

    # ── Domain alias override (DEBT-02) ──────────────────────────────────────
    # Comma-separated "keyword=domain" pairs; merges over built-in defaults.
    # Set to "-" to disable all aliases (explicit-URL source-lock only).
    # Example: CIMA_DEMO_DOMAIN_ALIASES=wikipedia=wikipedia.org,arxiv=arxiv.org

    # ── ColBERT late-interaction reranker (optional) ─────────────────────────
    # Empty string = disabled. When set, applies a second late-interaction
    # reranking pass after the CrossEncoder, on the CrossEncoder's top-K output.
    # Deploy any TEI-compatible ColBERT model (e.g. colbert-ir/colbertv2.0)
    # at this URL. Uses the same /rerank endpoint as the CrossEncoder reranker.
    colbert_url: str = ""
    colbert_timeout: float = 20.0
    colbert_max_batch: int = 32

    # ── Auth ──────────────────────────────────────────────────────────────────
    api_key: str = ""
    api_key_required: bool = True

    # ── Turn / iteration limits ───────────────────────────────────────────────
    # max_iterations: hard safety cap — primary stop condition is stall detection.
    #   The agent should keep trying while it makes progress; iterations alone are
    #   not a meaningful stop signal. Raise only if stall detection is unreliable.
    # max_stall_count: cumulative stall count across turns before user warning + reset.
    # max_strategy_retries: how many times the reasoning module may retry a strategy
    #   after a failure signal before giving up and switching (RM-INV-07).
    max_iterations:       int = 20
    max_stall_count:      int = 5
    max_strategy_retries: int = 1

    # ── Turn timeout ──────────────────────────────────────────────────────────
    # Wall-clock limit for handle_turn(). Safety valve against runaway turns.
    # 20 iterations × 600s LLM + tool calls ≈ 4 h worst case. Default 7200s = 2 h.
    turn_timeout_secs: int = 7200
    # Max tokens per LLM call — two budgets by pass type:
    #   llm_max_tokens_tool: tool-call iterations (Mode A). The model emits tool calls
    #       only; text content is prohibited by protocol. Budget covers any accidental
    #       preamble. iGPU @ ~5 tok/s: 512 tokens ≈ 102 s hard ceiling.
    #   llm_max_tokens: synthesis pass (Mode B). Full response + conclusions.
    #       iGPU @ ~5 tok/s: 3000 tokens ≈ 10 min max. 0 = unlimited.
    llm_max_tokens_tool: int = 512
    llm_max_tokens: int = 3000
    # Context window configured when starting llama.cpp (e.g. --ctx-size).
    # 0 disables this cap, but production/demo runs should set it explicitly.
    # Context selection never uses this whole window: it reserves llm_max_tokens
    # for the answer and context_budget_overhead for system/history wrapping.
    llm_context_window: int = 49152
    # Prompt-level visible answer length controls for demo/open-scenario runs.
    # These do not truncate model output mechanically; they shape the answer
    # contract sent to instruction-tuned local models.
    demo_answer_specific_max_words: int = 160
    demo_answer_default_max_words: int = 220
    demo_answer_summary_max_words: int = 700
    # Keep this false for local publication runs: citation extraction is already
    # deterministic and running another LLM JSON pass after long answers is slow
    # and can produce truncated JSON on constrained GPUs. Enable only when you
    # explicitly want LLM-derived durable conclusions.
    demo_llm_memory_pass: bool = False
    # Temperature — Ministral best practices: 0.0 for deterministic tool selection
    # (Mode A), 0.2 for synthesis quality (Mode B).
    llm_temperature_tool: float = 0.0   # Mode A: tool-call passes
    llm_temperature: float = 0.2        # Mode B: synthesis passes
    # repeat_penalty=1.0 disables the penalty entirely. Ministral uses JSON-native
    # tool calls which contain many repeated structural tokens (quotes, braces, colons).
    # A value > 1.0 silently corrupts tool call JSON — keep at 1.0 for this model.
    llm_repeat_penalty: float = 1.0
    # top_p: nucleus sampling cutoff. 1.0 = disabled (full distribution sampled).
    # At temperature=0.0 (Mode A) this is irrelevant. At temperature=0.2 (Mode B)
    # top_p=1.0 gives the model full vocabulary access for natural language quality.
    llm_top_p: float = 1.0

    # ── Context budget ────────────────────────────────────────────────────────
    # Ministral-3-14B-Instruct-2512 has 256k native context. The budget here is
    # the amount we fill; the rest is reserved for the model's own output.
    # Default 40000: keeps working-set small and forces selective retrieval.
    # Raise CIMA_DEMO_CONTEXT_BUDGET_MAX up to ~196608 if VRAM allows (256k - 60k output).
    context_budget_max: int = 40000
    context_budget_overhead: int = 4096

    # ── File upload limits ────────────────────────────────────────────────────
    max_file_size_mb: int = 10
    max_files_per_request: int = 5

    # ── Logging / debug diagnostics ───────────────────────────────────────────
    log_level: str = "INFO"
    # When enabled, demo runtime writes bounded prompt/generation diagnostics to
    # the per-run artifact directory under debug/*.json. Keep disabled by default
    # because prompts may contain source text.
    debug_trace: bool = False
    debug_trace_max_chars: int = 50000
    # OpenAI-compatible SSE wrapper timeout. This is distinct from the llama.cpp
    # HTTP read timeout: it controls how long the API wrapper waits without any
    # visible delta before classifying the turn as a stream timeout. 0 disables.
    oai_stream_timeout_secs: int = 7200

    # ── Workers ───────────────────────────────────────────────────────────────
    forget_cycle_interval_secs: int = 18000  # 5 hours default
    forget_cycle_stale_hours: int = 24       # only run on conversations idle ≥ N hours
    gc_thinning_age_hours: int = 24          # witness async-plane thinning threshold

    # ── Workspace TTL ─────────────────────────────────────────────────────────
    # Hours before workspace subdirectories are cleaned. 0 = never delete.
    # Default 24 matches the legacy daily 00:00 UTC clean cycle.
    workspace_ttl_hours: int = 24

    # ── Workspace (ephemeral scratch volume, cleared daily at 00:00 UTC) ──────
    workspace_dir: str = "./cima_demo/workspace"
    workspace_max_mb: int = 500
    demo_artifacts_dir: str = "./cima_demo/artifacts"

    # ── Geometric expansion ───────────────────────────────────────────────────
    # APP-D-08: cosine threshold for semantic bridge filter (0.0 = off / keep all)
    bridge_threshold: float = 0.15
    # Maximum C-Items scanned in the backward pass of fetch_neighbors (SPEC-1).
    # The backward pass scrolls the full conversation to find items whose
    # dependency_ids reference any seed. For long conversations this is O(n)
    # in memory; capping prevents RSS spikes without losing recall quality
    # (recent items are checked first; older items are rarely back-references).
    geo_backward_max: int = 500

    # ── Context drift detection ───────────────────────────────────────────────
    # Two orthogonal drift monitors — both disabled by setting threshold to 0.0.
    #
    # LOCAL drift: fires every iteration when the current context does not cover
    # the active step objective. Triggers re-retrieval with a gap query targeting
    # missing concepts. Prevents noise accumulation and tangential context.
    # Anchor: active plan step description, or user_message when no plan.
    # 0.0 = disabled. Recommended: 0.40–0.55.
    #
    # GLOBAL drift: fires every global_drift_check_every iterations when the
    # current context does not cover the ORIGINAL user objective — detects
    # course deviation ("building a bicycle when asked for a tricycle").
    # Does NOT auto-correct; injects an alignment reminder into the prompt so
    # the model can self-correct. User notification on severe drift.
    # Anchor: user_message (fixed throughout the turn).
    # 0.0 = disabled. Recommended: 0.25–0.40.

    # ── Global recall relevance gate ──────────────────────────────────────────
    # Minimum similarity score for global-scoped C-Items to enter the RAG pipeline.
    # Q2 (global recall) results below this threshold are dropped before RRF merge,
    # preventing unrelated knowledge domains (e.g. OAuth2 FACT appearing in a classical
    # history conversation) from filling the context.
    # Episodic items (Q1, Q3) are NEVER filtered — they always pass through.
    # 0.0 = disabled (legacy behaviour — all global items pass regardless of relevance).
    # Recommended: 0.30 (cosine similarity from Qdrant hybrid search).
    global_min_score: float = 0.30

    # ── SPLADE sparse embedding (Phase 2, APP-D-09) ───────────────────────────
    # Empty string = disabled; BM25 (fastembed) used instead.
    splade_url: str = ""
    splade_timeout: float = 30.0

    # ── Genetic Reasoning Module (RM-INV-11) ──────────────────────────────────

    # ── RAG Evolution ────────────────────────────────────────────────────────

    # ── ReasoningConfig feature flags (SP-01..05) ─────────────────────────────
    # graph_context_depth: dependency annotations per C-Item in context (0=off, 1=1-hop).
    # plan_best_of_n: plan scoring candidates (1=off, 3=best-of-3; +2 LLM calls/plan).
    # conflict_sc_n: self-consistency NLI calls per conflict (1=off, 3=tri; +2 LLM calls).
    graph_context_depth: int = 1
    plan_best_of_n: int = 1
    conflict_sc_n: int = 1

    # ── Kafka event bus (ADR-001 Phase 5) ────────────────────────────────────
    # kafka_enabled: set True to route domain events to Kafka instead of in-process bus.
    kafka_enabled: bool = False
    kafka_bootstrap: str = "kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
    kafka_topic: str = "cima_demo.domain_events"

    # ── Server / runtime profile ─────────────────────────────────────────────
    # runtime_mode:
    #   standalone — local single-node demonstrator using in-memory stores.
    #                By default it uses the deterministic rule LLM for CI, but
    #                can call llama.cpp via CIMA_DEMO_STANDALONE_LLM_BACKEND=llamacpp.
    #   full       — PostgreSQL/Qdrant/llama.cpp/TEI backed runtime.
    runtime_mode: str = "standalone"
    # Standalone LLM backend: "rule" keeps CI deterministic; "llamacpp" uses
    # the configured OpenAI-compatible llama.cpp server (llm_url/llm_model) while
    # retaining in-memory stores, so open_scenarios can exercise real generation
    # without requiring Postgres/Qdrant/TEI.
    standalone_llm_backend: str = "rule"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    # ── Remote debug ─────────────────────────────────────────────────────────
    remote_debug: bool = False#set True (or env CIMA_DEMO_REMOTE_DEBUG=true) to enable debugpy.

    # ── Demonstrator mode ─────────────────────────────────────────────────────
    # When true, only user/assistant text is exposed to clients. Internal deltas
    # (reasoning, tool traces, plan-step events) remain internal runtime signals.
    demo_mode: bool = True


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
