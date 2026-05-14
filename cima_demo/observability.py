"""KIMA observability — Prometheus metrics + OpenTelemetry tracing + TurnTrace.

Span helpers (H-17 / SPEC-7):
  trace_span(name, **attrs) — context manager that opens an OTEL span when
  tracing is active, or yields None silently when it is not.  Use this
  instead of importing opentelemetry directly so callers are insulated from
  import-time failures.


Metrics server: port 8001 (separate HTTP server via prometheus_client).
OTEL tracing:   gRPC export to OTEL_EXPORTER_OTLP_ENDPOINT (env var).
TurnTrace:      structured per-turn JSON log emitted at debug level.
All are fully optional — if deps are not installed the module degrades
silently so the main application is never blocked by observability failures.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Generator

log = logging.getLogger(__name__)
_trace_log = logging.getLogger("cima_demo.trace")


# ── TurnTrace ─────────────────────────────────────────────────────────────────

@dataclass
class TurnTrace:
    """Structured per-turn trace — emitted as JSON at the end of every turn.

    Fields are intentionally flat/serialisable: no nested objects, only
    primitives, lists of strings/dicts, and numeric counters.

    Consumed by: log aggregators, postmortem analysis, benchmark classification.
    """
    conversation_id:    str
    turn_id:            str
    mode:               str | None       # ExecutionMode value
    execution_stage:    str              # final ExecutionStage at turn end
    input_has_file:     bool
    iteration_count:    int
    artifact_count:     int              # web/memory artifacts fetched
    resolved_slot_count: int
    compute_done:       bool
    # Validator result
    answer_valid:       bool
    answer_error_class: str | None       # ValidationResult.error_class or None
    # Timing
    e2e_ms:             float
    ttft_ms:            float | None
    # Tools used (deduplicated, ordered by first use)
    tools_used:         list[str] = field(default_factory=list)
    # Stage transitions as list of stage strings in order
    stage_transitions:  list[str] = field(default_factory=list)
    # Final error class for the turn
    final_error_class:  str | None = None   # exception type name or None
    # Protocol validity of the final answer
    protocol_valid:     bool = True
    # Canonical terminal outcome — set deterministically by engine exit logic.
    # Never inferred from logs or free text.  Benchmark tools read this field.
    outcome_code:       str | None = None   # TurnOutcome value
    # Additional metrics
    extra:              dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalStageTrace:
    """Per-turn structured trace of the RAG pipeline stages.

    Emitted as part of TurnTrace.extra['retrieval'] when retrieval runs.
    Enables causal postmortem: for each run one can reconstruct
    q1/q2/q3 sizes → RRF merge → rerank output → expansion → final packing.
    """
    query_type:             str
    recall_top_k:           int
    rerank_top_n:           int
    # Stage sizes
    q1_size:                int = 0     # episodic recall (before RRF)
    q2_size:                int = 0     # global recall (before RRF)
    q3_size:                int = 0     # flagged recall (total)
    q3_relevant:            int = 0     # flagged after local-relevance gate
    after_rrf:              int = 0     # merged Q1+Q2 after RRF
    after_rerank:           int = 0     # after CrossEncoder rerank
    geometric_added:        int = 0     # items added by geometric expansion
    bridge_eligible:        int = 0     # items eligible for bridge lane
    after_expand:           int = 0     # total candidates before greedy_select
    items_selected:         int = 0     # items in final ContextPack
    # Quality signals
    coverage_score:         float = 0.0
    traceability_density:   float = 0.0
    reranker_available:     bool = True
    bridge_enabled:         bool = False
    bridge_alpha:           float = 0.5
    bridge_floor:           float = 0.0
    # Retry
    retry_count:            int = 0
    latency_ms:             int = 0


@contextmanager
def trace_span(
    name: str,
    **attributes: Any,
) -> Generator[Any, None, None]:
    """Open a named OTEL span if tracing is active; yield None silently otherwise.

    H-17 (SPEC-7): lightweight wrapper so callers don't need opentelemetry
    imports and don't break when the SDK is absent.

    Usage::

        with trace_span("cima_demo.retrieval", query_type="multi_hop") as span:
            ...
            if span:
                span.set_attribute("after_rrf", len(merged))
    """
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("cima_demo")
        with tracer.start_as_current_span(name) as span:
            for k, v in attributes.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
            yield span
    except Exception:
        yield None  # tracing unavailable — caller continues unaffected


def emit_retrieval_span(
    query_type: str,
    recall_top_k: int,
    rerank_top_n: int,
    latency_ms: int,
    *,
    q1: int = 0,
    q2: int = 0,
    q3: int = 0,
    after_rrf: int = 0,
    after_rerank: int = 0,
    after_expand: int = 0,
    items_selected: int = 0,
    coverage_score: float = 0.0,
    reranker_available: bool = True,
    bridge_enabled: bool = False,
    bridge_alpha: float = 0.5,
    retry_count: int = 0,
    conversation_id: str = "",
) -> None:
    """Emit a completed RAG-pipeline OTEL span with retrieval metrics.

    H-17 (SPEC-7): called at the end of RetrievalOrchestrator.retrieve() so
    the span carries final stage sizes without wrapping the entire async body.
    No-op when OTEL is unavailable.
    """
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("cima_demo")
        with tracer.start_as_current_span("cima_demo.retrieval") as span:
            span.set_attribute("query_type",          query_type)
            span.set_attribute("recall_top_k",        recall_top_k)
            span.set_attribute("rerank_top_n",        rerank_top_n)
            span.set_attribute("latency_ms",          latency_ms)
            span.set_attribute("q1_size",             q1)
            span.set_attribute("q2_size",             q2)
            span.set_attribute("q3_size",             q3)
            span.set_attribute("after_rrf",           after_rrf)
            span.set_attribute("after_rerank",        after_rerank)
            span.set_attribute("after_expand",        after_expand)
            span.set_attribute("items_selected",      items_selected)
            span.set_attribute("coverage_score",      coverage_score)
            span.set_attribute("reranker_available",  reranker_available)
            span.set_attribute("bridge_enabled",      bridge_enabled)
            span.set_attribute("bridge_alpha",        bridge_alpha)
            span.set_attribute("retry_count",         retry_count)
            if conversation_id:
                span.set_attribute("conversation_id", conversation_id)
    except Exception:
        pass  # tracing unavailable — non-fatal


def emit_tool_dispatch_span(
    tool_name: str,
    success: bool,
    latency_ms: int,
    *,
    conversation_id: str = "",
    state_changed: bool = True,
) -> None:
    """Emit a completed tool-dispatch OTEL span.

    H-17 (SPEC-7): called after each real tool call completes.
    """
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("cima_demo")
        with tracer.start_as_current_span("cima_demo.tool_dispatch") as span:
            span.set_attribute("tool_name",       tool_name)
            span.set_attribute("success",         success)
            span.set_attribute("latency_ms",      latency_ms)
            span.set_attribute("state_changed",   state_changed)
            if conversation_id:
                span.set_attribute("conversation_id", conversation_id)
    except Exception:
        pass


def emit_turn_trace(trace: TurnTrace) -> None:
    """Emit a TurnTrace as a structured JSON log line at DEBUG level.

    Uses logger 'cima_demo.trace' so it can be routed to a dedicated handler
    (e.g. a JSON file appender) without polluting the main log stream.
    """
    import json
    try:
        _trace_log.debug(
            "TURN_TRACE %s",
            json.dumps(asdict(trace), ensure_ascii=False, default=str),
        )
    except Exception as exc:
        log.debug("emit_turn_trace failed (non-fatal): %s", exc)

# ── Prometheus ─────────────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server as _prom_start

    _TURN_DURATION = Histogram(
        "kima_turn_duration_ms",
        "End-to-end turn duration in milliseconds",
        buckets=[500, 1_000, 5_000, 15_000, 30_000, 60_000, 120_000, 300_000, 600_000],
    )
    _TTFT = Histogram(
        "kima_ttft_ms",
        "Time to first token in milliseconds",
        buckets=[100, 500, 1_000, 2_000, 5_000, 15_000, 30_000, 60_000, 120_000, 300_000],
    )
    _TOOL_CALLS = Counter(
        "kima_tool_calls_total",
        "Total tool calls dispatched per tool",
        ["tool_name"],
    )
    _TOOL_SUCCESS = Counter(
        "kima_tool_success_total",
        "Successful tool calls per tool",
        ["tool_name"],
    )
    _TOOL_FAILURE = Counter(
        "kima_tool_failure_total",
        "Failed tool calls per tool",
        ["tool_name"],
    )
    _COMPLEXITY = Histogram(
        "kima_complexity_score",
        "Turn complexity score (depth × width / 2)",
        buckets=[0.0, 0.5, 1.0, 2.0, 3.0, 4.5],
    )
    _STEP_SUCCESS_RATE = Histogram(
        "kima_step_success_rate",
        "Fraction of tool calls that succeeded (0.0–1.0)",
        buckets=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    _ACTIVE_TURNS = Gauge(
        "kima_active_turns",
        "Turns currently being processed",
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    log.warning("prometheus-client not installed — Prometheus metrics disabled")


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus metrics HTTP server on *port*."""
    if not _PROM_AVAILABLE:
        log.warning("Cannot start metrics server: prometheus-client not installed")
        return
    try:
        _prom_start(port)
        log.info("Prometheus metrics server listening on port %d", port)
    except Exception as exc:
        log.error("Failed to start Prometheus metrics server on port %d: %s", port, exc)


def record_turn_metrics(metrics: dict[str, Any]) -> None:
    """Record TurnMetrics dict into Prometheus histograms/counters.

    Called from AgentOrchestrator after each completed turn.
    ``metrics`` is the same dict emitted in the TurnMetrics log line.
    """
    if not _PROM_AVAILABLE:
        return
    try:
        _TURN_DURATION.observe(metrics["e2e_ms"])
        if metrics.get("ttft_ms") is not None:
            _TTFT.observe(metrics["ttft_ms"])
        _COMPLEXITY.observe(metrics.get("complexity_score", 0.0))
        _STEP_SUCCESS_RATE.observe(metrics.get("step_success_rate", 1.0))
        for tool in metrics.get("tools_used", []):
            _TOOL_CALLS.labels(tool_name=tool).inc()
    except Exception as exc:
        log.debug("record_turn_metrics failed (non-fatal): %s", exc)


def record_tool_result(tool_name: str, *, success: bool) -> None:
    """Increment success/failure counter for a single tool dispatch."""
    if not _PROM_AVAILABLE:
        return
    try:
        if success:
            _TOOL_SUCCESS.labels(tool_name=tool_name).inc()
        else:
            _TOOL_FAILURE.labels(tool_name=tool_name).inc()
    except Exception:
        pass


def active_turn_inc() -> None:
    if _PROM_AVAILABLE:
        try:
            _ACTIVE_TURNS.inc()
        except Exception:
            pass


def active_turn_dec() -> None:
    if _PROM_AVAILABLE:
        try:
            _ACTIVE_TURNS.dec()
        except Exception:
            pass


# ── OpenTelemetry ──────────────────────────────────────────────────────────────
def setup_tracing() -> None:
    """Configure OTEL tracing from environment variables.

    Required env vars (already present in the pod):
      OTEL_EXPORTER_OTLP_ENDPOINT  — gRPC endpoint, e.g. http://otel-gateway:4317
      OTEL_SERVICE_NAME            — defaults to 'cima-demo-api'

    If OTEL_EXPORTER_OTLP_ENDPOINT is empty or unset, tracing is silently
    skipped (no packages are required to be installed in that case).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        log.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.getenv("OTEL_SERVICE_NAME", "cima-demo-api")
        resource = Resource.create({
            SERVICE_NAME:    service_name,
            SERVICE_VERSION: "0.1.0",
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        log.info("OTEL tracing initialized: service=%s endpoint=%s", service_name, endpoint)

        # FastAPI HTTP span instrumentation
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument()
            log.info("FastAPI OTEL instrumentation active")
        except ImportError:
            log.warning(
                "opentelemetry-instrumentation-fastapi not installed — "
                "HTTP request spans disabled"
            )

        # httpx instrumentation — covers TEI, llama.cpp, SearXNG outbound calls
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
            log.info("HTTPX OTEL instrumentation active")
        except ImportError:
            log.warning(
                "opentelemetry-instrumentation-httpx not installed — "
                "outbound HTTP spans disabled"
            )

    except ImportError as exc:
        log.warning("OpenTelemetry SDK not installed — tracing disabled (%s)", exc)
    except Exception as exc:
        log.error("OTEL tracing setup failed: %s", exc)
