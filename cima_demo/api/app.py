"""FastAPI application for CIMA Demonstrator."""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from jinja2 import Environment, FileSystemLoader

from cima_demo.observability import setup_tracing, start_metrics_server
from cima_demo.api.routers import chat, context, conversations, handoff, health, rag, runs, sources
from cima_demo.branding import PUBLIC_DOCS_PATH, PUBLIC_PROJECT_NAME
from cima_demo.api.settings import get_settings
from cima_demo.api.budgeting import build_effective_context_budget
from cima_demo.application.orchestrator import AgentOrchestrator
from cima_demo.application.plan_executor import PlanExecutor
from cima_demo.application.stream_manager import StreamManager
from cima_demo.cognitive.kernel.events import TurnEventType, make_event
from cima_demo.demo.context import DemoContextService
from cima_demo.demo.handoff import DemoHandoffService
from cima_demo.demo.harness.fakes import (
    HarnessContextBuilder,
    HarnessMemoryService,
    InMemoryCItemStore,
    InMemoryDemoDB,
    StandaloneRuleLLM,
)
from cima_demo.demo.lifecycle import DemoLifecycleAuditService
from cima_demo.demo.lineage import DemoLineageService
from cima_demo.demo.runtime import DemoRunJournal
from cima_demo.demo.runtime.controller import DemoTurnController
from cima_demo.domain.value_objects import ContextBudget, ForgetParams, PromotionPolicy
from cima_demo.geometry import (
    DemoGeometryService,
    DirectGeometryBoundary,
    GeometryCommandPublisher,
    GeometryReadModelService,
    NoOpGeometricExpander,
)
from cima_demo.infrastructure.events.direct import DirectSSEEventBus
from cima_demo.infrastructure.events.domain_bus import InProcessDomainEventBus
from cima_demo.infrastructure.events.kafka_bus import kafka_bus_from_env
from cima_demo.workers.workspace_cleaner import WorkspaceCleanerWorker
from cima_demo.witness_backend.ephemeral import EphemeralVectorRegistry
from cima_demo.witness_backend.ephemeral_runtime import EphemeralRuntimeMirror
from cima_demo.witness_backend.source_ingest import SourceRegistrationService

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=False, keep_trailing_newline=True)


def _render_system_prompt(
    strategy_section: str | None = None,
    include_external: bool = True,
    synthesis: bool = False,
    allowed_tools: list[str] | None = None,
) -> str:
    settings = get_settings()
    tmpl = _jinja_env.get_template("system_prompt.j2")
    return tmpl.render(
        strategy_section=strategy_section,
        include_external=include_external,
        synthesis=synthesis,
        allowed_tools=allowed_tools,
    )


def _common_runtime(
    *,
    app: FastAPI,
    settings: Any,
    db: Any,
    citem_store: Any,
    llm_adapter: Any,
    file_adapter: Any | None,
    chunking_port: Any,
    geometric_expander: Any,
    reranker_adapter: Any | None,
    nli_adapter: Any | None,
    colbert_adapter: Any | None,
    event_bus: Any,
    domain_event_bus: Any,
    geometry_reader: Any | None,
    geometry_commands: Any | None,
    ephemeral_runtime: Any | None,
    artifacts_path: Path,
    workspace_path: Path,
    forget_worker_factory: Callable[[Any, Any, DemoLifecycleAuditService], Any] | None = None,
) -> tuple[Any, Any | None, Any | None]:
    from cima_demo.retrieval.context_builder import ContextBuilder
    from cima_demo.retrieval.multi_hop import MultiHopAnalyzer
    from cima_demo.retrieval.orchestrator import RetrievalOrchestrator
    from cima_demo.retrieval.query_planner import QueryPlanner

    stream_manager = StreamManager(event_bus=event_bus)
    demo_lineage_service = DemoLineageService(rel_db=db)
    if file_adapter is None:
        memory_service = HarnessMemoryService(citem_store, db=db, lineage=demo_lineage_service)
    else:
        from cima_demo.memory.service import MemoryService
        memory_service = MemoryService(
            rel_db=db,
            citem_store=citem_store,
            llm_port=llm_adapter,
            file_processor=file_adapter,
            chunking_port=chunking_port,
            stream_manager=stream_manager,
            forget_params=ForgetParams.default(),
            promotion_policy=PromotionPolicy.default(),
            workspace_dir=workspace_path,
            workspace_max_mb=settings.workspace_max_mb,
            nli_port=nli_adapter,
            lineage_service=demo_lineage_service,
        )

    if isinstance(chunking_port, HarnessContextBuilder):
        context_builder = chunking_port
        plan_executor = PlanExecutor(rel_db=db)
        multi_hop_analyzer = None
    else:
        query_planner = QueryPlanner()
        retrieval_orchestrator = RetrievalOrchestrator(
            citem_store=citem_store,
            geometric_expander=NoOpGeometricExpander() if settings.demo_mode else geometric_expander,
            reranker_port=reranker_adapter,
            rel_db=db,
            global_min_score=settings.global_min_score,
            colbert_port=colbert_adapter,
        )
        multi_hop_analyzer = MultiHopAnalyzer(llm=llm_adapter)
        context_builder = ContextBuilder(
            query_planner=query_planner,
            retrieval_orchestrator=retrieval_orchestrator,
            rel_db=db,
            multi_hop_analyzer=multi_hop_analyzer,
        )
        plan_executor = PlanExecutor(rel_db=db)

    demo_run_journal = DemoRunJournal(rel_db=db, artifacts_root=artifacts_path)
    handoff_service = DemoHandoffService(
        rel_db=db,
        citem_store=citem_store,
        run_journal=demo_run_journal,
        artifacts_root=artifacts_path,
    )
    lifecycle_audit_service = DemoLifecycleAuditService(
        rel_db=db,
        citem_store=citem_store,
        memory_service=memory_service,
        artifacts_root=artifacts_path,
    )
    demo_context_service = DemoContextService(
        base_builder=context_builder,
        memory_service=memory_service,
        rel_db=db,
        run_journal=demo_run_journal,
        geometry_reader=geometry_reader,
        geometry_commands=geometry_commands,
        handoff_service=handoff_service,
        ephemeral_runtime=ephemeral_runtime,
    )
    demo_turn_controller = DemoTurnController(
        llm_port=llm_adapter,
        stream_manager=stream_manager,
        context_service=demo_context_service,
        memory_service=memory_service,
        context_budget=build_effective_context_budget(
            requested_context_tokens=settings.context_budget_max,
            reserve_output_tokens=settings.llm_max_tokens if settings.llm_max_tokens > 0 else None,
            overhead_tokens=settings.context_budget_overhead,
            settings=settings,
        ),
        run_journal=demo_run_journal,
        llm_temperature=settings.llm_temperature,
        llm_top_p=settings.llm_top_p,
        llm_repeat_penalty=settings.llm_repeat_penalty,
        llm_max_tokens=(settings.llm_max_tokens if settings.llm_max_tokens > 0 else None),
        answer_specific_max_words=settings.demo_answer_specific_max_words,
        answer_default_max_words=settings.demo_answer_default_max_words,
        answer_summary_max_words=settings.demo_answer_summary_max_words,
        debug_trace=settings.debug_trace,
        debug_trace_max_chars=settings.debug_trace_max_chars,
        llm_memory_pass=settings.demo_llm_memory_pass,
    ) if settings.demo_mode else None

    orchestrator = AgentOrchestrator(
        llm_port=llm_adapter,
        rel_db=db,
        memory_service=memory_service,
        context_builder=demo_context_service,
        stream_manager=stream_manager,
        plan_executor=plan_executor,
        context_budget=build_effective_context_budget(
            requested_context_tokens=settings.context_budget_max,
            reserve_output_tokens=settings.llm_max_tokens if settings.llm_max_tokens > 0 else None,
            overhead_tokens=settings.context_budget_overhead,
            settings=settings,
        ),
        system_prompt_factory=_render_system_prompt,
        turn_timeout_secs=settings.turn_timeout_secs,
        llm_max_tokens=settings.llm_max_tokens,
        llm_max_tokens_tool=settings.llm_max_tokens_tool,
        llm_temperature=settings.llm_temperature,
        llm_temperature_tool=settings.llm_temperature_tool,
        llm_repeat_penalty=settings.llm_repeat_penalty,
        llm_top_p=settings.llm_top_p,
        max_iterations=settings.max_iterations,
        max_stall_count=settings.max_stall_count,
        max_strategy_retries=settings.max_strategy_retries,
        domain_event_publisher=domain_event_bus,
        demo_run_journal=demo_run_journal,
        demo_turn_controller=demo_turn_controller,
    )

    forget_worker = None
    if forget_worker_factory is not None:
        forget_worker = forget_worker_factory(memory_service, db, lifecycle_audit_service)
    workspace_cleaner = WorkspaceCleanerWorker(workspace_dir=workspace_path, ttl_hours=settings.workspace_ttl_hours)

    app.state.db = db
    app.state.citem_store = citem_store
    app.state.orchestrator = orchestrator
    app.state.stream_manager = stream_manager
    app.state.demo_run_journal = demo_run_journal
    app.state.context_service = demo_context_service
    app.state.lineage_service = demo_lineage_service
    app.state.memory_service = memory_service
    app.state.demo_turn_controller = demo_turn_controller
    app.state.source_registration_service = SourceRegistrationService(
        db=db,
        lineage=demo_lineage_service,
        workspace_root=workspace_path,
    )
    app.state.geometry_reader = geometry_reader
    app.state.geometry_commands = geometry_commands
    app.state.handoff_service = handoff_service
    app.state.lifecycle_audit_service = lifecycle_audit_service
    app.state.llm = llm_adapter
    app.state.embedding = None
    app.state.reranker = reranker_adapter
    app.state.runtime_mode = settings.runtime_mode
    app.state.workspace_cleaner = workspace_cleaner
    app.state.forget_worker = forget_worker

    return workspace_cleaner, forget_worker, None


async def _bootstrap_standalone_runtime(app: FastAPI, settings: Any, workspace_path: Path, artifacts_path: Path) -> tuple[Any, Any | None, Any | None]:
    log.info("Bootstrapping standalone single-node CIMA Demonstrator runtime")
    db = InMemoryDemoDB()
    citem_store = InMemoryCItemStore()
    standalone_backend = str(getattr(settings, "standalone_llm_backend", "rule") or "rule").strip().lower()
    if standalone_backend in {"openai", "hosted_openai"}:
        from cima_demo.infrastructure.llm.openai_chat import OpenAIChatAdapter

        log.info(
            "Standalone runtime using OpenAI LLM backend base_url=%s model=%s",
            settings.openai_base_url,
            settings.llm_model,
        )
        llm_adapter = OpenAIChatAdapter(
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            organization=settings.openai_organization or None,
            project=settings.openai_project or None,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            retry_delay_base=settings.llm_retry_delay_base,
            retry_delay_max=settings.llm_retry_delay_max,
            debug_trace=settings.debug_trace,
            debug_trace_max_chars=settings.debug_trace_max_chars,
        )
    elif standalone_backend in {"llamacpp", "llama", "real", "mistral"}:
        from cima_demo.infrastructure.llm.llamacpp import LlamaCppAdapter

        log.info(
            "Standalone runtime using llama.cpp LLM backend url=%s model=%s",
            settings.llm_url,
            settings.llm_model,
        )
        llm_adapter = LlamaCppAdapter(
            base_url=settings.llm_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            retry_delay_base=settings.llm_retry_delay_base,
            retry_delay_max=settings.llm_retry_delay_max,
            response_prefill_enabled=settings.llm_response_prefill,
            debug_trace=settings.debug_trace,
            debug_trace_max_chars=settings.debug_trace_max_chars,
        )
    else:
        llm_adapter = StandaloneRuleLLM()
    geometry_service = DemoGeometryService(rel_db=db, citem_store=citem_store, embedding_port=None, k_max=4)
    geometry_boundary = DirectGeometryBoundary(geometry_service)
    event_bus = DirectSSEEventBus()
    domain_event_bus = InProcessDomainEventBus()
    context_builder = HarnessContextBuilder(store=citem_store, db=db, include_summaries=True)
    return _common_runtime(
        app=app,
        settings=settings,
        db=db,
        citem_store=citem_store,
        llm_adapter=llm_adapter,
        file_adapter=None,
        chunking_port=context_builder,
        geometric_expander=NoOpGeometricExpander(),
        reranker_adapter=None,
        nli_adapter=None,
        colbert_adapter=None,
        event_bus=event_bus,
        domain_event_bus=domain_event_bus,
        geometry_reader=geometry_boundary,
        geometry_commands=geometry_boundary,
        ephemeral_runtime=None,
        artifacts_path=artifacts_path,
        workspace_path=workspace_path,
        forget_worker_factory=None,
    )


async def _bootstrap_full_runtime(app: FastAPI, settings: Any, workspace_path: Path, artifacts_path: Path) -> tuple[Any, Any | None, Any | None]:
    from cima_demo.infrastructure.aliases.adapter import InProcessDomainAliasAdapter  # noqa: F401
    from cima_demo.infrastructure.embedding.tei import TEIAdapter
    from cima_demo.infrastructure.files.chunker import SemanticChunkerAdapter
    from cima_demo.infrastructure.files.processor import FileProcessingAdapter
    from cima_demo.infrastructure.llm.llamacpp import LlamaCppAdapter
    from cima_demo.infrastructure.postgres.migrations import run_migrations
    from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
    from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
    from cima_demo.infrastructure.qdrant.expander import DependencyIdsGeometricExpander
    from cima_demo.infrastructure.qdrant.qdrant import QdrantCItemAdapter
    from cima_demo.infrastructure.tokenizer import LlamaCppTokenizerClient
    from cima_demo.infrastructure.reranker.tei import TEIRerankerAdapter
    from cima_demo.infrastructure.nli.tei import TEINLIAdapter
    from cima_demo.workers.forget_worker import ForgetCycleWorker
    from qdrant_client import AsyncQdrantClient

    pg_pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    pg_adapter = PostgreSQLAdapter(pg_pool)
    await run_migrations(pg_pool)

    llm_provider = str(getattr(settings, "llm_provider", "llamacpp") or "llamacpp").strip().lower()
    if llm_provider == "openai":
        from cima_demo.infrastructure.llm.openai_chat import OpenAIChatAdapter

        llama_adapter = OpenAIChatAdapter(
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            organization=settings.openai_organization or None,
            project=settings.openai_project or None,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            retry_delay_base=settings.llm_retry_delay_base,
            retry_delay_max=settings.llm_retry_delay_max,
            debug_trace=settings.debug_trace,
            debug_trace_max_chars=settings.debug_trace_max_chars,
        )
    else:
        llama_adapter = LlamaCppAdapter(
            base_url=settings.llm_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            retry_delay_base=settings.llm_retry_delay_base,
            retry_delay_max=settings.llm_retry_delay_max,
            response_prefill_enabled=settings.llm_response_prefill,
            debug_trace=settings.debug_trace,
            debug_trace_max_chars=settings.debug_trace_max_chars,
        )
    tei_adapter = TEIAdapter(base_url=settings.tei_url, timeout=settings.tei_timeout)
    reranker_adapter = TEIRerankerAdapter(
        base_url=settings.tei_reranker_url,
        timeout=settings.tei_reranker_timeout,
        max_batch=settings.tei_reranker_max_batch,
        circuit_threshold=settings.tei_reranker_circuit_threshold,
        circuit_open_secs=settings.tei_reranker_circuit_open_secs,
    )
    nli_adapter = TEINLIAdapter(base_url=settings.tei_nli_url, timeout=settings.tei_nli_timeout) if settings.tei_nli_url else None
    colbert_adapter = TEIRerankerAdapter(base_url=settings.colbert_url, timeout=settings.colbert_timeout, max_batch=settings.colbert_max_batch) if settings.colbert_url else None

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    catalog = QdrantCollectionCatalog.from_settings(settings)
    _splade_port = None
    if settings.splade_url:
        from cima_demo.infrastructure.splade.splade import SPLADEAdapter
        _splade_port = SPLADEAdapter(base_url=settings.splade_url, timeout=settings.splade_timeout)
    citem_adapter = QdrantCItemAdapter(
        client=qdrant_client,
        embedding_port=tei_adapter,
        collection=catalog.local_citems,
        global_collection=catalog.global_citems,
        ephemeral_collection=catalog.ephemeral,
        sparse_embedding_port=_splade_port,
        dense_dim=settings.tei_embed_dim,
        rel_db=pg_adapter,
    )
    from cima_demo.infrastructure.qdrant.setup import ensure_collections
    from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
    import asyncio as _asyncio
    for attempt in range(1, 6):
        try:
            await ensure_collections(qdrant_client, catalog, settings.tei_embed_dim)
            break
        except Exception as exc:
            if attempt == 5:
                raise RuntimeError("Qdrant witness collections could not be created") from exc
            await _asyncio.sleep(attempt * 2)
    witness_plane = QdrantWitnessPlane(client=qdrant_client, catalog=catalog, dense_dim=settings.tei_embed_dim)
    ephemeral_runtime = EphemeralRuntimeMirror(
        plane=witness_plane,
        embedder=tei_adapter,
        registry=EphemeralVectorRegistry(pg_adapter),
        ttl_seconds=getattr(settings, "ephemeral_runtime_ttl_seconds", 900),
        max_items=getattr(settings, "ephemeral_runtime_max_items", 12),
        embedding_model_id=getattr(settings, "tei_model_id", "tei"),
        embedding_schema_version=1,
        conversation_reader=pg_adapter,
    )
    geometry_reader = GeometryReadModelService(pg_adapter)
    geometry_commands = GeometryCommandPublisher(pg_adapter)
    event_bus = DirectSSEEventBus()
    _kafka_bus = kafka_bus_from_env()
    if _kafka_bus is not None:
        await _kafka_bus.start()
        domain_event_bus: Any = _kafka_bus
    else:
        domain_event_bus = InProcessDomainEventBus()
    tokenizer_client = LlamaCppTokenizerClient(base_url=settings.llm_url, timeout=settings.llm_timeout)
    def _sync_count(text: str) -> int:
        return tokenizer_client.count_text_tokens_sync(text)
    chunker = SemanticChunkerAdapter(token_counter=_sync_count)
    return _common_runtime(
        app=app,
        settings=settings,
        db=pg_adapter,
        citem_store=citem_adapter,
        llm_adapter=llama_adapter,
        file_adapter=FileProcessingAdapter(),
        chunking_port=chunker,
        geometric_expander=DependencyIdsGeometricExpander(citem_store=citem_adapter, embedding_port=tei_adapter, bridge_threshold=settings.bridge_threshold, backward_max=settings.geo_backward_max),
        reranker_adapter=reranker_adapter,
        nli_adapter=nli_adapter,
        colbert_adapter=colbert_adapter,
        event_bus=event_bus,
        domain_event_bus=domain_event_bus,
        geometry_reader=geometry_reader,
        geometry_commands=geometry_commands,
        ephemeral_runtime=ephemeral_runtime,
        artifacts_path=artifacts_path,
        workspace_path=workspace_path,
        forget_worker_factory=lambda memory_service, db, lifecycle_audit_service: ForgetCycleWorker(
            memory_service=memory_service,
            rel_db=db,
            interval_secs=settings.forget_cycle_interval_secs,
            stale_hours=settings.forget_cycle_stale_hours,
            audit_service=lifecycle_audit_service,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_tracing()
    start_metrics_server(port=8001)
    workspace_path = Path(settings.workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    artifacts_path = Path(settings.demo_artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    if settings.runtime_mode == "standalone":
        workspace_cleaner, forget_worker, extra = await _bootstrap_standalone_runtime(app, settings, workspace_path, artifacts_path)
    else:
        workspace_cleaner, forget_worker, extra = await _bootstrap_full_runtime(app, settings, workspace_path, artifacts_path)

    if forget_worker is not None:
        forget_worker.start()
    workspace_cleaner.start()
    log.info("CIMA Demonstrator application started in %s mode", settings.runtime_mode)
    yield
    if forget_worker is not None:
        await forget_worker.stop()
    await workspace_cleaner.stop()
    if extra is not None and hasattr(extra, 'stop'):
        await extra.stop()
    log.info("CIMA Demonstrator application stopped")


def create_app() -> FastAPI:
    return FastAPI(title=PUBLIC_PROJECT_NAME, version="0.1.0", lifespan=lifespan, docs_url=PUBLIC_DOCS_PATH, redoc_url=None)


app = create_app()
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(conversations.cima_v1_router)
app.include_router(context.router)
app.include_router(context.cima_v1_router)
app.include_router(runs.router)
app.include_router(handoff.router)
app.include_router(sources.router)
app.include_router(rag.router)
app.include_router(health.router)
