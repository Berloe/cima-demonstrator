"""Puertos del bounded context de orquestación (ADR-001 v3.4).

DomainEventPublisherPort: publicación del stream autoritativo de eventos de dominio.
  Fase 1: InProcessDomainEventBus (cima_demo event bus implementation).
  Fase 5: KafkaEventBus.

ContextBuilderPort, MemoryServicePort, StreamManagerPort, ToolRegistryPort,
PlanExecutorPort: puertos de los servicios de aplicación que el kernel usa.
  Permiten que OrchestrationEngine sea independiente de las implementaciones
  concretas en cima_demo/application/ (ADR-001 §D-12, Fase 1).

Separado de EventBusPort (cima_demo/domain/ports.py), que publica KimaDeltas para SSE.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from cima_demo.cognitive.kernel.events import DomainEvent

if TYPE_CHECKING:
    from cima_demo.domain.entities import CItem, ContextView, IngestRequest, KimaDelta, Plan, TaskMemory
    from cima_demo.domain.value_objects import ContextBudget


# ── DomainEventPublisherPort ──────────────────────────────────────────────────

class DomainEventPublisherPort(Protocol):
    """Publica eventos de dominio autoritativos al bus externo.

    Contrato:
      - publish_batch devuelve los event_ids confirmados.
      - Los eventos no confirmados deben permanecer en el outbox del caller.
      - Implementaciones no deben lanzar excepciones; deben retornar lista vacía
        si la publicación falla completamente.
    """

    async def publish_batch(self, events: list[DomainEvent]) -> list[str]:
        """Publica un batch. Retorna event_ids entregados con confirmación."""
        ...

    async def publish(self, event: DomainEvent) -> bool:
        """Publica un único evento. Retorna True si confirmado."""
        ...


# Tipo para handlers de suscripción (usado por InProcessDomainEventBus)
DomainEventHandler = Callable[[DomainEvent], Awaitable[None]]


# ── ContextBuilderPort ────────────────────────────────────────────────────────

class ContextBuilderPort(Protocol):
    """Construye ContextView para inyección en el prompt LLM.

    Abstrae ContextBuilder (cima_demo/retrieval/context_builder.py).
    """

    async def build(
        self,
        phase: str,
        task_memory: TaskMemory,
        plan: Plan | None,
        query: str,
        conversation_id: str,
        budget: ContextBudget,
        history_contents: set[str] | None = None,
        global_objective: str = "",
        local_objective: str = "",
        exclude_ids: set[str] | None = None,
    ) -> ContextView:
        """Build ContextView for current turn iteration."""
        ...


# ── MemoryServicePort ─────────────────────────────────────────────────────────

class MemoryServicePort(Protocol):
    """Ingesta y actualización de C-Items en el plano K.

    Abstrae los métodos de MemoryService (cima_demo/memory/service.py)
    que usa OrchestrationEngine directamente durante el loop cognitivo.
    """

    async def ingest_citem(
        self,
        request: IngestRequest,
        skip_conflict_detection: bool = False,
    ) -> CItem | None:
        """Embed + upsert a C-Item. Returns None on duplicate."""
        ...

    async def ingest_batch(
        self,
        conclusions: list[dict[str, Any]],
        phase: str,
        conversation_id: str,
        turn_id: str,
    ) -> None:
        """Ingest multiple conclusions from LLM <conclusions> section."""
        ...

    async def ingest_web_content(
        self,
        url: str,
        text: str,
        title: str,
        conversation_id: str,
        phase: str,
        objective: str | None = None,
    ) -> Any:
        """Chunk and ingest fetched web content. Returns WebIngestionResult."""
        ...

    async def refresh_context(
        self,
        context_view: ContextView,
        task_memory: TaskMemory,
        conversation_id: str,
        current_goal: str | None = None,
        active_step: str | None = None,
        phase: str | None = None,
        semantic: bool = False,
        force_ids: set[str] | None = None,
    ) -> tuple[ContextView, int, str]:
        """Compress active C-Items via L1 or L2 refresh."""
        ...


# ── StreamManagerPort ─────────────────────────────────────────────────────────

class StreamManagerPort(Protocol):
    """Publica KimaDeltas al bus de eventos SSE.

    Abstrae StreamManager (cima_demo/application/stream_manager.py).
    """

    async def publish(self, delta: KimaDelta) -> None:
        """Publish one delta to the SSE event bus."""
        ...


# ── ToolRegistryPort ──────────────────────────────────────────────────────────

class ToolRegistryPort(Protocol):
    """Despacha llamadas de herramienta del LLM.

    Abstrae ToolRegistry (cima_demo/tools/registry.py).
    Any used for ToolCall/ToolResult to avoid importing application/tools
    from orchestration/domain (wrong direction). Engine imports the concrete
    types separately for runtime use.
    """

    def get_definitions(self, include_external: bool = True) -> list[dict[str, Any]]:
        """Return tool definitions for LLM system prompt injection."""
        ...

    def get_definitions_for_mode(self, mode: Any) -> list[dict[str, Any]]:
        """Return tool definitions filtered by ExecutionMode allowlist."""
        ...

    def mode_allows_web(self, mode: Any) -> bool:
        """True when the given ExecutionMode permits the web tool."""
        ...

    async def dispatch(self, tool_call: Any, conversation_id: str) -> Any:
        """Route ToolCall to its handler. Returns ToolResult."""
        ...


# ── PlanExecutorPort ──────────────────────────────────────────────────────────

class PlanExecutorPort(Protocol):
    """Gestiona el ciclo de vida de los planes de ejecución.

    Abstrae PlanExecutor (cima_demo/application/plan_executor.py).
    Solo los 4 métodos que OrchestrationEngine usa directamente.
    """

    async def create_plan(
        self,
        goal: str,
        steps: list[dict[str, Any]],
        conversation_id: str,
        *,
        auto_continue: bool = False,
    ) -> Plan:
        """Create a new Plan with steps and persist it."""
        ...

    async def start(self, plan: Plan, task_memory: TaskMemory) -> Plan:
        """Activate first step and set plan to RUNNING."""
        ...

    async def advance_step(
        self,
        plan: Plan,
        result_summary: str,
        task_memory: TaskMemory,
        conversation_id: str,
        turn_id: str,
    ) -> str:
        """Mark current step completed and activate next. Returns PlanStatus."""
        ...

    async def pause(self, plan: Plan, task_memory: TaskMemory) -> None:
        """Pause plan when ask_user interrupts."""
        ...
