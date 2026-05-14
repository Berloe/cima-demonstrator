"""InProcessDomainEventBus — implementación Fase 1 de DomainEventPublisherPort.

Bus in-process basado en callbacks async registrados.
Sin persistencia, sin replay. Equivalente funcional a DirectSSEEventBus
pero para eventos de dominio autoritativos (no KimaDeltas).

Diseño:
  - Handlers se registran vía subscribe().
  - publish/publish_batch llama a todos los handlers en orden de registro.
  - Un handler que lanza no bloquea a los demás; se loguea como debug.
  - publish_batch retorna todos los event_ids (best-effort: siempre confirmados
    en el bus in-process, independientemente de si algún handler falló).
"""
from __future__ import annotations

import asyncio
import logging

from cima_demo.cognitive.kernel.events import DomainEvent
from cima_demo.cognitive.kernel.ports import DomainEventHandler

log = logging.getLogger(__name__)


class InProcessDomainEventBus:
    """DomainEventPublisherPort in-process para Fase 1.

    Thread-safety: no requerida — todos los callers son corrutinas en el
    mismo event loop (FastAPI single-process).
    """

    def __init__(self) -> None:
        self._handlers: list[DomainEventHandler] = []

    def subscribe(self, handler: DomainEventHandler) -> None:
        """Registra un handler async. Se llama una vez en el wiring (lifespan)."""
        self._handlers.append(handler)

    async def publish(self, event: DomainEvent) -> bool:
        """Publica un evento a todos los handlers. Siempre retorna True."""
        await self._dispatch(event)
        return True

    async def publish_batch(self, events: list[DomainEvent]) -> list[str]:
        """Publica un batch. Retorna todos los event_ids (best-effort)."""
        tasks = [self._dispatch(e) for e in events]
        await asyncio.gather(*tasks, return_exceptions=True)
        return [e.event_id for e in events]

    async def _dispatch(self, event: DomainEvent) -> None:
        for handler in self._handlers:
            try:
                await handler(event)
            except Exception:
                log.debug(
                    "domain event handler failed (non-fatal) event_type=%s event_id=%s",
                    event.event_type, event.event_id,
                )
