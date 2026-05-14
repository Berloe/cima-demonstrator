"""StreamManager — wraps EventBusPort for typed delta publishing."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from cima_demo.domain.entities import KimaDelta
from cima_demo.domain.ports import EventBusPort


class StreamManager:
    """Thin wrapper over EventBusPort.

    Provides typed publish() and subscribe() for AgentOrchestrator and API layer.
    API-INV-01: subscribe() must be called BEFORE publish() (before create_task).
    """

    def __init__(self, event_bus: EventBusPort) -> None:
        self._bus = event_bus

    async def publish(self, delta: KimaDelta) -> None:
        await self._bus.publish(delta)

    def subscribe(self, conversation_id: str) -> AsyncGenerator[KimaDelta, None]:
        """Return async iterator of KimaDeltas for conversation_id."""
        return self._bus.subscribe(conversation_id)
