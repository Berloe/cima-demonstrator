"""DirectSSEEventBus -> EventBusPort (Phase 1 - in-process asyncio.Queue)."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from cima_demo.domain.entities import KimaDelta
from cima_demo.domain.ports import EventBusPort
from cima_demo.domain.value_objects import KimaDeltaType

log = logging.getLogger(__name__)


class DirectSSEEventBus(EventBusPort):
    """In-process SSE broker using asyncio.Queue per conversation.

    Phase 1: single-process FastAPI. No persistence, no replay.
    subscribe() must be called BEFORE publish() (API-INV-01).

    Queues are created ONLY in subscribe() and removed on disconnect.
    publish() silently discards deltas when no subscriber is active -
    this prevents orphaned queues from filling up when the orchestrator
    task outlives a disconnected HTTP client.

    Terminal delivery invariant:
    DONE must never be lost because API stream generators use it as the
    normal close signal. If the queue is full, older buffered deltas may be
    evicted so DONE and its sentinel can be enqueued.
    """

    def __init__(self, *, queue_maxsize: int = 4096) -> None:
        self._queue_maxsize = max(1, int(queue_maxsize))
        self._queues: dict[str, asyncio.Queue[KimaDelta | None]] = {}

    async def publish(self, delta: KimaDelta) -> None:
        q = self._queues.get(delta.conversation_id)
        if q is None:
            # No active subscriber - orchestrator task outlived the SSE connection.
            return

        # DONE is structural, not telemetry. It must reach the subscriber even
        # when a burst of TOKEN deltas has filled the queue.
        if delta.type == KimaDeltaType.DONE:
            self._force_put(q, delta, conversation_id=delta.conversation_id, label="DONE")
            # Best-effort sentinel for generic consumers. Routers also close when
            # they see the DONE delta itself, so the sentinel must not evict ERROR
            # or DONE when the queue is already saturated.
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
            return

        # ERROR is also terminal-ish from the client's point of view. Preserve it
        # over older buffered deltas, but do not add the None sentinel; the router
        # still drains until DONE when possible.
        if delta.type == KimaDeltaType.ERROR:
            self._force_put(q, delta, conversation_id=delta.conversation_id, label="ERROR")
            return

        try:
            q.put_nowait(delta)
            return
        except asyncio.QueueFull:
            pass

        # Visible output should normally not be dropped. Apply bounded
        # backpressure first; if the client has gone away or stopped draining,
        # the HTTP generator will cancel the orchestrator task.
        if delta.type in {KimaDeltaType.TOKEN, KimaDeltaType.REASONING}:
            try:
                await asyncio.wait_for(q.put(delta), timeout=5.0)
                return
            except asyncio.TimeoutError:
                log.warning(
                    "SSE queue full for conversation %s - timed out preserving %s delta; dropping it",
                    delta.conversation_id, delta.type,
                )
                return

        log.warning(
            "SSE queue full for conversation %s - dropping %s delta",
            delta.conversation_id, delta.type,
        )

    def _force_put(
        self,
        q: asyncio.Queue[KimaDelta | None],
        item: KimaDelta | None,
        *,
        conversation_id: str,
        label: str,
    ) -> None:
        """Put *item* even if the queue is full by evicting older non-terminal items."""
        max_attempts = max(1, q.maxsize or self._queue_maxsize) + 1
        evicted = 0
        for _ in range(max_attempts):
            try:
                q.put_nowait(item)
                if evicted:
                    log.warning(
                        "SSE queue full for conversation %s - evicted %d buffered delta(s) to deliver %s",
                        conversation_id, evicted, label,
                    )
                return
            except asyncio.QueueFull:
                if not self._evict_one_non_terminal(q):
                    break
                evicted += 1
        log.warning(
            "SSE queue full for conversation %s - could not deliver %s after evicting %d delta(s)",
            conversation_id, label, evicted,
        )

    def _evict_one_non_terminal(self, q: asyncio.Queue[KimaDelta | None]) -> bool:
        """Remove one buffered non-terminal item while preserving ERROR/DONE/None."""
        preserved: list[KimaDelta | None] = []
        removed = False
        try:
            while True:
                item = q.get_nowait()
                if self._is_terminal_item(item):
                    preserved.append(item)
                    continue
                removed = True
                break
        except asyncio.QueueEmpty:
            pass

        # Restore protected terminal items before returning. If one non-terminal
        # was removed, there is still exactly one free slot for the caller's item.
        for item in preserved:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                break
        return removed

    @staticmethod
    def _is_terminal_item(item: KimaDelta | None) -> bool:
        if item is None:
            return True
        return item.type in {KimaDeltaType.ERROR, KimaDeltaType.DONE}

    def subscribe(self, conversation_id: str) -> AsyncGenerator[KimaDelta, None]:
        # Important: create/register the queue synchronously in subscribe().
        # Async-generator bodies do not execute until the first __anext__(), so
        # creating the queue inside the generator body would still allow early
        # publishes to be lost despite callers subscribing before create_task().
        q: asyncio.Queue[KimaDelta | None] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._queues[conversation_id] = q

        async def _iter() -> AsyncGenerator[KimaDelta, None]:
            try:
                while True:
                    item = await q.get()
                    if item is None:
                        return
                    yield item
                    if item.type == KimaDeltaType.DONE:
                        return
            finally:
                if self._queues.get(conversation_id) is q:
                    self._queues.pop(conversation_id, None)

        return _iter()
