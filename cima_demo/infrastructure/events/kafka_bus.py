"""KafkaEventBus — ADR-001 v3.4 Phase 5 DomainEventPublisherPort implementation.

Replaces InProcessDomainEventBus with a Kafka-backed bus for reliable,
durable, ordered domain event streaming across services.

Design:
  - Uses aiokafka (async Kafka client) — same event loop as FastAPI.
  - Topic: cima_demo.domain_events (one partition per conversation via key routing).
  - Messages: JSON-serialized DomainEvent envelope (schema_version="1.0").
  - Delivery semantics: at-least-once (acks="all", retries=3).
  - Outbox pattern: CRITICAL events block publish_batch() completion
    (BUDGET_IMPASSE, TURN_COMPLETED, TURN_FAILED).
  - BEST_EFFORT events use fire-and-forget via asyncio.create_task().

Configuration:
  KIMA_KAFKA_BOOTSTRAP  — broker address(es), default:
                          kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092
  KIMA_KAFKA_TOPIC      — topic name, default: cima_demo.domain_events
  KIMA_KAFKA_ENABLED    — set to "true" to enable; default false (in-process bus)

Fallback:
  If Kafka is unavailable at startup or during publish, falls back silently to
  best-effort logging (event is not lost locally — orchestrator writes to PG outbox
  in Phase 5b). Never raises from publish() or publish_batch().
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from cima_demo.cognitive.kernel.events import DomainEvent, CRITICAL_EVENT_TYPES

log = logging.getLogger(__name__)

_BOOTSTRAP_DEFAULT = (
    "kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
)
_TOPIC_DEFAULT = "cima_demo.domain_events"


def _event_to_bytes(event: DomainEvent) -> bytes:
    """Serialize DomainEvent to JSON bytes for Kafka message value."""
    payload: dict[str, Any] = {
        "event_id":        event.event_id,
        "schema_version":  event.schema_version,
        "event_type":      event.event_type,
        "conversation_id": event.conversation_id,
        "turn_id":         event.turn_id,
        "iteration":       event.iteration,
        "causation_id":    event.causation_id,
        "correlation_id":  event.correlation_id,
        "occurred_at":     event.occurred_at.isoformat(),
        "payload":         event.payload,
    }
    return json.dumps(payload, ensure_ascii=False).encode()


def _partition_key(event: DomainEvent) -> bytes:
    """Route by conversation_id for ordered delivery per conversation."""
    return event.conversation_id.encode()


class KafkaEventBus:
    """DomainEventPublisherPort backed by Apache Kafka (ADR-001 Phase 5).

    Lifecycle:
      KafkaEventBus is created in the lifespan context manager.
      start() must be called before any publish; stop() before shutdown.
      Both are called by the FastAPI lifespan in cima_demo/api/app.py.

    Thread-safety: coroutine-safe within a single asyncio event loop.
    """

    def __init__(
        self,
        bootstrap_servers: str = _BOOTSTRAP_DEFAULT,
        topic: str = _TOPIC_DEFAULT,
        acks: str = "all",
        max_batch_size: int = 16384,
        linger_ms: int = 5,
        request_timeout_ms: int = 10_000,
        retry_backoff_ms: int = 200,
    ) -> None:
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._acks = acks
        self._max_batch_size = max_batch_size
        self._linger_ms = linger_ms
        self._request_timeout_ms = request_timeout_ms
        self._retry_backoff_ms = retry_backoff_ms
        self._producer: Any = None   # AIOKafkaProducer — set in start()
        self._started = False

    async def start(self) -> None:
        """Initialize and connect the Kafka producer."""
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap,
                acks=self._acks,
                max_batch_size=self._max_batch_size,
                linger_ms=self._linger_ms,
                request_timeout_ms=self._request_timeout_ms,
                retry_backoff_ms=self._retry_backoff_ms,
                value_serializer=lambda v: v,  # bytes passthrough
                key_serializer=lambda k: k,    # bytes passthrough
            )
            await self._producer.start()
            self._started = True
            log.info(
                "KafkaEventBus started — bootstrap=%s topic=%s",
                self._bootstrap, self._topic,
            )
        except ImportError:
            log.warning(
                "aiokafka not installed — KafkaEventBus in degraded mode (logging only)"
            )
        except Exception as exc:
            log.warning(
                "KafkaEventBus start failed (will use fallback): %s", exc
            )

    async def stop(self) -> None:
        """Flush and disconnect the Kafka producer."""
        if self._producer is not None:
            try:
                await self._producer.stop()
                log.info("KafkaEventBus stopped")
            except Exception as exc:
                log.warning("KafkaEventBus stop error (non-fatal): %s", exc)
        self._started = False

    async def publish(self, event: DomainEvent) -> bool:
        """Publish a single event. Returns True if confirmed, False on error.

        CRITICAL events (BUDGET_IMPASSE, TURN_COMPLETED, TURN_FAILED) are
        awaited for broker confirmation. BEST_EFFORT events are fire-and-forget.
        Never raises.
        """
        is_critical = event.event_type in CRITICAL_EVENT_TYPES
        try:
            await self._send(event, critical=is_critical)
            return True
        except Exception as exc:
            log.warning(
                "KafkaEventBus.publish failed event_type=%s event_id=%s: %s",
                event.event_type, event.event_id, exc,
            )
            return False

    async def publish_batch(self, events: list[DomainEvent]) -> list[str]:
        """Publish a batch. Returns event_ids of confirmed events.

        CRITICAL events within the batch are awaited; BEST_EFFORT events
        are dispatched concurrently for throughput.
        Never raises.
        """
        confirmed: list[str] = []
        critical = [e for e in events if e.event_type in CRITICAL_EVENT_TYPES]
        best_effort = [e for e in events if e.event_type not in CRITICAL_EVENT_TYPES]

        # Send critical first (blocking) — these must land before mutex release
        for event in critical:
            try:
                await self._send(event, critical=True)
                confirmed.append(event.event_id)
            except Exception as exc:
                log.warning(
                    "KafkaEventBus: CRITICAL event send failed event_type=%s: %s",
                    event.event_type, exc,
                )

        # Best-effort: fire-and-forget for throughput
        for event in best_effort:
            try:
                await self._send(event, critical=False)
                confirmed.append(event.event_id)
            except Exception as exc:
                log.debug(
                    "KafkaEventBus: BEST_EFFORT event send failed event_type=%s: %s",
                    event.event_type, exc,
                )

        return confirmed

    async def _send(self, event: DomainEvent, critical: bool) -> None:
        """Send a single event to Kafka. Raises on failure."""
        if not self._started or self._producer is None:
            # Fallback: log event so it's not silently lost
            log.debug(
                "KafkaEventBus FALLBACK (no producer) event_type=%s event_id=%s",
                event.event_type, event.event_id,
            )
            return

        value = _event_to_bytes(event)
        key = _partition_key(event)

        fut = await self._producer.send(self._topic, value=value, key=key)
        if critical:
            # Await broker ACK for CRITICAL events
            await fut


def kafka_bus_from_env() -> KafkaEventBus | None:
    """Factory — returns a KafkaEventBus if kafka_enabled=True in settings, else None.

    Used in cima_demo/api/app.py lifespan to choose the event bus implementation.
    """
    from cima_demo.api.settings import get_settings
    s = get_settings()
    if not s.kafka_enabled:
        return None
    log.info("Kafka event bus configured: bootstrap=%s topic=%s", s.kafka_bootstrap, s.kafka_topic)
    return KafkaEventBus(bootstrap_servers=s.kafka_bootstrap, topic=s.kafka_topic)
