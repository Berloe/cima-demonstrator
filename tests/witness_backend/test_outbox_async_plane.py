from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType, Producer, SourceRegisteredData
from cima_demo.witness_backend.outbox import build_outbox_record
from cima_demo.witness_backend.publisher import OutboxPublisher
from cima_demo.witness_backend.topic_catalog import TOPICS, cleanup_policy_for, is_compacted_topic


class _FakeOutboxStore:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.sent: list[int] = []
        self.errors: list[tuple[int, str]] = []
        self.effects: set[tuple[str, str, str]] = set()

    async def claim_outbox_batch(self, limit: int = 100) -> list[dict]:
        return list(self.rows[:limit])

    async def mark_outbox_sent(self, outbox_ids: list[int]) -> None:
        self.sent.extend(outbox_ids)

    async def mark_outbox_error(self, outbox_id: int, error: str) -> None:
        self.errors.append((outbox_id, error))

    async def begin_consumer_effect(self, *, consumer_name: str, event_id: str, effect_key: str) -> bool:
        key = (consumer_name, event_id, effect_key)
        if key in self.effects:
            return False
        self.effects.add(key)
        return True

    async def complete_consumer_effect(self, *, consumer_name: str, event_id: str, effect_key: str, details_json: dict | None = None) -> None:
        return None


class _FakeProducer:
    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.sent: list[tuple[str, bytes, bytes | None, list[tuple[str, bytes]] | None]] = []

    async def send_and_wait(self, topic: str, value: bytes | None, *, key: bytes | None = None, headers: list[tuple[str, bytes]] | None = None):
        idx = len(self.sent) + 1
        if idx in self.fail_on:
            raise RuntimeError("boom")
        self.sent.append((topic, value, key, headers))
        return None


def test_build_outbox_record_preserves_cloudevent_headers() -> None:
    payload = SourceRegisteredData(source_id=uuid4(), kind="chat_user")
    envelope = CloudEventEnvelope(
        type=EventType.MEMORY_SOURCE_REGISTERED,
        source=Producer.CIMA_API,
        subject="conv-1",
        dataschema="schemas/cima.memory.source.registered.v1.json",
        time=datetime.now(UTC),
        data=payload.model_dump(mode="json"),
    )
    record = build_outbox_record(topic=TOPICS.memory_events, message_key="conv-1", envelope=envelope)
    assert record.topic == TOPICS.memory_events
    assert record.headers_json["ce_type"] == EventType.MEMORY_SOURCE_REGISTERED.value
    assert record.payload_json["subject"] == "conv-1"


@pytest.mark.asyncio
async def test_outbox_publisher_marks_sent_and_error_rows() -> None:
    rows = [
        {"outbox_id": 1, "topic": TOPICS.memory_events, "message_key": "conv-1", "headers_json": {}, "payload_json": {"hello": 1}},
        {"outbox_id": 2, "topic": TOPICS.memory_events, "message_key": "conv-1", "headers_json": {}, "payload_json": {"hello": 2}},
    ]
    store = _FakeOutboxStore(rows)
    producer = _FakeProducer(fail_on={2})
    publisher = OutboxPublisher(store=store, producer=producer)

    report = await publisher.publish_once(limit=10)

    assert report.claimed == 2
    assert report.sent == 1
    assert report.errored == 1
    assert store.sent == [1]
    assert store.errors == [(2, "RuntimeError")]


@pytest.mark.asyncio
async def test_consumer_effect_ledger_rejects_duplicate_effects() -> None:
    store = _FakeOutboxStore()
    ledger = ConsumerEffectLedger(store)
    key = ConsumerEffectKey("geom-projector", "evt-1", "conv-1|local_citem|x")

    assert await ledger.begin(key) is True
    assert await ledger.begin(key) is False
    await ledger.complete(key, details_json={"state": "ok"})


@pytest.mark.asyncio
async def test_outbox_publisher_supports_tombstone_values() -> None:
    rows = [
        {"outbox_id": 1, "topic": TOPICS.geom_item_state, "message_key": "conv-1|local_citem|x", "headers_json": {}, "payload_json": None},
    ]
    store = _FakeOutboxStore(rows)
    producer = _FakeProducer()
    publisher = OutboxPublisher(store=store, producer=producer)

    report = await publisher.publish_once(limit=10)

    assert report.sent == 1
    assert producer.sent[0][1] is None


def test_geometry_state_topics_are_marked_compacted() -> None:
    assert is_compacted_topic(TOPICS.geom_item_state) is True
    assert is_compacted_topic(TOPICS.geom_cluster_state) is True
    assert cleanup_policy_for(TOPICS.geom_item_state) == "compact,delete"
    assert cleanup_policy_for(TOPICS.memory_events) == "delete"


@pytest.mark.asyncio
async def test_outbox_publisher_rejects_tombstone_on_non_compacted_topic() -> None:
    rows = [
        {"outbox_id": 1, "topic": TOPICS.memory_events, "message_key": "conv-1", "headers_json": {}, "payload_json": None},
    ]
    store = _FakeOutboxStore(rows)
    producer = _FakeProducer()
    publisher = OutboxPublisher(store=store, producer=producer)

    report = await publisher.publish_once(limit=10)

    assert report.sent == 0
    assert report.errored == 1
    assert store.errors == [(1, "ValueError")]
    assert producer.sent == []
