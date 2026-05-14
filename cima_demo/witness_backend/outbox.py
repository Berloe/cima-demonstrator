from __future__ import annotations

"""Outbox record helpers for the witness-backend async plane."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cima_demo.witness_backend.events import CloudEventEnvelope


class OutboxStatus(StrEnum):
    NEW = "NEW"
    CLAIMED = "CLAIMED"
    SENT = "SENT"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    topic: str
    message_key: str
    headers_json: dict[str, Any]
    payload_json: dict[str, Any] | None


def build_outbox_record(*, topic: str, message_key: str, envelope: CloudEventEnvelope) -> OutboxRecord:
    payload = envelope.model_dump(mode="json")
    headers = {
        "ce_specversion": envelope.specversion,
        "ce_id": str(envelope.id),
        "ce_type": envelope.type.value,
        "ce_source": envelope.source.value,
        "ce_subject": envelope.subject,
        "ce_time": envelope.time.isoformat(),
        "content-type": envelope.datacontenttype,
    }
    return OutboxRecord(
        topic=topic,
        message_key=message_key,
        headers_json=headers,
        payload_json=payload,
    )


def build_outbox_tombstone_record(*, topic: str, message_key: str) -> OutboxRecord:
    return OutboxRecord(topic=topic, message_key=message_key, headers_json={}, payload_json=None)
