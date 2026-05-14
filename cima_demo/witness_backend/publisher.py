from __future__ import annotations

"""Internal outbox publisher for the witness-backend async plane."""

from dataclasses import dataclass
from typing import Any, Protocol

from cima_demo.witness_backend.topic_catalog import is_compacted_topic


class OutboxStore(Protocol):
    async def claim_outbox_batch(self, limit: int = 100) -> list[dict[str, Any]]: ...

    async def mark_outbox_sent(self, outbox_ids: list[int]) -> None: ...

    async def mark_outbox_error(self, outbox_id: int, error: str) -> None: ...


class KafkaProducerLike(Protocol):
    async def send_and_wait(
        self,
        topic: str,
        value: bytes | None,
        *,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class PublishReport:
    claimed: int
    sent: int
    errored: int


class OutboxPublisher:
    def __init__(self, *, store: OutboxStore, producer: KafkaProducerLike) -> None:
        self._store = store
        self._producer = producer

    async def publish_once(self, *, limit: int = 100) -> PublishReport:
        rows = await self._store.claim_outbox_batch(limit)
        sent_ids: list[int] = []
        errors = 0
        for row in rows:
            outbox_id = int(row["outbox_id"])
            try:
                headers = [
                    (str(k), str(v).encode("utf-8"))
                    for k, v in dict(row.get("headers_json") or {}).items()
                ]
                payload = row["payload_json"]
                if payload is None and not is_compacted_topic(str(row["topic"])):
                    raise ValueError(f"TombstoneNotAllowed:{row['topic']}")
                await self._producer.send_and_wait(
                    row["topic"],
                    _encode_json(payload),
                    key=str(row["message_key"]).encode("utf-8"),
                    headers=headers,
                )
                sent_ids.append(outbox_id)
            except Exception as exc:  # pragma: no cover - covered via tests with fake producer
                errors += 1
                await self._store.mark_outbox_error(outbox_id, type(exc).__name__)
        if sent_ids:
            await self._store.mark_outbox_sent(sent_ids)
        return PublishReport(claimed=len(rows), sent=len(sent_ids), errored=errors)


def _encode_json(payload: dict[str, Any] | None) -> bytes | None:
    if payload is None:
        return None
    import json

    return json.dumps(payload, ensure_ascii=False).encode("utf-8")
