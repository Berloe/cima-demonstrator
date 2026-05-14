from __future__ import annotations

"""Async summary plane for the witness backend.

This tranche closes the remaining semantic leg of the approved async pipeline:
local C-item batches trigger summary requests, summary requests materialise local
summary rows with explicit origins, and summary change events can then be indexed
into the witness Qdrant plane.

The implementation is intentionally deterministic and storage-aware so the
async plane can progress without depending on an in-request-path LLM call.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.lifecycle_guard import complete_if_conversation_not_active
from cima_demo.witness_backend.events import (
    CItemCreatedData,
    CloudEventEnvelope,
    EventType,
    Producer,
    SummaryChangedData,
    SummaryRequestedData,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


class SummaryStoreLike(Protocol):
    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...

    async def list_local_citem_records(
        self,
        conversation_id: str,
        *,
        citem_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def list_local_summary_records(
        self,
        conversation_id: str,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        cluster_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def save_local_summary_record(self, summary_json: dict[str, Any]) -> None: ...

    async def save_local_summary_origin(self, origin_json: dict[str, Any]) -> None: ...

    async def delete_local_summary_origins(self, local_summary_id: str) -> None: ...

    async def load_geometry_read_model_item_states(
        self,
        conversation_id: str,
        ref_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def load_geometry_item_states(
        self,
        conversation_id: str,
        ref_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...


class TokenCounterLike(Protocol):
    def count_text_tokens_sync(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class BuiltSummary:
    summary_id: str
    level: str
    cluster_id: str | None
    epoch_no: int | None
    text: str
    token_count: int
    origin_citem_ids: list[str]


_TYPE_ORDER = [
    "DECISION",
    "CONSTRAINT",
    "DEFINITION",
    "PLAN_STEP",
    "RISK",
    "FACT",
    "HEDGED_FACT",
    "CONTEXT",
    "ATTRIBUTION",
    "EVALUATION",
    "QUESTION",
    "CODE_ARTIFACT",
]


class DeterministicSummaryBuilder:
    def __init__(self, *, token_counter: TokenCounterLike) -> None:
        self._counter = token_counter

    def build(
        self,
        *,
        conversation_id: str,
        level: str,
        citem_rows: list[dict[str, Any]],
        cluster_id: str | None = None,
        epoch_no: int | None = None,
        summary_id: str,
    ) -> BuiltSummary:
        ordered = sorted(
            citem_rows,
            key=lambda row: (
                float(row.get("salience", 0.0)),
                str(row.get("created_at") or ""),
                str(row.get("local_citem_id") or ""),
            ),
            reverse=True,
        )
        grouped: dict[str, list[str]] = {key: [] for key in _TYPE_ORDER}
        for row in ordered:
            item_type = str(row.get("type") or "FACT")
            if item_type not in grouped:
                grouped[item_type] = []
            grouped[item_type].append(_one_line(row.get("text") or ""))

        lines: list[str] = []
        heading = {
            "EPOCH": f"Epoch summary #{epoch_no or 0}",
            "CLUSTER": f"Cluster summary {cluster_id or 'cluster'}",
            "MASTER": "Master conversation summary",
        }[level]
        lines.append(heading)
        for item_type in _TYPE_ORDER:
            bucket = [text for text in grouped.get(item_type, []) if text]
            if not bucket:
                continue
            chosen = bucket[:3]
            lines.append(f"{item_type}: " + " | ".join(chosen))
        if level == "MASTER":
            tail = [text for text in grouped.get("QUESTION", []) if text][:2]
            if tail:
                lines.append("OPEN: " + " | ".join(tail))
        text = "\n".join(lines).strip()
        return BuiltSummary(
            summary_id=summary_id,
            level=level,
            cluster_id=cluster_id,
            epoch_no=epoch_no,
            text=text,
            token_count=max(1, self._counter.count_text_tokens_sync(text)),
            origin_citem_ids=[str(row["local_citem_id"]) for row in ordered if row.get("local_citem_id")],
        )


def _one_line(text: str) -> str:
    compact = " ".join(str(text).strip().split())
    if len(compact) <= 180:
        return compact
    return compact[:177].rstrip() + "..."


def _deterministic_event_id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts))


class MemorySummaryConsumer:
    def __init__(
        self,
        *,
        db: SummaryStoreLike,
        ledger: ConsumerEffectLedger,
        tokenizer: TokenCounterLike,
        producer: Producer = Producer.CIMA_WORKER,
    ) -> None:
        self._db = db
        self._ledger = ledger
        self._tokenizer = tokenizer
        self._producer = producer
        self._builder = DeterministicSummaryBuilder(token_counter=tokenizer)

    async def handle(self, payload_json: dict[str, Any]) -> None:
        envelope = CloudEventEnvelope.model_validate(payload_json)
        if envelope.type == EventType.MEMORY_CITEM_CREATED:
            await self._handle_citem_created(envelope)
            return
        if envelope.type == EventType.SUMMARY_REQUESTED:
            await self._handle_summary_requested(envelope)
            return

    async def _handle_citem_created(self, envelope: CloudEventEnvelope) -> None:
        data = CItemCreatedData.model_validate(envelope.data)
        effect_key = f"summary-schedule:{','.join(sorted(str(v) for v in data.citem_ids))}"
        key = ConsumerEffectKey("memory-summary-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        existing_epochs = await self._db.list_local_summary_records(envelope.subject, level="EPOCH")
        next_epoch = max([int(row.get("epoch_no") or 0) for row in existing_epochs] or [0]) + 1
        requests = [
            SummaryRequestedData(
                level="EPOCH",
                epoch_no=next_epoch,
                reason="EPOCH_CLOSED",
                priority="NORMAL",
                target_citem_ids=[uuid.UUID(str(v)) for v in data.citem_ids],
            ),
            SummaryRequestedData(
                level="MASTER",
                reason="PERIODIC",
                priority="NORMAL",
                target_citem_ids=[uuid.UUID(str(v)) for v in data.citem_ids],
            ),
        ]
        for req in requests:
            event_id = _deterministic_event_id(
                envelope.subject,
                EventType.SUMMARY_REQUESTED,
                req.level,
                req.cluster_id or "",
                str(req.epoch_no or 0),
                ",".join(sorted(str(v) for v in (req.target_citem_ids or []))),
            )
            outbox = CloudEventEnvelope(
                id=event_id,
                type=EventType.SUMMARY_REQUESTED,
                source=self._producer,
                subject=envelope.subject,
                dataschema="schemas/cima.summary.requested.v1.json",
                data=req.model_dump(mode="json"),
            )
            await self._db.append_outbox_event(
                topic=TOPICS.summary_cmd,
                message_key=conversation_key(envelope.subject),
                payload_json=outbox.model_dump(mode="json"),
            )
        await self._ledger.complete(key, details_json={"status": "scheduled", "request_count": len(requests)})

    async def _handle_summary_requested(self, envelope: CloudEventEnvelope) -> None:
        data = SummaryRequestedData.model_validate(envelope.data)
        effect_key = f"summary-build:{data.level}:{data.cluster_id or ''}:{data.epoch_no or 0}:{','.join(sorted(str(v) for v in (data.target_citem_ids or [])))}"
        key = ConsumerEffectKey("memory-summary-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        citem_rows = await self._resolve_citems(conversation_id=envelope.subject, request=data)
        if not citem_rows:
            await self._ledger.complete(key, details_json={"status": "no_targets", "level": data.level})
            return
        summary_id = _summary_id(conversation_id=envelope.subject, level=data.level, cluster_id=data.cluster_id, epoch_no=data.epoch_no)
        existing = await self._db.list_local_summary_records(envelope.subject, summary_ids=[summary_id])
        built = self._builder.build(
            conversation_id=envelope.subject,
            level=data.level,
            citem_rows=citem_rows,
            cluster_id=data.cluster_id,
            epoch_no=data.epoch_no,
            summary_id=summary_id,
        )
        now = datetime.now(UTC).isoformat()
        covers_json = {
            "origin_citem_ids": built.origin_citem_ids,
            "cluster_id": data.cluster_id,
            "epoch_no": data.epoch_no,
            "request_level": data.level,
        }
        await self._db.save_local_summary_record(
            {
                "local_summary_id": built.summary_id,
                "conversation_id": envelope.subject,
                "level": built.level,
                "cluster_id": built.cluster_id,
                "epoch_no": built.epoch_no,
                "text": built.text,
                "covers_json": covers_json,
                "created_at": existing[0].get("created_at") if existing else now,
                "updated_at": now,
                "vector_state": existing[0].get("vector_state", "NONE") if existing else "NONE",
            }
        )
        if existing:
            await self._db.delete_local_summary_origins(built.summary_id)
        for ordinal, citem_id in enumerate(built.origin_citem_ids):
            await self._db.save_local_summary_origin(
                {
                    "local_summary_id": built.summary_id,
                    "origin_kind": "local_citem",
                    "origin_id": citem_id,
                    "ordinal": ordinal,
                    "conversation_id": envelope.subject,
                }
            )
        changed_type = EventType.MEMORY_SUMMARY_UPDATED if existing else EventType.MEMORY_SUMMARY_CREATED
        changed = SummaryChangedData(
            summary_id=uuid.UUID(built.summary_id),
            level=built.level,  # type: ignore[arg-type]
            cluster_id=built.cluster_id,
            epoch_no=built.epoch_no,
        )
        event_id = _deterministic_event_id(envelope.subject, changed_type, built.summary_id, built.level)
        outbox = CloudEventEnvelope(
            id=event_id,
            type=changed_type,
            source=self._producer,
            subject=envelope.subject,
            dataschema=(
                "schemas/cima.memory.summary.updated.v1.json"
                if changed_type == EventType.MEMORY_SUMMARY_UPDATED
                else "schemas/cima.memory.summary.created.v1.json"
            ),
            data=changed.model_dump(mode="json"),
        )
        await self._db.append_outbox_event(
            topic=TOPICS.memory_events,
            message_key=conversation_key(envelope.subject),
            payload_json=outbox.model_dump(mode="json"),
        )
        await self._ledger.complete(
            key,
            details_json={"status": "created" if not existing else "updated", "summary_id": built.summary_id, "origin_count": len(built.origin_citem_ids)},
        )

    async def _resolve_citems(self, *, conversation_id: str, request: SummaryRequestedData) -> list[dict[str, Any]]:
        target_ids = [str(v) for v in (request.target_citem_ids or [])]
        if target_ids:
            rows = await self._db.list_local_citem_records(conversation_id, citem_ids=target_ids)
            if rows:
                return rows
        if request.level == "CLUSTER" and request.cluster_id:
            loader = getattr(self._db, "load_geometry_read_model_item_states", None) or getattr(self._db, "load_geometry_item_states")
            geom_rows = await loader(conversation_id)
            matched = [
                str(row["ref_id"])
                for row in geom_rows
                if row.get("ref_kind") == "local_citem"
                and (row.get("cluster_top1") == request.cluster_id or row.get("cluster_top2") == request.cluster_id)
            ]
            if matched:
                rows = await self._db.list_local_citem_records(conversation_id, citem_ids=matched)
                if rows:
                    return rows
        rows = await self._db.list_local_citem_records(conversation_id)
        if not rows:
            return []
        if request.level == "MASTER":
            return rows[: min(12, len(rows))]
        return rows[: min(8, len(rows))]


def _summary_id(*, conversation_id: str, level: str, cluster_id: str | None, epoch_no: int | None) -> str:
    if level == "MASTER":
        basis = f"{conversation_id}|MASTER"
    elif level == "CLUSTER":
        basis = f"{conversation_id}|CLUSTER|{cluster_id or ''}"
    else:
        basis = f"{conversation_id}|EPOCH|{epoch_no or 0}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, basis))
