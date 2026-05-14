from __future__ import annotations

"""Hard-delete scheduling and consumption for the witness-backend async plane.

This tranche introduces the minimal durable two-step protocol for full runtime:
- API marks the conversation as DELETING and writes a hard-delete request event
  to the CIMA outbox.
- A dedicated worker consumes conversation events, performs the purge and writes
  the completion event back to the outbox.

The design is intentionally incremental: it aligns the live runtime with the
approved async plane without forcing the full witness-backend table migration in
one step.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    ConversationHardDeleteCompletedData,
    ConversationHardDeleteCompletedStats,
    ConversationHardDeleteRequestedData,
    EventType,
    Producer,
    VectorDeletedData,
    VectorMeta,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


class HardDeleteSchedulerStore(Protocol):
    async def begin_hard_delete(self, conversation_id: str, *, delete_run_id: str) -> bool: ...

    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any],
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class HardDeleteRequestResult:
    conversation_id: str
    delete_run_id: str
    accepted: bool


class HardDeleteScheduler:
    def __init__(self, store: HardDeleteSchedulerStore, *, producer: Producer = Producer.CIMA_API) -> None:
        self._store = store
        self._producer = producer

    async def request(self, conversation_id: str, *, reason: str = "USER_REQUEST") -> HardDeleteRequestResult:
        delete_run_id = str(uuid4())
        accepted = await self._store.begin_hard_delete(conversation_id, delete_run_id=delete_run_id)
        if not accepted:
            return HardDeleteRequestResult(conversation_id=conversation_id, delete_run_id=delete_run_id, accepted=False)
        envelope = CloudEventEnvelope(
            type=EventType.CONVERSATION_HARD_DELETE_REQUESTED,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.conversation.hard_delete.requested.v1.json",
            time=datetime.now(UTC),
            data=ConversationHardDeleteRequestedData(
                delete_run_id=UUID(delete_run_id),
                mode="HARD",
                reason=reason if reason in {"USER_REQUEST", "RETENTION_POLICY"} else "USER_REQUEST",
            ).model_dump(mode="json"),
        )
        await self._store.append_outbox_event(
            topic=TOPICS.conversation_events,
            message_key=conversation_key(conversation_id),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )
        return HardDeleteRequestResult(conversation_id=conversation_id, delete_run_id=delete_run_id, accepted=True)


class HardDeleteStore(Protocol):
    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None: ...

    async def delete_conversation(self, conversation_id: str) -> None: ...

    async def list_ephemeral_vector_records(self, *, conversation_id: str | None = None, lifecycle_state: str | None = None) -> list[dict[str, Any]]: ...

    async def mark_ephemeral_vector_purged(self, ephemeral_id: str, *, purged_at: str | None = None) -> None: ...

    async def mark_hard_delete_completed(self, *, delete_run_id: str, stats_json: dict[str, Any] | None = None) -> None: ...

    async def mark_hard_delete_failed(self, *, delete_run_id: str, stats_json: dict[str, Any] | None = None) -> None: ...

    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any],
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...


class CItemDeletePort(Protocol):
    async def delete_by_conversation(self, conversation_id: str) -> int: ...


class GeometryCommandsPort(Protocol):
    async def purge_conversation(self, conversation_id: str, *, delete_run_id: str | None = None) -> None: ...


class HardDeleteConsumer:
    def __init__(
        self,
        *,
        store: HardDeleteStore,
        citem_store: CItemDeletePort,
        ledger: ConsumerEffectLedger,
        geometry_commands: GeometryCommandsPort | None = None,
    ) -> None:
        self._store = store
        self._citem_store = citem_store
        self._ledger = ledger
        self._geometry_commands = geometry_commands
        self._consumer_name = "conversation-hard-delete-worker"

    async def handle(self, *, payload_json: dict[str, Any] | None) -> None:
        if payload_json is None:
            return
        envelope = CloudEventEnvelope.model_validate(payload_json)
        if envelope.type != EventType.CONVERSATION_HARD_DELETE_REQUESTED:
            raise ValueError(f"Unsupported event type for hard delete worker: {envelope.type}")
        delete_run_id = str(envelope.data["delete_run_id"])
        key = ConsumerEffectKey(
            consumer_name=self._consumer_name,
            event_id=str(envelope.id),
            effect_key=f"{envelope.subject}|{delete_run_id}",
        )
        if not await self._ledger.begin(key):
            return
        try:
            stats = await self._purge_conversation(envelope.subject, delete_run_id=delete_run_id)
            await self._store.mark_hard_delete_completed(delete_run_id=delete_run_id, stats_json=stats)
            completed = CloudEventEnvelope(
                type=EventType.CONVERSATION_HARD_DELETE_COMPLETED,
                source=Producer.CIMA_WORKER,
                subject=envelope.subject,
                dataschema="schemas/cima.conversation.hard_delete.completed.v1.json",
                time=datetime.now(UTC),
                data=ConversationHardDeleteCompletedData(
                    delete_run_id=UUID(delete_run_id),
                    stats=ConversationHardDeleteCompletedStats(**stats),
                ).model_dump(mode="json"),
            )
            await self._store.append_outbox_event(
                topic=TOPICS.conversation_events,
                message_key=conversation_key(envelope.subject),
                headers_json={"ce_type": completed.type.value, "ce_source": completed.source.value},
                payload_json=completed.model_dump(mode="json"),
            )
            await self._ledger.complete(key, details_json={"delete_run_id": delete_run_id, **stats})
        except Exception as exc:
            await self._store.mark_hard_delete_failed(
                delete_run_id=delete_run_id,
                stats_json={"error_class": type(exc).__name__, "message": str(exc)},
            )
            raise

    async def _purge_conversation(self, conversation_id: str, *, delete_run_id: str) -> dict[str, int]:
        row = await self._store.get_conversation(conversation_id)
        if row is None:
            return {
                "postgres_rows_deleted": 0,
                "qdrant_points_deleted": 0,
                "blob_bytes_deleted": 0,
                "ephemeral_records_purged": 0,
                "geometry_purge_requested": 0,
                "qdrant_points_deleted_by_collection": {},
            }
        ephemeral_stats = await self._purge_ephemeral_lifecycle(conversation_id)
        qdrant_collection_counts = await self._local_qdrant_point_counts(conversation_id)
        qdrant_deleted = int(sum(qdrant_collection_counts.values()))
        await self._request_geometry_purge(conversation_id, delete_run_id=delete_run_id)
        await self._citem_store.delete_by_conversation(conversation_id)
        await self._store.delete_conversation(conversation_id)
        return {
            "postgres_rows_deleted": 1,
            "qdrant_points_deleted": qdrant_deleted + int(ephemeral_stats["ephemeral_records_purged"]),
            "blob_bytes_deleted": 0,
            "ephemeral_records_purged": int(ephemeral_stats["ephemeral_records_purged"]),
            "geometry_purge_requested": 1 if self._geometry_commands is not None else 0,
            "qdrant_points_deleted_by_collection": qdrant_collection_counts,
        }

    async def _purge_ephemeral_lifecycle(self, conversation_id: str) -> dict[str, int]:
        rows = await self._store.list_ephemeral_vector_records(conversation_id=conversation_id)
        if not rows:
            return {"ephemeral_records_purged": 0}
        purged_at = datetime.now(UTC).isoformat()
        purged = 0
        for row in rows:
            ephemeral_id = str(row.get("ephemeral_id") or "")
            if not ephemeral_id:
                continue
            await self._store.mark_ephemeral_vector_purged(ephemeral_id, purged_at=purged_at)
            purged += 1
            await self._emit_vector_deleted(
                conversation_id=conversation_id,
                ref_kind="ephemeral",
                ref_id=ephemeral_id,
                collection=str(row.get("qdrant_collection") or ""),
                scope=str((row.get("meta_json") or {}).get("scope") or "local"),
                item_type=(row.get("meta_json") or {}).get("type"),
                reason="HARD_DELETE",
            )
        return {"ephemeral_records_purged": purged}

    async def _local_qdrant_point_counts(self, conversation_id: str) -> dict[str, int]:
        catalog = getattr(self._citem_store, "catalog", None)
        list_points = getattr(self._citem_store, "list_point_ids_by_conversation", None)
        if catalog is None or list_points is None:
            deleted = int(await self._citem_store.delete_by_conversation(conversation_id) or 0)
            return {"local_scoped": deleted} if deleted else {}
        counts: dict[str, int] = {}
        for collection_name in (catalog.local_citems, catalog.local_summaries, catalog.chunks):
            point_ids = await list_points(collection_name=collection_name, conversation_id=conversation_id)
            if point_ids:
                counts[str(collection_name)] = len(point_ids)
        return counts

    async def _request_geometry_purge(self, conversation_id: str, *, delete_run_id: str) -> None:
        if self._geometry_commands is None:
            return
        await self._geometry_commands.purge_conversation(conversation_id, delete_run_id=delete_run_id)

    async def _emit_vector_deleted(
        self,
        *,
        conversation_id: str,
        ref_kind: str,
        ref_id: str,
        collection: str,
        scope: str,
        item_type: str | None,
        reason: str,
    ) -> None:
        envelope = CloudEventEnvelope(
            type=EventType.VECTOR_DELETED,
            source=Producer.CIMA_WORKER,
            subject=conversation_id,
            dataschema="schemas/cima.vector.deleted.v1.json",
            time=datetime.now(UTC),
            data=VectorDeletedData(
                ref_kind=ref_kind,  # type: ignore[arg-type]
                ref_id=UUID(ref_id),
                qdrant_collection=collection,
                reason=reason,  # type: ignore[arg-type]
                meta=VectorMeta(scope="global" if scope == "global" else "local", type=item_type),
            ).model_dump(mode="json"),
        )
        await self._store.append_outbox_event(
            topic=TOPICS.vector_events,
            message_key=conversation_key(conversation_id),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )
