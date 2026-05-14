from __future__ import annotations

"""Maintenance scheduling and GC actions for the witness-backend async plane.

This tranche closes part of the lifecycle gap left after the hard-delete path:
- durable maintenance_run records
- GC_THINNING_REQUESTED scheduling via outbox
- worker-side thinning that removes vectors from the approved local collections
  and marks witness rows as THINNED without destroying the semantic records
- GC_EPHEMERAL_EXPIRY_REQUESTED scheduling and execution against the dedicated
  ephemeral collection

The implementation is intentionally conservative: automatic thinning only acts
on local/chunk witness rows and never on promoted global memory.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    EventType,
    GcRequestedData,
    Producer,
    VectorDeletedData,
    VectorMeta,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


class MaintenanceSchedulerStore(Protocol):
    async def begin_maintenance_run(
        self,
        *,
        kind: str,
        conversation_id: str | None = None,
        maintenance_run_id: str,
    ) -> bool: ...

    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class MaintenanceRequestResult:
    maintenance_run_id: str
    kind: str
    conversation_id: str | None
    accepted: bool


class MaintenanceScheduler:
    def __init__(self, store: MaintenanceSchedulerStore, *, producer: Producer = Producer.CIMA_API) -> None:
        self._store = store
        self._producer = producer

    async def request_thinning(self, conversation_id: str) -> MaintenanceRequestResult:
        return await self._request(kind="THINNING", conversation_id=conversation_id)

    async def request_ephemeral_expiry(self) -> MaintenanceRequestResult:
        return await self._request(kind="EPHEMERAL_EXPIRY", conversation_id=None)

    async def request_reconcile(self, conversation_id: str | None = None) -> MaintenanceRequestResult:
        return await self._request(kind="RECONCILE", conversation_id=conversation_id)

    async def _request(self, *, kind: str, conversation_id: str | None) -> MaintenanceRequestResult:
        maintenance_run_id = str(uuid4())
        accepted = await self._store.begin_maintenance_run(
            kind=kind,
            conversation_id=conversation_id,
            maintenance_run_id=maintenance_run_id,
        )
        if not accepted:
            return MaintenanceRequestResult(
                maintenance_run_id=maintenance_run_id,
                kind=kind,
                conversation_id=conversation_id,
                accepted=False,
            )
        subject = conversation_id or "*"
        envelope = CloudEventEnvelope(
            type=(
                EventType.GC_THINNING_REQUESTED
                if kind == "THINNING"
                else EventType.GC_EPHEMERAL_EXPIRY_REQUESTED
                if kind == "EPHEMERAL_EXPIRY"
                else EventType.GC_RECONCILE_REQUESTED
            ),
            source=self._producer,
            subject=subject,
            dataschema="schemas/cima.gc.requested.v1.json",
            time=datetime.now(UTC),
            data=GcRequestedData(
                maintenance_run_id=UUID(maintenance_run_id),
                reason=kind,  # type: ignore[arg-type]
            ).model_dump(mode="json"),
        )
        await self._store.append_outbox_event(
            topic=TOPICS.gc_events,
            message_key=conversation_key(subject),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )
        return MaintenanceRequestResult(
            maintenance_run_id=maintenance_run_id,
            kind=kind,
            conversation_id=conversation_id,
            accepted=True,
        )


class MaintenanceStore(Protocol):
    async def list_conversations(self) -> list[dict[str, Any]]: ...

    async def mark_maintenance_run_running(self, *, maintenance_run_id: str) -> None: ...

    async def mark_maintenance_run_completed(self, *, maintenance_run_id: str, stats_json: dict[str, Any] | None = None) -> None: ...

    async def mark_maintenance_run_failed(self, *, maintenance_run_id: str, stats_json: dict[str, Any] | None = None) -> None: ...

    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...

    async def list_chunk_records(self, conversation_id: str, *, source_id: str | None = None) -> list[dict[str, Any]]: ...

    async def list_local_citem_records(self, conversation_id: str, *, citem_ids: list[str] | None = None) -> list[dict[str, Any]]: ...

    async def list_local_summary_records(
        self,
        conversation_id: str,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        cluster_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def list_global_citem_records(
        self,
        *,
        global_citem_ids: list[str] | None = None,
        semantic_identity_ids: list[str] | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def list_global_summary_records(
        self,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def update_chunk_vector_state(
        self,
        chunk_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def update_local_citem_vector_state(
        self,
        local_citem_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def update_local_summary_vector_state(
        self,
        local_summary_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def list_ephemeral_vector_records(
        self,
        *,
        conversation_id: str | None = None,
        lifecycle_state: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def list_due_ephemeral_vector_records(self, *, now: str | None = None) -> list[dict[str, Any]]: ...

    async def mark_ephemeral_vector_expired(self, ephemeral_id: str, *, expired_at: str | None = None) -> None: ...

    async def mark_ephemeral_vector_purged(self, ephemeral_id: str, *, purged_at: str | None = None) -> None: ...

    async def update_global_citem_vector_state(
        self,
        global_citem_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def update_global_summary_vector_state(
        self,
        global_summary_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...


class QdrantMaintenancePort(Protocol):
    catalog: QdrantCollectionCatalog

    async def delete_point_ids(self, *, collection_name: str, point_ids: list[str]) -> int: ...

    async def list_point_ids_by_conversation(self, *, collection_name: str, conversation_id: str) -> list[str]: ...

    async def list_all_point_ids(self, *, collection_name: str) -> list[str]: ...

    async def sweep_ephemeral_expired(self, *, now: datetime | None = None) -> int: ...


class MaintenanceConsumer:
    def __init__(
        self,
        *,
        store: MaintenanceStore,
        qdrant_plane: QdrantMaintenancePort,
        ledger: ConsumerEffectLedger,
        thinning_age_hours: int = 24,
    ) -> None:
        self._store = store
        self._qdrant = qdrant_plane
        self._ledger = ledger
        self._consumer_name = "gc-maintenance-worker"
        self._thinning_age = timedelta(hours=max(1, thinning_age_hours))

    async def handle(self, *, payload_json: dict[str, Any] | None) -> None:
        if payload_json is None:
            return
        envelope = CloudEventEnvelope.model_validate(payload_json)
        if envelope.type not in {EventType.GC_THINNING_REQUESTED, EventType.GC_EPHEMERAL_EXPIRY_REQUESTED, EventType.GC_RECONCILE_REQUESTED}:
            raise ValueError(f"Unsupported event type for maintenance worker: {envelope.type}")
        data = GcRequestedData.model_validate(envelope.data)
        key = ConsumerEffectKey(
            consumer_name=self._consumer_name,
            event_id=str(envelope.id),
            effect_key=f"{data.reason}|{envelope.subject}|{data.maintenance_run_id}",
        )
        if not await self._ledger.begin(key):
            return
        try:
            await self._store.mark_maintenance_run_running(maintenance_run_id=str(data.maintenance_run_id))
            if data.reason == "THINNING":
                stats = await self._handle_thinning(conversation_id=envelope.subject)
            elif data.reason == "EPHEMERAL_EXPIRY":
                stats = await self._handle_ephemeral_expiry()
            else:
                stats = await self._handle_reconcile(subject=envelope.subject)
            await self._store.mark_maintenance_run_completed(
                maintenance_run_id=str(data.maintenance_run_id),
                stats_json=stats,
            )
            await self._ledger.complete(key, details_json=stats)
        except Exception as exc:
            await self._store.mark_maintenance_run_failed(
                maintenance_run_id=str(data.maintenance_run_id),
                stats_json={"error_class": type(exc).__name__, "message": str(exc)},
            )
            raise

    async def _handle_thinning(self, *, conversation_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        stats = {
            "chunk_points_deleted": 0,
            "local_citem_points_deleted": 0,
            "local_summary_points_deleted": 0,
            "rows_marked_thinned": 0,
        }

        chunk_rows = [row for row in await self._store.list_chunk_records(conversation_id) if self._thinning_candidate(row, now=now)]
        citem_rows = [row for row in await self._store.list_local_citem_records(conversation_id) if self._thinning_candidate(row, now=now, pinned_field="is_pinned")]
        summary_rows = [row for row in await self._store.list_local_summary_records(conversation_id) if self._thinning_candidate(row, now=now, pinned_field="is_pinned")]

        chunk_ids = [str(row["chunk_id"]) for row in chunk_rows]
        citem_ids = [str(row["local_citem_id"]) for row in citem_rows]
        summary_ids = [str(row["local_summary_id"]) for row in summary_rows]

        stats["chunk_points_deleted"] = int(await self._qdrant.delete_point_ids(collection_name=self._qdrant.catalog.chunks, point_ids=chunk_ids) or 0)
        stats["local_citem_points_deleted"] = int(await self._qdrant.delete_point_ids(collection_name=self._qdrant.catalog.local_citems, point_ids=citem_ids) or 0)
        stats["local_summary_points_deleted"] = int(await self._qdrant.delete_point_ids(collection_name=self._qdrant.catalog.local_summaries, point_ids=summary_ids) or 0)

        for row in chunk_rows:
            await self._store.update_chunk_vector_state(str(row["chunk_id"]), vector_state="THINNED")
            stats["rows_marked_thinned"] += 1
            await self._emit_vector_deleted(
                conversation_id=conversation_id,
                ref_kind="chunk",
                ref_id=str(row["chunk_id"]),
                collection=self._qdrant.catalog.chunks,
                scope="local",
                item_type=row.get("chunk_kind"),
                reason="THINNING",
            )
        for row in citem_rows:
            await self._store.update_local_citem_vector_state(str(row["local_citem_id"]), vector_state="THINNED")
            stats["rows_marked_thinned"] += 1
            await self._emit_vector_deleted(
                conversation_id=conversation_id,
                ref_kind="local_citem",
                ref_id=str(row["local_citem_id"]),
                collection=self._qdrant.catalog.local_citems,
                scope="local",
                item_type=row.get("type"),
                reason="THINNING",
            )
        for row in summary_rows:
            await self._store.update_local_summary_vector_state(str(row["local_summary_id"]), vector_state="THINNED")
            stats["rows_marked_thinned"] += 1
            await self._emit_vector_deleted(
                conversation_id=conversation_id,
                ref_kind="local_summary",
                ref_id=str(row["local_summary_id"]),
                collection=self._qdrant.catalog.local_summaries,
                scope="local",
                item_type=row.get("level"),
                reason="THINNING",
            )
        return stats

    async def _handle_ephemeral_expiry(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        due_rows = await self._store.list_due_ephemeral_vector_records(now=now.isoformat())
        stats = {
            "ephemeral_due_records": len(due_rows),
            "ephemeral_records_marked_expired": 0,
            "ephemeral_records_marked_purged": 0,
            "ephemeral_points_deleted": 0,
        }
        if not due_rows:
            return stats

        point_ids = [str(row["ephemeral_id"]) for row in due_rows]
        for row in due_rows:
            if str(row.get("lifecycle_state") or "ACTIVE") == "ACTIVE":
                await self._store.mark_ephemeral_vector_expired(str(row["ephemeral_id"]), expired_at=now.isoformat())
                stats["ephemeral_records_marked_expired"] += 1
        stats["ephemeral_points_deleted"] = int(
            await self._qdrant.delete_point_ids(collection_name=self._qdrant.catalog.ephemeral, point_ids=point_ids) or 0
        )
        for row in due_rows:
            await self._store.mark_ephemeral_vector_purged(str(row["ephemeral_id"]), purged_at=now.isoformat())
            stats["ephemeral_records_marked_purged"] += 1
            meta_json = dict(row.get("meta_json") or {})
            await self._emit_vector_deleted(
                conversation_id=str(row.get("conversation_id") or "*"),
                ref_kind="ephemeral",
                ref_id=str(row["ephemeral_id"]),
                collection=self._qdrant.catalog.ephemeral,
                scope=str(meta_json.get("scope") or "local"),
                item_type=meta_json.get("type"),
                reason="EXPIRED",
            )
        return stats

    async def _handle_reconcile(self, *, subject: str) -> dict[str, Any]:
        stats = {
            "orphan_points_deleted": 0,
            "collections_checked": 0,
            "collections_with_orphans": 0,
            "conversations_checked": 0,
            "orphan_points_by_collection": {},
            "missing_indexed_rows_marked": 0,
            "missing_indexed_rows_by_collection": {},
            "ephemeral_rows_purged": 0,
        }
        if subject != "*":
            local_stats = await self._reconcile_local_scope(subject)
            self._merge_reconcile_stats(stats, local_stats)
            stats["conversations_checked"] = 1
            return stats

        conversation_ids = sorted(
            {
                str(row.get("conversation_id"))
                for row in await self._store.list_conversations()
                if row.get("conversation_id") and str(row.get("status") or "ACTIVE") != "DELETED"
            }
        )
        for conversation_id in conversation_ids:
            local_stats = await self._reconcile_local_scope(conversation_id)
            self._merge_reconcile_stats(stats, local_stats)
        stats["conversations_checked"] = len(conversation_ids)

        collection_map = {
            self._qdrant.catalog.global_citems: (
                await self._store.list_global_citem_records(),
                "global_citem",
                "global",
                "type",
                self._store.update_global_citem_vector_state,
            ),
            self._qdrant.catalog.global_summaries: (
                await self._store.list_global_summary_records(),
                "global_summary",
                "global",
                "level",
                self._store.update_global_summary_vector_state,
            ),
        }
        for collection, (rows, ref_kind, scope, type_field, update_missing) in collection_map.items():
            result = await self._reconcile_collection(
                collection_name=collection,
                rows=rows,
                ref_kind=ref_kind,
                scope=scope,
                type_field=type_field,
                conversation_id=None,
                update_missing=update_missing,
            )
            stats["collections_checked"] += 1
            if result["orphan_points_deleted"]:
                stats["collections_with_orphans"] += 1
                stats["orphan_points_by_collection"][collection] = result["orphan_points_deleted"]
                stats["orphan_points_deleted"] += result["orphan_points_deleted"]
            if result["missing_indexed_rows_marked"]:
                stats["missing_indexed_rows_by_collection"][collection] = result["missing_indexed_rows_marked"]
                stats["missing_indexed_rows_marked"] += result["missing_indexed_rows_marked"]
        return stats

    async def _reconcile_local_scope(self, conversation_id: str) -> dict[str, Any]:
        stats = {
            "orphan_points_deleted": 0,
            "collections_checked": 0,
            "collections_with_orphans": 0,
            "orphan_points_by_collection": {},
            "missing_indexed_rows_marked": 0,
            "missing_indexed_rows_by_collection": {},
            "ephemeral_rows_purged": 0,
        }
        collection_map = {
            self._qdrant.catalog.chunks: (
                await self._store.list_chunk_records(conversation_id),
                "chunk",
                "local",
                "chunk_kind",
                self._store.update_chunk_vector_state,
            ),
            self._qdrant.catalog.local_citems: (
                await self._store.list_local_citem_records(conversation_id),
                "local_citem",
                "local",
                "type",
                self._store.update_local_citem_vector_state,
            ),
            self._qdrant.catalog.local_summaries: (
                await self._store.list_local_summary_records(conversation_id),
                "local_summary",
                "local",
                "level",
                self._store.update_local_summary_vector_state,
            ),
        }
        for collection, (rows, ref_kind, scope, type_field, update_missing) in collection_map.items():
            result = await self._reconcile_collection(
                collection_name=collection,
                rows=rows,
                ref_kind=ref_kind,
                scope=scope,
                type_field=type_field,
                conversation_id=conversation_id,
                update_missing=update_missing,
            )
            stats["collections_checked"] += 1
            if result["orphan_points_deleted"]:
                stats["collections_with_orphans"] += 1
                stats["orphan_points_by_collection"][collection] = result["orphan_points_deleted"]
                stats["orphan_points_deleted"] += result["orphan_points_deleted"]
            if result["missing_indexed_rows_marked"]:
                stats["missing_indexed_rows_by_collection"][collection] = result["missing_indexed_rows_marked"]
                stats["missing_indexed_rows_marked"] += result["missing_indexed_rows_marked"]
        eph_result = await self._reconcile_ephemeral_collection(conversation_id=conversation_id)
        stats["collections_checked"] += 1
        if eph_result["orphan_points_deleted"]:
            stats["collections_with_orphans"] += 1
            stats["orphan_points_by_collection"][self._qdrant.catalog.ephemeral] = eph_result["orphan_points_deleted"]
            stats["orphan_points_deleted"] += eph_result["orphan_points_deleted"]
        if eph_result["ephemeral_rows_purged"]:
            stats["ephemeral_rows_purged"] += eph_result["ephemeral_rows_purged"]
        return stats

    @staticmethod
    def _merge_reconcile_stats(target: dict[str, Any], delta: dict[str, Any]) -> None:
        target["orphan_points_deleted"] += int(delta.get("orphan_points_deleted", 0) or 0)
        target["collections_checked"] += int(delta.get("collections_checked", 0) or 0)
        target["collections_with_orphans"] += int(delta.get("collections_with_orphans", 0) or 0)
        target["missing_indexed_rows_marked"] += int(delta.get("missing_indexed_rows_marked", 0) or 0)
        target["ephemeral_rows_purged"] += int(delta.get("ephemeral_rows_purged", 0) or 0)
        for collection, count in dict(delta.get("orphan_points_by_collection") or {}).items():
            target["orphan_points_by_collection"][collection] = int(target["orphan_points_by_collection"].get(collection, 0)) + int(count or 0)
        for collection, count in dict(delta.get("missing_indexed_rows_by_collection") or {}).items():
            target["missing_indexed_rows_by_collection"][collection] = int(target["missing_indexed_rows_by_collection"].get(collection, 0)) + int(count or 0)

    async def _reconcile_collection(
        self,
        *,
        collection_name: str,
        rows: list[dict[str, Any]],
        ref_kind: str,
        scope: str,
        type_field: str,
        conversation_id: str | None,
        update_missing: Any,
    ) -> dict[str, int]:
        expected_rows = [row for row in rows if str(row.get("vector_state") or "NONE") == "INDEXED"]
        expected_ids = {self._row_ref_id(row, ref_kind=ref_kind) for row in expected_rows}
        if conversation_id is not None:
            actual_ids = set(await self._qdrant.list_point_ids_by_conversation(collection_name=collection_name, conversation_id=conversation_id))
        else:
            actual_ids = set(await self._qdrant.list_all_point_ids(collection_name=collection_name))
        orphan_ids = sorted(actual_ids - expected_ids)
        missing_ids = sorted(expected_ids - actual_ids)
        row_by_id = {self._row_ref_id(row, ref_kind=ref_kind): row for row in rows}
        if orphan_ids:
            await self._qdrant.delete_point_ids(collection_name=collection_name, point_ids=orphan_ids)
            for orphan_id in orphan_ids:
                row = row_by_id.get(orphan_id)
                item_type = row.get(type_field) if row is not None else None
                orphan_conversation = conversation_id
                if orphan_conversation is None and row is not None:
                    orphan_conversation = row.get("origin_conversation_id") or row.get("conversation_id") or "*"
                if orphan_conversation is None:
                    orphan_conversation = "*"
                await self._emit_vector_deleted(
                    conversation_id=str(orphan_conversation),
                    ref_kind=ref_kind,
                    ref_id=orphan_id,
                    collection=collection_name,
                    scope=scope,
                    item_type=str(item_type) if item_type is not None else None,
                    reason="ORPHAN_CLEANUP",
                )
        for missing_id in missing_ids:
            row = row_by_id.get(missing_id)
            if row is None:
                continue
            await update_missing(missing_id, vector_state="THINNED")
            missing_conversation = conversation_id
            if missing_conversation is None:
                missing_conversation = row.get("origin_conversation_id") or row.get("conversation_id") or "*"
            await self._emit_vector_deleted(
                conversation_id=str(missing_conversation or "*"),
                ref_kind=ref_kind,
                ref_id=missing_id,
                collection=collection_name,
                scope=scope,
                item_type=str(row.get(type_field)) if row.get(type_field) is not None else None,
                reason="RECONCILE",
            )
        return {
            "orphan_points_deleted": len(orphan_ids),
            "missing_indexed_rows_marked": len(missing_ids),
        }

    async def _reconcile_ephemeral_collection(self, *, conversation_id: str | None) -> dict[str, int]:
        rows = await self._store.list_ephemeral_vector_records(conversation_id=conversation_id)
        expected_rows = [
            row
            for row in rows
            if str(row.get("vector_state") or "NONE") == "EPHEMERAL"
            and str(row.get("lifecycle_state") or "ACTIVE") in {"ACTIVE", "EXPIRED"}
        ]
        expected_ids = {str(row.get("ephemeral_id") or "") for row in expected_rows if row.get("ephemeral_id")}
        if conversation_id is not None:
            actual_ids = set(
                await self._qdrant.list_point_ids_by_conversation(
                    collection_name=self._qdrant.catalog.ephemeral,
                    conversation_id=conversation_id,
                )
            )
        else:
            actual_ids = set(await self._qdrant.list_all_point_ids(collection_name=self._qdrant.catalog.ephemeral))
        orphan_ids = sorted(actual_ids - expected_ids)
        missing_ids = sorted(expected_ids - actual_ids)
        row_by_id = {str(row.get("ephemeral_id") or ""): row for row in rows if row.get("ephemeral_id")}
        if orphan_ids:
            await self._qdrant.delete_point_ids(collection_name=self._qdrant.catalog.ephemeral, point_ids=orphan_ids)
            for orphan_id in orphan_ids:
                row = row_by_id.get(orphan_id)
                meta_json = dict(row.get("meta_json") or {}) if row is not None else {}
                orphan_conversation = conversation_id or str((row or {}).get("conversation_id") or "*")
                await self._emit_vector_deleted(
                    conversation_id=orphan_conversation,
                    ref_kind="ephemeral",
                    ref_id=orphan_id,
                    collection=self._qdrant.catalog.ephemeral,
                    scope=str(meta_json.get("scope") or "local"),
                    item_type=str(meta_json.get("type")) if meta_json.get("type") is not None else None,
                    reason="ORPHAN_CLEANUP",
                )
        purged = 0
        if missing_ids:
            now_iso = datetime.now(UTC).isoformat()
            for missing_id in missing_ids:
                row = row_by_id.get(missing_id)
                if row is None:
                    continue
                if str(row.get("lifecycle_state") or "ACTIVE") == "ACTIVE":
                    await self._store.mark_ephemeral_vector_expired(missing_id, expired_at=now_iso)
                await self._store.mark_ephemeral_vector_purged(missing_id, purged_at=now_iso)
                purged += 1
                meta_json = dict(row.get("meta_json") or {})
                await self._emit_vector_deleted(
                    conversation_id=str(row.get("conversation_id") or conversation_id or "*"),
                    ref_kind="ephemeral",
                    ref_id=missing_id,
                    collection=self._qdrant.catalog.ephemeral,
                    scope=str(meta_json.get("scope") or "local"),
                    item_type=str(meta_json.get("type")) if meta_json.get("type") is not None else None,
                    reason="RECONCILE",
                )
        return {
            "orphan_points_deleted": len(orphan_ids),
            "ephemeral_rows_purged": purged,
        }

    def _row_ref_id(self, row: dict[str, Any], *, ref_kind: str) -> str:
        field_map = {
            "chunk": "chunk_id",
            "local_citem": "local_citem_id",
            "local_summary": "local_summary_id",
            "global_citem": "global_citem_id",
            "global_summary": "global_summary_id",
        }
        return str(row[field_map[ref_kind]])

    def _thinning_candidate(self, row: dict[str, Any], *, now: datetime, pinned_field: str | None = None) -> bool:
        if str(row.get("vector_state") or "NONE") != "INDEXED":
            return False
        if pinned_field is not None and bool(row.get(pinned_field)):
            return False
        if bool(row.get("was_cited")):
            return False
        last_touch = row.get("last_used_at") or row.get("created_at")
        try:
            touched_at = datetime.fromisoformat(str(last_touch))
        except Exception:
            return False
        return now - touched_at >= self._thinning_age

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
                meta=VectorMeta(scope=scope, type=item_type),
            ).model_dump(mode="json"),
        )
        await self._store.append_outbox_event(
            topic=TOPICS.vector_events,
            message_key=conversation_key(conversation_id),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )
