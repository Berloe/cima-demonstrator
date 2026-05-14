from __future__ import annotations

"""Persistent EPHEMERAL vector registry for the witness backend.

R2.1 / R2.2 introduce an explicit persistence model for temporary vectors instead
of treating the dedicated Qdrant collection as the only source of truth. The
registry stores lifecycle rows in PostgreSQL and emits canonical VECTOR_UPSERTED
CloudEvents for the async plane.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    EventType,
    Producer,
    VectorMeta,
    VectorUpsertedData,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


class EphemeralRegistryStore(Protocol):
    async def save_ephemeral_vector_record(self, record_json: dict[str, Any]) -> None: ...

    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class EphemeralVectorLease:
    ephemeral_id: str
    conversation_id: str
    origin_ref_kind: str
    origin_ref_id: str | None
    qdrant_collection: str
    lifecycle_state: str
    expires_at: str
    accepted: bool


class EphemeralVectorRegistry:
    def __init__(self, store: EphemeralRegistryStore, *, producer: Producer = Producer.CIMA_WORKER) -> None:
        self._store = store
        self._producer = producer

    async def register(
        self,
        *,
        conversation_id: str,
        origin_ref_kind: str,
        origin_ref_id: str | None,
        qdrant_collection: str,
        embedding_model_id: str,
        embedding_schema_version: int,
        ttl_seconds: int,
        scope: str = "local",
        item_type: str | None = None,
        meta_json: dict[str, Any] | None = None,
        ephemeral_id: str | None = None,
        now: datetime | None = None,
    ) -> EphemeralVectorLease:
        created_at = (now or datetime.now(UTC))
        expires_at = created_at + timedelta(seconds=max(1, int(ttl_seconds)))
        lease_id = str(UUID(ephemeral_id)) if ephemeral_id is not None else str(uuid4())
        payload_meta = dict(meta_json or {})
        payload_meta.setdefault("scope", scope)
        if item_type is not None:
            payload_meta.setdefault("type", item_type)
        payload_meta.setdefault("origin_ref_kind", origin_ref_kind)
        if origin_ref_id is not None:
            payload_meta.setdefault("origin_ref_id", origin_ref_id)
        await self._store.save_ephemeral_vector_record(
            {
                "ephemeral_id": lease_id,
                "conversation_id": conversation_id,
                "origin_ref_kind": origin_ref_kind,
                "origin_ref_id": origin_ref_id,
                "qdrant_collection": qdrant_collection,
                "lifecycle_state": "ACTIVE",
                "vector_state": "EPHEMERAL",
                "embedding_model_id": embedding_model_id,
                "embedding_schema_version": embedding_schema_version,
                "eligible_for_geometry": False,
                "meta_json": payload_meta,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "expired_at": None,
                "purged_at": None,
            }
        )
        envelope = CloudEventEnvelope(
            type=EventType.VECTOR_UPSERTED,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.vector.upserted.v1.json",
            time=created_at,
            data=VectorUpsertedData(
                ref_kind="ephemeral",
                ref_id=UUID(lease_id),
                qdrant_collection=qdrant_collection,
                vector_state="EPHEMERAL",
                embedding_model_id=embedding_model_id,
                embedding_schema_version=embedding_schema_version,
                eligible_for_geometry=False,
                meta=VectorMeta(scope=scope, type=item_type),
            ).model_dump(mode="json"),
        )
        await self._store.append_outbox_event(
            topic=TOPICS.vector_events,
            message_key=conversation_key(conversation_id),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )
        return EphemeralVectorLease(
            ephemeral_id=lease_id,
            conversation_id=conversation_id,
            origin_ref_kind=origin_ref_kind,
            origin_ref_id=origin_ref_id,
            qdrant_collection=qdrant_collection,
            lifecycle_state="ACTIVE",
            expires_at=expires_at.isoformat(),
            accepted=True,
        )
