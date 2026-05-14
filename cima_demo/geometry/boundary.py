from __future__ import annotations

"""Geometry boundary adapters.

The geometry bounded context is consumed through two narrow ports:
- a read-side hints port used by CIMA selection/UI,
- a command port used to request recompute/purge through the async plane.

In standalone/demo mode we keep an in-process adapter for reproducibility.
In the full runtime the command side writes CloudEvents to the outbox so the
separate geometry service can consume them over Kafka.
"""

import inspect
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    EventType,
    GeometryPurgeData,
    GeometryRecomputeData,
    Producer,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


class GeometryHintsPort(Protocol):
    async def get_item_hints(self, *, conversation_id: str, ref_ids: list[str]) -> dict[str, dict[str, Any]]: ...

    async def load_all_item_hints(self, *, conversation_id: str) -> list[dict[str, Any]]: ...

    async def get_cluster_hints(self, *, conversation_id: str) -> list[dict[str, Any]]: ...


class GeometryCommandsPort(Protocol):
    async def schedule_recompute(self, conversation_id: str, *, reason: str = "context_snapshot") -> None: ...

    async def purge_conversation(self, conversation_id: str, *, delete_run_id: str | None = None) -> None: ...


class DirectGeometryBoundary(GeometryHintsPort, GeometryCommandsPort):
    """Compatibility adapter for the in-process/demo runtime."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def get_item_hints(self, *, conversation_id: str, ref_ids: list[str]) -> dict[str, dict[str, Any]]:
        return await self._service.get_item_hints(conversation_id=conversation_id, ref_ids=ref_ids)

    async def load_all_item_hints(self, *, conversation_id: str) -> list[dict[str, Any]]:
        return await self._service.load_all_item_hints(conversation_id=conversation_id)

    async def get_cluster_hints(self, *, conversation_id: str) -> list[dict[str, Any]]:
        return await self._service.get_cluster_hints(conversation_id=conversation_id)

    async def schedule_recompute(self, conversation_id: str, *, reason: str = "context_snapshot") -> None:
        maybe_awaitable = self._service.schedule_recompute(conversation_id, reason=reason)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    async def purge_conversation(self, conversation_id: str, *, delete_run_id: str | None = None) -> None:
        await self._service.purge_conversation(conversation_id)


class GeometryReadModelService(GeometryHintsPort):
    """Read-side facade over CIMA's geometry read model tables."""

    def __init__(self, rel_db: Any) -> None:
        self._db = rel_db

    async def get_item_hints(self, *, conversation_id: str, ref_ids: list[str]) -> dict[str, dict[str, Any]]:
        loader = getattr(self._db, "load_geometry_read_model_item_states", None) or getattr(self._db, "load_geometry_item_states")
        rows = await loader(conversation_id, ref_ids)
        return {str(row["ref_id"]): row for row in rows}

    async def load_all_item_hints(self, *, conversation_id: str) -> list[dict[str, Any]]:
        loader = getattr(self._db, "load_geometry_read_model_item_states", None) or getattr(self._db, "load_geometry_item_states")
        return await loader(conversation_id)

    async def get_cluster_hints(self, *, conversation_id: str) -> list[dict[str, Any]]:
        loader = getattr(self._db, "load_geometry_read_model_cluster_states", None) or getattr(self._db, "load_geometry_cluster_states")
        return await loader(conversation_id)


class GeometryCommandPublisher(GeometryCommandsPort):
    """Command-side adapter that writes geometry commands to the outbox.

    This keeps the geometry bounded context behind the async plane even when CIMA
    runs in-process as an API.
    """

    def __init__(self, rel_db: Any, *, producer: Producer = Producer.CIMA_API) -> None:
        self._db = rel_db
        self._producer = producer

    async def schedule_recompute(self, conversation_id: str, *, reason: str = "context_snapshot") -> None:
        normalized_reason = _normalize_geom_reason(reason)
        payload = GeometryRecomputeData(reason=normalized_reason).model_dump(mode="json")
        envelope = CloudEventEnvelope(
            type=EventType.GEOM_RECOMPUTE,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.geom.recompute.requested.v1.json",
            time=datetime.now(UTC),
            data=payload,
        )
        await self._db.append_outbox_event(
            topic=TOPICS.geom_cmd,
            message_key=conversation_key(conversation_id),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )

    async def purge_conversation(self, conversation_id: str, *, delete_run_id: str | None = None) -> None:
        payload = GeometryPurgeData(delete_run_id=delete_run_id or str(uuid4())).model_dump(mode="json")
        envelope = CloudEventEnvelope(
            type=EventType.GEOM_PURGE,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.geom.purge.requested.v1.json",
            time=datetime.now(UTC),
            data=payload,
        )
        await self._db.append_outbox_event(
            topic=TOPICS.geom_cmd,
            message_key=conversation_key(conversation_id),
            headers_json={"ce_type": envelope.type.value, "ce_source": envelope.source.value},
            payload_json=envelope.model_dump(mode="json"),
        )


def _normalize_geom_reason(reason: str) -> str:
    normalized = reason.strip().upper() if reason else "MANUAL"
    mapping = {
        "CONTEXT_SNAPSHOT": "DELTA_THRESHOLD",
        "MEMORY_APPLY": "DELTA_THRESHOLD",
        "CONTEXT_REFRESH": "DELTA_THRESHOLD",
        "EPOCH_CLOSED": "EPOCH_CLOSED",
        "MANUAL": "MANUAL",
        "RECOVERY": "RECOVERY",
        "DELTA_THRESHOLD": "DELTA_THRESHOLD",
    }
    return mapping.get(normalized, "MANUAL")
