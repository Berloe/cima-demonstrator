from __future__ import annotations

"""Witness-backend collection orchestration helpers for Qdrant.

The demonstrator still routes most live recall through the legacy CItem store,
but lifecycle and GC operations already need the approved physical collection
layout.  This helper centralises collection-wide operations so hard delete,
EPHEMERAL cleanup and future thinning jobs do not need to know collection names.
"""

from dataclasses import dataclass
from typing import Any, Protocol
from datetime import UTC, datetime

try:
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        HasIdCondition,
        MatchValue,
        PointStruct,
        DatetimeRange,
    )
except ModuleNotFoundError:
    class DatetimeRange:
        def __init__(self, *, lt=None, lte=None, gt=None, gte=None):
            self.lt = lt
            self.lte = lte
            self.gt = gt
            self.gte = gte

try:  # pragma: no cover - real runtime path
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchValue, PointStruct, Range
except ModuleNotFoundError:  # pragma: no cover - lightweight test fallback
    from typing import Any as AsyncQdrantClient

    class MatchValue:
        def __init__(self, *, value: object) -> None:
            self.value = value

    class Range:
        def __init__(self, *, lt: object | None = None) -> None:
            self.lt = lt

    class FieldCondition:
        def __init__(self, *, key: str, match: object | None = None, range: object | None = None) -> None:
            self.key = key
            self.match = match
            self.range = range

    class HasIdCondition:
        def __init__(self, *, has_id: list[str]) -> None:
            self.has_id = has_id

    class Filter:
        def __init__(self, *, must: list[object] | None = None) -> None:
            self.must = must or []

    class PointStruct:
        def __init__(self, *, id: str, payload: dict[str, object], vector: dict[str, object]) -> None:
            self.id = id
            self.payload = payload
            self.vector = vector

from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.setup import ensure_collections


@dataclass(frozen=True, slots=True)
class WitnessPlaneDeleteReport:
    local_points_deleted: int
    collections_touched: tuple[str, ...]


class SupportsQdrantCollections(Protocol):
    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: Any,
    ) -> Any: ...


class QdrantWitnessPlane:
    def __init__(self, *, client: AsyncQdrantClient, catalog: QdrantCollectionCatalog, dense_dim: int) -> None:
        self._client = client
        self._catalog = catalog
        self._dense_dim = dense_dim

    @property
    def catalog(self) -> QdrantCollectionCatalog:
        return self._catalog

    async def ensure_ready(self) -> None:
        await ensure_collections(self._client, self._catalog, self._dense_dim)

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all conversation-scoped vectors from the approved local collections.

        Qdrant filter deletes do not reliably return deleted counts, so we return
        0 and leave precise accounting to higher-level audits/reconciliation.
        """
        conversation_filter = Filter(
            must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
        )
        for collection in self._catalog.local_scoped():
            await self._client.delete(collection_name=collection, points_selector=conversation_filter)
        return 0

    async def upsert_points(self, *, collection_name: str, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        rows = [
            PointStruct(id=str(point["id"]), payload=dict(point.get("payload") or {}), vector={"content": list(point["vector"])})
            for point in points
        ]
        await self._client.upsert(collection_name=collection_name, points=rows)

    async def set_payload_fields(self, *, collection_name: str, point_ids: list[str], payload: dict[str, Any]) -> None:
        if not point_ids:
            return
        if hasattr(self._client, "set_payload"):
            await self._client.set_payload(collection_name=collection_name, points=point_ids, payload=payload)

    async def delete_point_ids(self, *, collection_name: str, point_ids: list[str]) -> int:
        if not point_ids:
            return 0
        point_filter = Filter(must=[HasIdCondition(has_id=[str(v) for v in point_ids])])
        await self._client.delete(collection_name=collection_name, points_selector=point_filter)
        return len(point_ids)

    async def list_point_ids_by_conversation(self, *, collection_name: str, conversation_id: str) -> list[str]:
        rows: list[str] = []
        scroll_filter = Filter(must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))])
        offset = None
        while True:
            batch, offset = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            rows.extend(str(row.id) for row in batch)
            if offset is None:
                break
        return rows

    async def list_all_point_ids(self, *, collection_name: str) -> list[str]:
        rows: list[str] = []
        offset = None
        while True:
            batch, offset = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=[]),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            rows.extend(str(row.id) for row in batch)
            if offset is None:
                break
        return rows

    async def sweep_ephemeral_expired(self, *, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(UTC)
        expiry_filter = Filter(
            must=[FieldCondition(key="expires_at", range=DatetimeRange(lt=cutoff))]
        )
        await self._client.delete(
            collection_name=self._catalog.ephemeral,
            points_selector=expiry_filter,
        )
        return 0