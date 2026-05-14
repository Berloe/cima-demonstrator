from __future__ import annotations

"""Schema-aware outbox store wrappers.

CIMA and Geometry each own their outbox table. We keep the publisher generic and
select the correct schema through a tiny wrapper instead of duplicating logic.
"""

from typing import Any


class DatabaseOutboxStore:
    def __init__(self, db: Any, *, schema_name: str) -> None:
        if schema_name not in {"cima", "geom"}:
            raise ValueError(f"Unsupported outbox schema: {schema_name}")
        self._db = db
        self._schema_name = schema_name

    async def claim_outbox_batch(self, limit: int = 100) -> list[dict[str, Any]]:
        if self._schema_name == "geom":
            return await self._db.claim_geom_outbox_batch(limit)
        return await self._db.claim_outbox_batch(limit)

    async def mark_outbox_sent(self, outbox_ids: list[int]) -> None:
        if self._schema_name == "geom":
            await self._db.mark_geom_outbox_sent(outbox_ids)
            return
        await self._db.mark_outbox_sent(outbox_ids)

    async def mark_outbox_error(self, outbox_id: int, error: str) -> None:
        if self._schema_name == "geom":
            await self._db.mark_geom_outbox_error(outbox_id, error)
            return
        await self._db.mark_outbox_error(outbox_id, error)
