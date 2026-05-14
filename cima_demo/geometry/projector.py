from __future__ import annotations

"""Read-model projector for geometry compacted topics.

The geometry service remains a hard-bounded context. CIMA consumes its compacted
item and cluster state topics and materialises a local read model for selection
and UI, instead of calling the geometry runtime over business HTTP endpoints.
"""

from typing import Any

from cima_demo.witness_backend.events import (
    GeometryClusterStateData,
    GeometryItemStateData,
    GeometryRunCompletedData,
)


class GeometryReadModelProjector:
    def __init__(self, rel_db: Any) -> None:
        self._db = rel_db

    async def apply_run_completed(self, conversation_id: str, payload: dict[str, Any]) -> None:
        state = GeometryRunCompletedData.model_validate(payload)
        saver = getattr(self._db, "save_geometry_read_model_run", None)
        if saver is None:
            return
        await saver({
            "conversation_id": conversation_id,
            **state.model_dump(mode="json"),
        })

    async def apply_item_state(self, conversation_id: str, payload: dict[str, Any]) -> None:
        state = GeometryItemStateData.model_validate(payload)
        saver = getattr(self._db, "save_geometry_read_model_item_state", None) or getattr(self._db, "save_geometry_item_state")
        await saver(
            {
                "conversation_id": conversation_id,
                **state.model_dump(mode="json"),
            }
        )

    async def apply_cluster_state(self, conversation_id: str, payload: dict[str, Any]) -> None:
        state = GeometryClusterStateData.model_validate(payload)
        saver = getattr(self._db, "save_geometry_read_model_cluster_state", None) or getattr(self._db, "save_geometry_cluster_state")
        await saver(
            {
                "conversation_id": conversation_id,
                "run_id": str(state.run_id),
                "cluster_id": state.cluster_id,
                "mass": state.mass,
                "medoid_ref_kind": state.medoid.ref_kind,
                "medoid_ref_id": str(state.medoid.ref_id),
                "summary_id": str(state.summary_id) if state.summary_id is not None else None,
                "updated_at": state.updated_at.isoformat(),
                "label": None,
            }
        )

    async def delete_item_state(self, conversation_id: str, *, ref_kind: str, ref_id: str) -> None:
        deleter = getattr(self._db, "delete_geometry_read_model_item_state", None)
        if deleter is None:
            return
        await deleter(conversation_id, ref_kind, ref_id)

    async def delete_cluster_state(self, conversation_id: str, *, cluster_id: str) -> None:
        deleter = getattr(self._db, "delete_geometry_read_model_cluster_state", None)
        if deleter is None:
            return
        await deleter(conversation_id, cluster_id)

    async def purge_conversation(self, conversation_id: str) -> None:
        deleter = getattr(self._db, "delete_geometry_read_model_conversation", None) or getattr(self._db, "delete_geometry_conversation")
        await deleter(conversation_id)
