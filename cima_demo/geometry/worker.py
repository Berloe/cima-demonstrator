from __future__ import annotations

"""Event-driven geometry worker for the hard-boundary deployment mode.

The worker consumes compact, id-only events, recomputes geometry using the core
algorithm, and emits compacted state events plus a run summary. It intentionally
contains no business HTTP API.
"""

from datetime import UTC, datetime
from typing import Any

from cima_demo.geometry.service import DemoGeometryService
from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    EventEnvelope,
    EventType,
    GeometryClusterMedoid,
    GeometryClusterStateData,
    GeometryItemStateData,
    GeometryPurgeData,
    GeometryRecomputeData,
    GeometryRunCompletedData,
    GeometryRunMetrics,
    Producer,
    SummaryRequestedData,
)
from cima_demo.witness_backend.topic_catalog import (
    TOPICS,
    conversation_key,
    geom_cluster_state_key,
    geom_item_state_key,
)


class GeometryCommandProcessor:
    """Translate geometry commands into run + compacted state events."""

    def __init__(self, service: DemoGeometryService) -> None:
        self._service = service

    async def handle(self, envelope: EventEnvelope) -> list[tuple[str, str, dict[str, Any] | None]]:
        if envelope.type == EventType.GEOM_RECOMPUTE:
            return await self._handle_recompute(envelope)
        if envelope.type == EventType.GEOM_PURGE:
            return await self._handle_purge(envelope)
        raise ValueError(f"Unsupported geometry command: {envelope.type}")

    async def _handle_recompute(self, envelope: EventEnvelope) -> list[tuple[str, str, dict[str, Any] | None]]:
        cmd = GeometryRecomputeData.model_validate(envelope.data)
        run = await self._service.recompute(
            conversation_id=envelope.subject,
            reason=cmd.reason.lower(),
        )
        all_rows = await self._service.load_all_item_hints(conversation_id=envelope.subject)
        cluster_rows = await self._service.get_cluster_hints(conversation_id=envelope.subject)

        now = datetime.now(UTC)
        run_payload = GeometryRunCompletedData(
            run_id=run.run_id,
            algo_version=run.algo_version,
            universe_hash=f"{envelope.subject}:{run.run_id}",
            params={"k_used": run.cluster_count},
            metrics=GeometryRunMetrics(
                n_vectors=run.n_items,
                core_size=run.core_count,
                bridge_count=run.bridge_count,
                core_mass_frac=0.0,
            ),
        )

        run_event = CloudEventEnvelope(
            type=EventType.GEOM_RUN_COMPLETED,
            source=Producer.CIMA_GEOMETRY,
            subject=envelope.subject,
            dataschema="schemas/cima.geom.run.completed.v1.json",
            time=now,
            data=run_payload.model_dump(mode="json"),
        )
        out: list[tuple[str, str, dict[str, Any] | None]] = [
            (TOPICS.geom_run, conversation_key(envelope.subject), run_event.model_dump(mode="json")),
        ]
        for row in all_rows:
            ref_id = str(row["ref_id"])
            item_payload = GeometryItemStateData(
                run_id=row["run_id"],
                algo_version=run.algo_version,
                ref_kind=row["ref_kind"],
                ref_id=row["ref_id"],
                cluster_top1=row["cluster_top1"],
                cluster_top2=row.get("cluster_top2"),
                w1=row["w1"],
                w2=row.get("w2"),
                margin=row["margin"],
                is_core=row["is_core"],
                is_bridge_candidate=row["is_bridge_candidate"],
                centrality=row.get("centrality"),
                updated_at=now,
            )
            item_event = CloudEventEnvelope(
                type=EventType.GEOM_ITEM_STATE,
                source=Producer.CIMA_GEOMETRY,
                subject=envelope.subject,
                dataschema="schemas/cima.geom.item_state.v1.json",
                time=now,
                data=item_payload.model_dump(mode="json"),
            )
            out.append((TOPICS.geom_item_state, geom_item_state_key(envelope.subject, row["ref_kind"], ref_id), item_event.model_dump(mode="json")))
        for row in cluster_rows:
            cluster_payload = GeometryClusterStateData(
                run_id=row["run_id"],
                algo_version=run.algo_version,
                cluster_id=row["cluster_id"],
                mass=row["mass"],
                medoid=GeometryClusterMedoid(ref_kind="local_citem", ref_id=row["medoid_ref_id"]),
                summary_id=row.get("summary_id"),
                updated_at=now,
            )
            cluster_event = CloudEventEnvelope(
                type=EventType.GEOM_CLUSTER_STATE,
                source=Producer.CIMA_GEOMETRY,
                subject=envelope.subject,
                dataschema="schemas/cima.geom.cluster_state.v1.json",
                time=now,
                data=cluster_payload.model_dump(mode="json"),
            )
            out.append((TOPICS.geom_cluster_state, geom_cluster_state_key(envelope.subject, row["cluster_id"]), cluster_event.model_dump(mode="json")))
            summary_cmd = SummaryRequestedData(
                level="CLUSTER",
                cluster_id=row["cluster_id"],
                reason="GEOM_CLUSTER_CHANGED",
                priority="NORMAL",
            )
            summary_event = CloudEventEnvelope(
                type=EventType.SUMMARY_REQUESTED,
                source=Producer.CIMA_GEOMETRY,
                subject=envelope.subject,
                dataschema="schemas/cima.summary.requested.v1.json",
                time=now,
                data=summary_cmd.model_dump(mode="json"),
            )
            out.append((TOPICS.summary_cmd, conversation_key(envelope.subject), summary_event.model_dump(mode="json")))
        return out

    async def _handle_purge(self, envelope: EventEnvelope) -> list[tuple[str, str, dict[str, Any] | None]]:
        GeometryPurgeData.model_validate(envelope.data)
        item_rows = await self._service.load_all_item_hints(conversation_id=envelope.subject)
        cluster_rows = await self._service.get_cluster_hints(conversation_id=envelope.subject)
        await self._service.purge_conversation(envelope.subject)
        out: list[tuple[str, str, dict[str, Any] | None]] = []
        for row in item_rows:
            out.append(
                (
                    TOPICS.geom_item_state,
                    geom_item_state_key(envelope.subject, row["ref_kind"], str(row["ref_id"])),
                    None,
                )
            )
        for row in cluster_rows:
            out.append(
                (
                    TOPICS.geom_cluster_state,
                    geom_cluster_state_key(envelope.subject, row["cluster_id"]),
                    None,
                )
            )
        return out
