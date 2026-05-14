from __future__ import annotations

from typing import Any

from cima_demo.geometry.projector import GeometryReadModelProjector
from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType
from cima_demo.witness_backend.topic_catalog import TOPICS


class GeometryReadModelProjectorConsumer:
    """Consume geometry topics and materialise CIMA-side read models idempotently."""

    def __init__(self, *, projector: GeometryReadModelProjector, ledger: ConsumerEffectLedger) -> None:
        self._projector = projector
        self._ledger = ledger
        self._consumer_name = "geom-read-model-projector"

    async def handle(self, *, topic: str, message_key: str, payload_json: dict[str, Any] | None) -> None:
        effect = ConsumerEffectKey(self._consumer_name, self._event_identity(topic, message_key, payload_json), f"{topic}:{message_key}")
        should_apply = await self._ledger.begin(effect)
        if not should_apply:
            return
        if topic == TOPICS.geom_item_state:
            if payload_json is None:
                conversation_id, ref_kind, ref_id = message_key.split("|", 2)
                await self._projector.delete_item_state(conversation_id, ref_kind=ref_kind, ref_id=ref_id)
            else:
                envelope = CloudEventEnvelope.model_validate(payload_json)
                await self._projector.apply_item_state(envelope.subject, envelope.data)
        elif topic == TOPICS.geom_cluster_state:
            if payload_json is None:
                conversation_id, cluster_id = message_key.split("|", 1)
                await self._projector.delete_cluster_state(conversation_id, cluster_id=cluster_id)
            else:
                envelope = CloudEventEnvelope.model_validate(payload_json)
                await self._projector.apply_cluster_state(envelope.subject, envelope.data)
        elif topic == TOPICS.geom_run:
            if payload_json is not None:
                envelope = CloudEventEnvelope.model_validate(payload_json)
                await self._projector.apply_run_completed(envelope.subject, envelope.data)
        elif topic == TOPICS.conversation_events and payload_json is not None:
            envelope = CloudEventEnvelope.model_validate(payload_json)
            if envelope.type == EventType.CONVERSATION_HARD_DELETE_REQUESTED:
                await self._projector.purge_conversation(envelope.subject)
        else:
            raise ValueError(f"Unsupported projector topic: {topic}")
        await self._ledger.complete(effect, details_json={"topic": topic, "message_key": message_key})

    @staticmethod
    def _event_identity(topic: str, message_key: str, payload_json: dict[str, Any] | None) -> str:
        if payload_json is None:
            return f"tombstone:{topic}:{message_key}"
        envelope = CloudEventEnvelope.model_validate(payload_json)
        return str(envelope.id)
