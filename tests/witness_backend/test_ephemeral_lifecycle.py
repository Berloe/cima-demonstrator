from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.witness_backend.ephemeral import EphemeralVectorRegistry
from cima_demo.witness_backend.events import EventType
from cima_demo.witness_backend.topic_catalog import TOPICS


@pytest.mark.asyncio
async def test_ephemeral_registry_persists_active_lease_and_emits_vector_upserted() -> None:
    db = InMemoryDemoDB()
    conversation_id = str(uuid4())
    await db.create_conversation(conversation_id)
    registry = EphemeralVectorRegistry(db)

    lease = await registry.register(
        conversation_id=conversation_id,
        origin_ref_kind="chunk",
        origin_ref_id=str(uuid4()),
        qdrant_collection="cima_ephemeral",
        embedding_model_id="tei-test",
        embedding_schema_version=2,
        ttl_seconds=600,
        scope="local",
        item_type="FACT",
        now=datetime.now(UTC),
    )

    row = db.ephemeral_vector_records[lease.ephemeral_id]
    assert row["conversation_id"] == conversation_id
    assert row["lifecycle_state"] == "ACTIVE"
    assert row["vector_state"] == "EPHEMERAL"
    assert row["eligible_for_geometry"] is False
    assert row["meta_json"]["scope"] == "local"
    assert row["meta_json"]["type"] == "FACT"
    assert UUID(lease.ephemeral_id)

    outbox = db.outbox_rows[-1]
    assert outbox["topic"] == TOPICS.vector_events
    assert outbox["payload_json"]["type"] == EventType.VECTOR_UPSERTED.value
    assert outbox["payload_json"]["data"]["ref_kind"] == "ephemeral"
    assert outbox["payload_json"]["data"]["vector_state"] == "EPHEMERAL"
    assert outbox["payload_json"]["data"]["eligible_for_geometry"] is False
