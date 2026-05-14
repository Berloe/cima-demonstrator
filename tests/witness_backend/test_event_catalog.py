from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from cima_demo.witness_backend.events import (
    CloudEventEnvelope,
    EventType,
    GeometryRecomputeData,
    Producer,
    SourceRegisteredData,
    TraceContext,
    VectorMeta,
    VectorUpsertedData,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, geom_cluster_state_key, geom_item_state_key


def test_topic_catalog_and_compacted_keys_are_stable() -> None:
    assert TOPICS.memory_events == "cima.memory.events.v1"
    assert TOPICS.geom_item_state == "cima.geom.item_state.v1"
    assert TOPICS.geom_cluster_state == "cima.geom.cluster_state.v1"
    assert geom_item_state_key("conv-1", "local_citem", "abc") == "conv-1|local_citem|abc"
    assert geom_cluster_state_key("conv-1", "c_001") == "conv-1|c_001"


def test_cloudevent_envelope_and_payload_models_validate() -> None:
    payload = VectorUpsertedData(
        ref_kind="local_citem",
        ref_id=uuid4(),
        qdrant_collection="cima_local_citems",
        vector_state="INDEXED",
        embedding_model_id="tei-mistral",
        embedding_schema_version=1,
        eligible_for_geometry=True,
        meta=VectorMeta(scope="local", type="DECISION"),
    )
    envelope = CloudEventEnvelope(
        type=EventType.VECTOR_UPSERTED,
        source=Producer.CIMA_WORKER,
        subject="conv-1",
        dataschema="schemas/cima.vector.upserted.v1.json",
        time=datetime.now(UTC),
        data=payload.model_dump(mode="json"),
    )
    assert envelope.specversion == "1.0"
    assert isinstance(envelope.id, UUID)
    cmd = GeometryRecomputeData(reason="MANUAL")
    assert cmd.override_params is None


def test_source_registered_payload_separates_display_and_process_concerns() -> None:
    payload = SourceRegisteredData(
        source_id=uuid4(),
        kind="chat_user",
        external_provider="librechat",
        external_message_id="msg-1",
        revision_no=0,
        displayable=True,
        processable=True,
    )
    assert payload.displayable is True
    assert payload.processable is True


def test_source_registered_data_canonicalizes_legacy_file_alias() -> None:
    payload = SourceRegisteredData(source_id=uuid4(), kind="file")
    assert payload.kind == "file_text"


def test_source_registered_data_canonicalizes_dataset_document_alias() -> None:
    payload = SourceRegisteredData(source_id=uuid4(), kind="dataset_document")
    assert payload.kind == "file_text"
