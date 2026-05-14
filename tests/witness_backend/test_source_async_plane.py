from __future__ import annotations

from pathlib import Path

import pytest

from cima_demo.demo.harness.fakes import InMemoryDemoDB
from cima_demo.demo.lineage import DemoLineageService
from cima_demo.infrastructure.files.chunker import SemanticChunkerAdapter
from cima_demo.infrastructure.files.processor import FileProcessingAdapter
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.events import CloudEventEnvelope, EventType
from cima_demo.witness_backend.source_ingest import MemorySourceConsumer, SourceRegistrationService
from cima_demo.witness_backend.topic_catalog import TOPICS


@pytest.mark.asyncio
async def test_register_text_emits_source_registered_and_preserves_display_text(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    conversation_id = "11111111-1111-1111-1111-111111111111"
    await db.create_conversation(conversation_id)
    service = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )

    result = await service.register_text(
        conversation_id=conversation_id,
        text="Linea 1\n\nLinea 2",
        role="user",
        source_kind="chat_user",
        external_provider="librechat",
        external_message_id="msg-1",
    )

    source_row = db.demo_sources[result.source_id]
    assert source_row["display_text"] == "Linea 1\n\nLinea 2"
    assert source_row["process_text"] == "Linea 1\n\nLinea 2"
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert db.outbox_rows[-1]["topic"] == TOPICS.memory_events
    assert envelope.type == EventType.MEMORY_SOURCE_REGISTERED
    assert envelope.subject == conversation_id


@pytest.mark.asyncio
async def test_register_text_canonicalizes_file_alias_to_file_text(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    conversation_id = "12121212-1212-1212-1212-121212121212"
    await db.create_conversation(conversation_id)
    service = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )

    result = await service.register_text(
        conversation_id=conversation_id,
        text="Document content",
        role=None,
        source_kind="file",
        displayable=False,
        processable=True,
    )

    source_row = db.demo_sources[result.source_id]
    assert source_row["source_kind"] == "file_text"
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert envelope.data["kind"] == "file_text"


@pytest.mark.asyncio
async def test_register_file_upload_persists_blob_and_emits_file_uploaded(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    conversation_id = "22222222-2222-2222-2222-222222222222"
    await db.create_conversation(conversation_id)
    service = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )

    result = await service.register_file_upload(
        conversation_id=conversation_id,
        filename="notes.txt",
        mime_type="text/plain",
        content=b"alpha\n\nbeta\n",
    )

    assert Path(result.blob_path).exists()
    record = await db.get_file_record(result.file_id)
    assert record is not None
    assert record.status == "QUEUED"
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert envelope.type == EventType.MEMORY_FILE_UPLOADED
    assert envelope.subject == conversation_id


@pytest.mark.asyncio
async def test_memory_source_consumer_file_uploaded_creates_file_text_source_and_requeues_source_registered(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    conversation_id = "33333333-3333-3333-3333-333333333333"
    await db.create_conversation(conversation_id)
    registration = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )
    await registration.register_file_upload(
        conversation_id=conversation_id,
        filename="spec.txt",
        mime_type="text/plain",
        content=b"Alpha paragraph.\n\nBeta paragraph.",
    )
    upload_event = db.outbox_rows[-1]["payload_json"]
    consumer = MemorySourceConsumer(
        db=db,
        chunker=SemanticChunkerAdapter(token_counter=lambda text: max(1, len(text.split()))),
        file_processor=FileProcessingAdapter(),
        ledger=ConsumerEffectLedger(db),
    )

    await consumer.handle(upload_event)

    assert any(row.get("source_kind") == "file_text" for row in db.demo_sources.values())
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert envelope.type == EventType.MEMORY_SOURCE_REGISTERED
    record = next(iter(db.file_records.values()))
    assert record["status"] == "PROCESSING"


@pytest.mark.asyncio
async def test_memory_source_consumer_source_registered_creates_chunk_manifests_and_chunk_event(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    conversation_id = "44444444-4444-4444-4444-444444444444"
    await db.create_conversation(conversation_id)
    registration = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )
    result = await registration.register_text(
        conversation_id=conversation_id,
        text="Primero.\n\nSegundo bloque con mas texto.",
        role="user",
        source_kind="chat_user",
    )
    source_event = db.outbox_rows[-1]["payload_json"]
    consumer = MemorySourceConsumer(
        db=db,
        chunker=SemanticChunkerAdapter(token_counter=lambda text: max(1, len(text.split()))),
        file_processor=FileProcessingAdapter(),
        ledger=ConsumerEffectLedger(db),
    )

    await consumer.handle(source_event)

    chunks = await db.list_chunk_records(conversation_id, source_id=result.source_id)
    assert chunks
    envelope = CloudEventEnvelope.model_validate(db.outbox_rows[-1]["payload_json"])
    assert envelope.type == EventType.MEMORY_CHUNK_CREATED
    payload_chunk_ids = envelope.data["chunk_ids"]
    assert len(payload_chunk_ids) == len(chunks)


@pytest.mark.asyncio
async def test_memory_source_consumer_skips_late_source_events_for_deleting_conversation(tmp_path: Path) -> None:
    db = InMemoryDemoDB()
    conversation_id = "45454545-4444-4444-4444-444444444444"
    await db.create_conversation(conversation_id)
    registration = SourceRegistrationService(
        db=db,
        lineage=DemoLineageService(db),
        workspace_root=tmp_path,
    )
    result = await registration.register_text(
        conversation_id=conversation_id,
        text="Primero. Segundo bloque.",
        role="user",
        source_kind="chat_user",
    )
    source_event = db.outbox_rows[-1]["payload_json"]
    db.conversations[conversation_id]["status"] = "DELETING"
    consumer = MemorySourceConsumer(
        db=db,
        chunker=SemanticChunkerAdapter(token_counter=lambda value: max(1, len(value.split()))),
        file_processor=FileProcessingAdapter(),
        ledger=ConsumerEffectLedger(db),
    )

    await consumer.handle(source_event)

    chunks = await db.list_chunk_records(conversation_id, source_id=result.source_id)
    assert chunks == []
