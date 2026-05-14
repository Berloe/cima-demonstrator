from __future__ import annotations

"""Async source/file registration foundation for the witness backend.

This tranche adds the missing first leg of the approved async plane:
source/file registration writes durable state and outbox entries, while a worker
converts registered sources into chunk manifests without putting heavy parsing in
request handlers.
"""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cima_demo.demo.lineage import DemoLineageService
from cima_demo.domain.entities import FileRecord
from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.lifecycle_guard import complete_if_conversation_not_active
from cima_demo.witness_backend.events import (
    ChunkCreatedData,
    CloudEventEnvelope,
    EventType,
    FileUploadedData,
    Producer,
    SourceRegisteredData,
    TraceContext,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ChunkerLike(Protocol):
    async def chunk(self, text: str, filename: str, doc_type: str) -> list[Any]: ...


class FileProcessorLike(Protocol):
    def extract_text(self, content: bytes, filename: str, mime_type: str) -> str: ...


class SourceStoreLike(Protocol):
    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...

    async def save_file_record(self, record: FileRecord) -> None: ...

    async def update_file_record(
        self,
        file_id: str,
        *,
        status: str,
        chunk_count: int = 0,
        citem_ids: list[str] | None = None,
        error_message: str | None = None,
    ) -> None: ...

    async def get_file_record(self, file_id: str) -> FileRecord | None: ...

    async def load_demo_sources(self, conversation_id: str, source_ids: list[str]) -> list[dict[str, Any]]: ...

    async def save_chunk_record(self, chunk_json: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class RegisterTextResult:
    source_id: str
    outbox_id: int
    span_id: str | None


@dataclass(frozen=True, slots=True)
class RegisterFileResult:
    file_id: str
    outbox_id: int
    blob_path: str


class SourceRegistrationService:
    def __init__(
        self,
        *,
        db: SourceStoreLike,
        lineage: DemoLineageService,
        workspace_root: Path,
        producer: Producer = Producer.CIMA_API,
    ) -> None:
        self._db = db
        self._lineage = lineage
        self._workspace_root = workspace_root
        self._producer = producer

    @staticmethod
    def _canonical_source_kind(source_kind: str) -> str:
        aliases = {
            "file": "file_text",
            "dataset_document": "file_text",
            "document": "file_text",
            "doc": "file_text",
            "text": "chat_user",
            "chat": "chat_user",
            "assistant": "chat_assistant",
            "user": "chat_user",
        }
        return aliases.get(source_kind, source_kind)

    async def register_text(
        self,
        *,
        conversation_id: str,
        text: str,
        role: str | None,
        source_kind: str,
        external_provider: str | None = None,
        external_conversation_id: str | None = None,
        external_message_id: str | None = None,
        displayable: bool = True,
        processable: bool = True,
        trace: TraceContext | None = None,
    ) -> RegisterTextResult:
        canonical_source_kind = self._canonical_source_kind(source_kind)
        source, full_span = await self._lineage.register_text_source(
            conversation_id=conversation_id,
            source_kind=canonical_source_kind,
            role=role,
            display_text=text if displayable else None,
            process_text=text if processable else None,
            origin_ref=external_message_id,
            metadata={
                "external_provider": external_provider,
                "external_conversation_id": external_conversation_id,
                "external_message_id": external_message_id,
            },
        )
        payload = SourceRegisteredData(
            source_id=uuid.UUID(source.source_id),
            kind=canonical_source_kind,  # type: ignore[arg-type]
            external_provider=external_provider,
            external_conversation_id=external_conversation_id,
            external_message_id=external_message_id,
            revision_no=0,
            displayable=displayable,
            processable=processable,
        )
        envelope = CloudEventEnvelope(
            type=EventType.MEMORY_SOURCE_REGISTERED,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.memory.source.registered.v1.json",
            data=payload.model_dump(mode="json"),
        )
        outbox_id = await self._db.append_outbox_event(
            topic=TOPICS.memory_events,
            message_key=conversation_key(conversation_id),
            payload_json=envelope.model_dump(mode="json"),
            headers_json=_trace_headers(trace),
        )
        return RegisterTextResult(source_id=source.source_id, outbox_id=outbox_id, span_id=full_span.span_id if full_span else None)

    async def register_file_upload(
        self,
        *,
        conversation_id: str,
        filename: str,
        mime_type: str,
        content: bytes,
        trace: TraceContext | None = None,
    ) -> RegisterFileResult:
        file_id = str(uuid.uuid4())
        blob_path = self._persist_blob(conversation_id=conversation_id, file_id=file_id, filename=filename, content=content)
        record = FileRecord(
            file_id=file_id,
            conversation_id=conversation_id,
            filename=filename,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
            status="QUEUED",
            blob_path=blob_path,
        )
        await self._db.save_file_record(record)
        payload = FileUploadedData(
            file_id=uuid.UUID(file_id),
            filename=filename,
            mime_type=record.mime_type,
            sha256=record.content_hash,
            size_bytes=record.size_bytes,
        )
        envelope = CloudEventEnvelope(
            type=EventType.MEMORY_FILE_UPLOADED,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.memory.file.uploaded.v1.json",
            data=payload.model_dump(mode="json"),
        )
        outbox_id = await self._db.append_outbox_event(
            topic=TOPICS.memory_events,
            message_key=conversation_key(conversation_id),
            payload_json=envelope.model_dump(mode="json"),
            headers_json=_trace_headers(trace),
        )
        return RegisterFileResult(file_id=file_id, outbox_id=outbox_id, blob_path=blob_path)

    def _persist_blob(self, *, conversation_id: str, file_id: str, filename: str, content: bytes) -> str:
        safe_name = _SAFE_NAME_RE.sub("_", filename).strip("._") or "upload.bin"
        target_dir = self._workspace_root / conversation_id / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{file_id}-{safe_name}"
        target.write_bytes(content)
        return str(target)


class MemorySourceConsumer:
    def __init__(
        self,
        *,
        db: SourceStoreLike,
        chunker: ChunkerLike,
        file_processor: FileProcessorLike,
        ledger: ConsumerEffectLedger,
        producer: Producer = Producer.CIMA_WORKER,
        chunker_version: int = 1,
        normalizer_version: int = 1,
    ) -> None:
        self._db = db
        self._chunker = chunker
        self._file_processor = file_processor
        self._ledger = ledger
        self._producer = producer
        self._chunker_version = chunker_version
        self._normalizer_version = normalizer_version
        self._lineage = DemoLineageService(db)  # deterministic adapter over persisted demo sources/spans

    async def handle(self, payload_json: dict[str, Any]) -> None:
        envelope = CloudEventEnvelope.model_validate(payload_json)
        if envelope.type == EventType.MEMORY_FILE_UPLOADED:
            await self._handle_file_uploaded(envelope)
            return
        if envelope.type == EventType.MEMORY_SOURCE_REGISTERED:
            await self._handle_source_registered(envelope)
            return

    async def _handle_file_uploaded(self, envelope: CloudEventEnvelope) -> None:
        data = FileUploadedData.model_validate(envelope.data)
        effect_key = f"file-uploaded:{data.file_id}:parse-v1"
        key = ConsumerEffectKey(
            consumer_name="memory-source-consumer",
            event_id=str(envelope.id),
            effect_key=effect_key,
        )
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        record = await self._db.get_file_record(str(data.file_id))
        if record is None or not record.blob_path:
            await self._ledger.complete(key, details_json={"status": "missing_file_record"})
            return
        await self._db.update_file_record(str(data.file_id), status="PROCESSING")
        try:
            file_bytes = Path(record.blob_path).read_bytes()
            extracted = self._file_processor.extract_text(file_bytes, record.filename, record.mime_type)
            source, _full_span = await self._lineage.register_text_source(
                conversation_id=record.conversation_id,
                source_kind="file_text",
                role=None,
                display_text=None,
                process_text=extracted,
                origin_ref=record.file_id,
                metadata={
                    "filename": record.filename,
                    "mime_type": record.mime_type,
                    "content_hash": record.content_hash,
                },
            )
            payload = SourceRegisteredData(
                source_id=uuid.UUID(source.source_id),
                kind="file_text",
                external_provider="cima-api",
                external_conversation_id=record.conversation_id,
                external_message_id=record.file_id,
                revision_no=0,
                displayable=False,
                processable=True,
            )
            outbox = CloudEventEnvelope(
                type=EventType.MEMORY_SOURCE_REGISTERED,
                source=self._producer,
                subject=record.conversation_id,
                dataschema="schemas/cima.memory.source.registered.v1.json",
                data=payload.model_dump(mode="json"),
            )
            await self._db.append_outbox_event(
                topic=TOPICS.memory_events,
                message_key=conversation_key(record.conversation_id),
                payload_json=outbox.model_dump(mode="json"),
            )
            await self._ledger.complete(key, details_json={"status": "queued_source", "source_id": source.source_id})
        except Exception as exc:
            await self._db.update_file_record(str(data.file_id), status="FAILED", error_message=type(exc).__name__)
            raise

    async def _handle_source_registered(self, envelope: CloudEventEnvelope) -> None:
        data = SourceRegisteredData.model_validate(envelope.data)
        if not data.processable:
            return
        effect_key = f"source-registered:{data.source_id}:chunk-v{self._chunker_version}"
        key = ConsumerEffectKey(
            consumer_name="memory-source-consumer",
            event_id=str(envelope.id),
            effect_key=effect_key,
        )
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        rows = await self._db.load_demo_sources(envelope.subject, [str(data.source_id)])
        if not rows:
            await self._ledger.complete(key, details_json={"status": "missing_source"})
            return
        row = rows[0]
        text = row.get("process_text") or row.get("display_text") or ""
        if not text.strip():
            await self._ledger.complete(key, details_json={"status": "empty_source"})
            return
        meta = dict(row.get("metadata") or {})
        filename = str(meta.get("filename") or f"source-{data.source_id}")
        doc_type = str(meta.get("doc_type") or ("text" if data.kind != "chat_user" and data.kind != "chat_assistant" else "chat"))
        chunks = await self._chunker.chunk(text, filename, doc_type)
        span_map = await self._lineage.register_spans_from_chunks(
            conversation_id=envelope.subject,
            source_id=str(data.source_id),
            process_text=text,
            chunks=chunks,
        )
        chunk_ids: list[uuid.UUID] = []
        origin_file_id = row.get("origin_ref")
        kind = "doc_chunk" if data.kind == "file_text" else "chat_chunk"
        for chunk in chunks:
            chunk_id = uuid.uuid4()
            span = span_map.get(getattr(chunk, "index", 0))
            chunk_ids.append(chunk_id)
            await self._db.save_chunk_record(
                {
                    "chunk_id": str(chunk_id),
                    "conversation_id": envelope.subject,
                    "source_id": str(data.source_id),
                    "file_id": origin_file_id,
                    "source_span_id": span.span_id if span is not None else None,
                    "chunk_kind": kind,
                    "chunk_index": int(getattr(chunk, "index", len(chunk_ids) - 1)),
                    "page_num": getattr(chunk, "page_num", None),
                    "section_hint": getattr(chunk, "section_hint", None),
                    "normalizer_version": self._normalizer_version,
                    "chunker_version": self._chunker_version,
                    "vector_state": "NONE",
                }
            )
        if origin_file_id:
            await self._db.update_file_record(str(origin_file_id), status="READY", chunk_count=len(chunk_ids))
        payload = ChunkCreatedData(
            chunk_ids=chunk_ids,
            chunker_version=self._chunker_version,
            normalizer_version=self._normalizer_version,
            origin_kind="file_text" if data.kind == "file_text" else "chat",
        )
        outbox = CloudEventEnvelope(
            type=EventType.MEMORY_CHUNK_CREATED,
            source=self._producer,
            subject=envelope.subject,
            dataschema="schemas/cima.memory.chunk.created.v1.json",
            data=payload.model_dump(mode="json"),
        )
        await self._db.append_outbox_event(
            topic=TOPICS.memory_events,
            message_key=conversation_key(envelope.subject),
            payload_json=outbox.model_dump(mode="json"),
        )
        await self._ledger.complete(key, details_json={"status": "chunked", "chunk_count": len(chunk_ids)})


def _trace_headers(trace: TraceContext | None) -> dict[str, Any]:
    if trace is None:
        return {}
    return json.loads(trace.model_dump_json())
