"""FileIngester — file upload extraction + chunking + ingestion (SPEC-5 split)."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cima_demo.domain.entities import FileRecord, IngestRequest
from cima_demo.domain.ports import ChunkingPort, CItemStorePort, FileProcessingPort, RelDBPort
from cima_demo.domain.value_objects import ChunkKind, ItemType
from cima_demo.memory.ingestion.web import (
    CHUNK_KIND_IMPORTANCE,
    _classify_chunk_kind,
    strip_images_from_text,
)

if TYPE_CHECKING:
    from cima_demo.application.stream_manager import StreamManager

log = logging.getLogger(__name__)


class FileIngester:
    """Extract text → chunk → ingest chain for uploaded files."""

    def __init__(
        self,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        file_processor: FileProcessingPort,
        chunking_port: ChunkingPort,
        stream_manager: "StreamManager",
        workspace_dir: Path | None = None,
        workspace_max_mb: int = 500,
        ingest_citem_fn: Any = None,
        lineage_service: Any | None = None,
    ) -> None:
        self._db = rel_db
        self._cstore = citem_store
        self._file_processor = file_processor
        self._chunker = chunking_port
        self._stream = stream_manager
        self._workspace_dir = workspace_dir
        self._workspace_max_bytes = workspace_max_mb * 1024 * 1024
        self._ingest_citem_fn = ingest_citem_fn
        self._extract_sem = asyncio.Semaphore(4)
        self._lineage = lineage_service

    async def ingest_files(
        self,
        files: list[tuple[bytes, str, str]],
        conversation_id: str,
        user_message: str,
        turn_id: str,
        progress_cb: object = None,
    ) -> None:
        from cima_demo.infrastructure.files.processor import infer_doc_type

        async def _emit(msg: str) -> None:
            if progress_cb is not None:
                try:
                    await progress_cb(msg)  # type: ignore[call-arg]
                except Exception:
                    pass

        expanded: list[tuple[bytes, str, str]] = []
        for file_bytes, filename, mime_type in files:
            if infer_doc_type(filename, mime_type) == "zip":
                await _emit(f"Descomprimiendo {filename}…")
                members = await asyncio.to_thread(
                    self._extract_zip,
                    file_bytes, filename, conversation_id,
                )
                expanded.extend(members)
            else:
                expanded.append((file_bytes, filename, mime_type))
        files = expanded

        for file_bytes, filename, mime_type in files:
            doc_type = infer_doc_type(filename, mime_type)
            file_kb = len(file_bytes) / 1024
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            record = FileRecord(
                conversation_id=conversation_id,
                filename=filename,
                mime_type=mime_type,
                size_bytes=len(file_bytes),
                content_hash=file_hash,
                status="QUEUED",
            )
            try:
                await self._db.save_file_record(record)
            except Exception:
                log.debug("save_file_record failed for %s (non-fatal)", filename)

            try:
                await _emit(f"Extrayendo texto de {filename} ({file_kb:.0f} KB)…")
                await self._db.update_file_record(record.file_id, status="PROCESSING")

                async with self._extract_sem:
                    text: str = await asyncio.to_thread(
                        self._file_processor.extract_text,
                        file_bytes,
                        filename,
                        mime_type,
                    )

                text = strip_images_from_text(text)
                _demo_source = None
                _demo_full_span = None
                if self._lineage is not None:
                    try:
                        _demo_source, _demo_full_span = await self._lineage.register_text_source(
                            conversation_id=conversation_id,
                            source_kind="file_text",
                            role=None,
                            display_text=None,
                            process_text=text,
                            origin_ref=record.file_id,
                            metadata={
                                "filename": filename,
                                "mime_type": mime_type,
                                "doc_type": doc_type,
                            },
                        )
                    except Exception:
                        log.exception("demo lineage source registration failed for %s", filename)


                if not text.strip():
                    log.warning("ingest_files: empty extraction for %s", filename)
                    await self._db.update_file_record(
                        record.file_id, status="FAILED",
                        error_message="extraction produced no text",
                    )
                    continue

                meta_request = IngestRequest(
                    content=(
                        f"Document ingested: «{filename}» "
                        f"(type: {doc_type}, size: {file_kb:.1f} KB, "
                        f"chars extracted: {len(text)})"
                    ),
                    item_type=ItemType.FACT,
                    phase_ingested="IDLE",
                    actor="user",
                    conversation_id=conversation_id,
                    motivation=f"File upload metadata anchor for {filename}",
                    confidence=1.0,
                    source_id=_demo_source.source_id if _demo_source is not None else None,
                    source_span_ids=[_demo_full_span.span_id] if _demo_full_span is not None else [],
                    lineage_meta={"kind": "file_metadata", "filename": filename},
                )
                meta_citem = await self._ingest_citem_fn(meta_request, skip_conflict_detection=True)
                if meta_citem is None:
                    await _emit(f"{filename} ya estaba indexado, omitiendo.")
                    continue

                await _emit(f"Segmentando {filename} ({len(text):,} chars)…")
                chunks = await self._chunker.chunk(text, filename, doc_type)
                _chunk_spans: dict[int, Any] = {}
                if self._lineage is not None and _demo_source is not None:
                    try:
                        _chunk_spans = await self._lineage.register_spans_from_chunks(
                            conversation_id=conversation_id,
                            source_id=_demo_source.source_id,
                            process_text=text,
                            chunks=chunks,
                        )
                    except Exception:
                        log.exception("demo lineage span registration failed for %s", filename)

                prev_id: str = meta_citem.citem_id
                chunk_citem_ids: list[str] = [meta_citem.citem_id]
                n_skipped = 0
                for chunk in chunks:
                    chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                    if await self._cstore.exists_by_hash(chunk_hash, conversation_id):
                        n_skipped += 1
                        continue
                    loc_parts: list[str] = [f"chunk {chunk.index + 1}/{len(chunks)}"]
                    if chunk.page_num is not None:
                        loc_parts.append(f"page {chunk.page_num}")
                    if chunk.section_hint:
                        loc_parts.append(f"section «{chunk.section_hint[:60]}»")
                    motivation = f"{filename} — {', '.join(loc_parts)}"

                    kind = _classify_chunk_kind(chunk.text)
                    try:
                        ck = ChunkKind(kind)
                        item_type = ItemType.FACT if ck.evidence_eligible else ItemType.OBSERVATION
                        importance = CHUNK_KIND_IMPORTANCE.get(kind, 0.30)
                    except ValueError:
                        item_type = ItemType.OBSERVATION
                        importance = 0.30

                    _span = _chunk_spans.get(chunk.index)
                    request = IngestRequest(
                        content=chunk.text,
                        item_type=item_type,
                        phase_ingested="IDLE",
                        actor="user",
                        conversation_id=conversation_id,
                        motivation=motivation,
                        confidence=1.0,
                        importance_override=importance,
                        dependency_ids=[prev_id],
                        chunk_kind=kind,
                        source_id=_demo_source.source_id if _demo_source is not None else None,
                        source_span_ids=[_span.span_id] if _span is not None else [],
                        lineage_meta={
                            "filename": filename,
                            "doc_type": doc_type,
                            "page_num": getattr(chunk, "page_num", None),
                            "section_hint": getattr(chunk, "section_hint", None),
                            "chunk_index": chunk.index,
                        },
                    )
                    chunk_citem = await self._ingest_citem_fn(request, skip_conflict_detection=True)
                    if chunk_citem is None:
                        n_skipped += 1
                        continue
                    prev_id = chunk_citem.citem_id
                    chunk_citem_ids.append(chunk_citem.citem_id)

                n_new = len(chunks) - n_skipped
                await _emit(
                    f"{filename} — {n_new} segmentos indexados"
                    + (f", {n_skipped} duplicados omitidos" if n_skipped else "") + "."
                )

                summary_request = IngestRequest(
                    content=(
                        f"File «{filename}» has been processed and stored in memory. "
                        f"Type: {doc_type}. {len(chunks)} chunk(s) indexed. "
                        f"You can now answer questions about its content."
                    ),
                    item_type=ItemType.OBSERVATION,
                    phase_ingested="IDLE",
                    actor="system",
                    conversation_id=conversation_id,
                    motivation=f"File ingestion complete: {filename}",
                    confidence=1.0,
                    dependency_ids=[meta_citem.citem_id] if meta_citem else [],
                )
                await self._ingest_citem_fn(summary_request, skip_conflict_detection=True)

                await self._db.update_file_record(
                    record.file_id,
                    status="READY",
                    chunk_count=len(chunks),
                    citem_ids=chunk_citem_ids,
                )

                log.info(
                    "ingest_files: %s (%s) → %d chunks ingested, %d skipped (duplicate)",
                    filename, doc_type, n_new, n_skipped,
                )

            except Exception as e:
                log.warning("ingest_files failed for %s: %s", filename, e)
                try:
                    await self._db.update_file_record(
                        record.file_id, status="FAILED",
                        error_message=str(e)[:500],
                    )
                except Exception:
                    pass

    def _extract_zip(
        self,
        zip_bytes: bytes,
        zip_filename: str,
        conversation_id: str,
    ) -> list[tuple[bytes, str, str]]:
        import io
        import mimetypes
        from cima_demo.infrastructure.files.processor import infer_doc_type

        _BINARY_TYPES = frozenset({"image", "zip"})
        _MAX_BYTES = self._workspace_max_bytes

        ws_dest: Path | None = None
        if self._workspace_dir is not None:
            ws_dest = self._workspace_dir / conversation_id
            ws_dest.mkdir(parents=True, exist_ok=True)

        results: list[tuple[bytes, str, str]] = []
        total_bytes = 0
        ingestable_names: list[str] = []
        binary_names: list[str] = []

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for info in zf.infolist():
                    name = info.filename
                    parts = Path(name).parts
                    if any(p in ("..", "/") or Path(p).is_absolute() for p in parts):
                        log.warning("ZIP: skipping unsafe member path: %s", name)
                        continue
                    if name.endswith("/"):
                        if ws_dest is not None:
                            (ws_dest / name).mkdir(parents=True, exist_ok=True)
                        continue

                    total_bytes += info.file_size
                    if total_bytes > _MAX_BYTES:
                        log.warning(
                            "ZIP: extraction aborted — extracted size exceeded limit (%d MB)",
                            _MAX_BYTES // (1024 * 1024),
                        )
                        break

                    member_bytes = zf.read(name)

                    if ws_dest is not None:
                        dest_file = (ws_dest / name).resolve()
                        if not dest_file.is_relative_to(ws_dest.resolve()):
                            log.warning("ZIP: resolved path escapes workspace, skipping: %s", name)
                            continue
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        dest_file.write_bytes(member_bytes)

                    mime, _ = mimetypes.guess_type(name)
                    mime = mime or "application/octet-stream"
                    doc_type = infer_doc_type(name, mime)
                    if doc_type in _BINARY_TYPES:
                        binary_names.append(name)
                    else:
                        results.append((member_bytes, name, mime))
                        ingestable_names.append(name)

        except zipfile.BadZipFile as exc:
            log.warning("ZIP: bad zip file %s: %s", zip_filename, exc)
            return []

        ws_note = (
            f"Workspace path: {ws_dest} (ephemeral — cleared daily at 00:00 UTC). "
            "Use workspace_ls to verify current state before executing scripts."
            if ws_dest is not None
            else "No workspace configured — files not available for execution."
        )
        obs_parts = [f"ZIP «{zip_filename}» extracted."]
        if ingestable_names:
            obs_parts.append(
                f"Ingested into memory (permanent, recall via RAG): "
                + ", ".join(ingestable_names[:20])
                + ("..." if len(ingestable_names) > 20 else "")
            )
        if binary_names:
            obs_parts.append(
                f"Binary files (not ingested, workspace only): "
                + ", ".join(binary_names[:20])
                + ("..." if len(binary_names) > 20 else "")
            )
        obs_parts.append(ws_note)
        obs_parts.append(
            "IMPORTANT: memory recall works regardless of workspace state; "
            "script execution requires workspace files to exist — verify with workspace_ls."
        )
        log.info(
            "ZIP %s: %d ingestable, %d binary, dest=%s",
            zip_filename, len(ingestable_names), len(binary_names), ws_dest,
        )

        obs_text = "\n".join(obs_parts)
        results.append((obs_text.encode(), f"_zip_obs_{zip_filename}.txt", "text/plain"))

        return results
