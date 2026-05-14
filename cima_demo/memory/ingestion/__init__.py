"""IngestionService — backward-compatible facade over SPEC-5 sub-modules.

Sub-modules:
  ingestion.core     — IngestionCore (ingest_citem, ingest_batch)
  ingestion.web      — WebIngester (ingest_web_content) + helpers
  ingestion.files    — FileIngester (ingest_files, _extract_zip)
  ingestion.conflict — ConflictDetector (detect, resolve, NLI)

Re-exports for backward compatibility (used by service.py, engine.py, tests):
  WebIngestionResult, strip_images_from_text, _classify_chunk_kind, _score_relevance
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cima_demo.application.stream_manager import StreamManager

from cima_demo.domain.ports import (
    ChunkingPort,
    CItemStorePort,
    FileProcessingPort,
    LLMPort,
    NLIPort,
    RelDBPort,
)
from cima_demo.domain.value_objects import ForgetParams, PromotionPolicy

from cima_demo.memory.ingestion.conflict import ConflictDetector
from cima_demo.memory.ingestion.core import IngestionCore
from cima_demo.memory.ingestion.files import FileIngester
from cima_demo.memory.ingestion.web import (
    WebIngester,
    WebIngestionResult,
    _classify_chunk_kind,
    _score_relevance,
    strip_images_from_text,
)

__all__ = [
    "IngestionService",
    "WebIngestionResult",
    "strip_images_from_text",
    "_classify_chunk_kind",
    "_score_relevance",
    "ConflictDetector",
    "IngestionCore",
    "FileIngester",
    "WebIngester",
]


class IngestionService:
    """Backward-compatible facade delegating to IngestionCore, WebIngester,
    FileIngester, and ConflictDetector.

    Drop-in replacement: same __init__ signature, same method surface.
    """

    def __init__(
        self,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        llm_port: LLMPort,
        file_processor: FileProcessingPort,
        chunking_port: ChunkingPort,
        stream_manager: "StreamManager",
        workspace_dir: Path | None = None,
        workspace_max_mb: int = 500,
        nli_port: NLIPort | None = None,
        forget_params: ForgetParams | None = None,
        promotion_policy: PromotionPolicy | None = None,
        ingest_citem_fn: Any | None = None,
        lineage_service: Any | None = None,
    ) -> None:
        self._conflict_detector = ConflictDetector(
            rel_db=rel_db,
            citem_store=citem_store,
            nli_port=nli_port,
            llm_port=llm_port,
        )
        self._core = IngestionCore(
            citem_store=citem_store,
            llm_port=llm_port,
            conflict_detector=self._conflict_detector,
            ingest_citem_fn=ingest_citem_fn,
            lineage_recorder=lineage_service,
        )
        self._web = WebIngester(
            chunking_port=chunking_port,
            citem_store=citem_store,
            ingest_citem_fn=ingest_citem_fn if ingest_citem_fn is not None else self._core.ingest_citem,
        )
        self._files = FileIngester(
            rel_db=rel_db,
            citem_store=citem_store,
            file_processor=file_processor,
            chunking_port=chunking_port,
            stream_manager=stream_manager,
            workspace_dir=workspace_dir,
            workspace_max_mb=workspace_max_mb,
            ingest_citem_fn=ingest_citem_fn if ingest_citem_fn is not None else self._core.ingest_citem,
            lineage_service=lineage_service,
        )

    # ── Delegation ───────────────────────────────────────────────────────────

    async def ingest_citem(self, request: Any, skip_conflict_detection: bool = False) -> Any:
        return await self._core.ingest_citem(request, skip_conflict_detection)

    async def ingest_batch(self, conclusions: list[dict[str, Any]], phase: str, conversation_id: str, turn_id: str) -> None:
        return await self._core.ingest_batch(conclusions, phase, conversation_id, turn_id)

    async def ingest_files(self, files: list[tuple[bytes, str, str]], conversation_id: str, user_message: str, turn_id: str, progress_cb: object = None) -> None:
        return await self._files.ingest_files(files, conversation_id, user_message, turn_id, progress_cb)

    async def ingest_web_content(self, url: str, text: str, title: str, conversation_id: str, phase: str, objective: str | None = None) -> WebIngestionResult:
        return await self._web.ingest_web_content(url, text, title, conversation_id, phase, objective)

    async def resolve_conflict(self, citem_id: str) -> None:
        return await self._conflict_detector.resolve_conflict(citem_id)
