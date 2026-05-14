"""MemoryService — thin facade delegating to IngestionService, LifecycleService, SummaryService.

Re-exports module-level symbols consumed by external callers:
  WebIngestionResult, strip_images_from_text, _classify_chunk_kind, _score_relevance
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cima_demo.application.stream_manager import StreamManager

from cima_demo.domain.entities import (
    CItem,
    ContextView,
    IngestRequest,
    TaskMemory,
)
from cima_demo.domain.ports import (
    ChunkingPort,
    CItemStorePort,
    FileProcessingPort,
    LLMPort,
    NLIPort,
    RelDBPort,
)
from cima_demo.domain.value_objects import ForgetParams, PromotionPolicy

# ── Re-export module-level symbols used by external callers ───────────────────
# engine.py: from cima_demo.memory.service import WebIngestionResult, strip_images_from_text
# tool_dispatcher.py: from cima_demo.memory.service import strip_images_from_text
# tests/: _classify_chunk_kind, _score_relevance, strip_images_from_text
from cima_demo.memory.ingestion import (  # noqa: F401  (re-export)
    WebIngestionResult,
    _classify_chunk_kind,
    _score_relevance,
    strip_images_from_text,
)

log = logging.getLogger(__name__)


class MemoryService:
    """Única clase con visibilidad sobre RelDBPort Y CItemStorePort.

    Thin facade — delegates all logic to:
      IngestionService  — C-Item ingestion + conflict detection
      LifecycleService  — forget cycle, dedup, promotions
      SummaryService    — context refresh and L1/L2 summarization
    """

    def __init__(
        self,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        llm_port: LLMPort,
        file_processor: FileProcessingPort,
        chunking_port: ChunkingPort,
        stream_manager: "StreamManager",
        forget_params: ForgetParams | None = None,
        promotion_policy: PromotionPolicy | None = None,
        workspace_dir: Path | None = None,
        workspace_max_mb: int = 500,
        nli_port: NLIPort | None = None,
        lineage_service: Any | None = None,
    ) -> None:
        from cima_demo.memory.ingestion import IngestionService
        from cima_demo.memory.lifecycle import LifecycleService
        from cima_demo.memory.summary import SummaryService

        _facade = self

        async def _ingest_via_facade(request: object, skip_conflict_detection: bool = False) -> object:
            return await _facade.ingest_citem(request, skip_conflict_detection)  # type: ignore[arg-type]

        self._lineage = lineage_service
        self._ingestion = IngestionService(
            rel_db=rel_db,
            citem_store=citem_store,
            llm_port=llm_port,
            file_processor=file_processor,
            chunking_port=chunking_port,
            stream_manager=stream_manager,
            workspace_dir=workspace_dir,
            workspace_max_mb=workspace_max_mb,
            nli_port=nli_port,
            forget_params=forget_params,
            promotion_policy=promotion_policy,
            ingest_citem_fn=_ingest_via_facade,
            lineage_service=lineage_service,
        )
        self._lifecycle = LifecycleService(
            rel_db=rel_db,
            citem_store=citem_store,
            stream_manager=stream_manager,
            forget_params=forget_params,
            promotion_policy=promotion_policy,
        )
        self._summary = SummaryService(
            rel_db=rel_db,
            citem_store=citem_store,
            llm_port=llm_port,
            stream_manager=stream_manager,
            forget_params=forget_params,
            lineage_service=lineage_service,
        )

    # ── Ingestion delegation ──────────────────────────────────────────────────

    async def ingest_citem(
        self,
        request: IngestRequest,
        skip_conflict_detection: bool = False,
    ) -> CItem | None:
        return await self._ingestion.ingest_citem(request, skip_conflict_detection)

    async def ingest_batch(
        self,
        conclusions: list[dict[str, Any]],
        phase: str,
        conversation_id: str,
        turn_id: str,
    ) -> None:
        return await self._ingestion.ingest_batch(conclusions, phase, conversation_id, turn_id)

    async def ingest_files(
        self,
        files: list[tuple[bytes, str, str]],
        conversation_id: str,
        user_message: str,
        turn_id: str,
        progress_cb: object = None,
    ) -> None:
        return await self._ingestion.ingest_files(
            files, conversation_id, user_message, turn_id, progress_cb
        )

    async def ingest_web_content(
        self,
        url: str,
        text: str,
        title: str,
        conversation_id: str,
        phase: str,
        objective: str | None = None,
    ) -> "WebIngestionResult":
        return await self._ingestion.ingest_web_content(
            url, text, title, conversation_id, phase, objective
        )

    async def resolve_conflict(self, citem_id: str) -> None:
        return await self._ingestion.resolve_conflict(citem_id)

    async def fetch_by_conversation(
        self,
        conversation_id: str,
        scope_status: str = "active",
    ) -> list[CItem]:
        """Fetch all C-Items for a conversation from the citem store."""
        return await self._lifecycle._cstore.fetch_by_conversation(
            conversation_id, scope_status=scope_status
        )

    # ── Lifecycle delegation ──────────────────────────────────────────────────

    async def run_forget_cycle(self, conversation_id: str) -> tuple[int, int]:
        return await self._lifecycle.run_forget_cycle(conversation_id)

    async def run_forget_cycle_detailed(self, conversation_id: str) -> dict[str, Any]:
        return await self._lifecycle.run_forget_cycle_detailed(conversation_id)

    async def run_dedup_cycle(self, conversation_id: str) -> int:
        return await self._lifecycle.run_dedup_cycle(conversation_id)

    async def run_dedup_cycle_detailed(self, conversation_id: str) -> dict[str, Any]:
        return await self._lifecycle.run_dedup_cycle_detailed(conversation_id)

    async def check_promotions(
        self,
        conversation_id: str,
        chm_reference_counts: dict[str, int],
    ) -> tuple[int, int]:
        return await self._lifecycle.check_promotions(conversation_id, chm_reference_counts)

    async def check_promotions_detailed(
        self,
        conversation_id: str,
        chm_reference_counts: dict[str, int],
    ) -> dict[str, Any]:
        return await self._lifecycle.check_promotions_detailed(conversation_id, chm_reference_counts)

    # ── Summary delegation ────────────────────────────────────────────────────

    async def trigger_l2_check(self, conversation_id: str) -> bool:
        """A-10 L2 AutoPromote check — callable from background workers (SPEC-6)."""
        return await self._summary.trigger_l2_check(conversation_id)

    async def refresh_context(
        self,
        context_view: ContextView,
        task_memory: TaskMemory,
        conversation_id: str,
        current_goal: str | None = None,
        active_step: str | None = None,
        phase: str | None = None,
        semantic: bool = False,
        force_ids: set[str] | None = None,
    ) -> tuple[ContextView, int, str]:
        return await self._summary.refresh_context(
            context_view,
            task_memory,
            conversation_id,
            current_goal,
            active_step,
            phase,
            semantic,
            force_ids,
        )
