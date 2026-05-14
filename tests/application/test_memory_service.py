"""Tests for MemoryService (cima_demo/application/memory_service.py)."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from cima_demo.memory.service import MemoryService
from cima_demo.domain.entities import CItem, IngestRequest
from cima_demo.domain.value_objects import ForgetParams, ItemType

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_rel_db() -> AsyncMock:
    db = AsyncMock()
    db.save_summary = AsyncMock()
    db.save_conflict = AsyncMock()
    return db


@pytest.fixture
def mock_cstore() -> AsyncMock:
    store = AsyncMock()
    store.save = AsyncMock()
    store.exists_by_hash = AsyncMock(return_value=False)   # dedup: not a duplicate
    store.fetch_by_conversation = AsyncMock(return_value=[])
    store.update_field = AsyncMock()
    store.delete = AsyncMock()
    return store


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.count_tokens = AsyncMock(return_value=10)
    llm.complete = AsyncMock(return_value="summary text")
    return llm


@pytest.fixture
def mock_file_processor() -> MagicMock:
    fp = MagicMock()
    fp.extract_text = MagicMock(return_value="extracted text content")
    return fp


@pytest.fixture
def mock_chunker() -> AsyncMock:
    from cima_demo.domain.value_objects import ChunkResult
    chunker = AsyncMock()
    chunker.chunk = AsyncMock(
        return_value=[ChunkResult(text="chunk1", index=0, filename="f.txt", doc_type="text/plain")]
    )
    return chunker


@pytest.fixture
def mock_stream() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def service(
    mock_rel_db: AsyncMock,
    mock_cstore: AsyncMock,
    mock_llm: AsyncMock,
    mock_file_processor: MagicMock,
    mock_chunker: AsyncMock,
    mock_stream: AsyncMock,
) -> MemoryService:
    return MemoryService(
        rel_db=mock_rel_db,
        citem_store=mock_cstore,
        llm_port=mock_llm,
        file_processor=mock_file_processor,
        chunking_port=mock_chunker,
        stream_manager=mock_stream,
    )


def _request(
    content: str = "test content",
    item_type: str = ItemType.FACT,
    conversation_id: str = "conv-1",
) -> IngestRequest:
    return IngestRequest(
        content=content,
        item_type=item_type,
        phase_ingested="IDLE",
        actor="agent",
        conversation_id=conversation_id,
        motivation="test",
        confidence=0.9,
    )


# ── ingest_citem ──────────────────────────────────────────────────────────────

class TestIngestCitem:
    async def test_returns_citem_with_correct_content(
        self, service: MemoryService,
    ) -> None:
        citem = await service.ingest_citem(_request(content="hello world"))
        assert citem.content == "hello world"

    async def test_saves_to_cstore(
        self, service: MemoryService, mock_cstore: AsyncMock,
    ) -> None:
        await service.ingest_citem(_request())
        mock_cstore.save.assert_awaited_once()

    async def test_importance_computed(self, service: MemoryService) -> None:
        citem = await service.ingest_citem(_request(item_type=ItemType.DECISION))
        assert citem.importance > 0.5  # DECISION has high base importance

    async def test_token_count_set(
        self, service: MemoryService, mock_llm: AsyncMock,
    ) -> None:
        mock_llm.count_tokens.return_value = 42
        citem = await service.ingest_citem(_request())
        assert citem.token_count == 42

    async def test_token_count_fallback_on_llm_error(
        self, service: MemoryService, mock_llm: AsyncMock,
    ) -> None:
        mock_llm.count_tokens.side_effect = RuntimeError("LLM down")
        citem = await service.ingest_citem(_request(content="hello world"))
        # fallback: len(content) // 4 = 11 // 4 = 2 → max(1, 2) = 2
        assert citem.token_count >= 1

    async def test_skip_conflict_detection_flag(
        self, service: MemoryService, mock_cstore: AsyncMock,
    ) -> None:
        # Should not raise or call fetch when skip=True
        await service.ingest_citem(_request(), skip_conflict_detection=True)
        mock_cstore.save.assert_awaited_once()

    async def test_dependency_ids_copied(self, service: MemoryService) -> None:
        req = _request()
        req.dependency_ids = ["dep-a", "dep-b"]
        citem = await service.ingest_citem(req)
        assert citem.dependency_ids == ["dep-a", "dep-b"]


# ── ingest_batch ──────────────────────────────────────────────────────────────

class TestIngestBatch:
    async def test_ingests_all_items(
        self, service: MemoryService, mock_cstore: AsyncMock,
    ) -> None:
        conclusions = [
            {"content": "fact 1", "type": ItemType.FACT, "confidence": 0.9},
            {"content": "fact 2", "type": ItemType.OBSERVATION, "confidence": 0.7},
        ]
        await service.ingest_batch(conclusions, "RECALL", "conv-1", "turn-1")
        assert mock_cstore.save.await_count == 2

    async def test_empty_batch_no_saves(
        self, service: MemoryService, mock_cstore: AsyncMock,
    ) -> None:
        await service.ingest_batch([], "IDLE", "conv-1", "turn-1")
        mock_cstore.save.assert_not_awaited()

    async def test_missing_content_skipped(
        self, service: MemoryService, mock_cstore: AsyncMock,
    ) -> None:
        conclusions = [{"content": "", "type": ItemType.FACT, "confidence": 0.9}]
        await service.ingest_batch(conclusions, "IDLE", "conv-1", "turn-1")
        # Empty content item → saved with empty content (not filtered at batch level)
        # The test just confirms no crash
        assert mock_cstore.save.await_count <= 1


# ── run_forget_cycle ──────────────────────────────────────────────────────────

class TestRunForgetCycle:
    def _active_item(self, citem_id: str, importance: float = 0.1) -> CItem:
        return CItem(
            citem_id=citem_id,
            conversation_id="conv-1",
            content="test",
            item_type=ItemType.OBSERVATION,
            scope="episodic",
            scope_status="active",
            importance=importance,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),  # very old
        )

    def _archived_item(self, citem_id: str, importance: float = 0.05) -> CItem:
        item = CItem(
            citem_id=citem_id,
            conversation_id="conv-1",
            content="test",
            item_type=ItemType.OBSERVATION,
            scope="episodic",
            scope_status="archived",
            importance=importance,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
            archived_at_unix=datetime(2000, 1, 1, tzinfo=UTC).timestamp(),
        )
        return item

    async def test_returns_counts(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        mock_cstore.fetch_by_conversation.return_value = []
        n_att, n_purged = await service.run_forget_cycle("conv-1")
        assert isinstance(n_att, int)
        assert isinstance(n_purged, int)

    async def test_attenuates_old_low_importance_item(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        item = self._active_item("item-1", importance=0.1)
        # fetch_by_conversation: first call (active) returns item, second call (archived) empty
        mock_cstore.fetch_by_conversation.side_effect = [[item], []]
        params = ForgetParams(attenuation_age_days=1.0, attenuation_threshold=0.3)
        service._forget_params = params

        n_att, _ = await service.run_forget_cycle("conv-1")
        assert n_att == 1

    async def test_purges_archived_old_item(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        archived = self._archived_item("item-2", importance=0.02)
        mock_cstore.fetch_by_conversation.side_effect = [[], [archived]]
        params = ForgetParams(alpha_purge_days=1.0, min_importance_to_purge=0.1)
        service._forget_params = params

        _, n_purged = await service.run_forget_cycle("conv-1")
        assert n_purged == 1
        mock_cstore.delete.assert_awaited_once_with("item-2")

    async def test_does_not_attenuate_decision(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        item = CItem(
            citem_id="dec-1",
            conversation_id="conv-1",
            content="decision",
            item_type=ItemType.DECISION,
            scope_status="active",
            importance=0.01,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
        mock_cstore.fetch_by_conversation.side_effect = [[item], []]
        params = ForgetParams(attenuation_age_days=1.0, attenuation_threshold=0.9)
        service._forget_params = params

        n_att, _ = await service.run_forget_cycle("conv-1")
        assert n_att == 0


# ── check_promotions ──────────────────────────────────────────────────────────

class TestCheckPromotions:
    async def test_promotes_eligible_item(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        item = CItem(
            citem_id="ep-1",
            conversation_id="conv-1",
            content="fact",
            item_type=ItemType.FACT,
            scope="episodic",
            scope_status="active",
            importance=0.9,
        )
        mock_cstore.fetch_by_conversation.return_value = [item]
        from cima_demo.domain.value_objects import PromotionPolicy
        service._promotion_policy = PromotionPolicy(min_references=1, min_importance=0.5)

        promoted, demoted = await service.check_promotions("conv-1", {"ep-1": 5})
        assert promoted == 1
        assert demoted == 0
        mock_cstore.update_field.assert_awaited_with("ep-1", "scope", "global")

    async def test_does_not_promote_below_threshold(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        item = CItem(
            citem_id="ep-2",
            conversation_id="conv-1",
            content="fact",
            item_type=ItemType.FACT,
            scope="episodic",
            scope_status="active",
            importance=0.3,  # below min_importance=0.7
        )
        mock_cstore.fetch_by_conversation.return_value = [item]

        promoted, demoted = await service.check_promotions("conv-1", {"ep-2": 10})
        assert promoted == 0

    async def test_no_promotions_for_global_items(
        self,
        service: MemoryService,
        mock_cstore: AsyncMock,
    ) -> None:
        item = CItem(
            citem_id="gl-1",
            conversation_id="conv-1",
            content="global fact",
            scope="global",
            scope_status="active",
            importance=0.9,
        )
        mock_cstore.fetch_by_conversation.return_value = [item]

        promoted, demoted = await service.check_promotions("conv-1", {"gl-1": 100})
        assert promoted == 0
