"""ConflictDetector — Jaccard + NLI conflict detection for C-Items (SPEC-5 split)."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cima_demo.domain.entities import CItem, IngestRequest

from cima_demo.domain.entities import ConflictLogEntry
from cima_demo.domain.errors import ConflictDetectionError, NLIUnavailableError
from cima_demo.domain.ports import CItemStorePort, LLMPort, NLIPort, RelDBPort
from cima_demo.domain.value_objects import ItemType

log = logging.getLogger(__name__)

_CONFLICT_STOPWORDS = frozenset({
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
    "do", "does", "did", "will", "would", "could", "should", "my", "your",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not", "with",
    "from", "that", "this", "it", "its", "by", "way", "there", "here", "about",
    # Spanish
    "el", "la", "los", "las", "un", "una", "es", "son", "fue", "era",
    "en", "de", "del", "al", "y", "o", "no", "se", "mi", "tu", "su",
    "lo", "le", "me", "nos", "por", "para", "con", "sin", "que", "más",
    "si", "ya", "pero", "como", "muy", "hay", "ha", "han", "ser", "tener",
    "está", "están", "dicho", "todo", "todos", "cada", "otra", "otro",
})

_JACCARD_CONFLICT_THRESHOLD = 0.20
_MIN_CONFLICT_WORDS = 2

CONFLICTABLE = frozenset({
    ItemType.FACT,
    ItemType.DERIVED,
    ItemType.HYPOTHESIS,
    ItemType.DECISION,
    ItemType.CONSTRAINT,
    ItemType.OBSERVATION,
})


def conflict_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r'\w+', text.lower())
        if w not in _CONFLICT_STOPWORDS and len(w) > 2
    }


def jaccard_similarity(a: str, b: str) -> float:
    wa, wb = conflict_words(a), conflict_words(b)
    if len(wa) < _MIN_CONFLICT_WORDS or len(wb) < _MIN_CONFLICT_WORDS:
        return 0.0
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


class ConflictDetector:
    """Jaccard filter + hybrid NLI conflict detection."""

    def __init__(
        self,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        nli_port: NLIPort | None = None,
        llm_port: LLMPort | None = None,
    ) -> None:
        self._db = rel_db
        self._cstore = citem_store
        self._nli = nli_port
        self._llm = llm_port

    async def resolve_conflict(self, citem_id: str) -> None:
        await self._cstore.update_field(citem_id, "conflict_status", "resolved")
        log.debug("resolve_conflict: cleared conflict_status for %s", citem_id[:8])

    async def detect_conflict(self, new_item: "CItem", request: "IngestRequest") -> None:
        try:
            related = await self._cstore.fetch_by_conversation(
                request.conversation_id, scope_status="active"
            )
            candidates = [
                item for item in related
                if item.citem_id != new_item.citem_id
                and item.item_type in CONFLICTABLE
                and new_item.item_type in CONFLICTABLE
                and item.conflict_status != "flagged"
            ]
            candidates.sort(key=lambda x: x.created_at, reverse=True)
            candidates = candidates[:20]

            for existing in candidates:
                jaccard = jaccard_similarity(new_item.content, existing.content)
                if jaccard < _JACCARD_CONFLICT_THRESHOLD:
                    continue
                is_contradiction = await self._nli_is_contradiction(
                    existing.content, new_item.content
                )
                if not is_contradiction:
                    continue
                entry = ConflictLogEntry(
                    conversation_id=request.conversation_id,
                    item_a_id=existing.citem_id,
                    item_b_id=new_item.citem_id,
                    conflict_type="CONTRADICTION",
                )
                await self._db.save_conflict(entry)
                await self._cstore.update_field(
                    existing.citem_id, "conflict_status", "flagged"
                )
                await self._cstore.update_field(
                    new_item.citem_id, "conflict_status", "flagged"
                )
                log.info(
                    "Conflict detected (jaccard=%.3f, nli=CONTRADICTION): %s ↔ %s",
                    jaccard, existing.citem_id, new_item.citem_id,
                )
                break

        except ConflictDetectionError:
            raise
        except Exception as exc:
            raise ConflictDetectionError(f"Conflict detection failed: {exc}") from exc

    async def _nli_is_contradiction(self, text_a: str, text_b: str) -> bool:
        from cima_demo.infrastructure.nli.tei import CONTRADICTION as _CONTRADICTION

        if self._nli is not None:
            try:
                label = await self._nli.classify(text_a, text_b)
                if label == _CONTRADICTION:
                    return True
                return False
            except NLIUnavailableError as exc:
                log.warning("TEI NLI unavailable, falling back to LLM: %s", exc)

        return await self._llm_nli_classify(text_a, text_b)

    async def _llm_nli_classify(self, text_a: str, text_b: str) -> bool:
        if self._llm is None:
            return False
        from cima_demo.domain.entities import LLMMessage
        prompt = (
            "Do the following two statements contradict each other?\n\n"
            f"Statement A: {text_a}\n\n"
            f"Statement B: {text_b}\n\n"
            "Answer with exactly one word: CONTRADICTION or NEUTRAL"
        )
        try:
            answer = await self._llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=10,
            )
            return "CONTRADICTION" in answer.upper()
        except Exception as exc:
            log.warning("LLM NLI classify failed: %s", exc)
            return False
