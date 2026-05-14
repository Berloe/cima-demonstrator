"""WebIngester — web content chunking + evidence extraction (SPEC-5 split)."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass as _dc
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from cima_demo.domain.entities import CItem, IngestRequest
from cima_demo.domain.ports import ChunkingPort, CItemStorePort
from cima_demo.domain.value_objects import ChunkKind, ItemType

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ── Image stripping ──────────────────────────────────────────────────────────

_IMAGE_MD_RE = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')
_IMAGE_URL_RE = re.compile(
    r'https?://\S+\.(?:png|jpe?g|gif|svg|webp|ico|bmp)(?:\?\S*)?',
    re.IGNORECASE,
)


def strip_images_from_text(text: str) -> str:
    """Remove image markdown and bare image URLs from extracted text.

    Alt text is preserved when descriptive (len > 4) — figure captions survive.
    """
    text = _IMAGE_MD_RE.sub(
        lambda m: m.group(1).strip() if len(m.group(1).strip()) > 4 else '',
        text,
    )
    text = _IMAGE_URL_RE.sub('', text)
    return text


# ── Chunk classifier ─────────────────────────────────────────────────────────

_CK_TABLE_RE = re.compile(r'^\|.+\|.+\|', re.MULTILINE)
_CK_HEADING_RE = re.compile(r'^#{1,6}\s+\S', re.MULTILINE)
_CK_INFOBOX_RE = re.compile(
    r'^\*{0,2}([A-Z][A-Za-z\s\-]+)\*{0,2}\s*[:：]\s+\S', re.MULTILINE
)
_CK_NAV_RE = re.compile(
    r'(?:'
    r'navigation menu'
    r'|retrieved from'
    r'|jump to content'
    r'|view history'
    r'|edit source'
    r'|read\s*\|?\s*edit'
    r'|toggle the table'
    r'|article\s*\|\s*talk'
    r'|read\s*\|\s*view source'
    r'|create account'
    r'|log in'
    r'|search wikipedia'
    r'|donate to wikipedia'
    r'|personal tools'
    r'|contents\s*\[\s*hide\s*\]'
    r'|this page was last edited'
    r'|text is available under'
    r'|privacy policy'
    r'|cookie\s+(?:settings?|preferences?|policy|notice|consent)'
    r'|accept\s+(?:all\s+)?cookies?'
    r'|terms\s+(?:of\s+)?(?:service|use)\b'
    r'|all\s+rights\s+reserved'
    r'|copyright\s+©'
    r'|follow\s+us\s+on\b'
    r'|subscribe\s+to\s+(?:our\s+)?newsletter'
    r'|sign\s+up\s+for\s+(?:our\s+)?newsletter'
    r'|add\s+to\s+(?:cart|basket|bag|wishlist)'
    r'|share\s+this\s+(?:article|post|page|story)'
    r'|filed\s+under\b'
    r')',
    re.IGNORECASE,
)
_CK_CATEGORY_RE = re.compile(
    r'^(?:categories?|hidden categories?)[:：]',
    re.IGNORECASE | re.MULTILINE,
)
_CK_REF_RE = re.compile(
    r'(?:^\s*\[\d+\]|^\s*\^|\bISBN\b|\bDOI\b|\bdoi\.org\b)',
    re.IGNORECASE | re.MULTILINE,
)
_CK_LANGUAGE_LIST_RE = re.compile(
    r'^[\w\u00C0-\u024F\u0400-\u04FF][\w\u00C0-\u024F\u0400-\u04FF\s\-]{1,35}\s*$',
    re.MULTILINE | re.UNICODE,
)
_CK_SENTENCE_SIGNAL_RE = re.compile(
    r'[.!?,:;()0-9]|\b(?:is|are|was|were|the|a|an|in|of|on|at|to|for|with|and|or|but)\b',
    re.IGNORECASE,
)


def _classify_chunk_kind(text: str) -> str:
    """Classify a web-content chunk into a ChunkKind value."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ChunkKind.RAW_FALLBACK.value

    n = len(lines)

    if _CK_NAV_RE.search(text):
        return ChunkKind.NAV_BOILERPLATE.value
    if _CK_CATEGORY_RE.search(text):
        return ChunkKind.CATEGORY.value

    _candidate_lines = [
        l for l in lines
        if len(l.strip()) <= 38
        and _CK_LANGUAGE_LIST_RE.match(l.strip())
        and not _CK_SENTENCE_SIGNAL_RE.search(l)
    ]
    if len(_candidate_lines) >= 5 and len(_candidate_lines) / max(n, 1) >= 0.45:
        return ChunkKind.NAV_BOILERPLATE.value

    if n >= 6:
        _short = sum(1 for l in lines if len(l.strip()) <= 50)
        _punct = sum(l.count('.') + l.count(',') + l.count(';') for l in lines)
        if _short / n >= 0.80 and _punct / max(n, 1) < 0.5:
            return ChunkKind.NAV_BOILERPLATE.value

    table_lines = sum(1 for l in lines if _CK_TABLE_RE.match(l))
    heading_lines = sum(1 for l in lines if _CK_HEADING_RE.match(l))
    infobox_lines = sum(1 for l in lines if _CK_INFOBOX_RE.match(l))
    ref_lines = sum(1 for l in lines if _CK_REF_RE.search(l))

    if ref_lines / n >= 0.25:
        return ChunkKind.REFERENCE_LIST.value
    if table_lines / n >= 0.40:
        return ChunkKind.TABLE_ROW.value
    if infobox_lines / n >= 0.35:
        return ChunkKind.INFOBOX_FIELD.value
    if heading_lines / n >= 0.60 and n <= 5:
        return ChunkKind.HEADING.value

    return ChunkKind.BODY_PARAGRAPH.value


CHUNK_KIND_IMPORTANCE: dict[str, float] = {
    ChunkKind.INFOBOX_FIELD.value:   0.75,
    ChunkKind.TABLE_ROW.value:       0.70,
    ChunkKind.BODY_PARAGRAPH.value:  0.60,
    ChunkKind.HEADING.value:         0.40,
    ChunkKind.CAPTION.value:         0.40,
    ChunkKind.REFERENCE_LIST.value:  0.10,
    ChunkKind.NAV_BOILERPLATE.value: 0.05,
    ChunkKind.CATEGORY.value:        0.05,
    ChunkKind.RAW_FALLBACK.value:    0.30,
}


def _score_relevance(chunk_text: str, query_tokens: frozenset[str]) -> float:
    """Lexical relevance: fraction of query tokens found in the chunk text."""
    if not query_tokens:
        return 0.0
    chunk_words = frozenset(re.findall(r'\b\w+\b', chunk_text.lower()))
    hits = len(query_tokens & chunk_words)
    return hits / len(query_tokens)


# ── WebIngestionResult ───────────────────────────────────────────────────────

@_dc
class WebIngestionResult:
    """Return value of ingest_web_content."""
    citem_ids: list[str]
    evidence_atoms: list[tuple[str, str]]
    discarded: dict[str, int]

    @property
    def empty(self) -> bool:
        return not self.citem_ids


# ── WebIngester ──────────────────────────────────────────────────────────────

class WebIngester:
    """Chunk and ingest fetched web content as a chain of C-Items."""

    def __init__(
        self,
        chunking_port: ChunkingPort,
        citem_store: CItemStorePort,
        ingest_citem_fn: Any,
    ) -> None:
        self._chunker = chunking_port
        self._cstore = citem_store
        self._ingest_citem_fn = ingest_citem_fn

    async def ingest_web_content(
        self,
        url: str,
        text: str,
        title: str,
        conversation_id: str,
        phase: str,
        objective: str | None = None,
    ) -> WebIngestionResult:
        _EMPTY = WebIngestionResult(citem_ids=[], evidence_atoms=[], discarded={})
        if not text.strip():
            return _EMPTY

        meta_content = (
            f"Web page fetched: {url}\n"
            f"Title: {title}\n"
            f"Content size: {len(text):,} chars"
        )
        meta_request = IngestRequest(
            content=meta_content,
            item_type=ItemType.OBSERVATION,
            phase_ingested=phase,
            actor="web",
            conversation_id=conversation_id,
            motivation=f"web_fetch:{url}",
            confidence=1.0,
            importance_override=0.45,
        )
        meta_citem = await self._ingest_citem_fn(meta_request, skip_conflict_detection=True)
        if meta_citem is None:
            return _EMPTY
        all_ids: list[str] = [meta_citem.citem_id]

        chunks = await self._chunker.chunk(text, url, "web")

        prev_id = meta_citem.citem_id
        n_skipped = 0
        evidence_atoms: list[tuple[str, str]] = []
        discarded: dict[str, int] = {}
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            if await self._cstore.exists_by_hash(chunk_hash, conversation_id):
                n_skipped += 1
                continue
            motivation = f"{url} — chunk {chunk.index + 1}/{len(chunks)}"
            if chunk.section_hint:
                motivation += f" — {chunk.section_hint[:60]}"

            kind = _classify_chunk_kind(chunk.text)
            try:
                ck = ChunkKind(kind)
                item_type = ItemType.FACT if ck.evidence_eligible else ItemType.OBSERVATION
                importance = CHUNK_KIND_IMPORTANCE.get(kind, 0.30)
            except ValueError:
                ck = None
                item_type = ItemType.OBSERVATION
                importance = 0.30

            if ck is not None and not ck.prompt_eligible:
                discarded[kind] = discarded.get(kind, 0) + 1
                continue

            request = IngestRequest(
                content=chunk.text,
                item_type=item_type,
                phase_ingested=phase,
                actor="web",
                conversation_id=conversation_id,
                motivation=motivation,
                confidence=1.0,
                importance_override=importance,
                dependency_ids=[prev_id],
                chunk_kind=kind,
            )
            chunk_citem = await self._ingest_citem_fn(request, skip_conflict_detection=True)
            if chunk_citem is None:
                n_skipped += 1
                continue
            prev_id = chunk_citem.citem_id
            all_ids.append(chunk_citem.citem_id)

            if ck is not None and ck.evidence_eligible:
                evidence_atoms.append((chunk.text, kind))

        if objective:
            _q_tokens = frozenset(re.findall(r'\b\w{3,}\b', objective.lower()))
            _KIND_BOOST = {"infobox_field": 0.30, "table_row": 0.20}
            evidence_atoms.sort(
                key=lambda t: _score_relevance(t[0], _q_tokens) + _KIND_BOOST.get(t[1], 0.0),
                reverse=True,
            )
        else:
            _KIND_PRIORITY = {"infobox_field": 0, "table_row": 1, "body_paragraph": 2}
            evidence_atoms.sort(key=lambda t: _KIND_PRIORITY.get(t[1], 9))

        log.debug(
            "ingest_web_content: %s — %d chunks indexed (%d evidence, %d noise), %d duplicates",
            url, len(chunks) - n_skipped, len(evidence_atoms), sum(discarded.values()), n_skipped,
        )
        return WebIngestionResult(
            citem_ids=all_ids,
            evidence_atoms=evidence_atoms,
            discarded=discarded,
        )
