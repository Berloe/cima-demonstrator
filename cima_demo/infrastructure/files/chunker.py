"""SemanticChunkerAdapter → ChunkingPort (T-08, §3.11 — resolves APP-D-06)."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable

log = logging.getLogger(__name__)

from cima_demo.domain.ports import ChunkingPort
from cima_demo.domain.value_objects import ChunkResult

# Matches [PAGE N] markers injected by FileProcessingAdapter._extract_pdf()
# at the START of a line (after split on double-newline, may be first line of block)
_PAGE_MARKER_RE = re.compile(r"^\[PAGE\s+(\d+)\]\s*\n?", re.IGNORECASE | re.MULTILINE)

# Heading heuristics: markdown headers or short lines (<= 80 chars, no period at end)
_HEADING_RE = re.compile(r"^(#{1,4}\s+.{1,70}|[A-Z][A-Z\s\d,\-]{3,60})\s*$")


class SemanticChunkerAdapter(ChunkingPort):
    """Paragraph-boundary chunker with page-number and section tracking.

    Target 300 tokens; hard range 100-420.
    Respects [PAGE N] markers from PDF extraction.
    Uses heading heuristics to annotate section_hint on each chunk.

    token_counter: sync callable str → int.
    """

    TARGET_TOKENS = 300
    MIN_TOKENS    = 100
    MAX_TOKENS    = 420

    def __init__(self, token_counter: Callable[[str], int]) -> None:
        self._count = token_counter

    async def chunk(self, text: str, filename: str, doc_type: str) -> list[ChunkResult]:
        import asyncio
        return await asyncio.to_thread(self._chunk_sync, text, filename, doc_type)

    def _chunk_sync(self, text: str, filename: str, doc_type: str) -> list[ChunkResult]:
        if not text.strip():
            return []

        # ── 1. Segment text into (page_num, page_text) sections ──────────────
        # Split on [PAGE N] markers to get per-page content.
        # Result: [(page_num_or_None, text_block), ...]
        segments = self._split_by_pages(text)

        # ── 2. Explode segments into annotated paragraphs ────────────────────
        # Each element: (text, page_num, current_section)
        annotated: list[tuple[str, int | None, str | None]] = []
        current_section: str | None = None

        for page_num, block in segments:
            paras = [p.strip() for p in block.split("\n\n") if p.strip()]
            for para in paras:
                # Detect headings to track section context
                if _HEADING_RE.match(para) and self._count(para) < 30:
                    current_section = para.strip()
                annotated.append((para, page_num, current_section))

        if not annotated:
            return []

        # ── 3. Greedy token-budget chunking ──────────────────────────────────
        chunks: list[ChunkResult] = []
        buffer = ""
        buffer_page: int | None = None
        buffer_section: str | None = None
        chunk_index = 0

        def _flush(buf: str, page: int | None, section: str | None) -> None:
            nonlocal chunk_index
            clean = buf.strip()
            if not clean:
                return
            if self._is_noise(clean, doc_type):
                log.debug(
                    "chunker: discarded noise chunk [%s] page=%s len=%d: %r…",
                    filename, page, len(clean), clean[:80],
                )
                return
            chunks.append(ChunkResult(
                text=clean,
                index=chunk_index,
                filename=filename,
                doc_type=doc_type,
                page_num=page,
                section_hint=section,
            ))
            chunk_index += 1

        for para, page, section in annotated:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para

            if self._count(candidate) <= self.MAX_TOKENS:
                buffer = candidate
                if buffer_page is None:
                    buffer_page = page
                if buffer_section is None:
                    buffer_section = section
            else:
                if buffer and self._count(buffer) >= self.MIN_TOKENS:
                    _flush(buffer, buffer_page, buffer_section)
                    buffer = para
                    buffer_page = page
                    buffer_section = section
                else:
                    # Paragraph too long — split at sentence boundary
                    combined = (buffer + " " + para).strip() if buffer else para
                    buffer = ""
                    for sent in self._split_sentences(combined):
                        trial = (buffer + " " + sent).strip() if buffer else sent
                        if self._count(trial) <= self.MAX_TOKENS:
                            buffer = trial
                        else:
                            if buffer:
                                _flush(buffer, buffer_page, buffer_section)
                            if self._count(sent) > self.MAX_TOKENS:
                                # Single sentence exceeds hard max — emit as-is; unavoidable
                                log.debug(
                                    "chunker: sentence exceeds MAX_TOKENS (%d tokens), "
                                    "emitting oversized chunk for %s",
                                    self._count(sent), filename,
                                )
                                _flush(sent, buffer_page, buffer_section)
                                buffer = ""
                            else:
                                buffer = sent
                    buffer_page = page
                    buffer_section = section

        _flush(buffer, buffer_page, buffer_section)
        return chunks

    @staticmethod
    def _split_by_pages(text: str) -> list[tuple[int | None, str]]:
        """Split text into (page_num, content) segments using [PAGE N] markers.

        If no markers present, returns a single segment with page_num=None.
        """
        # Find all [PAGE N] marker positions
        matches = list(_PAGE_MARKER_RE.finditer(text))
        if not matches:
            return [(None, text)]

        segments: list[tuple[int | None, str]] = []

        # Text before first marker (if any)
        if matches[0].start() > 0:
            pre = text[:matches[0].start()].strip()
            if pre:
                segments.append((None, pre))

        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()
            if content:
                segments.append((page_num, content))

        return segments

    @staticmethod
    def _is_noise(text: str, doc_type: str) -> bool:
        """Return True if chunk is structural noise unfit for memory.

        Conservative — three hard filters only. Verbatim text is never altered.
        Skipped entirely for structured formats where all content is meaningful.
        """
        # Structured formats: every token is content — never filter
        if doc_type in ("code", "json", "csv"):
            return False

        words = text.split()
        n = len(words)
        if n == 0:
            return True

        # 1. Extreme repetition — page-number lists, breadcrumb repetitions
        #    ("1 2 3 4 5 … 48 49 50" or "Home Home Home …")
        unique_ratio = len({w.lower() for w in words}) / n
        if unique_ratio < 0.25:
            return True

        # 2. Symbol-dominated — separator bars, icon-label rows, copyright blocks
        #    ("© 2024 | Privacy | Terms | ® | ™ | ···")
        non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if non_alnum / max(len(text), 1) > 0.45:
            return True

        # 3. Mostly numeric tokens — table-of-contents page references
        #    ("Introduction 1  Methods 3  Results 7  Discussion 12 …")
        numeric_tokens = sum(1 for w in words if re.fullmatch(r"\d+\.?\d*", w))
        if numeric_tokens / n > 0.50:
            return True

        return False

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]
