"""Tests for FileProcessingAdapter (cima_demo/infrastructure/files/adapter.py)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from cima_demo.domain.errors import FileProcessingError
from cima_demo.infrastructure.files.processor import FileProcessingAdapter


@pytest.fixture
def adapter() -> FileProcessingAdapter:
    return FileProcessingAdapter()


# ── sys.modules helpers ───────────────────────────────────────────────────────

def _mock_pypdf(reader: MagicMock) -> dict[str, MagicMock]:
    """Return sys.modules patch dict that provides a mock PdfReader."""
    mod = MagicMock()
    mod.PdfReader = MagicMock(return_value=reader)
    return {"pypdf": mod}


def _mock_docx_paragraphs(texts: list[str]) -> tuple[dict[str, MagicMock], MagicMock]:
    """Build (sys_modules_patch, doc_mock) for DOCX paragraph extraction tests.

    The adapter iterates ``doc.element.body`` (internal XML body) and wraps each
    ``<w:p>`` element in ``docx.text.paragraph.Paragraph``. This helper sets up
    both the element list and the Paragraph constructor mock so the test receives
    ``para.text == text`` for each block.
    """
    blocks: list[MagicMock] = []
    para_mocks: list[MagicMock] = []
    for text in texts:
        block = MagicMock()
        block.tag = "{w}p"  # split("}")[-1] == "p" — adapter checks this
        para = MagicMock()
        para.text = text
        blocks.append(block)
        para_mocks.append(para)

    doc = MagicMock()
    doc.element.body = blocks

    docx_mod = MagicMock()
    docx_mod.Document = MagicMock(return_value=doc)

    block_to_para = {id(b): p for b, p in zip(blocks, para_mocks)}
    text_para_mod = MagicMock()
    text_para_mod.Paragraph = MagicMock(
        side_effect=lambda b, d: block_to_para.get(id(b), MagicMock()),
    )

    table_mod = MagicMock()

    modules = {
        "docx": docx_mod,
        "docx.table": table_mod,
        "docx.text.paragraph": text_para_mod,
    }
    return modules, doc


def _mock_ocr(ocr_text: str) -> dict[str, MagicMock]:
    """Return sys.modules patch dict for pytesseract + PIL."""
    pil_image_mod = MagicMock()
    pil_image_mod.open = MagicMock(return_value=MagicMock())
    pil_mod = MagicMock()
    pil_mod.Image = pil_image_mod
    tess_mod = MagicMock()
    tess_mod.image_to_string = MagicMock(return_value=ocr_text)
    return {"pytesseract": tess_mod, "PIL": pil_mod, "PIL.Image": pil_image_mod}


def _absent(*names: str) -> dict[str, None]:
    """Mark modules as absent (import will raise ImportError)."""
    return {name: None for name in names}


# ── Plain text / CSV / Markdown ───────────────────────────────────────────────

class TestPlainText:
    def test_extracts_utf8_text(self, adapter: FileProcessingAdapter) -> None:
        result = adapter.extract_text(b"hello world", "note.txt", "text/plain")
        assert result == "hello world"

    def test_extracts_markdown(self, adapter: FileProcessingAdapter) -> None:
        content = b"# Title\n\nBody text"
        result = adapter.extract_text(content, "doc.md", "text/markdown")
        assert "Title" in result

    def test_replaces_invalid_utf8(self, adapter: FileProcessingAdapter) -> None:
        result = adapter.extract_text(b"\xff\xfe", "bad.txt", "text/plain")
        assert isinstance(result, str)  # no crash — replacement chars

    def test_csv_returns_structured_blocks(self, adapter: FileProcessingAdapter) -> None:
        result = adapter.extract_text(b"a,b,c\n1,2,3", "data.csv", "text/csv")
        # Header line: "[CSV: data.csv] 3 columns: a, b, c"
        assert "[CSV: data.csv]" in result
        assert "3 columns" in result
        # Data block: each field on its own line as "key: value"
        assert "a: 1" in result
        assert "b: 2" in result
        assert "c: 3" in result


# ── PDF ───────────────────────────────────────────────────────────────────────

class TestPdf:
    def _make_reader(self, texts: list[str]) -> MagicMock:
        pages = []
        for text in texts:
            page = MagicMock()
            page.extract_text.return_value = text
            page.images = []
            pages.append(page)
        reader = MagicMock()
        reader.pages = pages
        return reader

    def test_extract_pdf_by_extension(self, adapter: FileProcessingAdapter) -> None:
        reader = self._make_reader(["extracted text"])
        with patch.dict(sys.modules, _mock_pypdf(reader)):
            result = adapter.extract_text(b"%PDF", "doc.pdf", "application/pdf")
        # PDF extractor adds [PAGE N] markers
        assert "extracted text" in result
        assert "[PAGE 1]" in result

    def test_extract_pdf_by_mime(self, adapter: FileProcessingAdapter) -> None:
        reader = self._make_reader(["page content"])
        with patch.dict(sys.modules, _mock_pypdf(reader)):
            result = adapter.extract_text(b"%PDF", "file", "application/pdf")
        assert "page content" in result

    def test_multi_page_pdf_joined_with_newlines(self, adapter: FileProcessingAdapter) -> None:
        reader = self._make_reader(["Page one", "Page two"])
        with patch.dict(sys.modules, _mock_pypdf(reader)):
            result = adapter.extract_text(b"%PDF", "multi.pdf", "application/pdf")
        assert "Page one" in result
        assert "Page two" in result

    def test_pages_with_no_text_excluded(self, adapter: FileProcessingAdapter) -> None:
        reader = self._make_reader(["", "Real content"])
        with patch.dict(sys.modules, _mock_pypdf(reader)):
            result = adapter.extract_text(b"%PDF", "doc.pdf", "application/pdf")
        # Empty page 1 is excluded; page 2 appears with its marker
        assert "Real content" in result
        assert "[PAGE 1]" not in result
        assert "[PAGE 2]" in result

    def test_scanned_pdf_falls_back_to_ocr(self, adapter: FileProcessingAdapter) -> None:
        """All pages return empty text → OCR fallback is called."""
        mock_image = MagicMock()
        mock_image.data = b"\x89PNG\r\n"
        page = MagicMock()
        page.extract_text.return_value = ""
        page.images = [mock_image]
        reader = MagicMock()
        reader.pages = [page]

        mods = {**_mock_pypdf(reader), **_mock_ocr("OCR extracted text")}
        with patch.dict(sys.modules, mods):
            result = adapter.extract_text(b"%PDF", "scan.pdf", "application/pdf")
        assert "OCR extracted text" in result

    def test_scanned_pdf_graceful_fallback_without_pytesseract(
        self, adapter: FileProcessingAdapter,
    ) -> None:
        page = MagicMock()
        page.extract_text.return_value = ""
        page.images = [MagicMock(data=b"img")]
        reader = MagicMock()
        reader.pages = [page]

        mods = {**_mock_pypdf(reader), **_absent("pytesseract", "PIL", "PIL.Image")}
        with patch.dict(sys.modules, mods):
            result = adapter.extract_text(b"%PDF", "scan.pdf", "application/pdf")
        assert result == ""  # graceful empty string

    def test_pypdf_import_error_raises_file_processing_error(
        self, adapter: FileProcessingAdapter,
    ) -> None:
        with patch.dict(sys.modules, _absent("pypdf")), \
             pytest.raises(FileProcessingError, match="pypdf"):
            adapter.extract_text(b"%PDF", "doc.pdf", "application/pdf")


# ── DOCX ──────────────────────────────────────────────────────────────────────

class TestDocx:
    def test_extract_docx_by_extension(self, adapter: FileProcessingAdapter) -> None:
        mods, _ = _mock_docx_paragraphs(["Paragraph text"])
        with patch.dict(sys.modules, mods):
            result = adapter.extract_text(b"PK", "doc.docx", "application/octet-stream")
        assert "Paragraph text" in result

    def test_empty_paragraphs_excluded(self, adapter: FileProcessingAdapter) -> None:
        mods, _ = _mock_docx_paragraphs(["", "Real", "  "])
        with patch.dict(sys.modules, mods):
            result = adapter.extract_text(b"PK", "doc.docx", "application/octet-stream")
        assert result == "Real"

    def test_docx_import_error_raises_file_processing_error(
        self, adapter: FileProcessingAdapter,
    ) -> None:
        with patch.dict(sys.modules, _absent("docx")), \
             pytest.raises(FileProcessingError, match="python-docx"):
            adapter.extract_text(
                b"PK", "doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )


# ── OCR (images) ──────────────────────────────────────────────────────────────

class TestOcr:
    def test_extracts_text_from_png(self, adapter: FileProcessingAdapter) -> None:
        with patch.dict(sys.modules, _mock_ocr("scanned text")):
            result = adapter.extract_text(b"\x89PNG", "image.png", "image/png")
        assert result == "scanned text"

    def test_extracts_text_from_jpeg(self, adapter: FileProcessingAdapter) -> None:
        with patch.dict(sys.modules, _mock_ocr("jpeg text")):
            result = adapter.extract_text(b"\xff\xd8", "photo.jpg", "image/jpeg")
        assert result == "jpeg text"

    def test_ocr_graceful_fallback_without_pytesseract(
        self, adapter: FileProcessingAdapter,
    ) -> None:
        with patch.dict(sys.modules, _absent("pytesseract", "PIL", "PIL.Image")):
            result = adapter.extract_text(b"\x89PNG", "image.png", "image/png")
        assert result == ""

    def test_image_detected_by_extension_not_mime(
        self, adapter: FileProcessingAdapter,
    ) -> None:
        with patch.dict(sys.modules, _mock_ocr("ext text")):
            result = adapter.extract_text(b"\x89PNG", "scan.png", "application/octet-stream")
        assert result == "ext text"


# ── supported_mime_types ──────────────────────────────────────────────────────

class TestSupportedMimeTypes:
    def test_includes_pdf(self, adapter: FileProcessingAdapter) -> None:
        assert "application/pdf" in adapter.supported_mime_types()

    def test_includes_images(self, adapter: FileProcessingAdapter) -> None:
        mimes = adapter.supported_mime_types()
        assert "image/png" in mimes
        assert "image/jpeg" in mimes

    def test_returns_list(self, adapter: FileProcessingAdapter) -> None:
        assert isinstance(adapter.supported_mime_types(), list)
