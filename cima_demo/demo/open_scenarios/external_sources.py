from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from cima_demo.infrastructure.files.processor import FileProcessingAdapter


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def as_text(self) -> str:
        return "\n".join(self._parts)


@dataclass(slots=True)
class ExternalSourceSpec:
    source_id: str
    source_type: str
    title: str | None = None
    path: str | None = None
    url: str | None = None
    text: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExternalDocument:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "metadata": self.metadata,
        }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_manifest(path: Path) -> list[ExternalSourceSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("sources") or []
    specs: list[ExternalSourceSpec] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest entry {idx} must be an object")
        source_id = str(row.get("source_id") or row.get("id") or f"source-{idx}")
        source_type = str(row.get("source_type") or row.get("type") or "").strip().lower()
        if source_type not in {"text", "file", "url"}:
            raise ValueError(f"Unsupported source_type for {source_id}: {source_type}")
        specs.append(
            ExternalSourceSpec(
                source_id=source_id,
                source_type=source_type,
                title=(str(row.get("title")) if row.get("title") is not None else None),
                path=(str(row.get("path")) if row.get("path") is not None else None),
                url=(str(row.get("url")) if row.get("url") is not None else None),
                text=(str(row.get("text")) if row.get("text") is not None else None),
                tags=[str(tag) for tag in row.get("tags") or []],
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return specs


def _normalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                collapsed.append("")
            blank = True
            continue
        collapsed.append(line)
        blank = False
    return "\n".join(collapsed).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_html_as_text(text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return parser.as_text().strip()


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or parsed.netloc or "external-url"
    return name


def _title_from_path(path: Path) -> str:
    return path.stem or path.name


def _load_text_source(spec: ExternalSourceSpec) -> ExternalDocument:
    if not spec.text or not spec.text.strip():
        raise ValueError(f"text source {spec.source_id} does not contain text")
    text = _normalize_whitespace(spec.text)
    title = spec.title or spec.source_id
    metadata = {**spec.metadata, "source_type": "text", "tags": list(spec.tags), "sha256": _sha256_text(text)}
    return ExternalDocument(doc_id=spec.source_id, title=title, text=text, metadata=metadata)


def _load_file_source(spec: ExternalSourceSpec, *, base_dir: Path) -> ExternalDocument:
    if not spec.path:
        raise ValueError(f"file source {spec.source_id} does not contain path")
    path = Path(spec.path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"file source {spec.source_id} not found: {path}")
    content = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "text/plain"
    processor = FileProcessingAdapter()
    text = _normalize_whitespace(processor.extract_text(content, path.name, mime_type))
    title = spec.title or _title_from_path(path)
    metadata = {
        **spec.metadata,
        "source_type": "file",
        "path": str(path),
        "mime_type": mime_type,
        "tags": list(spec.tags),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return ExternalDocument(doc_id=spec.source_id, title=title, text=text, metadata=metadata)


async def _load_url_source(spec: ExternalSourceSpec, *, timeout_seconds: float) -> ExternalDocument:
    if not spec.url:
        raise ValueError(f"url source {spec.source_id} does not contain url")
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        response = await client.get(spec.url)
        response.raise_for_status()
        body = response.text
        content_type = response.headers.get("content-type", "")
    text = body
    if "html" in content_type.lower() or spec.url.lower().endswith((".html", ".htm")):
        text = _render_html_as_text(body)
    text = _normalize_whitespace(text)
    title = spec.title or _title_from_url(spec.url)
    metadata = {
        **spec.metadata,
        "source_type": "url",
        "url": spec.url,
        "content_type": content_type,
        "tags": list(spec.tags),
        "sha256": _sha256_text(text),
    }
    return ExternalDocument(doc_id=spec.source_id, title=title, text=text, metadata=metadata)


async def materialize_external_sources(
    *,
    manifest_path: Path,
    out_root: Path,
    timeout_seconds: float,
) -> dict[str, Path]:
    _ensure_dir(out_root)
    specs = _read_manifest(manifest_path)
    docs: list[ExternalDocument] = []
    source_root = manifest_path.parent
    for spec in specs:
        if spec.source_type == "text":
            docs.append(_load_text_source(spec))
        elif spec.source_type == "file":
            docs.append(_load_file_source(spec, base_dir=source_root))
        else:
            docs.append(await _load_url_source(spec, timeout_seconds=timeout_seconds))
    docs_path = out_root / "external_documents.jsonl"
    docs_path.write_text("\n".join(json.dumps(doc.to_dict(), ensure_ascii=False) for doc in docs), encoding="utf-8")
    manifest = {
        "schema_version": "cima_demo.external_sources_manifest.v1",
        "source_manifest": str(manifest_path),
        "document_count": len(docs),
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "sha256": doc.metadata.get("sha256"),
                "source_type": doc.metadata.get("source_type"),
            }
            for doc in docs
        ],
        "artifacts": {
            "documents": str(docs_path),
        },
    }
    manifest_out = out_root / "external_documents_manifest.json"
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"documents": docs_path, "manifest": manifest_out}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize external source manifests into normalized documents.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest with text/file/url sources")
    parser.add_argument("--out", type=Path, required=True, help="Directory for normalized external documents")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


async def _run(args: argparse.Namespace) -> int:
    outputs = await materialize_external_sources(
        manifest_path=args.manifest,
        out_root=args.out,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    import asyncio
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
