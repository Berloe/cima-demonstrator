from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import OpenCaseDocument, OpenCaseEvidence, OpenScenarioCase


_DEFINITION_RE = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9_\- /]{1,80}?)\s+(?:is|are|refers to|means|denotes)\s+(?P<body>.+)$"
)
_REQUIREMENT_RE = re.compile(
    r"\b(must|shall|required to|should not|must not|is required to|needs to)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(must not|should not|cannot|disabled|forbidden|prohibited)\b", re.IGNORECASE)
_MODAL_RE = re.compile(r"\b(must|shall|required to|is required to|needs to|should)\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 \-]{3,80})$")


@dataclass(slots=True)
class ExternalCaseCandidate:
    candidate_id: str
    dataset_id: str
    case_family: str
    status: str
    confidence: float
    prompt: str
    gold_answers: list[str]
    gold_label: str | None
    documents: list[OpenCaseDocument] = field(default_factory=list)
    gold_evidence: list[OpenCaseEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset_id": self.dataset_id,
            "case_family": self.case_family,
            "status": self.status,
            "confidence": self.confidence,
            "prompt": self.prompt,
            "gold_answers": list(self.gold_answers),
            "gold_label": self.gold_label,
            "documents": [
                {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text, "metadata": dict(doc.metadata)}
                for doc in self.documents
            ],
            "gold_evidence": [
                {"doc_id": ev.doc_id, "text": ev.text, "metadata": dict(ev.metadata)}
                for ev in self.gold_evidence
            ],
            "metadata": dict(self.metadata),
        }

    def to_open_case(self) -> OpenScenarioCase:
        return OpenScenarioCase(
            case_id=self.candidate_id,
            dataset_id=self.dataset_id,
            task_family=self.case_family,
            split="external",
            prompt=self.prompt,
            gold_answers=list(self.gold_answers),
            gold_label=self.gold_label,
            documents=list(self.documents),
            gold_evidence=list(self.gold_evidence),
            metadata=dict(self.metadata),
        )


def _read_docs(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", _normalize_text(text)) if part.strip()]
    return parts


def _sentence_spans(paragraph: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", paragraph)
    return [piece.strip() for piece in pieces if piece.strip()]


def _is_heading(paragraph: str) -> bool:
    return bool(_HEADING_RE.match(paragraph.strip()))


def _iter_heading_blocks(text: str) -> list[tuple[str | None, list[str]]]:
    blocks: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_body: list[str] = []
    for paragraph in _paragraphs(text):
        if _is_heading(paragraph):
            if current_heading is not None or current_body:
                blocks.append((current_heading, current_body))
            current_heading = paragraph.strip().lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(paragraph)
    if current_heading is not None or current_body:
        blocks.append((current_heading, current_body))
    return blocks


def _definition_candidates(doc: OpenCaseDocument, *, dataset_id: str) -> list[ExternalCaseCandidate]:
    candidates: list[ExternalCaseCandidate] = []
    for p_idx, paragraph in enumerate(_paragraphs(doc.text)):
        for s_idx, sentence in enumerate(_sentence_spans(paragraph)):
            match = _DEFINITION_RE.match(sentence)
            if not match:
                continue
            subject = match.group("subject").strip(" :-")
            if len(subject.split()) > 12:
                continue
            candidates.append(
                ExternalCaseCandidate(
                    candidate_id=f"{dataset_id}::{doc.doc_id}::definition::{p_idx}-{s_idx}",
                    dataset_id=dataset_id,
                    case_family="external_definition_grounding",
                    status="auto_ready",
                    confidence=0.95,
                    prompt=f"According to '{doc.title}', what is {subject}?",
                    gold_answers=[sentence],
                    gold_label=None,
                    documents=[doc],
                    gold_evidence=[OpenCaseEvidence(doc_id=doc.doc_id, text=sentence, metadata={"paragraph_index": p_idx})],
                    metadata={"subject": subject, "source_case_type": "definition"},
                )
            )
    return candidates


def _requirement_subject(sentence: str, heading: str | None) -> str:
    if heading:
        return heading
    prefix = _NEGATION_RE.sub("", _MODAL_RE.split(sentence, maxsplit=1)[0]).strip(" :-")
    if 3 <= len(prefix) <= 80:
        return prefix
    words = sentence.split()
    return " ".join(words[: min(8, len(words))]).strip(" :-")


def _requirement_candidates(doc: OpenCaseDocument, *, dataset_id: str) -> list[ExternalCaseCandidate]:
    candidates: list[ExternalCaseCandidate] = []
    for block_idx, (heading, body) in enumerate(_iter_heading_blocks(doc.text)):
        for para_offset, paragraph in enumerate(body):
            for s_idx, sentence in enumerate(_sentence_spans(paragraph)):
                if not _REQUIREMENT_RE.search(sentence):
                    continue
                subject = _requirement_subject(sentence, heading)
                candidates.append(
                    ExternalCaseCandidate(
                        candidate_id=f"{dataset_id}::{doc.doc_id}::requirement::{block_idx}-{para_offset}-{s_idx}",
                        dataset_id=dataset_id,
                        case_family="external_requirement_grounding",
                        status="auto_ready",
                        confidence=0.9,
                        prompt=f"According to '{doc.title}', what requirement is stated about {subject}?",
                        gold_answers=[sentence],
                        gold_label=None,
                        documents=[doc],
                        gold_evidence=[OpenCaseEvidence(doc_id=doc.doc_id, text=sentence, metadata={"heading": heading, "block_index": block_idx})],
                        metadata={"subject": subject, "heading": heading, "source_case_type": "requirement"},
                    )
                )
    return candidates


def _conflict_candidates(doc: OpenCaseDocument, *, dataset_id: str) -> list[ExternalCaseCandidate]:
    candidates: list[ExternalCaseCandidate] = []
    sentences: list[tuple[int, str]] = []
    for p_idx, paragraph in enumerate(_paragraphs(doc.text)):
        for sentence in _sentence_spans(paragraph):
            if _MODAL_RE.search(sentence) or _NEGATION_RE.search(sentence):
                sentences.append((p_idx, sentence))
    for idx, (p_idx, left) in enumerate(sentences):
        left_prefix = _MODAL_RE.split(left, maxsplit=1)[0].strip(" :-").lower()
        if not left_prefix:
            continue
        for q_idx, right in sentences[idx + 1 :]:
            right_prefix = _MODAL_RE.split(right, maxsplit=1)[0].strip(" :-").lower()
            if not right_prefix or left_prefix != right_prefix:
                continue
            left_neg = bool(_NEGATION_RE.search(left))
            right_neg = bool(_NEGATION_RE.search(right))
            if left_neg == right_neg:
                continue
            candidates.append(
                ExternalCaseCandidate(
                    candidate_id=f"{dataset_id}::{doc.doc_id}::conflict::{idx}-{q_idx}",
                    dataset_id=dataset_id,
                    case_family="external_conflict_detection",
                    status="needs_review",
                    confidence=0.55,
                    prompt=f"Do the statements about '{left_prefix}' conflict in '{doc.title}'? Answer using the evidence.",
                    gold_answers=[left, right],
                    gold_label="conflict_candidate",
                    documents=[doc],
                    gold_evidence=[
                        OpenCaseEvidence(doc_id=doc.doc_id, text=left, metadata={"paragraph_index": p_idx}),
                        OpenCaseEvidence(doc_id=doc.doc_id, text=right, metadata={"paragraph_index": q_idx}),
                    ],
                    metadata={"subject": left_prefix, "source_case_type": "conflict_candidate"},
                )
            )
    return candidates


def generate_external_cases(
    *,
    docs_path: Path,
    out_root: Path,
    dataset_id: str = "external_generated",
    max_cases_per_doc: int | None = None,
) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    docs = [OpenCaseDocument(doc_id=str(row["doc_id"]), title=str(row["title"]), text=str(row["text"]), metadata=dict(row.get("metadata") or {})) for row in _read_docs(docs_path)]
    candidates: list[ExternalCaseCandidate] = []
    auto_ready: list[OpenScenarioCase] = []
    for doc in docs:
        mined: list[ExternalCaseCandidate] = []
        mined.extend(_definition_candidates(doc, dataset_id=dataset_id))
        mined.extend(_requirement_candidates(doc, dataset_id=dataset_id))
        mined.extend(_conflict_candidates(doc, dataset_id=dataset_id))
        if max_cases_per_doc is not None:
            auto = [case for case in mined if case.status == "auto_ready"][:max_cases_per_doc]
            review = [case for case in mined if case.status != "auto_ready"][: max(0, max_cases_per_doc // 2)]
            mined = auto + review
        candidates.extend(mined)
        auto_ready.extend(case.to_open_case() for case in mined if case.status == "auto_ready")
    candidates_path = out_root / "external_case_candidates.jsonl"
    candidates_path.write_text("\n".join(json.dumps(case.to_dict(), ensure_ascii=False) for case in candidates), encoding="utf-8")
    cases_path = out_root / "external_cases.jsonl"
    cases_path.write_text("\n".join(case.to_json() for case in auto_ready), encoding="utf-8")
    manifest = {
        "schema_version": "cima_demo.external_casegen_manifest.v1",
        "dataset_id": dataset_id,
        "document_count": len(docs),
        "candidate_count": len(candidates),
        "auto_ready_case_count": len(auto_ready),
        "artifacts": {
            "candidates": str(candidates_path),
            "cases": str(cases_path),
        },
        "case_family_counts": {
            family: sum(1 for case in candidates if case.case_family == family)
            for family in sorted({case.case_family for case in candidates})
        },
    }
    manifest_path = out_root / "external_case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"candidates": candidates_path, "cases": cases_path, "manifest": manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate executable external test cases from normalized external documents.")
    parser.add_argument("--documents", type=Path, required=True, help="external_documents.jsonl from external_sources.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-id", default="external_generated")
    parser.add_argument("--max-cases-per-doc", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = generate_external_cases(
        docs_path=args.documents,
        out_root=args.out,
        dataset_id=args.dataset_id,
        max_cases_per_doc=args.max_cases_per_doc,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
