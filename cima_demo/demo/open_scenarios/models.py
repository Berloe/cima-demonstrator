from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class DatasetSource:
    kind: str  # hf_dataset | hf_parquet | git_repo | direct_files
    ref: str
    config: str | None = None
    revision: str | None = None


@dataclass(slots=True)
class OpenDatasetSpec:
    dataset_id: str
    title: str
    task_family: str
    source: DatasetSource
    recommended_role: str
    supports_normalization: bool
    notes: str = ""


@dataclass(slots=True)
class OpenCaseDocument:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenCaseEvidence:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenScenarioCase:
    case_id: str
    dataset_id: str
    task_family: str
    split: str
    prompt: str
    gold_answers: list[str] = field(default_factory=list)
    gold_label: str | None = None
    documents: list[OpenCaseDocument] = field(default_factory=list)
    gold_evidence: list[OpenCaseEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "task_family": self.task_family,
            "split": self.split,
            "prompt": self.prompt,
            "gold_answers": list(self.gold_answers),
            "gold_label": self.gold_label,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "metadata": dict(doc.metadata),
                }
                for doc in self.documents
            ],
            "gold_evidence": [
                {
                    "doc_id": ev.doc_id,
                    "text": ev.text,
                    "metadata": dict(ev.metadata),
                }
                for ev in self.gold_evidence
            ],
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass(slots=True)
class DownloadArtifact:
    dataset_id: str
    path: Path
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizationResult:
    dataset_id: str
    output_path: Path
    case_count: int
    skipped_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
