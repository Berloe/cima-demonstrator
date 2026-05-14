"""Frozen datasets for the CIMA Demonstrator harness."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DATASET_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass(slots=True)
class HarnessDocument:
    doc_id: str
    title: str
    text: str


@dataclass(slots=True)
class HarnessCItemSpec:
    citem_id: str
    content: str
    item_type: str
    source_doc_id: str
    dependencies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HarnessSummarySpec:
    summary_id: str
    level: int
    content: str
    origin_citem_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HarnessLLMSpec:
    need_proposal: dict[str, Any]
    memory_proposal: dict[str, Any]
    answer: str
    resume_answer: str | None = None


@dataclass(slots=True)
class HarnessExpectation:
    answer_contains: list[str] = field(default_factory=list)
    forbidden_answer_contains: list[str] = field(default_factory=list)
    resume_contains: list[str] = field(default_factory=list)
    required_markers: list[str] = field(default_factory=list)
    min_evidence_coverage: float = 0.8
    requires_handoff: bool = False
    requires_zoom: bool = False
    requires_zoom_out: bool = False


@dataclass(slots=True)
class HarnessScenario:
    scenario_id: str
    title: str
    description: str
    window_tokens: int
    query: str
    resume_query: str | None
    llm: HarnessLLMSpec
    documents: list[HarnessDocument] = field(default_factory=list)
    citems: list[HarnessCItemSpec] = field(default_factory=list)
    summaries: list[HarnessSummarySpec] = field(default_factory=list)
    expectations: HarnessExpectation = field(default_factory=HarnessExpectation)

    @property
    def corpus_tokens(self) -> int:
        return sum(max(1, len(doc.text.split())) for doc in self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "window_tokens": self.window_tokens,
            "query": self.query,
            "resume_query": self.resume_query,
            "llm": {
                "need_proposal": dict(self.llm.need_proposal),
                "memory_proposal": dict(self.llm.memory_proposal),
                "answer": self.llm.answer,
                "resume_answer": self.llm.resume_answer,
            },
            "documents": [doc.__dict__ for doc in self.documents],
            "citems": [
                {
                    "citem_id": item.citem_id,
                    "content": item.content,
                    "item_type": item.item_type,
                    "source_doc_id": item.source_doc_id,
                    "dependencies": list(item.dependencies),
                }
                for item in self.citems
            ],
            "summaries": [
                {
                    "summary_id": item.summary_id,
                    "level": item.level,
                    "content": item.content,
                    "origin_citem_ids": list(item.origin_citem_ids),
                }
                for item in self.summaries
            ],
            "expectations": {
                "answer_contains": list(self.expectations.answer_contains),
                "forbidden_answer_contains": list(self.expectations.forbidden_answer_contains),
                "resume_contains": list(self.expectations.resume_contains),
                "required_markers": list(self.expectations.required_markers),
                "min_evidence_coverage": self.expectations.min_evidence_coverage,
                "requires_handoff": self.expectations.requires_handoff,
                "requires_zoom": self.expectations.requires_zoom,
                "requires_zoom_out": self.expectations.requires_zoom_out,
            },
        }


def _scenario_from_payload(payload: dict[str, Any]) -> HarnessScenario:
    return HarnessScenario(
        scenario_id=str(payload["scenario_id"]),
        title=str(payload["title"]),
        description=str(payload.get("description", "")),
        window_tokens=int(payload.get("window_tokens", 128)),
        query=str(payload["query"]),
        resume_query=(str(payload["resume_query"]).strip() if payload.get("resume_query") else None),
        llm=HarnessLLMSpec(
            need_proposal=dict(payload.get("llm", {}).get("need_proposal", {})),
            memory_proposal=dict(payload.get("llm", {}).get("memory_proposal", {})),
            answer=str(payload.get("llm", {}).get("answer", "")),
            resume_answer=(
                str(payload.get("llm", {}).get("resume_answer")).strip()
                if payload.get("llm", {}).get("resume_answer")
                else None
            ),
        ),
        documents=[HarnessDocument(**doc) for doc in payload.get("documents", [])],
        citems=[HarnessCItemSpec(**item) for item in payload.get("citems", [])],
        summaries=[HarnessSummarySpec(**item) for item in payload.get("summaries", [])],
        expectations=HarnessExpectation(**payload.get("expectations", {})),
    )


def dataset_dir() -> Path:
    return _DATASET_DIR


def list_scenario_files() -> list[Path]:
    return sorted(_DATASET_DIR.glob("scenario_*.json"))


def load_scenario(path_or_name: str | Path) -> HarnessScenario:
    path = Path(path_or_name)
    if not path.exists():
        path = _DATASET_DIR / str(path_or_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _scenario_from_payload(payload)


def load_all_scenarios() -> list[HarnessScenario]:
    return [load_scenario(path) for path in list_scenario_files()]
