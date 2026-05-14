"""Reporting utilities for the CIMA Demonstrator harness."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acceptance import ScenarioAcceptance
from .datasets import HarnessScenario


@dataclass(slots=True)
class ScenarioExecutionResult:
    scenario: HarnessScenario
    conversation_id: str
    run_id: str
    answer_text: str
    resumed_answer_text: str | None
    artifacts_dir: Path
    corpus_tokens: int
    context_snapshot: dict[str, Any] | None
    budget_trace: dict[str, Any] | None
    answer_lineage: dict[str, Any] | None
    zoom_result: dict[str, Any] | None
    zoom_out_result: dict[str, Any] | None
    handoff_manifest: dict[str, Any] | None
    handoff_validation: dict[str, Any] | None
    handoff_restore: dict[str, Any] | None
    visible_transcript: list[dict[str, Any]]
    acceptance: ScenarioAcceptance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "title": self.scenario.title,
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
            "answer_text": self.answer_text,
            "resumed_answer_text": self.resumed_answer_text,
            "artifacts_dir": str(self.artifacts_dir),
            "corpus_tokens": self.corpus_tokens,
            "context_snapshot": self.context_snapshot,
            "budget_trace": self.budget_trace,
            "answer_lineage": self.answer_lineage,
            "zoom_result": self.zoom_result,
            "zoom_out_result": self.zoom_out_result,
            "handoff_manifest": self.handoff_manifest,
            "handoff_validation": self.handoff_validation,
            "handoff_restore": self.handoff_restore,
            "visible_transcript": self.visible_transcript,
            "acceptance": self.acceptance.to_dict() if self.acceptance is not None else None,
        }


def build_acceptance_report(results: list[ScenarioExecutionResult]) -> dict[str, Any]:
    passed = [result for result in results if result.acceptance is not None and result.acceptance.passed]
    return {
        "schema_version": "cima_demo.acceptance_report.v1",
        "scenario_count": len(results),
        "passed_count": len(passed),
        "failed_count": len(results) - len(passed),
        "all_passed": len(passed) == len(results),
        "scenarios": [result.to_dict() for result in results],
    }


def write_acceptance_report(root: Path, payload: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "acceptance_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_demo_report(root: Path, results: list[ScenarioExecutionResult], acceptance_report: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CIMA Demonstrator — Demo Harness Report",
        "",
        f"Scenarios: {acceptance_report['scenario_count']}",
        f"Passed: {acceptance_report['passed_count']}",
        f"Failed: {acceptance_report['failed_count']}",
        "",
    ]
    for result in results:
        acc = result.acceptance
        status = "PASS" if acc is not None and acc.passed else "FAIL"
        lines.extend([
            f"## {result.scenario.scenario_id} — {result.scenario.title}",
            "",
            f"Status: **{status}**",
            f"Run ID: `{result.run_id}`",
            f"Artifacts: `{result.artifacts_dir}`",
            f"Corpus tokens: {result.corpus_tokens}",
            f"Answer: {result.answer_text}",
            "",
        ])
        if result.resumed_answer_text:
            lines.append(f"Resumed answer: {result.resumed_answer_text}")
            lines.append("")
        if acc is not None:
            lines.append("Checks:")
            for check in acc.checks:
                mark = "[x]" if check.passed else "[ ]"
                lines.append(f"- {mark} {check.name}")
            lines.append("")
    path = root / "demo_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
