"""Acceptance checks for frozen demonstrator scenarios."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .datasets import HarnessScenario


@dataclass(slots=True)
class ScenarioCheck:
    name: str
    passed: bool
    observed: Any
    expected: Any
    invalidates: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "expected": self.expected,
            "invalidates": self.invalidates,
        }


@dataclass(slots=True)
class ScenarioAcceptance:
    scenario_id: str
    passed: bool
    checks: list[ScenarioCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class ScenarioObserved:
    answer_text: str
    answer_lineage: dict[str, Any] | None
    context_snapshot: dict[str, Any] | None
    budget_trace: dict[str, Any] | None
    corpus_tokens: int
    visible_transcript: list[dict[str, Any]]
    zoom_result: dict[str, Any] | None = None
    zoom_out_result: dict[str, Any] | None = None
    handoff_manifest: dict[str, Any] | None = None
    handoff_validation: dict[str, Any] | None = None
    handoff_restore: dict[str, Any] | None = None
    resumed_answer_text: str | None = None


def evaluate_scenario(scenario: HarnessScenario, observed: ScenarioObserved) -> ScenarioAcceptance:
    checks: list[ScenarioCheck] = []
    answer_lower = observed.answer_text.lower()
    for expected in scenario.expectations.answer_contains:
        checks.append(
            ScenarioCheck(
                name=f"answer_contains:{expected}",
                passed=expected.lower() in answer_lower,
                observed=observed.answer_text,
                expected=expected,
                invalidates="The answer does not contain the expected grounded statement.",
            )
        )

    for forbidden in scenario.expectations.forbidden_answer_contains:
        checks.append(
            ScenarioCheck(
                name=f"answer_forbids:{forbidden}",
                passed=forbidden.lower() not in answer_lower,
                observed=observed.answer_text,
                expected=f"must not contain: {forbidden}",
                invalidates="The answer overclaims or contradicts the expected safe-degradation posture.",
            )
        )

    snapshot = observed.context_snapshot or {}
    budget_trace = observed.budget_trace or {}
    checks.append(
        ScenarioCheck(
            name="context_within_budget",
            passed=int(budget_trace.get("tokens_used", 0)) <= int(budget_trace.get("available_for_content", 0) or 0),
            observed={
                "tokens_used": budget_trace.get("tokens_used"),
                "available_for_content": budget_trace.get("available_for_content"),
            },
            expected="tokens_used <= available_for_content",
            invalidates="The prompt budget is violated.",
        )
    )

    if scenario.scenario_id.startswith("A_"):
        checks.append(
            ScenarioCheck(
                name="corpus_exceeds_window",
                passed=observed.corpus_tokens >= scenario.window_tokens * 10,
                observed={"corpus_tokens": observed.corpus_tokens, "window_tokens": scenario.window_tokens},
                expected="corpus_tokens >= 10x effective window",
                invalidates="The scenario does not prove virtual long-context under bounded window.",
            )
        )
        checks.append(
            ScenarioCheck(
                name="selected_context_smaller_than_corpus",
                passed=int(budget_trace.get("tokens_used", 0)) < observed.corpus_tokens,
                observed={"selected_tokens": budget_trace.get("tokens_used"), "corpus_tokens": observed.corpus_tokens},
                expected="selected context smaller than corpus",
                invalidates="The scenario may still be injecting the whole corpus.",
            )
        )

    answer_lineage = observed.answer_lineage or {}
    cited = list(answer_lineage.get("cited_markers") or [])
    for marker in scenario.expectations.required_markers:
        checks.append(
            ScenarioCheck(
                name=f"required_marker:{marker}",
                passed=marker in cited or marker in observed.answer_text,
                observed={"cited_markers": cited, "answer_text": observed.answer_text},
                expected=marker,
                invalidates="The required marker is missing from the grounded answer.",
            )
        )

    if scenario.scenario_id.startswith("B_"):
        lineage_rows = list(answer_lineage.get("lineage") or [])
        checks.append(
            ScenarioCheck(
                name="answer_lineage_present",
                passed=bool(answer_lineage) and bool(lineage_rows),
                observed=answer_lineage,
                expected="answer lineage with selected items",
                invalidates="Traceability would remain narrative without durable answer lineage.",
            )
        )
        checks.append(
            ScenarioCheck(
                name="lineage_resolves_to_sources",
                passed=bool(answer_lineage.get("resolved_source_count", 0)) and bool(answer_lineage.get("resolved_span_count", 0)),
                observed={
                    "resolved_source_count": answer_lineage.get("resolved_source_count", 0),
                    "resolved_span_count": answer_lineage.get("resolved_span_count", 0),
                },
                expected="> 0 resolved sources and spans",
                invalidates="Markers do not resolve down to sources and spans.",
            )
        )

    if scenario.scenario_id.startswith("C_"):
        validation = observed.handoff_validation or {}
        restore = observed.handoff_restore or {}
        checks.append(
            ScenarioCheck(
                name="handoff_validation",
                passed=bool(validation.get("valid")),
                observed=validation,
                expected="valid handoff",
                invalidates="The handoff manifest is not portable or self-consistent.",
            )
        )
        checks.append(
            ScenarioCheck(
                name="handoff_evidence_coverage",
                passed=float(validation.get("evidence_coverage", 0.0) or 0.0) >= scenario.expectations.min_evidence_coverage,
                observed=validation.get("evidence_coverage"),
                expected=scenario.expectations.min_evidence_coverage,
                invalidates="The restored state is not backed by enough evidence coverage.",
            )
        )
        checks.append(
            ScenarioCheck(
                name="handoff_restore_valid",
                passed=bool(restore.get("valid")),
                observed=restore,
                expected="restore valid",
                invalidates="The task cannot be reconstructed into a new conversation.",
            )
        )
        for expected in scenario.expectations.resume_contains:
            checks.append(
                ScenarioCheck(
                    name=f"resume_contains:{expected}",
                    passed=expected.lower() in (observed.resumed_answer_text or "").lower(),
                    observed=observed.resumed_answer_text,
                    expected=expected,
                    invalidates="The resumed run does not continue from restored state convincingly.",
                )
            )

    if scenario.expectations.requires_zoom:
        checks.append(
            ScenarioCheck(
                name="zoom_has_evidence",
                passed=bool((observed.zoom_result or {}).get("markers_added")) and bool((observed.zoom_result or {}).get("evidence_block")),
                observed=observed.zoom_result,
                expected="zoom markers and evidence block",
                invalidates="Zoom is not acting as a real evidence operator.",
            )
        )
    if scenario.expectations.requires_zoom_out:
        checks.append(
            ScenarioCheck(
                name="zoom_out_has_perspective",
                passed=bool((observed.zoom_out_result or {}).get("markers_added")) and bool((observed.zoom_out_result or {}).get("perspective_block")),
                observed=observed.zoom_out_result,
                expected="zoom_out markers and perspective block",
                invalidates="Zoom-out is not acting as a real perspective operator.",
            )
        )

    checks.append(
        ScenarioCheck(
            name="visible_transcript_clean",
            passed=all(row.get("role") in {"user", "assistant"} for row in observed.visible_transcript),
            observed=observed.visible_transcript,
            expected="only user/assistant roles",
            invalidates="The visible transcript leaks internal runtime state.",
        )
    )

    passed = all(check.passed for check in checks)
    return ScenarioAcceptance(scenario_id=scenario.scenario_id, passed=passed, checks=checks)
