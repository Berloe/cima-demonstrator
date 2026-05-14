"""FinalAnswerValidator — fail-closed answer quality gate."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Patterns that indicate the model leaked protocol internals into the answer
_TOOL_CALL_LEAK_RE = re.compile(
    r'\[TOOL_CALLS\]|\btool_call_id\b|"function"\s*:\s*\{',
    re.IGNORECASE,
)
_JSON_TOOL_ARGS_RE = re.compile(
    r'^\s*\{.*"(?:action|url|queries|code|expression)"\s*:',
    re.DOTALL | re.IGNORECASE,
)
_PROTOCOL_TAGS_RE = re.compile(
    r'<(?:phase|conclusions|response|think|thinking|reasoning)[\s>]',
    re.IGNORECASE,
)
# Minimum visible answer length (chars) — empty / near-empty answers are invalid
_MIN_ANSWER_CHARS = 4


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _candidate_verified_compute_strings(final_compute_result: Any) -> tuple[str, ...]:
    rendered = str(getattr(final_compute_result, "rendered_value", "") or "").strip()
    value = str(getattr(final_compute_result, "value", "") or "").strip()
    unit = str(getattr(final_compute_result, "unit", "") or "").strip()

    candidates: list[str] = []
    for candidate in (
        rendered,
        f"{value} {unit}".strip() if value else "",
        value,
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _answer_mentions_verified_compute_result(answer: str, final_compute_result: Any) -> bool:
    normalized_answer = _normalize_text(answer)
    for candidate in _candidate_verified_compute_strings(final_compute_result):
        normalized_candidate = _normalize_text(candidate)
        if not normalized_candidate:
            continue
        pattern = re.compile(rf"(?<![0-9a-z]){re.escape(normalized_candidate)}(?![0-9a-z])")
        if pattern.search(normalized_answer):
            return True
    return False


@dataclass
class ValidationResult:
    valid: bool
    error_class: str | None = None   # "empty" | "tool_call_leak" | "protocol_tag" | "no_evidence" | "no_compute" | "compute_mismatch"
    detail: str | None = None


def validate_final_answer(
    answer: str,
    *,
    requires_evidence: bool = False,
    resolved_slot_count: int = 0,
    artifact_count: int = 0,
    slot_contract_required: bool = False,
    compute_done: bool = True,
    final_compute_result: Any | None = None,
) -> ValidationResult:
    """Validate the assembled final answer text.

    Checks (in order):
      1. Not empty / near-empty
      2. No [TOOL_CALLS] or raw tool JSON leaked into answer
      3. No protocol tags (<phase>, <conclusions>, <response> etc.)
      4. If requires_evidence: at least one evidence source was obtained
      5. If slot_contract_required: compute must have completed. The canonical
         signal is FinalComputeResult; compute_done remains a legacy fallback.
      6. When a verified FinalComputeResult is present, the answer must surface it

    Returns ValidationResult with valid=True when all checks pass.
    """
    text = answer.strip()
    matches_verified_compute = (
        final_compute_result is not None
        and _answer_mentions_verified_compute_result(text, final_compute_result)
    )

    # 1. Empty
    if len(text) < _MIN_ANSWER_CHARS and not matches_verified_compute:
        return ValidationResult(
            valid=False,
            error_class="empty",
            detail=f"Answer has {len(text)} chars (min {_MIN_ANSWER_CHARS})",
        )

    # 2. Tool-call leak
    if _TOOL_CALL_LEAK_RE.search(text):
        return ValidationResult(
            valid=False,
            error_class="tool_call_leak",
            detail="Answer contains raw tool-call syntax",
        )
    if _JSON_TOOL_ARGS_RE.match(text):
        return ValidationResult(
            valid=False,
            error_class="tool_call_leak",
            detail="Answer is a JSON tool-args object",
        )

    # 3. Protocol tag leak
    if _PROTOCOL_TAGS_RE.search(text):
        return ValidationResult(
            valid=False,
            error_class="protocol_tag",
            detail="Answer contains internal protocol tags",
        )

    # 4. Evidence requirement
    if requires_evidence and resolved_slot_count == 0 and artifact_count == 0:
        return ValidationResult(
            valid=False,
            error_class="no_evidence",
            detail="Task requires evidence but no artifacts or slots were resolved",
        )

    # 5. Compute requirement: the canonical signal is FinalComputeResult.
    # compute_done remains for backward compatibility with older callers that
    # have not yet been migrated to the stronger final-result contract.
    compute_ready = final_compute_result is not None or compute_done
    if slot_contract_required and not compute_ready:
        return ValidationResult(
            valid=False,
            error_class="no_compute",
            detail=(
                "Task requires a verified numeric result via compute() but no "
                "successful compute call was recorded. Resolve all slots then "
                "call compute() before delivering the answer."
            ),
        )

    # 6. Final compute consistency: when the pipeline has already promoted a
    # canonical FinalComputeResult, the surfaced answer must include it.
    if slot_contract_required and final_compute_result is not None:
        if not matches_verified_compute:
            expected = getattr(final_compute_result, "rendered_value", None) or getattr(final_compute_result, "value", None)
            return ValidationResult(
                valid=False,
                error_class="compute_mismatch",
                detail=f"Answer does not include the verified compute result: {expected}",
            )

    return ValidationResult(valid=True)
