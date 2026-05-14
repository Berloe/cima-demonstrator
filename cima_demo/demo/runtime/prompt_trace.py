"""Prompt tracing and linting for CIMA demonstrator LLM calls.

These helpers deliberately persist the exact logical messages sent through the
LLMPort (system/user/etc.) so publication runs can distinguish prompt defects
from model non-compliance.  They never persist transport headers or secrets.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from cima_demo.domain.entities import LLMMessage

SCHEMA_VERSION = "cima_demo.llm_call.v1"
PROMPT_LINT_SCHEMA_VERSION = "cima_demo.prompt_lint.v1"
PROMPT_TEMPLATE_VERSION = "cima_demo.answer_prompt.v4.grounding.v1"

_MARKER_RE = re.compile(r"\[(S|E|P)[1-9][0-9]*\]")
_MARKER_GROUP_RE = re.compile(r"\[((?:[SEP][1-9][0-9]*)(?:\s*,\s*[SEP][1-9][0-9]*)*)\]")

_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cite_closest_partial_evidence", re.compile(r"cite\s+(?:the\s+)?(?:closest|nearest)\s+partial\s+evidence", re.IGNORECASE)),
    ("force_citation_on_insufficient_evidence", re.compile(r"(?:not\s+enough\s+info|insufficient\s+evidence)[^\n]{0,160}\[(?:S|E|P)\d+\]", re.IGNORECASE)),
    ("marker_range_instruction", re.compile(r"\[(?:S|E|P)1\]\s*(?:through|to|-)\s*\[(?:S|E|P)(?:max|\d+)\]", re.IGNORECASE)),
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def canonical_json_sha256(value: Any) -> str:
    return _sha256_text(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def message_to_dict(message: LLMMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content_parts if message.content_parts else (message.content or ""),
    }
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    return payload


def messages_to_dicts(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]




def _marker_uid_from_row(row: dict[str, Any]) -> str:
    marker = str(row.get("marker") or "").strip()
    namespace = str(row.get("marker_namespace") or row.get("support_source") or "runtime_context").strip()
    return str(row.get("marker_uid") or (f"{namespace}:{marker}" if marker else "")).strip()


def _visible_support_signature(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    source_ids = tuple(str(v) for v in list(row.get("source_ids") or row.get("resolved_source_ids") or []) if str(v))
    span_ids = tuple(str(v) for v in list(row.get("span_ids") or row.get("resolved_span_ids") or []) if str(v))
    return (
        str(row.get("ref_kind") or "").strip(),
        str(row.get("ref_id") or "").strip(),
        source_ids,
        span_ids,
    )


def _row_has_verified_prompt_anchor(row: dict[str, Any]) -> bool:
    return bool(
        row.get("prompt_offsets_exact") is True
        and row.get("visible_slice_verified") is True
        and str(row.get("prompt_sha256") or "")
        and str(row.get("visible_slice_sha256") or "")
    )


def _visible_marker_label_collisions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_marker: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        marker = str(row.get("marker") or "").strip()
        if marker:
            by_marker.setdefault(marker, []).append(row)
    out: dict[str, list[str]] = {}
    for marker, marker_rows in by_marker.items():
        if len({_visible_support_signature(row) for row in marker_rows}) > 1:
            out[marker] = sorted(_marker_uid_from_row(row) for row in marker_rows if _marker_uid_from_row(row))
    return out


def _registry_signature(row: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    return (
        str(row.get("ref_kind") or row.get("kind") or ""),
        str(row.get("ref_id") or ""),
        str(row.get("resolution_status") or ""),
        tuple(str(v) for v in list(row.get("source_ids") or row.get("resolved_source_ids") or []) if str(v)),
        tuple(str(v) for v in list(row.get("spans") or row.get("resolved_span_ids") or []) if str(v)),
    )


def _registry_label_collisions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_marker: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        marker = str(row.get("marker") or "").strip()
        if marker:
            by_marker.setdefault(marker, []).append(row)
    out: dict[str, list[str]] = {}
    for marker, marker_rows in by_marker.items():
        if len({_registry_signature(row) for row in marker_rows}) > 1:
            out[marker] = sorted(_marker_uid_from_row(row) for row in marker_rows if _marker_uid_from_row(row))
    return out

def _prompt_text(messages: list[LLMMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.content_parts if message.content_parts else (message.content or "")
        if isinstance(content, str):
            chunks.append(content)
        else:
            chunks.append(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str))
    return "\n\n".join(chunks)


def extract_marker_literals(text: str) -> list[str]:
    out: list[str] = []
    for match in _MARKER_RE.finditer(text or ""):
        marker = match.group(0)[1:-1]
        if marker not in out:
            out.append(marker)
    for group in _MARKER_GROUP_RE.finditer(text or ""):
        for part in group.group(1).split(","):
            marker = part.strip()
            if marker and marker not in out:
                out.append(marker)
    return out


def build_prompt_lint(
    *,
    call_kind: str,
    messages: list[LLMMessage],
    allowed_markers: list[str] | set[str] | None = None,
    require_answer_grounding: bool = False,
    marker_registry: list[dict[str, Any]] | None = None,
    visible_marker_support: list[dict[str, Any]] | None = None,
    task_family: str | None = None,
    output_format: str | None = None,
) -> dict[str, Any]:
    """Return deterministic prompt lint evidence.

    The linter is intentionally conservative. It does not try to prove prompt
    quality; it proves the invariants that matter for CIMA publication runs:
    closed citation set, no marker-range shortcuts, no dangerous repair policy,
    and explicit grounding/abstention rules when the call produces an answer.
    """
    allowed = [str(m) for m in (allowed_markers or []) if str(m)]
    allowed_set = set(allowed)
    registry_list = [dict(row) for row in (marker_registry or []) if isinstance(row, dict) and str(row.get("marker") or "")]
    registry_label_collisions = _registry_label_collisions(registry_list)
    registry_by_marker: dict[str, list[dict[str, Any]]] = {}
    for row in registry_list:
        registry_by_marker.setdefault(str(row.get("marker") or ""), []).append(row)
    visible_support_rows = [dict(row) for row in (visible_marker_support or []) if isinstance(row, dict) and str(row.get("marker") or "")]
    visible_marker_label_collisions = _visible_marker_label_collisions(visible_support_rows)
    visible_rows = {
        str(row.get("marker") or ""): row
        for row in visible_support_rows
        if str(row.get("marker") or "") not in visible_marker_label_collisions
        and _row_has_verified_prompt_anchor(row)
    }

    def _has_citable_registry_row(marker: str) -> bool:
        rows = registry_by_marker.get(marker) or []
        if marker in registry_label_collisions:
            return False
        return any(
            row.get("citable") is True
            and str(row.get("resolution_status") or "") in {"source_span", "summary_witness"}
            for row in rows
        )

    allowed_without_resolution = [marker for marker in allowed if not _has_citable_registry_row(marker)]
    # Strict C3V invariant: an allowed marker must have a verified prompt-visible
    # anchor.  If no visible support rows were supplied, every allowed marker is
    # missing support; do not treat an empty support set as "not checked".
    allowed_without_visible_support = [
        marker for marker in allowed
        if marker not in visible_rows
    ]
    missing_visible_marker_support = bool(allowed and not visible_support_rows)
    text = _prompt_text(messages)
    marker_literals = extract_marker_literals(text)
    unknown_markers = sorted([m for m in marker_literals if m not in allowed_set])
    declared_allowed_line_present = True
    if require_answer_grounding and allowed:
        allowed_text = ", ".join(f"[{m}]" for m in allowed)
        declared_allowed_line_present = allowed_text in text and "Allowed citation markers" in text

    dangerous = []
    for name, pattern in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            dangerous.append(name)

    uses_marker_range = "marker_range_instruction" in dangerous
    grounding_policy_present = bool(re.search(r"CIMA\s+GROUNDING\s+POLICY|Use only the CIMA-provided context", text, re.IGNORECASE))
    training_policy_present = bool(re.search(r"training-time knowledge|external facts as factual support|must not rely on .*knowledge", text, re.IGNORECASE))
    abstention_policy_present = bool(re.search(r"NOT\s+ENOUGH\s+INFO|insufficient[-\s]+evidence|do not guess", text, re.IGNORECASE))
    closed_marker_policy_present = bool(re.search(r"Allowed citation markers are exactly and only|closed list|No other citation markers exist", text, re.IGNORECASE))
    inferred_output_format = (output_format or "").strip().lower()
    if not inferred_output_format:
        contract_match = re.search(
            r"output\s+contract\s*:\s*(?:\n|.){0,500}?^\s*format\s*[:=]\s*([a-z0-9_-]+)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if contract_match:
            inferred_output_format = contract_match.group(1).strip().lower()
    if not inferred_output_format:
        match = re.search(r"output[_\s-]*format\s*[:=]\s*([a-z0-9_-]+)", text, re.IGNORECASE)
        if match:
            inferred_output_format = match.group(1).strip().lower()
    inferred_task_family = (task_family or "").strip().lower()
    normalized_task_family = re.sub(r"[^a-z0-9]+", "_", inferred_task_family).strip("_")
    numeric_task_family = normalized_task_family in {
        "numeric",
        "number",
        "calculation",
        "arithmetic",
        "counting",
        "extractive_numeric",
        "direct_arithmetic",
        "prompt_contained_quant",
        "source_bound_quant",
        "executionmode_source_bound_quant",
        "executionmode_prompt_contained_quant",
        "executionmode_direct_arithmetic",
    }
    output_format_number_mismatch = inferred_output_format == "number" and not numeric_task_family
    contradictory_abstention_citation_policy = bool(
        re.search(
            r"(?:NOT\s+ENOUGH\s+INFO|insufficient[-\s]+evidence|not\s+enough\s+info)[^\n]{0,120}"
            r"(?:must|always|required|force)[^\n]{0,80}(?:cite|citation|marker)",
            text,
            re.IGNORECASE,
        )
        and not re.search(r"do not force (?:a )?citation|without forcing citations|no citation is required", text, re.IGNORECASE)
    )

    failures: list[str] = []
    if require_answer_grounding:
        if not grounding_policy_present:
            failures.append("missing_grounding_policy")
        if not training_policy_present:
            failures.append("missing_training_knowledge_exclusion")
        if not abstention_policy_present:
            failures.append("missing_abstention_policy")
        if allowed and not declared_allowed_line_present:
            failures.append("allowed_markers_not_rendered_exactly")
        if allowed and not closed_marker_policy_present:
            failures.append("missing_closed_marker_policy")
        if allowed_without_resolution:
            failures.append("allowed_markers_without_source_span_or_summary_witness")
        if missing_visible_marker_support:
            failures.append("missing_visible_marker_support")
        if allowed_without_visible_support:
            failures.append("allowed_markers_without_prompt_visible_support")
        if registry_label_collisions:
            failures.append("duplicate_marker_registry_label")
        if visible_marker_label_collisions:
            failures.append("visible_marker_label_collision")
        if output_format_number_mismatch:
            failures.append("output_format_number_for_non_numeric_task")
        if contradictory_abstention_citation_policy:
            failures.append("contradictory_abstention_and_citation_policy")
    if unknown_markers and allowed_set:
        failures.append("unknown_marker_literals_in_prompt")
    if uses_marker_range:
        failures.append("uses_marker_range")
    if dangerous:
        # marker_range is already listed above, but keep the explicit dangerous
        # names as evidence for audit UX.
        failures.append("dangerous_prompt_instruction")

    return {
        "schema_version": PROMPT_LINT_SCHEMA_VERSION,
        "call_kind": call_kind,
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
        "allowed_markers": allowed,
        "allowed_marker_count": len(allowed),
        "allowed_markers_without_resolution": allowed_without_resolution,
        "allowed_markers_without_visible_support": allowed_without_visible_support,
        "missing_visible_marker_support": missing_visible_marker_support,
        "visible_marker_support_checked": bool(visible_support_rows),
        "visible_marker_support_count": len(visible_support_rows),
        "visible_marker_label_collisions": visible_marker_label_collisions,
        "duplicate_marker_registry_labels": registry_label_collisions,
        "verified_visible_marker_support_count": len(visible_rows),
        "marker_registry_checked": bool(registry_list),
        "marker_literals_in_prompt": marker_literals,
        "unknown_marker_literals_in_prompt": unknown_markers,
        "declared_allowed_line_present": declared_allowed_line_present,
        "grounding_policy_present": grounding_policy_present,
        "training_knowledge_exclusion_present": training_policy_present,
        "abstention_policy_present": abstention_policy_present,
        "closed_marker_policy_present": closed_marker_policy_present,
        "uses_marker_range": uses_marker_range,
        "dangerous_prompt_instruction_names": dangerous,
        "task_family": inferred_task_family,
        "output_format": inferred_output_format,
        "output_format_number_mismatch": output_format_number_mismatch,
        "contradictory_abstention_citation_policy": contradictory_abstention_citation_policy,
        "message_count": len(messages),
        "prompt_chars": len(text),
        "prompt_sha256": _sha256_text(text),
        "prompt_messages_sha256": canonical_json_sha256(messages_to_dicts(messages)),
    }


def build_llm_call_record(
    *,
    run_id: str,
    conversation_id: str,
    turn_id: str,
    call_kind: str,
    messages: list[LLMMessage],
    params: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    allowed_markers: list[str] | set[str] | None = None,
    context_id: str | None = None,
    prompt_lint: dict[str, Any] | None = None,
    response_text: str | None = None,
    response_json: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    msg_dicts = messages_to_dicts(messages)
    return {
        "schema_version": SCHEMA_VERSION,
        "call_id": str(uuid.uuid4()),
        "call_kind": call_kind,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "runtime": runtime or {},
        "params": params or {},
        "context_id": context_id,
        "allowed_markers": [str(m) for m in (allowed_markers or []) if str(m)],
        "messages": msg_dicts,
        "prompt_sha256": canonical_json_sha256(msg_dicts),
        "prompt_lint": prompt_lint or {},
        "response_text": response_text,
        "response_json": response_json,
        "response_sha256": _sha256_text(response_text or json.dumps(response_json or {}, ensure_ascii=False, sort_keys=True)),
        "error": error,
    }
