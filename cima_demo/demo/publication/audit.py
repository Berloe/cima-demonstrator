from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunEvidenceRow:
    case_id: str
    dataset_id: str
    has_context: bool
    marker_count: int
    resolved_source_count: int
    resolved_span_count: int
    unresolved_ref_count: int
    resolution_mode: str
    context_tokens: int
    available_context_tokens: int
    budget_passed: bool
    generation_checked: bool
    generation_passed: bool
    generation_failure_kind: str
    citation_checked: bool
    citation_passed: bool
    # Citation Block Coverage: fraction of answer blocks that carry at least one citation
    answer_block_count: int
    uncited_block_count: int
    citation_block_passed: bool   # True iff uncited_block_count == 0 and answer_block_count > 0
    # Evidence Utilization Rate: fraction of available ContextView markers actually cited
    available_marker_count_cit: int
    cited_marker_count: int
    evidence_utilization_rate: float  # cited / available; 0.0 when available == 0
    cleanup_checked: bool
    cleanup_passed: bool
    zoom_checked: bool
    zoom_passed: bool
    zoom_out_checked: bool
    zoom_out_passed: bool  # legacy/composite: artifact plus lineage evidence
    zoom_out_artifact_passed: bool
    zoom_out_lineage_passed: bool
    summary_evidenced: bool
    cleanup_final_verified: bool
    # C3-SAN-v1 split: raw model output vs published answer
    c3_raw_model_checked: bool
    c3_raw_model_passed: bool
    c3_published_checked: bool
    c3_published_passed: bool
    c3_sanitization_applied: bool  # True when C3 removed markers before publication
    c3a_abstention_checked: bool
    c3a_abstention_passed: bool
    answer_type: str
    published_integrity_passed: bool
    publication_status: str
    publishable: bool
    blocked_by_cima: bool
    blocked_reason: str
    invalid_published_as_valid: bool
    publication_gate_declared: bool
    prompt_trace_available: bool
    prompt_lint_passed: bool
    allowed_markers_lineage_passed: bool
    allowed_marker_count_lineage: int
    uncitable_allowed_marker_count: int
    dropped_uncitable_marker_count: int
    auxiliary_item_count: int
    input_context_item_count: int
    retained_context_item_count: int
    gate_cost_denominator_known: bool
    allowed_marker_retention_denominator_known: bool
    dropped_allowed_marker_count: int
    allowed_marker_retention_rate: float | None
    context_item_retention_rate: float
    abstention_due_to_no_allowed_markers: bool
    visible_prompt_support_passed: bool  # deprecated internal alias for visible_marker_anchor_passed
    visible_prompt_support_coverage: float  # deprecated alias for visible_marker_anchor_coverage
    visible_marker_anchor_coverage: float
    verified_visible_support_rate: float
    visible_same_label_distinct_slice_count: int
    cited_without_visible_support_count: int
    allowed_without_visible_support_count: int
    auxiliary_not_in_prompt_passed: bool
    summary_direct_witness_passed: bool
    lineage_stage_passed: bool
    runtime_context_marker_lineage_passed: bool
    runtime_zoom_lineage_passed: bool
    llm_call_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "has_context": self.has_context,
            "marker_count": self.marker_count,
            "resolved_source_count": self.resolved_source_count,
            "resolved_span_count": self.resolved_span_count,
            "unresolved_ref_count": self.unresolved_ref_count,
            "resolution_mode": self.resolution_mode,
            "context_tokens": self.context_tokens,
            "available_context_tokens": self.available_context_tokens,
            "budget_passed": self.budget_passed,
            "generation_checked": self.generation_checked,
            "generation_passed": self.generation_passed,
            "generation_failure_kind": self.generation_failure_kind,
            "citation_checked": self.citation_checked,
            "citation_passed": self.citation_passed,
            "answer_block_count": self.answer_block_count,
            "uncited_block_count": self.uncited_block_count,
            "citation_block_passed": self.citation_block_passed,
            "available_marker_count_cit": self.available_marker_count_cit,
            "cited_marker_count": self.cited_marker_count,
            "evidence_utilization_rate": round(self.evidence_utilization_rate, 4),
            "cleanup_checked": self.cleanup_checked,
            "cleanup_passed": self.cleanup_passed,
            "zoom_checked": self.zoom_checked,
            "zoom_passed": self.zoom_passed,
            "zoom_out_checked": self.zoom_out_checked,
            "zoom_out_passed": self.zoom_out_passed,
            "zoom_out_artifact_passed": self.zoom_out_artifact_passed,
            "zoom_out_lineage_passed": self.zoom_out_lineage_passed,
            "summary_evidenced": self.summary_evidenced,
            "cleanup_final_verified": self.cleanup_final_verified,
            "c3_raw_model_checked": self.c3_raw_model_checked,
            "c3_raw_model_passed": self.c3_raw_model_passed,
            "c3_published_checked": self.c3_published_checked,
            "c3_published_passed": self.c3_published_passed,
            "c3_sanitization_applied": self.c3_sanitization_applied,
            "c3a_abstention_checked": self.c3a_abstention_checked,
            "c3a_abstention_passed": self.c3a_abstention_passed,
            "answer_type": self.answer_type,
            "published_integrity_passed": self.published_integrity_passed,
            "publication_status": self.publication_status,
            "publishable": self.publishable,
            "blocked_by_cima": self.blocked_by_cima,
            "blocked_reason": self.blocked_reason,
            "invalid_published_as_valid": self.invalid_published_as_valid,
            "publication_gate_declared": self.publication_gate_declared,
            "prompt_trace_available": self.prompt_trace_available,
            "prompt_lint_passed": self.prompt_lint_passed,
            "allowed_markers_lineage_passed": self.allowed_markers_lineage_passed,
            "allowed_marker_count_lineage": self.allowed_marker_count_lineage,
            "uncitable_allowed_marker_count": self.uncitable_allowed_marker_count,
            "dropped_uncitable_marker_count": self.dropped_uncitable_marker_count,
            "auxiliary_item_count": self.auxiliary_item_count,
            "input_context_item_count": self.input_context_item_count,
            "retained_context_item_count": self.retained_context_item_count,
            "gate_cost_denominator_known": self.gate_cost_denominator_known,
            "allowed_marker_retention_denominator_known": self.allowed_marker_retention_denominator_known,
            "dropped_allowed_marker_count": self.dropped_allowed_marker_count,
            "allowed_marker_retention_rate": (round(self.allowed_marker_retention_rate, 4) if self.allowed_marker_retention_rate is not None else None),
            "context_item_retention_rate": round(self.context_item_retention_rate, 4),
            "abstention_due_to_no_allowed_markers": self.abstention_due_to_no_allowed_markers,
            "visible_marker_anchor_passed": self.visible_prompt_support_passed,
            "visible_prompt_support_passed_deprecated_alias": self.visible_prompt_support_passed,
            "visible_marker_anchor_coverage": round(self.visible_marker_anchor_coverage, 4),
            "visible_prompt_support_coverage_deprecated_alias": round(self.visible_prompt_support_coverage, 4),
            "verified_visible_support_rate": round(self.verified_visible_support_rate, 4),
            "visible_same_label_distinct_slice_count": self.visible_same_label_distinct_slice_count,
            "cited_without_visible_support_count": self.cited_without_visible_support_count,
            "allowed_without_visible_support_count": self.allowed_without_visible_support_count,
            "auxiliary_not_in_prompt_passed": self.auxiliary_not_in_prompt_passed,
            "summary_direct_witness_passed": self.summary_direct_witness_passed,
            "lineage_stage_passed": self.lineage_stage_passed,
            "runtime_context_marker_lineage_passed": self.runtime_context_marker_lineage_passed,
            "runtime_zoom_lineage_passed": self.runtime_zoom_lineage_passed,
            "llm_call_count": self.llm_call_count,
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _iter_run_dirs(root: Path) -> list[Path]:
    if (root / "run_manifest.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("run_manifest.json"))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _context_marker_count(context: dict[str, Any] | None) -> int:
    if not context:
        return 0
    markers = context.get("markers") or []
    if isinstance(markers, list):
        return len([m for m in markers if str(m)])
    return 0


def _marker_resolution_by_marker(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key in ("marker_resolution", "zoom_out_marker_resolution"):
        for row in payload.get(key) or []:
            if isinstance(row, dict) and str(row.get("marker") or ""):
                out[str(row["marker"])] = dict(row)
    return out


def _resolution_row_has_source_span_support(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    if list(row.get("unresolved_ref_ids") or []):
        return False
    return _as_int(row.get("resolved_source_count")) > 0 and _as_int(row.get("resolved_span_count")) > 0


def _resolution_row_has_summary_lineage(row: dict[str, Any] | None) -> bool:
    if not _resolution_row_has_source_span_support(row):
        return False
    if str((row or {}).get("ref_kind") or "") not in {"summary", "local_summary", "global_summary"}:
        return False
    citem_ids = [str(v) for v in list((row or {}).get("citem_ids") or []) if str(v)]
    if not citem_ids:
        return False
    if list((row or {}).get("unresolved_citem_ids") or []):
        return False
    witnesses = [dict(v) for v in list((row or {}).get("citem_witnesses") or []) if isinstance(v, dict)]
    if not witnesses:
        return False
    by_citem = {str(w.get("citem_id") or ""): w for w in witnesses if str(w.get("citem_id") or "")}
    return all(list((by_citem.get(cid) or {}).get("source_ids") or []) and list((by_citem.get(cid) or {}).get("span_ids") or []) for cid in citem_ids)


def _resolution_row_is_citable(row: dict[str, Any] | None) -> bool:
    if not _resolution_row_has_source_span_support(row):
        return False
    kind = str((row or {}).get("ref_kind") or (row or {}).get("summary_ref_kind") or "")
    if kind in {"summary", "local_summary", "global_summary", "summary_chunk"}:
        return _resolution_row_has_summary_lineage(row)
    return True


def _payload_marker_lineage_passed(payload: dict[str, Any] | None, field: str) -> bool:
    if not isinstance(payload, dict):
        return True
    markers = [str(marker) for marker in payload.get(field) or [] if str(marker)]
    if not markers:
        return True
    by_marker = _marker_resolution_by_marker(payload)
    return all(_resolution_row_is_citable(by_marker.get(marker)) for marker in markers)


def _visible_marker_support_rows(context: dict[str, Any] | None, citation: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in (context, citation):
        if not isinstance(payload, dict):
            continue
        for row in payload.get("visible_marker_support") or []:
            if isinstance(row, dict) and str(row.get("marker") or ""):
                rows.append(dict(row))
    return rows


def _visible_row_verified(row: dict[str, Any]) -> bool:
    return bool(
        row.get("prompt_offsets_exact") is True
        and row.get("visible_slice_verified") is True
        and str(row.get("prompt_sha256") or "")
        and str(row.get("visible_slice_sha256") or "")
    )




def _visible_support_signature(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    source_ids = tuple(str(v) for v in list(row.get("source_ids") or row.get("resolved_source_ids") or []) if str(v))
    span_ids = tuple(str(v) for v in list(row.get("span_ids") or row.get("resolved_span_ids") or []) if str(v))
    return (
        str(row.get("ref_kind") or "").strip(),
        str(row.get("ref_id") or "").strip(),
        source_ids,
        span_ids,
    )


def _same_label_distinct_slice_count(rows: list[dict[str, Any]]) -> int:
    by_marker: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        marker = str(row.get("marker") or "").strip()
        if marker:
            by_marker.setdefault(marker, []).append(row)
    count = 0
    for marker_rows in by_marker.values():
        if len(marker_rows) < 2:
            continue
        signatures = {_visible_support_signature(row) for row in marker_rows}
        slice_hashes = {str(row.get("visible_slice_sha256") or "") for row in marker_rows if str(row.get("visible_slice_sha256") or "")}
        if len(signatures) == 1 and len(slice_hashes) > 1:
            count += 1
    return count


def _visible_marker_support_markers(context: dict[str, Any] | None, citation: dict[str, Any] | None) -> set[str]:
    return {str(row.get("marker") or "") for row in _visible_marker_support_rows(context, citation) if _visible_row_verified(row)}


def _visible_marker_support_uids(context: dict[str, Any] | None, citation: dict[str, Any] | None) -> set[str]:
    return {str(row.get("marker_uid") or "") for row in _visible_marker_support_rows(context, citation) if str(row.get("marker_uid") or "") and _visible_row_verified(row)}


def _prompt_text_from_llm_calls(llm_calls: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for call in llm_calls or []:
        for message in call.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            else:
                try:
                    chunks.append(json.dumps(content, ensure_ascii=False, sort_keys=True))
                except Exception:
                    chunks.append(str(content))
    return "\n\n".join(chunks)


def _auxiliary_not_in_prompt(context: dict[str, Any] | None, llm_calls: list[dict[str, Any]]) -> bool:
    if not isinstance(context, dict):
        return True
    auxiliary_items = [item for item in context.get("auxiliary_items") or [] if isinstance(item, dict)]
    if not auxiliary_items:
        return True
    prompt_text = _prompt_text_from_llm_calls(llm_calls)
    if "AUXILIARY CONTEXT (NOT CITABLE)" in prompt_text:
        return False
    for item in auxiliary_items:
        content = str(item.get("content") or "").strip()
        if content and len(content) >= 24 and content[:160] in prompt_text:
            return False
    return True


def _summary_direct_witness_passed(context: dict[str, Any] | None, zoom_out: dict[str, Any] | None) -> bool:
    summary_rows: list[dict[str, Any]] = []
    for payload in (context, zoom_out):
        if not isinstance(payload, dict):
            continue
        for row in (payload.get("marker_resolution") or []) + (payload.get("zoom_out_marker_resolution") or []):
            if isinstance(row, dict) and str(row.get("ref_kind") or row.get("summary_ref_kind") or "") in {"summary", "local_summary", "global_summary", "summary_chunk"}:
                summary_rows.append(row)
    if not summary_rows:
        return True
    return all(_resolution_row_has_summary_lineage(row) for row in summary_rows)


def _allowed_marker_lineage_stats(
    *,
    citation: dict[str, Any] | None,
    context: dict[str, Any] | None,
    zoom: dict[str, Any] | None,
    zoom_out: dict[str, Any] | None,
) -> dict[str, Any]:
    allowed = [str(marker) for marker in ((citation or {}).get("allowed_markers") or []) if str(marker)]
    rows: dict[str, dict[str, Any]] = {}
    for payload in (context, zoom, zoom_out):
        for marker, row in _marker_resolution_by_marker(payload).items():
            # Prefer a citable row over an unresolved duplicate for the same
            # label across context/zoom artifacts. The contract property is
            # that every allowed label has a source/span or summary witness in
            # the runtime artifact set.
            if marker not in rows or (not _resolution_row_is_citable(rows.get(marker)) and _resolution_row_is_citable(row)):
                rows[marker] = row
    bad = [marker for marker in allowed if not _resolution_row_is_citable(rows.get(marker))]
    return {
        "allowed_marker_count": len(allowed),
        "uncitable_allowed_markers": bad,
        "uncitable_allowed_marker_count": len(bad),
        "passed": bool(allowed and not bad) if allowed else True,
    }


def _zoom_out_lineage_ok(zoom_out: dict[str, Any] | None) -> bool:
    if not isinstance(zoom_out, dict):
        return False
    markers = [str(v) for v in zoom_out.get("markers_added") or [] if str(v)]
    if not markers or not str(zoom_out.get("perspective_block") or "").strip():
        return False
    if zoom_out.get("summary_lineage_valid") is True:
        return True
    by_marker = _marker_resolution_by_marker(zoom_out)
    return all(_resolution_row_has_summary_lineage(by_marker.get(marker)) for marker in markers)


def _context_summary_lineage_ok(context: dict[str, Any] | None) -> bool:
    if not isinstance(context, dict):
        return False
    by_marker = _marker_resolution_by_marker(context)
    for item in context.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("ref_kind") or "") not in {"summary", "local_summary", "global_summary"}:
            continue
        marker = str(item.get("marker") or "")
        if marker and _resolution_row_has_summary_lineage(by_marker.get(marker)):
            return True
    return False


def _summary_evidenced(context: dict[str, Any] | None, zoom_out: dict[str, Any] | None) -> bool:
    return _context_summary_lineage_ok(context) or _zoom_out_lineage_ok(zoom_out)


def _cleanup_final_verified(cleanup: dict[str, Any] | None) -> bool:
    if not isinstance(cleanup, dict):
        return False
    if cleanup.get("final_verified") is True:
        return True
    audit = cleanup.get("final_audit") if isinstance(cleanup.get("final_audit"), dict) else {}
    consistency = audit.get("consistency") if isinstance(audit.get("consistency"), dict) else cleanup.get("consistency")
    if not isinstance(consistency, dict):
        return False
    return bool(
        consistency.get("cleanup_ok") is True
        and consistency.get("conversation_deleted") is True
        and consistency.get("qdrant_zeroed") is True
    )


def analyze_runs(root: Path) -> dict[str, Any]:
    rows: list[RunEvidenceRow] = []
    dataset_counts: dict[str, int] = {}
    for run_dir in _iter_run_dirs(root):
        manifest = _load_json(run_dir / "run_manifest.json") or {}
        context = _load_json(run_dir / "runtime_context.json") or _load_json(run_dir / "context.json")
        generation = _load_json(run_dir / "generation_contract.json")
        citation = _load_json(run_dir / "citation_contract.json")
        cleanup = _load_json(run_dir / "cleanup.json")
        zoom = _load_json(run_dir / "runtime_zoom.json") or _load_json(run_dir / "zoom.json")
        zoom_out = _load_json(run_dir / "runtime_zoom_out.json") or _load_json(run_dir / "zoom_out.json")
        prompt_lint = _load_json(run_dir / "prompt_lint.json")
        llm_calls = _load_jsonl(run_dir / "llm_calls.jsonl")
        prompt_trace_available = bool(llm_calls)
        prompt_lint_passed = bool((prompt_lint or {}).get("passed"))
        allowed_lineage = _allowed_marker_lineage_stats(
            citation=citation,
            context=context,
            zoom=zoom,
            zoom_out=zoom_out,
        )
        runtime_context_marker_lineage_passed = _payload_marker_lineage_passed(context, "markers")
        runtime_zoom_lineage_passed = _payload_marker_lineage_passed(zoom, "markers_added")

        dataset_id = str(manifest.get("dataset_id") or "unknown")
        case_id = str(manifest.get("case_id") or run_dir.name)
        dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + 1

        token_usage = context.get("token_usage") if isinstance(context, dict) else {}
        context_tokens = _as_int((token_usage or {}).get("context"))
        available = _as_int((token_usage or {}).get("available_for_content"))
        budget_passed = bool(context is not None and (available <= 0 or context_tokens <= available))
        context_drop_metrics = (context or {}).get("context_drop_metrics") if isinstance((context or {}).get("context_drop_metrics"), dict) else {}
        dropped_uncitable_marker_count = _as_int((context_drop_metrics or {}).get("dropped_uncitable_marker_count"))
        auxiliary_item_count = _as_int((context_drop_metrics or {}).get("auxiliary_item_count"))
        input_context_item_count = _as_int((context_drop_metrics or {}).get("input_item_count"))
        gate_cost_denominator_known = input_context_item_count > 0
        retained_context_item_count = _context_marker_count(context)

        zoom_passed = bool(
            zoom
            and str(zoom.get("resolution_mode") or "empty") != "empty"
            and _as_int(zoom.get("resolved_source_count")) > 0
            and _as_int(zoom.get("resolved_span_count")) > 0
            and not list(zoom.get("unresolved_ref_ids") or [])
        )
        zoom_out_artifact_passed = bool(
            zoom_out
            and str(zoom_out.get("resolution_mode") or "empty") != "empty"
            and str(zoom_out.get("perspective_block") or "").strip()
        )
        zoom_out_lineage_passed = _zoom_out_lineage_ok(zoom_out)
        zoom_out_passed = bool(zoom_out_artifact_passed and zoom_out_lineage_passed)
        cleanup_final_verified = _cleanup_final_verified(cleanup)

        # C3-SAN-v1 split fields from citation_contract
        _c3_raw = (citation or {}).get("c3_raw_model") or {}
        _c3_pub = (citation or {}).get("c3_published") or {}
        c3_raw_model_checked = bool(_c3_raw)
        # In C3-SAN-v1, c3_raw_model.passed may mean "passed after deterministic
        # sanitation".  Raw-model obedience is tracked separately as
        # raw_model_passed; fall back to NOOP_PASS for legacy artifacts.
        c3_raw_model_passed = bool(_c3_raw.get("raw_model_passed")) if _c3_raw else False
        if _c3_raw and "raw_model_passed" not in _c3_raw:
            c3_raw_model_passed = str(_c3_raw.get("status") or "") == "NOOP_PASS"
        c3_published_checked = bool(_c3_pub)
        c3_published_passed = bool(_c3_pub.get("passed")) if _c3_pub else False
        c3_sanitization_applied = bool(
            (_c3_raw or {}).get("deterministic_sanitization_applied")
            or (citation or {}).get("deterministic_sanitization_applied")
        )
        _c3a = (citation or {}).get("c3a_traceable_abstention") or {}
        c3a_abstention_checked = bool(_c3a.get("checked") or _c3a.get("applicable"))
        c3a_abstention_passed = bool(_c3a.get("passed")) if c3a_abstention_checked else False
        answer_type = str((citation or {}).get("answer_type") or _c3a.get("answer_type") or "unknown")
        published_integrity_passed = bool(
            (citation or {}).get("published_integrity_passed")
            or (c3_published_checked and c3_published_passed)
            or c3a_abstention_passed
        )
        publication_gate = (citation or {}).get("publication_gate") if isinstance((citation or {}).get("publication_gate"), dict) else {}
        publication_gate_declared = bool(publication_gate)
        publication_status = str(publication_gate.get("publication_status") or (citation or {}).get("publication_status") or ("publishable" if published_integrity_passed else "blocked"))
        publishable = bool(publication_gate.get("publishable")) if "publishable" in publication_gate else bool((citation or {}).get("publishable") if "publishable" in (citation or {}) else published_integrity_passed)
        blocked_by_cima = bool(publication_gate.get("blocked_by_cima")) if "blocked_by_cima" in publication_gate else bool((citation or {}).get("blocked_by_cima") if "blocked_by_cima" in (citation or {}) else (citation is not None and not publishable))
        blocked_reason = str(publication_gate.get("blocked_reason") or (citation or {}).get("blocked_reason") or "")
        invalid_published_as_valid = bool(
            publication_gate.get("invalid_published_as_valid")
            or (citation or {}).get("invalid_published_as_valid")
            or (publication_status == "publishable" and not published_integrity_passed)
        )

        # CBC / EUR fields from citation_contract
        answer_block_count = _as_int((citation or {}).get("answer_block_count"))
        uncited_block_count = _as_int((citation or {}).get("uncited_answer_block_count"))
        citation_block_passed = bool(
            citation is not None and answer_block_count > 0 and uncited_block_count == 0
        )
        available_marker_count_cit = _as_int((citation or {}).get("available_marker_count"))
        candidate_allowed_marker_count = _as_int(
            (citation or {}).get("candidate_allowed_marker_count")
            or (context_drop_metrics or {}).get("candidate_allowed_marker_count")
            or (context_drop_metrics or {}).get("pre_gate_allowed_marker_count")
        )
        allowed_marker_retention_denominator_known = candidate_allowed_marker_count > 0
        if allowed_marker_retention_denominator_known:
            dropped_allowed_marker_count = max(0, candidate_allowed_marker_count - available_marker_count_cit)
            allowed_marker_retention_rate = min(1.0, available_marker_count_cit / candidate_allowed_marker_count)
        else:
            # Do not divide allowed markers by context items.  They are distinct
            # denominators; legacy artifacts lack the pre-gate marker candidate
            # count and must report marker retention as unknown.
            dropped_allowed_marker_count = 0
            allowed_marker_retention_rate = None
        if gate_cost_denominator_known:
            context_item_retention_rate = min(1.0, retained_context_item_count / input_context_item_count)
        else:
            context_item_retention_rate = 1.0
        cited_marker_count = len(list((citation or {}).get("cited_markers") or []))
        evidence_utilization_rate = (
            cited_marker_count / available_marker_count_cit
            if available_marker_count_cit > 0 else 0.0
        )
        visible_support_rows_all = _visible_marker_support_rows(context, citation)
        verified_support_rows = [row for row in visible_support_rows_all if _visible_row_verified(row)]
        visible_same_label_distinct_slice_count = _same_label_distinct_slice_count(verified_support_rows)
        visible_markers = {str(row.get("marker") or "") for row in verified_support_rows if str(row.get("marker") or "")}
        cited_markers = [str(v) for v in list((citation or {}).get("cited_markers") or []) if str(v)]
        allowed_markers = [str(v) for v in list((citation or {}).get("allowed_markers") or []) if str(v)]
        cited_without_visible_support = [m for m in cited_markers if m not in visible_markers]
        # Strict C3V: if allowed markers exist and no verified support exists,
        # every allowed marker lacks a verified prompt-visible anchor.
        allowed_without_visible_support = [m for m in allowed_markers if m not in visible_markers]
        visible_marker_anchor_coverage = (
            (len(cited_markers) - len(cited_without_visible_support)) / len(cited_markers)
            if cited_markers else 1.0
        )
        verified_visible_support_rate = (
            len(verified_support_rows) / len(visible_support_rows_all)
            if visible_support_rows_all else 1.0
        )
        # Backwards-compatible metric name. This is marker-anchor coverage, not
        # semantic claim entailment. The claim matrix states that distinction.
        visible_prompt_support_coverage = visible_marker_anchor_coverage
        visible_prompt_support_passed = bool(not cited_without_visible_support and not allowed_without_visible_support)
        abstention_due_to_no_allowed_markers = bool(answer_type == "insufficient_evidence" and available_marker_count_cit == 0)
        auxiliary_not_in_prompt_passed = _auxiliary_not_in_prompt(context, llm_calls)
        summary_direct_witness_passed = _summary_direct_witness_passed(context, zoom_out)
        lineage_stage_passed = bool(
            runtime_context_marker_lineage_passed
            and runtime_zoom_lineage_passed
            and bool(allowed_lineage.get("passed"))
            and visible_prompt_support_passed
            and summary_direct_witness_passed
            and auxiliary_not_in_prompt_passed
        )

        rows.append(RunEvidenceRow(
            case_id=case_id,
            dataset_id=dataset_id,
            has_context=context is not None,
            marker_count=_context_marker_count(context),
            resolved_source_count=_as_int((context or {}).get("resolved_source_count")),
            resolved_span_count=_as_int((context or {}).get("resolved_span_count")),
            unresolved_ref_count=len(list((context or {}).get("unresolved_ref_ids") or [])),
            resolution_mode=str((context or {}).get("resolution_mode") or "empty"),
            context_tokens=context_tokens,
            available_context_tokens=available,
            budget_passed=budget_passed,
            generation_checked=generation is not None,
            generation_passed=bool((generation or {}).get("passed")) if generation is not None else bool((run_dir / "chat.json").exists()),
            generation_failure_kind=str((generation or {}).get("failure_kind") or ""),
            citation_checked=citation is not None,
            citation_passed=published_integrity_passed if citation is not None else False,
            answer_block_count=answer_block_count,
            uncited_block_count=uncited_block_count,
            citation_block_passed=citation_block_passed,
            available_marker_count_cit=available_marker_count_cit,
            cited_marker_count=cited_marker_count,
            evidence_utilization_rate=evidence_utilization_rate,
            cleanup_checked=cleanup is not None,
            cleanup_passed=cleanup_final_verified,
            zoom_checked=zoom is not None,
            zoom_passed=zoom_passed,
            zoom_out_checked=zoom_out is not None,
            zoom_out_passed=zoom_out_passed,
            zoom_out_artifact_passed=zoom_out_artifact_passed,
            zoom_out_lineage_passed=zoom_out_lineage_passed,
            summary_evidenced=_summary_evidenced(context, zoom_out),
            cleanup_final_verified=cleanup_final_verified,
            c3_raw_model_checked=c3_raw_model_checked,
            c3_raw_model_passed=c3_raw_model_passed,
            c3_published_checked=c3_published_checked,
            c3_published_passed=c3_published_passed,
            c3_sanitization_applied=c3_sanitization_applied,
            c3a_abstention_checked=c3a_abstention_checked,
            c3a_abstention_passed=c3a_abstention_passed,
            answer_type=answer_type,
            published_integrity_passed=published_integrity_passed,
            publication_status=publication_status,
            publishable=publishable,
            blocked_by_cima=blocked_by_cima,
            blocked_reason=blocked_reason,
            invalid_published_as_valid=invalid_published_as_valid,
            publication_gate_declared=publication_gate_declared,
            prompt_trace_available=prompt_trace_available,
            prompt_lint_passed=prompt_lint_passed,
            allowed_markers_lineage_passed=bool(allowed_lineage.get("passed")),
            allowed_marker_count_lineage=_as_int(allowed_lineage.get("allowed_marker_count")),
            uncitable_allowed_marker_count=_as_int(allowed_lineage.get("uncitable_allowed_marker_count")),
            dropped_uncitable_marker_count=dropped_uncitable_marker_count,
            auxiliary_item_count=auxiliary_item_count,
            input_context_item_count=input_context_item_count,
            retained_context_item_count=retained_context_item_count,
            gate_cost_denominator_known=gate_cost_denominator_known,
            allowed_marker_retention_denominator_known=allowed_marker_retention_denominator_known,
            dropped_allowed_marker_count=dropped_allowed_marker_count,
            allowed_marker_retention_rate=allowed_marker_retention_rate,
            context_item_retention_rate=context_item_retention_rate,
            abstention_due_to_no_allowed_markers=abstention_due_to_no_allowed_markers,
            visible_prompt_support_passed=visible_prompt_support_passed,
            visible_prompt_support_coverage=visible_prompt_support_coverage,
            visible_marker_anchor_coverage=visible_marker_anchor_coverage,
            verified_visible_support_rate=verified_visible_support_rate,
            visible_same_label_distinct_slice_count=visible_same_label_distinct_slice_count,
            cited_without_visible_support_count=len(cited_without_visible_support),
            allowed_without_visible_support_count=len(allowed_without_visible_support),
            auxiliary_not_in_prompt_passed=auxiliary_not_in_prompt_passed,
            summary_direct_witness_passed=summary_direct_witness_passed,
            lineage_stage_passed=lineage_stage_passed,
            runtime_context_marker_lineage_passed=runtime_context_marker_lineage_passed,
            runtime_zoom_lineage_passed=runtime_zoom_lineage_passed,
            llm_call_count=len(llm_calls),
        ))

    total = len(rows)

    def rate(predicate) -> float:
        if total == 0:
            return 0.0
        return sum(1 for row in rows if predicate(row)) / total

    def status(value: float, *, demonstrated: float = 0.95, partial: float = 0.50) -> str:
        if total == 0:
            return "not_demonstrated"
        if value >= demonstrated:
            return "demonstrated"
        if value >= partial:
            return "partial"
        return "not_demonstrated"

    claims = [
        {
            "claim_id": "CIMA-C1",
            "claim": "Bounded ContextView under explicit budget",
            "status": status(rate(lambda r: r.has_context and r.budget_passed)),
            "evidence_rate": rate(lambda r: r.has_context and r.budget_passed),
            "evidence": "context.json token_usage.context <= token_usage.available_for_content",
            "publication_note": "Claim only for the configured runtime profile and tokenizer/budget strategy.",
        },
        {
            "claim_id": "CIMA-C2",
            "claim": "Source/span lineage for selected ContextView markers",
            "status": status(rate(lambda r: r.has_context and r.marker_count > 0 and r.resolved_source_count > 0 and r.resolved_span_count > 0 and r.unresolved_ref_count == 0)),
            "evidence_rate": rate(lambda r: r.has_context and r.marker_count > 0 and r.resolved_source_count > 0 and r.resolved_span_count > 0 and r.unresolved_ref_count == 0),
            "evidence": "context.json marker_resolution + resolved_source_ids + resolved_span_ids",
            "publication_note": "This is operational lineage, not a legal compliance guarantee.",
        },
        {
            "claim_id": "CIMA-C0",
            "claim": "LLM generation completed without timeout/error",
            "status": status(rate(lambda r: (not r.generation_checked) or r.generation_passed), demonstrated=0.98),
            "evidence_rate": rate(lambda r: (not r.generation_checked) or r.generation_passed),
            "evidence": "generation_contract.json and chat/chat_error artifacts",
            "publication_note": "Separates model/runtime failure from CIMA memory-navigation evidence.",
        },
        {
            "claim_id": "CIMA-C3",
            "claim": "Published answer integrity: factual answers are cited, abstentions are traced",
            "status": status(rate(lambda r: r.citation_checked and r.published_integrity_passed), demonstrated=0.98),
            "evidence_rate": rate(lambda r: r.citation_checked and r.published_integrity_passed),
            "evidence": "citation_contract.json published_integrity_passed = C3 factual citation OR C3A traceable abstention",
            "publication_note": (
                "CIMA does not optimize against gold answers. If a factual answer is published, "
                "C3 requires valid traceable citations. If no factual answer is supported, C3A allows "
                "an honest insufficient-evidence abstention with an operational trace; no citation is forced."
            ),
        },
        {
            "claim_id": "CIMA-C3V",
            "claim": "Published citation markers refer only to verified prompt-visible marker anchors",
            "status": status(rate(lambda r: r.citation_checked and r.visible_prompt_support_passed), demonstrated=0.98),
            "evidence_rate": rate(lambda r: r.citation_checked and r.visible_prompt_support_passed),
            "evidence": "citation_contract.json visible_marker_support with verified prompt offsets + cited_markers_without_visible_support",
            "publication_note": (
                "This is a structural marker-anchor claim: the cited marker was part of the exact prompt-visible support. "
                "It is not a claim of semantic entailment between each answer claim and the cited evidence."
            ),
        },
        {
            "claim_id": "CIMA-C3-ENTAILMENT",
            "claim": "Each factual claim is semantically entailed by its cited visible evidence",
            "status": "not_claimed",
            "evidence_rate": 0.0,
            "evidence": "No claim-by-claim entailment validator is included in this demonstrator slice.",
            "publication_note": "CIMA structures evidence and traceability; semantic entailment validation is a separate validation layer and is not claimed here.",
        },
        {
            "claim_id": "CIMA-C3-PUB-GATE",
            "claim": "Generated outputs are either publishable under C3/C3A or explicitly blocked",
            "status": status(rate(lambda r: r.citation_checked and (r.publishable or r.blocked_by_cima) and not r.invalid_published_as_valid), demonstrated=0.98),
            "evidence_rate": rate(lambda r: r.citation_checked and (r.publishable or r.blocked_by_cima) and not r.invalid_published_as_valid),
            "evidence": "citation_contract.json publication_gate / publication_status / blocked_reason",
            "publication_note": (
                "CIMA separates generation from publication. A model output that fails C3/C3A is not counted as a valid publication; "
                "it is marked blocked with a reason such as uncited_block_after_sanitizer or empty_generation."
            ),
        },
        {
            "claim_id": "CIMA-C3-SAN",
            "claim": "C3-SAN-v1 deterministic sanitization: published answer passes citation invariants",
            "status": status(rate(lambda r: r.c3_published_checked and r.c3_published_passed), demonstrated=0.98),
            "evidence_rate": rate(lambda r: r.c3_published_checked and r.c3_published_passed),
            "evidence": "citation_contract.json c3_published.passed (invariants B–F)",
            "publication_note": (
                "C3-SAN-v1 guarantees no invalid markers remain in the published answer (invariant B) "
                "and no markers were added or text mutated (C/E). "
                "c3_raw_model records the raw LLM output before sanitization; "
                "c3_published records the final output after the deterministic strip. "
                "Factual correctness and claim entailment are not claimed."
            ),
        },
        {
            "claim_id": "CIMA-C3A",
            "claim": "Traceable abstention for insufficient-evidence answers",
            "status": status(rate(lambda r: (not r.c3a_abstention_checked) or r.c3a_abstention_passed), demonstrated=0.98),
            "evidence_rate": rate(lambda r: (not r.c3a_abstention_checked) or r.c3a_abstention_passed),
            "evidence": "citation_contract.json c3a_traceable_abstention.insufficiency_trace",
            "publication_note": (
                "Applies only to pure insufficient-evidence answers such as NOT ENOUGH INFO. "
                "The trace records the inspected context/markers; it is not presented as factual support."
            ),
        },
        {
            "claim_id": "CIMA-C4",
            "claim": "On-demand drill-down from ContextView to source/span evidence",
            "status": status(rate(lambda r: r.zoom_checked and r.zoom_passed)),
            "evidence_rate": rate(lambda r: r.zoom_checked and r.zoom_passed),
            "evidence": "zoom.json evidence_block + resolved_source_ids + resolved_span_ids",
            "publication_note": (
                "Zoom targets are ContextView markers; the operator navigates from any cited marker "
                "to its stored source span and raw evidence block. The zoom.json artifact records "
                "resolved_source_ids and resolved_span_ids. No separate target_marker input parameter "
                "is required: the operator selects from the active ContextView state."
            ),
        },
        {
            "claim_id": "CIMA-C5A",
            "claim": "Zoom-out artifact is produced as an L1 abstraction candidate",
            "status": status(rate(lambda r: r.zoom_out_checked and r.zoom_out_artifact_passed), demonstrated=0.95, partial=0.50),
            "evidence_rate": rate(lambda r: r.zoom_out_checked and r.zoom_out_artifact_passed),
            "evidence": "runtime_zoom_out.json/zoom_out.json perspective_block + markers_added",
            "publication_note": "This demonstrates production of a zoom-out/L1 projection artifact. It does not by itself demonstrate direct summary witness lineage.",
        },
        {
            "claim_id": "CIMA-C5B",
            "claim": "Zoom-out L1 abstraction has direct summary witness lineage to source/span evidence",
            "status": status(rate(lambda r: r.zoom_out_checked and r.zoom_out_artifact_passed and r.zoom_out_lineage_passed and r.summary_direct_witness_passed), demonstrated=0.95, partial=0.50),
            "evidence_rate": rate(lambda r: r.zoom_out_checked and r.zoom_out_artifact_passed and r.zoom_out_lineage_passed and r.summary_direct_witness_passed),
            "evidence": "zoom_out marker_resolution citem_witnesses/source_ids/span_ids for each effective summary input",
            "publication_note": "This is the strict lineage claim. Legacy summary_lineage_valid alone is not sufficient for C5B; deeper pyramids remain future work.",
        },
        {
            "claim_id": "CIMA-C6",
            "claim": "Local cleanup/purge is idempotent at run level",
            "status": status(rate(lambda r: r.cleanup_checked and r.cleanup_passed)),
            "evidence_rate": rate(lambda r: r.cleanup_checked and r.cleanup_passed),
            "evidence": "cleanup.json final_audit.consistency cleanup_ok/conversation_deleted/qdrant_zeroed",
            "publication_note": "This verifies local hard-delete convergence for the run; it does not claim full archive/thinning/global-memory lifecycle.",
        },
        {
            "claim_id": "CIMA-M1",
            "claim": "Answer block citation coverage over traceable markers",
            "status": status(rate(lambda r: r.citation_block_passed)),
            "evidence_rate": rate(lambda r: r.citation_block_passed),
            "evidence": "citation_contract.json answer_block_count vs uncited_answer_block_count",
            "publication_note": (
                "Citation Block Coverage measures whether every paragraph-level unit of the answer "
                "has at least one citation mapping it to a stored evidence marker. "
                "CBC = 1.0 means no paragraph-level floating claims. This remains a marker-level contract, not claim-level semantic entailment."
            ),
        },
        {
            "claim_id": "CIMA-N1",
            "claim": "Full EU AI Act compliance",
            "status": "not_claimed",
            "evidence_rate": None,
            "evidence": "Out of scope: CIMA provides runtime evidence artifacts, not organizational/legal compliance by itself.",
            "publication_note": "Position as compliance-enabling evidence substrate only.",
        },
        {
            "claim_id": "CIMA-N2",
            "claim": "Complete TaskMemory / global reusable memory / CHM handoff",
            "status": "not_claimed",
            "evidence_rate": None,
            "evidence": "Not part of this demonstrator evidence slice.",
            "publication_note": "Keep as future/extension work unless separately evidenced.",
        },
    ]

    # EUR per-dataset breakdown (mean, min, max)
    eur_by_dataset: dict[str, list[float]] = {}
    for row in rows:
        if row.citation_checked and row.available_marker_count_cit > 0:
            eur_by_dataset.setdefault(row.dataset_id, []).append(row.evidence_utilization_rate)
    eur_stats = {
        ds: {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
        for ds, vals in sorted(eur_by_dataset.items())
    }
    eur_all = [r.evidence_utilization_rate for r in rows if r.citation_checked and r.available_marker_count_cit > 0]
    eur_global = {
        "n": len(eur_all),
        "mean": round(sum(eur_all) / len(eur_all), 4) if eur_all else 0.0,
        "min": round(min(eur_all), 4) if eur_all else 0.0,
        "max": round(max(eur_all), 4) if eur_all else 0.0,
    }

    return {
        "schema_version": "cima_demo.publication_evidence.v2",
        "metric_notes": {
            "visible_marker_anchor_coverage": "Primary C3V metric: structural coverage of cited markers with verified prompt-visible anchors; not semantic entailment.",
            "visible_prompt_support_coverage_deprecated_alias": "Deprecated compatibility alias for visible_marker_anchor_coverage; do not interpret as claim-evidence entailment.",
            "allowed_marker_retention_rate_mean": "Gate-cost metric: available allowed markers after citability/visibility gates divided by explicit pre-gate candidate allowed markers; null when that denominator is absent.",
            "context_item_retention_rate_mean": "Gate-cost metric: retained ContextView marker count after context gating divided by input context item count.",
        },
        "runs_root": str(root),
        "run_count": total,
        "dataset_counts": dataset_counts,
        "rates": {
            "bounded_context": rate(lambda r: r.has_context and r.budget_passed),
            "source_span_lineage": rate(lambda r: r.has_context and r.marker_count > 0 and r.resolved_source_count > 0 and r.resolved_span_count > 0 and r.unresolved_ref_count == 0),
            "generation_success": rate(lambda r: (not r.generation_checked) or r.generation_passed),
            "published_integrity": rate(lambda r: r.citation_checked and r.published_integrity_passed),
            "citation_contract": rate(lambda r: r.citation_checked and r.citation_passed),  # backward-compatible alias
            "publication_gate_declared": rate(lambda r: r.citation_checked and r.publication_gate_declared),
            "publishable_outputs": rate(lambda r: r.citation_checked and r.publishable),
            "blocked_outputs": rate(lambda r: r.citation_checked and r.blocked_by_cima),
            "invalid_published_as_valid": rate(lambda r: r.citation_checked and r.invalid_published_as_valid),
            "citation_block_coverage": rate(lambda r: r.citation_block_passed),
            "zoom_to_evidence": rate(lambda r: r.zoom_checked and r.zoom_passed),
            "zoom_out_summary": rate(lambda r: r.zoom_out_checked and r.zoom_out_passed and r.summary_evidenced),  # legacy composite
            "zoom_out_artifact": rate(lambda r: r.zoom_out_checked and r.zoom_out_artifact_passed),
            "zoom_out_direct_witness_lineage": rate(lambda r: r.zoom_out_checked and r.zoom_out_artifact_passed and r.zoom_out_lineage_passed and r.summary_direct_witness_passed),
            "cleanup": rate(lambda r: r.cleanup_checked and r.cleanup_passed),
            "prompt_trace_available": rate(lambda r: r.prompt_trace_available),
            "prompt_lint_passed": rate(lambda r: r.prompt_trace_available and r.prompt_lint_passed),
            "allowed_markers_resolve_to_source_span_or_summary_witness": rate(lambda r: r.allowed_markers_lineage_passed),
            "runtime_context_marker_lineage": rate(lambda r: r.runtime_context_marker_lineage_passed),
            "runtime_zoom_lineage": rate(lambda r: r.runtime_zoom_lineage_passed),
            "visible_marker_anchor_passed": rate(lambda r: r.visible_prompt_support_passed),
            "visible_marker_anchor_coverage": rate(lambda r: r.visible_prompt_support_passed),
            "visible_prompt_support_coverage_deprecated_alias": rate(lambda r: r.visible_prompt_support_passed),
            "verified_visible_support_rate": sum(row.verified_visible_support_rate for row in rows) / total if total else 0.0,
            "allowed_marker_retention_rate_mean": (
                round(sum(row.allowed_marker_retention_rate for row in rows if row.allowed_marker_retention_rate is not None) / sum(1 for row in rows if row.allowed_marker_retention_rate is not None), 4)
                if any(row.allowed_marker_retention_rate is not None for row in rows) else None
            ),
            "context_item_retention_rate_mean": (
                round(sum(row.context_item_retention_rate for row in rows if row.gate_cost_denominator_known) / sum(1 for row in rows if row.gate_cost_denominator_known), 4)
                if any(row.gate_cost_denominator_known for row in rows) else None
            ),
            "runs_with_abstention_due_to_no_allowed_markers": rate(lambda r: r.abstention_due_to_no_allowed_markers),
            "auxiliary_not_in_prompt": rate(lambda r: r.auxiliary_not_in_prompt_passed),
            "summary_direct_witness": rate(lambda r: r.summary_direct_witness_passed),
            "lineage_stage_passed": rate(lambda r: r.lineage_stage_passed),
            "runs_with_uncitable_context_drops": rate(lambda r: r.dropped_uncitable_marker_count > 0),
            "runs_with_auxiliary_non_citable_context": rate(lambda r: r.auxiliary_item_count > 0),
            "c3_raw_model_passed": rate(lambda r: r.c3_raw_model_checked and r.c3_raw_model_passed),
            "c3_published_passed": rate(lambda r: r.c3_published_checked and r.c3_published_passed),
            "c3_sanitization_applied": rate(lambda r: r.c3_sanitization_applied),
            "c3a_abstention_passed": rate(lambda r: (not r.c3a_abstention_checked) or r.c3a_abstention_passed),
        },
        "evidence_utilization": {
            "global": eur_global,
            "by_dataset": eur_stats,
        },
        "gate_cost_accounting": {
            "runs_with_gate_cost_denominator": sum(1 for row in rows if row.gate_cost_denominator_known),
            "total_input_context_items": sum(row.input_context_item_count for row in rows if row.gate_cost_denominator_known),
            "total_retained_context_items": sum(row.retained_context_item_count for row in rows if row.gate_cost_denominator_known),
            "runs_with_allowed_marker_retention_denominator": sum(1 for row in rows if row.allowed_marker_retention_denominator_known),
            "total_dropped_allowed_markers": sum(row.dropped_allowed_marker_count for row in rows if row.allowed_marker_retention_denominator_known),
            "mean_allowed_marker_retention_rate": (
                round(sum(row.allowed_marker_retention_rate for row in rows if row.allowed_marker_retention_rate is not None) / sum(1 for row in rows if row.allowed_marker_retention_rate is not None), 4)
                if any(row.allowed_marker_retention_rate is not None for row in rows) else None
            ),
            "mean_context_item_retention_rate": (
                round(sum(row.context_item_retention_rate for row in rows if row.gate_cost_denominator_known) / sum(1 for row in rows if row.gate_cost_denominator_known), 4)
                if any(row.gate_cost_denominator_known for row in rows) else None
            ),
            "runs_with_abstention_due_to_no_allowed_markers": sum(1 for row in rows if row.abstention_due_to_no_allowed_markers),
        },
        "drop_accounting": {
            "total_dropped_uncitable_markers": sum(row.dropped_uncitable_marker_count for row in rows),
            "total_auxiliary_items": sum(row.auxiliary_item_count for row in rows),
            "total_cited_without_visible_support": sum(row.cited_without_visible_support_count for row in rows),
            "total_allowed_without_visible_support": sum(row.allowed_without_visible_support_count for row in rows),
            "total_same_label_distinct_visible_slice_warnings": sum(row.visible_same_label_distinct_slice_count for row in rows),
        },
        "lineage_stage_accounting": {
            "runtime_context_marker_lineage_passed": rate(lambda r: r.runtime_context_marker_lineage_passed),
            "runtime_zoom_lineage_passed": rate(lambda r: r.runtime_zoom_lineage_passed),
            "allowed_marker_lineage_passed": rate(lambda r: r.allowed_markers_lineage_passed),
            "visible_marker_anchor_passed": rate(lambda r: r.visible_prompt_support_passed),
            "summary_direct_witness_passed": rate(lambda r: r.summary_direct_witness_passed),
            "zoom_out_artifact_passed": rate(lambda r: r.zoom_out_artifact_passed),
            "zoom_out_direct_witness_lineage_passed": rate(lambda r: r.zoom_out_artifact_passed and r.zoom_out_lineage_passed and r.summary_direct_witness_passed),
            "auxiliary_not_in_prompt_passed": rate(lambda r: r.auxiliary_not_in_prompt_passed),
            "end_to_end_lineage_stage_passed": rate(lambda r: r.lineage_stage_passed),
        },
        "claims": claims,
        "rows": [row.to_dict() for row in rows],
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "publication_evidence_report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "publication_run_metrics.json").write_text(
        json.dumps(report.get("rows", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "publication_claim_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["claim_id", "claim", "status", "evidence_rate", "evidence", "publication_note"])
        writer.writeheader()
        for claim in report.get("claims", []):
            writer.writerow(claim)
    rows = report.get("rows", [])
    if rows:
        with (out_dir / "publication_run_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    _EDUCATION_4_DATASETS = {"explainmeetsum", "qmsum"}
    dataset_counts = report.get("dataset_counts") or {}
    education_4_runs = sum(v for k, v in dataset_counts.items() if k in _EDUCATION_4_DATASETS)
    total_runs = report.get("run_count", 0)

    md_lines = [
        "# CIMA Demonstrator Publication Evidence Report",
        "",
        f"Runs analyzed: **{total_runs}**",
        "",
        "## Dataset coverage",
        "",
    ]
    for dataset_id, count in sorted(dataset_counts.items()):
        md_lines.append(f"- `{dataset_id}`: {count}")
    num_datasets = len(dataset_counts)
    if education_4_runs > 0 and total_runs > 0:
        md_lines.append("")
        md_lines.append("### Document diversity note")
        md_lines.append("")
        if num_datasets > 1:
            other_runs = total_runs - education_4_runs
            md_lines.append(
                f"The corpus spans {num_datasets} datasets with distinct question types: "
                f"HotpotQA (multi-hop QA), QASPER (scientific QA), "
                f"ExplainMeetSum (abstractive meeting explanation), and QMSum "
                f"(query-focused meeting summarization). FEVER is intentionally excluded "
                f"because the current integration lacks FEVER wiki-pages and FEVER-standard "
                f"claim-to-evidence retrieval. ExplainMeetSum and QMSum both draw from the "
                f"same parliamentary transcript (`education_4`), contributing {education_4_runs} "
                f"of {total_runs} cases against a single long source document under two "
                f"different summarization strategies. The remaining {other_runs} cases "
                f"(HotpotQA, QASPER) each use distinct source documents. Dataset diversity "
                f"in this corpus is therefore diversity of *question type* and *ContextView "
                f"construction pattern*; source-document diversity is partial and declared."
            )
        else:
            md_lines.append(
                f"This evaluation slice uses {total_runs} cases from ExplainMeetSum, "
                f"all derived from the same parliamentary transcript (`education_4`) "
                f"under a long-context budget. "
                f"It is a single-dataset, single-document evaluation targeting the "
                f"long-context retrieval regime. Dataset and source-document diversity "
                f"are not claimed for this slice; its purpose is to demonstrate bounded "
                f"ContextView and zoom properties at extended context length."
            )
    md_lines.extend(["", "## Claim matrix", "", "| Claim | Status | Evidence rate | Evidence |", "|---|---:|---:|---|"])
    for claim in report.get("claims", []):
        rate_value = claim.get("evidence_rate")
        rate_text = "n/a" if rate_value is None else f"{float(rate_value):.3f}"
        md_lines.append(f"| {claim.get('claim_id')} — {claim.get('claim')} | {claim.get('status')} | {rate_text} | {claim.get('evidence')} |")

    # Evidence Utilization Rate section
    eur_data = report.get("evidence_utilization") or {}
    eur_global = eur_data.get("global") or {}
    eur_by_ds = eur_data.get("by_dataset") or {}
    if eur_global.get("n"):
        md_lines.extend([
            "",
            "## Evidence Utilization Rate (EUR)",
            "",
            (
                "EUR measures the fraction of ContextView markers that end up cited in the answer "
                "(cited_markers / available_marker_count). It is a descriptive efficiency metric, "
                "not a conformance gate: low EUR indicates generous context selection; "
                "high EUR indicates tight question-specific selection. "
                "CBC = 1.0 holds independently of EUR."
            ),
            "",
            f"**Global** (n={eur_global['n']}): "
            f"mean={eur_global['mean']:.3f}, "
            f"min={eur_global['min']:.3f}, "
            f"max={eur_global['max']:.3f}",
            "",
            "| Dataset | n | mean EUR | min | max |",
            "|---|---:|---:|---:|---:|",
        ])
        for ds, stats in sorted(eur_by_ds.items()):
            md_lines.append(
                f"| `{ds}` | {stats['n']} | {stats['mean']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
            )
        md_lines.extend([
            "",
            (
                "*Interpretation:* factual/multi-hop datasets such as HotpotQA can produce low EUR "
                "because a precise answer may use few markers from a broad ContextView. "
                "Summarization datasets (QMSum, ExplainMeetSum) can produce higher EUR because "
                "comprehensive answers mobilize more of the available evidence. "
                "This stratification is expected and reflects different ContextView construction "
                "patterns, not a difference in traceability quality. FEVER is excluded from "
                "this report until FEVER-standard evidence materialization and retrieval exist."
            ),
        ])

    md_lines.extend([
        "",
        "## Publication guardrail",
        "",
        "This report demonstrates operational evidence artifacts. It must not be worded as full legal compliance, full TaskMemory, full lifecycle, or full global-memory evidence unless those features are separately demonstrated.",
    ])
    (out_dir / "publication_evidence_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a publication-evidence claim matrix from CIMA open-scenario runs.")
    parser.add_argument("--runs", type=Path, required=True, help="Directory containing run_manifest.json artifacts.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for publication evidence artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_runs(args.runs)
    write_outputs(report, args.out)
    print(json.dumps({
        "out": str(args.out),
        "run_count": report.get("run_count", 0),
        "rates": report.get("rates", {}),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
