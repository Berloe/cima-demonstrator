"""Canonical citable evidence marker registry for CIMA demo runtime artifacts.

The registry separates two different questions that used to be conflated:

* Is a marker useful context for the model?
* Is a marker admissible as a citation marker in ``allowed_markers``?

Only the second class is allowed to use an ``S#`` marker.  Non-citable items may
still be retained as auxiliary context, but they are explicitly recorded as
uncitable and cannot enter the closed citation set.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

SUMMARY_KINDS = {"summary", "local_summary", "global_summary", "summary_chunk"}
CITABLE_STATUSES = {"source_span", "summary_witness"}


@dataclass(frozen=True, slots=True)
class EvidenceMarker:
    marker: str
    kind: str
    ref_id: str
    citable: bool
    resolution_status: str
    source_ids: list[str]
    spans: list[str]
    lineage_refs: list[str]
    reason_if_not_citable: str = ""
    effective_citem_ids: list[str] = field(default_factory=list)
    unresolved_citem_ids: list[str] = field(default_factory=list)
    citem_witnesses: list[dict[str, Any]] = field(default_factory=list)
    duplicate_marker_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _is_summary_kind(kind: str) -> bool:
    return kind.lower() in SUMMARY_KINDS


def _witnesses_cover_effective_citems(row: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Return whether every effective C-item in a summary row has source/span.

    New runtime rows include ``citem_witnesses`` so the check is direct.  Older
    rows only contain aggregate counts, which is intentionally insufficient for
    strict summary citability: aggregate source/span coverage can hide a missing
    direct witness for one of the summary's effective inputs.
    """
    if not row:
        return False, []
    citem_ids = _dedupe(list(row.get("citem_ids") or row.get("effective_citem_ids") or []))
    if not citem_ids:
        return False, []
    witnesses_raw = row.get("citem_witnesses") or []
    if not witnesses_raw:
        return False, citem_ids
    by_citem: dict[str, dict[str, Any]] = {}
    for witness in witnesses_raw:
        if not isinstance(witness, dict):
            continue
        citem_id = str(witness.get("citem_id") or "").strip()
        if citem_id:
            by_citem[citem_id] = witness
    missing: list[str] = []
    for citem_id in citem_ids:
        witness = by_citem.get(citem_id) or {}
        if not list(witness.get("source_ids") or []) or not list(witness.get("span_ids") or []):
            missing.append(citem_id)
    explicit_unresolved = _dedupe(list(row.get("unresolved_citem_ids") or []))
    missing = _dedupe([*missing, *explicit_unresolved])
    return not missing, missing


def marker_resolution_status(row: dict[str, Any] | None) -> str:
    if not row:
        return "unresolved"
    if list(row.get("unresolved_ref_ids") or []):
        return "unresolved"
    if int(row.get("resolved_source_count") or 0) <= 0 or int(row.get("resolved_span_count") or 0) <= 0:
        return "unresolved"
    kind = str(row.get("ref_kind") or row.get("summary_ref_kind") or "").lower()
    if _is_summary_kind(kind):
        covered, _missing = _witnesses_cover_effective_citems(row)
        if covered:
            return "summary_witness"
        return "unresolved"
    return "source_span"


def _reason_for(row: dict[str, Any] | None, status: str, *, duplicate_marker_count: int = 1) -> str:
    if duplicate_marker_count > 1:
        return "duplicate_marker_ambiguous"
    if status in CITABLE_STATUSES:
        return ""
    if not row:
        return "missing_marker_resolution"
    if list(row.get("unresolved_ref_ids") or []):
        return "unresolved_ref_ids"
    if int(row.get("resolved_source_count") or 0) <= 0:
        return "missing_source"
    if int(row.get("resolved_span_count") or 0) <= 0:
        return "missing_span"
    kind = str(row.get("ref_kind") or row.get("summary_ref_kind") or "").lower()
    if _is_summary_kind(kind):
        if not list(row.get("citem_ids") or row.get("effective_citem_ids") or []):
            return "summary_missing_effective_input_witness"
        _covered, missing = _witnesses_cover_effective_citems(row)
        if missing:
            return "summary_missing_direct_witness_for_effective_inputs"
        return "summary_missing_strict_witness_detail"
    return "unresolved"


def _row_key(row: dict[str, Any] | None) -> tuple[str, str, str]:
    row = row or {}
    return (str(row.get("marker") or ""), str(row.get("ref_kind") or ""), str(row.get("ref_id") or ""))


def _select_row_for_marker(
    *,
    marker: str,
    item: dict[str, Any],
    rows_by_marker: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, int]:
    rows = rows_by_marker.get(marker) or []
    if not rows:
        return None, 0
    exact = [
        row for row in rows
        if str(row.get("ref_kind") or "") == str(item.get("ref_kind") or "")
        and str(row.get("ref_id") or "") == str(item.get("ref_id") or "")
    ]
    if len(exact) == 1:
        return exact[0], len(rows)
    if len(rows) == 1:
        return rows[0], 1
    # Ambiguous marker labels are deliberately not resolved by "best effort".
    # CIMA marker identity must be unique within the active ContextView.
    return None, len(rows)


def build_registry(*, items: list[dict[str, Any]], marker_resolution: list[dict[str, Any]]) -> list[EvidenceMarker]:
    rows_by_marker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in marker_resolution or []:
        marker = str(row.get("marker") or "").strip()
        if marker:
            rows_by_marker[marker].append(dict(row))

    markers: list[str] = []
    by_marker: dict[str, dict[str, Any]] = {}
    marker_counts: Counter[str] = Counter()
    for item in items or []:
        marker = str(item.get("marker") or "").strip()
        if not marker:
            continue
        markers.append(marker)
        marker_counts[marker] += 1
        by_marker.setdefault(marker, dict(item))
    for marker in rows_by_marker:
        if marker not in by_marker:
            markers.append(marker)
            marker_counts[marker] += 1
            by_marker[marker] = {}

    out: list[EvidenceMarker] = []
    seen: set[str] = set()
    for marker in markers:
        if marker in seen:
            continue
        seen.add(marker)
        item = by_marker.get(marker) or {}
        row, row_duplicate_count = _select_row_for_marker(marker=marker, item=item, rows_by_marker=rows_by_marker)
        duplicate_marker_count = max(marker_counts.get(marker, 1), row_duplicate_count or 1)
        ambiguous_duplicate = duplicate_marker_count > 1 and row is None
        kind = str(item.get("ref_kind") or (row or {}).get("ref_kind") or "citem")
        ref_id = str(item.get("ref_id") or (row or {}).get("ref_id") or "")
        status = "unresolved" if ambiguous_duplicate else marker_resolution_status(row)
        source_ids = _dedupe(list((row or {}).get("resolved_source_ids") or []))
        spans = _dedupe(list((row or {}).get("resolved_span_ids") or []))
        effective_citem_ids = _dedupe(list((row or {}).get("citem_ids") or (row or {}).get("effective_citem_ids") or []))
        unresolved_citem_ids = _dedupe(list((row or {}).get("unresolved_citem_ids") or []))
        _covered, missing = _witnesses_cover_effective_citems(row) if _is_summary_kind(kind) else (True, [])
        unresolved_citem_ids = _dedupe([*unresolved_citem_ids, *missing])
        lineage_refs = _dedupe(effective_citem_ids + list((row or {}).get("summary_ids") or []))
        citem_witnesses = [dict(v) for v in list((row or {}).get("citem_witnesses") or []) if isinstance(v, dict)]
        reason = _reason_for(row, status, duplicate_marker_count=duplicate_marker_count if ambiguous_duplicate else 1)
        out.append(EvidenceMarker(
            marker=marker,
            kind=kind,
            ref_id=ref_id,
            citable=status in CITABLE_STATUSES,
            resolution_status=status,
            source_ids=source_ids,
            spans=spans,
            lineage_refs=lineage_refs,
            reason_if_not_citable=reason,
            effective_citem_ids=effective_citem_ids,
            unresolved_citem_ids=unresolved_citem_ids,
            citem_witnesses=citem_witnesses,
            duplicate_marker_count=duplicate_marker_count,
        ))
    return out


def citable_markers(registry: list[EvidenceMarker]) -> set[str]:
    return {entry.marker for entry in registry if entry.citable and entry.resolution_status in CITABLE_STATUSES}


def split_citable_and_auxiliary_items(
    *,
    items: list[dict[str, Any]],
    marker_resolution: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    registry = build_registry(items=items, marker_resolution=marker_resolution)
    keep = citable_markers(registry)
    registry_by_marker = {entry.marker: entry.to_dict() for entry in registry}
    citable_items: list[dict[str, Any]] = []
    auxiliary_items: list[dict[str, Any]] = []
    for item in items or []:
        marker = str(item.get("marker") or "")
        materialized = dict(item)
        if marker in keep:
            citable_items.append(materialized)
        else:
            reg = dict(registry_by_marker.get(marker) or {})
            materialized["uncitable_marker"] = marker
            materialized["citation_marker_removed"] = True
            materialized["reason_if_not_citable"] = str(reg.get("reason_if_not_citable") or "unresolved")
            # Never expose an S#/P# marker for auxiliary context.
            materialized.pop("marker", None)
            auxiliary_items.append(materialized)
    filtered_resolution = [dict(row) for row in marker_resolution or [] if str(row.get("marker") or "") in keep]
    return citable_items, auxiliary_items, filtered_resolution, [entry.to_dict() for entry in registry]


def filter_citable_items(*, items: list[dict[str, Any]], marker_resolution: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    citable_items, _auxiliary_items, filtered_resolution, registry = split_citable_and_auxiliary_items(
        items=items,
        marker_resolution=marker_resolution,
    )
    return citable_items, filtered_resolution, registry
