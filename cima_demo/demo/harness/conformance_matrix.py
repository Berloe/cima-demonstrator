"""Conformance-matrix writers that distinguish implemented from demonstrated."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


STATUS_YES = "yes"
STATUS_PARTIAL = "partial"
STATUS_NO = "no"


@dataclass(slots=True)
class ConformanceRow:
    item_id: str
    item_label: str
    category: str
    implemented_status: str
    demonstrated_status: str
    roadmap_targets: list[str]
    scenario_ids: list[str]
    evidence_paths: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConformanceMatrix:
    schema_version: str
    generated_at: str
    summary: dict[str, Any]
    rows: list[ConformanceRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "rows": [row.to_dict() for row in self.rows],
        }


def _load_manifests(root: Path) -> dict[str, dict[str, Any]]:
    index_path = root / "evidence_book_index.json"
    if not index_path.exists():
        return {}
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    manifests: dict[str, dict[str, Any]] = {}
    for entry in index_payload.get("manifests") or []:
        path = root / str(entry.get("path") or "")
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[str(payload.get("scenario_id") or "")] = payload
    return manifests


def _manifest_path(root: Path, scenario_id: str) -> str | None:
    manifests = _load_manifests(root)
    payload = manifests.get(scenario_id)
    if payload is None:
        return None
    for entry in json.loads((root / "evidence_book_index.json").read_text(encoding="utf-8")).get("manifests") or []:
        if entry.get("scenario_id") == scenario_id:
            return str(entry.get("path"))
    return None


def _has_passed(manifests: dict[str, dict[str, Any]], scenario_id: str) -> bool:
    payload = manifests.get(scenario_id)
    return bool(payload and payload.get("acceptance_passed") is True)


def build_conformance_matrix(*, root: Path) -> ConformanceMatrix:
    manifests = _load_manifests(root)

    def evidence_for(*scenario_ids: str) -> list[str]:
        index_payload = {}
        index_path = root / "evidence_book_index.json"
        if index_path.exists():
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        by_id = {str(entry.get("scenario_id") or ""): str(entry.get("path") or "") for entry in index_payload.get("manifests") or []}
        return [by_id[sid] for sid in scenario_ids if sid in by_id]

    rows = [
        ConformanceRow(
            item_id="C-07",
            item_label="ContextView",
            category="construct",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "A_LONG_CONTEXT_VIRTUAL") else STATUS_NO,
            roadmap_targets=["R3.1"],
            scenario_ids=["A_LONG_CONTEXT_VIRTUAL"],
            evidence_paths=evidence_for("A_LONG_CONTEXT_VIRTUAL"),
            note="Bounded active context is implemented in runtime and demonstrated by the long-context scenario.",
        ),
        ConformanceRow(
            item_id="O-03",
            item_label="Select",
            category="operator",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "A_LONG_CONTEXT_VIRTUAL") else STATUS_NO,
            roadmap_targets=["R3.1"],
            scenario_ids=["A_LONG_CONTEXT_VIRTUAL"],
            evidence_paths=evidence_for("A_LONG_CONTEXT_VIRTUAL"),
            note="Budget-bounded selection is exercised by Scenario A.",
        ),
        ConformanceRow(
            item_id="P-01",
            item_label="Bounded attention",
            category="invariant",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "A_LONG_CONTEXT_VIRTUAL") else STATUS_NO,
            roadmap_targets=["R3.1"],
            scenario_ids=["A_LONG_CONTEXT_VIRTUAL"],
            evidence_paths=evidence_for("A_LONG_CONTEXT_VIRTUAL"),
            note="The active window remains below corpus size and is explicitly measured.",
        ),
        ConformanceRow(
            item_id="P-02",
            item_label="Traceability / auditability",
            category="invariant",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "B_TRACEABILITY") else STATUS_NO,
            roadmap_targets=["R3.2"],
            scenario_ids=["B_TRACEABILITY"],
            evidence_paths=evidence_for("B_TRACEABILITY"),
            note="Answer lineage, summary-resolution artifacts, and lineage edges are all exported for Scenario B.",
        ),
        ConformanceRow(
            item_id="O-04",
            item_label="Zoom",
            category="operator",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "D_ZOOM_OPERATORS") else STATUS_NO,
            roadmap_targets=["R3.4"],
            scenario_ids=["D_ZOOM_OPERATORS"],
            evidence_paths=evidence_for("D_ZOOM_OPERATORS"),
            note="Zoom is demonstrated with marker-level resolution evidence in Scenario D.",
        ),
        ConformanceRow(
            item_id="O-05",
            item_label="Zoom-out",
            category="operator",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "D_ZOOM_OPERATORS") else STATUS_NO,
            roadmap_targets=["R3.4"],
            scenario_ids=["D_ZOOM_OPERATORS"],
            evidence_paths=evidence_for("D_ZOOM_OPERATORS"),
            note="Zoom-out is demonstrated alongside Zoom under the same artifact discipline.",
        ),
        ConformanceRow(
            item_id="O-09",
            item_label="Handoff",
            category="operator",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_PARTIAL if _has_passed(manifests, "C_HANDOFF_CONTINUITY") else STATUS_NO,
            roadmap_targets=["R3.3"],
            scenario_ids=["C_HANDOFF_CONTINUITY"],
            evidence_paths=evidence_for("C_HANDOFF_CONTINUITY"),
            note="Handoff is mechanically evidenced, but strong semantic-equivalence claims remain intentionally modest.",
        ),
        ConformanceRow(
            item_id="O-10",
            item_label="Restore",
            category="operator",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_PARTIAL if _has_passed(manifests, "C_HANDOFF_CONTINUITY") else STATUS_NO,
            roadmap_targets=["R3.3"],
            scenario_ids=["C_HANDOFF_CONTINUITY"],
            evidence_paths=evidence_for("C_HANDOFF_CONTINUITY"),
            note="Restore is evidenced with validation and diff artifacts, but equivalence remains a bounded claim.",
        ),
        ConformanceRow(
            item_id="P-03",
            item_label="Continuity across handoffs",
            category="invariant",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_PARTIAL if _has_passed(manifests, "C_HANDOFF_CONTINUITY") else STATUS_NO,
            roadmap_targets=["R3.3"],
            scenario_ids=["C_HANDOFF_CONTINUITY"],
            evidence_paths=evidence_for("C_HANDOFF_CONTINUITY"),
            note="Continuity is demonstrated up to evidence coverage and restore diff, not as a strong universal equivalence theorem.",
        ),
        ConformanceRow(
            item_id="G-4",
            item_label="Held-out scenario",
            category="publication_gate",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "E_HELD_OUT_BOUNDARY") else STATUS_NO,
            roadmap_targets=["R3.5"],
            scenario_ids=["E_HELD_OUT_BOUNDARY"],
            evidence_paths=evidence_for("E_HELD_OUT_BOUNDARY"),
            note="A held-out scenario now runs under the same artifact discipline as the baseline harness.",
        ),
        ConformanceRow(
            item_id="G-5",
            item_label="Negative scenario",
            category="publication_gate",
            implemented_status=STATUS_YES,
            demonstrated_status=STATUS_YES if _has_passed(manifests, "F_NEGATIVE_CONFLICT") else STATUS_NO,
            roadmap_targets=["R3.6"],
            scenario_ids=["F_NEGATIVE_CONFLICT"],
            evidence_paths=evidence_for("F_NEGATIVE_CONFLICT"),
            note="A negative scenario now checks safe degradation and forbids overclaiming phrases.",
        ),
        ConformanceRow(
            item_id="C-10",
            item_label="Authority domain",
            category="construct",
            implemented_status=STATUS_NO,
            demonstrated_status=STATUS_NO,
            roadmap_targets=[],
            scenario_ids=[],
            evidence_paths=[],
            note="Still outside the current demonstrator scope; kept explicit to avoid silent overclaiming.",
        ),
    ]

    summary = {
        "implemented_yes": sum(1 for row in rows if row.implemented_status == STATUS_YES),
        "implemented_partial": sum(1 for row in rows if row.implemented_status == STATUS_PARTIAL),
        "implemented_no": sum(1 for row in rows if row.implemented_status == STATUS_NO),
        "demonstrated_yes": sum(1 for row in rows if row.demonstrated_status == STATUS_YES),
        "demonstrated_partial": sum(1 for row in rows if row.demonstrated_status == STATUS_PARTIAL),
        "demonstrated_no": sum(1 for row in rows if row.demonstrated_status == STATUS_NO),
    }

    return ConformanceMatrix(
        schema_version="cima_demo.conformance_matrix.v1",
        generated_at=_utc_now(),
        summary=summary,
        rows=rows,
    )


def write_conformance_matrix(*, root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    matrix = build_conformance_matrix(root=root)
    json_path = root / "conformance_matrix.json"
    json_path.write_text(json.dumps(matrix.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CIMA Demonstrator — Conformance Matrix",
        "",
        f"Generated at: {matrix.generated_at}",
        f"Implemented yes/partial/no: `{matrix.summary['implemented_yes']}` / `{matrix.summary['implemented_partial']}` / `{matrix.summary['implemented_no']}`",
        f"Demonstrated yes/partial/no: `{matrix.summary['demonstrated_yes']}` / `{matrix.summary['demonstrated_partial']}` / `{matrix.summary['demonstrated_no']}`",
        "",
        "This matrix distinguishes implementation status from demonstration status to avoid overclaiming.",
        "",
    ]
    for row in matrix.rows:
        evidence = ", ".join(f"`{path}`" for path in row.evidence_paths) if row.evidence_paths else "None"
        lines.extend([
            f"## {row.item_id} — {row.item_label}",
            "",
            f"- Category: `{row.category}`",
            f"- Implemented: `{row.implemented_status}`",
            f"- Demonstrated: `{row.demonstrated_status}`",
            f"- Roadmap targets: `{', '.join(row.roadmap_targets) if row.roadmap_targets else '-'}`",
            f"- Scenarios: `{', '.join(row.scenario_ids) if row.scenario_ids else '-'}`",
            f"- Evidence: {evidence}",
            f"- Note: {row.note}",
            "",
        ])
    md_path = root / "conformance_matrix.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
