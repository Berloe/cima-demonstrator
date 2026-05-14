"""Evidence-book writers for reproducible demonstrator scenarios."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .report_builder import ScenarioExecutionResult


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _scenario_dataset_file(scenario_id: str) -> str:
    lower = scenario_id.lower()
    if lower.startswith("a_"):
        return "scenario_a_long_context_virtual.json"
    if lower.startswith("b_"):
        return "scenario_b_traceability.json"
    if lower.startswith("c_"):
        return "scenario_c_handoff_continuity.json"
    if lower.startswith("d_"):
        return "scenario_d_zoom_operators.json"
    if lower.startswith("e_"):
        return "scenario_e_held_out_boundary.json"
    if lower.startswith("f_"):
        return "scenario_f_negative_conflict.json"
    return ""


def _roadmap_targets(scenario_id: str) -> list[str]:
    lower = scenario_id.lower()
    if lower.startswith("a_"):
        return ["R3.1"]
    if lower.startswith("b_"):
        return ["R3.2"]
    if lower.startswith("c_"):
        return ["R3.3"]
    if lower.startswith("d_"):
        return ["R3.4"]
    if lower.startswith("e_"):
        return ["R3.5"]
    if lower.startswith("f_"):
        return ["R3.6"]
    return []


@dataclass(slots=True)
class ScenarioEvidenceManifest:
    schema_version: str
    generated_at: str
    scenario_id: str
    title: str
    roadmap_targets: list[str]
    acceptance_passed: bool
    acceptance_checks: list[dict[str, Any]]
    dataset: dict[str, Any]
    source_run: dict[str, Any]
    continuation_run: dict[str, Any] | None
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceBookIndex:
    schema_version: str
    generated_at: str
    scenario_count: int
    roadmap_coverage: dict[str, list[str]]
    manifests: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_scenario_evidence_manifest(*, result: ScenarioExecutionResult, artifacts_root: Path, dataset_root: Path) -> ScenarioEvidenceManifest:
    scenario = result.scenario
    scenario_file = dataset_root / _scenario_dataset_file(scenario.scenario_id)
    dataset_info = {
        "file_name": scenario_file.name if scenario_file.exists() else None,
        "sha256": _sha256_file(scenario_file) if scenario_file.exists() else None,
        "bytes": scenario_file.stat().st_size if scenario_file.exists() else None,
    }
    acceptance = result.acceptance
    checks = [] if acceptance is None else [check.to_dict() for check in acceptance.checks]

    source_inventory = _artifact_inventory(result.artifacts_dir)
    source_run = {
        "conversation_id": result.conversation_id,
        "run_id": result.run_id,
        "artifacts_dir": str(result.artifacts_dir),
        "artifact_inventory": source_inventory,
    }

    continuation_run: dict[str, Any] | None = None
    target_run_id = None
    target_conversation_id = None
    if result.handoff_restore:
        target_run_id = result.handoff_restore.get("target_run_id")
        target_conversation_id = result.handoff_restore.get("target_conversation_id")
    if target_run_id and target_conversation_id:
        target_dir = Path(artifacts_root) / str(target_conversation_id) / str(target_run_id)
        continuation_run = {
            "conversation_id": str(target_conversation_id),
            "run_id": str(target_run_id),
            "artifacts_dir": str(target_dir),
            "artifact_inventory": _artifact_inventory(target_dir),
        }

    selected_tokens = 0
    if result.budget_trace:
        selected_tokens = int(result.budget_trace.get("tokens_used") or 0)
    metrics: dict[str, Any] = {
        "window_tokens": int(scenario.window_tokens),
        "corpus_tokens": int(result.corpus_tokens),
        "selected_tokens": selected_tokens,
        "selected_smaller_than_corpus": bool(selected_tokens < result.corpus_tokens) if selected_tokens else False,
    }
    if scenario.window_tokens:
        metrics["corpus_to_window_ratio"] = float(result.corpus_tokens) / float(scenario.window_tokens)
    if selected_tokens:
        metrics["corpus_to_selected_ratio"] = float(result.corpus_tokens) / float(selected_tokens)

    lower = scenario.scenario_id.lower()
    if lower.startswith("a_"):
        metrics["scenario_focus"] = "bounded_context_under_long_corpus"
        metrics["requires_zoom"] = bool(scenario.expectations.requires_zoom)
        metrics["requires_zoom_out"] = bool(scenario.expectations.requires_zoom_out)
        metrics["required_markers"] = list(scenario.expectations.required_markers)
        metrics["context_snapshot_id"] = None if result.context_snapshot is None else result.context_snapshot.get("context_id")
    if lower.startswith("b_"):
        answer_lineage = result.answer_lineage or {}
        marker_resolution = list(answer_lineage.get("marker_resolution") or [])
        summary_resolution_files = sorted(
            row["relative_path"]
            for row in source_inventory
            if str(row.get("relative_path") or "").startswith("summary_resolution_")
        )
        metrics.update(
            {
                "scenario_focus": "traceability_from_answer_to_source_spans",
                "answer_lineage_id": answer_lineage.get("answer_lineage_id"),
                "answer_lineage_present": bool(answer_lineage),
                "resolved_source_count": int(answer_lineage.get("resolved_source_count") or 0),
                "resolved_span_count": int(answer_lineage.get("resolved_span_count") or 0),
                "marker_resolution_count": len(marker_resolution),
                "lineage_resolution_mode": str(answer_lineage.get("resolution_mode") or "empty"),
                "unresolved_ref_ids": list(answer_lineage.get("unresolved_ref_ids") or []),
                "summary_resolution_artifacts": summary_resolution_files,
                "lineage_edges_artifact_present": any(
                    row["relative_path"] == "lineage_edges.jsonl" for row in source_inventory
                ),
                "required_markers": list(scenario.expectations.required_markers),
            }
        )
    if lower.startswith("c_"):
        validation = result.handoff_validation or {}
        restore = result.handoff_restore or {}
        metrics.update(
            {
                "scenario_focus": "portable_handoff_continuity",
                "handoff_id": None if result.handoff_manifest is None else result.handoff_manifest.get("handoff_id"),
                "handoff_valid": bool(validation.get("valid")),
                "evidence_coverage": float(validation.get("evidence_coverage") or 0.0),
                "restore_valid": bool(restore.get("valid")),
                "restore_diff": dict(restore.get("diff") or {}),
                "resume_answer_present": bool(result.resumed_answer_text),
                "required_markers": list(scenario.expectations.required_markers),
            }
        )
    if lower.startswith("d_"):
        zoom = result.zoom_result or {}
        zoom_out = result.zoom_out_result or {}
        metrics.update(
            {
                "scenario_focus": "zoom_in_and_zoom_out_navigation",
                "zoom_valid": bool(zoom.get("markers_added")) and bool(zoom.get("evidence_block")),
                "zoom_out_valid": bool(zoom_out.get("markers_added")) and bool(zoom_out.get("perspective_block")),
                "zoom_marker_count": len(list(zoom.get("markers_added") or [])),
                "zoom_out_marker_count": len(list(zoom_out.get("markers_added") or [])),
                "zoom_resolution_mode": str(zoom.get("resolution_mode") or "empty"),
                "zoom_out_resolution_mode": str(zoom_out.get("resolution_mode") or "empty"),
                "zoom_marker_resolution_count": len(list(zoom.get("marker_resolution") or [])),
                "zoom_out_marker_resolution_count": len(list(zoom_out.get("marker_resolution") or [])),
                "required_markers": list(scenario.expectations.required_markers),
            }
        )
    if lower.startswith("e_"):
        answer_lineage = result.answer_lineage or {}
        metrics.update(
            {
                "scenario_focus": "held_out_scenario_under_same_artifact_discipline",
                "held_out": True,
                "answer_lineage_present": bool(answer_lineage),
                "resolved_source_count": int(answer_lineage.get("resolved_source_count") or 0),
                "resolved_span_count": int(answer_lineage.get("resolved_span_count") or 0),
                "lineage_resolution_mode": str(answer_lineage.get("resolution_mode") or "empty"),
                "required_markers": list(scenario.expectations.required_markers),
            }
        )
    if lower.startswith("f_"):
        answer_lineage = result.answer_lineage or {}
        answer_text_lower = result.answer_text.lower()
        forbidden_hits = [
            token
            for token in list(scenario.expectations.forbidden_answer_contains)
            if token.lower() in answer_text_lower
        ]
        metrics.update(
            {
                "scenario_focus": "negative_safe_degradation_under_conflict",
                "safe_degradation_present": all(
                    token.lower() in answer_text_lower for token in list(scenario.expectations.answer_contains)
                ),
                "forbidden_phrase_hits": forbidden_hits,
                "resolved_source_count": int(answer_lineage.get("resolved_source_count") or 0),
                "resolved_span_count": int(answer_lineage.get("resolved_span_count") or 0),
                "marker_resolution_count": len(list(answer_lineage.get("marker_resolution") or [])),
                "lineage_resolution_mode": str(answer_lineage.get("resolution_mode") or "empty"),
                "required_markers": list(scenario.expectations.required_markers),
            }
        )

    return ScenarioEvidenceManifest(
        schema_version="cima_demo.scenario_evidence_manifest.v1",
        generated_at=_utc_now(),
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        roadmap_targets=_roadmap_targets(scenario.scenario_id),
        acceptance_passed=bool(acceptance.passed) if acceptance is not None else False,
        acceptance_checks=checks,
        dataset=dataset_info,
        source_run=source_run,
        continuation_run=continuation_run,
        metrics=metrics,
    )


def write_scenario_evidence_manifest(*, result: ScenarioExecutionResult, artifacts_root: Path, dataset_root: Path) -> Path:
    manifest = build_scenario_evidence_manifest(result=result, artifacts_root=artifacts_root, dataset_root=dataset_root)
    path = result.artifacts_dir / "scenario_evidence_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_evidence_book_index(*, root: Path, scenario_manifest_paths: list[Path]) -> EvidenceBookIndex:
    coverage: dict[str, list[str]] = {}
    manifests: list[dict[str, Any]] = []
    for path in sorted(scenario_manifest_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = str(payload.get("scenario_id") or "")
        for target in payload.get("roadmap_targets") or []:
            coverage.setdefault(str(target), []).append(scenario_id)
        manifests.append(
            {
                "scenario_id": scenario_id,
                "path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "acceptance_passed": bool(payload.get("acceptance_passed", False)),
            }
        )
    for value in coverage.values():
        value.sort()
    return EvidenceBookIndex(
        schema_version="cima_demo.evidence_book_index.v1",
        generated_at=_utc_now(),
        scenario_count=len(scenario_manifest_paths),
        roadmap_coverage=coverage,
        manifests=manifests,
    )


def write_evidence_book(*, root: Path, results: list[ScenarioExecutionResult], dataset_root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_paths = [write_scenario_evidence_manifest(result=result, artifacts_root=root, dataset_root=dataset_root) for result in results]
    index = build_evidence_book_index(root=root, scenario_manifest_paths=manifest_paths)
    index_path = root / "evidence_book_index.json"
    index_path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CIMA Demonstrator — Evidence Book",
        "",
        f"Generated at: {index.generated_at}",
        f"Scenario manifests: {index.scenario_count}",
        "",
    ]
    for entry in index.manifests:
        payload = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        lines.extend(
            [
                f"## {entry['scenario_id']}",
                "",
                f"Manifest: `{entry['path']}`",
                f"SHA256: `{entry['sha256']}`",
                f"Acceptance passed: `{entry['acceptance_passed']}`",
                f"Roadmap targets: `{', '.join(payload.get('roadmap_targets') or [])}`",
                f"Focus: `{payload.get('metrics', {}).get('scenario_focus', '')}`",
                "",
            ]
        )
    md_path = root / "evidence_book.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"index": index_path, "markdown": md_path}
