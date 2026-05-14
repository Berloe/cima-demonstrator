from __future__ import annotations

import json
from pathlib import Path

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_evidence_book_writes_reproducible_manifests_for_r3_2_and_r3_4(tmp_path: Path) -> None:
    selected_ids = {"B_TRACEABILITY", "D_ZOOM_OPERATORS"}
    scenarios = [scenario for scenario in load_all_scenarios() if scenario.scenario_id in selected_ids]
    runner = DemoScenarioRunner(artifacts_root=tmp_path, scenarios=scenarios)

    results = await runner.run_all()

    assert {result.scenario.scenario_id for result in results} == selected_ids

    evidence_index_path = tmp_path / "evidence_book_index.json"
    evidence_md_path = tmp_path / "evidence_book.md"
    assert evidence_index_path.exists()
    assert evidence_md_path.exists()

    evidence_index = json.loads(evidence_index_path.read_text(encoding="utf-8"))
    assert evidence_index["schema_version"] == "cima_demo.evidence_book_index.v1"
    assert evidence_index["roadmap_coverage"]["R3.2"] == ["B_TRACEABILITY"]
    assert evidence_index["roadmap_coverage"]["R3.4"] == ["D_ZOOM_OPERATORS"]

    manifests: dict[str, dict] = {}
    for entry in evidence_index["manifests"]:
        manifest_path = tmp_path / entry["path"]
        assert manifest_path.exists()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[payload["scenario_id"]] = payload
        assert entry["sha256"]
        assert entry["bytes"] > 0
        assert payload["acceptance_passed"] is True

    b_manifest = manifests["B_TRACEABILITY"]
    assert b_manifest["roadmap_targets"] == ["R3.2"]
    assert b_manifest["dataset"]["sha256"]
    assert b_manifest["metrics"]["scenario_focus"] == "traceability_from_answer_to_source_spans"
    assert b_manifest["metrics"]["answer_lineage_present"] is True
    assert b_manifest["metrics"]["resolved_source_count"] > 0
    assert b_manifest["metrics"]["resolved_span_count"] > 0
    assert b_manifest["metrics"]["lineage_edges_artifact_present"] is True
    assert b_manifest["metrics"]["summary_resolution_artifacts"]
    b_artifacts = {row["relative_path"] for row in b_manifest["source_run"]["artifact_inventory"]}
    assert any(path.startswith("answer_lineage_") for path in b_artifacts)
    assert "lineage_edges.jsonl" in b_artifacts

    d_manifest = manifests["D_ZOOM_OPERATORS"]
    assert d_manifest["roadmap_targets"] == ["R3.4"]
    assert d_manifest["dataset"]["sha256"]
    assert d_manifest["metrics"]["scenario_focus"] == "zoom_in_and_zoom_out_navigation"
    assert d_manifest["metrics"]["zoom_valid"] is True
    assert d_manifest["metrics"]["zoom_out_valid"] is True
    assert d_manifest["metrics"]["zoom_marker_count"] > 0
    assert d_manifest["metrics"]["zoom_out_marker_count"] > 0
    assert d_manifest["metrics"]["zoom_marker_resolution_count"] > 0
    assert d_manifest["metrics"]["zoom_out_marker_resolution_count"] > 0
    d_artifacts = {row["relative_path"] for row in d_manifest["source_run"]["artifact_inventory"]}
    assert any(path.startswith("zoom_trace_") for path in d_artifacts)
    assert any(path.startswith("zoom_out_trace_") for path in d_artifacts)
