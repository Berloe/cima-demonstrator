from __future__ import annotations

import json
from pathlib import Path

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_evidence_book_writes_reproducible_manifests_for_r3_5_and_r3_6(tmp_path: Path) -> None:
    selected_ids = {"E_HELD_OUT_BOUNDARY", "F_NEGATIVE_CONFLICT"}
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
    assert evidence_index["roadmap_coverage"]["R3.5"] == ["E_HELD_OUT_BOUNDARY"]
    assert evidence_index["roadmap_coverage"]["R3.6"] == ["F_NEGATIVE_CONFLICT"]

    manifests: dict[str, dict] = {}
    for entry in evidence_index["manifests"]:
        manifest_path = tmp_path / entry["path"]
        assert manifest_path.exists()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[payload["scenario_id"]] = payload
        assert entry["sha256"]
        assert entry["bytes"] > 0
        assert payload["acceptance_passed"] is True

    e_manifest = manifests["E_HELD_OUT_BOUNDARY"]
    assert e_manifest["roadmap_targets"] == ["R3.5"]
    assert e_manifest["dataset"]["sha256"]
    assert e_manifest["metrics"]["scenario_focus"] == "held_out_scenario_under_same_artifact_discipline"
    assert e_manifest["metrics"]["held_out"] is True
    assert e_manifest["metrics"]["answer_lineage_present"] is True
    assert e_manifest["metrics"]["resolved_source_count"] > 0
    assert e_manifest["metrics"]["resolved_span_count"] > 0
    e_artifacts = {row["relative_path"] for row in e_manifest["source_run"]["artifact_inventory"]}
    assert any(path.startswith("answer_lineage_") for path in e_artifacts)
    assert any(path.startswith("zoom_trace_") for path in e_artifacts)

    f_manifest = manifests["F_NEGATIVE_CONFLICT"]
    assert f_manifest["roadmap_targets"] == ["R3.6"]
    assert f_manifest["dataset"]["sha256"]
    assert f_manifest["metrics"]["scenario_focus"] == "negative_safe_degradation_under_conflict"
    assert f_manifest["metrics"]["safe_degradation_present"] is True
    assert f_manifest["metrics"]["forbidden_phrase_hits"] == []
    assert f_manifest["metrics"]["resolved_source_count"] > 0
    assert f_manifest["metrics"]["resolved_span_count"] > 0
    f_artifacts = {row["relative_path"] for row in f_manifest["source_run"]["artifact_inventory"]}
    assert any(path.startswith("answer_lineage_") for path in f_artifacts)
    assert any(path.startswith("zoom_trace_") for path in f_artifacts)
