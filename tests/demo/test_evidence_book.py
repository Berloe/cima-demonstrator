from __future__ import annotations

import json
from pathlib import Path

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_evidence_book_writes_reproducible_manifests_for_r3_1_and_r3_3(tmp_path: Path) -> None:
    selected_ids = {"A_LONG_CONTEXT_VIRTUAL", "C_HANDOFF_CONTINUITY"}
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
    assert evidence_index["roadmap_coverage"]["R3.1"] == ["A_LONG_CONTEXT_VIRTUAL"]
    assert evidence_index["roadmap_coverage"]["R3.3"] == ["C_HANDOFF_CONTINUITY"]

    manifests: dict[str, dict] = {}
    for entry in evidence_index["manifests"]:
        manifest_path = tmp_path / entry["path"]
        assert manifest_path.exists()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[payload["scenario_id"]] = payload
        assert entry["sha256"]
        assert entry["bytes"] > 0
        assert payload["acceptance_passed"] is True

    a_manifest = manifests["A_LONG_CONTEXT_VIRTUAL"]
    assert a_manifest["roadmap_targets"] == ["R3.1"]
    assert a_manifest["dataset"]["sha256"]
    assert a_manifest["metrics"]["scenario_focus"] == "bounded_context_under_long_corpus"
    assert a_manifest["metrics"]["selected_smaller_than_corpus"] is True
    a_artifacts = {row["relative_path"] for row in a_manifest["source_run"]["artifact_inventory"]}
    assert "run_manifest.json" in a_artifacts
    assert "visible_transcript.jsonl" in a_artifacts
    assert any(path.startswith("context_snapshot_") for path in a_artifacts)
    assert any(path.startswith("budget_trace_") for path in a_artifacts)
    assert any(path.startswith("answer_lineage_") for path in a_artifacts)

    c_manifest = manifests["C_HANDOFF_CONTINUITY"]
    assert c_manifest["roadmap_targets"] == ["R3.3"]
    assert c_manifest["metrics"]["scenario_focus"] == "portable_handoff_continuity"
    assert c_manifest["metrics"]["handoff_valid"] is True
    assert c_manifest["metrics"]["restore_valid"] is True
    assert c_manifest["metrics"]["evidence_coverage"] >= 0.8
    assert c_manifest["continuation_run"] is not None
    c_source_artifacts = {row["relative_path"] for row in c_manifest["source_run"]["artifact_inventory"]}
    c_target_artifacts = {row["relative_path"] for row in c_manifest["continuation_run"]["artifact_inventory"]}
    assert any(path.startswith("handoff_manifest_") for path in c_source_artifacts)
    assert any(path.startswith("handoff_validation_") for path in c_source_artifacts)
    assert any(path.startswith("reconstruction_diff_") for path in c_target_artifacts)
    assert "visible_transcript.jsonl" in c_target_artifacts
