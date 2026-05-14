from __future__ import annotations

import json
from pathlib import Path

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_final_handoff_writes_closure_summary_for_release_and_audit(tmp_path: Path) -> None:
    runner = DemoScenarioRunner(artifacts_root=tmp_path, scenarios=load_all_scenarios())

    await runner.run_all()

    json_path = tmp_path / "final_handoff.json"
    md_path = tmp_path / "final_handoff.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cima_demo.final_handoff.v1"
    assert payload["all_scenarios_passed"] is True
    assert payload["roadmap_status"] == {
        "R0": "closed",
        "R1": "closed",
        "R2": "closed",
        "R3": "closed",
        "R4": "closed",
        "R5": "closed",
    }
    assert payload["release_candidate"]["index"] == "release_candidate_index.json"
    assert payload["release_candidate"]["bundle"] == "release_candidate_bundle.zip"
    assert payload["release_candidate"]["repo_snapshot"] == "release_repo_snapshot.zip"
    labels = {entry["label"] for entry in payload["verification_entrypoints"]}
    assert labels == {"Acceptance package", "Conformance matrix", "Release candidate"}
    assert any(item["item_id"] == "C-10" for item in payload["known_limits"])
