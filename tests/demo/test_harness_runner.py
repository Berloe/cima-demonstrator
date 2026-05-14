from __future__ import annotations

import json
from pathlib import Path

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_harness_runs_frozen_scenarios_and_writes_reports(tmp_path: Path) -> None:
    runner = DemoScenarioRunner(artifacts_root=tmp_path, scenarios=load_all_scenarios())
    results = await runner.run_all()

    assert len(results) >= 3
    assert all(result.acceptance is not None for result in results)
    assert all(result.acceptance.passed for result in results)

    acceptance_report = json.loads((tmp_path / "acceptance_report.json").read_text(encoding="utf-8"))
    assert acceptance_report["all_passed"] is True
    assert acceptance_report["passed_count"] == len(results)
    assert (tmp_path / "demo_report.md").exists()

    for result in results:
        run_dir = result.artifacts_dir
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "run_phases.jsonl").exists()
        assert (run_dir / "visible_transcript.jsonl").exists()
        assert (run_dir / "lineage_edges.jsonl").exists()
        if result.context_snapshot is not None:
            context_id = result.context_snapshot["context_id"]
            assert (run_dir / f"context_snapshot_{context_id}.json").exists()
            assert (run_dir / f"context_pack_{context_id}.txt").exists()
            assert (run_dir / f"budget_trace_{context_id}.json").exists()
        assert any(path.name.startswith("answer_lineage_") for path in run_dir.iterdir())
        if result.scenario.expectations.requires_handoff:
            assert any(path.name.startswith("handoff_manifest_") for path in run_dir.iterdir())
            assert any(path.name.startswith("handoff_validation_") for path in run_dir.iterdir())
