from __future__ import annotations

import json
from pathlib import Path

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_conformance_matrix_distinguishes_implemented_from_demonstrated(tmp_path: Path) -> None:
    runner = DemoScenarioRunner(artifacts_root=tmp_path, scenarios=load_all_scenarios())

    results = await runner.run_all()

    assert len(results) >= 6
    json_path = tmp_path / "conformance_matrix.json"
    md_path = tmp_path / "conformance_matrix.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cima_demo.conformance_matrix.v1"
    rows = {row["item_id"]: row for row in payload["rows"]}

    assert rows["C-07"]["implemented_status"] == "yes"
    assert rows["C-07"]["demonstrated_status"] == "yes"
    assert rows["P-02"]["implemented_status"] == "yes"
    assert rows["P-02"]["demonstrated_status"] == "yes"
    assert rows["O-09"]["implemented_status"] == "yes"
    assert rows["O-09"]["demonstrated_status"] == "partial"
    assert rows["P-03"]["demonstrated_status"] == "partial"
    assert rows["G-4"]["implemented_status"] == "yes"
    assert rows["G-4"]["demonstrated_status"] == "yes"
    assert rows["G-5"]["implemented_status"] == "yes"
    assert rows["G-5"]["demonstrated_status"] == "yes"
    assert rows["C-10"]["implemented_status"] == "no"
    assert rows["C-10"]["demonstrated_status"] == "no"

    assert payload["summary"]["implemented_yes"] >= 10
    assert payload["summary"]["demonstrated_yes"] >= 7
    assert payload["summary"]["demonstrated_partial"] >= 1

    markdown = md_path.read_text(encoding="utf-8")
    assert "Implemented: `yes`" in markdown
    assert "Demonstrated: `partial`" in markdown
    assert "Authority domain" in markdown
