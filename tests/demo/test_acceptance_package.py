from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_acceptance_package_writes_integrated_bundle_and_index(tmp_path: Path) -> None:
    runner = DemoScenarioRunner(artifacts_root=tmp_path, scenarios=load_all_scenarios())

    results = await runner.run_all()

    assert len(results) >= 6
    index_path = tmp_path / "acceptance_package_index.json"
    md_path = tmp_path / "acceptance_package.md"
    bundle_path = tmp_path / "acceptance_package_bundle.zip"
    assert index_path.exists()
    assert md_path.exists()
    assert bundle_path.exists()

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cima_demo.acceptance_package_index.v1"
    assert payload["scenario_count"] == len(results)
    assert payload["all_passed"] is True
    assert payload["bundle"]["relative_path"] == "acceptance_package_bundle.zip"
    assert payload["bundle"]["sha256"]
    assert payload["bundle"]["bytes"] > 0
    assert payload["roadmap_coverage"]["R3.1"] == ["A_LONG_CONTEXT_VIRTUAL"]
    assert payload["roadmap_coverage"]["R3.6"] == ["F_NEGATIVE_CONFLICT"]

    inventory_paths = {entry["relative_path"] for entry in payload["package_inventory"]}
    assert "acceptance_report.json" in inventory_paths
    assert "demo_report.md" in inventory_paths
    assert "evidence_book_index.json" in inventory_paths
    assert "evidence_book.md" in inventory_paths
    assert "conformance_matrix.json" in inventory_paths
    assert "conformance_matrix.md" in inventory_paths
    assert any(path.endswith("scenario_evidence_manifest.json") for path in inventory_paths)

    with ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
    assert "acceptance_report.json" in names
    assert "demo_report.md" in names
    assert "evidence_book_index.json" in names
    assert "evidence_book.md" in names
    assert "conformance_matrix.json" in names
    assert "conformance_matrix.md" in names
    assert any(name.endswith("scenario_evidence_manifest.json") for name in names)
    assert all(not name.endswith("acceptance_package_bundle.zip") for name in names)
