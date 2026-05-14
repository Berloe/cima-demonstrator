from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from cima_demo.demo.harness import DemoScenarioRunner, load_all_scenarios


@pytest.mark.asyncio
async def test_release_candidate_writes_repo_snapshot_docs_and_bundle(tmp_path: Path) -> None:
    runner = DemoScenarioRunner(artifacts_root=tmp_path, scenarios=load_all_scenarios())

    results = await runner.run_all()

    assert len(results) >= 6
    index_path = tmp_path / "release_candidate_index.json"
    md_path = tmp_path / "release_candidate.md"
    bundle_path = tmp_path / "release_candidate_bundle.zip"
    repo_snapshot_path = tmp_path / "release_repo_snapshot.zip"
    assert index_path.exists()
    assert md_path.exists()
    assert bundle_path.exists()
    assert repo_snapshot_path.exists()

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cima_demo.release_candidate_index.v1"
    assert payload["all_passed"] is True
    assert payload["scenario_count"] == len(results)
    assert payload["repo_snapshot"]["relative_path"] == "release_repo_snapshot.zip"
    assert payload["repo_snapshot"]["file_count"] > 0
    assert payload["evidence_bundle"]["relative_path"] == "acceptance_package_bundle.zip"
    assert payload["bundle"]["relative_path"] == "release_candidate_bundle.zip"
    assert payload["conformance_summary"]["implemented_yes"] >= 1
    assert any(item["item_id"] == "C-10" for item in payload["known_limits"])
    assert any(entry["relative_path"].startswith("doc/") for entry in payload["documentation_inventory"])

    with ZipFile(repo_snapshot_path) as zf:
        names = set(zf.namelist())
    assert "README.md" in names
    assert "pyproject.toml" in names
    assert any(name.startswith("cima_demo/") for name in names)

    with ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
    assert "acceptance_package_bundle.zip" in names
    assert "release_repo_snapshot.zip" in names
    assert any(name.startswith("doc/") for name in names)
    assert all(not name.endswith("release_candidate_bundle.zip") for name in names)


def test_repo_snapshot_excludes_artifacts_root_inside_repo(tmp_path: Path) -> None:
    from cima_demo.demo.harness.release_candidate import _write_repo_snapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("demo", encoding="utf-8")
    (repo / "cima_demo").mkdir()
    (repo / "cima_demo" / "__init__.py").write_text("", encoding="utf-8")

    artifacts = repo / "artifacts"
    artifacts.mkdir()
    (artifacts / "acceptance_package_bundle.zip").write_bytes(b"not source")
    (artifacts / "previous_run.txt").write_text("generated", encoding="utf-8")

    snapshot_path, file_count = _write_repo_snapshot(root=artifacts, repo_root=repo)

    assert snapshot_path.exists()
    assert file_count == 2
    with ZipFile(snapshot_path) as zf:
        names = set(zf.namelist())
    assert names == {"README.md", "cima_demo/__init__.py"}
    assert all(not name.startswith("artifacts/") for name in names)
