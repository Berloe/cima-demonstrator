"""Integrated acceptance-package writers for the demonstrator evidence book."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .report_builder import ScenarioExecutionResult


PACKAGE_FILE_NAMES = {
    "acceptance_package_index.json",
    "acceptance_package.md",
    "acceptance_package_bundle.zip",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory_paths(root: Path) -> list[Path]:
    excluded = PACKAGE_FILE_NAMES
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )


@dataclass(slots=True)
class AcceptancePackageIndex:
    schema_version: str
    generated_at: str
    scenario_count: int
    all_passed: bool
    roadmap_coverage: dict[str, list[str]]
    package_inventory: list[dict[str, Any]]
    bundle: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_package_inventory(*, root: Path, manifest_paths: list[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(manifest_paths):
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return inventory


def _roadmap_coverage_from_manifests(*, root: Path, manifest_paths: list[Path]) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {}
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = str(payload.get("scenario_id") or "")
        for target in payload.get("roadmap_targets") or []:
            coverage.setdefault(str(target), []).append(scenario_id)
    for scenario_ids in coverage.values():
        scenario_ids.sort()
    return coverage


def write_acceptance_package(
    *,
    root: Path,
    results: list[ScenarioExecutionResult],
    scenario_manifest_paths: list[Path],
    acceptance_report_path: Path,
    demo_report_path: Path,
    evidence_book_index_path: Path,
    evidence_book_md_path: Path,
    conformance_matrix_json_path: Path,
    conformance_matrix_md_path: Path,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)

    bundle_path = root / "acceptance_package_bundle.zip"
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in _inventory_paths(root):
            zf.write(path, arcname=path.relative_to(root).as_posix())

    package_inventory_paths = [
        acceptance_report_path,
        demo_report_path,
        evidence_book_index_path,
        evidence_book_md_path,
        conformance_matrix_json_path,
        conformance_matrix_md_path,
        *scenario_manifest_paths,
    ]
    package_inventory = _build_package_inventory(root=root, manifest_paths=package_inventory_paths)
    roadmap_coverage = _roadmap_coverage_from_manifests(root=root, manifest_paths=scenario_manifest_paths)

    index = AcceptancePackageIndex(
        schema_version="cima_demo.acceptance_package_index.v1",
        generated_at=_utc_now(),
        scenario_count=len(results),
        all_passed=all(result.acceptance is not None and result.acceptance.passed for result in results),
        roadmap_coverage=roadmap_coverage,
        package_inventory=package_inventory,
        bundle={
            "relative_path": bundle_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(bundle_path),
            "bytes": bundle_path.stat().st_size,
        },
    )
    index_path = root / "acceptance_package_index.json"
    index_path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CIMA Demonstrator — Acceptance Package",
        "",
        f"Generated at: {index.generated_at}",
        f"Scenarios: {index.scenario_count}",
        f"All passed: `{index.all_passed}`",
        f"Bundle: `{index.bundle['relative_path']}`",
        f"Bundle SHA256: `{index.bundle['sha256']}`",
        "",
        "## Roadmap coverage",
        "",
    ]
    for target in sorted(index.roadmap_coverage):
        joined = ", ".join(index.roadmap_coverage[target])
        lines.append(f"- **{target}**: {joined}")
    lines.extend([
        "",
        "## Package inventory",
        "",
    ])
    for entry in index.package_inventory:
        lines.append(
            f"- `{entry['relative_path']}` — {entry['bytes']} bytes — `{entry['sha256']}`"
        )
    md_path = root / "acceptance_package.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"index": index_path, "markdown": md_path, "bundle": bundle_path}
