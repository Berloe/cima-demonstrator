"""Release-candidate packaging for the demonstrator repo + docs + evidence bundle."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


GENERATED_FILE_NAMES = {
    "acceptance_package_index.json",
    "acceptance_package.md",
    "acceptance_package_bundle.zip",
    "release_candidate_index.json",
    "release_candidate.md",
    "release_candidate_bundle.zip",
    "release_repo_snapshot.zip",
}

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Python 3.8-compatible Path.is_relative_to()."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _iter_repo_files(repo_root: Path, *, exclude_roots: list[Path] | None = None) -> list[Path]:
    """Return source files for the repo snapshot without generated artifacts.

    The acceptance/release artifacts can be written inside the repository during
    local runs.  If the snapshot writer includes its own output directory,
    ``release_repo_snapshot.zip`` may be discovered while it is still being
    written and the zip grows without bound.  Excluding the artifact root is
    therefore a correctness guard, not just a cosmetic filter.
    """
    excludes = [root.resolve() for root in (exclude_roots or []) if root.exists()]
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(_is_relative_to(path, excluded) for excluded in excludes):
            continue
        if path.name in GENERATED_FILE_NAMES:
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def _iter_doc_files(repo_root: Path) -> list[Path]:
    doc_root = repo_root / "doc"
    if not doc_root.exists():
        return []
    files: list[Path] = []
    for path in doc_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


@dataclass(slots=True)
class ReleaseCandidateIndex:
    schema_version: str
    generated_at: str
    all_passed: bool
    scenario_count: int
    repo_snapshot: dict[str, Any]
    evidence_bundle: dict[str, Any]
    documentation_inventory: list[dict[str, Any]]
    conformance_summary: dict[str, Any]
    known_limits: list[dict[str, Any]]
    bundle: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_file_entry(base: Path, path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(base).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_repo_snapshot(*, root: Path, repo_root: Path) -> tuple[Path, int]:
    snapshot_path = root / "release_repo_snapshot.zip"
    file_count = 0
    with ZipFile(snapshot_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in _iter_repo_files(repo_root, exclude_roots=[root]):
            file_count += 1
            zf.write(path, arcname=path.relative_to(repo_root).as_posix())
    return snapshot_path, file_count


def _load_acceptance_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_conformance_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _known_limits_from_conformance(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    limits: list[dict[str, Any]] = []
    for row in matrix.get("rows") or []:
        if row.get("implemented_status") == "yes" and row.get("demonstrated_status") == "yes":
            continue
        limits.append(
            {
                "item_id": row.get("item_id"),
                "item_label": row.get("item_label"),
                "implemented_status": row.get("implemented_status"),
                "demonstrated_status": row.get("demonstrated_status"),
                "note": row.get("note") or "",
            }
        )
    return limits


def write_release_candidate(
    *,
    root: Path,
    repo_root: Path,
    acceptance_report_path: Path,
    acceptance_package_bundle_path: Path,
    conformance_matrix_json_path: Path,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    repo_snapshot_path, repo_file_count = _write_repo_snapshot(root=root, repo_root=repo_root)

    bundle_path = root / "release_candidate_bundle.zip"
    doc_files = _iter_doc_files(repo_root)
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(acceptance_package_bundle_path, arcname=acceptance_package_bundle_path.name)
        zf.write(repo_snapshot_path, arcname=repo_snapshot_path.name)
        for path in doc_files:
            zf.write(path, arcname=Path("doc") / path.relative_to(repo_root / "doc"))

    acceptance_report = _load_acceptance_report(acceptance_report_path)
    conformance_matrix = _load_conformance_matrix(conformance_matrix_json_path)
    documentation_inventory = [_build_file_entry(repo_root, path) for path in doc_files]

    index = ReleaseCandidateIndex(
        schema_version="cima_demo.release_candidate_index.v1",
        generated_at=_utc_now(),
        all_passed=bool(acceptance_report.get("all_passed") is True),
        scenario_count=int(acceptance_report.get("scenario_count") or 0),
        repo_snapshot={
            "relative_path": repo_snapshot_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(repo_snapshot_path),
            "bytes": repo_snapshot_path.stat().st_size,
            "file_count": repo_file_count,
        },
        evidence_bundle={
            "relative_path": acceptance_package_bundle_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(acceptance_package_bundle_path),
            "bytes": acceptance_package_bundle_path.stat().st_size,
        },
        documentation_inventory=documentation_inventory,
        conformance_summary=conformance_matrix.get("summary") or {},
        known_limits=_known_limits_from_conformance(conformance_matrix),
        bundle={
            "relative_path": bundle_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(bundle_path),
            "bytes": bundle_path.stat().st_size,
        },
    )

    index_path = root / "release_candidate_index.json"
    index_path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CIMA Demonstrator — Release Candidate",
        "",
        f"Generated at: {index.generated_at}",
        f"All scenarios passed: `{index.all_passed}`",
        f"Scenario count: `{index.scenario_count}`",
        f"Repo snapshot: `{index.repo_snapshot['relative_path']}` ({index.repo_snapshot['file_count']} files)",
        f"Repo snapshot SHA256: `{index.repo_snapshot['sha256']}`",
        f"Evidence bundle: `{index.evidence_bundle['relative_path']}`",
        f"Evidence bundle SHA256: `{index.evidence_bundle['sha256']}`",
        f"Release bundle: `{index.bundle['relative_path']}`",
        f"Release bundle SHA256: `{index.bundle['sha256']}`",
        "",
        "## Conformance summary",
        "",
        f"- Implemented yes/partial/no: `{index.conformance_summary.get('implemented_yes', 0)}` / `{index.conformance_summary.get('implemented_partial', 0)}` / `{index.conformance_summary.get('implemented_no', 0)}`",
        f"- Demonstrated yes/partial/no: `{index.conformance_summary.get('demonstrated_yes', 0)}` / `{index.conformance_summary.get('demonstrated_partial', 0)}` / `{index.conformance_summary.get('demonstrated_no', 0)}`",
        "",
        "## Known explicit limits",
        "",
    ]
    if index.known_limits:
        for item in index.known_limits:
            lines.append(
                f"- **{item['item_id']} — {item['item_label']}**: implemented `{item['implemented_status']}`, demonstrated `{item['demonstrated_status']}` — {item['note']}"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Documentation inventory",
        "",
    ])
    for entry in documentation_inventory:
        lines.append(
            f"- `{entry['relative_path']}` — {entry['bytes']} bytes — `{entry['sha256']}`"
        )
    md_path = root / "release_candidate.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "index": index_path,
        "markdown": md_path,
        "bundle": bundle_path,
        "repo_snapshot": repo_snapshot_path,
    }
