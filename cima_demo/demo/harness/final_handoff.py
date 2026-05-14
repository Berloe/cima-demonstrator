"""Final handoff writers for release / audit continuation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class FinalHandoff:
    schema_version: str
    generated_at: str
    release_candidate: dict[str, Any]
    roadmap_status: dict[str, str]
    all_scenarios_passed: bool
    known_limits: list[dict[str, Any]]
    verification_entrypoints: list[dict[str, str]]
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_final_handoff(
    *,
    root: Path,
    release_candidate_index_path: Path,
    acceptance_package_index_path: Path,
    conformance_matrix_json_path: Path,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    release_index = _load_json(release_candidate_index_path)
    acceptance_index = _load_json(acceptance_package_index_path)
    conformance_matrix = _load_json(conformance_matrix_json_path)

    roadmap_status = {
        "R0": "closed",
        "R1": "closed",
        "R2": "closed",
        "R3": "closed",
        "R4": "closed",
        "R5": "closed",
    }

    handoff = FinalHandoff(
        schema_version="cima_demo.final_handoff.v1",
        generated_at=_utc_now(),
        release_candidate={
            "index": release_candidate_index_path.relative_to(root).as_posix(),
            "bundle": release_index.get("bundle", {}).get("relative_path"),
            "repo_snapshot": release_index.get("repo_snapshot", {}).get("relative_path"),
        },
        roadmap_status=roadmap_status,
        all_scenarios_passed=bool(acceptance_index.get("all_passed") is True),
        known_limits=release_index.get("known_limits") or [],
        verification_entrypoints=[
            {
                "label": "Acceptance package",
                "path": acceptance_package_index_path.relative_to(root).as_posix(),
            },
            {
                "label": "Conformance matrix",
                "path": conformance_matrix_json_path.relative_to(root).as_posix(),
            },
            {
                "label": "Release candidate",
                "path": release_candidate_index_path.relative_to(root).as_posix(),
            },
        ],
        next_step="Release freeze / external audit using the acceptance package, conformance matrix, and release bundle as the canonical entrypoints.",
    )

    json_path = root / "final_handoff.json"
    json_path.write_text(json.dumps(handoff.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CIMA Demonstrator — Final Handoff",
        "",
        f"Generated at: {handoff.generated_at}",
        f"All scenarios passed: `{handoff.all_scenarios_passed}`",
        "",
        "## Release artifacts",
        "",
        f"- Release candidate index: `{handoff.release_candidate['index']}`",
        f"- Release candidate bundle: `{handoff.release_candidate['bundle']}`",
        f"- Repo snapshot: `{handoff.release_candidate['repo_snapshot']}`",
        "",
        "## Roadmap status",
        "",
    ]
    for item_id, status in handoff.roadmap_status.items():
        lines.append(f"- **{item_id}**: `{status}`")
    lines.extend([
        "",
        "## Known explicit limits",
        "",
    ])
    if handoff.known_limits:
        for item in handoff.known_limits:
            lines.append(
                f"- **{item['item_id']} — {item['item_label']}**: implemented `{item['implemented_status']}`, demonstrated `{item['demonstrated_status']}` — {item['note']}"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Verification entrypoints",
        "",
    ])
    for entry in handoff.verification_entrypoints:
        lines.append(f"- **{entry['label']}**: `{entry['path']}`")
    lines.extend([
        "",
        "## Next step",
        "",
        handoff.next_step,
        "",
    ])
    md_path = root / "final_handoff.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
