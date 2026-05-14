from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from .catalog import DATASET_SPECS, get_dataset_spec, resolve_dataset_ids
from .models import DownloadArtifact


PARQUET_SPLITS = ("train", "validation", "test")

FEVER_V1_FILES = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "labelled_dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "unlabelled_dev": "https://fever.ai/download/fever/shared_task_dev_public.jsonl",
    "unlabelled_test": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "paper_dev": "https://fever.ai/download/fever/paper_dev.jsonl",
    "paper_test": "https://fever.ai/download/fever/paper_test.jsonl",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_saved_hf_dataset(path: Path) -> bool:
    """Return True only for a complete datasets.save_to_disk target."""
    return (path / "dataset_dict.json").exists() or (path / "state.json").exists()


def _remove_tree(path: Path) -> None:
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], check=True)


def _require_hf_datasets() -> Any:
    try:
        from datasets import DatasetDict, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when dependency missing
        raise RuntimeError(
            "The optional dependency 'datasets' is required for Hugging Face-backed open scenarios. "
            "Install it with: pip install -r requirements-open-scenarios.txt"
        ) from exc
    return load_dataset, DatasetDict



def _is_saved_direct_dataset(path: Path, expected_files: dict[str, str]) -> bool:
    manifest = path / "cima_download_metadata.json"
    if not manifest.exists():
        return False
    return all((path / f"{name}.jsonl").exists() for name in expected_files)


def _download_url(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:  # noqa: S310 - controlled dataset URLs
            shutil.copyfileobj(response, handle)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _download_direct_files_dataset(
    dataset_id: str,
    target: Path,
    *,
    files: dict[str, str],
    force: bool,
    note: str | None = None,
) -> dict[str, Any]:
    if target.exists() and not force and _is_saved_direct_dataset(target, files):
        metadata = {"path": str(target), "format": "direct_files", "files": sorted(files)}
        manifest = target / "cima_download_metadata.json"
        try:
            metadata.update(json.loads(manifest.read_text(encoding="utf-8")))
        except Exception:
            pass
        return metadata
    if target.exists():
        _remove_tree(target)
    _ensure_dir(target)

    downloaded: dict[str, dict[str, Any]] = {}
    for name, url in files.items():
        destination = target / f"{name}.jsonl"
        _download_url(url, destination)
        downloaded[name] = {"url": url, "path": str(destination), "bytes": destination.stat().st_size}

    metadata = {
        "path": str(target),
        "dataset_id": dataset_id,
        "format": "direct_files",
        "files": downloaded,
        "note": note,
    }
    (target / "cima_download_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata

def _git_clone(url: str, target: Path, *, force: bool) -> dict[str, Any]:
    if target.exists():
        if not force:
            commit = subprocess.run(
                ["git", "-C", str(target), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            return {"path": str(target), "commit": commit.stdout.strip()}
        _remove_tree(target)
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    commit = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"path": str(target), "commit": commit.stdout.strip()}


def _download_hf_dataset(dataset_id: str, target: Path, *, force: bool) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_id)
    if target.exists() and not force and _is_saved_hf_dataset(target):
        return {"path": str(target), "source": spec.source.ref, "config": spec.source.config}
    if target.exists():
        _remove_tree(target)
    load_dataset, _dataset_dict_cls = _require_hf_datasets()
    ds = load_dataset(spec.source.ref, spec.source.config)
    ds.save_to_disk(str(target))
    return {"path": str(target), "source": spec.source.ref, "config": spec.source.config}


def _split_is_missing(exc: Exception) -> bool:
    message = str(exc).lower()
    missing_markers = (
        "corresponds to no data",
        "no data files",
        "at least one valid data file must be specified",
        "all the data_files are invalid",
        "unable to find",
        "couldn't find",
        "does not exist",
        "not found",
        "empty data_files",
    )
    return any(marker in message for marker in missing_markers)


def _load_one_parquet_split(load_dataset: Any, pattern: str) -> Any:
    # A single data_files string is exposed by datasets as split="train".
    # We assign that Dataset to the real split name in the DatasetDict outside.
    return load_dataset("parquet", data_files=pattern, split="train")


def _download_hf_parquet_dataset(dataset_id: str, target: Path, *, force: bool) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_id)
    if target.exists() and not force and _is_saved_hf_dataset(target):
        manifest = target / "cima_download_metadata.json"
        metadata = {"path": str(target), "source": spec.source.ref, "config": spec.source.config, "format": "parquet"}
        if manifest.exists():
            try:
                metadata.update(json.loads(manifest.read_text(encoding="utf-8")))
            except Exception:
                pass
        return metadata

    if target.exists():
        _remove_tree(target)

    split_patterns = {
        split: f"hf://datasets/{spec.source.ref}@refs/convert/parquet/{spec.source.config}/{split}/*.parquet"
        for split in PARQUET_SPLITS
    }

    load_dataset, dataset_dict_cls = _require_hf_datasets()
    loaded_splits: dict[str, Any] = {}
    skipped_splits: dict[str, str] = {}
    fatal_errors: dict[str, str] = {}

    for split, pattern in split_patterns.items():
        try:
            loaded_splits[split] = _load_one_parquet_split(load_dataset, pattern)
        except Exception as exc:  # pragma: no cover - exact exception type varies across datasets versions
            if _split_is_missing(exc):
                skipped_splits[split] = str(exc)
                continue
            fatal_errors[split] = str(exc)

    if not loaded_splits:
        details = {"skipped_splits": skipped_splits, "fatal_errors": fatal_errors}
        raise RuntimeError(
            f"No parquet splits could be downloaded for dataset '{dataset_id}' ({spec.source.ref}/{spec.source.config}). "
            f"Details: {json.dumps(details, ensure_ascii=False)}"
        )

    # If at least one split loaded, treat non-missing errors as hard failures. This prevents silently
    # masking schema/network/auth errors while still tolerating legitimately absent splits such as test.
    if fatal_errors:
        raise RuntimeError(
            f"Some parquet splits failed for dataset '{dataset_id}' ({spec.source.ref}/{spec.source.config}). "
            f"Loaded splits: {sorted(loaded_splits)}. Fatal errors: {json.dumps(fatal_errors, ensure_ascii=False)}"
        )

    ds = dataset_dict_cls(loaded_splits)
    ds.save_to_disk(str(target))
    metadata = {
        "path": str(target),
        "source": spec.source.ref,
        "config": spec.source.config,
        "format": "parquet",
        "ref": "refs/convert/parquet",
        "splits": sorted(loaded_splits),
        "skipped_splits": sorted(skipped_splits),
    }
    (target / "cima_download_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def download_dataset(dataset_id: str, *, root: Path, force: bool = False) -> DownloadArtifact:
    spec = get_dataset_spec(dataset_id)
    if not spec.supports_normalization:
        raise ValueError(
            f"Dataset '{dataset_id}' is excluded from the main CIMA demonstrator evaluation. {spec.notes}"
        )
    dataset_root = root / dataset_id
    _ensure_dir(dataset_root.parent)
    if spec.source.kind == "git_repo":
        metadata = _git_clone(spec.source.ref, dataset_root / "repo", force=force)
        return DownloadArtifact(dataset_id=dataset_id, path=dataset_root / "repo", kind="git_repo", metadata=metadata)
    if dataset_id == "fever":
        metadata = _download_direct_files_dataset(
            dataset_id,
            dataset_root / "raw",
            files=FEVER_V1_FILES,
            force=force,
            note=(
                "Downloaded FEVER v1.0 claim files directly from fever.ai because the Hugging Face "
                "dataset script path is not supported by modern datasets versions. Normalization emits "
                "claim-only executable cases and preserves unresolved Wikipedia evidence references as metadata."
            ),
        )
        return DownloadArtifact(dataset_id=dataset_id, path=dataset_root / "raw", kind="direct_files", metadata=metadata)
    if spec.source.kind == "hf_dataset":
        metadata = _download_hf_dataset(dataset_id, dataset_root / "hf", force=force)
        return DownloadArtifact(dataset_id=dataset_id, path=dataset_root / "hf", kind="hf_dataset", metadata=metadata)
    if spec.source.kind == "hf_parquet":
        metadata = _download_hf_parquet_dataset(dataset_id, dataset_root / "hf", force=force)
        return DownloadArtifact(dataset_id=dataset_id, path=dataset_root / "hf", kind="hf_parquet", metadata=metadata)
    raise ValueError(f"Unsupported source kind: {spec.source.kind}")


def write_download_manifest(root: Path, artifacts: list[DownloadArtifact]) -> Path:
    payload = {
        "datasets": [
            {
                "dataset_id": artifact.dataset_id,
                "kind": artifact.kind,
                "path": str(artifact.path),
                "metadata": artifact.metadata,
            }
            for artifact in artifacts
        ]
    }
    path = root / "download_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download open publication scenario datasets.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory for raw downloads.")
    parser.add_argument(
        "--datasets",
        default="all",
        help=f"Comma-separated dataset ids. Available: {', '.join(DATASET_SPECS)}",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if target exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_ids = resolve_dataset_ids([item.strip() for item in args.datasets.split(",") if item.strip()])
    _ensure_dir(args.root)
    artifacts = [download_dataset(dataset_id, root=args.root, force=args.force) for dataset_id in dataset_ids]
    manifest = write_download_manifest(args.root, artifacts)
    print(json.dumps({"downloaded": [a.dataset_id for a in artifacts], "manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
