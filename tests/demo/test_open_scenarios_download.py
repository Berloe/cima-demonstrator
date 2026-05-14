from __future__ import annotations

import json

import pytest

from cima_demo.demo.open_scenarios import download
from cima_demo.demo.open_scenarios.normalize import _resolve_split


class _FakeSplit:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDatasetDict(dict):
    def save_to_disk(self, path: str) -> None:
        from pathlib import Path

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "dataset_dict.json").write_text(json.dumps({"splits": sorted(self.keys())}), encoding="utf-8")


def test_hf_parquet_download_skips_missing_splits(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_load_dataset(_kind: str, *, data_files: str, split: str):
        assert _kind == "parquet"
        assert split == "train"
        calls.append(data_files)
        if "/test/" in data_files:
            raise ValueError('Instruction "test" corresponds to no data!')
        if "/validation/" in data_files:
            return _FakeSplit("validation")
        if "/train/" in data_files:
            return _FakeSplit("train")
        raise AssertionError(data_files)

    monkeypatch.setattr(download, "_require_hf_datasets", lambda: (fake_load_dataset, _FakeDatasetDict))

    metadata = download._download_hf_parquet_dataset("hotpotqa", tmp_path / "hotpotqa" / "hf", force=False)

    assert metadata["splits"] == ["train", "validation"]
    assert metadata["skipped_splits"] == ["test"]
    assert (tmp_path / "hotpotqa" / "hf" / "dataset_dict.json").exists()
    assert len(calls) == 3


def test_hf_parquet_download_skips_empty_wildcard_splits(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_load_dataset(_kind: str, *, data_files: str, split: str):
        assert _kind == "parquet"
        assert split == "train"
        calls.append(data_files)
        if "/test/" in data_files:
            raise ValueError(
                "At least one valid data file must be specified, "
                "all the data_files are invalid: {'train': []}"
            )
        if "/validation/" in data_files:
            return _FakeSplit("validation")
        if "/train/" in data_files:
            return _FakeSplit("train")
        raise AssertionError(data_files)

    monkeypatch.setattr(download, "_require_hf_datasets", lambda: (fake_load_dataset, _FakeDatasetDict))

    metadata = download._download_hf_parquet_dataset("hotpotqa", tmp_path / "hotpotqa" / "hf", force=False)

    assert metadata["splits"] == ["train", "validation"]
    assert metadata["skipped_splits"] == ["test"]
    assert (tmp_path / "hotpotqa" / "hf" / "dataset_dict.json").exists()
    assert len(calls) == 3


def test_resolve_split_falls_back_from_test_to_validation():
    dataset = {"train": [1], "validation": [2]}
    split_data, source_split = _resolve_split(dataset, "test", dataset_id="hotpotqa")
    assert split_data == [2]
    assert source_split == "validation"


def test_fever_download_is_rejected_until_standard_integration_exists(tmp_path):
    with pytest.raises(ValueError, match="Dataset 'fever' is excluded"):
        download.download_dataset('fever', root=tmp_path, force=False)
