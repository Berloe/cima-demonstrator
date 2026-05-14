from __future__ import annotations

import pytest

from cima_demo.demo.open_scenarios.catalog import DATASET_SPECS, EXCLUDED_DATASET_IDS, MAIN_EVALUATION_DATASET_IDS, resolve_dataset_ids


def test_open_scenario_catalog_contains_expected_ids() -> None:
    assert {"qmsum", "explainmeetsum", "qasper", "fever", "hotpotqa"}.issubset(DATASET_SPECS)
    assert "fever" in EXCLUDED_DATASET_IDS
    assert not DATASET_SPECS["fever"].supports_normalization


def test_resolve_dataset_ids_all_expands_main_evaluation_only() -> None:
    resolved = resolve_dataset_ids(["all"])
    assert tuple(resolved) == MAIN_EVALUATION_DATASET_IDS
    assert "fever" not in resolved


def test_resolve_dataset_ids_rejects_fever_until_standard_integration_exists() -> None:
    with pytest.raises(ValueError, match="Excluded dataset requested: fever"):
        resolve_dataset_ids(["fever"])
