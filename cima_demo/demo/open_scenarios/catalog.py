from __future__ import annotations

from .models import DatasetSource, OpenDatasetSpec


MAIN_EVALUATION_DATASET_IDS: tuple[str, ...] = (
    "hotpotqa",
    "qasper",
    "explainmeetsum",
    "qmsum",
)

EXCLUDED_DATASET_IDS: dict[str, str] = {
    "fever": (
        "FEVER is excluded from the main CIMA demonstrator evaluation because the "
        "current integration has only FEVER claim files and metadata references, "
        "not the FEVER wiki-pages corpus and claim-to-evidence retrieval required "
        "for FEVER-standard. Do not run it as claim-only or oracle evidence in the "
        "main traceability analysis."
    )
}


DATASET_SPECS: dict[str, OpenDatasetSpec] = {
    "qmsum": OpenDatasetSpec(
        dataset_id="qmsum",
        title="QMSum",
        task_family="meeting_query_summary",
        source=DatasetSource(kind="git_repo", ref="https://github.com/Yale-LILY/QMSum.git"),
        recommended_role="primary",
        supports_normalization=True,
        notes="Official GitHub repository; train/val/test data lives under data/ALL.",
    ),
    "explainmeetsum": OpenDatasetSpec(
        dataset_id="explainmeetsum",
        title="ExplainMeetSum",
        task_family="explainable_meeting_summary",
        source=DatasetSource(kind="git_repo", ref="https://github.com/hkim-etri/ExplainMeetSum.git"),
        recommended_role="primary",
        supports_normalization=True,
        notes=(
            "Official GitHub repository. Uses QMSum plus evidence annotations; if built data is not already present, "
            "the repository's conversion step may be required."
        ),
    ),
    "qasper": OpenDatasetSpec(
        dataset_id="qasper",
        title="QASPER",
        task_family="evidence_qa",
        source=DatasetSource(kind="hf_parquet",ref="allenai/qasper",config="qasper",),
        recommended_role="complement",
        supports_normalization=True,
        notes="Hugging Face dataset with title, abstract, full_text, and qas fields.",
    ),
    "fever": OpenDatasetSpec(
        dataset_id="fever",
        title="FEVER",
        task_family="claim_verification",
        source=DatasetSource(kind="direct_files", ref="https://fever.ai/download/fever"),
        recommended_role="excluded_pending_standard_integration",
        supports_normalization=False,
        notes=(
            "Excluded from the main CIMA demonstrator evaluation. FEVER-standard requires "
            "the FEVER wiki-pages corpus plus retrieval from claim to evidence sentences. "
            "The bundled direct claim files alone are insufficient for a meaningful CIMA run."
        ),
    ),
    "hotpotqa": OpenDatasetSpec(
        dataset_id="hotpotqa",
        title="HotpotQA",
        task_family="multi_hop_qa",
        source=DatasetSource(kind="hf_parquet", ref="hotpotqa/hotpot_qa", config="distractor"),
        recommended_role="optional",
        supports_normalization=True,
        notes="Hugging Face distractor split includes context paragraphs and supporting facts.",
    ),
}


def resolve_dataset_ids(dataset_ids: list[str] | None) -> list[str]:
    if not dataset_ids or dataset_ids == ["all"] or dataset_ids == ["main"]:
        return list(MAIN_EVALUATION_DATASET_IDS)

    requested = [dataset_id.strip() for dataset_id in dataset_ids if dataset_id and dataset_id.strip()]
    missing = [dataset_id for dataset_id in requested if dataset_id not in DATASET_SPECS]
    if missing:
        raise KeyError(f"Unknown dataset ids: {', '.join(sorted(missing))}")

    excluded = [dataset_id for dataset_id in requested if dataset_id in EXCLUDED_DATASET_IDS]
    if excluded:
        reasons = "; ".join(f"{dataset_id}: {EXCLUDED_DATASET_IDS[dataset_id]}" for dataset_id in excluded)
        raise ValueError(f"Excluded dataset requested: {reasons}")

    unsupported = [dataset_id for dataset_id in requested if not DATASET_SPECS[dataset_id].supports_normalization]
    if unsupported:
        raise ValueError(f"Unsupported dataset ids for main evaluation: {', '.join(sorted(unsupported))}")

    return requested


def get_dataset_spec(dataset_id: str) -> OpenDatasetSpec:
    try:
        return DATASET_SPECS[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset id: {dataset_id}") from exc
