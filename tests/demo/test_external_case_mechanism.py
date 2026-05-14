from __future__ import annotations

import asyncio
import json
from pathlib import Path

from cima_demo.demo.open_scenarios.external_casegen import generate_external_cases
from cima_demo.demo.open_scenarios.external_sources import materialize_external_sources


async def test_external_sources_materialize_text_and_file(tmp_path: Path) -> None:
    source_file = tmp_path / "policy.md"
    source_file.write_text(
        "# Policy\n\nCIMA is a memory architecture.\n\nDeployments must preserve traceability.",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "inline-1",
                        "source_type": "text",
                        "title": "Inline",
                        "text": "TaskMemory is a stability axis.",
                    },
                    {
                        "source_id": "file-1",
                        "source_type": "file",
                        "path": str(source_file),
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    outputs = await materialize_external_sources(manifest_path=manifest, out_root=out, timeout_seconds=1.0)
    docs = [json.loads(line) for line in outputs["documents"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(docs) == 2
    assert {doc["doc_id"] for doc in docs} == {"inline-1", "file-1"}
    assert all(doc["metadata"]["sha256"] for doc in docs)


def test_external_casegen_emits_auto_ready_and_review(tmp_path: Path) -> None:
    docs_path = tmp_path / "external_documents.jsonl"
    documents = [
        {
            "doc_id": "doc-1",
            "title": "Spec",
            "text": (
                "# Definitions\n\nTaskMemory is the stability axis of the task.\n\n"
                "# Requirements\n\nDeployments must preserve traceability across summaries.\n\n"
                "Caching must be enabled. Caching must not be enabled for deleted conversations."
            ),
            "metadata": {"sha256": "abc", "source_type": "text"},
        }
    ]
    docs_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in documents), encoding="utf-8")
    outputs = generate_external_cases(docs_path=docs_path, out_root=tmp_path / "cases", dataset_id="ext")
    cases = [json.loads(line) for line in outputs["cases"].read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [json.loads(line) for line in outputs["candidates"].read_text(encoding="utf-8").splitlines() if line.strip()]
    families = {case["task_family"] for case in cases}
    assert "external_definition_grounding" in families
    assert "external_requirement_grounding" in families
    assert any(candidate["case_family"] == "external_conflict_detection" and candidate["status"] == "needs_review" for candidate in candidates)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["auto_ready_case_count"] == len(cases)


def test_external_sources_cli_and_casegen_are_stable(tmp_path: Path) -> None:
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {"sources": [{"source_id": "t1", "source_type": "text", "text": "CIMA is a memory architecture."}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    docs_out = tmp_path / "docs"
    asyncio.run(materialize_external_sources(manifest_path=source_manifest, out_root=docs_out, timeout_seconds=1.0))
    outputs = generate_external_cases(
        docs_path=docs_out / "external_documents.jsonl",
        out_root=tmp_path / "cases",
        dataset_id="ext_small",
    )
    assert outputs["candidates"].exists()
    assert outputs["cases"].exists()
