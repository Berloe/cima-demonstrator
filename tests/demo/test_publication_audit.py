from __future__ import annotations

import json
from pathlib import Path

from cima_demo.demo.publication.audit import analyze_runs, write_outputs


def test_publication_audit_builds_claim_matrix(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"case_id": "case-1", "dataset_id": "demo"}), encoding="utf-8")
    (run_dir / "context.json").write_text(json.dumps({
        "markers": ["S1"],
        "resolved_source_count": 1,
        "resolved_span_count": 1,
        "unresolved_ref_ids": [],
        "resolution_mode": "witness_first",
        "token_usage": {"context": 100, "available_for_content": 120},
    }), encoding="utf-8")
    (run_dir / "citation_contract.json").write_text(json.dumps({
        "passed": True,
        "published_integrity_passed": True,
        "publication_status": "publishable",
        "publishable": True,
        "blocked_by_cima": False,
        "invalid_published_as_valid": False,
        "publication_gate": {"publication_status": "publishable", "publishable": True, "blocked_by_cima": False},
    }), encoding="utf-8")
    (run_dir / "cleanup.json").write_text(json.dumps({
        "ok": True,
        "final_verified": True,
        "final_audit": {
            "status": "ok",
            "consistency": {"cleanup_ok": True, "conversation_deleted": True, "qdrant_zeroed": True},
        },
    }), encoding="utf-8")
    (run_dir / "zoom.json").write_text(json.dumps({
        "resolution_mode": "witness_first",
        "resolved_source_count": 1,
        "resolved_span_count": 1,
        "unresolved_ref_ids": [],
    }), encoding="utf-8")
    (run_dir / "zoom_out.json").write_text(json.dumps({
        "resolution_mode": "witness_first",
        "perspective_block": "[P1] L1 summary",
        "markers_added": ["P1"],
        "summary_lineage_valid": True,
        "marker_resolution": [{
            "marker": "P1",
            "ref_kind": "local_summary",
            "ref_id": "summary-1",
            "resolved_source_count": 1,
            "resolved_span_count": 1,
            "unresolved_ref_ids": [],
            "citem_ids": ["C1"],
        }],
    }), encoding="utf-8")

    report = analyze_runs(tmp_path / "runs")
    assert report["run_count"] == 1
    assert report["rates"]["bounded_context"] == 1.0
    assert report["rates"]["source_span_lineage"] == 1.0
    assert report["rates"]["zoom_out_summary"] == 1.0
    assert report["rates"]["cleanup"] == 1.0
    assert report["rates"]["publication_gate_declared"] == 1.0
    assert report["rates"]["publishable_outputs"] == 1.0
    assert report["rates"]["invalid_published_as_valid"] == 0.0
    claims = {claim["claim_id"]: claim for claim in report["claims"]}
    assert claims["CIMA-C5A"]["status"] == "demonstrated"
    assert claims["CIMA-C5B"]["status"] == "not_demonstrated"
    assert any(claim["status"] == "not_claimed" for claim in report["claims"])

    out = tmp_path / "publication"
    write_outputs(report, out)
    assert (out / "publication_evidence_report.md").exists()
    assert (out / "publication_claim_matrix.csv").exists()


def test_publication_audit_flags_allowed_markers_when_visible_support_is_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "case-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"case_id": "case-1", "dataset_id": "demo"}), encoding="utf-8")
    (run_dir / "runtime_context.json").write_text(json.dumps({
        "markers": ["S1"],
        "token_usage": {"context": 50, "available_for_content": 100},
        "marker_resolution": [{
            "marker": "S1",
            "ref_kind": "citem",
            "ref_id": "c1",
            "resolved_source_ids": ["src1"],
            "resolved_span_ids": ["sp1"],
            "resolved_source_count": 1,
            "resolved_span_count": 1,
            "unresolved_ref_ids": [],
        }],
        "visible_marker_support": [],
    }), encoding="utf-8")
    (run_dir / "citation_contract.json").write_text(json.dumps({
        "passed": True,
        "published_integrity_passed": True,
        "publication_status": "publishable",
        "publishable": True,
        "blocked_by_cima": False,
        "invalid_published_as_valid": False,
        "publication_gate": {"publication_status": "publishable", "publishable": True, "blocked_by_cima": False},
        "allowed_markers": ["S1"],
        "available_marker_count": 1,
        "cited_markers": ["S1"],
    }), encoding="utf-8")

    report = analyze_runs(tmp_path / "runs")
    row = report["rows"][0]

    assert row["allowed_without_visible_support_count"] == 1
    assert row["cited_without_visible_support_count"] == 1
    assert row["visible_marker_anchor_passed"] is False
    assert report["drop_accounting"]["total_allowed_without_visible_support"] == 1
    assert report["rates"]["visible_marker_anchor_passed"] == 0.0
