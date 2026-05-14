from __future__ import annotations

from cima_demo.demo.context.evidence_marker_registry import filter_citable_items
from cima_demo.demo.runtime.prompt_trace import build_prompt_lint
from cima_demo.domain.entities import LLMMessage


def test_evidence_marker_registry_filters_uncitable_citem_and_keeps_summary_witness() -> None:
    items = [
        {"marker": "S1", "ref_kind": "citem", "ref_id": "c1", "content": "direct"},
        {"marker": "S2", "ref_kind": "citem", "ref_id": "missing", "content": "missing"},
        {"marker": "S3", "ref_kind": "summary", "ref_id": "sum1", "content": "summary"},
    ]
    rows = [
        {
            "marker": "S1",
            "ref_kind": "citem",
            "ref_id": "c1",
            "resolved_source_ids": ["src1"],
            "resolved_span_ids": ["span1"],
            "resolved_source_count": 1,
            "resolved_span_count": 1,
            "unresolved_ref_ids": [],
            "citem_ids": ["c1"],
            "summary_ids": [],
        },
        {
            "marker": "S2",
            "ref_kind": "citem",
            "ref_id": "missing",
            "resolved_source_ids": [],
            "resolved_span_ids": [],
            "resolved_source_count": 0,
            "resolved_span_count": 0,
            "unresolved_ref_ids": ["missing"],
            "citem_ids": ["missing"],
            "summary_ids": [],
        },
        {
            "marker": "S3",
            "ref_kind": "summary",
            "ref_id": "sum1",
            "resolved_source_ids": ["src1"],
            "resolved_span_ids": ["span1", "span2"],
            "resolved_source_count": 1,
            "resolved_span_count": 2,
            "unresolved_ref_ids": [],
            "citem_ids": ["c1", "c2"],
            "citem_witnesses": [
                {"citem_id": "c1", "source_ids": ["src1"], "span_ids": ["span1"]},
                {"citem_id": "c2", "source_ids": ["src1"], "span_ids": ["span2"]},
            ],
            "summary_ids": ["sum1"],
        },
    ]
    filtered_items, filtered_rows, registry = filter_citable_items(items=items, marker_resolution=rows)
    assert [item["marker"] for item in filtered_items] == ["S1", "S3"]
    assert [row["marker"] for row in filtered_rows] == ["S1", "S3"]
    registry_by_marker = {row["marker"]: row for row in registry}
    assert registry_by_marker["S1"]["resolution_status"] == "source_span"
    assert registry_by_marker["S2"]["citable"] is False
    assert registry_by_marker["S3"]["resolution_status"] == "summary_witness"


def test_prompt_lint_rejects_allowed_marker_without_evidence_resolution() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S1", "citable": False, "resolution_status": "unresolved"}],
    )
    assert not lint["passed"]
    assert "allowed_markers_without_source_span_or_summary_witness" in lint["failures"]


def test_prompt_lint_accepts_allowed_marker_with_source_span_resolution() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S1", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[{"marker": "S1", "prompt_offsets_exact": True, "visible_slice_verified": True, "prompt_sha256": "p", "visible_slice_sha256": "abc"}],
    )
    assert lint["passed"]
    assert not lint["allowed_markers_without_resolution"]


def test_prompt_lint_rejects_allowed_marker_when_visible_support_rows_are_missing() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "[S1] Visible-looking text is not enough without structured support.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S1", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[],
    )
    assert not lint["passed"]
    assert "missing_visible_marker_support" in lint["failures"]
    assert "allowed_markers_without_prompt_visible_support" in lint["failures"]
    assert lint["allowed_markers_without_visible_support"] == ["S1"]


def test_prompt_lint_rejects_allowed_marker_without_prompt_visible_support() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S1", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[{"marker": "S2", "prompt_offsets_exact": True, "visible_slice_verified": True, "prompt_sha256": "p", "visible_slice_sha256": "abc"}],
    )
    assert not lint["passed"]
    assert "allowed_markers_without_prompt_visible_support" in lint["failures"]
    assert lint["allowed_markers_without_visible_support"] == ["S1"]


def test_prompt_lint_accepts_evidence_marker_with_visible_support() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "[E1] Visible zoom evidence.\n"
                    "Allowed citation markers are exactly and only: [E1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["E1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "E1", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[{"marker": "E1", "prompt_offsets_exact": True, "visible_slice_verified": True, "prompt_sha256": "p", "visible_slice_sha256": "abc"}],
    )
    assert lint["passed"]
    assert lint["marker_literals_in_prompt"] == ["E1"]


def test_visible_support_does_not_discover_markers_from_extra_block_text() -> None:
    from types import SimpleNamespace

    from cima_demo.demo.runtime.controller import DemoTurnController

    controller = object.__new__(DemoTurnController)
    context_view = SimpleNamespace(
        items=[],
        visible_marker_support=[],
        runtime_extra_visible_marker_support=[],
    )

    support = controller._visible_marker_support(context_view, extra_blocks=["[S9] literal source text, not structured support"])

    assert "S9" not in support


def test_posthoc_anchor_is_not_marked_exact() -> None:
    from cima_demo.demo.runtime.controller import DemoTurnController

    controller = object.__new__(DemoTurnController)
    support = {
        "runtime_context:S1": {
            "marker": "S1",
            "marker_namespace": "runtime_context",
            "marker_uid": "runtime_context:S1",
            "visible_text_preview": "Visible support text",
            "visible_char_count": 20,
        }
    }
    prompt = "User task\n\nContext pack:\n[S1] Visible support text\n\n## CIMA GROUNDING POLICY"

    anchored = controller._anchor_visible_support_to_prompt(support, prompt)

    assert anchored["runtime_context:S1"]["prompt_offsets_exact"] is False
    assert anchored["runtime_context:S1"]["prompt_anchor_method"] == "posthoc_marker_preview"


def test_rendered_prompt_support_offsets_are_exact_by_construction() -> None:
    from cima_demo.demo.runtime.controller import DemoTurnController

    controller = object.__new__(DemoTurnController)
    part = "Context pack:\n[S1] Visible support text"
    rows = [
        {
            "marker": "S1",
            "marker_namespace": "runtime_context",
            "marker_uid": "runtime_context:S1",
            "prompt_char_start": 0,
            "prompt_char_end": len("[S1] Visible support text"),
            "visible_text_preview": "Visible support text",
        }
    ]

    prompt, support = controller._render_prompt_parts_with_support([("User task", [], 0), (part, rows, len("Context pack:\n"))])
    row = support["runtime_context:S1"]

    assert row["prompt_offsets_exact"] is True
    assert row["prompt_anchor_method"] == "render_shift_verified"
    assert prompt[row["prompt_char_start"]:row["prompt_char_end"]] == "[S1] Visible support text"
    assert row["visible_slice_sha256"]


def test_prompt_lint_rejects_duplicate_registry_marker_label() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "[S1] First support.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[
            {"marker": "S1", "marker_uid": "runtime_context:S1", "ref_kind": "citem", "ref_id": "c1", "citable": True, "resolution_status": "source_span"},
            {"marker": "S1", "marker_uid": "zoom:S1", "ref_kind": "citem", "ref_id": "c2", "citable": True, "resolution_status": "source_span"},
        ],
        visible_marker_support=[{"marker": "S1", "marker_uid": "runtime_context:S1", "prompt_offsets_exact": True, "visible_slice_verified": True, "prompt_sha256": "p", "visible_slice_sha256": "s"}],
    )
    assert not lint["passed"]
    assert "duplicate_marker_registry_label" in lint["failures"]


def test_prompt_lint_rejects_unverified_visible_support() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "[S1] Support.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S1", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[{"marker": "S1", "prompt_offsets_exact": False, "visible_slice_verified": False, "prompt_sha256": "p", "visible_slice_sha256": "s"}],
    )
    assert not lint["passed"]
    assert "allowed_markers_without_prompt_visible_support" in lint["failures"]


def test_prompt_lint_allows_repeated_label_when_evidence_identity_is_same() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "[S1] Short visible support.\n\n[S1] Expanded visible support.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[
            {
                "marker": "S1",
                "marker_uid": "runtime_context:S1",
                "ref_kind": "citem",
                "ref_id": "c1",
                "source_ids": ["src1"],
                "spans": ["span1"],
                "citable": True,
                "resolution_status": "source_span",
            }
        ],
        visible_marker_support=[
            {
                "marker": "S1",
                "marker_uid": "runtime_context:S1",
                "ref_kind": "citem",
                "ref_id": "c1",
                "source_ids": ["src1"],
                "span_ids": ["span1"],
                "prompt_offsets_exact": True,
                "visible_slice_verified": True,
                "prompt_sha256": "p",
                "visible_slice_sha256": "slice-a",
            },
            {
                "marker": "S1",
                "marker_uid": "zoom:S1",
                "ref_kind": "citem",
                "ref_id": "c1",
                "source_ids": ["src1"],
                "span_ids": ["span1"],
                "prompt_offsets_exact": True,
                "visible_slice_verified": True,
                "prompt_sha256": "p",
                "visible_slice_sha256": "slice-b",
            },
        ],
    )

    assert lint["passed"]
    assert lint["visible_marker_label_collisions"] == {}


def test_prompt_lint_does_not_require_generic_s1_example_and_detects_number_format_from_contract_block() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "User task:\nWhat colors are worn by the Oregon Duck?\n\n"
                    "Output contract:\n  format: number\n  required_evidence: true\n\n"
                    "Context pack:\n[S2] The mascot wears green and yellow.\n\n"
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "Allowed citation markers are exactly and only: [S2]. No other citation markers exist. "
                    "For every factual paragraph or bullet, cite at least one marker from the closed list."
                ),
            ),
        ],
        allowed_markers=["S2"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S2", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[{"marker": "S2", "prompt_offsets_exact": True, "visible_slice_verified": True, "prompt_sha256": "p", "visible_slice_sha256": "s"}],
    )
    assert not lint["passed"]
    assert lint["output_format"] == "number"
    assert "output_format_number_for_non_numeric_task" in lint["failures"]
    assert lint["unknown_marker_literals_in_prompt"] == []


def test_prompt_lint_allows_number_format_for_numeric_task_family() -> None:
    lint = build_prompt_lint(
        call_kind="answer_generation",
        messages=[
            LLMMessage(role="system", content="Use only the CIMA-provided context."),
            LLMMessage(
                role="user",
                content=(
                    "Output contract:\n  format: number\n\n"
                    "Context pack:\n[S1] The value is 42.\n\n"
                    "## CIMA GROUNDING POLICY\n"
                    "Use only the CIMA-provided context. You must not rely on training-time knowledge. "
                    "If there is insufficient evidence, do not guess.\n"
                    "Allowed citation markers are exactly and only: [S1]. No other citation markers exist. "
                    "For every factual paragraph or bullet, cite at least one marker from the closed list."
                ),
            ),
        ],
        allowed_markers=["S1"],
        require_answer_grounding=True,
        marker_registry=[{"marker": "S1", "citable": True, "resolution_status": "source_span"}],
        visible_marker_support=[{"marker": "S1", "prompt_offsets_exact": True, "visible_slice_verified": True, "prompt_sha256": "p", "visible_slice_sha256": "s"}],
        task_family="SOURCE_BOUND_QUANT",
    )
    assert lint["passed"]
    assert lint["output_format"] == "number"
    assert not lint["output_format_number_mismatch"]
