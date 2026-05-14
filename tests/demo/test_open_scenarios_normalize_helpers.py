from __future__ import annotations

import json
from pathlib import Path

from cima_demo.demo.open_scenarios.normalize import _render_meeting_transcript, normalize_dataset


def test_render_meeting_transcript_handles_structured_turns() -> None:
    transcript, turns = _render_meeting_transcript(
        {
            "meeting_transcripts": [
                {"speaker": "A", "content": "Hello"},
                {"speaker_name": "B", "text": "World"},
            ]
        }
    )
    assert turns == ["A: Hello", "B: World"]
    assert transcript == "A: Hello\nB: World"


def test_normalize_dataset_rejects_fever_until_standard_integration_exists(tmp_path: Path) -> None:
    try:
        normalize_dataset(
            "fever",
            raw_root=tmp_path,
            out_root=tmp_path / "out",
            splits=["test"],
            limit_per_split=1,
            seed=17,
        )
    except ValueError as exc:
        assert "Dataset 'fever' is excluded" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("FEVER normalization must be rejected until FEVER-standard integration exists")


def test_qasper_normalization_extracts_nested_answers_and_falls_back_from_empty_test(tmp_path: Path, monkeypatch) -> None:
    from cima_demo.demo.open_scenarios import normalize as mod

    class FakeDataset(list):
        pass

    fake = {
        "test": FakeDataset([
            {
                "id": "paper-test",
                "title": "No gold",
                "abstract": "A",
                "full_text": {"section_name": ["Intro"], "paragraphs": [["No answers here."]]},
                "qas": {"question": ["Unanswered?"], "question_id": ["qt"], "answers": [[]]},
            }
        ]),
        "validation": FakeDataset([
            {
                "id": "paper-val",
                "title": "Paper",
                "abstract": "Abstract text.",
                "full_text": {"section_name": ["Intro"], "paragraphs": [["Paragraph evidence."]]},
                "qas": {
                    "question": ["What is reported?"],
                    "question_id": ["q1"],
                    "answers": [[
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["an extractive answer"],
                                "yes_no": None,
                                "free_form_answer": "",
                                "evidence": ["Paragraph evidence."],
                                "highlighted_evidence": ["Paragraph evidence."],
                            },
                            "annotation_id": "a1",
                        }
                    ]],
                },
            }
        ]),
    }

    monkeypatch.setattr(mod, "_require_hf_datasets", lambda: (lambda _path: fake))
    result = normalize_dataset(
        "qasper",
        raw_root=tmp_path,
        out_root=tmp_path / "out",
        splits=["test"],
        limit_per_split=None,
        seed=17,
    )
    assert result.case_count == 1
    payload = json.loads((tmp_path / "out" / "qasper" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["metadata"]["source_split"] == "validation"
    assert payload["gold_answers"] == ["an extractive answer"]
    assert payload["gold_evidence"][0]["text"] == "Paragraph evidence."


def test_explainmeetsum_normalization_reads_nested_explainable_qmsum(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "explainmeetsum" / "repo" / "data" / "ExplainMeetSum" / "test"
    root.mkdir(parents=True)
    (root / "meeting1.json").write_text(
        json.dumps(
            {
                "meeting_transcripts": [{"speaker": "A", "content": "Opening. Evidence sentence."}],
                "explainable_qmsum": {
                    "general_query_list": [
                        {
                            "query": "Summarize the meeting.",
                            "answer": "The meeting opened.",
                            "explainable_answer": [
                                {
                                    "answer_sentence": "The meeting opened.",
                                    "evidence": [
                                        {"type": "CES", "content": "Opening."},
                                        {"type": "CES", "content": "Evidence sentence."},
                                    ],
                                }
                            ],
                        }
                    ],
                    "specific_query_list": [],
                },
            }
        ),
        encoding="utf-8",
    )
    result = normalize_dataset(
        "explainmeetsum",
        raw_root=tmp_path / "raw",
        out_root=tmp_path / "out",
        splits=["test"],
        limit_per_split=None,
        seed=17,
    )
    assert result.case_count == 1
    payload = json.loads((tmp_path / "out" / "explainmeetsum" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["prompt"] == "Summarize the meeting."
    assert payload["gold_answers"] == ["The meeting opened."]
    assert [item["text"] for item in payload["gold_evidence"]] == ["Opening.", "Evidence sentence."]


def test_qasper_normalization_handles_hf_columnar_qas_and_answers(tmp_path: Path, monkeypatch) -> None:
    from cima_demo.demo.open_scenarios import normalize as mod

    class FakeDataset(list):
        pass

    fake = {
        "validation": FakeDataset([
            {
                "id": "paper-col",
                "title": "Columnar Paper",
                "abstract": "Abstract text.",
                "full_text": {
                    "section_name": ["Intro"],
                    "paragraphs": [["Paragraph evidence."]],
                },
                "qas": {
                    "question": ["What does it report?"],
                    "question_id": ["q-col"],
                    "question_writer": ["wq"],
                    "answers": {
                        "annotation_id": [["ann1", "ann2"]],
                        "worker_id": [["worker1", "worker2"]],
                        "answer": [
                            {
                                "unanswerable": [False, False],
                                "extractive_spans": [["extractive answer"], []],
                                "yes_no": [None, True],
                                "free_form_answer": ["", ""],
                                "evidence": [["Paragraph evidence."], []],
                                "highlighted_evidence": [["Highlighted evidence."], []],
                            }
                        ],
                    },
                },
            }
        ])
    }

    monkeypatch.setattr(mod, "_require_hf_datasets", lambda: (lambda _path: fake))
    result = normalize_dataset(
        "qasper",
        raw_root=tmp_path,
        out_root=tmp_path / "out",
        splits=["test"],
        limit_per_split=None,
        seed=17,
    )
    assert result.case_count == 1
    payload = json.loads((tmp_path / "out" / "qasper" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["prompt"] == "What does it report?"
    assert "extractive answer" in payload["gold_answers"]
    assert "Yes" in payload["gold_answers"]
    assert [item["text"] for item in payload["gold_evidence"]] == ["Paragraph evidence.", "Highlighted evidence."]
