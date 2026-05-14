from cima_demo.demo.open_scenarios.normalize import _canonicalize_explainmeetsum_prompt


def test_explainmeetsum_canonicalizes_private_setting_intuition_typo() -> None:
    prompt, metadata = _canonicalize_explainmeetsum_prompt(
        "What are the pros and cons of intuition in a private setting?",
        answer="The discussion concerns home tuition and alternative provision.",
        evidence_texts=["home tuition is sometimes used for excluded pupils"],
        transcript_text="",
    )

    assert "tuition" in prompt.lower()
    assert "intuition" not in prompt.lower()
    assert metadata["prompt_canonicalized"] is True
    assert metadata["original_prompt"].startswith("What are")


def test_explainmeetsum_does_not_canonicalize_unrelated_intuition() -> None:
    prompt, metadata = _canonicalize_explainmeetsum_prompt(
        "How did intuition affect the decision?",
        answer="They relied on judgement.",
        evidence_texts=[],
        transcript_text="No education provision terms here.",
    )

    assert prompt == "How did intuition affect the decision?"
    assert metadata == {}
