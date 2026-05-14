from __future__ import annotations

from typing import Any

import httpx
import pytest

from cima_demo.demo.open_scenarios.execute import OpenScenarioExecutor, RunnerConfig


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - should not be called by cleanup tests
        raise AssertionError("cleanup must not raise for HTTP status handling")


class _FakeDeleteClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.closed = False
        self.paths: list[str] = []

    async def delete(self, path: str, **_kwargs: Any) -> _FakeResponse:
        self.paths.append(path)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_open_execute_cleanup_read_error_is_recorded_not_raised(monkeypatch) -> None:
    executor = OpenScenarioExecutor(
        RunnerConfig(
            base_url="http://example.test",
            api_key=None,
            model="cima_demo",
            mode="both",
            max_context_tokens=128,
            reserve_output_tokens=32,
            settle_seconds=0,
            cleanup=True,
        )
    )
    original_client = _FakeDeleteClient([httpx.ReadError("connection closed"), _FakeResponse(404)])
    executor._client = original_client  # type: ignore[assignment]

    async def fake_reset_client() -> None:
        await original_client.aclose()
        executor._client = _FakeDeleteClient([_FakeResponse(404)])  # type: ignore[assignment]

    monkeypatch.setattr(executor, "_reset_client", fake_reset_client)

    result = await executor._delete_conversation("conv-1")

    assert result["ok"] is False
    assert result["final_verified"] is False
    assert result["attempts"][0]["error_class"] == "ReadError"
    assert original_client.closed is True


@pytest.mark.asyncio
async def test_open_execute_cleanup_all_failures_are_best_effort(monkeypatch) -> None:
    executor = OpenScenarioExecutor(
        RunnerConfig(
            base_url="http://example.test",
            api_key=None,
            model="cima_demo",
            mode="both",
            max_context_tokens=128,
            reserve_output_tokens=32,
            settle_seconds=0,
            cleanup=True,
        )
    )
    fake_client = _FakeDeleteClient([httpx.ReadError("connection closed"), _FakeResponse(500, "boom")])
    executor._client = fake_client  # type: ignore[assignment]

    async def fake_reset_client() -> None:
        executor._client = _FakeDeleteClient([_FakeResponse(500, "boom")])  # type: ignore[assignment]

    monkeypatch.setattr(executor, "_reset_client", fake_reset_client)

    result = await executor._delete_conversation("conv-1")

    assert result["ok"] is False
    assert [attempt.get("error_class") for attempt in result["attempts"]] == ["ReadError", None]
    assert result["attempts"][1]["status_code"] == 500


@pytest.mark.asyncio
async def test_open_execute_cleanup_rejects_async_202_without_final_audit() -> None:
    executor = OpenScenarioExecutor(
        RunnerConfig(
            base_url="http://example.test",
            api_key=None,
            model="cima_demo",
            mode="both",
            max_context_tokens=128,
            reserve_output_tokens=32,
            settle_seconds=0,
            cleanup=True,
        )
    )
    fake_client = _FakeDeleteClient([_FakeResponse(202), _FakeResponse(202)])
    executor._client = fake_client  # type: ignore[assignment]

    result = await executor._delete_conversation("conv-1")

    assert result["ok"] is False
    assert result["final_verified"] is False
    assert [attempt["status_code"] for attempt in result["attempts"]] == [202, 202]
    assert [attempt["ok"] for attempt in result["attempts"]] == [False, False]
    assert result["accepted_path"] is None


class _FakePostResponse:
    status_code = 202

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"accepted": True}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakePostClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any] | None]] = []

    async def post(self, path: str, json: dict[str, Any] | None = None, **_kwargs: Any) -> _FakePostResponse:
        self.posts.append((path, json))
        return _FakePostResponse()


@pytest.mark.asyncio
async def test_register_document_uses_file_text_source_kind() -> None:
    executor = OpenScenarioExecutor(
        RunnerConfig(
            base_url="http://example.test",
            api_key=None,
            model="cima_demo",
            mode="both",
            max_context_tokens=128,
            reserve_output_tokens=32,
            settle_seconds=0,
            cleanup=False,
        )
    )
    fake_client = _FakePostClient()
    executor._client = fake_client  # type: ignore[assignment]

    await executor._register_document(
        conversation_id="conv-1",
        document={"doc_id": "doc-1", "text": "Document text"},
        request_id="req-1",
    )

    assert fake_client.posts[0][0] == "/cima/v1/sources/register_text"
    assert fake_client.posts[0][1] is not None
    assert fake_client.posts[0][1]["source_kind"] == "file_text"
    assert fake_client.posts[0][1]["displayable"] is False
    assert fake_client.posts[0][1]["processable"] is True


def test_citation_contract_from_artifacts_flags_uncited_answer() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {
        "markers": [{"marker": "S1"}],
        "text": "[S1] Supporting fact",
    }
    chat = {"choices": [{"message": {"content": "Supporting fact is important."}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["requires_citations"] is True
    assert result["passed"] is False
    assert result["uncited_answer_block_count"] == 1


def test_citation_contract_from_artifacts_accepts_valid_context_marker() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}], "text": "[S1] Supporting fact"}
    chat = {"choices": [{"message": {"content": "Supporting fact is important. [S1]"}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["passed"] is True
    assert result["valid_cited_markers"] == ["S1"]


def test_open_execute_parser_accepts_max_cases_alias() -> None:
    from pathlib import Path
    from cima_demo.demo.open_scenarios.execute import build_parser

    args = build_parser().parse_args([
        "--cases", "cases",
        "--out", "runs",
        "--base-url", "http://example.test",
        "--max-cases", "1",
    ])

    assert args.cases == Path("cases")
    assert args.out == Path("runs")
    assert args.limit == 1


def test_citation_contract_ignores_short_summary_framing_and_repair_preface() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}, {"marker": "S2"}], "text": "[S1] A fact. [S2] Another fact."}
    chat = {"choices": [{"message": {"content": (
        "Here is the corrected summary with valid citations:\n\n"
        "The meeting discussed the main issues and evidence gaps.\n\n"
        "- The first supported point is grounded. [S1]\n\n"
        "- The second supported point is grounded. [S2]"
    )}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["passed"] is True
    assert result["uncited_answer_block_count"] == 0


def test_citation_contract_ignores_evidence_scope_note_without_citation() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}], "text": "[S1] Rural transport was discussed."}
    chat = {"choices": [{"message": {"content": (
        "- The discussion identified transport as a practical barrier for rural EOTAS learners. [S1]\n\n"
        "Evidence is limited to Powys and Torfaen; broader Welsh trends remain unspecified."
    )}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["passed"] is True
    assert result["uncited_answer_block_count"] == 0


def test_citation_contract_treats_pure_abstention_as_c3a_trace_not_cited_support() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {
        "context_id": "ctx-view-1",
        "markers": [{"marker": "S1"}, {"marker": "S2"}],
        "text": "[S1] Candidate evidence. [S2] More candidate evidence.",
    }
    zoom = {"markers_added": ["S1"]}
    chat = {"choices": [{"message": {"content": "NOT ENOUGH INFO"}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context, zoom_payload=zoom)

    assert result["passed"] is True
    assert result["published_integrity_passed"] is True
    assert result["answer_type"] == "insufficient_evidence"
    assert result["requires_citations"] is False
    assert result["factual_citations_required"] is False
    assert result["c3a_traceable_abstention"]["passed"] is True
    assert result["c3a_traceable_abstention"]["insufficiency_trace"]["inspected_markers"] == ["S1", "S2", "S1"] or result["c3a_traceable_abstention"]["insufficiency_trace"]["inspected_markers"]


def test_citation_contract_keeps_factual_uncited_answer_failed() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}], "text": "[S1] Supporting fact"}
    chat = {"choices": [{"message": {"content": "The answer is a factual claim without citation."}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["answer_type"] == "factual_answer"
    assert result["requires_citations"] is True
    assert result["passed"] is False
    assert result["c3a_traceable_abstention"]["applicable"] is False



def test_citation_contract_marks_unsupported_factual_output_blocked() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}], "text": "[S1] Supporting fact"}
    chat = {"choices": [{"message": {"content": "The answer is a factual claim without citation."}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["published_integrity_passed"] is False
    assert result["publication_status"] == "blocked"
    assert result["publishable"] is False
    assert result["blocked_by_cima"] is True
    assert result["blocked_reason"] == "uncited_block"
    assert result["invalid_published_as_valid"] is False


def test_citation_contract_blocks_when_sanitizer_would_leave_uncited_block() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}], "text": "[S1] Supporting fact"}
    chat = {"choices": [{"message": {"content": "The answer relies only on an invalid citation. [S8]"}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["deterministic_sanitization_applied"] is True
    assert result["publication_status"] == "blocked"
    assert result["publishable"] is False
    assert result["blocked_reason"] == "uncited_block_after_sanitizer"
    assert result["invalid_published_as_valid"] is False


def test_citation_contract_publishable_after_deterministic_marker_strip() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    context = {"markers": [{"marker": "S1"}], "text": "[S1] Supporting fact"}
    chat = {"choices": [{"message": {"content": "The answer is supported. [S1][S3]"}}]}

    result = _citation_contract_from_artifacts(chat_payload=chat, context_payload=context)

    assert result["deterministic_sanitization_applied"] is True
    assert result["published_answer_text"] == "The answer is supported. [S1]"
    assert result["publication_status"] == "publishable"
    assert result["publishable"] is True
    assert result["blocked_by_cima"] is False
    assert result["blocked_reason"] is None


def test_citation_contract_prefers_runtime_contract_over_exported_context_marker_drift() -> None:
    from cima_demo.demo.open_scenarios.execute import _citation_contract_from_artifacts

    # Exported pre-chat context may come from a different ContextView than the
    # answer-generation prompt.  The runtime contract is authoritative because
    # it is produced inside the same chat pass as the LLM prompt.
    exported_context = {"markers": [{"marker": "S1"}, {"marker": "S2"}], "text": "[S1] A. [S2] B."}
    chat = {"choices": [{"message": {"content": "Supported by runtime prompt marker. [S3]"}}]}
    runtime_contract = {
        "schema_version": "cima_demo.citation_contract.v2",
        "allowed_markers": ["S1", "S2", "S3"],
        "published_answer_text": "Supported by runtime prompt marker. [S3]",
        "passed": True,
        "publishable": True,
    }
    prompt_trace = {"llm_calls": [{"call_kind": "answer_generation", "allowed_markers": ["S1", "S2", "S3"]}]}

    result = _citation_contract_from_artifacts(
        chat_payload=chat,
        context_payload=exported_context,
        runtime_citation_contract=runtime_contract,
        prompt_trace_payload=prompt_trace,
    )

    assert result["source"] == "runtime_chat_citation_contract"
    assert result["allowed_markers"] == ["S1", "S2", "S3"]
    assert result["marker_registry_consistency"]["prompt_equals_citation_contract"] is True
    assert result["marker_registry_consistency"]["citation_contract_equals_reconstructed_artifacts"] is False


def test_context_payload_from_runtime_snapshot_preserves_marker_resolution_shape() -> None:
    from cima_demo.demo.open_scenarios.execute import _context_payload_from_runtime_snapshot

    snapshot = {
        "context_id": "ctx-runtime",
        "context_text": "[S1] Runtime evidence",
        "markers": ["S1"],
        "run_id": "run-1",
        "turn_id": "turn-1",
        "budget": {"tokens_used": 12, "available_for_content": 100, "overhead_tokens": 3},
        "marker_resolution": [{"marker": "S1", "resolved_source_ids": ["src"], "resolved_span_ids": ["span"]}],
        "resolved_source_ids": ["src"],
        "resolved_span_ids": ["span"],
        "resolved_source_count": 1,
        "resolved_span_count": 1,
        "unresolved_ref_ids": [],
        "resolution_mode": "mixed",
    }

    result = _context_payload_from_runtime_snapshot(snapshot)

    assert result is not None
    assert result["context_id"] == "ctx-runtime"
    assert result["context_pack"] == "[S1] Runtime evidence"
    assert result["markers"] == ["S1"]
    assert result["marker_resolution"][0]["marker"] == "S1"
    assert result["source"] == "runtime_chat_context_snapshot"
