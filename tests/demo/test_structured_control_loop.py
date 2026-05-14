from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from cima_demo.demo.runtime.controller import DemoTurnController
from cima_demo.domain.entities import ContextView, KimaDelta, LLMMessage, Plan, TaskMemory
from cima_demo.domain.value_objects import ContextBudget, ItemType
from cima_demo.cognitive.kernel.state import TurnRuntime
from cima_demo.infrastructure.llm.llamacpp import LlamaCppAdapter


class _FakeStreamManager:
    def __init__(self) -> None:
        self.published: list[KimaDelta] = []

    async def publish(self, delta: KimaDelta) -> None:
        self.published.append(delta)


class _FakeContextService:
    def __init__(self) -> None:
        self.snapshot_id = "ctx-1"
        self.memory_calls: list[dict[str, Any]] = []

    async def build(self, **_: Any) -> ContextView:
        view = ContextView(
            text="[S1] Main fact\n[S2] Secondary fact",
            tokens_used=42,
            coverage_score=0.9,
            citem_ids=["c1", "c2"],
            items=[
                {"marker": "S1", "ref_kind": "citem", "ref_id": "c1", "content": "Main fact"},
                {"marker": "S2", "ref_kind": "citem", "ref_id": "c2", "content": "Secondary fact"},
            ],
        )
        setattr(view, "evidence_marker_registry", [
            {"marker": "S1", "kind": "citem", "ref_kind": "citem", "ref_id": "c1", "citable": True, "resolution_status": "source_span", "source_ids": ["src1"], "spans": ["span1"], "lineage_refs": ["c1"]},
            {"marker": "S2", "kind": "citem", "ref_kind": "citem", "ref_id": "c2", "citable": True, "resolution_status": "source_span", "source_ids": ["src2"], "spans": ["span2"], "lineage_refs": ["c2"]},
        ])
        s1 = "[S1] Main fact"
        s2 = "[S2] Secondary fact"
        setattr(view, "visible_marker_support", [
            {"marker": "S1", "marker_namespace": "runtime_context", "marker_uid": "runtime_context:S1", "ref_kind": "citem", "ref_id": "c1", "visible_text_preview": "Main fact", "visible_char_count": len(s1), "prompt_char_start": 0, "prompt_char_end": len(s1)},
            {"marker": "S2", "marker_namespace": "runtime_context", "marker_uid": "runtime_context:S2", "ref_kind": "citem", "ref_id": "c2", "visible_text_preview": "Secondary fact", "visible_char_count": len(s2), "prompt_char_start": len(s1) + 1, "prompt_char_end": len(s1) + 1 + len(s2)},
        ])
        return view

    def last_snapshot_id(self) -> str:
        return self.snapshot_id

    async def zoom(self, **_: Any) -> dict[str, Any]:
        return {
            "evidence_block": "[S1] Main fact",
            "markers_added": ["S1"],
            "token_usage": {"evidence": 5},
            "marker_resolution": [{
                "marker": "S1",
                "ref_kind": "citem",
                "ref_id": "c1",
                "resolved_source_count": 1,
                "resolved_span_count": 1,
                "resolved_source_ids": ["src1"],
                "resolved_span_ids": ["span1"],
                "unresolved_ref_ids": [],
                "citem_ids": ["c1"],
            }],
        }

    async def zoom_out(self, **_: Any) -> dict[str, Any]:
        return {
            "perspective_block": "[P1] Big picture",
            "markers_added": ["P1"],
            "token_usage": {"perspective": 7},
            "summary_lineage_valid": True,
            "marker_resolution": [{
                "marker": "P1",
                "ref_kind": "local_summary",
                "ref_id": "summary-1",
                "resolved_source_count": 1,
                "resolved_span_count": 1,
                "resolved_source_ids": ["src1"],
                "resolved_span_ids": ["span1"],
                "unresolved_ref_ids": [],
                "citem_ids": ["c1"],
                "citem_witnesses": [{"citem_id": "c1", "source_ids": ["src1"], "span_ids": ["span1"]}],
            }],
            "zoom_out_marker_resolution": [{
                "marker": "P1",
                "ref_kind": "local_summary",
                "ref_id": "summary-1",
                "resolved_source_count": 1,
                "resolved_span_count": 1,
                "resolved_source_ids": ["src1"],
                "resolved_span_ids": ["span1"],
                "unresolved_ref_ids": [],
                "citem_ids": ["c1"],
                "citem_witnesses": [{"citem_id": "c1", "source_ids": ["src1"], "span_ids": ["span1"]}],
            }],
        }

    async def apply_memory(self, **kwargs: Any) -> dict[str, Any]:
        self.memory_calls.append(kwargs)
        return {"accepted": kwargs.get("conclude", []), "rejected": []}


class _FakeJournal:
    def __init__(self) -> None:
        self.phases: list[tuple[str, dict[str, Any]]] = []

    async def append_phase(self, *, run_id: str, conversation_id: str, phase_name: str, payload: dict[str, Any] | None = None) -> int:
        self.phases.append((phase_name, payload or {}))
        return len(self.phases)


class _FakeLLM:
    def __init__(self) -> None:
        self.structured_calls: list[list[LLMMessage]] = []
        self.stream_calls: list[list[LLMMessage]] = []

    async def complete_structured(self, messages: list[LLMMessage], **_: Any) -> dict[str, Any]:
        self.structured_calls.append(messages)
        if len(self.structured_calls) == 1:
            return {
                "needs_zoom": True,
                "zoom_markers": ["S1"],
                "needs_zoom_out": True,
                "focus": "main fact",
                "reason": "Need the direct evidence and global perspective",
            }
        return {
            "cited_markers": ["S1", "P1"],
            "conclusions": [
                {"kind": "FACT", "content": "Main fact is relevant", "confidence": 0.9},
                {"kind": "NOTE", "content": "Perspective used", "confidence": 0.6},
            ],
        }

    async def stream_text(self, messages: list[LLMMessage], **_: Any):
        self.stream_calls.append(messages)
        for token in ["Answer ", "grounded ", "in [S1] and [P1]."]:
            yield token

    async def complete(self, messages: list[LLMMessage], **_: Any) -> str:
        return "fallback"


@pytest.mark.asyncio
async def test_demo_turn_controller_runs_structured_control_answer_and_memory() -> None:
    llm = _FakeLLM()
    stream = _FakeStreamManager()
    context = _FakeContextService()
    journal = _FakeJournal()
    controller = DemoTurnController(
        llm_port=llm,  # type: ignore[arg-type]
        stream_manager=stream,
        context_service=context,  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),  # unused directly
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        run_journal=journal,  # type: ignore[arg-type]
        llm_max_tokens=256,
    )
    rt = TurnRuntime(
        conversation_id="conv-1",
        turn_id="turn-1",
        run_id="run-1",
        user_message="What matters here?",
        phase="recall",
    )
    rt.output_contract = SimpleNamespace(
        format="text",
        representation="plain_text",
        base_unit=None,
        display_scale=None,
        rounding_rule=None,
        precision=None,
        required_evidence=True,
    )
    task_memory = TaskMemory(conversation_id="conv-1")

    await controller.run_turn(rt, task_memory, plan=None)

    assert rt.assistant_reply_buffer == "Answer grounded in [S1] and [P1]."
    assert rt.cited_markers == ["S1", "P1"]
    assert rt.conclusions_types_seen == ["FACT", "NOTE"]
    assert any(delta.token == "Answer " for delta in stream.published)
    phase_names = [name for name, _ in journal.phases]
    assert phase_names == [
        "CONTEXT_0",
        "CONTROL_PASS",
        "ENRICH_ZOOM",
        "ENRICH_ZOOM_OUT",
        "ANSWER",
        "CITATION_CONTRACT",
        "MEMORY_APPLY",
        "COMMIT",
    ]
    assert context.memory_calls[0]["conclude"] == [
        "FACT: Main fact is relevant",
        "NOTE: Perspective used",
    ]


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    @asynccontextmanager
    async def stream(self, method: str, url: str, json: dict[str, Any]):
        yield _FakeStreamResponse(self._lines)


@pytest.mark.asyncio
async def test_llamacpp_adapter_complete_structured_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LlamaCppAdapter("http://localhost:8080")

    async def _fake_complete(messages: list[LLMMessage], **_: Any) -> str:
        return "```json\n{\"needs_zoom\": true, \"zoom_markers\": [\"S1\"]}\n```"

    monkeypatch.setattr(adapter, "complete", _fake_complete)
    payload = await adapter.complete_structured([LLMMessage(role="user", content="x")])
    assert payload == {"needs_zoom": True, "zoom_markers": ["S1"]}


@pytest.mark.asyncio
async def test_llamacpp_adapter_stream_text_yields_visible_chunks() -> None:
    adapter = LlamaCppAdapter("http://localhost:8080")
    adapter._client = _FakeClient([
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "hidden"}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "world"}, "finish_reason": None}]}),
        "data: [DONE]",
    ])
    chunks = []
    async for chunk in adapter.stream_text([LLMMessage(role="user", content="hi")]):
        chunks.append(chunk)
    assert chunks == ["Hello ", "world"]


class _RepairLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0
        self.complete_calls = 0

    async def complete_structured(self, messages: list[LLMMessage], **_: Any) -> dict[str, Any]:
        self.structured_calls += 1
        if self.structured_calls == 1:
            return {"needs_zoom": False, "zoom_markers": [], "needs_zoom_out": False, "focus": None, "reason": "enough"}
        return {"cited_markers": ["S1"], "conclusions": []}

    async def stream_text(self, messages: list[LLMMessage], **_: Any):
        self.stream_calls += 1
        for token in ["The ", "main fact ", "matters."]:
            yield token

    async def complete(self, messages: list[LLMMessage], **_: Any) -> str:
        self.complete_calls += 1
        return "The main fact matters. [S1]"


@pytest.mark.asyncio
async def test_demo_turn_controller_repairs_uncited_answers_before_publishing() -> None:
    llm = _RepairLLM()
    stream = _FakeStreamManager()
    context = _FakeContextService()
    journal = _FakeJournal()
    controller = DemoTurnController(
        llm_port=llm,  # type: ignore[arg-type]
        stream_manager=stream,
        context_service=context,  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        run_journal=journal,  # type: ignore[arg-type]
        llm_max_tokens=256,
    )
    rt = TurnRuntime(
        conversation_id="conv-repair",
        turn_id="turn-repair",
        run_id="run-repair",
        user_message="What matters here?",
        phase="recall",
    )
    task_memory = TaskMemory(conversation_id="conv-repair")

    await controller.run_turn(rt, task_memory, plan=None)

    assert rt.assistant_reply_buffer == "The main fact matters. [S1]"
    assert stream.published[0].token == "The main fact matters. [S1]"
    assert rt.demo_citation_contract["passed"] is True
    assert rt.demo_citation_contract["repair_attempted"] is True
    assert rt.demo_citation_contract["repaired_from"]["passed"] is False
    assert llm.complete_calls == 1


def test_demo_context_service_drops_current_turn_item_and_renumbers_markers() -> None:
    from cima_demo.demo.context.service import _drop_current_turn_items

    view = ContextView(
        text="CONTEXT\n\n[S1] Summarize the whole meeting.\n\n[S2] DOCUMENT doc: first evidence.\n\n[S3] DOCUMENT doc: second evidence.",
        tokens_used=30,
        coverage_score=0.9,
        citem_ids=["u1", "d1", "d2"],
        items=[
            {"marker": "S1", "ref_kind": "citem", "ref_id": "u1", "actor": "user", "content": "Summarize the whole meeting."},
            {"marker": "S2", "ref_kind": "citem", "ref_id": "d1", "actor": "agent", "content": "DOCUMENT doc: first evidence."},
            {"marker": "S3", "ref_kind": "citem", "ref_id": "d2", "actor": "agent", "content": "DOCUMENT doc: second evidence."},
        ],
    )

    filtered = _drop_current_turn_items(view, {"Summarize the whole meeting."})

    assert [item["marker"] for item in filtered.items] == ["S1", "S2"]
    assert [item["ref_id"] for item in filtered.items] == ["d1", "d2"]
    assert "Summarize the whole meeting" not in filtered.text
    assert "[S1] DOCUMENT doc: first evidence." in filtered.text
    assert "[S2] DOCUMENT doc: second evidence." in filtered.text

class _FramingLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0
        self.complete_calls = 0

    async def complete_structured(self, messages: list[LLMMessage], **_: Any) -> dict[str, Any]:
        self.structured_calls += 1
        return {"needs_zoom": False, "zoom_markers": [], "needs_zoom_out": False, "focus": None, "reason": "enough"}

    async def stream_text(self, messages: list[LLMMessage], **_: Any):
        self.stream_calls += 1
        for token in [
            "The meeting highlighted key issues for vulnerable learners.\n\n",
            "- Main fact matters for the answer. [S1]",
        ]:
            yield token

    async def complete(self, messages: list[LLMMessage], **_: Any) -> str:
        self.complete_calls += 1
        return "SHOULD NOT REPAIR [S1]"


@pytest.mark.asyncio
async def test_demo_turn_controller_does_not_repair_framing_only_uncited_text() -> None:
    llm = _FramingLLM()
    stream = _FakeStreamManager()
    context = _FakeContextService()
    journal = _FakeJournal()
    controller = DemoTurnController(
        llm_port=llm,  # type: ignore[arg-type]
        stream_manager=stream,
        context_service=context,  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        run_journal=journal,  # type: ignore[arg-type]
        llm_max_tokens=256,
        llm_memory_pass=False,
    )
    rt = TurnRuntime(
        conversation_id="conv-framing",
        turn_id="turn-framing",
        run_id="run-framing",
        user_message="Summarize the meeting.",
        phase="recall",
    )

    await controller.run_turn(rt, TaskMemory(conversation_id="conv-framing"), plan=None)

    assert rt.demo_citation_contract["passed"] is True
    assert rt.demo_citation_contract["repair_attempted"] is False
    assert llm.complete_calls == 0
    assert "SHOULD NOT REPAIR" not in rt.assistant_reply_buffer


def test_demo_turn_controller_sanitizes_repair_prefaces() -> None:
    controller = DemoTurnController(
        llm_port=_RepairLLM(),  # type: ignore[arg-type]
        stream_manager=_FakeStreamManager(),
        context_service=_FakeContextService(),  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        llm_max_tokens=256,
    )

    cleaned = controller._sanitize_visible_answer(
        "Here is the corrected summary with valid citations:\n\n---\n\n- Main fact. [S1]\n"
    )

    assert cleaned == "- Main fact. [S1]"

class _TrailingCloserLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0
        self.complete_calls = 0

    async def complete_structured(self, messages: list[LLMMessage], **_: Any) -> dict[str, Any]:
        self.structured_calls += 1
        return {"needs_zoom": False, "zoom_markers": [], "needs_zoom_out": False, "focus": None, "reason": "enough"}

    async def stream_text(self, messages: list[LLMMessage], **_: Any):
        self.stream_calls += 1
        yield "- Main fact matters. [S1]\n\n"
        yield "The meeting emphasized the need for sustained investment and collaboration."

    async def complete(self, messages: list[LLMMessage], **_: Any) -> str:
        self.complete_calls += 1
        return "SHOULD NOT REPAIR [S1]"


@pytest.mark.asyncio
async def test_demo_turn_controller_sanitizes_trailing_uncited_closer_without_repair() -> None:
    llm = _TrailingCloserLLM()
    controller = DemoTurnController(
        llm_port=llm,  # type: ignore[arg-type]
        stream_manager=_FakeStreamManager(),
        context_service=_FakeContextService(),  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        llm_max_tokens=256,
        llm_memory_pass=False,
    )
    rt = TurnRuntime(
        conversation_id="conv-trailing",
        turn_id="turn-trailing",
        run_id="run-trailing",
        user_message="Summarize the meeting.",
        phase="recall",
    )

    await controller.run_turn(rt, TaskMemory(conversation_id="conv-trailing"), plan=None)

    assert rt.demo_citation_contract["passed"] is True
    assert rt.demo_citation_contract["repair_attempted"] is False
    assert rt.demo_citation_contract["sanitize_only"] is True
    assert llm.complete_calls == 0
    assert "The meeting emphasized" not in rt.assistant_reply_buffer
    assert "SHOULD NOT REPAIR" not in rt.assistant_reply_buffer


def test_demo_turn_controller_sanitizes_repair_meta_notes() -> None:
    controller = DemoTurnController(
        llm_port=_RepairLLM(),  # type: ignore[arg-type]
        stream_manager=_FakeStreamManager(),
        context_service=_FakeContextService(),  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        llm_max_tokens=256,
    )

    cleaned = controller._sanitize_visible_answer(
        "Here is the corrected summary with valid citations:\n\n---\n\n"
        "- Main fact. [S1]\n\n"
        "*(Note: Claims about data deficiencies lacked direct contextual support and were omitted.)*"
    )

    assert cleaned == "- Main fact. [S1]"

class _InvalidMarkerLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0
        self.complete_calls = 0

    async def complete_structured(self, messages: list[LLMMessage], **_: Any) -> dict[str, Any]:
        self.structured_calls += 1
        return {"needs_zoom": False, "zoom_markers": [], "needs_zoom_out": False, "focus": None, "reason": "enough"}

    async def stream_text(self, messages: list[LLMMessage], **_: Any):
        self.stream_calls += 1
        yield "The answer uses the main fact [S1][S3]."

    async def complete(self, messages: list[LLMMessage], **_: Any) -> str:
        self.complete_calls += 1
        return "SHOULD NOT REPAIR [S1]"


@pytest.mark.asyncio
async def test_demo_turn_controller_strips_invalid_markers_deterministically_without_repair() -> None:
    llm = _InvalidMarkerLLM()
    controller = DemoTurnController(
        llm_port=llm,  # type: ignore[arg-type]
        stream_manager=_FakeStreamManager(),
        context_service=_FakeContextService(),  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        llm_max_tokens=256,
        llm_memory_pass=False,
    )
    rt = TurnRuntime(
        conversation_id="conv-invalid-marker",
        turn_id="turn-invalid-marker",
        run_id="run-invalid-marker",
        user_message="What matters here?",
        phase="recall",
    )

    await controller.run_turn(rt, TaskMemory(conversation_id="conv-invalid-marker"), plan=None)

    assert rt.assistant_reply_buffer == "The answer uses the main fact [S1]."
    assert rt.demo_citation_contract["passed"] is True
    assert rt.demo_citation_contract["repair_attempted"] is False
    assert rt.demo_citation_contract["deterministic_sanitization_applied"] is True
    report = rt.demo_citation_contract["citation_sanitization_reports"][0]
    assert report["method"] == "deterministic_strip_invalid_markers"
    assert report["removed_invalid_markers"] == ["S3"]
    assert report["passed_after"] is True
    assert llm.complete_calls == 0


def test_demo_turn_controller_strips_invalid_markers_without_substituting_evidence() -> None:
    controller = DemoTurnController(
        llm_port=_RepairLLM(),  # type: ignore[arg-type]
        stream_manager=_FakeStreamManager(),
        context_service=_FakeContextService(),  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        llm_max_tokens=256,
    )

    stripped, report = controller._strip_invalid_citation_markers(
        "Claim A [S1, S3]. Claim B [S4].",
        {"S1", "S2"},
    )

    assert stripped == "Claim A [S1]. Claim B."
    assert report["removed_invalid_markers"] == ["S3", "S4"]
    assert report["preserved_valid_markers_in_mixed_groups"] == ["S1"]


def test_demo_turn_controller_sanitizes_evidence_scope_meta_note() -> None:
    controller = DemoTurnController(
        llm_port=_RepairLLM(),  # type: ignore[arg-type]
        stream_manager=_FakeStreamManager(),
        context_service=_FakeContextService(),  # type: ignore[arg-type]
        memory_service=SimpleNamespace(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        llm_max_tokens=256,
    )

    cleaned = controller._sanitize_visible_answer(
        "- Rural transport barriers were discussed. [S1]\n\n"
        "Evidence is limited to Powys and Torfaen; broader Welsh trends remain unspecified."
    )

    assert cleaned == "- Rural transport barriers were discussed. [S1]"
