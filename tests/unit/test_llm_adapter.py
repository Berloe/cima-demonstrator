"""Unit tests for LlamaCppAdapter._parse_sse (cima_demo/infrastructure/llm/adapter.py).

All tests are offline — no HTTP server required. SSE lines are synthesised
directly and fed into the parser's async generator.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from cima_demo.domain.value_objects import LLMEventType
from cima_demo.infrastructure.llm.llamacpp import LlamaCppAdapter


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeResponse:
    """Minimal mock for httpx.Response that yields pre-defined SSE lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


def _chunk(
    content: str = "",
    reasoning_content: str = "",
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> str:
    """Build a single SSE data line for a chat completion chunk."""
    delta: dict = {}
    if content:
        delta["content"] = content
    if reasoning_content:
        delta["reasoning_content"] = reasoning_content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    choice: dict = {"delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return "data: " + json.dumps({"choices": [choice]})


_DONE = "data: [DONE]"


async def _collect(lines: list[str]) -> list:
    """Run the parser and collect all emitted LLMEvents."""
    adapter = LlamaCppAdapter("http://localhost:8080")
    events = []
    async for ev in adapter._parse_sse(_FakeResponse(lines)):
        events.append(ev)
    return events


def _types(events: list) -> list[LLMEventType]:
    return [e.type for e in events]


def _tokens(events: list, event_type: LLMEventType) -> str:
    return "".join(e.token or "" for e in events if e.type == event_type)


# ── Normal protocol flow ──────────────────────────────────────────────────────

class TestNormalFlow:
    async def test_think_then_response_emits_reasoning_and_token(self) -> None:
        lines = [
            _chunk("<think>\nStep one\n</think>"),
            _chunk("<response>"),
            _chunk("Hello user"),
            _chunk("</response>"),
            _chunk("<conclusions>[]</conclusions>"),
            _chunk("<phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        types = _types(events)
        assert LLMEventType.REASONING in types
        assert LLMEventType.TOKEN in types
        assert LLMEventType.CONCLUSIONS in types
        assert LLMEventType.PHASE_DECL in types
        assert LLMEventType.DONE in types

    async def test_response_content_correct(self) -> None:
        lines = [
            _chunk("<think>reasoning</think><response>"),
            _chunk("Answer text"),
            _chunk("</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        token_text = _tokens(events, LLMEventType.TOKEN)
        assert "Answer text" in token_text

    async def test_conclusions_parsed_as_list(self) -> None:
        conclusions_json = json.dumps([
            {"type": "FACT", "content": "sky is blue", "confidence": 0.9}
        ])
        lines = [
            _chunk("<response>ok</response>"),
            _chunk(f"<conclusions>{conclusions_json}</conclusions>"),
            _chunk("<phase>RECALL</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        conc_events = [e for e in events if e.type == LLMEventType.CONCLUSIONS]
        assert len(conc_events) == 1
        assert conc_events[0].conclusions[0]["content"] == "sky is blue"

    async def test_phase_decl_emitted(self) -> None:
        lines = [
            _chunk("<response>ok</response><conclusions>[]</conclusions>"),
            _chunk("<phase>PLANNING</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        phase_events = [e for e in events if e.type == LLMEventType.PHASE_DECL]
        assert len(phase_events) == 1
        assert phase_events[0].phase_decl == "PLANNING"

    async def test_done_always_last(self) -> None:
        lines = [
            _chunk("<response>ok</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        assert events[-1].type == LLMEventType.DONE


# ── Fallback: no <response> tag ───────────────────────────────────────────────

class TestFallbackNoResponseTag:
    async def test_content_without_response_tag_becomes_token(self) -> None:
        """When LLM omits <response>, buffered content is emitted as TOKEN fallback."""
        lines = [
            _chunk("This is the answer without any tags."),
            _DONE,
        ]
        events = await _collect(lines)
        token_text = _tokens(events, LLMEventType.TOKEN)
        assert "This is the answer without any tags." in token_text

    async def test_no_response_tag_no_reasoning_events_emitted(self) -> None:
        """Buffered pre-response content must become TOKEN, not REASONING."""
        lines = [
            _chunk("Plain answer"),
            _DONE,
        ]
        events = await _collect(lines)
        # No REASONING emitted (buffer flushed as TOKEN)
        assert not any(e.type == LLMEventType.REASONING for e in events)
        assert any(e.type == LLMEventType.TOKEN for e in events)

    async def test_empty_stream_still_emits_done(self) -> None:
        events = await _collect([_DONE])
        assert events[-1].type == LLMEventType.DONE


# ── Native reasoning_content (R1 / QwQ) ──────────────────────────────────────

class TestNativeReasoning:
    async def test_reasoning_content_emits_reasoning_event(self) -> None:
        lines = [
            _chunk(reasoning_content="Step 1: think about it"),
            _chunk(reasoning_content=" Step 2: conclude"),
            _chunk("<response>Final answer</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        reasoning_text = _tokens(events, LLMEventType.REASONING)
        assert "Step 1" in reasoning_text
        assert "Step 2" in reasoning_text

    async def test_reasoning_content_does_not_block_response_tag(self) -> None:
        """Native reasoning tokens must not interfere with <response> detection in content."""
        lines = [
            _chunk(reasoning_content="native think"),
            _chunk("<response>visible</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        assert any(e.type == LLMEventType.TOKEN for e in events)
        token_text = _tokens(events, LLMEventType.TOKEN)
        assert "visible" in token_text

    async def test_reasoning_content_stripped_of_think_tags(self) -> None:
        lines = [
            _chunk(reasoning_content="<think>inner thought</think>"),
            _chunk("<response>ok</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        reasoning_text = _tokens(events, LLMEventType.REASONING)
        assert "<think>" not in reasoning_text
        assert "inner thought" in reasoning_text

    async def test_reasoning_and_content_in_same_chunk(self) -> None:
        """Some models emit both fields in the same delta chunk."""
        lines = [
            _chunk(reasoning_content="thinking", content="<response>"),
            _chunk(content="answer</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        assert any(e.type == LLMEventType.REASONING for e in events)
        assert any(e.type == LLMEventType.TOKEN for e in events)


# ── Think tag stripping ───────────────────────────────────────────────────────

class TestThinkTagStripping:
    async def test_think_tags_stripped_from_reasoning(self) -> None:
        lines = [
            _chunk("<think>\nmy thought\n</think>"),
            _chunk("<response>ok</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        reasoning_text = _tokens(events, LLMEventType.REASONING)
        assert "<think>" not in reasoning_text
        assert "</think>" not in reasoning_text
        assert "my thought" in reasoning_text

    async def test_thinking_and_reasoning_variants_stripped(self) -> None:
        for tag in ("<thinking>", "</thinking>", "<reasoning>", "</reasoning>"):
            lines = [
                _chunk(f"{tag}content{tag}"),
                _chunk("<response>ok</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
                _DONE,
            ]
            events = await _collect(lines)
            reasoning_text = _tokens(events, LLMEventType.REASONING)
            assert tag not in reasoning_text, f"Tag {tag!r} leaked into REASONING"


# ── Tool calls ────────────────────────────────────────────────────────────────

class TestToolCalls:
    async def test_tool_call_event_emitted(self) -> None:
        tool_call_delta = [
            {"index": 0, "id": "call-1", "function": {"name": "datetime_tool", "arguments": ""}}
        ]
        lines = [
            _chunk(tool_calls=tool_call_delta),
            _chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"action":"now"}'}}]),
            _chunk(finish_reason="tool_calls"),
            _DONE,
        ]
        events = await _collect(lines)
        tc_events = [e for e in events if e.type == LLMEventType.TOOL_CALL]
        assert len(tc_events) == 1
        assert tc_events[0].tool_name == "datetime_tool"
        assert '"action"' in (tc_events[0].tool_args or "")

    async def test_multiple_tool_calls(self) -> None:
        lines = [
            _chunk(tool_calls=[
                {"index": 0, "id": "c1", "function": {"name": "math_eval", "arguments": ""}},
                {"index": 1, "id": "c2", "function": {"name": "datetime_tool", "arguments": ""}},
            ]),
            _chunk(tool_calls=[
                {"index": 0, "function": {"arguments": '{"expression":"2+2"}'}},
                {"index": 1, "function": {"arguments": '{"action":"now"}'}},
            ]),
            _chunk(finish_reason="tool_calls"),
            _DONE,
        ]
        events = await _collect(lines)
        tc_events = [e for e in events if e.type == LLMEventType.TOOL_CALL]
        names = {e.tool_name for e in tc_events}
        assert names == {"math_eval", "datetime_tool"}


# ── Chunk boundary safety ─────────────────────────────────────────────────────

class TestChunkBoundary:
    async def test_response_tag_split_across_chunks(self) -> None:
        """<response> split at the boundary must still be detected correctly."""
        lines = [
            _chunk("before<resp"),   # partial tag
            _chunk("onse>"),         # completes the tag
            _chunk("content"),
            _chunk("</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        assert any(e.type == LLMEventType.TOKEN for e in events)
        token_text = _tokens(events, LLMEventType.TOKEN)
        assert "content" in token_text

    async def test_multi_chunk_response_content_assembled(self) -> None:
        lines = [
            _chunk("<response>"),
            _chunk("part1 "),
            _chunk("part2 "),
            _chunk("part3"),
            _chunk("</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        token_text = _tokens(events, LLMEventType.TOKEN)
        assert "part1" in token_text
        assert "part2" in token_text
        assert "part3" in token_text

    async def test_strategy_fail_tag_parsed(self) -> None:
        lines = [
            _chunk('<response><strategy_fail type="convergence">'),
            _chunk("query too vague"),
            _chunk("</strategy_fail></response>"),
            _chunk("<conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        # strategy_fail is inside <response>, so TOKEN events carry it
        token_text = _tokens(events, LLMEventType.TOKEN)
        assert "strategy_fail" in token_text or "convergence" in token_text

    async def test_invalid_json_chunk_skipped(self) -> None:
        lines = [
            "data: {broken json",
            _chunk("<response>ok</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        # Parser must not crash; DONE must still be emitted
        assert events[-1].type == LLMEventType.DONE

    async def test_non_data_lines_ignored(self) -> None:
        lines = [
            ": heartbeat",
            "event: ping",
            _chunk("<response>ok</response><conclusions>[]</conclusions><phase>IDLE</phase>"),
            _DONE,
        ]
        events = await _collect(lines)
        assert events[-1].type == LLMEventType.DONE
