"""Tests for non-streaming /v1/chat/completions (stream=false).

Invariants verified:
  - stream=false returns 200 with object=chat.completion (no 501)
  - X-Conversation-Id header is present in the response
  - Only TOKEN delta.content is accumulated; reasoning/thought are ignored
  - A failing turn (exception from _stream_openai) raises HTTP 500, not silent 200
  - Lock is acquired before the stream and released by the orchestrator's finally block
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from cima_demo.api.routers.chat import (
    OpenAICompletionRequest,
    _openai_completions_nonstreaming,
)
from cima_demo.branding import PUBLIC_MODEL_ID


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db(turn_lock_ok: bool = True) -> AsyncMock:
    db = AsyncMock()
    db.create_conversation = AsyncMock()
    db.get_conversation = AsyncMock(return_value={"conversation_id": "conv", "status": "ACTIVE"})
    db.try_set_turn_in_progress = AsyncMock(return_value=turn_lock_ok)
    db.release_turn_in_progress = AsyncMock()
    return db


def _sse_chunk(content: str) -> str:
    data = {
        "id": "cmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "cima_demo",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data)}\n\n"


def _sse_reasoning(text: str) -> str:
    data = {
        "id": "cmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "cima_demo",
        "choices": [{"index": 0, "delta": {"reasoning": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data)}\n\n"


async def _fake_stream_ok(*args, **kwargs):
    yield _sse_chunk("Hello")
    yield _sse_chunk(", world!")
    yield _sse_reasoning("some internal reasoning")
    yield "data: [DONE]\n\n"


async def _fake_stream_error(*args, **kwargs):
    raise RuntimeError("LLM unavailable")
    yield  # make it an async generator


async def _fake_stream_empty(*args, **kwargs):
    yield "data: [DONE]\n\n"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNonStreamingEndpoint:
    """test_openai_nonstream_returns_chat_completion"""

    @pytest.mark.asyncio
    async def test_returns_chat_completion_object(self):
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        with patch("cima_demo.api.routers.chat._stream_openai", _fake_stream_ok):
            result = await _openai_completions_nonstreaming(
                body=body,
                x_conversation_id=None,
                orchestrator=MagicMock(),
                stream_manager=MagicMock(),
                db=_make_db(),
            )

        assert isinstance(result, JSONResponse)
        data = json.loads(result.body)
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello, world!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["model"] == PUBLIC_MODEL_ID

    """test_openai_nonstream_no_501"""

    @pytest.mark.asyncio
    async def test_no_501_when_stream_false(self):
        # The 501 guard is gone; we should never raise 501 for stream=false.
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
        )
        with patch("cima_demo.api.routers.chat._stream_openai", _fake_stream_ok):
            result = await _openai_completions_nonstreaming(
                body=body,
                x_conversation_id=None,
                orchestrator=MagicMock(),
                stream_manager=MagicMock(),
                db=_make_db(),
            )
        assert result.status_code == 200

    """test_conversation_id_in_header"""

    @pytest.mark.asyncio
    async def test_conversation_id_returned_in_header(self):
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            conversation_id="conv-fixed-id",
        )
        with patch("cima_demo.api.routers.chat._stream_openai", _fake_stream_ok):
            result = await _openai_completions_nonstreaming(
                body=body,
                x_conversation_id=None,
                orchestrator=MagicMock(),
                stream_manager=MagicMock(),
                db=_make_db(),
            )

        assert result.headers.get("x-conversation-id") == "conv-fixed-id"
        data = json.loads(result.body)
        assert data["conversation_id"] == "conv-fixed-id"

    """test_real_turn_error_not_silent_200"""

    @pytest.mark.asyncio
    async def test_stream_exception_raises_http_500(self):
        """If the stream raises, we must NOT return 200 with empty content."""
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        with patch("cima_demo.api.routers.chat._stream_openai", _fake_stream_error):
            with pytest.raises(HTTPException) as exc_info:
                await _openai_completions_nonstreaming(
                    body=body,
                    x_conversation_id=None,
                    orchestrator=MagicMock(),
                    stream_manager=MagicMock(),
                    db=_make_db(),
                )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_reasoning_tokens_not_included_in_content(self):
        """delta.reasoning must not bleed into the accumulated content."""
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "x"}],
            stream=False,
        )
        with patch("cima_demo.api.routers.chat._stream_openai", _fake_stream_ok):
            result = await _openai_completions_nonstreaming(
                body=body, x_conversation_id=None,
                orchestrator=MagicMock(), stream_manager=MagicMock(),
                db=_make_db(),
            )
        data = json.loads(result.body)
        assert "some internal reasoning" not in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_lock_not_acquired_for_existing_conversation_returns_409(self):
        """409 when an existing conversation is already running a turn."""
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "x"}],
            stream=False,
            conversation_id="conv-existing",
        )
        with pytest.raises(HTTPException) as exc_info:
            await _openai_completions_nonstreaming(
                body=body, x_conversation_id=None,
                orchestrator=MagicMock(), stream_manager=MagicMock(),
                db=_make_db(turn_lock_ok=False),
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty_content(self):
        """Empty stream → 200 with empty content (valid: turn may produce no text)."""
        body = OpenAICompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        with patch("cima_demo.api.routers.chat._stream_openai", _fake_stream_empty):
            result = await _openai_completions_nonstreaming(
                body=body, x_conversation_id=None,
                orchestrator=MagicMock(), stream_manager=MagicMock(),
                db=_make_db(),
            )
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["choices"][0]["message"]["content"] == ""
