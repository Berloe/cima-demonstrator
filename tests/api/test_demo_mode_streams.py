from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cima_demo.api.routers.chat import list_models, _stream_kima, _stream_openai
from cima_demo.api.settings import Settings
from cima_demo.api.settings import Settings as AliasSettings
from cima_demo.branding import PUBLIC_MODEL_ID, PUBLIC_OWNER
from cima_demo.domain.entities import KimaDelta
from cima_demo.domain.value_objects import KimaDeltaType


class _FakeStreamManager:
    def __init__(self, deltas: list[KimaDelta]):
        self._deltas = deltas

    def subscribe(self, _conversation_id: str):
        async def _gen():
            for delta in self._deltas:
                yield delta
        return _gen()


@pytest.mark.asyncio
async def test_list_models_uses_cima_demo_branding() -> None:
    payload = await list_models(None)
    assert payload["data"][0]["id"] == PUBLIC_MODEL_ID
    assert payload["data"][0]["owned_by"] == PUBLIC_OWNER


@pytest.mark.asyncio
async def test_demo_mode_native_sse_filters_internal_deltas() -> None:
    deltas = [
        KimaDelta(type=KimaDeltaType.REASONING, conversation_id="conv-1", token="hidden"),
        KimaDelta(type=KimaDeltaType.THOUGHT, conversation_id="conv-1", tool_name="web_search", thought='{"q": "x"}'),
        KimaDelta(type=KimaDeltaType.TOKEN, conversation_id="conv-1", token="visible"),
        KimaDelta(type=KimaDeltaType.TOOL_RESULT, conversation_id="conv-1", tool_name="web_search", tool_result="done"),
        KimaDelta(type=KimaDeltaType.DONE, conversation_id="conv-1"),
    ]
    stream_manager = _FakeStreamManager(deltas)
    orchestrator = AsyncMock()

    with patch("cima_demo.api.routers.chat.get_settings", return_value=SimpleNamespace(demo_mode=True)):
        chunks = [
            chunk
            async for chunk in _stream_kima(
                conversation_id="conv-1",
                user_message="hello",
                file_data=None,
                orchestrator=orchestrator,
                stream_manager=stream_manager,
            )
        ]

    joined = "".join(chunks)
    assert "event: TOKEN" in joined
    assert "visible" in joined
    assert "event: REASONING" not in joined
    assert "event: THOUGHT" not in joined
    assert "event: TOOL_RESULT" not in joined


@pytest.mark.asyncio
async def test_demo_mode_openai_stream_hides_reasoning_and_tool_traces() -> None:
    deltas = [
        KimaDelta(type=KimaDeltaType.REASONING, conversation_id="conv-2", token="hidden"),
        KimaDelta(type=KimaDeltaType.THOUGHT, conversation_id="conv-2", tool_name="web_search", thought='{"q": "x"}'),
        KimaDelta(type=KimaDeltaType.TOOL_RESULT, conversation_id="conv-2", tool_name="web_search", tool_result="done"),
        KimaDelta(type=KimaDeltaType.TOKEN, conversation_id="conv-2", token="visible"),
        KimaDelta(type=KimaDeltaType.DONE, conversation_id="conv-2"),
    ]
    stream_manager = _FakeStreamManager(deltas)
    orchestrator = AsyncMock()

    with patch("cima_demo.api.routers.chat.get_settings", return_value=SimpleNamespace(demo_mode=True)):
        chunks = [
            chunk
            async for chunk in _stream_openai(
                conversation_id="conv-2",
                user_message="hello",
                attached_files=None,
                chatcmpl_id="chatcmpl-test",
                created_ts=1,
                orchestrator=orchestrator,
                stream_manager=stream_manager,
            )
        ]

    joined = "".join(chunks)
    assert "\"reasoning\"" not in joined
    assert "visible" in joined
    assert "web_search" not in joined


def test_settings_accept_legacy_kima_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIMA_DEMO_LOG_LEVEL", raising=False)
    monkeypatch.setenv("KIMA_LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.log_level == "DEBUG"


def test_cima_demo_package_alias_exposes_settings() -> None:
    settings = AliasSettings()
    assert settings.demo_mode is True
