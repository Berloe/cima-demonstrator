from __future__ import annotations

import json

import httpx
import pytest

from cima_demo.domain.entities import LLMMessage
from cima_demo.domain.value_objects import LLMEventType
from cima_demo.infrastructure.llm.openai_chat import OpenAIChatAdapter


@pytest.mark.asyncio
async def test_openai_complete_uses_bearer_auth_and_no_llamacpp_params() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["client_request_id"] = request.headers.get("x-client-request-id")
        payload = json.loads(request.content.decode())
        seen["payload"] = payload
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    adapter = OpenAIChatAdapter(api_key="sk-test", model="gpt-test")
    await adapter._client.aclose()  # type: ignore[attr-defined]
    adapter._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url="https://api.openai.test",
        transport=httpx.MockTransport(handler),
        headers=adapter._headers(),  # type: ignore[attr-defined]
    )

    result = await adapter.complete([LLMMessage(role="user", content="hello")], max_tokens=12)

    assert result == "ok"
    assert seen["authorization"] == "Bearer sk-test"
    assert seen["client_request_id"]
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-test"
    assert payload["max_completion_tokens"] == 12
    assert "cache_prompt" not in payload
    assert "repeat_penalty" not in payload


@pytest.mark.asyncio
async def test_openai_stream_chat_yields_tokens_and_done() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = b"".join([
            b'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ])
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = OpenAIChatAdapter(api_key="sk-test", model="gpt-test")
    await adapter._client.aclose()  # type: ignore[attr-defined]
    adapter._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url="https://api.openai.test",
        transport=httpx.MockTransport(handler),
        headers=adapter._headers(),  # type: ignore[attr-defined]
    )

    events = [event async for event in adapter.stream_chat([LLMMessage(role="user", content="hello")])]

    assert [event.token for event in events if event.type == LLMEventType.TOKEN] == ["hel", "lo"]
    assert events[-1].type == LLMEventType.DONE
