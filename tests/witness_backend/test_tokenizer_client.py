from __future__ import annotations

import httpx
import pytest

from cima_demo.domain.entities import LLMMessage
from cima_demo.infrastructure.tokenizer import LlamaCppTokenizerClient


@pytest.mark.asyncio
async def test_llamacpp_tokenizer_client_counts_chat_tokens_exactly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<s>[INST] hello [/INST]"})
        if request.url.path == "/tokenize":
            payload = request.read().decode()
            assert "hello" in payload
            return httpx.Response(200, json={"tokens": [1, 2, 3, 4]})
        raise AssertionError(f"unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = LlamaCppTokenizerClient("http://llama.local")
    client._async_client = httpx.AsyncClient(base_url="http://llama.local", transport=transport)
    client._sync_client = httpx.Client(base_url="http://llama.local", transport=transport)
    try:
        count = await client.count_chat_tokens([LLMMessage(role="user", content="hello")])
        assert count == 4
        assert client.count_text_tokens_sync("hello") == 4
    finally:
        await client.aclose()
