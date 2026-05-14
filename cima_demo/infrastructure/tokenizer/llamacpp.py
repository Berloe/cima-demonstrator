from __future__ import annotations

"""Exact tokenization helpers backed by the same llama.cpp server used for inference.

The goal is to avoid heuristic token accounting when the runtime model is a GGUF
served by llama.cpp-compatible endpoints. The client uses the documented
`/apply-template` and `/tokenize` endpoints whenever available.
"""

from typing import Any

import httpx

from cima_demo.domain.entities import LLMMessage


class LlamaCppTokenizerClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._async_client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._sync_client = httpx.Client(base_url=self._base_url, timeout=timeout)

    @staticmethod
    def _messages_to_dicts(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            item: dict[str, Any] = {"role": msg.role, "content": msg.content_parts if msg.content_parts else msg.content}
            if msg.name:
                item["name"] = msg.name
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                item["tool_calls"] = msg.tool_calls
            out.append(item)
        return out

    async def aclose(self) -> None:
        await self._async_client.aclose()
        self._sync_client.close()

    async def apply_chat_template(self, messages: list[LLMMessage], *, add_generation_prompt: bool = True) -> str:
        resp = await self._async_client.post(
            "/apply-template",
            json={
                "messages": self._messages_to_dicts(messages),
                "add_generation_prompt": add_generation_prompt,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("llama.cpp /apply-template did not return a prompt string")
        return prompt

    async def tokenize_text(self, text: str) -> list[int]:
        resp = await self._async_client.post("/tokenize", json={"content": text})
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("llama.cpp /tokenize did not return a token list")
        return [int(tok) for tok in tokens]

    async def count_text_tokens(self, text: str) -> int:
        return len(await self.tokenize_text(text))

    async def count_chat_tokens(self, messages: list[LLMMessage], *, add_generation_prompt: bool = True) -> int:
        prompt = await self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        return await self.count_text_tokens(prompt)

    def count_text_tokens_sync(self, text: str) -> int:
        resp = self._sync_client.post("/tokenize", json={"content": text})
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("llama.cpp /tokenize did not return a token list")
        return len(tokens)
