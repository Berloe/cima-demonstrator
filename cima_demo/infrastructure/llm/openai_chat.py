"""OpenAI Chat Completions adapter for the CIMA Demonstrator LLMPort.

This adapter intentionally keeps the same domain-facing contract as the
llama.cpp adapter, but only sends parameters accepted by the OpenAI API.  It is
mainly a publication/evaluation bridge so open-scenario runs can compare the
same CIMA memory/context substrate against hosted OpenAI models.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from cima_demo.domain.entities import LLMEvent, LLMMessage
from cima_demo.domain.errors import LLMContextOverflowError, LLMUnavailableError
from cima_demo.domain.ports import LLMPort
from cima_demo.domain.value_objects import LLMEventType

log = logging.getLogger(__name__)


class OpenAIChatAdapter(LLMPort):
    """LLMPort implementation backed by OpenAI's Chat Completions API.

    Notes:
    - CIMA's current runtime speaks an OpenAI-chat-compatible internal message
      protocol.  The Responses API is the preferred public API for new OpenAI
      apps, but the chat adapter is a lower-risk compatibility layer for this
      demonstrator because it preserves existing message/tool semantics.
    - No llama.cpp-only options are sent (`cache_prompt`, `repeat_penalty`,
      assistant-prefill hacks).  This avoids 400s against the hosted API.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com",
        organization: str | None = None,
        project: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 4,
        retry_delay_base: float = 1.0,
        retry_delay_max: float = 20.0,
        debug_trace: bool = False,
        debug_trace_max_chars: int = 50000,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CIMA_DEMO_OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAIChatAdapter requires OPENAI_API_KEY or CIMA_DEMO_OPENAI_API_KEY")
        self._api_key = resolved_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._organization = organization or os.getenv("OPENAI_ORGANIZATION") or os.getenv("CIMA_DEMO_OPENAI_ORGANIZATION")
        self._project = project or os.getenv("OPENAI_PROJECT") or os.getenv("CIMA_DEMO_OPENAI_PROJECT")
        self._max_retries = max(0, int(max_retries))
        self._retry_delay_base = float(retry_delay_base)
        self._retry_delay_max = float(retry_delay_max)
        self._debug_trace = bool(debug_trace)
        self._debug_trace_max_chars = max(1000, int(debug_trace_max_chars))
        connect_timeout = min(30.0, float(timeout))
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout, connect=connect_timeout, read=timeout, write=timeout, pool=connect_timeout),
        )
        self._active_response: httpx.Response | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        if self._project:
            headers["OpenAI-Project"] = self._project
        return headers

    @staticmethod
    def _messages_to_dicts(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.role
            # Chat Completions supports system/user/assistant/tool.  Some newer
            # APIs use developer; if the caller ever provides it, pass it through.
            d: dict[str, Any] = {
                "role": role,
                "content": m.content_parts if m.content_parts else (m.content or ""),
            }
            if m.name:
                d["name"] = m.name
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            out.append(d)
        return out

    def _message_debug_summary(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            content = message.content or ""
            summary.append({
                "index": idx,
                "role": message.role,
                "chars": len(content),
                "preview": content[: min(len(content), self._debug_trace_max_chars)],
                "truncated": len(content) > self._debug_trace_max_chars,
                "has_content_parts": bool(message.content_parts),
            })
        return summary

    def abort(self) -> None:
        resp = self._active_response
        if resp is not None:
            self._active_response = None
            asyncio.ensure_future(resp.aclose())

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "adapter": self.__class__.__name__,
            "base_url": self._base_url,
            "model_requested": self._model,
            "model_resolved": self._model,
            "organization_configured": bool(self._organization),
            "project_configured": bool(self._project),
            "max_retries": self._max_retries,
            "retry_delay_base": self._retry_delay_base,
            "retry_delay_max": self._retry_delay_max,
        }

    def _retry_wait(self, attempt: int) -> float:
        return min(self._retry_delay_base * (2.0 ** attempt), self._retry_delay_max)

    def _base_payload(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages_to_dicts(messages),
            "stream": bool(stream),
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            # OpenAI's newer chat-compatible models accept max_completion_tokens.
            payload["max_completion_tokens"] = int(max_tokens)
        if tools:
            payload["tools"] = tools
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    async def _post_chat_with_fallbacks(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with small compatibility fallbacks for model-specific params."""
        attempt_payloads: list[dict[str, Any]] = [payload]
        if "max_completion_tokens" in payload:
            alt = dict(payload)
            alt["max_tokens"] = alt.pop("max_completion_tokens")
            attempt_payloads.append(alt)
        # Some reasoning-family endpoints may reject explicit temperature/top_p.
        stripped = dict(payload)
        stripped.pop("temperature", None)
        stripped.pop("top_p", None)
        attempt_payloads.append(stripped)

        last_resp: httpx.Response | None = None
        seen_serialized: set[str] = set()
        for p in attempt_payloads:
            key = json.dumps(p, sort_keys=True, default=str)
            if key in seen_serialized:
                continue
            seen_serialized.add(key)
            headers = {"X-Client-Request-Id": str(uuid.uuid4())}
            resp = await self._client.post("/v1/chat/completions", json=p, headers=headers)
            last_resp = resp
            if resp.status_code < 400:
                return resp
            if resp.status_code == 413:
                raise LLMContextOverflowError("Prompt too large (413)")
            if resp.status_code >= 500:
                return resp
            body = resp.text[:1200]
            lowered = body.lower()
            # Continue only for parameter-shape errors that the alternate payloads may fix.
            if not any(term in lowered for term in ("unsupported", "unknown parameter", "max_completion_tokens", "max_tokens", "temperature", "top_p")):
                return resp
        assert last_resp is not None
        return last_resp

    async def stream_chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int | None = None,
        prefill_response: bool = False,
    ) -> AsyncGenerator[LLMEvent, None]:
        del repeat_penalty, prefill_response  # OpenAI API does not support llama.cpp-specific controls here.
        payload = self._base_payload(
            messages=messages,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )
        if self._debug_trace:
            approx_chars = sum(len(m.content or "") for m in messages)
            log.info(
                "OpenAI stream_chat request model=%s messages=%d approx_prompt_chars=%d max_tokens=%s temperature=%.2f",
                self._model, len(messages), approx_chars, max_tokens, temperature,
            )
            log.debug("OpenAI stream_chat message summaries: %s", json.dumps(self._message_debug_summary(messages), ensure_ascii=False))

        for attempt in range(self._max_retries + 1):
            yielded_any = False
            pending_tool_calls: dict[int, dict[str, str]] = {}
            try:
                headers = {"X-Client-Request-Id": str(uuid.uuid4())}
                async with self._client.stream("POST", "/v1/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status_code == 413:
                        raise LLMContextOverflowError("Prompt too large (413)")
                    if resp.status_code >= 500:
                        body = await resp.aread()
                        log.error("OpenAI %d — body: %s", resp.status_code, body.decode(errors="replace")[:2000])
                        raise LLMUnavailableError(f"OpenAI returned {resp.status_code}")
                    if 400 <= resp.status_code < 500:
                        body = await resp.aread()
                        raise LLMUnavailableError(f"OpenAI returned {resp.status_code}: {body.decode(errors='replace')[:500]}")
                    self._active_response = resp
                    try:
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                yield LLMEvent(type=LLMEventType.DONE)
                                return
                            try:
                                chunk = json.loads(data)
                            except Exception:
                                continue
                            choice = (chunk.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            for tc in delta.get("tool_calls") or []:
                                index = int(tc.get("index", 0))
                                buf = pending_tool_calls.setdefault(index, {"id": "", "name": "", "args": ""})
                                if tc.get("id"):
                                    buf["id"] = tc.get("id") or ""
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    buf["name"] += fn.get("name") or ""
                                if fn.get("arguments"):
                                    buf["args"] += fn.get("arguments") or ""
                            content = delta.get("content") or ""
                            if content:
                                yielded_any = True
                                yield LLMEvent(type=LLMEventType.TOKEN, token=content)
                            finish = choice.get("finish_reason")
                            if finish == "tool_calls":
                                for tc in pending_tool_calls.values():
                                    yield LLMEvent(
                                        type=LLMEventType.TOOL_CALL,
                                        tool_name=tc.get("name") or None,
                                        tool_args=tc.get("args") or None,
                                        tool_call_id=tc.get("id") or None,
                                    )
                                pending_tool_calls.clear()
                            if finish is not None:
                                yield LLMEvent(type=LLMEventType.DONE, truncated=(finish == "length"))
                                return
                    finally:
                        self._active_response = None
                return
            except LLMContextOverflowError:
                raise
            except (LLMUnavailableError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if yielded_any:
                    raise LLMUnavailableError(f"OpenAI failed mid-stream: {exc}") from exc
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_wait(attempt))
                    continue
                raise LLMUnavailableError(f"OpenAI unreachable after {self._max_retries + 1} attempts: {exc}") from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(f"OpenAI timeout: {exc}") from exc

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload = self._base_payload(
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            stream=False,
            response_format=response_format,
        )
        if self._debug_trace:
            approx_chars = sum(len(m.content or "") for m in messages)
            log.info(
                "OpenAI complete request model=%s messages=%d approx_prompt_chars=%d max_tokens=%s temperature=%.2f response_format=%s",
                self._model, len(messages), approx_chars, max_tokens, temperature, bool(response_format),
            )
        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = await self._post_chat_with_fallbacks(payload)
                if resp.status_code >= 500:
                    raise LLMUnavailableError(f"OpenAI returned {resp.status_code}")
                if resp.status_code == 413:
                    raise LLMContextOverflowError("Prompt too large (413)")
                resp.raise_for_status()
                data = resp.json()
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                log.debug("OpenAI complete response elapsed=%.1fs chars=%d", time.monotonic() - t0, len(content))
                return content
            except LLMContextOverflowError:
                raise
            except (LLMUnavailableError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_wait(attempt))
                    continue
                raise LLMUnavailableError(f"OpenAI complete unavailable after {self._max_retries + 1} attempts: {exc}") from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(f"OpenAI timeout: {exc}") from exc
        raise LLMUnavailableError("OpenAI complete: retry loop exhausted")

    async def count_tokens(self, text: str) -> int:
        # Avoid adding a hard tiktoken dependency to the demonstrator package.
        # This is intentionally conservative for budget assembly; exact billing
        # usage remains available from OpenAI response metadata when exposed by
        # the outer API path.
        return max(1, (len(text) + 3) // 4)

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/v1/models", timeout=10.0)
            return 200 <= resp.status_code < 300
        except Exception:
            return False
