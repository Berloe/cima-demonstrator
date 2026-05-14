"""LlamaCppAdapter → LLMPort (KIMA_Infrastructure_Layer_v0.6 §3.3)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx

from cima_demo.domain.entities import LLMEvent, LLMMessage
from cima_demo.domain.errors import LLMContextOverflowError, LLMUnavailableError
from cima_demo.domain.ports import LLMPort
from cima_demo.domain.value_objects import LLMEventType

log = logging.getLogger(__name__)

_RESPONSE_OPEN = "<response>"
_RESPONSE_CLOSE = "</response>"
_CONCLUSIONS_OPEN = "<conclusions>"
_CONCLUSIONS_CLOSE = "</conclusions>"
_PHASE_OPEN = "<phase>"
_PHASE_CLOSE = "</phase>"
_STRATEGY_FAIL_OPEN_PREFIX = "<strategy_fail"
_STRATEGY_FAIL_CLOSE = "</strategy_fail>"
_STRATEGY_FAIL_TYPE_RE = re.compile(r'type=["\']([^"\']+)["\']')
# Strip <think>, </think>, <thinking>, </thinking>, <reasoning>, </reasoning> from REASONING tokens
_THINK_TAG_RE = re.compile(r'</?(?:think|thinking|reasoning)>\n?', re.IGNORECASE)
# Detect think-block boundaries for visible-content filtering (fallback TOKEN path).
_THINK_OPEN_RE = re.compile(r'<(?:think|thinking|reasoning)>', re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r'</(?:think|thinking|reasoning)>', re.IGNORECASE)
_TOOL_CALL_LEAK_RE = re.compile(
    r'\[TOOL_CALLS\]|\btool_call_id\b|"function"\s*:\s*\{',
    re.IGNORECASE,
)
_JSON_TOOL_ARGS_RE = re.compile(
    r'^\s*\{.*"(?:action|url|urls|queries|code|expression)"\s*:',
    re.DOTALL | re.IGNORECASE,
)
_CONCLUSIONS_BLOCK_RE = re.compile(
    r'<conclusions>.*?</conclusions>',
    re.DOTALL | re.IGNORECASE,
)
_PROTOCOL_FALLBACK_RE = re.compile(
    r'</?(?:phase|conclusions|response|think|thinking|reasoning)[\s>]',
    re.IGNORECASE,
)
# Safe lookahead: the longest tag we must avoid splitting across chunk boundaries.
# Covers <response> (10), </reasoning> (12) and all think-family tags.
_SAFE_LOOKAHEAD = max(
    len(_RESPONSE_OPEN),          # 10
    len("</reasoning>"),           # 12
    len("</thinking>"),            # 11
)                                  # 12 — retain at least this many trailing chars so that
# the longest think-family closing tag is never split across chunk boundaries


def _extract_visible(text: str, in_think: bool) -> tuple[str, bool]:
    """Return (visible_text, ends_in_think) for fallback TOKEN filtering.

    Strips text that falls inside <think>/<thinking>/<reasoning> blocks.
    Content between think tags is internal reasoning — must not be shown
    to the user if <response> was never emitted.
    """
    out: list[str] = []
    rest = text
    while rest:
        if in_think:
            m = _THINK_CLOSE_RE.search(rest)
            if m:
                rest = rest[m.end():]
                in_think = False
            else:
                break  # entire remainder is inside a think block
        else:
            m = _THINK_OPEN_RE.search(rest)
            if m:
                out.append(rest[:m.start()])
                rest = rest[m.end():]
                in_think = True
            else:
                out.append(rest)
                break
    return "".join(out), in_think


class LlamaCppAdapter(LLMPort):
    """httpx-based adapter for llama.cpp OpenAI-compat API.

    Parses streaming SSE and emits LLMEvent objects:
      - REASONING:   native reasoning_content tokens (DeepSeek-R1, QwQ, enable_thinking)
                     or text before </think> in pre_response mode (tool pass only)
      - TOKEN:       visible answer text — emitted directly in synthesis pass
      - TOOL_CALL:   OpenAI tool_calls format
      - CONCLUSIONS: content inside <conclusions> (post_response; synthesis pass should not emit these)
      - PHASE_DECL:  content inside <phase>
      - DONE:        end of stream

    Output protocol (system_prompt.j2 — plain-text synthesis, no XML):
      Tool pass (Mode A):   [tool_calls only — no text]
      Synthesis pass (Mode B): plain answer text — no <response>, no <conclusions>, no XML

    Parser states:
      pre_response  — tool-pass default; text is buffered as REASONING; </think> triggers TOKEN mode
      in_response   — TOKEN mode; used when response_prefilled=True (synthesis pass or actual prefill)
      post_response — after </response>; structural parsing for <conclusions>, <phase>

    response_prefilled=True is passed for any synthesis pass (prefill_response=True), regardless of
    whether the <response> prefix was physically injected, so plain-text answers stream as TOKEN
    immediately rather than being held in REASONING until DONE.

    Retry policy (LLMUnavailableError / connection errors):
      Exponential backoff: delay = min(retry_delay_base * 2^attempt, retry_delay_max).
      Only retries when no tokens have been yielded yet (full restart from scratch).
      Retries on 5xx, ConnectError, RemoteProtocolError.
      Never retries LLMContextOverflowError (413) — won't fix itself.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "mistral",
        timeout: float = 120.0,
        max_retries: int = 12,
        retry_delay_base: float = 5.0,
        retry_delay_max: float = 120.0,
        response_prefill_enabled: bool = False,
        debug_trace: bool = False,
        debug_trace_max_chars: int = 50000,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_base = retry_delay_base
        self._retry_delay_max = retry_delay_max
        # Guard: prefill is incompatible with native-reasoning models (DeepSeek-R1, QwQ…)
        # because it suppresses their CoT phase. Callers must NOT enable this when
        # native_reasoning=True. Stored here so stream_chat can gate on it regardless
        # of what the orchestrator requests.
        self._response_prefill_enabled = response_prefill_enabled
        self._debug_trace = bool(debug_trace)
        self._debug_trace_max_chars = max(1000, int(debug_trace_max_chars))
        # Capability registry: tracks whether prefill works on this backend.
        # None = untested; True = confirmed working; False = currently disabled.
        # H-08 fix: prefill is disabled with a time-based cool-down rather than
        # permanently, so a transient 400 (server loading, OOM) doesn't permanently
        # degrade synthesis quality for the adapter lifetime.
        self._prefill_capability_confirmed: bool | None = None  # None = untested
        # Monotonic timestamp after which prefill is re-eligible if it was disabled.
        # 0.0 = never disabled (or cool-down expired).
        self._prefill_disabled_until: float = 0.0
        # Cool-down period in seconds before re-testing a disabled prefill backend.
        self._prefill_cooldown_secs: float = 600.0  # 10 minutes
        connect_timeout = min(30.0, float(timeout))
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout, read=timeout, write=timeout, pool=connect_timeout),
        )
        # Track the active streaming response so abort() can close it immediately.
        self._active_response: httpx.Response | None = None

    def abort(self) -> None:
        """Close the active HTTP stream immediately.

        Called fire-and-forget from the orchestrator's finally block when the
        client disconnects (stop button pressed).  Schedules aclose() as an
        asyncio task — does not await, so it is safe to call from a finally
        block even during CancelledError propagation.
        """
        resp = self._active_response
        if resp is not None:
            self._active_response = None
            asyncio.ensure_future(resp.aclose())
            log.debug("LlamaCppAdapter.abort(): closing active stream")

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": "llamacpp",
            "adapter": self.__class__.__name__,
            "base_url": self._base_url,
            "model_requested": self._model,
            "model_resolved": self._model,
            "max_retries": self._max_retries,
            "response_prefill_enabled": self._response_prefill_enabled,
            "prefill_capability_confirmed": self._prefill_capability_confirmed,
        }

    def _retry_wait(self, attempt: int) -> float:
        """Exponential backoff: base * 2^attempt, capped at max."""
        return min(self._retry_delay_base * (2.0 ** attempt), self._retry_delay_max)

    @staticmethod
    def _messages_to_dicts(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        result = []
        for m in messages:
            # Use content_parts (multimodal) when available, otherwise plain text content.
            d: dict[str, Any] = {"role": m.role, "content": m.content_parts if m.content_parts else m.content}
            if m.name:
                d["name"] = m.name
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            result.append(d)
        return result

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
        msg_dicts = self._messages_to_dicts(messages)
        # Apply prefill only if:
        #   1. orchestrator requested it
        #   2. adapter was configured with response_prefill_enabled=True
        #   3. capability cool-down has not disabled it (H-08: time-based, not permanent)
        _now_mono = time.monotonic()
        if self._prefill_capability_confirmed is False and _now_mono >= self._prefill_disabled_until:
            self._prefill_capability_confirmed = None  # re-probe after cool-down
            log.info(
                "Prefill cool-down expired for %s/%s — re-testing prefill capability.",
                self._base_url, self._model,
            )
        _prefill_disabled = self._prefill_capability_confirmed is False
        _do_prefill = (
            prefill_response
            and self._response_prefill_enabled
            and not _prefill_disabled
        )
        if _do_prefill:
            # Assistant prefill: force the model to start output after <response>,
            # eliminating pre-response preamble generation (~50s saved for Instruct models).
            # Only safe when tools=None (synthesis pass), since prefill prevents tool calls.
            msg_dicts = [*msg_dicts, {"role": "assistant", "content": "<response>\n"}]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": msg_dicts,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            # Enable KV-cache reuse for the stable prefix (system prompt + history).
            # Saves 40-120 s of prefill on subsequent iterations within a turn
            # and between turns that share the same system prompt.
            "cache_prompt": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools

        if self._debug_trace:
            approx_chars = sum(len(m.content or "") for m in messages)
            log.info(
                "LLM DEBUG stream_chat request base_url=%s model=%s messages=%d approx_prompt_chars=%d max_tokens=%s temperature=%.2f",
                self._base_url, self._model, len(messages), approx_chars, max_tokens, temperature,
            )
            log.debug("LLM DEBUG stream_chat message summaries: %s", json.dumps(self._message_debug_summary(messages), ensure_ascii=False))

        log.debug(
            "LLM stream_chat request — messages=%d, tools=%d, temperature=%.2f, "
            "top_p=%.2f, repeat_penalty=%.2f, max_tokens=%s\npayload:\n%s",
            len(messages),
            len(tools) if tools else 0,
            temperature,
            top_p,
            repeat_penalty,
            max_tokens,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

        # Retry loop — exponential backoff on model unavailability.
        # Restarts the full request from scratch each attempt.
        # Only retries when no tokens have been yielded (pre-first-token window).
        # Retriable: 5xx, ConnectError, RemoteProtocolError.
        # Non-retriable: 413 (LLMContextOverflowError), 4xx, TimeoutException.

        _prefill_fallback_used = False
        for _attempt in range(self._max_retries + 1):
            _yielded_any = False
            _resp_log: list[str] = []
            _tool_log: list[str] = []
            _t_stream_start = time.monotonic()
            _t_first_delta: float | None = None
            # On prefill-fallback attempts, strip the assistant prefill message
            _active_msg_dicts = msg_dicts[:-1] if (_do_prefill and _prefill_fallback_used) else msg_dicts
            _active_payload = {**payload, "messages": _active_msg_dicts}
            try:
                async with self._client.stream("POST", "/v1/chat/completions", json=_active_payload) as resp:
                    if resp.status_code == 413:
                        raise LLMContextOverflowError("Prompt too large (413)")
                    if resp.status_code >= 500:
                        body = await resp.aread()
                        log.error("llama.cpp %d — body: %s", resp.status_code, body.decode(errors="replace")[:2000])
                        raise LLMUnavailableError(f"LLM returned {resp.status_code}")
                    if 400 <= resp.status_code < 500:
                        body = await resp.aread()
                        body_str = body.decode(errors="replace")[:2000]
                        log.error("llama.cpp %d — body: %s", resp.status_code, body_str)
                        if resp.status_code == 400 and _do_prefill and not _prefill_fallback_used:
                            log.warning(
                                "400 on prefill attempt — retrying once without assistant prefill "
                                "(attempt %d/%d)", _attempt + 1, self._max_retries + 1,
                            )
                            _prefill_fallback_used = True
                            # H-08: disable prefill with a cool-down instead of permanently,
                            # so a transient 400 (OOM, model loading) doesn't degrade synthesis
                            # for the entire adapter lifetime.
                            if self._prefill_capability_confirmed is None:
                                self._prefill_capability_confirmed = False
                                self._prefill_disabled_until = time.monotonic() + self._prefill_cooldown_secs
                                log.warning(
                                    "Prefill capability disabled for %s/%s after 400 "
                                    "— will re-test after %.0fs cool-down.",
                                    self._base_url, self._model, self._prefill_cooldown_secs,
                                )
                            continue
                        raise LLMUnavailableError(
                            f"LLM returned {resp.status_code}: {body_str[:200]}"
                        )
                    resp.raise_for_status()

                    self._active_response = resp
                    # Record first successful prefill so we don't retry disable logic again.
                    if _do_prefill and self._prefill_capability_confirmed is None:
                        self._prefill_capability_confirmed = True
                        log.debug("Prefill capability confirmed for %s/%s", self._base_url, self._model)
                    try:
                        async for event in self._parse_sse(
                            resp,
                            # synthesis pass (prefill_response=True) always starts in TOKEN mode:
                            # the prompt forbids <response> tags, so plain text must flow directly
                            # to TOKEN regardless of whether we physically injected the prefix.
                            response_prefilled=prefill_response,
                        ):
                            _yielded_any = True
                            # Side-accumulate for response logging — yield is NOT delayed
                            if event.type == LLMEventType.TOKEN and event.token:
                                if _t_first_delta is None:
                                    _t_first_delta = time.monotonic()
                                _resp_log.append(event.token)
                            elif event.type == LLMEventType.TOOL_CALL and event.tool_name:
                                _tool_log.append(f"{event.tool_name}({event.tool_args or ''})")
                            elif event.type == LLMEventType.DONE:
                                _t_done2 = time.monotonic()
                                log.debug(
                                    "LLM stream response — first_delta=%.1fs, total=%.1fs, chars=%d\n%s",
                                    (_t_first_delta - _t_stream_start) if _t_first_delta else -1.0,
                                    _t_done2 - _t_stream_start,
                                    sum(len(t) for t in _resp_log),
                                    "".join(_resp_log),
                                )
                                if _tool_log:
                                    log.debug("LLM stream tool_calls: %s", " | ".join(_tool_log))
                            yield event
                    finally:
                        self._active_response = None
                return  # success

            except LLMContextOverflowError:
                raise  # never retry — prompt is too large
            except (LLMUnavailableError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if _yielded_any:
                    # Tokens already sent upstream — can't restart cleanly; let caller handle it
                    raise LLMUnavailableError(f"LLM failed mid-stream: {exc}") from exc
                if _attempt < self._max_retries:
                    delay = self._retry_wait(_attempt)
                    log.warning(
                        "LLM unavailable (attempt %d/%d) — retrying in %.0fs: %s",
                        _attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise LLMUnavailableError(f"LLM unreachable after {self._max_retries + 1} attempts: {exc}") from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(f"LLM timeout: {exc}") from exc

    async def _parse_sse(
        self, resp: httpx.Response, *, response_prefilled: bool = False
    ) -> AsyncIterator[LLMEvent]:
        """Parse SSE stream for the KIMA output protocol.

        Parser states:
          pre_response  — tool-pass default (response_prefilled=False); text buffered as REASONING.
                          </think> trigger switches to TOKEN mode for native reasoning models.
                          Fallback: content emitted as sanitized TOKEN at DONE if no </think> seen.
          in_response   — synthesis-pass mode (response_prefilled=True); text emitted as TOKEN directly.
          post_response — after </response>; structural parsing for <conclusions>, <phase>.

        response_prefilled=True is used for all synthesis passes so plain-text answers stream
        token-by-token as TOKEN events rather than accumulating in REASONING until DONE.
        The name reflects the historical prefill-injection use; the semantic is now "synthesis mode".
        """
        pre_response = not response_prefilled
        in_response = response_prefilled
        in_post_think = False   # <think> block that appears AFTER </response>
        in_conclusions = False
        in_phase = False
        in_strategy_fail = False
        post_think_buf: list[str] = []
        conclusions_buf: list[str] = []
        phase_buf: list[str] = []
        strategy_fail_buf: list[str] = []
        strategy_fail_type: str | None = None
        pending_tool_calls: dict[int, dict[str, Any]] = {}
        buf = ""
        _raw_chars = 0
        _response_tag_seen = response_prefilled
        # Buffer for pre-response REASONING; flushed when <response> found or at DONE fallback.
        _reasoning_accum: list[str] = []
        # Fallback-visible buffer: only non-think content accumulated for TOKEN fallback.
        # Tracks think-block state across streaming chunks so we never emit reasoning
        # that was inside <think> blocks as user-visible TOKEN when <response> is absent.
        _visible_accum: list[str] = []
        _in_think_block: bool = False
        # True when at least one delta.reasoning_content chunk has been received.
        # Set by native-reasoning models (DeepSeek-R1, QwQ, Llama-3.3-thinking via llama.cpp).
        _native_reasoning_seen = False
        # Timing
        _t_start = time.monotonic()
        _t_first_token: float | None = None
        # When response is prefilled, tag was "seen" at t=0
        _t_response_tag: float | None = _t_start if response_prefilled else None
        _total_chars = 0
        _finish_reason: str | None = None  # last non-null finish_reason from any chunk

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                _t_done = time.monotonic()
                log.info(
                    "LLM stream DONE — total=%.1fs, TTFT=%.1fs, response_tag_at=%.1fs, "
                    "chars=%d, response_tag_seen=%s",
                    _t_done - _t_start,
                    (_t_first_token - _t_start) if _t_first_token else -1.0,
                    (_t_response_tag - _t_start) if _t_response_tag else -1.0,
                    _total_chars,
                    _response_tag_seen,
                )
                if buf:
                    if pre_response:
                        # Remaining buf at DONE without <response>: accumulate for
                        # fallback TOKEN only — do NOT emit REASONING.  If <response>
                        # was never seen, this content becomes user-visible TOKEN; it
                        # must not also appear in the thinking panel as REASONING.
                        visible, _in_think_block = _extract_visible(buf, _in_think_block)
                        if visible.strip():
                            _visible_accum.append(visible)
                        clean = _THINK_TAG_RE.sub("", buf)
                        if clean:
                            _reasoning_accum.append(clean)  # triggers fallback logic below
                    elif in_response:
                        yield LLMEvent(type=LLMEventType.TOKEN, token=buf)
                if pre_response and (_reasoning_accum or _visible_accum):
                    # No <response> tag seen — plain-text synthesis (item 9).
                    # Emit all visible non-think content as TOKEN, but sanitize
                    # protocol debris first so raw tool syntax is never surfaced.
                    fallback_text = "".join(_visible_accum).strip()
                    if not fallback_text and _reasoning_accum:
                        # Entire output was inside think blocks — strip tags and use as fallback.
                        fallback_text = _THINK_TAG_RE.sub("", "".join(_reasoning_accum)).strip()
                    if fallback_text:
                        fallback_text = _CONCLUSIONS_BLOCK_RE.sub("", fallback_text)
                        fallback_text = _PROTOCOL_FALLBACK_RE.sub("", fallback_text)
                        fallback_text = fallback_text.replace(_RESPONSE_OPEN, "").replace(_RESPONSE_CLOSE, "").strip()
                    log.debug(
                        "LLM stream plain-text synthesis — no <response> tag "
                        "(%d reasoning chars, %d visible chars, native=%s)",
                        sum(len(c) for c in _reasoning_accum),
                        len(fallback_text),
                        _native_reasoning_seen,
                    )
                    if fallback_text and not _TOOL_CALL_LEAK_RE.search(fallback_text) and not _JSON_TOOL_ARGS_RE.match(fallback_text):
                        yield LLMEvent(type=LLMEventType.TOKEN, token=fallback_text)
                    elif fallback_text:
                        log.warning("Suppressing plain-text fallback that still contains tool/protocol syntax")
                yield LLMEvent(
                    type=LLMEventType.DONE,
                    truncated=(_finish_reason == "length"),
                )
                return

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})

            # Track time-to-first-token (any content: text or tool_calls)
            if _t_first_token is None and (
                delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
            ):
                _t_first_token = time.monotonic()
                log.info("LLM TTFT=%.1fs", _t_first_token - _t_start)

            # Tool calls (OpenAI format)
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in pending_tool_calls:
                    pending_tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "args": "",
                    }
                pending_tool_calls[idx]["args"] += tc.get("function", {}).get("arguments", "")

            # Native reasoning_content (DeepSeek-R1, QwQ, Llama-3.3-thinking via llama.cpp ≥b4450).
            # These tokens are definitively the model's native CoT — stream as REASONING immediately,
            # no buffering required since they cannot contain the <response> tag.
            native_tok = delta.get("reasoning_content") or ""
            if native_tok:
                clean = _THINK_TAG_RE.sub("", native_tok)
                if clean:
                    _native_reasoning_seen = True
                    yield LLMEvent(type=LLMEventType.REASONING, token=clean)

            finish = choice.get("finish_reason")
            if finish:
                _finish_reason = finish  # track last non-null finish_reason
            if finish == "tool_calls":
                for tc in pending_tool_calls.values():
                    yield LLMEvent(
                        type=LLMEventType.TOOL_CALL,
                        tool_name=tc["name"],
                        tool_args=tc["args"],
                        tool_call_id=tc["id"],
                    )
                pending_tool_calls.clear()
                continue

            token = delta.get("content") or ""
            if not token:
                continue

            _total_chars += len(token)

            # Diagnostic: log first 300 raw chars once
            if _raw_chars < 300:
                sample = token[: 300 - _raw_chars]
                _raw_chars += len(sample)
                if _raw_chars >= 300:
                    log.info("LLM raw output (first 300 chars): %r", buf + sample)

            buf += token

            # Process buf — loop until no more complete structural elements remain
            while buf:
                if pre_response:
                    # Determine which comes first in buf: <response> or </think> close.
                    # </think> close is a defensive fallback for models that write <think>
                    # before <response> (old schema) and then produce content after </think>.
                    response_pos = buf.find(_RESPONSE_OPEN)
                    think_close_m = _THINK_CLOSE_RE.search(buf)
                    think_close_pos = think_close_m.start() if think_close_m else -1

                    use_think_close = (
                        think_close_pos != -1
                        and (response_pos == -1 or think_close_pos < response_pos)
                    )

                    if response_pos != -1 and not use_think_close:
                        if not _response_tag_seen:
                            _response_tag_seen = True
                            _t_response_tag = time.monotonic()
                            log.info(
                                "LLM <response> tag found — reasoning_chars=%d, elapsed=%.1fs",
                                _total_chars,
                                _t_response_tag - _t_start,
                            )
                        before, _, buf = buf.partition(_RESPONSE_OPEN)
                        if before:
                            clean = _THINK_TAG_RE.sub("", before)
                            if clean:
                                _reasoning_accum.append(clean)
                                yield LLMEvent(type=LLMEventType.REASONING, token=clean)
                            visible, _in_think_block = _extract_visible(before, _in_think_block)
                            if visible.strip():
                                _visible_accum.append(visible)
                        _reasoning_accum = []
                        _visible_accum = []
                        pre_response = False
                        in_response = True
                    elif use_think_close:
                        # </think> seen before <response>: treat everything after as TOKEN.
                        # Emit content before </think> as REASONING, then switch to TOKEN mode.
                        before = buf[:think_close_pos]
                        buf = buf[think_close_m.end():]
                        if before:
                            clean = _THINK_TAG_RE.sub("", before)
                            if clean:
                                _reasoning_accum.append(clean)
                                yield LLMEvent(type=LLMEventType.REASONING, token=clean)
                        _reasoning_accum = []
                        _visible_accum = []
                        pre_response = False
                        in_response = True
                        log.debug(
                            "LLM </think> used as TOKEN trigger (no <response> seen before it)"
                        )
                    else:
                        # Buffer only the trailing partial that could be the start
                        # of <response> or </think> split across chunks.
                        safe_len = len(buf) - _SAFE_LOOKAHEAD
                        if safe_len > 0:
                            chunk = buf[:safe_len]
                            clean = _THINK_TAG_RE.sub("", chunk)
                            if clean:
                                _reasoning_accum.append(clean)
                                yield LLMEvent(type=LLMEventType.TOKEN, token=clean)
                            visible, _in_think_block = _extract_visible(chunk, _in_think_block)
                            if visible.strip():
                                _visible_accum.append(visible)
                            buf = buf[safe_len:]
                        break
                    continue

                if in_response:
                    if _RESPONSE_CLOSE in buf:
                        before, _, buf = buf.partition(_RESPONSE_CLOSE)
                        if before:
                            yield LLMEvent(type=LLMEventType.TOKEN, token=before)
                        in_response = False
                    else:
                        safe_len = len(buf) - (len(_RESPONSE_CLOSE) - 1)
                        if safe_len > 0:
                            yield LLMEvent(type=LLMEventType.TOKEN, token=buf[:safe_len])
                            buf = buf[safe_len:]
                        break
                    continue

                # post_response: <think> block that follows </response> (reasoning trace)
                if in_post_think:
                    m = _THINK_CLOSE_RE.search(buf)
                    if m:
                        before = buf[:m.start()]
                        buf = buf[m.end():]
                        if before:
                            clean = _THINK_TAG_RE.sub("", before)
                            if clean:
                                yield LLMEvent(type=LLMEventType.REASONING, token=clean)
                        in_post_think = False
                        post_think_buf = []
                    else:
                        # stream safe portion as REASONING
                        safe_len = len(buf) - _SAFE_LOOKAHEAD
                        if safe_len > 0:
                            clean = _THINK_TAG_RE.sub("", buf[:safe_len])
                            if clean:
                                yield LLMEvent(type=LLMEventType.REASONING, token=clean)
                            buf = buf[safe_len:]
                        break
                    continue

                # post_response: structural tags only
                if in_conclusions:
                    if _CONCLUSIONS_CLOSE in buf:
                        before, _, buf = buf.partition(_CONCLUSIONS_CLOSE)
                        conclusions_buf.append(before)
                        raw = "".join(conclusions_buf).strip()
                        try:
                            parsed: list[dict[str, Any]] = json.loads(raw)
                            if not isinstance(parsed, list):
                                parsed = []
                        except (json.JSONDecodeError, ValueError):
                            parsed = [
                                {"content": ln.strip(), "type": "OBSERVATION", "confidence": 0.8}
                                for ln in raw.splitlines()
                                if ln.strip()
                            ]
                        yield LLMEvent(
                            type=LLMEventType.CONCLUSIONS,
                            conclusions=[c for c in parsed if isinstance(c, dict) and c.get("content")],
                        )
                        in_conclusions = False
                        conclusions_buf = []
                    else:
                        conclusions_buf.append(buf)
                        buf = ""
                    continue

                if in_phase:
                    if _PHASE_CLOSE in buf:
                        before, _, buf = buf.partition(_PHASE_CLOSE)
                        phase_buf.append(before)
                        yield LLMEvent(
                            type=LLMEventType.PHASE_DECL,
                            phase_decl="".join(phase_buf).strip(),
                        )
                        in_phase = False
                        phase_buf = []
                    else:
                        phase_buf.append(buf)
                        buf = ""
                    continue

                if in_strategy_fail:
                    if _STRATEGY_FAIL_CLOSE in buf:
                        before, _, buf = buf.partition(_STRATEGY_FAIL_CLOSE)
                        strategy_fail_buf.append(before)
                        yield LLMEvent(
                            type=LLMEventType.STRATEGY_FAIL,
                            strategy_fail_type=strategy_fail_type,
                            strategy_fail_reason="".join(strategy_fail_buf).strip(),
                        )
                        in_strategy_fail = False
                        strategy_fail_buf = []
                        strategy_fail_type = None
                    else:
                        strategy_fail_buf.append(buf)
                        buf = ""
                    continue

                # Scan for next structural tag
                lt_pos = buf.find("<")
                if lt_pos == -1:
                    buf = ""
                    break

                buf = buf[lt_pos:]
                gt_pos = buf.find(">")
                if gt_pos == -1:
                    break  # incomplete tag — wait for more tokens

                tag = buf[: gt_pos + 1]
                buf = buf[gt_pos + 1:]

                if tag == _CONCLUSIONS_OPEN:
                    in_conclusions = True
                    conclusions_buf = []
                elif tag == _PHASE_OPEN:
                    in_phase = True
                    phase_buf = []
                elif tag.startswith(_STRATEGY_FAIL_OPEN_PREFIX):
                    in_strategy_fail = True
                    strategy_fail_buf = []
                    m = _STRATEGY_FAIL_TYPE_RE.search(tag)
                    strategy_fail_type = m.group(1) if m else None
                elif _THINK_OPEN_RE.match(tag):
                    # <think> block after </response>: emit as REASONING (post-hoc trace)
                    in_post_think = True
                    post_think_buf = []
                # unknown post-response tags are silently consumed

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages_to_dicts(messages),
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # H-16: forward OpenAI-compat response_format when caller requests
        # guaranteed-valid JSON.  llama.cpp server honours
        # {"type": "json_object"} since mainline builds from early 2024;
        # older/alternate backends silently ignore the field.
        if response_format is not None:
            payload["response_format"] = response_format
        if self._debug_trace:
            approx_chars = sum(len(m.content or "") for m in messages)
            log.info(
                "LLM DEBUG complete request base_url=%s model=%s messages=%d approx_prompt_chars=%d max_tokens=%s temperature=%.2f response_format=%s",
                self._base_url, self._model, len(messages), approx_chars, max_tokens, temperature, bool(response_format),
            )
            log.debug("LLM DEBUG complete message summaries: %s", json.dumps(self._message_debug_summary(messages), ensure_ascii=False))
        log.debug(
            "LLM complete request — messages=%d, temperature=%.2f, max_tokens=%d\npayload:\n%s",
            len(messages),
            temperature,
            max_tokens,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        for _attempt in range(self._max_retries + 1):
            try:
                _t0 = time.monotonic()
                resp = await self._client.post("/v1/chat/completions", json=payload)
                if resp.status_code == 413:
                    raise LLMContextOverflowError("Prompt too large (413)")
                if resp.status_code >= 500:
                    raise LLMUnavailableError(f"LLM returned {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                log.debug(
                    "LLM complete response — elapsed=%.1fs, chars=%d\n%s",
                    time.monotonic() - _t0,
                    len(content),
                    content,
                )
                return content
            except LLMContextOverflowError:
                raise
            except (LLMUnavailableError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if _attempt < self._max_retries:
                    delay = self._retry_wait(_attempt)
                    log.warning(
                        "LLM complete unavailable (attempt %d/%d) — retrying in %.0fs: %s",
                        _attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise LLMUnavailableError(f"LLM unreachable after {self._max_retries + 1} attempts: {exc}") from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(f"LLM timeout: {exc}") from exc
        raise LLMUnavailableError("LLM complete: retry loop exhausted")

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = await self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format or {"type": "json_object"},
        )
        from cima_demo.domain.ports import _extract_json_object

        return _extract_json_object(raw)

    async def stream_text(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages_to_dicts(messages),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "cache_prompt": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if self._debug_trace:
            approx_chars = sum(len(m.content or "") for m in messages)
            log.info(
                "LLM DEBUG stream_text request base_url=%s model=%s messages=%d approx_prompt_chars=%d max_tokens=%s temperature=%.2f",
                self._base_url, self._model, len(messages), approx_chars, max_tokens, temperature,
            )
            log.debug("LLM DEBUG stream_text message summaries: %s", json.dumps(self._message_debug_summary(messages), ensure_ascii=False))

        log.debug(
            "LLM stream_text request — messages=%d temperature=%.2f top_p=%.2f repeat_penalty=%.2f max_tokens=%s",
            len(messages), temperature, top_p, repeat_penalty, max_tokens,
        )
        for _attempt in range(self._max_retries + 1):
            yielded_any = False
            try:
                async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                    if resp.status_code == 413:
                        raise LLMContextOverflowError("Prompt too large (413)")
                    if resp.status_code >= 500:
                        body = await resp.aread()
                        log.error("llama.cpp %d — body: %s", resp.status_code, body.decode(errors="replace")[:2000])
                        raise LLMUnavailableError(f"LLM returned {resp.status_code}")
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            payload_chunk = json.loads(data)
                        except Exception:
                            continue
                        choice = (payload_chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        chunk = delta.get("content") or ""
                        if chunk:
                            yielded_any = True
                            yield chunk
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            return
            except LLMContextOverflowError:
                raise
            except (LLMUnavailableError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if yielded_any:
                    raise
                if _attempt < self._max_retries:
                    delay = self._retry_wait(_attempt)
                    await asyncio.sleep(delay)
                    continue
                raise LLMUnavailableError(f"LLM stream_text unavailable after {self._max_retries + 1} attempts: {exc}") from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(f"LLM timeout: {exc}") from exc
        raise LLMUnavailableError("LLM stream_text: retry loop exhausted")

    async def count_tokens(self, text: str) -> int:
        """Use llama.cpp /tokenize endpoint for exact count."""
        try:
            resp = await self._client.post("/tokenize", json={"content": text})
            resp.raise_for_status()
            data = resp.json()
            return len(data.get("tokens", []))
        except Exception:
            # Fallback: rough estimate (4 chars ≈ 1 token)
            return max(1, len(text) // 4)

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
