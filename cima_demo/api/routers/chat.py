"""Chat endpoints for CIMA Demonstrator."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import time
from enum import Enum
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from cima_demo.api.auth import verify_api_key, verify_api_key_openai
from cima_demo.api.budgeting import build_effective_context_budget
from cima_demo.api.conversation_guard import ensure_active_conversation
from cima_demo.branding import (
    PUBLIC_CHAT_PATH,
    PUBLIC_MODEL_ID,
    PUBLIC_OWNER,
)
from cima_demo.api.dependencies import get_db, get_orchestrator, get_stream_manager
from cima_demo.api.settings import get_settings
from cima_demo.application.orchestrator import AgentOrchestrator
from cima_demo.application.stream_manager import StreamManager
from cima_demo.domain.entities import KimaDelta
from cima_demo.domain.ports import RelDBPort
from cima_demo.domain.value_objects import KimaDeltaType

router = APIRouter(tags=["chat"])

_STREAM_TIMEOUT_SECS = 7200  # fallback default; runtime value comes from CIMA_DEMO_OAI_STREAM_TIMEOUT_SECS
_HEARTBEAT_INTERVAL  = 10.0  # keepalive SSE comment — evita read-timeout en el cliente


# ── OpenAI-compat model list ──────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models(_: None = Depends(verify_api_key_openai)) -> dict:
    """Return a single model entry so clients discover the demonstrator model."""
    from datetime import UTC, datetime
    now = int(datetime.now(UTC).timestamp())
    return {
        "object": "list",
        "data": [{
            "id":       PUBLIC_MODEL_ID,
            "object":   "model",
            "created":  now,
            "owned_by": PUBLIC_OWNER,
        }],
    }


# ── CIMA Demonstrator Native SSE ─────────────────────────────────────────────

@router.post(PUBLIC_CHAT_PATH)
async def kima_chat(
    message: str = Form(..., min_length=1, max_length=32_768),
    files: list[UploadFile] = File(default=[]),
    x_conversation_id: str = Header(..., alias="X-Conversation-Id"),
    _auth: None = Depends(verify_api_key),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    stream_manager: StreamManager = Depends(get_stream_manager),
    db: RelDBPort = Depends(get_db),
) -> StreamingResponse:
    """Execute a demonstrator turn and stream only the client-visible deltas."""
    conversation_id = x_conversation_id

    # Validate conversation exists
    row = ensure_active_conversation(await db.get_conversation(conversation_id))

    # Read files BEFORE acquiring mutex (API-INV-12 / R-02)
    _settings = get_settings()
    max_files = _settings.max_files_per_request
    max_file_bytes = _settings.max_file_size_mb * 1024 * 1024
    if len(files) > max_files:
        raise HTTPException(status_code=422, detail=f"Max {max_files} files allowed")

    file_data: list[tuple[bytes, str, str]] = []
    for upload in files:
        content = await upload.read()
        if len(content) > max_file_bytes:
            raise HTTPException(
                status_code=422,
                detail=f"File {upload.filename!r} exceeds {_settings.max_file_size_mb}MB",
            )
        file_data.append((
            content,
            upload.filename or "unnamed",
            upload.content_type or "application/octet-stream",
        ))

    # Atomic turn mutex (P-01 — eliminates race condition)
    if not await db.try_set_turn_in_progress(conversation_id):
        raise HTTPException(
            status_code=409,
            detail={"code": "TURN_IN_PROGRESS", "message": "Conversation has an active turn"},
        )

    _t_req = time.monotonic()
    log.info(
        "→ CIMA/chat conv=%s msg_chars=%d files=%d",
        conversation_id, len(message), len(file_data),
    )
    log.debug("→ CIMA/chat body: %r", message[:500])

    # Build StreamingResponse — release mutex on construction failure (API-INV-12)
    try:
        response = StreamingResponse(
            _stream_kima(
                conversation_id=conversation_id,
                user_message=message,
                file_data=file_data or None,
                orchestrator=orchestrator,
                stream_manager=stream_manager,
                t_req=_t_req,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                "X-Conversation-Id": conversation_id,
            },
        )
    except Exception:
        await db.release_turn_in_progress(conversation_id)
        raise

    return response


def _demo_mode_enabled() -> bool:
    return bool(get_settings().demo_mode)


def _is_client_visible_delta(delta: KimaDelta, demo_mode: bool) -> bool:
    if not demo_mode:
        return True
    return delta.type in {
        KimaDeltaType.TOKEN,
        KimaDeltaType.ERROR,
        KimaDeltaType.DONE,
    }


def _demo_openai_heartbeat_chunk() -> str:
    return ": keepalive\n\n"


async def _stream_kima(
    conversation_id: str,
    user_message: str,
    file_data: list[tuple[bytes, str, str]] | None,
    orchestrator: AgentOrchestrator,
    stream_manager: StreamManager,
    t_req: float = 0.0,
) -> AsyncIterator[str]:
    """SSE generator for CIMA Demonstrator native SSE.

    API-INV-01: subscribe BEFORE launching the Task.
    Uses the shielded-task pattern (same as _stream_openai) so heartbeat timeouts
    never cancel the async generator's __anext__() coroutine mid-flight.
    """
    handle_task: asyncio.Task[None] | None = None
    _n_deltas = 0
    _n_heartbeats = 0
    _t0 = t_req or time.monotonic()
    demo_mode = _demo_mode_enabled()

    # Subscribe before launching task (API-INV-01)
    sub = stream_manager.subscribe(conversation_id)

    handle_task = asyncio.create_task(
        orchestrator.handle_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            attached_files=file_data,
        )
    )

    # Persistent task for the current __anext__() call — same pattern as _stream_openai.
    # Never cancelled by heartbeat timeouts, preventing async-generator corruption.
    _pending: asyncio.Task[KimaDelta | None] | None = None
    # True only when the while loop exits on a DONE delta or exhausted generator —
    # used by finally to distinguish normal completion from GeneratorExit (client disconnect).
    _clean_done = False
    try:
        while True:
            if _pending is None:
                _pending = asyncio.ensure_future(_anext_or_none(sub))
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(_pending), timeout=_HEARTBEAT_INTERVAL
                )
                _pending = None
            except asyncio.TimeoutError:
                _n_heartbeats += 1
                log.debug("← CIMA/chat heartbeat #%d conv=%s elapsed=%.1fs",
                          _n_heartbeats, conversation_id, time.monotonic() - _t0)
                if handle_task is not None and handle_task.done():
                    log.warning(
                        "← CIMA/chat handle_task done but no terminal delta was received; closing stream conv=%s",
                        conversation_id,
                    )
                    _clean_done = True
                    break
                yield ": keepalive\n\n"
                continue

            if result is None:
                _clean_done = True
                break

            delta = result
            _n_deltas += 1
            _elapsed = time.monotonic() - _t0
            if delta.type == KimaDeltaType.TOKEN and delta.token:
                log.debug("← CIMA/chat TOKEN #%d +%.1fs chars=%d",
                          _n_deltas, _elapsed, len(delta.token))
            else:
                log.debug("← CIMA/chat %s #%d +%.1fs", delta.type, _n_deltas, _elapsed)

            if _is_client_visible_delta(delta, demo_mode):
                yield _delta_to_sse(delta)
            if delta.type == KimaDeltaType.DONE:
                _clean_done = True
                break

    except asyncio.CancelledError:
        log.info("← CIMA/chat CANCELLED conv=%s elapsed=%.1fs deltas=%d",
                 conversation_id, time.monotonic() - _t0, _n_deltas)
        # Cancel the orchestrator task so its finally block releases the mutex promptly
        # instead of holding it until the full inference completes (may be minutes).
        if handle_task is not None and not handle_task.done():
            handle_task.cancel()
        return
    finally:
        if _pending is not None and not _pending.done():
            _pending.cancel()
        # GeneratorExit (LibreChat stop button → aclose()) lands here, not in CancelledError.
        # Cancel the orchestrator so the mutex is released promptly.
        if not _clean_done and handle_task is not None and not handle_task.done():
            log.info("← CIMA/chat early exit (GeneratorExit?) — cancelling handle_task conv=%s",
                     conversation_id)
            handle_task.cancel()

    # Check if task raised an exception
    if handle_task is not None and handle_task.done() and not handle_task.cancelled():
        exc = handle_task.exception()
        if exc is not None:
            yield _error_sse(conversation_id, "INTERNAL_ERROR", str(exc), recoverable=False)

    log.info("← CIMA/chat DONE conv=%s elapsed=%.1fs deltas=%d heartbeats=%d",
             conversation_id, time.monotonic() - _t0, _n_deltas, _n_heartbeats)
    yield "\n"  # close SSE stream


def _delta_to_sse(delta: KimaDelta) -> str:
    """Serialize KimaDelta to SSE string using the KimaEnvelope wire format."""
    t = delta.type.value if isinstance(delta.type, Enum) else delta.type
    inner: dict[str, object] = {}

    if t == KimaDeltaType.REASONING:
        inner = {"text": delta.token or ""}
    elif t == KimaDeltaType.TOKEN:
        inner = {"text": delta.token or ""}
    elif t == KimaDeltaType.THOUGHT:
        try:
            params = json.loads(delta.thought or "{}")
        except Exception:
            params = {}
        inner = {"tool_name": delta.tool_name or "", "params": params}
    elif t == KimaDeltaType.TOOL_RESULT:
        inner = {
            "tool_name":     delta.tool_name or "",
            "success":       delta.error_message is None,
            "summary":       delta.tool_result or "",
            "error_message": delta.error_message,
        }
    elif t == KimaDeltaType.PLAN_STEP:
        inner = {
            "plan_id":          delta.plan_id or "",
            "step_seq":         delta.step_index if delta.step_index is not None else 0,
            "step_description": delta.step_description or "",
            "total_steps":      delta.total_steps if delta.total_steps is not None else 1,
            "event":            delta.step_status or "advanced",
        }
    elif t == KimaDeltaType.STALL:
        inner = {"reason": delta.stall_message or "", "iteration": 0}
    elif t == KimaDeltaType.CONTEXT_REFRESH:
        inner = {"tokens_before": 0, "tokens_after": 0, "items_summarized": 0}
    elif t == KimaDeltaType.ERROR:
        inner = {
            "code":        delta.error_code or "UNKNOWN",
            "message":     delta.error_message or "",
            "recoverable": delta.error_code not in ("CONTEXT_OVERFLOW", "INTERNAL_ERROR"),
        }

    envelope: dict[str, object] = {
        "conversation_id": delta.conversation_id,
        "turn_id":         "",
        "trace_id":        "",
        "type":            t,
        "timestamp":       datetime.now(UTC).isoformat(),
        "payload":         inner,
    }
    return f"event: {t}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"


def _error_sse(
    conversation_id: str,
    code: str,
    message: str,
    recoverable: bool,
) -> str:
    envelope: dict[str, object] = {
        "conversation_id": conversation_id,
        "turn_id":         "",
        "trace_id":        "",
        "type":            "ERROR",
        "timestamp":       datetime.now(UTC).isoformat(),
        "payload": {
            "code":        code,
            "message":     message,
            "recoverable": recoverable,
        },
    }
    return f"event: ERROR\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"


# ── OpenAI-compatible SSE ─────────────────────────────────────────────────────

class OpenAIMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None
    # LibreChat sends non-image attachments (PDF, DOCX, …) here, NOT in content.
    # formatMessage() merges image_urls → content but leaves documents separate.
    documents: list[Any] | None = None


class OpenAICompletionRequest(BaseModel):
    model: str = PUBLIC_MODEL_ID
    messages: list[OpenAIMessage] = Field(..., min_length=1)
    stream: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # OpenAI-compatible output cap. In demo mode it also defines the reserved
    # answer budget used to shrink the context pack before the model call.
    max_tokens: int | None = Field(default=None, ge=1)
    conversation_id: str | None = None
    # Demonstrator extension fields used by open_scenarios and local eval. They
    # are ignored by generic clients but let the harness request smaller model
    # windows without changing server-wide environment variables.
    max_context_tokens: int | None = Field(default=None, ge=1)
    reserve_output_tokens: int | None = Field(default=None, ge=1)
    overhead_tokens: int | None = Field(default=None, ge=0)


def _request_budget(body: OpenAICompletionRequest):
    settings = get_settings()
    reserve = body.reserve_output_tokens or body.max_tokens or settings.llm_max_tokens
    explicit_context_window = body.max_context_tokens is not None
    overhead = body.overhead_tokens if body.overhead_tokens is not None else (0 if explicit_context_window else settings.context_budget_overhead)
    return build_effective_context_budget(
        requested_context_tokens=body.max_context_tokens or settings.context_budget_max,
        reserve_output_tokens=reserve,
        overhead_tokens=overhead,
        settings=settings,
    ), reserve


def _decode_data_uri(
    url: str,
    fallback_fname: str = "",
    hint_fname: str = "",
) -> tuple[bytes, str, str] | None:
    """Decode a data URI into (bytes, filename, mime). Returns None on failure."""
    if not url.startswith("data:"):
        return None
    try:
        header, encoded = url.split(",", 1)
        mime = header.split(";")[0][len("data:"):]
        raw = base64.b64decode(encoded)
        ext = mimetypes.guess_extension(mime) or ""
        fname = hint_fname or fallback_fname or f"attachment{ext}"
        return raw, fname, mime
    except Exception as exc:
        # HCR-1: log decode failures so we can distinguish "file not in request"
        # from "file in request but base64/header parsing failed".
        log.warning(
            "attachment decode failure fname=%r mime_prefix=%r bytes_b64=%d exc=%s",
            hint_fname or fallback_fname,
            url[:40],
            len(url),
            exc,
        )
        return None


def _parse_content(msg: OpenAIMessage) -> tuple[str, list[tuple[bytes, str, str]]]:
    """Extract plain text and embedded files from an OpenAI message.

    Handles:
    - String content (plain text, no files)
    - Vision-style list content with image_url parts
    - LibreChat documents field (non-image attachments: PDF, DOCX, …)
      LibreChat's formatMessage() merges image_urls into content but leaves
      documents as a separate message.documents field.
    """
    content = msg.content
    files: list[tuple[bytes, str, str]] = []

    # ── Parse content (text + inline images) ─────────────────────────────────
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text_parts: list[str] = []
        _file_parts_seen = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                text_parts.append(part.get("text") or "")
            elif ptype in ("image_url", "file", "input_file"):
                _file_parts_seen += 1
                decoded = _extract_file_from_part(part)
                if decoded:
                    files.append(decoded)
        text = " ".join(text_parts)
        # HCR-1: log content-part summary so we can distinguish "no file parts
        # in the request" from "file parts present but decode failed".
        if _file_parts_seen > 0 or files:
            log.info(
                "attachment parse: content file_parts_seen=%d decoded=%d",
                _file_parts_seen, len(files),
            )

    # ── Parse documents (LibreChat non-image attachments) ────────────────────
    # LibreChat puts PDFs, DOCX, etc. in message.documents, not in content.
    if msg.documents:
        _doc_parts_seen = len(msg.documents)
        _docs_before = len(files)
        for part in msg.documents:
            if not isinstance(part, dict):
                continue
            decoded = _extract_file_from_part(part)
            if decoded:
                files.append(decoded)
        log.info(
            "attachment parse: documents doc_parts=%d decoded=%d",
            _doc_parts_seen, len(files) - _docs_before,
        )

    return text, files


def _extract_file_from_part(part: dict) -> tuple[bytes, str, str] | None:
    """Extract (bytes, filename, mime) from a single content-array part dict.

    Handles:
    - image_url: {"type": "image_url", "image_url": {"url": "data:..."}}
    - input_file: {"type": "input_file", "file_data": "base64...", "media_type": "..."}
    - LibreChat file: {"type": "file", "file": {"filename": "...", "file_data": "data:..."}}
    """
    ptype = part.get("type", "")
    hint_fname = part.get("filename") or part.get("name") or ""

    # Standard vision format
    if ptype == "image_url":
        url_obj = part.get("image_url") or {}
        url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
        result = _decode_data_uri(url, hint_fname=hint_fname)
        if result is None:
            log.warning("attachment extract: image_url part decoded to None fname=%r url_prefix=%r", hint_fname, url[:60])
        return result

    # Responses API / input_file format: file_data is bare base64, not a data URI
    if ptype == "input_file":
        raw_b64 = part.get("file_data", "")
        mime = part.get("media_type", "application/octet-stream")
        if raw_b64:
            url = f"data:{mime};base64,{raw_b64}"
            result = _decode_data_uri(url, hint_fname=hint_fname)
            if result is None:
                log.warning("attachment extract: input_file part decoded to None fname=%r mime=%r b64_len=%d", hint_fname, mime, len(raw_b64))
            return result
        log.warning("attachment extract: input_file part has empty file_data fname=%r", hint_fname)
        return None

    # LibreChat custom-endpoint document format:
    # {"type": "file", "file": {"filename": "doc.pdf", "file_data": "data:application/pdf;base64,..."}}
    if ptype == "file":
        # Nested file object (LibreChat)
        file_obj = part.get("file")
        if isinstance(file_obj, dict):
            url = file_obj.get("file_data", "")
            fname = file_obj.get("filename") or hint_fname
            result = _decode_data_uri(url, hint_fname=fname)
            if result is None:
                log.warning("attachment extract: file/nested part decoded to None fname=%r url_prefix=%r", fname, url[:60])
            return result
        # Flat file_data (other clients)
        raw_b64 = part.get("file_data", "")
        if raw_b64:
            mime = part.get("media_type", "application/octet-stream")
            url = f"data:{mime};base64,{raw_b64}"
            result = _decode_data_uri(url, hint_fname=hint_fname)
            if result is None:
                log.warning("attachment extract: file/flat part decoded to None fname=%r mime=%r", hint_fname, mime)
            return result
        log.warning("attachment extract: file part has no file_data fname=%r keys=%s", hint_fname, list(part.keys()))
        return None

    return None


@router.post("/v1/chat/completions")
async def openai_completions(
    body: OpenAICompletionRequest,
    x_conversation_id: str | None = Header(default=None, alias="X-Conversation-Id"),
    _auth: None = Depends(verify_api_key_openai),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    stream_manager: StreamManager = Depends(get_stream_manager),
    db: RelDBPort = Depends(get_db),
) -> Response:
    if not body.stream:
        return await _openai_completions_nonstreaming(
            body=body,
            x_conversation_id=x_conversation_id,
            orchestrator=orchestrator,
            stream_manager=stream_manager,
            db=db,
        )

    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Find the last user message; extract text + any embedded files
    last_user = next(
        (m for m in reversed(body.messages) if m.role == "user"), None
    )
    if last_user is None:
        raise HTTPException(status_code=400, detail="No user message found")
    user_message, attached_files = _parse_content(last_user)
    if not user_message.strip():
        # Fall back: find any non-empty user message
        for m in reversed(body.messages):
            if m.role == "user":
                text, fls = _parse_content(m)
                if text.strip():
                    user_message, attached_files = text, fls
                    break
    existing_id = x_conversation_id or body.conversation_id
    if existing_id:
        # Upsert: create the conversation if the client-provided ID is not yet known.
        # This handles the first turn from LibreChat, which generates its own stable UUID
        # and sends it as conversation_id before KIMA has recorded it.
        await db.create_conversation(existing_id)  # idempotent — ON CONFLICT DO NOTHING
        ensure_active_conversation(await db.get_conversation(existing_id))
        if not await db.try_set_turn_in_progress(existing_id):
            raise HTTPException(status_code=409, detail={"code": "TURN_IN_PROGRESS"})
        conversation_id = existing_id
    else:
        conversation_id = str(uuid.uuid4())
        await db.create_conversation(conversation_id)
        if not await db.try_set_turn_in_progress(conversation_id):
            raise HTTPException(status_code=500, detail="Failed to acquire turn lock")

    chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(datetime.now(UTC).timestamp())
    _t_req = time.monotonic()

    log.info(
        "→ OAI/completions conv=%s msg_chars=%d files=%d msgs=%d",
        conversation_id, len(user_message), len(attached_files), len(body.messages),
    )
    log.debug("→ OAI/completions user_message: %r", user_message[:500])

    try:
        response = StreamingResponse(
            _stream_openai(
                conversation_id=conversation_id,
                user_message=user_message,
                attached_files=attached_files or None,
                chatcmpl_id=chatcmpl_id,
                created_ts=created_ts,
                orchestrator=orchestrator,
                stream_manager=stream_manager,
                context_budget_override=_request_budget(body)[0],
                llm_max_tokens_override=_request_budget(body)[1],
                t_req=_t_req,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                "X-Conversation-Id": conversation_id,
            },
        )
    except Exception:
        await db.release_turn_in_progress(conversation_id)
        raise

    return response


async def _anext_or_none(gen: Any) -> "KimaDelta | None":
    """Wrap __anext__() so StopAsyncIteration becomes None (Tasks cannot raise it)."""
    try:
        return await gen.__anext__()
    except StopAsyncIteration:
        return None


async def _stream_openai(
    conversation_id: str,
    user_message: str,
    chatcmpl_id: str,
    created_ts: int,
    orchestrator: AgentOrchestrator,
    stream_manager: StreamManager,
    attached_files: list[tuple[bytes, str, str]] | None = None,
    context_budget_override: Any | None = None,
    llm_max_tokens_override: int | None = None,
    t_req: float = 0.0,
) -> AsyncIterator[str]:
    """OpenAI-compatible SSE for the demonstrator.

    LibreChat's agents framework detects delta.reasoning and emits on_reasoning_delta
    events, which the frontend renders as a collapsible ContentTypes.THINK panel.

    Heartbeat design: a persistent asyncio.Task wraps each __anext__() call and is
    reused across heartbeat timeouts via asyncio.shield().  This prevents the
    wait_for() cancellation from propagating into the async generator and triggering
    its finally-block (which would destroy the queue and end the subscription early).
    """
    sub = stream_manager.subscribe(conversation_id)
    demo_mode = _demo_mode_enabled()
    stream_timeout_secs = float(getattr(get_settings(), "oai_stream_timeout_secs", _STREAM_TIMEOUT_SECS) or 0)
    if stream_timeout_secs <= 0:
        stream_timeout_secs = 0.0
    _n_deltas = 0
    _n_heartbeats = 0
    _t0 = t_req or time.monotonic()

    handle_task: asyncio.Task[None] = asyncio.create_task(
        orchestrator.handle_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            attached_files=attached_files,
            context_budget_override=context_budget_override,
            llm_max_tokens_override=llm_max_tokens_override,
        )
    )

    # First chunk: role announcement + conversation_id so LibreChat can reuse it
    first_chunk: dict[str, object] = {
        "id":              chatcmpl_id,
        "object":          "chat.completion.chunk",
        "created":         created_ts,
        "model":           PUBLIC_MODEL_ID,
        "conversation_id": conversation_id,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    elapsed = 0.0
    # Persistent task for the current __anext__() call.  Created once per item;
    # kept alive across heartbeat timeouts so the generator is never cancelled.
    _pending: asyncio.Task[KimaDelta | None] | None = None
    # True only on normal loop exit (DONE delta or exhausted generator).
    # Used by finally to detect GeneratorExit (LibreChat stop button) on early exit.
    _clean_done = False
    try:
        while True:
            if _pending is None:
                _pending = asyncio.ensure_future(_anext_or_none(sub))

            try:
                result = await asyncio.wait_for(
                    asyncio.shield(_pending), timeout=_HEARTBEAT_INTERVAL
                )
                _pending = None  # task consumed — create a new one next iteration
                elapsed = 0.0
            except asyncio.TimeoutError:
                # _pending is still alive (shielded) - generator not cancelled.
                # If the orchestrator is already done and the queue is empty, the
                # terminal DONE delta was missed or evicted; close from task state
                # instead of heartbeating forever.
                elapsed += _HEARTBEAT_INTERVAL
                _n_heartbeats += 1
                log.debug("← OAI/completions heartbeat #%d conv=%s elapsed=%.1fs",
                          _n_heartbeats, conversation_id, time.monotonic() - _t0)
                if handle_task.done():
                    log.warning(
                        "← OAI/completions handle_task done but no terminal delta was received; closing stream conv=%s",
                        conversation_id,
                    )
                    _clean_done = True
                    break
                if stream_timeout_secs > 0 and elapsed >= stream_timeout_secs:
                    msg = (
                        f"OAI_STREAM_TIMEOUT after {stream_timeout_secs:.0f}s without a visible delta "
                        f"(conv={conversation_id}, deltas={_n_deltas})"
                    )
                    log.warning("← OAI/completions %s elapsed=%.1fs", msg, time.monotonic() - _t0)
                    if not handle_task.done():
                        handle_task.cancel()
                    # Do not synthesize a fake assistant answer such as [TIMEOUT].
                    # Non-streaming callers receive an HTTP error; streaming callers
                    # see the connection fail, which is a generation failure rather
                    # than a valid answer.
                    raise TimeoutError(msg)
                if demo_mode:
                    yield _demo_openai_heartbeat_chunk()
                else:
                    # Heartbeat as data keeps legacy clients from timing out.
                    yield _openai_chunk(chatcmpl_id, created_ts, delta={"reasoning": " "})
                continue

            if result is None:
                # Generator exhausted (StopAsyncIteration converted by _anext_or_none)
                _clean_done = True
                break

            delta = result
            _n_deltas += 1
            _elapsed = time.monotonic() - _t0

            if delta.type == KimaDeltaType.REASONING and delta.token:
                log.debug("← OAI/completions REASONING #%d +%.1fs chars=%d",
                          _n_deltas, _elapsed, len(delta.token))
                if not demo_mode:
                    yield _openai_chunk(chatcmpl_id, created_ts, delta={"reasoning": delta.token})
            elif delta.type == KimaDeltaType.TOKEN and delta.token:
                log.debug("← OAI/completions TOKEN #%d +%.1fs chars=%d",
                          _n_deltas, _elapsed, len(delta.token))
                yield _openai_chunk(chatcmpl_id, created_ts, delta={"content": delta.token})
            elif delta.type == KimaDeltaType.THOUGHT:
                # Show tool call in progress as a reasoning token so LibreChat
                # renders it in the collapsible thinking panel (not main content).
                name = delta.tool_name or "tool"
                log.debug("← OAI/completions THOUGHT #%d +%.1fs tool=%s",
                          _n_deltas, _elapsed, name)
                try:
                    params = json.loads(delta.thought or "{}")
                    param_str = ", ".join(
                        f"{k}={str(v)!r}" for k, v in params.items()
                    )
                except Exception:
                    param_str = delta.thought or ""
                if not demo_mode:
                    yield _openai_chunk(
                        chatcmpl_id, created_ts,
                        delta={"reasoning": f"\n→ {name}({param_str})\n"},
                    )
            elif delta.type == KimaDeltaType.TOOL_RESULT:
                _ok = delta.success if delta.success is not None else delta.error_message is None
                icon = "✓" if _ok else "✗"
                summary = (delta.tool_result or "")
                log.debug("← OAI/completions TOOL_RESULT #%d +%.1fs tool=%s ok=%s",
                          _n_deltas, _elapsed, delta.tool_name, _ok)
                if not demo_mode:
                    yield _openai_chunk(
                        chatcmpl_id, created_ts,
                        delta={"reasoning": f"  {icon} {delta.tool_name}: {summary}\n"},
                    )
            elif delta.type == KimaDeltaType.STALL:
                log.debug("← OAI/completions STALL #%d +%.1fs", _n_deltas, _elapsed)
                msg = f"\n⚠ {delta.stall_message or 'Iteration limit reached'}\n"
                if not demo_mode:
                    yield _openai_chunk(chatcmpl_id, created_ts, delta={"reasoning": msg})
            elif delta.type == KimaDeltaType.STRATEGY_SWITCH:
                log.debug("← OAI/completions STRATEGY_SWITCH #%d +%.1fs", _n_deltas, _elapsed)
                if not demo_mode:
                    yield _openai_chunk(
                        chatcmpl_id, created_ts,
                        delta={"reasoning": "\n↻ Strategy switch — retrying\n"},
                    )
            elif delta.type == KimaDeltaType.ERROR:
                log.info("← OAI/completions ERROR #%d +%.1fs code=%s msg=%r",
                         _n_deltas, _elapsed, delta.error_code, delta.error_message)
                error_msg = f"[ERROR: {delta.error_message or delta.error_code}]"
                yield _openai_chunk(chatcmpl_id, created_ts, delta={"content": error_msg}, finish="stop")
                # Drain until DONE so the orchestrator's finally block (mutex release)
                # completes before data:[DONE] reaches the client.
                # Without this, the client starts the next turn while turn_in_progress=True → 409.
                # Reuse the shielded-task pattern here too.
                _drain: asyncio.Task[KimaDelta | None] | None = None
                try:
                    for _ in range(200):  # bounded drain — max 200 items
                        if _drain is None:
                            _drain = asyncio.ensure_future(_anext_or_none(sub))
                        drain = await asyncio.wait_for(asyncio.shield(_drain), timeout=120.0)
                        _drain = None
                        if drain is None or drain.type == KimaDeltaType.DONE:
                            break
                except (asyncio.TimeoutError, Exception):
                    pass
                finally:
                    if _drain is not None and not _drain.done():
                        _drain.cancel()
                break
            elif delta.type == KimaDeltaType.DONE:
                _clean_done = True
                break
            else:
                log.debug("← OAI/completions %s #%d +%.1fs", delta.type, _n_deltas, _elapsed)
    except asyncio.CancelledError:
        log.info("← OAI/completions CANCELLED conv=%s elapsed=%.1fs deltas=%d",
                 conversation_id, time.monotonic() - _t0, _n_deltas)
        # Cancel the orchestrator task so its finally block releases the mutex
        # promptly instead of holding it until inference finishes (may be minutes).
        if not handle_task.done():
            handle_task.cancel()
        return
    finally:
        # Cancel the pending __anext__() task on any exit path so the generator's
        # finally-block (queue cleanup) runs promptly instead of leaking.
        if _pending is not None and not _pending.done():
            _pending.cancel()
        # GeneratorExit (LibreChat stop button → aclose()) is NOT caught by CancelledError.
        # On any early exit, cancel the orchestrator so the mutex is released promptly.
        if not _clean_done and not handle_task.done():
            log.info("← OAI/completions early exit (GeneratorExit?) — cancelling handle_task conv=%s",
                     conversation_id)
            handle_task.cancel()

    # Drain: wait for orchestrator finally block (save_task_memory / mutex release)
    # before sending [DONE] to the client. Without this, the client starts the next
    # turn while turn_in_progress=True is still in DB → 409.
    # The orchestrator publishes DONE as its last action — after mutex release —
    # so handle_task should be done (or nearly done) at this point.
    if not handle_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(handle_task), timeout=15.0)
        except asyncio.CancelledError:
            # Task was already cancelled (e.g. by the timeout-break path in finally).
            # Expected — do not let this escape the generator or data:[DONE] is never sent.
            pass
        except asyncio.TimeoutError:
            # Orchestrator still running 15 s after client received DONE delta —
            # force-cancel it so its finally block runs and releases the mutex
            # before [DONE] is flushed to the client.
            if not handle_task.done():
                log.warning(
                    "← OAI/completions drain timeout — cancelling handle_task conv=%s",
                    conversation_id,
                )
                handle_task.cancel()
                try:
                    await asyncio.wait_for(handle_task, timeout=30.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
        except Exception:
            pass

    log.info("← OAI/completions DONE conv=%s elapsed=%.1fs deltas=%d heartbeats=%d",
             conversation_id, time.monotonic() - _t0, _n_deltas, _n_heartbeats)

    # Final chunk + DONE
    yield _openai_chunk(chatcmpl_id, created_ts, delta={}, finish="stop")
    yield "data: [DONE]\n\n"


def _openai_chunk(
    chatcmpl_id: str,
    created_ts: int,
    delta: dict[str, object],
    finish: str | None = None,
) -> str:
    chunk = {
        "id":      chatcmpl_id,
        "object":  "chat.completion.chunk",
        "created": created_ts,
        "model":   PUBLIC_MODEL_ID,
        "choices": [{
            "index":         0,
            "delta":         delta,
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _openai_completions_nonstreaming(
    body: "OpenAICompletionRequest",
    x_conversation_id: str | None,
    orchestrator: "AgentOrchestrator",
    stream_manager: "StreamManager",
    db: "RelDBPort",
) -> JSONResponse:
    """Buffer _stream_openai and return a standard chat.completion JSON.

    Invariants:
    - Never returns 200 silently when the turn fails — exceptions propagate as 500.
    - Lock semantics identical to the streaming path: acquired here, released by the
      orchestrator's finally block (which runs when the generator is fully consumed).
    - Only TOKEN delta.content is accumulated; reasoning/thought/tool_result are ignored.
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_user = next((m for m in reversed(body.messages) if m.role == "user"), None)
    if last_user is None:
        raise HTTPException(status_code=400, detail="No user message found")
    user_message, attached_files = _parse_content(last_user)
    if not user_message.strip():
        for m in reversed(body.messages):
            if m.role == "user":
                text, fls = _parse_content(m)
                if text.strip():
                    user_message, attached_files = text, fls
                    break

    existing_id = x_conversation_id or body.conversation_id
    if existing_id:
        await db.create_conversation(existing_id)
        ensure_active_conversation(await db.get_conversation(existing_id))
        if not await db.try_set_turn_in_progress(existing_id):
            raise HTTPException(status_code=409, detail={"code": "TURN_IN_PROGRESS"})
        conversation_id = existing_id
    else:
        conversation_id = str(uuid.uuid4())
        await db.create_conversation(conversation_id)
        if not await db.try_set_turn_in_progress(conversation_id):
            raise HTTPException(status_code=500, detail="Failed to acquire turn lock")

    chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(datetime.now(UTC).timestamp())

    content_parts: list[str] = []
    try:
        _budget_override, _reserved_output = _request_budget(body)
        async for chunk_str in _stream_openai(
            conversation_id=conversation_id,
            user_message=user_message,
            attached_files=attached_files or None,
            chatcmpl_id=chatcmpl_id,
            created_ts=created_ts,
            orchestrator=orchestrator,
            stream_manager=stream_manager,
            context_budget_override=_budget_override,
            llm_max_tokens_override=_reserved_output,
        ):
            if not chunk_str.startswith("data: ") or chunk_str.startswith("data: [DONE]"):
                continue
            try:
                chunk_data = json.loads(chunk_str[6:].strip())
            except json.JSONDecodeError:
                continue
            for choice in chunk_data.get("choices", []):
                token = choice.get("delta", {}).get("content", "")
                if token:
                    content_parts.append(token)
    except Exception as exc:
        # Do NOT swallow — a failing turn must not silently return 200 with no content.
        log.error(
            "non-streaming turn failed conv=%s: %s", conversation_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Turn failed: {exc}") from exc

    return JSONResponse(
        content={
            "id":              chatcmpl_id,
            "object":          "chat.completion",
            "created":         created_ts,
            "model":           PUBLIC_MODEL_ID,
            "conversation_id": conversation_id,
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        headers={"X-Conversation-Id": conversation_id},
    )

