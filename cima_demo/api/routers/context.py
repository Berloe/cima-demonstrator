"""Explicit context endpoints for the CIMA Demonstrator."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from cima_demo.api.auth import verify_api_key
from cima_demo.api.budgeting import build_effective_context_budget
from cima_demo.api.dependencies import get_context_service, get_db, get_run_journal
from cima_demo.api.settings import get_settings
from cima_demo.api.conversation_guard import ensure_active_conversation
from cima_demo.branding import PUBLIC_CONTEXT_PREFIX
from cima_demo.domain.entities import TaskMemory
from cima_demo.domain.value_objects import ContextBudget

router = APIRouter(prefix=PUBLIC_CONTEXT_PREFIX, tags=["context"])


class ContextGetRequest(BaseModel):
    conversation_id: str
    request_id: str | None = None
    query: str | None = None
    user_text: str | None = None
    mode: str | None = None
    phase: str = "recall"
    max_context_tokens: int | None = None
    reserve_output_tokens: int | None = None
    overhead_tokens: int | None = None
    tokenizer_id: str | None = None
    model_id: str | None = None
    selected_artifact_ids: list[str] = Field(default_factory=list)
    selected_scope: str | None = None
    global_objective: str = ""
    local_objective: str = ""
    exclude_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_query(self):
        query = (self.query or self.user_text or "").strip()
        if not query:
            raise ValueError("query or user_text is required")
        self.query = query
        return self


class ZoomRequest(BaseModel):
    context_id: str
    zoom_targets: list[str] = Field(default_factory=list)
    max_evidence_tokens: int = 800


class ZoomOutRequest(BaseModel):
    context_id: str
    targets: list[str] = Field(default_factory=lambda: ["MASTER"])
    max_perspective_tokens: int = 800


class MemoryApplyRequest(BaseModel):
    conversation_id: str
    conclude: list[str] = Field(default_factory=list)
    phase: str = "synthesis"
    turn_id: str | None = None


async def _materialize_context(body: ContextGetRequest, *, db, context_service, run_journal):
    conversation = ensure_active_conversation(await db.get_conversation(body.conversation_id))
    task_memory = await db.load_task_memory(body.conversation_id) or TaskMemory(conversation_id=body.conversation_id)
    plan = None
    active_plan_id = getattr(task_memory, "active_plan_id", None)
    if active_plan_id:
        plan = await db.load_plan(active_plan_id)
    run_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    manifest = await run_journal.open_skeleton_run(
        run_id=run_id,
        conversation_id=body.conversation_id,
        turn_id=turn_id,
        user_message=body.query,
        attached_files=[],
    )
    token = context_service.bind_run(
        run_id=run_id,
        conversation_id=body.conversation_id,
        turn_id=turn_id,
        query_text=body.query,
    )
    try:
        explicit_context_window = body.max_context_tokens is not None
        budget = build_effective_context_budget(
            requested_context_tokens=body.max_context_tokens or 4096,
            reserve_output_tokens=body.reserve_output_tokens,
            # For explicit probe windows, the caller is usually sizing the
            # context pack itself. Reserve the answer, but do not silently spend
            # most of a small test window on estimated system overhead unless
            # the caller asked for it.
            overhead_tokens=(body.overhead_tokens if body.overhead_tokens is not None else (0 if explicit_context_window else 512)),
            settings=get_settings(),
        )
        payload = await context_service.get_context(
            conversation_id=body.conversation_id,
            query=body.query,
            phase=body.phase,
            task_memory=task_memory,
            plan=plan,
            budget=budget,
            global_objective=body.global_objective,
            local_objective=body.local_objective,
            exclude_ids=set(body.exclude_ids),
        )
    finally:
        context_service.reset_run(token)
    manifest.status = "context_only"
    manifest.cognitive_phase = body.phase
    manifest.execution_mode = "context_probe"
    manifest.finished_at = datetime.now(UTC)
    await run_journal.finalize_run(manifest)
    payload["run_id"] = run_id
    payload["turn_id"] = turn_id
    return payload


@router.post('/get')
async def get_context(
    body: ContextGetRequest,
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
    context_service=Depends(get_context_service),
    run_journal=Depends(get_run_journal),
):
    return await _materialize_context(body, db=db, context_service=context_service, run_journal=run_journal)


@router.post('/zoom')
async def zoom_context(
    body: ZoomRequest,
    _auth: None = Depends(verify_api_key),
    context_service=Depends(get_context_service),
):
    return await context_service.zoom(
        context_id=body.context_id,
        zoom_targets=body.zoom_targets,
        max_evidence_tokens=body.max_evidence_tokens,
    )


@router.post('/zoom_out')
async def zoom_out_context(
    body: ZoomOutRequest,
    _auth: None = Depends(verify_api_key),
    context_service=Depends(get_context_service),
):
    return await context_service.zoom_out(
        context_id=body.context_id,
        targets=body.targets,
        max_perspective_tokens=body.max_perspective_tokens,
    )


@router.post('/memory/apply')
async def apply_memory(
    body: MemoryApplyRequest,
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
    context_service=Depends(get_context_service),
):
    ensure_active_conversation(await db.get_conversation(body.conversation_id))
    return await context_service.apply_memory(
        conversation_id=body.conversation_id,
        conclude=body.conclude,
        phase=body.phase,
        turn_id=body.turn_id or str(uuid.uuid4()),
    )


@router.get('/{context_id}')
async def get_context_snapshot(
    context_id: str,
    _auth: None = Depends(verify_api_key),
    context_service=Depends(get_context_service),
):
    snapshot = await context_service.load_context_snapshot_public(context_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Context snapshot not found")
    return snapshot


# Compatibility surface for the CIMA v1 context contract.
cima_v1_router = APIRouter(prefix="/cima/v1/context", tags=["context:v1-compat"])


@cima_v1_router.post('/get')
async def get_context_v1(
    body: ContextGetRequest,
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
    context_service=Depends(get_context_service),
    run_journal=Depends(get_run_journal),
):
    return await _materialize_context(body, db=db, context_service=context_service, run_journal=run_journal)


@cima_v1_router.post('/zoom')
async def zoom_context_v1(
    body: ZoomRequest,
    _auth: None = Depends(verify_api_key),
    context_service=Depends(get_context_service),
):
    return await context_service.zoom(
        context_id=body.context_id,
        zoom_targets=body.zoom_targets,
        max_evidence_tokens=body.max_evidence_tokens,
    )


@cima_v1_router.post('/zoom_out')
async def zoom_out_context_v1(
    body: ZoomOutRequest,
    _auth: None = Depends(verify_api_key),
    context_service=Depends(get_context_service),
):
    return await context_service.zoom_out(
        context_id=body.context_id,
        targets=body.targets,
        max_perspective_tokens=body.max_perspective_tokens,
    )
