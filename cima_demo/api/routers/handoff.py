"""Portable handoff endpoints for CIMA Demonstrator."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from cima_demo.api.auth import verify_api_key
from cima_demo.api.dependencies import get_handoff_service
from cima_demo.branding import PUBLIC_HANDOFF_PREFIX

router = APIRouter(prefix=PUBLIC_HANDOFF_PREFIX, tags=["handoff"])


class CreateHandoffRequest(BaseModel):
    conversation_id: str
    source_run_id: str
    rationale: str | None = None


class RestoreHandoffRequest(BaseModel):
    handoff_id: str
    target_conversation_id: str
    target_run_id: str | None = None


class ValidateHandoffRequest(BaseModel):
    handoff_id: str


@router.post('/create')
async def create_handoff(body: CreateHandoffRequest, _auth: None = Depends(verify_api_key), handoff_service=Depends(get_handoff_service)):
    try:
        manifest = await handoff_service.create_handoff(
            conversation_id=body.conversation_id,
            source_run_id=body.source_run_id,
            rationale=body.rationale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return manifest.to_dict()


@router.post('/validate')
async def validate_handoff(body: ValidateHandoffRequest, _auth: None = Depends(verify_api_key), handoff_service=Depends(get_handoff_service)):
    validation = await handoff_service.validate_handoff(handoff_id=body.handoff_id)
    return validation.to_dict()


@router.post('/restore')
async def restore_handoff(body: RestoreHandoffRequest, _auth: None = Depends(verify_api_key), handoff_service=Depends(get_handoff_service)):
    restore = await handoff_service.restore_handoff(
        handoff_id=body.handoff_id,
        target_conversation_id=body.target_conversation_id,
        target_run_id=body.target_run_id,
    )
    return restore.to_dict()
