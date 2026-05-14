"""Run journal inspection endpoints for the CIMA Demonstrator."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cima_demo.api.auth import verify_api_key
from cima_demo.api.dependencies import get_context_service, get_lifecycle_audit_service, get_run_journal
from cima_demo.branding import PUBLIC_RUNS_PREFIX

router = APIRouter(prefix=PUBLIC_RUNS_PREFIX, tags=["runs"])


@router.get('/{run_id}')
async def get_run_manifest(
    run_id: str,
    _auth: None = Depends(verify_api_key),
    run_journal=Depends(get_run_journal),
):
    bundle = await run_journal.load_bundle(run_id)
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return bundle.manifest


@router.get('/{run_id}/bundle')
async def get_run_bundle(
    run_id: str,
    _auth: None = Depends(verify_api_key),
    run_journal=Depends(get_run_journal),
):
    bundle = await run_journal.load_bundle(run_id)
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return bundle.to_dict()


@router.get('/{run_id}/context-snapshots')
async def get_run_context_snapshots(
    run_id: str,
    _auth: None = Depends(verify_api_key),
    context_service=Depends(get_context_service),
):
    snapshots = await context_service.load_context_snapshots_for_run_public(run_id)
    if not snapshots:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No context snapshots for run")
    return {"run_id": run_id, "snapshots": snapshots}


@router.get('/{run_id}/replay')
async def replay_run(
    run_id: str,
    _auth: None = Depends(verify_api_key),
    run_journal=Depends(get_run_journal),
    context_service=Depends(get_context_service),
):
    bundle = await run_journal.load_bundle(run_id)
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    snapshots = await context_service.load_context_snapshots_for_run_public(run_id)
    return {
        "run": bundle.manifest,
        "phases": bundle.phases,
        "checkpoints": bundle.checkpoints,
        "context_snapshots": snapshots,
    }


@router.get('/conversations/{conversation_id}/prompt-trace')
async def get_latest_prompt_trace(
    conversation_id: str,
    _auth: None = Depends(verify_api_key),
    run_journal=Depends(get_run_journal),
):
    trace = await run_journal.load_latest_prompt_trace(conversation_id)
    if trace is None or not trace.get("prompt_trace_available"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No prompt trace for conversation")
    return trace


@router.get('/conversations/{conversation_id}/gc-audits')
async def get_gc_audits(
    conversation_id: str,
    _auth: None = Depends(verify_api_key),
    lifecycle_audit_service=Depends(get_lifecycle_audit_service),
):
    audits = await lifecycle_audit_service.load_audits(conversation_id)
    return {"conversation_id": conversation_id, "gc_audits": audits}
