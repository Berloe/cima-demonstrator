"""Conversation CRUD endpoints for CIMA Demonstrator."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any

from cima_demo.api.auth import verify_api_key
from cima_demo.branding import PUBLIC_CONVERSATIONS_PREFIX
from cima_demo.api.dependencies import get_citem_store, get_db, get_geometry_commands, get_lifecycle_audit_service
from cima_demo.api.settings import get_settings
from cima_demo.witness_backend.hard_delete import HardDeleteScheduler
from cima_demo.domain.ports import CItemStorePort, RelDBPort

router = APIRouter(prefix=PUBLIC_CONVERSATIONS_PREFIX, tags=["conversations"])


class CreateConversationRequest(BaseModel):
    metadata: dict[str, object] | None = Field(default=None)


class ConversationUpsertRequest(BaseModel):
    conversation_id: str | None = None
    external_system: str | None = None
    external_conversation_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = Field(default=None)


class ConversationResponse(BaseModel):
    conversation_id: str
    created_at: str
    last_turn_at: str | None
    awaiting_user_input: bool
    turn_in_progress: bool
    turn_count: int


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest | None = None,
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
) -> ConversationResponse:
    """Create a new conversation. Returns conversation_id."""
    conversation_id = str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    return ConversationResponse(
        conversation_id=conversation_id,
        created_at=datetime.now(UTC).isoformat(),
        last_turn_at=None,
        awaiting_user_input=False,
        turn_in_progress=False,
        turn_count=0,
    )


@router.get("")
async def list_conversations(
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
) -> list[ConversationResponse]:
    """List all conversations."""
    rows = await db.list_conversations()
    return [
        ConversationResponse(
            conversation_id=str(r["conversation_id"]),
            created_at=r["created_at"].isoformat() if hasattr(r.get("created_at"), "isoformat") else str(r.get("created_at", "")),
            last_turn_at=r["last_turn_at"].isoformat() if r.get("last_turn_at") and hasattr(r["last_turn_at"], "isoformat") else None,
            awaiting_user_input=bool(r.get("awaiting_user_input", False)),
            turn_in_progress=bool(r.get("turn_in_progress", False)),
            turn_count=int(r.get("turn_count", 0)),
        )
        for r in rows
    ]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
) -> ConversationResponse:
    """Get a single conversation by ID."""
    row = await db.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        conversation_id=str(row["conversation_id"]),
        created_at=row["created_at"].isoformat() if hasattr(row.get("created_at"), "isoformat") else str(row.get("created_at", "")),
        last_turn_at=row["last_turn_at"].isoformat() if row.get("last_turn_at") and hasattr(row["last_turn_at"], "isoformat") else None,
        awaiting_user_input=bool(row.get("awaiting_user_input", False)),
        turn_in_progress=bool(row.get("turn_in_progress", False)),
        turn_count=int(row.get("turn_count", 0)),
    )


@router.get("/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    max_turns: int = 20,
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
) -> dict[str, object]:
    """Return ordered turn history for a conversation (FE-D-08)."""
    row = await db.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await db.load_recent_history(conversation_id, max_turns=max_turns)
    return {"messages": messages}


async def _delete_conversation_sync_audited(
    *,
    conversation_id: str,
    db: RelDBPort,
    citem_store: CItemStorePort,
    geometry_commands,
    lifecycle_audit_service,
    notes_prefix: list[str] | None = None,
) -> Response:
    """Synchronously hard-delete a conversation and persist a GC proof.

    This is the publication-demonstrator cleanup path.  It is deliberately
    stronger than the production/full-runtime async hard-delete path: the call
    only returns success after relational state, vector state and geometry state
    have been reconciled and a ``demo_gc_audits`` record has been written.
    """
    row = await db.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not await db.try_set_turn_in_progress(conversation_id):
        raise HTTPException(
            status_code=409,
            detail={"code": "TURN_IN_PROGRESS", "message": "Cannot delete while turn is active"},
        )

    before_counts = await lifecycle_audit_service.collect_counts(conversation_id)
    qdrant_deleted = int(before_counts.get("citems_total", 0))
    notes: list[str] = list(notes_prefix or [])
    try:
        deleted_count = await citem_store.delete_by_conversation(conversation_id)
        if deleted_count:
            qdrant_deleted = int(deleted_count)
        notes.append(f"qdrant_delete_requested={qdrant_deleted}")
        try:
            await geometry_commands.purge_conversation(conversation_id)
            notes.append("geometry_purge=completed")
        except Exception as geom_exc:
            notes.append(f"geometry_purge=error:{type(geom_exc).__name__}")
        await db.delete_conversation(conversation_id)
        # Reconcile sweep for late/duplicate cleanup on the single-node
        # demonstrator.  This keeps the publication claim tied to final state,
        # not just to a delete request.
        await citem_store.delete_by_conversation(conversation_id)
        try:
            await geometry_commands.purge_conversation(conversation_id)
        except Exception:
            pass
        after_counts = await lifecycle_audit_service.collect_counts(conversation_id)
        audit = await lifecycle_audit_service.audit_delete_outcome(
            conversation_id=conversation_id,
            before_counts=before_counts,
            after_counts=after_counts,
            metrics={
                "mode": "sync_publication_delete",
                "qdrant_delete_requested": qdrant_deleted,
                "reconcile_sweep": True,
            },
            notes=notes,
        )
        if not bool(audit.consistency.get("cleanup_ok", False)):
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "GC_RECONCILE_FAILED",
                    "message": "Conversation deleted but residual demonstrator artifacts remain",
                    "consistency": audit.consistency,
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        after_counts = await lifecycle_audit_service.collect_counts(conversation_id)
        await lifecycle_audit_service.audit_delete_outcome(
            conversation_id=conversation_id,
            before_counts=before_counts,
            after_counts=after_counts,
            metrics={
                "mode": "sync_publication_delete",
                "qdrant_delete_requested": qdrant_deleted,
                "reconcile_sweep": False,
            },
            notes=notes,
            error_class=type(exc).__name__,
        )
        await db.release_turn_in_progress(conversation_id)
        raise

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
    citem_store: CItemStorePort = Depends(get_citem_store),
    geometry_commands = Depends(get_geometry_commands),
    lifecycle_audit_service = Depends(get_lifecycle_audit_service),
) -> Response:
    """Delete a conversation and all its data (U-01).

    In full runtime this public UI endpoint preserves the async hard-delete
    plane.  The /cima/v1 compatibility endpoint used by the publication harness
    intentionally forces the synchronous audited path so C6 is proven by final
    state rather than by HTTP 202 acceptance.
    """
    row = await db.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    settings = get_settings()
    if settings.runtime_mode == "full" and hasattr(db, "begin_hard_delete"):
        if not await db.try_set_turn_in_progress(conversation_id):
            raise HTTPException(
                status_code=409,
                detail={"code": "TURN_IN_PROGRESS", "message": "Cannot delete while turn is active"},
            )
        scheduler = HardDeleteScheduler(db)
        result = await scheduler.request(conversation_id, reason="USER_REQUEST")
        await db.release_turn_in_progress(conversation_id)
        if not result.accepted:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "conversation_id": conversation_id,
                    "status": "deleting",
                    "accepted": False,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "conversation_id": conversation_id,
                "status": "deleting",
                "accepted": True,
                "delete_run_id": result.delete_run_id,
            },
        )

    # Standalone demonstrator path remains synchronous and audited.
    return await _delete_conversation_sync_audited(
        conversation_id=conversation_id,
        db=db,
        citem_store=citem_store,
        geometry_commands=geometry_commands,
        lifecycle_audit_service=lifecycle_audit_service,
        notes_prefix=["endpoint=/cima_demo/conversations"],
    )

# Compatibility surface used by the OpenAI/Open-Scenarios harness.
# The demonstrator public UI keeps /cima_demo/*, but /cima/v1/* must be real
# endpoints rather than relying on client-side fallback noise.
cima_v1_router = APIRouter(prefix="/cima/v1/conversations", tags=["conversation:v1-compat"])


@cima_v1_router.post("/upsert", status_code=status.HTTP_200_OK)
async def upsert_conversation_v1(
    body: ConversationUpsertRequest | None = None,
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
) -> ConversationResponse:
    body = body or ConversationUpsertRequest()
    conversation_id = body.conversation_id or str(uuid.uuid4())
    await db.create_conversation(conversation_id)
    row = await db.get_conversation(conversation_id)
    created_at = datetime.now(UTC).isoformat()
    if row is not None and row.get("created_at") is not None:
        created = row.get("created_at")
        created_at = created.isoformat() if hasattr(created, "isoformat") else str(created)
    return ConversationResponse(
        conversation_id=conversation_id,
        created_at=created_at,
        last_turn_at=None if row is None or not row.get("last_turn_at") else (row["last_turn_at"].isoformat() if hasattr(row["last_turn_at"], "isoformat") else str(row["last_turn_at"])),
        awaiting_user_input=bool(row.get("awaiting_user_input", False)) if row else False,
        turn_in_progress=bool(row.get("turn_in_progress", False)) if row else False,
        turn_count=int(row.get("turn_count", 0)) if row else 0,
    )


@cima_v1_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_v1(
    conversation_id: str,
    purge: bool = True,
    _auth: None = Depends(verify_api_key),
    db: RelDBPort = Depends(get_db),
    citem_store: CItemStorePort = Depends(get_citem_store),
    geometry_commands = Depends(get_geometry_commands),
    lifecycle_audit_service = Depends(get_lifecycle_audit_service),
) -> Response:
    # Publication harness / CIMA v1 compatibility path: hard delete is
    # synchronous and audited even when the full runtime also supports an async
    # hard-delete plane.  This aligns the C6 claim with final, inspectable state:
    # DELETE success means a demo_gc_audits row proves cleanup_ok,
    # conversation_deleted and qdrant_zeroed.
    return await _delete_conversation_sync_audited(
        conversation_id=conversation_id,
        db=db,
        citem_store=citem_store,
        geometry_commands=geometry_commands,
        lifecycle_audit_service=lifecycle_audit_service,
        notes_prefix=[
            "endpoint=/cima/v1/conversations",
            f"purge={bool(purge)}",
        ],
    )
