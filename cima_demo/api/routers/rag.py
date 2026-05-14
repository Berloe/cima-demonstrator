from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, File, Form, Header, HTTPException, UploadFile, status

from cima_demo.api.auth import verify_api_key
from cima_demo.api.conversation_guard import ensure_active_conversation
from cima_demo.api.dependencies import get_db, get_source_registration_service
from cima_demo.witness_backend.events import TraceContext

router = APIRouter(tags=["rag"])


def _pick_conversation_id(form_value: str | None, header_value: str | None, payload_value: str | None = None) -> str:
    conversation_id = (form_value or payload_value or header_value or "").strip()
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    return conversation_id


def _trace(*, request_id: str | None, conversation_id: str) -> TraceContext:
    return TraceContext(
        request_id=request_id or conversation_id,
        correlation_id=request_id or conversation_id,
        actor_kind="user",
    )


async def _register_uploads(
    *,
    uploads: list[UploadFile],
    conversation_id: str,
    request_id: str | None,
    service,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    trace = _trace(request_id=request_id, conversation_id=conversation_id)
    for upload in uploads:
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail=f"Empty file upload: {upload.filename or 'upload.bin'}")
        result = await service.register_file_upload(
            conversation_id=conversation_id,
            filename=upload.filename or "upload.bin",
            mime_type=upload.content_type or "application/octet-stream",
            content=payload,
            trace=trace,
        )
        documents.append(
            {
                "file_id": result.file_id,
                "conversation_id": conversation_id,
                "filename": upload.filename or "upload.bin",
                "status": "queued",
                "outbox_id": result.outbox_id,
            }
        )
    return documents


@router.get("/health")
async def rag_health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/embed", status_code=status.HTTP_202_ACCEPTED)
async def embed(
    payload: dict[str, Any] = Body(default_factory=dict),
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
    service=Depends(get_source_registration_service),
    x_librechat_conversation_id: str | None = Header(default=None, alias="X-LibreChat-Conversation-Id"),
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
):
    """LibreChat RAG compatibility endpoint for raw text ingestion.

    In the agreed Option B, retrieval context is built exclusively by CIMA later;
    this endpoint only materializes a text source into the ingestion pipeline.
    """
    conv_id = _pick_conversation_id(
        None,
        x_librechat_conversation_id,
        str(payload.get("conversation_id") or payload.get("conversationId") or "") or None,
    )
    ensure_active_conversation(await db.get_conversation(conv_id))
    text = str(payload.get("text") or payload.get("content") or "").strip()
    if not text and isinstance(payload.get("chunks"), list):
        text = "\n\n".join(str(item).strip() for item in payload["chunks"] if str(item).strip())
    if not text:
        return {"accepted": True, "conversation_id": conv_id, "status": "noop", "reason": "empty_text"}
    result = await service.register_text(
        conversation_id=conv_id,
        text=text,
        role=None,
        source_kind="file_text",
        external_provider="librechat_rag",
        external_conversation_id=conv_id,
        external_message_id=str(payload.get("file_id") or payload.get("document_id") or payload.get("id") or "") or None,
        displayable=False,
        processable=True,
        trace=_trace(request_id=x_request_id, conversation_id=conv_id),
    )
    return {
        "accepted": True,
        "conversation_id": conv_id,
        "source_id": result.source_id,
        "status": "queued",
        "outbox_id": result.outbox_id,
    }


@router.post("/embed-upload", status_code=status.HTTP_202_ACCEPTED)
async def embed_upload(
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    conversation_id: str | None = Form(default=None),
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
    service=Depends(get_source_registration_service),
    x_librechat_conversation_id: str | None = Header(default=None, alias="X-LibreChat-Conversation-Id"),
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
):
    """LibreChat RAG compatibility upload endpoint.

    Accepts both the singular ``file`` field used by the local OpenAPI and the
    repeated ``files`` field used by some LibreChat RAG clients.
    """
    conv_id = _pick_conversation_id(conversation_id, x_librechat_conversation_id)
    ensure_active_conversation(await db.get_conversation(conv_id))
    uploads: list[UploadFile] = []
    if file is not None:
        uploads.append(file)
    if files:
        uploads.extend(files)
    if not uploads:
        raise HTTPException(status_code=422, detail="file or files is required")
    documents = await _register_uploads(
        uploads=uploads,
        conversation_id=conv_id,
        request_id=x_request_id,
        service=service,
    )
    first = documents[0]
    return {
        "accepted": True,
        "conversation_id": conv_id,
        "file_id": first["file_id"],
        "status": "queued",
        "outbox_id": first["outbox_id"],
        "documents": documents,
    }


def _empty_query_response(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    query = "" if payload is None else str(payload.get("query") or payload.get("text") or "")
    return {
        "query": query,
        "context": "",
        "text": "",
        "sources": [],
        "source_documents": [],
        "sourceDocuments": [],
        "results": [],
    }


@router.post("/query")
async def query(payload: dict[str, Any] = Body(default_factory=dict), _auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    """Option B: keep LibreChat-compatible shape but do not inject parallel RAG context."""
    return _empty_query_response(payload)


@router.post("/query_multiple")
async def query_multiple(payload: dict[str, Any] = Body(default_factory=dict), _auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    return _empty_query_response(payload)


@router.post("/search")
async def search(payload: dict[str, Any] = Body(default_factory=dict), _auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    return _empty_query_response(payload)


@router.post("/delete")
async def delete(payload: dict[str, Any] = Body(default_factory=dict), _auth: None = Depends(verify_api_key), db=Depends(get_db)) -> dict[str, Any]:
    file_id = str(payload.get("file_id") or payload.get("document_id") or payload.get("id") or "").strip()
    if not file_id:
        return {"accepted": True, "status": "noop", "reason": "missing_file_id"}
    return await _delete_document_record(file_id=file_id, db=db)


@router.get("/documents/{file_id}")
async def get_document(
    file_id: str,
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
):
    record = await db.get_file_record(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "file_id": record.file_id,
        "conversation_id": record.conversation_id,
        "filename": record.filename,
        "mime_type": record.mime_type,
        "size_bytes": record.size_bytes,
        "status": record.status,
        "chunk_count": record.chunk_count,
        "blob_path": record.blob_path,
        "error_message": record.error_message,
    }


async def _delete_document_record(*, file_id: str, db) -> dict[str, Any]:
    record = await db.get_file_record(file_id)
    if record is None:
        return {"accepted": True, "file_id": file_id, "status": "not_found"}
    if hasattr(db, "update_file_record"):
        await db.update_file_record(file_id, status="DELETED", chunk_count=getattr(record, "chunk_count", 0), citem_ids=getattr(record, "citem_ids", []))
    return {"accepted": True, "file_id": file_id, "conversation_id": record.conversation_id, "status": "deleted"}


@router.delete("/documents/{file_id}")
async def delete_document(
    file_id: str,
    _auth: None = Depends(verify_api_key),
    db=Depends(get_db),
):
    return await _delete_document_record(file_id=file_id, db=db)
