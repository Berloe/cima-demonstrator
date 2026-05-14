from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status


def ensure_active_conversation(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    state = str(row.get("status") or "ACTIVE").upper()
    if state != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONVERSATION_DELETING", "status": state},
        )
    return row
