"""API Key authentication (KIMA_API_Layer_v0.5 §10)."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from cima_demo.api.settings import get_settings


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = get_settings()
    if not settings.api_key_required or (settings.runtime_mode == "standalone" and not settings.api_key):
        return
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def verify_api_key_openai(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Accept X-API-Key or Authorization: Bearer <key>."""
    settings = get_settings()
    if not settings.api_key_required or (settings.runtime_mode == "standalone" and not settings.api_key):
        return
    key = x_api_key
    if not key and authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
    if not key or key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
