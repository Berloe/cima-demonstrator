"""Health endpoints: GET /health, GET /readyz (KIMA_API_Layer_v0.5 §9)."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request

from cima_demo.api.dependencies import get_citem_store, get_db
from cima_demo.domain.ports import CItemStorePort, RelDBPort

router = APIRouter()


@router.get("/health")
async def health(
    db: RelDBPort = Depends(get_db),
    citem_store: CItemStorePort = Depends(get_citem_store),
) -> dict[str, object]:
    """Liveness probe — checks all critical dependencies."""
    checks: dict[str, str] = {}
    overall = "ok"

    for name, checker in [
        ("postgres", db.ping),
        ("qdrant",   citem_store.ping),
    ]:
        try:
            ok = await checker()
            checks[name] = "ok" if ok else "error: ping returned False"
            if not ok:
                overall = "degraded"
        except Exception as exc:
            checks[name] = f"error: {exc}"
            overall = "degraded"

    return {
        "status":    overall,
        "checks":    checks,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    """Readiness probe — checks if the app is ready to serve traffic."""
    try:
        db: RelDBPort = request.app.state.db
        ok = await db.ping()
        if not ok:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="Database not ready")
    except AttributeError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="App not initialized") from exc

    return {"status": "ready"}
