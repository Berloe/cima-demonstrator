"""CIMA Demonstrator entry point — starts uvicorn with the FastAPI application."""
from __future__ import annotations

import logging

import uvicorn

from cima_demo.api.settings import get_settings


def main() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    # httpx/httpcore son extremadamente verbosos en DEBUG — silenciarlos a WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Silenciar health checks (/readyz, /healthz) del access log de uvicorn
    class _HealthFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not (
                ("GET /readyz" in msg or "GET /healthz" in msg)
                and "200" in msg
            )

    for _name in ("uvicorn.access", "uvicorn"):
        logging.getLogger(_name).addFilter(_HealthFilter())

    uvicorn.run(
        "cima_demo.api.app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    settings = get_settings()
    if settings.remote_debug:
        import debugpy
        debugpy.listen(("0.0.0.0", settings.remote_debug_port))
        debugpy.wait_for_client()
    main()
