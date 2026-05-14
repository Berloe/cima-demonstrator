"""WorkspaceCleanerWorker — periodic reset of the ephemeral workspace volume."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


class WorkspaceCleanerWorker:
    """Deletes per-conversation subdirectories under *workspace_dir* based on TTL.

    Design:
    - ttl_hours=0: disabled — nothing is ever deleted.
    - ttl_hours>0: runs at 00:00 UTC daily; removes entries older than ttl_hours.
    - workspace_dir itself is never deleted.
    """

    def __init__(self, workspace_dir: Path, *, ttl_hours: int = 24) -> None:
        self._workspace = workspace_dir
        self._ttl_hours = ttl_hours
        self._task: asyncio.Task[None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._ttl_hours == 0:
            log.info("WorkspaceCleanerWorker disabled (ttl_hours=0)")
            return
        if self._task is not None and not self._task.done():
            log.warning("WorkspaceCleanerWorker already running")
            return
        self._task = asyncio.create_task(self._loop(), name="workspace_cleaner")
        next_run = self._seconds_until_midnight()
        log.info(
            "WorkspaceCleanerWorker started — ttl=%dh, next clean in %.0fs (00:00 UTC)",
            self._ttl_hours, next_run,
        )

    async def stop(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        log.info("WorkspaceCleanerWorker stopped")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _seconds_until_midnight() -> float:
        now = datetime.now(UTC)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (tomorrow - now).total_seconds()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._seconds_until_midnight())
            try:
                await asyncio.to_thread(self._clean)
            except Exception:
                log.exception("WorkspaceCleanerWorker: clean failed")

    def _clean(self) -> None:
        if not self._workspace.exists():
            return
        cutoff = datetime.now(UTC) - timedelta(hours=self._ttl_hours)
        removed = 0
        errors = 0
        for entry in self._workspace.iterdir():
            try:
                mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=UTC)
                if mtime >= cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except Exception:
                log.exception("WorkspaceCleanerWorker: failed to remove %s", entry)
                errors += 1
        log.info(
            "WorkspaceCleanerWorker: cleaned %d entries (%d errors, ttl=%dh) from %s",
            removed, errors, self._ttl_hours, self._workspace,
        )
