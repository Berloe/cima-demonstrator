"""ForgetCycleWorker — background task executing CIMA A-7 forget cycle (§4.2)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from cima_demo.memory.service import MemoryService
from cima_demo.domain.ports import RelDBPort

log = logging.getLogger(__name__)


class ForgetCycleWorker:
    """Periodically runs memory lifecycle maintenance.

    Pass 1 — scope transitions (ALL eligible conversations):
        Promotions (episodic → global) and demotions (global → episodic).

    Pass 2 — forget + dedup (STALE conversations only):
        Attenuation, purge, duplicate cleanup and L2 autopromotion checks.

    When an audit service is provided, every per-conversation maintenance pass is
    turned into a durable GC trace artifact plus a DB audit row.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        rel_db: RelDBPort,
        interval_secs: int = 18000,
        stale_hours: int = 24,
        audit_service: Any | None = None,
    ) -> None:
        self._memory = memory_service
        self._db = rel_db
        self._interval = interval_secs
        self._stale_delta = timedelta(hours=stale_hours)
        self._task: asyncio.Task[None] | None = None
        self._audit = audit_service

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            log.warning("ForgetCycleWorker already running")
            return
        self._task = asyncio.create_task(self._loop(), name="forget_cycle_worker")
        log.info(
            "ForgetCycleWorker started (interval=%ds, stale=%dh)",
            self._interval,
            int(self._stale_delta.total_seconds()) // 3600,
        )

    async def stop(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        log.info("ForgetCycleWorker stopped")

    async def _loop(self) -> None:
        while True:
            try:
                await self._run_once()
            except Exception:
                log.exception("ForgetCycleWorker: unhandled error in cycle")
            await asyncio.sleep(self._interval)

    async def _run_once(self) -> None:
        cutoff = datetime.now(UTC) - self._stale_delta
        rows = await self._db.list_conversations()
        eligible = [r for r in rows if not r.get("turn_in_progress")]

        n_promoted = n_demoted = 0
        for row in eligible:
            conv_id = str(row["conversation_id"])
            try:
                chm_counts = await self._db.load_chm_refs(conv_id)
                if self._audit is not None:
                    record = await self._audit.run_scope_transition_cycle(conv_id, chm_counts)
                    metrics = record.metrics
                    n_promoted += int(metrics.get("n_promoted", 0))
                    n_demoted += int(metrics.get("n_demoted", 0))
                else:
                    promoted, demoted = await self._memory.check_promotions(conv_id, chm_counts)
                    n_promoted += promoted
                    n_demoted += demoted
            except Exception:
                log.exception("ForgetCycleWorker pass1: scope transitions failed for %s", conv_id)

        log.info(
            "ForgetCycleWorker pass1: %d conversations — %d promoted, %d demoted",
            len(eligible), n_promoted, n_demoted,
        )

        n_attenuated = n_purged = n_deduped = n_stale = n_l2_promoted = 0
        for row in eligible:
            last_active: datetime | None = row.get("last_turn_at") or row.get("created_at")
            if last_active is None:
                continue
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=UTC)
            if last_active >= cutoff:
                continue

            conv_id = str(row["conversation_id"])
            try:
                if self._audit is not None:
                    record = await self._audit.run_stale_maintenance_cycle(conv_id)
                    metrics = record.metrics
                    forget = metrics.get("forget", {}) if isinstance(metrics, dict) else {}
                    dedup = metrics.get("dedup", {}) if isinstance(metrics, dict) else {}
                    n_attenuated += int(forget.get("n_attenuated", 0))
                    n_purged += int(forget.get("n_purged", 0))
                    n_deduped += int(dedup.get("n_archived", 0))
                    if bool(metrics.get("l2_triggered", False)):
                        n_l2_promoted += 1
                else:
                    att, purged = await self._memory.run_forget_cycle(conv_id)
                    n_attenuated += att
                    n_purged += purged
                    n_deduped += await self._memory.run_dedup_cycle(conv_id)
                    if await self._memory.trigger_l2_check(conv_id):
                        n_l2_promoted += 1
                n_stale += 1
            except Exception:
                log.exception("ForgetCycleWorker pass2: forget/dedup failed for %s", conv_id)

        log.info(
            "ForgetCycleWorker pass2: %d stale conversations — %d attenuated, %d purged, %d deduped",
            n_stale, n_attenuated, n_purged, n_deduped,
        )
        if n_l2_promoted:
            log.info(
                "ForgetCycleWorker pass3: L2 AutoPromote triggered for %d conversation(s)",
                n_l2_promoted,
            )
