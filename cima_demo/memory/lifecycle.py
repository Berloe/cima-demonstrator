"""LifecycleService — forget cycle, deduplication, and promotions."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cima_demo.application.stream_manager import StreamManager

from cima_demo.domain.entities import CItem
from cima_demo.domain.operations import (
    can_purge,
    is_promotion_eligible,
    should_attenuate,
)
from cima_demo.domain.ports import (
    CItemStorePort,
    RelDBPort,
)
from cima_demo.domain.value_objects import (
    ForgetParams,
    PromotionPolicy,
)

log = logging.getLogger(__name__)


class LifecycleService:
    """Forget cycle, deduplication, and promotion management.

    Extracted from MemoryService to own:
    run_forget_cycle, run_dedup_cycle, check_promotions.
    """

    def __init__(
        self,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        stream_manager: "StreamManager",
        forget_params: ForgetParams | None = None,
        promotion_policy: PromotionPolicy | None = None,
    ) -> None:
        self._db = rel_db
        self._cstore = citem_store
        self._stream = stream_manager
        self._forget_params = forget_params or ForgetParams.default()
        self._promotion_policy = promotion_policy or PromotionPolicy.default()

    async def _audit_event(
        self,
        *,
        conversation_id: str,
        citem_id: str,
        event_type: str,
        old_value: str | None = None,
        new_value: str | None = None,
    ) -> None:
        callback = getattr(self._db, "append_citem_audit_event", None)
        if callback is None:
            return
        try:
            await callback(
                conversation_id=conversation_id,
                citem_id=citem_id,
                event_type=event_type,
                old_value=old_value,
                new_value=new_value,
            )
        except Exception:
            log.debug("LifecycleService audit event failed for %s/%s", conversation_id, citem_id, exc_info=True)

    # ── Forget cycle ──────────────────────────────────────────────────────────

    async def run_forget_cycle(self, conversation_id: str) -> tuple[int, int]:
        report = await self.run_forget_cycle_detailed(conversation_id)
        return int(report["n_attenuated"]), int(report["n_purged"])

    async def run_forget_cycle_detailed(self, conversation_id: str) -> dict[str, object]:
        """Attenuation + purge cycle with item-level audit details."""
        now = datetime.now(UTC).timestamp()
        params = self._forget_params

        active_items = await self._cstore.fetch_by_conversation(
            conversation_id, scope_status="active"
        )
        attenuated_ids: list[str] = []
        archived_ids: list[str] = []
        for item in active_items:
            age_days = (now - item.created_at.timestamp()) / 86400.0
            if should_attenuate(item, age_days, params):
                old_status = item.scope_status
                item.attenuate(threshold=params.attenuation_threshold)
                await self._cstore.update_field(item.citem_id, "importance", item.importance)
                attenuated_ids.append(item.citem_id)
                if item.scope_status == "archived":
                    await self._cstore.update_field(item.citem_id, "scope_status", "archived")
                    archived_ids.append(item.citem_id)
                    await self._audit_event(
                        conversation_id=conversation_id,
                        citem_id=item.citem_id,
                        event_type="ARCHIVED",
                        old_value=f"scope_status:{old_status}",
                        new_value="scope_status:archived",
                    )
                log.debug(
                    "Attenuated %s → importance=%.4f scope_status=%s",
                    item.citem_id, item.importance, item.scope_status,
                )

        archived_items = await self._cstore.fetch_by_conversation(
            conversation_id, scope_status="archived"
        )
        purged_ids: list[str] = []
        for item in archived_items:
            archived_at_ts = item.archived_at_unix or item.created_at.timestamp()
            days_since_archived = (now - archived_at_ts) / 86400.0
            if can_purge(item, days_since_archived, params):
                await self._audit_event(
                    conversation_id=conversation_id,
                    citem_id=item.citem_id,
                    event_type="PURGED",
                    old_value="scope_status:archived",
                    new_value="deleted",
                )
                await self._cstore.delete(item.citem_id)
                purged_ids.append(item.citem_id)

        log.info(
            "Forget cycle for %s: %d attenuated, %d purged",
            conversation_id, len(attenuated_ids), len(purged_ids),
        )
        return {
            "conversation_id": conversation_id,
            "n_attenuated": len(attenuated_ids),
            "n_archived": len(archived_ids),
            "n_purged": len(purged_ids),
            "attenuated_ids": attenuated_ids,
            "archived_ids": archived_ids,
            "purged_ids": purged_ids,
        }

    # ── Deduplication ─────────────────────────────────────────────────────────

    async def run_dedup_cycle(self, conversation_id: str) -> int:
        report = await self.run_dedup_cycle_detailed(conversation_id)
        return int(report["n_archived"])

    async def run_dedup_cycle_detailed(self, conversation_id: str) -> dict[str, object]:
        """Archive duplicate active C-Items (same content_hash), keeping the oldest."""
        active_items = await self._cstore.fetch_by_conversation(
            conversation_id, scope_status="active"
        )
        by_hash: dict[str, list[CItem]] = {}
        for item in active_items:
            if item.content_hash:
                by_hash.setdefault(item.content_hash, []).append(item)

        archived_duplicate_ids: list[str] = []
        for items in by_hash.values():
            if len(items) <= 1:
                continue
            items.sort(key=lambda x: x.created_at)
            for dup in items[1:]:
                await self._cstore.update_field(dup.citem_id, "scope_status", "archived")
                archived_duplicate_ids.append(dup.citem_id)
                await self._audit_event(
                    conversation_id=conversation_id,
                    citem_id=dup.citem_id,
                    event_type="ARCHIVED",
                    old_value="scope_status:active",
                    new_value="scope_status:archived",
                )

        if archived_duplicate_ids:
            log.info(
                "run_dedup_cycle: %d duplicate(s) archived for %s",
                len(archived_duplicate_ids), conversation_id,
            )
        return {
            "conversation_id": conversation_id,
            "n_archived": len(archived_duplicate_ids),
            "archived_duplicate_ids": archived_duplicate_ids,
        }

    # ── Promotion ─────────────────────────────────────────────────────────────

    async def check_promotions(
        self,
        conversation_id: str,
        chm_reference_counts: dict[str, int],
    ) -> tuple[int, int]:
        report = await self.check_promotions_detailed(conversation_id, chm_reference_counts)
        return int(report["n_promoted"]), int(report["n_demoted"])

    async def check_promotions_detailed(
        self,
        conversation_id: str,
        chm_reference_counts: dict[str, int],
    ) -> dict[str, object]:
        """Promote eligible episodic items to global; demote stale global items."""
        active_items = await self._cstore.fetch_by_conversation(
            conversation_id, scope_status="active"
        )
        promoted_ids: list[str] = []
        demoted_ids: list[str] = []
        for item in active_items:
            if item.scope == "episodic":
                count = chm_reference_counts.get(item.citem_id, 0)
                if is_promotion_eligible(item, count, self._promotion_policy):
                    item.promote_to_global()
                    await self._cstore.update_field(item.citem_id, "scope", "global")
                    promoted_ids.append(item.citem_id)
                    await self._audit_event(
                        conversation_id=conversation_id,
                        citem_id=item.citem_id,
                        event_type="PROMOTED",
                        old_value="scope:episodic",
                        new_value="scope:global",
                    )
            elif item.scope == "global":
                if item.importance < self._promotion_policy.min_importance:
                    await self._cstore.update_field(item.citem_id, "scope", "episodic")
                    demoted_ids.append(item.citem_id)
                    await self._audit_event(
                        conversation_id=conversation_id,
                        citem_id=item.citem_id,
                        event_type="DEMOTED",
                        old_value="scope:global",
                        new_value="scope:episodic",
                    )
        if promoted_ids or demoted_ids:
            log.info(
                "check_promotions: %d promoted, %d demoted for %s",
                len(promoted_ids), len(demoted_ids), conversation_id,
            )
        return {
            "conversation_id": conversation_id,
            "n_promoted": len(promoted_ids),
            "n_demoted": len(demoted_ids),
            "promoted_ids": promoted_ids,
            "demoted_ids": demoted_ids,
        }
