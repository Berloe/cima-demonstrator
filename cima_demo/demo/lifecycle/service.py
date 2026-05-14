"""Durable GC / lifecycle auditing for CIMA Demonstrator.

This service turns background lifecycle actions into durable, inspectable
artifacts and database rows. It does not own the visible transcript and it does
not participate in turn authority; it only governs memory lifecycle evidence.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from cima_demo.demo.contracts import GCAuditRecord
from cima_demo.domain.ports import CItemStorePort, RelDBPort


class DemoLifecycleAuditService:
    def __init__(
        self,
        *,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        memory_service: Any,
        artifacts_root: Path,
    ) -> None:
        self._db = rel_db
        self._store = citem_store
        self._memory = memory_service
        self._root = Path(artifacts_root)

    def _trace_path(self, conversation_id: str) -> Path:
        return self._root / "gc" / f"gc_trace_{conversation_id}.json"

    def _load_trace(self, conversation_id: str) -> dict[str, Any]:
        path = self._trace_path(conversation_id)
        if not path.exists():
            return {"conversation_id": conversation_id, "events": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"conversation_id": conversation_id, "events": []}

    def _write_trace(self, conversation_id: str, payload: dict[str, Any]) -> None:
        path = self._trace_path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def collect_counts(self, conversation_id: str) -> dict[str, Any]:
        counts = await self._db.load_demo_conversation_counts(conversation_id)
        items = await self._store.fetch_by_conversation(conversation_id, scope_status=None)
        counts = dict(counts)
        counts.update({
            "citems_total": len(items),
            "citems_active": sum(1 for item in items if item.scope_status == "active"),
            "citems_archived": sum(1 for item in items if item.scope_status == "archived"),
            "citems_global": sum(1 for item in items if item.scope == "global"),
            "citems_episodic": sum(1 for item in items if item.scope == "episodic"),
        })
        return counts

    async def record(self, record: GCAuditRecord) -> GCAuditRecord:
        payload = record.to_dict()
        await self._db.save_demo_gc_audit(payload)
        trace = self._load_trace(record.conversation_id)
        events = list(trace.get("events", []))
        events.append(payload)
        trace = {
            "conversation_id": record.conversation_id,
            "event_count": len(events),
            "events": events,
        }
        self._write_trace(record.conversation_id, trace)
        return record

    async def run_scope_transition_cycle(self, conversation_id: str, chm_reference_counts: dict[str, int]) -> GCAuditRecord:
        before = await self.collect_counts(conversation_id)
        report = await self._memory.check_promotions_detailed(conversation_id, chm_reference_counts)
        after = await self.collect_counts(conversation_id)
        record = GCAuditRecord(
            audit_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            action="scope_transition_cycle",
            phase="promotion",
            before_counts=before,
            after_counts=after,
            metrics=report,
            consistency={
                "promotion_delta_matches": int(report.get("n_promoted", 0)) == max(0, int(after.get("citems_global", 0)) - int(before.get("citems_global", 0))),
            },
            notes=[f"promoted={report.get('n_promoted', 0)}", f"demoted={report.get('n_demoted', 0)}"],
        )
        return await self.record(record)

    async def run_stale_maintenance_cycle(self, conversation_id: str) -> GCAuditRecord:
        before = await self.collect_counts(conversation_id)
        forget_report = await self._memory.run_forget_cycle_detailed(conversation_id)
        dedup_report = await self._memory.run_dedup_cycle_detailed(conversation_id)
        l2_triggered = bool(await self._memory.trigger_l2_check(conversation_id))
        after = await self.collect_counts(conversation_id)
        metrics = {
            "forget": forget_report,
            "dedup": dedup_report,
            "l2_triggered": l2_triggered,
        }
        record = GCAuditRecord(
            audit_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            action="stale_maintenance_cycle",
            phase="forget",
            before_counts=before,
            after_counts=after,
            metrics=metrics,
            consistency={
                "post_counts_collected": True,
                "archived_nonnegative": int(after.get("citems_archived", 0)) >= 0,
            },
            notes=[
                f"attenuated={forget_report.get('n_attenuated', 0)}",
                f"purged={forget_report.get('n_purged', 0)}",
                f"dedup_archived={dedup_report.get('n_archived', 0)}",
                f"l2_triggered={l2_triggered}",
            ],
        )
        return await self.record(record)

    async def audit_delete_outcome(
        self,
        *,
        conversation_id: str,
        before_counts: dict[str, Any],
        after_counts: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        notes: list[str] | None = None,
        error_class: str | None = None,
    ) -> GCAuditRecord:
        metrics = metrics or {}
        notes = list(notes or [])
        relevant_after = {k: int(v or 0) for k, v in after_counts.items() if k != "citems_total"}
        residual_relational = {k: v for k, v in relevant_after.items() if k.startswith("citems_") is False and v > 0}
        cleanup_ok = not residual_relational and int(after_counts.get("citems_total", 0)) == 0 and int(after_counts.get("conversations", 0)) == 0
        record = GCAuditRecord(
            audit_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            action="conversation_delete",
            status="ok" if cleanup_ok and error_class is None else "error",
            phase="delete",
            before_counts=before_counts,
            after_counts=after_counts,
            metrics=metrics,
            consistency={
                "cleanup_ok": cleanup_ok,
                "conversation_deleted": int(after_counts.get("conversations", 0)) == 0,
                "qdrant_zeroed": int(after_counts.get("citems_total", 0)) == 0,
                "residual_relational": residual_relational,
            },
            notes=notes,
            error_class=error_class,
        )
        return await self.record(record)
    async def load_audits(self, conversation_id: str) -> list[dict[str, Any]]:
        return await self._db.load_demo_gc_audits(conversation_id)

