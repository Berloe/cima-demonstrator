"""PostgreSQLAdapter → RelDBPort (KIMA_Infrastructure_Layer_v0.6 §3.1)."""
from __future__ import annotations

import json
import logging
import uuid as _uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg

from cima_demo.domain.entities import (
    ConflictLogEntry,
    Plan,
    PlanStep,
    SummaryNode,
    TaskMemory,
)
from cima_demo.domain.ports import RelDBPort
from cima_demo.domain.value_objects import CognitivePhase

log = logging.getLogger(__name__)


def _parse_ts(value: object) -> "datetime | None":
    """Convert an ISO-8601 string or datetime to datetime, or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None



def _parse_jsonb(value: object, default: object = None) -> object:
    """Safely load a JSONB column that may be returned as a str (TEXT column) or already parsed."""
    import json as _json
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return _json.loads(value)
        except (ValueError, TypeError):
            return default
    return value

class PostgreSQLAdapter(RelDBPort):
    """asyncpg-based implementation of RelDBPort.

    Handles: conversations, task_memory, summary_pyramid, plans/steps,
             conflict_log, retrieval_telemetry.
    Does NOT handle citems — those live exclusively in Qdrant.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Conversation ──────────────────────────────────────────────────────────

    async def create_conversation(self, conversation_id: str) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                    INSERT INTO conversations (conversation_id, created_at, status)
                    VALUES ($1, NOW(), 'ACTIVE')
                    ON CONFLICT (conversation_id) DO NOTHING
                    """,
                conversation_id,
            )
            await conn.execute(
                """
                    INSERT INTO task_memory (conversation_id)
                    VALUES ($1)
                    ON CONFLICT (conversation_id) DO NOTHING
                    """,
                conversation_id,
            )

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            """
            SELECT c.conversation_id, c.created_at, c.status,
                   c.delete_run_id, c.delete_requested_at, c.delete_completed_at,
                   tm.turn_count, tm.last_turn_at, tm.awaiting_user_input,
                   tm.turn_in_progress
            FROM conversations c
            LEFT JOIN task_memory tm ON tm.conversation_id = c.conversation_id
            WHERE c.conversation_id = $1
            """,
            conversation_id,
        )
        if row is None:
            return None
        return dict(row)

    async def list_conversations(self) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT c.conversation_id, c.created_at, c.status,
                   c.delete_run_id, c.delete_requested_at, c.delete_completed_at,
                   COALESCE(tm.turn_count, 0)          AS turn_count,
                   tm.last_turn_at,
                   COALESCE(tm.awaiting_user_input, FALSE) AS awaiting_user_input,
                   COALESCE(tm.turn_in_progress, FALSE)    AS turn_in_progress
            FROM conversations c
            LEFT JOIN task_memory tm ON tm.conversation_id = c.conversation_id
            ORDER BY c.created_at DESC
            """
        )
        return [dict(r) for r in rows]

    async def delete_conversation(self, conversation_id: str) -> None:
        """Hard-delete relational state for a conversation.

        Some operational tables intentionally do not have FK cascades in older
        demo deployments (notably task_metadata and retrieval_telemetry).  The
        publication cleanup path is evidence-driven, so these residual rows must
        be removed before the final GC audit is computed.

        C-Items/vector state are still deleted via
        CItemStorePort.delete_by_conversation(); this method owns only the
        relational side.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM retrieval_telemetry WHERE conversation_id = $1",
                conversation_id,
            )
            await conn.execute(
                "DELETE FROM task_metadata WHERE conversation_id = $1",
                conversation_id,
            )
            await conn.execute(
                "DELETE FROM conversations WHERE conversation_id = $1",
                conversation_id,
            )

    async def begin_hard_delete(self, conversation_id: str, *, delete_run_id: str) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                UPDATE conversations
                SET    status = 'DELETING',
                       delete_run_id = $2,
                       delete_requested_at = NOW()
                WHERE  conversation_id = $1
                  AND  COALESCE(status, 'ACTIVE') = 'ACTIVE'
                RETURNING conversation_id
                """,
                conversation_id,
                _uuid.UUID(delete_run_id),
            )
            if row is None:
                return False
            await conn.execute(
                """
                INSERT INTO cima.delete_run (delete_run_id, conversation_id, status, requested_at, stats_json)
                VALUES ($1, $2, 'REQUESTED', NOW(), '{}'::jsonb)
                ON CONFLICT (delete_run_id) DO NOTHING
                """,
                _uuid.UUID(delete_run_id),
                _uuid.UUID(conversation_id),
            )
            return True

    async def mark_hard_delete_completed(self, *, delete_run_id: str, stats_json: dict[str, Any] | None = None) -> None:
        await self._pool.execute(
            """
            UPDATE cima.delete_run
            SET    status = 'SUCCEEDED',
                   completed_at = NOW(),
                   stats_json = $2::jsonb
            WHERE  delete_run_id = $1
            """,
            _uuid.UUID(delete_run_id),
            json.dumps(stats_json or {}),
        )

    async def mark_hard_delete_failed(self, *, delete_run_id: str, stats_json: dict[str, Any] | None = None) -> None:
        await self._pool.execute(
            """
            UPDATE cima.delete_run
            SET    status = 'FAILED',
                   completed_at = NOW(),
                   stats_json = $2::jsonb
            WHERE  delete_run_id = $1
            """,
            _uuid.UUID(delete_run_id),
            json.dumps(stats_json or {}),
        )

    async def begin_maintenance_run(
        self,
        *,
        kind: str,
        conversation_id: str | None = None,
        maintenance_run_id: str,
    ) -> bool:
        if kind not in {"THINNING", "RECONCILE", "EPHEMERAL_EXPIRY", "ORPHAN_CLEANUP"}:
            raise ValueError(f"Unsupported maintenance kind: {kind}")
        async with self._pool.acquire() as conn, conn.transaction():
            if conversation_id is not None:
                row = await conn.fetchrow(
                    """
                    SELECT conversation_id
                    FROM conversations
                    WHERE conversation_id = $1
                    """,
                    conversation_id,
                )
                if row is None:
                    return False
            await conn.execute(
                """
                INSERT INTO cima.maintenance_run (maintenance_run_id, conversation_id, kind, status, requested_at, stats_json)
                VALUES ($1, $2, $3, 'REQUESTED', NOW(), '{}'::jsonb)
                ON CONFLICT (maintenance_run_id) DO NOTHING
                """,
                _uuid.UUID(maintenance_run_id),
                _uuid.UUID(conversation_id) if conversation_id is not None else None,
                kind,
            )
            return True

    async def mark_maintenance_run_running(self, *, maintenance_run_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE cima.maintenance_run
            SET    status = 'RUNNING'
            WHERE  maintenance_run_id = $1
            """,
            _uuid.UUID(maintenance_run_id),
        )

    async def mark_maintenance_run_completed(self, *, maintenance_run_id: str, stats_json: dict[str, Any] | None = None) -> None:
        await self._pool.execute(
            """
            UPDATE cima.maintenance_run
            SET    status = 'SUCCEEDED',
                   completed_at = NOW(),
                   stats_json = $2::jsonb
            WHERE  maintenance_run_id = $1
            """,
            _uuid.UUID(maintenance_run_id),
            json.dumps(stats_json or {}),
        )

    async def mark_maintenance_run_failed(self, *, maintenance_run_id: str, stats_json: dict[str, Any] | None = None) -> None:
        await self._pool.execute(
            """
            UPDATE cima.maintenance_run
            SET    status = 'FAILED',
                   completed_at = NOW(),
                   stats_json = $2::jsonb
            WHERE  maintenance_run_id = $1
            """,
            _uuid.UUID(maintenance_run_id),
            json.dumps(stats_json or {}),
        )

    async def save_ephemeral_vector_record(self, record_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.ephemeral_vector (
                ephemeral_id, conversation_id, origin_ref_kind, origin_ref_id, qdrant_collection,
                lifecycle_state, vector_state, embedding_model_id, embedding_schema_version,
                eligible_for_geometry, meta_json, created_at, expires_at, expired_at, purged_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11::jsonb, COALESCE($12::timestamptz, NOW()), $13::timestamptz, $14::timestamptz, $15::timestamptz
            )
            ON CONFLICT (ephemeral_id) DO UPDATE SET
                origin_ref_kind = EXCLUDED.origin_ref_kind,
                origin_ref_id = EXCLUDED.origin_ref_id,
                qdrant_collection = EXCLUDED.qdrant_collection,
                lifecycle_state = EXCLUDED.lifecycle_state,
                vector_state = EXCLUDED.vector_state,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                eligible_for_geometry = EXCLUDED.eligible_for_geometry,
                meta_json = EXCLUDED.meta_json,
                expires_at = EXCLUDED.expires_at,
                expired_at = EXCLUDED.expired_at,
                purged_at = EXCLUDED.purged_at
            """,
            _uuid.UUID(record_json["ephemeral_id"]),
            _uuid.UUID(record_json["conversation_id"]),
            record_json["origin_ref_kind"],
            _uuid.UUID(record_json["origin_ref_id"]) if record_json.get("origin_ref_id") else None,
            record_json["qdrant_collection"],
            record_json.get("lifecycle_state", "ACTIVE"),
            record_json.get("vector_state", "EPHEMERAL"),
            record_json.get("embedding_model_id"),
            int(record_json["embedding_schema_version"]) if record_json.get("embedding_schema_version") is not None else None,
            bool(record_json.get("eligible_for_geometry", False)),
            json.dumps(record_json.get("meta_json", {}), ensure_ascii=False),
            _parse_ts(record_json.get("created_at")),
            _parse_ts(record_json["expires_at"]) if "expires_at" in record_json else None,
            _parse_ts(record_json.get("expired_at")),
            _parse_ts(record_json.get("purged_at")),
        )

    async def list_ephemeral_vector_records(
        self,
        *,
        conversation_id: str | None = None,
        lifecycle_state: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if conversation_id is not None:
            clauses.append(f"conversation_id = ${len(params) + 1}::uuid")
            params.append(_uuid.UUID(conversation_id))
        if lifecycle_state is not None:
            clauses.append(f"lifecycle_state = ${len(params) + 1}")
            params.append(lifecycle_state)
        rows = await self._pool.fetch(
            f"""
            SELECT ephemeral_id, conversation_id, origin_ref_kind, origin_ref_id, qdrant_collection,
                   lifecycle_state, vector_state, embedding_model_id, embedding_schema_version,
                   eligible_for_geometry, meta_json, created_at, expires_at, expired_at, purged_at
            FROM cima.ephemeral_vector
            WHERE {' AND '.join(clauses)}
            ORDER BY expires_at ASC, ephemeral_id ASC
            """,
            *params,
        )
        return [
            {
                "ephemeral_id": str(r["ephemeral_id"]),
                "conversation_id": str(r["conversation_id"]),
                "origin_ref_kind": r["origin_ref_kind"],
                "origin_ref_id": str(r["origin_ref_id"]) if r["origin_ref_id"] else None,
                "qdrant_collection": r["qdrant_collection"],
                "lifecycle_state": r["lifecycle_state"],
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
                "eligible_for_geometry": bool(r["eligible_for_geometry"]),
                "meta_json": dict(_parse_jsonb(r["meta_json"], {})),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
                "expired_at": r["expired_at"].isoformat() if r["expired_at"] else None,
                "purged_at": r["purged_at"].isoformat() if r["purged_at"] else None,
            }
            for r in rows
        ]

    async def list_due_ephemeral_vector_records(self, *, now: str | None = None) -> list[dict[str, Any]]:
        cutoff = now or datetime.now(UTC).isoformat()
        rows = await self._pool.fetch(
            """
            SELECT ephemeral_id, conversation_id, origin_ref_kind, origin_ref_id, qdrant_collection,
                   lifecycle_state, vector_state, embedding_model_id, embedding_schema_version,
                   eligible_for_geometry, meta_json, created_at, expires_at, expired_at, purged_at
            FROM cima.ephemeral_vector
            WHERE (lifecycle_state = 'ACTIVE' AND expires_at <= $1::timestamptz)
               OR (lifecycle_state = 'EXPIRED' AND purged_at IS NULL)
            ORDER BY expires_at ASC, ephemeral_id ASC
            """,
            cutoff,
        )
        return [
            {
                "ephemeral_id": str(r["ephemeral_id"]),
                "conversation_id": str(r["conversation_id"]),
                "origin_ref_kind": r["origin_ref_kind"],
                "origin_ref_id": str(r["origin_ref_id"]) if r["origin_ref_id"] else None,
                "qdrant_collection": r["qdrant_collection"],
                "lifecycle_state": r["lifecycle_state"],
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
                "eligible_for_geometry": bool(r["eligible_for_geometry"]),
                "meta_json": dict(_parse_jsonb(r["meta_json"], {})),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
                "expired_at": r["expired_at"].isoformat() if r["expired_at"] else None,
                "purged_at": r["purged_at"].isoformat() if r["purged_at"] else None,
            }
            for r in rows
        ]

    async def mark_ephemeral_vector_expired(self, ephemeral_id: str, *, expired_at: str | None = None) -> None:
        await self._pool.execute(
            """
            UPDATE cima.ephemeral_vector
            SET lifecycle_state = 'EXPIRED',
                expired_at = COALESCE(expired_at, $2::timestamptz, NOW())
            WHERE ephemeral_id = $1
              AND lifecycle_state = 'ACTIVE'
            """,
            _uuid.UUID(ephemeral_id),
            expired_at,
        )

    async def mark_ephemeral_vector_purged(self, ephemeral_id: str, *, purged_at: str | None = None) -> None:
        await self._pool.execute(
            """
            UPDATE cima.ephemeral_vector
            SET lifecycle_state = 'PURGED',
                purged_at = COALESCE($2::timestamptz, purged_at, NOW())
            WHERE ephemeral_id = $1
              AND lifecycle_state IN ('ACTIVE', 'EXPIRED')
            """,
            _uuid.UUID(ephemeral_id),
            purged_at,
        )

    # ── Turn mutex ────────────────────────────────────────────────────────────

    async def try_set_turn_in_progress(self, conversation_id: str) -> bool:
        """Atomic CAS — UPDATE WHERE turn_in_progress=FALSE (J-04)."""
        row = await self._pool.fetchrow(
            """
            UPDATE task_memory
            SET    turn_in_progress = TRUE,
                   updated_at       = NOW()
            WHERE  conversation_id  = $1
              AND  turn_in_progress = FALSE
            RETURNING conversation_id
            """,
            conversation_id,
        )
        return row is not None

    async def set_turn_finished(self, conversation_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE task_memory
            SET    turn_in_progress = FALSE,
                   turn_count       = turn_count + 1,
                   last_turn_at     = NOW(),
                   updated_at       = NOW()
            WHERE  conversation_id = $1
            """,
            conversation_id,
        )

    async def release_turn_in_progress(self, conversation_id: str) -> None:
        """Force-release turn mutex (error path — idempotent, §3.1.2)."""
        await self._pool.execute(
            """
            UPDATE task_memory
            SET    turn_in_progress = FALSE,
                   updated_at       = NOW()
            WHERE  conversation_id = $1
            """,
            conversation_id,
        )

    # ── TaskMemory ────────────────────────────────────────────────────────────

    async def load_task_memory(self, conversation_id: str) -> TaskMemory | None:
        row = await self._pool.fetchrow(
            """
            SELECT conversation_id, turn_count, phase, active_plan_id,
                   awaiting_user_input, turn_in_progress, stall_count,
                   last_turn_at, created_at
            FROM task_memory
            WHERE conversation_id = $1
            """,
            conversation_id,
        )
        if row is None:
            return None
        return TaskMemory(
            conversation_id=str(row["conversation_id"]),
            turn_count=row["turn_count"] or 0,
            phase=row["phase"] or CognitivePhase.IDLE,
            active_plan_id=str(row["active_plan_id"]) if row["active_plan_id"] else None,
            awaiting_user_input=row["awaiting_user_input"] or False,
            turn_in_progress=row["turn_in_progress"] or False,
            stall_count=row["stall_count"] or 0,
            last_turn_at=row["last_turn_at"],
            created_at=row["created_at"] or datetime.now(UTC),
        )

    async def save_task_memory(self, task_memory: TaskMemory) -> None:
        await self._pool.execute(
            """
            INSERT INTO task_memory (
                conversation_id, turn_count, phase, active_plan_id,
                awaiting_user_input, turn_in_progress, stall_count,
                last_turn_at, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (conversation_id) DO UPDATE SET
                turn_count          = EXCLUDED.turn_count,
                phase               = EXCLUDED.phase,
                active_plan_id      = EXCLUDED.active_plan_id,
                awaiting_user_input = EXCLUDED.awaiting_user_input,
                turn_in_progress    = EXCLUDED.turn_in_progress,
                stall_count         = EXCLUDED.stall_count,
                last_turn_at        = EXCLUDED.last_turn_at,
                updated_at          = NOW()
            """,
            task_memory.conversation_id,
            task_memory.turn_count,
            task_memory.phase,
            task_memory.active_plan_id,
            task_memory.awaiting_user_input,
            task_memory.turn_in_progress,
            task_memory.stall_count,
            task_memory.last_turn_at,
            task_memory.created_at,
        )

    # ── History ───────────────────────────────────────────────────────────────

    async def append_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        created_at: str | None = None,
    ) -> None:
        ts = datetime.fromisoformat(created_at) if created_at else datetime.now(UTC)
        async with self._pool.acquire() as conn:
            seq = await conn.fetchval(
                """
                SELECT COALESCE(MAX(sequence), -1) + 1
                FROM conversation_turns
                WHERE conversation_id = $1
                """,
                conversation_id,
            )
            await conn.execute(
                """
                INSERT INTO conversation_turns
                    (conversation_id, sequence, user_message, assistant_reply, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                conversation_id, seq, user_message, assistant_message, ts,
            )

    async def load_recent_history(
        self,
        conversation_id: str,
        max_turns: int = 10,
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT user_message, assistant_reply, created_at
            FROM conversation_turns
            WHERE conversation_id = $1
            ORDER BY sequence DESC
            LIMIT $2
            """,
            conversation_id,
            max_turns,
        )
        # Build messages in chronological order
        messages: list[dict[str, Any]] = [
            {
                "role": role,
                "content": content,
                "timestamp": str(row["created_at"]),
            }
            for row in reversed(rows)
            for role, content in (
                ("user", row["user_message"]),
                ("assistant", row["assistant_reply"]),
            )
        ]

        # Trim oldest messages to fit within token_budget (INFRA-D-02)
        # Use index-based scan (O(n)) instead of list.pop(0) (O(n²)).
        if token_budget is not None and messages:
            total = sum(len(m["content"]) // 4 for m in messages)
            start = 0
            while start < len(messages) and total > token_budget:
                total -= len(messages[start]["content"]) // 4
                start += 1
                # Drop in pairs (user + assistant) to keep conversation coherent
                if start < len(messages) and messages[start]["role"] == "assistant":
                    total -= len(messages[start]["content"]) // 4
                    start += 1
            messages = messages[start:]

        return messages

    # ── Summary pyramid ───────────────────────────────────────────────────────

    def _row_to_summary_node(self, row: asyncpg.Record) -> SummaryNode:
        raw_origin_ids: list[Any] = row.get("origin_ids") or []
        node = SummaryNode(
            node_id=str(row["node_id"]),
            conversation_id=str(row["conversation_id"]),
            level=row["level"],
            content=row["content"] or "",
            token_count=row["token_count"] or 0,
            created_at=row["created_at"],
            parent_id=str(row["parent_id"]) if row.get("parent_id") else None,
            origin_citem_ids=[str(oid) for oid in raw_origin_ids],
        )
        setattr(node, "summary_resolution_mode", "legacy_fallback")
        setattr(node, "summary_ref_kind", "legacy_summary")
        setattr(node, "summary_scope", "legacy")
        return node

    async def save_summary(self, node: SummaryNode) -> None:
        # asyncpg maps list[str] to text[], not uuid[] — must pass uuid.UUID objects
        origin_ids_arr = (
            [_uuid.UUID(oid) for oid in node.origin_citem_ids]
            if node.origin_citem_ids else []
        )
        await self._pool.execute(
            """
            INSERT INTO summary_nodes
                (node_id, conversation_id, level, text, token_count,
                 origin_ids, parent_ids, confidence, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, '[]', 1.0, $7, NOW())
            ON CONFLICT (node_id) DO UPDATE SET
                text        = EXCLUDED.text,
                token_count = EXCLUDED.token_count,
                origin_ids  = EXCLUDED.origin_ids,
                updated_at  = NOW()
            """,
            node.node_id,
            node.conversation_id,
            node.level,
            node.content,
            node.token_count,
            origin_ids_arr,
            node.created_at,
        )

    async def _legacy_summary_nodes(
        self,
        conversation_id: str,
        *,
        level: int | None = None,
        parentless_only: bool = False,
    ) -> list[SummaryNode]:
        conditions = ["conversation_id = $1"]
        params: list[Any] = [conversation_id]
        if level is not None:
            conditions.append(f"level = ${len(params) + 1}")
            params.append(level)
        if parentless_only:
            conditions.append("parent_ids = '[]'::jsonb")
        where = " AND ".join(conditions)
        order = "level DESC, updated_at DESC" if parentless_only else "level ASC, updated_at DESC"
        rows = await self._pool.fetch(
            f"""
            SELECT node_id, conversation_id, level, text AS content,
                   token_count, origin_ids, created_at, parent_ids->>0 AS parent_id
            FROM summary_nodes
            WHERE {where}
            ORDER BY {order}
            """,
            *params,
        )
        return [self._row_to_summary_node(r) for r in rows]

    async def _witness_summary_nodes(
        self,
        conversation_id: str,
        *,
        level: int | None = None,
        parentless_only: bool = False,
    ) -> list[SummaryNode]:
        level_map = {"EPOCH": 1, "CLUSTER": 2, "MASTER": 3}
        target_levels = {level} if level is not None else None
        witness_nodes: list[SummaryNode] = []

        local_rows = await self.list_local_summary_records(conversation_id)
        for row in local_rows:
            node_level = level_map.get(str(row.get("level") or "EPOCH"), 1)
            if target_levels is not None and node_level not in target_levels:
                continue
            if parentless_only and row.get("parent_id"):
                continue
            covers = dict(row.get("covers_json") or {})
            origin_ids = [str(v) for v in covers.get("origin_citem_ids", []) if v]
            node = SummaryNode(
                node_id=str(row["local_summary_id"]),
                conversation_id=str(row["conversation_id"]),
                level=node_level,
                content=str(row.get("text") or ""),
                token_count=max(1, len(str(row.get("text") or "").split())),
                created_at=(datetime.fromisoformat(row["created_at"]) if _parse_ts(row.get("created_at")) else datetime.now(UTC)),
                parent_id=None,
                origin_citem_ids=origin_ids,
            )
            setattr(node, "summary_resolution_mode", "witness_first")
            setattr(node, "summary_ref_kind", "local_summary")
            setattr(node, "summary_scope", "local")
            witness_nodes.append(node)

        global_rows = await self.list_global_summary_records(origin_conversation_id=conversation_id)
        for row in global_rows:
            node_level = level_map.get(str(row.get("level") or "MASTER"), 3)
            if target_levels is not None and node_level not in target_levels:
                continue
            if parentless_only and row.get("parent_id"):
                continue
            covers = dict(row.get("covers_json") or {})
            origin_ids = [str(v) for v in covers.get("origin_global_citem_ids", []) if v]
            node = SummaryNode(
                node_id=str(row["global_summary_id"]),
                conversation_id=str(conversation_id),
                level=node_level,
                content=str(row.get("text") or ""),
                token_count=max(1, len(str(row.get("text") or "").split())),
                created_at=(datetime.fromisoformat(row["created_at"]) if _parse_ts(row.get("created_at")) else datetime.now(UTC)),
                parent_id=None,
                origin_citem_ids=origin_ids,
            )
            setattr(node, "summary_resolution_mode", "witness_first")
            setattr(node, "summary_ref_kind", "global_summary")
            setattr(node, "summary_scope", "global")
            witness_nodes.append(node)

        deduped: list[SummaryNode] = []
        seen: set[str] = set()
        for node in sorted(witness_nodes, key=lambda n: (n.level, n.created_at), reverse=parentless_only):
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            deduped.append(node)
        if not parentless_only:
            deduped = sorted(deduped, key=lambda n: (n.level, n.created_at), reverse=True)
        return deduped

    async def load_summaries(
        self,
        conversation_id: str,
        level: int | None = None,
    ) -> list[SummaryNode]:
        witness_nodes = await self._witness_summary_nodes(conversation_id, level=level, parentless_only=False)
        if witness_nodes:
            return sorted(witness_nodes, key=lambda node: (node.level, node.created_at), reverse=True)
        return await self._legacy_summary_nodes(conversation_id, level=level, parentless_only=False)

    async def set_summary_parent(self, node_id: str, parent_id: str) -> None:
        """A-10 L2 AutoPromote: mark node as absorbed into parent."""
        await self._pool.execute(
            """
            UPDATE summary_nodes
            SET parent_ids = jsonb_build_array($2::text),
                updated_at = NOW()
            WHERE node_id = $1
            """,
            node_id,
            parent_id,
        )

    async def fetch_nodes_at_level(
        self,
        level: int,
        conversation_id: str,
        parentless_only: bool = False,
        limit: int | None = None,
    ) -> list[SummaryNode]:
        """N-01: SELECT with optional parentless filter and LIMIT."""
        witness_nodes = await self._witness_summary_nodes(
            conversation_id,
            level=level,
            parentless_only=parentless_only,
        )
        rows = witness_nodes if witness_nodes else await self._legacy_summary_nodes(
            conversation_id,
            level=level,
            parentless_only=parentless_only,
        )
        return rows[:limit] if limit is not None else rows

    async def fetch_pyramid_tops(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[SummaryNode]:
        """N-02: Nodes without parents, highest level first.

        Transitional behaviour: if witness-backend local/global summaries exist,
        surface them ahead of legacy summary_nodes so the running runtime can
        consume witness summaries without waiting for the full persistence
        migration.
        """
        witness_nodes = await self._witness_summary_nodes(conversation_id, parentless_only=True)
        legacy_nodes = [] if witness_nodes else await self._legacy_summary_nodes(conversation_id, parentless_only=True)
        combined = witness_nodes or legacy_nodes
        combined = sorted(combined, key=lambda node: (node.level, node.created_at), reverse=True)
        return combined[:limit] if limit is not None else combined

    # ── Plans ─────────────────────────────────────────────────────────────────

    async def save_plan(self, plan: Plan) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                    INSERT INTO plans (plan_id, conversation_id, goal, status, auto_continue, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (plan_id) DO UPDATE SET
                        status        = EXCLUDED.status,
                        auto_continue = EXCLUDED.auto_continue,
                        updated_at    = NOW()
                    """,
                plan.plan_id,
                plan.conversation_id,
                plan.goal,
                plan.status.lower() if isinstance(plan.status, str) else plan.status,
                plan.auto_continue,
                plan.created_at,
            )
            for seq, step in enumerate(plan.steps):
                await conn.execute(
                    """
                        INSERT INTO plan_steps
                            (step_id, plan_id, sequence, description, tool_name,
                             tool_params, status, result_summary, procedure_citem_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (step_id) DO UPDATE SET
                            status         = EXCLUDED.status,
                            result_summary = EXCLUDED.result_summary
                        """,
                    step.step_id,
                    plan.plan_id,
                    seq,
                    step.description,
                    step.tool_name,
                    json.dumps(step.tool_args),
                    step.status.lower() if isinstance(step.status, str) else step.status,
                    step.result_summary,
                    step.procedure_citem_id,
                )

    async def save_plan_with_task_memory(
        self,
        plan: Plan,
        task_memory: TaskMemory,
    ) -> None:
        """Atomically persist Plan + TaskMemory (INFRA-D-01)."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO plans (plan_id, conversation_id, goal, status, auto_continue, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (plan_id) DO UPDATE SET
                    status        = EXCLUDED.status,
                    auto_continue = EXCLUDED.auto_continue,
                    updated_at    = NOW()
                """,
                plan.plan_id,
                plan.conversation_id,
                plan.goal,
                plan.status.lower() if isinstance(plan.status, str) else plan.status,
                plan.auto_continue,
                plan.created_at,
            )
            for seq, step in enumerate(plan.steps):
                await conn.execute(
                    """
                    INSERT INTO plan_steps
                        (step_id, plan_id, sequence, description, tool_name,
                         tool_params, status, result_summary, procedure_citem_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (step_id) DO UPDATE SET
                        status         = EXCLUDED.status,
                        result_summary = EXCLUDED.result_summary
                    """,
                    step.step_id,
                    plan.plan_id,
                    seq,
                    step.description,
                    step.tool_name,
                    json.dumps(step.tool_args),
                    step.status.lower() if isinstance(step.status, str) else step.status,
                    step.result_summary,
                    step.procedure_citem_id,
                )
            await conn.execute(
                """
                INSERT INTO task_memory (
                    conversation_id, turn_count, phase, active_plan_id,
                    awaiting_user_input, turn_in_progress, stall_count,
                    last_turn_at, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (conversation_id) DO UPDATE SET
                    turn_count          = EXCLUDED.turn_count,
                    phase               = EXCLUDED.phase,
                    active_plan_id      = EXCLUDED.active_plan_id,
                    awaiting_user_input = EXCLUDED.awaiting_user_input,
                    turn_in_progress    = EXCLUDED.turn_in_progress,
                    stall_count         = EXCLUDED.stall_count,
                    last_turn_at        = EXCLUDED.last_turn_at,
                    updated_at          = NOW()
                """,
                task_memory.conversation_id,
                task_memory.turn_count,
                task_memory.phase,
                task_memory.active_plan_id,
                task_memory.awaiting_user_input,
                task_memory.turn_in_progress,
                task_memory.stall_count,
                task_memory.last_turn_at,
                task_memory.created_at,
            )

    async def load_plan(self, plan_id: str) -> Plan | None:
        plan_row = await self._pool.fetchrow(
            "SELECT plan_id, conversation_id, goal, status, auto_continue, created_at FROM plans WHERE plan_id = $1",
            plan_id,
        )
        if plan_row is None:
            return None
        step_rows = await self._pool.fetch(
            """
            SELECT step_id, plan_id, description, tool_name, tool_params,
                   status, result_summary, procedure_citem_id, sequence
            FROM plan_steps
            WHERE plan_id = $1
            ORDER BY sequence
            """,
            plan_id,
        )
        steps = [
            PlanStep(
                step_id=str(r["step_id"]),
                plan_id=str(r["plan_id"]),
                description=r["description"],
                tool_name=r["tool_name"],
                tool_args=json.loads(r["tool_params"] or "{}"),
                status=r["status"].upper(),
                result_summary=r["result_summary"],
                procedure_citem_id=str(r["procedure_citem_id"]) if r["procedure_citem_id"] else None,
            )
            for r in step_rows
        ]
        return Plan(
            plan_id=str(plan_row["plan_id"]),
            conversation_id=str(plan_row["conversation_id"]),
            goal=plan_row["goal"],
            status=plan_row["status"].upper(),
            steps=steps,
            created_at=plan_row["created_at"],
            auto_continue=bool(plan_row["auto_continue"]),
        )

    async def list_auto_plans(self) -> list[tuple[str, str, str, int, int, str | None]]:
        """DEBT-01: returns auto-continue candidates with step context."""
        sql = """
            SELECT p.conversation_id::text,
                   p.plan_id::text,
                   ps_active.description,
                   ps_active.sequence AS active_step_idx,
                   (SELECT count(*) FROM plan_steps WHERE plan_id = p.plan_id) AS total_steps,
                   ps_prev.result_summary AS prev_result
            FROM plans p
            JOIN LATERAL (
                SELECT description, sequence
                FROM plan_steps
                WHERE plan_id = p.plan_id AND status = 'active'
                ORDER BY sequence
                LIMIT 1
            ) ps_active ON TRUE
            LEFT JOIN LATERAL (
                SELECT result_summary
                FROM plan_steps
                WHERE plan_id = p.plan_id AND status = 'completed'
                ORDER BY sequence DESC
                LIMIT 1
            ) ps_prev ON TRUE
            JOIN task_memory tm ON tm.conversation_id = p.conversation_id
            WHERE p.auto_continue = TRUE
              AND p.status = 'running'
              AND tm.turn_in_progress = FALSE
              AND tm.awaiting_user_input = FALSE
            ORDER BY p.updated_at ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return [
            (
                r["conversation_id"],
                r["plan_id"],
                r["description"],
                int(r["active_step_idx"]),
                int(r["total_steps"]),
                r["prev_result"],
            )
            for r in rows
        ]

    async def save_plan_step(self, step: PlanStep) -> None:
        await self._pool.execute(
            """
            INSERT INTO plan_steps
                (step_id, plan_id, sequence, description, tool_name,
                 tool_params, status, result_summary, procedure_citem_id)
            VALUES ($1, $2, 0, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (step_id) DO UPDATE SET
                status         = EXCLUDED.status,
                result_summary = EXCLUDED.result_summary
            """,
            step.step_id,
            step.plan_id,
            step.description,
            step.tool_name,
            json.dumps(step.tool_args),
            step.status.lower() if isinstance(step.status, str) else step.status,
            step.result_summary,
            step.procedure_citem_id,
        )

    async def update_plan_step_attempts(self, step_id: str, attempts: int) -> None:
        """N-03: Lightweight attempts update."""
        await self._pool.execute(
            "UPDATE plan_steps SET attempts = $1 WHERE step_id = $2",
            attempts, step_id,
        )

    # ── Conflict log ──────────────────────────────────────────────────────────

    async def save_conflict(self, entry: ConflictLogEntry) -> None:
        await self._pool.execute(
            """
            INSERT INTO conflict_log
                (entry_id, conversation_id, citem_a_id, citem_b_id,
                 conflict_type, detected_at, notes)
            VALUES ($1, $2, $3, $4, $5, $6, '')
            ON CONFLICT (entry_id) DO UPDATE SET
                resolved_at     = EXCLUDED.resolved_at,
                resolver_actor  = 'agent',
                notes           = COALESCE(conflict_log.notes, '')
            """,
            entry.entry_id,
            entry.conversation_id,
            entry.item_a_id,
            entry.item_b_id,
            entry.conflict_type,
            entry.created_at,
        )

    async def load_conflicts(
        self,
        conversation_id: str,
        resolved: bool | None = None,
    ) -> list[ConflictLogEntry]:
        if resolved is None:
            rows = await self._pool.fetch(
                """
                SELECT entry_id, conversation_id, citem_a_id, citem_b_id,
                       conflict_type, detected_at, resolved_at, notes
                FROM conflict_log
                WHERE conversation_id = $1
                ORDER BY detected_at DESC
                """,
                conversation_id,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT entry_id, conversation_id, citem_a_id, citem_b_id,
                       conflict_type, detected_at, resolved_at, notes
                FROM conflict_log
                WHERE conversation_id = $1
                  AND (resolved_at IS NOT NULL) = $2
                ORDER BY detected_at DESC
                """,
                conversation_id,
                resolved,
            )
        return [
            ConflictLogEntry(
                entry_id=str(r["entry_id"]),
                conversation_id=str(r["conversation_id"]),
                item_a_id=str(r["citem_a_id"]),
                item_b_id=str(r["citem_b_id"]),
                conflict_type=r["conflict_type"],
                resolved=r["resolved_at"] is not None,
                created_at=r["detected_at"],
                resolved_at=r["resolved_at"],
            )
            for r in rows
        ]

    # ── Telemetry ─────────────────────────────────────────────────────────────

    async def save_retrieval_telemetry(
        self,
        conversation_id: str,
        query_type: str,
        recall_top_k: int,
        rerank_top_n: int,
        items_selected: int,
        coverage_score: float,
        retry_count: int,
        latency_ms: int,
        candidates_before_rerank: int = 0,
        candidates_after_rerank: int = 0,
        candidates_after_expand: int = 0,
        pack_total_tokens: int = 0,
        geometric_expand: bool = False,
        reranker_available: bool = True,
        traceability_density: float = 1.0,
        # Bridge / strategy fields (Retrieval Instrumentation D)
        q3_relevant_count: int = 0,
        bridge_enabled: bool = False,
        bridge_alpha: float = 0.5,
        bridge_floor: float = 0.0,
        bridge_candidates_eligible: int = 0,
        direct_strategy: str | None = None,
    ) -> None:
        try:
            await self._pool.execute(
                """
                INSERT INTO retrieval_telemetry (
                    conversation_id, turn_id, query_type,
                    recall_top_k, rerank_top_n, geometric_expand,
                    candidates_before_rerank, candidates_after_rerank,
                    candidates_after_expand, pack_total_tokens,
                    coverage_score, retry_count, reranker_available, latency_ms,
                    traceability_density,
                    q3_relevant_count, bridge_enabled, bridge_alpha,
                    bridge_floor, bridge_candidates, direct_strategy
                ) VALUES (
                    $1, gen_random_uuid(), $2,
                    $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11, $12, $13,
                    $14,
                    $15, $16, $17,
                    $18, $19, $20
                )
                """,
                conversation_id, query_type,
                recall_top_k, rerank_top_n, geometric_expand,
                candidates_before_rerank, candidates_after_rerank,
                candidates_after_expand, pack_total_tokens,
                coverage_score, retry_count, reranker_available, latency_ms,
                traceability_density,
                q3_relevant_count, bridge_enabled, bridge_alpha,
                bridge_floor, bridge_candidates_eligible, direct_strategy,
            )
        except Exception:
            log.exception("save_retrieval_telemetry failed — non-fatal")

    # ── Turn metadata ─────────────────────────────────────────────────────────

    async def load_turn_metadata(self, conversation_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            "SELECT data FROM task_metadata WHERE conversation_id = $1",
            conversation_id,
        )
        if row is None:
            return None
        data = row["data"]
        if isinstance(data, str):
            result: dict[str, Any] = json.loads(data)
            return result
        return dict(data)

    async def save_turn_metadata(self, conversation_id: str, json_data: str) -> None:
        await self._pool.execute(
            """
            INSERT INTO task_metadata (conversation_id, data, updated_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (conversation_id) DO UPDATE SET
                data       = EXCLUDED.data,
                updated_at = NOW()
            """,
            conversation_id,
            json_data,
        )

    async def load_chm_refs(self, conversation_id: str) -> dict[str, int]:
        rows = await self._pool.fetch(
            "SELECT citem_id, reference_count FROM chm_refs WHERE conversation_id = $1",
            conversation_id,
        )
        return {str(r["citem_id"]): r["reference_count"] for r in rows}

    async def save_chm_refs(self, conversation_id: str, citem_ids: list[str]) -> None:
        if not citem_ids:
            return
        try:
            await self._pool.executemany(
                """
                INSERT INTO chm_refs (conversation_id, citem_id, reference_count)
                VALUES ($1, $2, 1)
                ON CONFLICT (conversation_id, citem_id)
                DO UPDATE SET reference_count = chm_refs.reference_count + 1
                """,
                [(_uuid.UUID(conversation_id), _uuid.UUID(cid)) for cid in citem_ids],
            )
        except Exception:
            # Graceful degradation: column may not exist on pre-migration instances.
            log.debug("save_chm_refs failed (non-fatal) for %s", conversation_id)

    # ── Demonstrator run journal ─────────────────────────────────────────────

    async def create_demo_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        turn_id: str,
        status: str,
        user_message: str,
        manifest_json: dict[str, Any],
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_runs (
                run_id, conversation_id, turn_id, status, user_message, manifest, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW(), NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                status = EXCLUDED.status,
                user_message = EXCLUDED.user_message,
                manifest = EXCLUDED.manifest,
                updated_at = NOW()
            """,
            _uuid.UUID(run_id),
            _uuid.UUID(conversation_id),
            _uuid.UUID(turn_id),
            status,
            user_message,
            json.dumps(manifest_json, ensure_ascii=False),
        )

    async def append_demo_run_phase(
        self,
        *,
        run_id: str,
        phase_name: str,
        payload_json: dict[str, Any],
    ) -> int:
        async with self._pool.acquire() as conn, conn.transaction():
            seq = await conn.fetchval(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM demo_run_phases WHERE run_id = $1",
                _uuid.UUID(run_id),
            )
            await conn.execute(
                """
                INSERT INTO demo_run_phases (run_id, sequence, phase_name, payload, created_at)
                VALUES ($1, $2, $3, $4::jsonb, NOW())
                """,
                _uuid.UUID(run_id),
                seq,
                phase_name,
                json.dumps(payload_json, ensure_ascii=False),
            )
            await conn.execute(
                """
                UPDATE demo_runs
                SET phase_count = GREATEST(phase_count, $2), updated_at = NOW()
                WHERE run_id = $1
                """,
                _uuid.UUID(run_id),
                seq,
            )
        return int(seq)

    async def save_demo_checkpoint(
        self,
        *,
        run_id: str,
        checkpoint_id: str,
        checkpoint_kind: str,
        state_json: dict[str, Any],
    ) -> int:
        async with self._pool.acquire() as conn, conn.transaction():
            seq = await conn.fetchval(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM demo_checkpoints WHERE run_id = $1",
                _uuid.UUID(run_id),
            )
            await conn.execute(
                """
                INSERT INTO demo_checkpoints (checkpoint_id, run_id, sequence, checkpoint_kind, state, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
                """,
                _uuid.UUID(checkpoint_id),
                _uuid.UUID(run_id),
                seq,
                checkpoint_kind,
                json.dumps(state_json, ensure_ascii=False),
            )
            await conn.execute(
                """
                UPDATE demo_runs
                SET checkpoint_count = GREATEST(checkpoint_count, $2), updated_at = NOW()
                WHERE run_id = $1
                """,
                _uuid.UUID(run_id),
                seq,
            )
        return int(seq)

    async def touch_demo_run_counters(
        self,
        *,
        run_id: str,
        checkpoint_count: int | None = None,
        phase_count: int | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE demo_runs
            SET checkpoint_count = COALESCE($2, checkpoint_count),
                phase_count = COALESCE($3, phase_count),
                updated_at = NOW()
            WHERE run_id = $1
            """,
            _uuid.UUID(run_id),
            checkpoint_count,
            phase_count,
        )

    async def update_demo_run_manifest(
        self,
        *,
        run_id: str,
        status: str,
        cognitive_phase: str | None,
        execution_mode: str | None,
        active_plan_id: str | None,
        assistant_reply: str,
        error_class: str | None,
        manifest_json: dict[str, Any],
        finished_at: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE demo_runs
            SET status = $2,
                cognitive_phase = $3,
                execution_mode = $4,
                active_plan_id = $5,
                assistant_reply = $6,
                error_class = $7,
                manifest = $8::jsonb,
                finished_at = COALESCE($9::timestamptz, finished_at),
                updated_at = NOW()
            WHERE run_id = $1
            """,
            _uuid.UUID(run_id),
            status,
            cognitive_phase,
            execution_mode,
            _uuid.UUID(active_plan_id) if active_plan_id else None,
            assistant_reply,
            error_class,
            json.dumps(manifest_json, ensure_ascii=False),
            _parse_ts(finished_at),
        )

    async def load_demo_run(self, run_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            """
            SELECT run_id, conversation_id, turn_id, status, user_message, cognitive_phase,
                   execution_mode, active_plan_id, assistant_reply, error_class,
                   checkpoint_count, phase_count, created_at, updated_at, finished_at, manifest
            FROM demo_runs
            WHERE run_id = $1
            """,
            _uuid.UUID(run_id),
        )
        if row is None:
            return None
        manifest = row["manifest"] or {}
        data = dict(manifest) if not isinstance(manifest, str) else json.loads(manifest)
        data.update({
            "run_id": str(row["run_id"]),
            "conversation_id": str(row["conversation_id"]),
            "turn_id": str(row["turn_id"]),
            "status": row["status"],
            "cognitive_phase": row["cognitive_phase"],
            "execution_mode": row["execution_mode"],
            "active_plan_id": str(row["active_plan_id"]) if row["active_plan_id"] else None,
            "assistant_reply": row["assistant_reply"] or "",
            "error_class": row["error_class"],
            "checkpoint_count": row["checkpoint_count"] or 0,
            "phase_count": row["phase_count"] or 0,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        })
        return data

    async def load_demo_run_phases(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT sequence, phase_name, payload, created_at
            FROM demo_run_phases
            WHERE run_id = $1
            ORDER BY sequence ASC
            """,
            _uuid.UUID(run_id),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row["payload"] or {}
            out.append({
                "run_id": run_id,
                "sequence": row["sequence"],
                "phase_name": row["phase_name"],
                "payload": dict(payload) if not isinstance(payload, str) else json.loads(payload),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })
        return out

    async def load_demo_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT checkpoint_id, sequence, checkpoint_kind, state, created_at
            FROM demo_checkpoints
            WHERE run_id = $1
            ORDER BY sequence ASC
            """,
            _uuid.UUID(run_id),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            state = row["state"] or {}
            out.append({
                "run_id": run_id,
                "checkpoint_id": str(row["checkpoint_id"]),
                "sequence": row["sequence"],
                "checkpoint_kind": row["checkpoint_kind"],
                "state": dict(state) if not isinstance(state, str) else json.loads(state),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })
        return out

    # ── Demonstrator lineage / context artifacts ────────────────────────────

    async def save_demo_source(self, source_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_sources (
                source_id, conversation_id, source_kind, role, origin_ref,
                display_text, process_text, metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, COALESCE($9::timestamptz, NOW()))
            ON CONFLICT (source_id) DO UPDATE SET
                display_text = EXCLUDED.display_text,
                process_text = EXCLUDED.process_text,
                metadata = EXCLUDED.metadata
            """,
            _uuid.UUID(source_json["source_id"]),
            _uuid.UUID(source_json["conversation_id"]),
            source_json["source_kind"],
            source_json.get("role"),
            source_json.get("origin_ref"),
            source_json.get("display_text"),
            source_json.get("process_text"),
            json.dumps(source_json.get("metadata", {}), ensure_ascii=False),
            _parse_ts(source_json.get("created_at")),
        )

    async def save_demo_source_span(self, span_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_source_spans (
                span_id, source_id, conversation_id, span_kind, char_start, char_end,
                locator, preview_text, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, COALESCE($9::timestamptz, NOW()))
            ON CONFLICT (span_id) DO UPDATE SET
                char_start = EXCLUDED.char_start,
                char_end = EXCLUDED.char_end,
                locator = EXCLUDED.locator,
                preview_text = EXCLUDED.preview_text
            """,
            _uuid.UUID(span_json["span_id"]),
            _uuid.UUID(span_json["source_id"]),
            _uuid.UUID(span_json["conversation_id"]),
            span_json["span_kind"],
            int(span_json.get("char_start", 0)),
            int(span_json.get("char_end", 0)),
            json.dumps(span_json.get("locator", {}), ensure_ascii=False),
            span_json.get("preview_text", ""),
            _parse_ts(span_json.get("created_at")),
        )

    async def save_demo_lineage_edge(self, edge_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_lineage_edges (
                edge_id, conversation_id, src_kind, src_id, dst_kind, dst_id,
                relation, metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, COALESCE($9::timestamptz, NOW()))
            ON CONFLICT (edge_id) DO NOTHING
            """,
            _uuid.UUID(edge_json["edge_id"]),
            _uuid.UUID(edge_json["conversation_id"]),
            edge_json["src_kind"],
            edge_json["src_id"],
            edge_json["dst_kind"],
            edge_json["dst_id"],
            edge_json["relation"],
            json.dumps(edge_json.get("metadata", {}), ensure_ascii=False),
            _parse_ts(edge_json.get("created_at")),
        )

    async def save_demo_summary_resolution(self, resolution_json: dict[str, Any]) -> None:
        origin_ids = [_uuid.UUID(v) for v in resolution_json.get("origin_citem_ids", [])]
        await self._pool.execute(
            """
            INSERT INTO demo_summary_resolutions (
                summary_id, conversation_id, summary_text, origin_citem_ids, metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5::jsonb, COALESCE($6::timestamptz, NOW()))
            ON CONFLICT (summary_id) DO UPDATE SET
                summary_text = EXCLUDED.summary_text,
                origin_citem_ids = EXCLUDED.origin_citem_ids,
                metadata = EXCLUDED.metadata
            """,
            _uuid.UUID(resolution_json["summary_id"]),
            _uuid.UUID(resolution_json["conversation_id"]),
            resolution_json.get("summary_text", ""),
            origin_ids,
            json.dumps(resolution_json.get("metadata", {}), ensure_ascii=False),
            _parse_ts(resolution_json.get("created_at")),
        )

    async def save_demo_context_snapshot(self, snapshot_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_context_snapshots (
                context_id, run_id, conversation_id, turn_id, query_text, phase,
                context_text, markers, items, budget,
                resolved_source_ids, resolved_span_ids, resolved_source_count,
                resolved_span_count, unresolved_ref_ids, marker_resolution, resolution_mode, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10::jsonb,
                $11::jsonb, $12::jsonb, $13, $14, $15::jsonb, $16::jsonb, $17,
                COALESCE($18::timestamptz, NOW())
            )
            ON CONFLICT (context_id) DO UPDATE SET
                context_text = EXCLUDED.context_text,
                markers = EXCLUDED.markers,
                items = EXCLUDED.items,
                budget = EXCLUDED.budget,
                resolved_source_ids = EXCLUDED.resolved_source_ids,
                resolved_span_ids = EXCLUDED.resolved_span_ids,
                resolved_source_count = EXCLUDED.resolved_source_count,
                resolved_span_count = EXCLUDED.resolved_span_count,
                unresolved_ref_ids = EXCLUDED.unresolved_ref_ids,
                marker_resolution = EXCLUDED.marker_resolution,
                resolution_mode = EXCLUDED.resolution_mode
            """,
            _uuid.UUID(snapshot_json["context_id"]),
            _uuid.UUID(snapshot_json["run_id"]),
            _uuid.UUID(snapshot_json["conversation_id"]),
            _uuid.UUID(snapshot_json["turn_id"]),
            snapshot_json.get("query_text", ""),
            snapshot_json.get("phase"),
            snapshot_json.get("context_text", ""),
            json.dumps(snapshot_json.get("markers", []), ensure_ascii=False),
            json.dumps(snapshot_json.get("items", []), ensure_ascii=False),
            json.dumps(snapshot_json.get("budget", {}), ensure_ascii=False),
            json.dumps(snapshot_json.get("resolved_source_ids", []), ensure_ascii=False),
            json.dumps(snapshot_json.get("resolved_span_ids", []), ensure_ascii=False),
            int(snapshot_json.get("resolved_source_count", 0) or 0),
            int(snapshot_json.get("resolved_span_count", 0) or 0),
            json.dumps(snapshot_json.get("unresolved_ref_ids", []), ensure_ascii=False),
            json.dumps(snapshot_json.get("marker_resolution", []), ensure_ascii=False),
            str(snapshot_json.get("resolution_mode", "empty") or "empty"),
            _parse_ts(snapshot_json.get("created_at")),
        )

    async def load_demo_context_snapshot(self, context_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            """
            SELECT context_id, run_id, conversation_id, turn_id, query_text, phase,
                   context_text, markers, items, budget,
                   resolved_source_ids, resolved_span_ids, resolved_source_count,
                   resolved_span_count, unresolved_ref_ids, marker_resolution, resolution_mode, created_at
            FROM demo_context_snapshots
            WHERE context_id = $1
            """,
            _uuid.UUID(context_id),
        )
        if row is None:
            return None
        return {
            "context_id": str(row["context_id"]),
            "run_id": str(row["run_id"]),
            "conversation_id": str(row["conversation_id"]),
            "turn_id": str(row["turn_id"]),
            "query_text": row["query_text"],
            "phase": row["phase"],
            "context_text": row["context_text"],
            "markers": list(_parse_jsonb(row["markers"], []) or []),
            "items": list(_parse_jsonb(row["items"], []) or []),
            "budget": dict(_parse_jsonb(row["budget"], {}) or {}),
            "resolved_source_ids": list(_parse_jsonb(row["resolved_source_ids"], []) or []),
            "resolved_span_ids": list(_parse_jsonb(row["resolved_span_ids"], []) or []),
            "resolved_source_count": int(row["resolved_source_count"] or 0),
            "resolved_span_count": int(row["resolved_span_count"] or 0),
            "unresolved_ref_ids": list(_parse_jsonb(row["unresolved_ref_ids"], []) or []),
            "marker_resolution": list(_parse_jsonb(row["marker_resolution"], []) or []),
            "resolution_mode": str(row["resolution_mode"] or "empty"),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def save_demo_answer_lineage(self, answer_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_answer_lineage (
                answer_lineage_id, conversation_id, run_id, response_turn_id,
                context_id, answer_text, cited_markers, lineage,
                resolved_source_ids, resolved_span_ids,
                resolved_source_count, resolved_span_count, unresolved_ref_ids, marker_resolution, resolution_mode,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb,
                      $9::jsonb, $10::jsonb, $11, $12, $13::jsonb, $14::jsonb, $15,
                      COALESCE($16::timestamptz, NOW()))
            ON CONFLICT (answer_lineage_id) DO UPDATE SET
                answer_text = EXCLUDED.answer_text,
                cited_markers = EXCLUDED.cited_markers,
                lineage = EXCLUDED.lineage,
                resolved_source_ids = EXCLUDED.resolved_source_ids,
                resolved_span_ids = EXCLUDED.resolved_span_ids,
                resolved_source_count = EXCLUDED.resolved_source_count,
                resolved_span_count = EXCLUDED.resolved_span_count,
                unresolved_ref_ids = EXCLUDED.unresolved_ref_ids,
                marker_resolution = EXCLUDED.marker_resolution,
                resolution_mode = EXCLUDED.resolution_mode
            """,
            _uuid.UUID(answer_json["answer_lineage_id"]),
            _uuid.UUID(answer_json["conversation_id"]),
            _uuid.UUID(answer_json["run_id"]),
            _uuid.UUID(answer_json["response_turn_id"]) if answer_json.get("response_turn_id") else None,
            _uuid.UUID(answer_json["context_id"]) if answer_json.get("context_id") else None,
            answer_json.get("answer_text", ""),
            json.dumps(answer_json.get("cited_markers", []), ensure_ascii=False),
            json.dumps(answer_json.get("lineage", []), ensure_ascii=False),
            json.dumps(answer_json.get("resolved_source_ids", []), ensure_ascii=False),
            json.dumps(answer_json.get("resolved_span_ids", []), ensure_ascii=False),
            int(answer_json.get("resolved_source_count", 0) or 0),
            int(answer_json.get("resolved_span_count", 0) or 0),
            json.dumps(answer_json.get("unresolved_ref_ids", []), ensure_ascii=False),
            json.dumps(answer_json.get("marker_resolution", []), ensure_ascii=False),
            str(answer_json.get("resolution_mode") or "empty"),
            _parse_ts(answer_json.get("created_at")),
        )

    async def load_latest_demo_context_snapshot_for_run(self, run_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            """
            SELECT context_id, run_id, conversation_id, turn_id, query_text, phase,
                   context_text, markers, items, budget,
                   resolved_source_ids, resolved_span_ids, resolved_source_count,
                   resolved_span_count, unresolved_ref_ids, marker_resolution, resolution_mode, created_at
            FROM demo_context_snapshots
            WHERE run_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            _uuid.UUID(run_id),
        )
        if row is None:
            return None
        return {
            "context_id": str(row["context_id"]),
            "run_id": str(row["run_id"]),
            "conversation_id": str(row["conversation_id"]),
            "turn_id": str(row["turn_id"]),
            "query_text": row["query_text"],
            "phase": row["phase"],
            "context_text": row["context_text"],
            "markers": list(_parse_jsonb(row["markers"], []) or []),
            "items": list(_parse_jsonb(row["items"], []) or []),
            "budget": dict(_parse_jsonb(row["budget"], {}) or {}),
            "resolved_source_ids": list(_parse_jsonb(row["resolved_source_ids"], []) or []),
            "resolved_span_ids": list(_parse_jsonb(row["resolved_span_ids"], []) or []),
            "resolved_source_count": int(row["resolved_source_count"] or 0),
            "resolved_span_count": int(row["resolved_span_count"] or 0),
            "unresolved_ref_ids": list(_parse_jsonb(row["unresolved_ref_ids"], []) or []),
            "marker_resolution": list(_parse_jsonb(row["marker_resolution"], []) or []),
            "resolution_mode": str(row["resolution_mode"] or "empty"),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def load_demo_context_snapshots_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT context_id, run_id, conversation_id, turn_id, query_text, phase,
                   context_text, markers, items, budget,
                   resolved_source_ids, resolved_span_ids, resolved_source_count,
                   resolved_span_count, unresolved_ref_ids, marker_resolution, resolution_mode, created_at
            FROM demo_context_snapshots
            WHERE run_id = $1
            ORDER BY created_at ASC
            """,
            _uuid.UUID(run_id),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({
                "context_id": str(row["context_id"]),
                "run_id": str(row["run_id"]),
                "conversation_id": str(row["conversation_id"]),
                "turn_id": str(row["turn_id"]),
                "query_text": row["query_text"],
                "phase": row["phase"],
                "context_text": row["context_text"],
                "markers": list(_parse_jsonb(row["markers"], []) or []),
                "items": list(_parse_jsonb(row["items"], []) or []),
                "budget": dict(_parse_jsonb(row["budget"], {}) or {}),
                "resolved_source_ids": list(_parse_jsonb(row["resolved_source_ids"], []) or []),
                "resolved_span_ids": list(_parse_jsonb(row["resolved_span_ids"], []) or []),
                "resolved_source_count": int(row["resolved_source_count"] or 0),
                "resolved_span_count": int(row["resolved_span_count"] or 0),
                "unresolved_ref_ids": list(_parse_jsonb(row["unresolved_ref_ids"], []) or []),
                "marker_resolution": list(_parse_jsonb(row["marker_resolution"], []) or []),
                "resolution_mode": str(row["resolution_mode"] or "empty"),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })
        return out

    async def load_demo_sources(self, conversation_id: str, source_ids: list[str]) -> list[dict[str, Any]]:
        if not source_ids:
            return []
        rows = await self._pool.fetch(
            """
            SELECT source_id, conversation_id, source_kind, role, origin_ref,
                   display_text, process_text, metadata, created_at
            FROM demo_sources
            WHERE conversation_id = $1 AND source_id = ANY($2::uuid[])
            ORDER BY created_at ASC
            """,
            _uuid.UUID(conversation_id),
            [_uuid.UUID(v) for v in source_ids],
        )
        return [{
            "source_id": str(r["source_id"]),
            "conversation_id": str(r["conversation_id"]),
            "source_kind": r["source_kind"],
            "role": r["role"],
            "origin_ref": r["origin_ref"],
            "display_text": r["display_text"],
            "process_text": r["process_text"],
            "metadata": dict(_parse_jsonb(r["metadata"], {})),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        } for r in rows]

    async def load_demo_source_spans(self, conversation_id: str, span_ids: list[str]) -> list[dict[str, Any]]:
        if not span_ids:
            return []
        rows = await self._pool.fetch(
            """
            SELECT span_id, source_id, conversation_id, span_kind, char_start, char_end,
                   locator, preview_text, created_at
            FROM demo_source_spans
            WHERE conversation_id = $1 AND span_id = ANY($2::uuid[])
            ORDER BY created_at ASC
            """,
            _uuid.UUID(conversation_id),
            [_uuid.UUID(v) for v in span_ids],
        )
        return [{
            "span_id": str(r["span_id"]),
            "source_id": str(r["source_id"]),
            "conversation_id": str(r["conversation_id"]),
            "span_kind": r["span_kind"],
            "char_start": int(r["char_start"]),
            "char_end": int(r["char_end"]),
            "locator": dict(_parse_jsonb(r["locator"], {})),
            "preview_text": r["preview_text"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        } for r in rows]

    async def load_demo_lineage_edges(
        self,
        conversation_id: str,
        *,
        src_kind: str | None = None,
        src_ids: list[str] | None = None,
        dst_kind: str | None = None,
        dst_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["conversation_id = $1"]
        args: list[Any] = [_uuid.UUID(conversation_id)]
        idx = 2
        if src_kind is not None:
            clauses.append(f"src_kind = ${idx}")
            args.append(src_kind)
            idx += 1
        if src_ids:
            clauses.append(f"src_id = ANY(${idx}::text[])")
            args.append(list(src_ids))
            idx += 1
        if dst_kind is not None:
            clauses.append(f"dst_kind = ${idx}")
            args.append(dst_kind)
            idx += 1
        if dst_ids:
            clauses.append(f"dst_id = ANY(${idx}::text[])")
            args.append(list(dst_ids))
            idx += 1
        query = "SELECT edge_id, conversation_id, src_kind, src_id, dst_kind, dst_id, relation, metadata, created_at FROM demo_lineage_edges WHERE " + " AND ".join(clauses) + " ORDER BY created_at ASC"
        rows = await self._pool.fetch(query, *args)
        return [{
            "edge_id": str(r["edge_id"]),
            "conversation_id": str(r["conversation_id"]),
            "src_kind": r["src_kind"],
            "src_id": r["src_id"],
            "dst_kind": r["dst_kind"],
            "dst_id": r["dst_id"],
            "relation": r["relation"],
            "metadata": dict(_parse_jsonb(r["metadata"], {})),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        } for r in rows]

    async def load_demo_summary_resolutions(self, conversation_id: str, summary_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if summary_ids:
            rows = await self._pool.fetch(
                """
                SELECT summary_id, conversation_id, summary_text, origin_citem_ids, metadata, created_at
                FROM demo_summary_resolutions
                WHERE conversation_id = $1 AND summary_id = ANY($2::uuid[])
                ORDER BY created_at ASC
                """,
                _uuid.UUID(conversation_id),
                [_uuid.UUID(v) for v in summary_ids],
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT summary_id, conversation_id, summary_text, origin_citem_ids, metadata, created_at
                FROM demo_summary_resolutions
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                """,
                _uuid.UUID(conversation_id),
            )
        return [{
            "summary_id": str(r["summary_id"]),
            "conversation_id": str(r["conversation_id"]),
            "summary_text": r["summary_text"],
            "origin_citem_ids": [str(v) for v in (r["origin_citem_ids"] or [])],
            "metadata": dict(_parse_jsonb(r["metadata"], {})),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        } for r in rows]

    async def save_demo_handoff_manifest(self, manifest_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_handoff_manifests (
                handoff_id, conversation_id, source_run_id, context_id, checksum, manifest, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, COALESCE($7::timestamptz, NOW()))
            ON CONFLICT (handoff_id) DO UPDATE SET
                checksum = EXCLUDED.checksum,
                manifest = EXCLUDED.manifest
            """,
            _uuid.UUID(manifest_json["handoff_id"]),
            _uuid.UUID(manifest_json["conversation_id"]),
            _uuid.UUID(manifest_json["source_run_id"]),
            _uuid.UUID(manifest_json["context_id"]) if manifest_json.get("context_id") else None,
            manifest_json.get("checksum", ""),
            json.dumps(manifest_json, ensure_ascii=False),
            _parse_ts(manifest_json.get("created_at")),
        )

    async def load_demo_handoff_manifest(self, handoff_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            "SELECT manifest FROM demo_handoff_manifests WHERE handoff_id = $1",
            _uuid.UUID(handoff_id),
        )
        if row is None:
            return None
        manifest = row["manifest"] or {}
        return dict(manifest) if not isinstance(manifest, str) else json.loads(manifest)

    async def save_demo_handoff_validation(self, validation_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_handoff_validations (
                handoff_id, valid, issues, evidence_coverage, validation, created_at
            ) VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, COALESCE($6::timestamptz, NOW()))
            ON CONFLICT (handoff_id) DO UPDATE SET
                valid = EXCLUDED.valid,
                issues = EXCLUDED.issues,
                evidence_coverage = EXCLUDED.evidence_coverage,
                validation = EXCLUDED.validation,
                created_at = EXCLUDED.created_at
            """,
            _uuid.UUID(validation_json["handoff_id"]),
            bool(validation_json.get("valid", False)),
            json.dumps(validation_json.get("issues", []), ensure_ascii=False),
            float(validation_json.get("evidence_coverage", 0.0) or 0.0),
            json.dumps(validation_json, ensure_ascii=False),
            _parse_ts(validation_json.get("validated_at") or _parse_ts(validation_json.get("created_at"))),
        )

    async def load_demo_handoff_validation(self, handoff_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            "SELECT validation FROM demo_handoff_validations WHERE handoff_id = $1",
            _uuid.UUID(handoff_id),
        )
        if row is None:
            return None
        payload = row["validation"] or {}
        return dict(payload) if not isinstance(payload, str) else json.loads(payload)

    async def save_demo_handoff_restore(self, restore_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_handoff_restores (
                restore_id, handoff_id, target_conversation_id, target_run_id, valid,
                reconstructed_task_state, diff, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, COALESCE($8::timestamptz, NOW()))
            ON CONFLICT (restore_id) DO UPDATE SET
                valid = EXCLUDED.valid,
                reconstructed_task_state = EXCLUDED.reconstructed_task_state,
                diff = EXCLUDED.diff,
                created_at = EXCLUDED.created_at
            """,
            _uuid.UUID(restore_json["restore_id"]),
            _uuid.UUID(restore_json["handoff_id"]),
            _uuid.UUID(restore_json["target_conversation_id"]),
            _uuid.UUID(restore_json["target_run_id"]) if restore_json.get("target_run_id") else None,
            bool(restore_json.get("valid", False)),
            json.dumps(restore_json.get("reconstructed_task_state", {}), ensure_ascii=False),
            json.dumps(restore_json.get("diff", {}), ensure_ascii=False),
            _parse_ts(restore_json.get("restored_at") or _parse_ts(restore_json.get("created_at"))),
        )

    async def save_demo_gc_audit(self, audit_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO demo_gc_audits (
                audit_id, conversation_id, run_id, action, status, phase, before_counts, after_counts,
                metrics, consistency, notes, error_class, audit, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb, $12, $13::jsonb, COALESCE($14::timestamptz, NOW()))
            ON CONFLICT (audit_id) DO UPDATE SET
                status = EXCLUDED.status,
                phase = EXCLUDED.phase,
                before_counts = EXCLUDED.before_counts,
                after_counts = EXCLUDED.after_counts,
                metrics = EXCLUDED.metrics,
                consistency = EXCLUDED.consistency,
                notes = EXCLUDED.notes,
                error_class = EXCLUDED.error_class,
                audit = EXCLUDED.audit,
                created_at = EXCLUDED.created_at
            """,
            _uuid.UUID(audit_json["audit_id"]),
            _uuid.UUID(audit_json["conversation_id"]),
            _uuid.UUID(audit_json["run_id"]) if audit_json.get("run_id") else None,
            audit_json.get("action", ""),
            audit_json.get("status", "ok"),
            audit_json.get("phase"),
            json.dumps(audit_json.get("before_counts", {}), ensure_ascii=False),
            json.dumps(audit_json.get("after_counts", {}), ensure_ascii=False),
            json.dumps(audit_json.get("metrics", {}), ensure_ascii=False),
            json.dumps(audit_json.get("consistency", {}), ensure_ascii=False),
            json.dumps(audit_json.get("notes", []), ensure_ascii=False),
            audit_json.get("error_class"),
            json.dumps(audit_json, ensure_ascii=False),
            _parse_ts(audit_json.get("created_at")),
        )

    async def load_demo_gc_audits(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT audit
            FROM demo_gc_audits
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            """,
            _uuid.UUID(conversation_id),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row["audit"] or {}
            out.append(dict(payload) if not isinstance(payload, str) else json.loads(payload))
        return out

    async def load_demo_conversation_counts(self, conversation_id: str) -> dict[str, Any]:
        row = await self._pool.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM conversations WHERE conversation_id = $1) AS conversations,
                (SELECT COUNT(*) FROM task_memory WHERE conversation_id = $1) AS task_memory,
                (SELECT COUNT(*) FROM conversation_turns WHERE conversation_id = $1) AS conversation_turns,
                (SELECT COUNT(*) FROM summary_nodes WHERE conversation_id = $1) AS summary_nodes,
                (SELECT COUNT(*) FROM task_metadata WHERE conversation_id = $1) AS task_metadata,
                (SELECT COUNT(*) FROM plans WHERE conversation_id = $1) AS plans,
                (SELECT COUNT(*) FROM plan_steps ps JOIN plans p ON p.plan_id = ps.plan_id WHERE p.conversation_id = $1) AS plan_steps,
                (SELECT COUNT(*) FROM conflict_log WHERE conversation_id = $1) AS conflict_log,
                (SELECT COUNT(*) FROM chm_refs WHERE conversation_id = $1) AS chm_refs,
                (SELECT COUNT(*) FROM retrieval_telemetry WHERE conversation_id = $1) AS retrieval_telemetry,
                (SELECT COUNT(*) FROM file_registry WHERE conversation_id = $1) AS file_registry,
                (SELECT COUNT(*) FROM demo_runs WHERE conversation_id = $1) AS demo_runs,
                (SELECT COUNT(*) FROM demo_run_phases p JOIN demo_runs r ON r.run_id = p.run_id WHERE r.conversation_id = $1) AS demo_run_phases,
                (SELECT COUNT(*) FROM demo_checkpoints c JOIN demo_runs r ON r.run_id = c.run_id WHERE r.conversation_id = $1) AS demo_checkpoints,
                (SELECT COUNT(*) FROM demo_sources WHERE conversation_id = $1) AS demo_sources,
                (SELECT COUNT(*) FROM demo_source_spans WHERE conversation_id = $1) AS demo_source_spans,
                (SELECT COUNT(*) FROM demo_lineage_edges WHERE conversation_id = $1) AS demo_lineage_edges,
                (SELECT COUNT(*) FROM demo_summary_resolutions WHERE conversation_id = $1) AS demo_summary_resolutions,
                (SELECT COUNT(*) FROM demo_context_snapshots WHERE conversation_id = $1) AS demo_context_snapshots,
                (SELECT COUNT(*) FROM demo_answer_lineage WHERE conversation_id = $1) AS demo_answer_lineage,
                (SELECT COUNT(*) FROM demo_handoff_manifests WHERE conversation_id = $1) AS demo_handoff_manifests,
                (SELECT COUNT(*) FROM demo_handoff_validations v JOIN demo_handoff_manifests m ON m.handoff_id = v.handoff_id WHERE m.conversation_id = $1) AS demo_handoff_validations,
                (SELECT COUNT(*) FROM demo_handoff_restores r JOIN demo_handoff_manifests m ON m.handoff_id = r.handoff_id WHERE m.conversation_id = $1) AS demo_handoff_restores_source,
                (SELECT COUNT(*) FROM demo_handoff_restores WHERE target_conversation_id = $1) AS demo_handoff_restores_target,
                (SELECT COUNT(*) FROM geom.runs WHERE conversation_id = $1) AS geometry_runs,
                (SELECT COUNT(*) FROM geom.item_state WHERE conversation_id = $1) AS geometry_item_states,
                (SELECT COUNT(*) FROM geom.cluster_state WHERE conversation_id = $1) AS geometry_cluster_states,
                (SELECT COUNT(*) FROM cima_rm.geom_run WHERE conversation_id = $1) AS geometry_read_model_runs,
                (SELECT COUNT(*) FROM cima_rm.geom_item_state WHERE conversation_id = $1) AS geometry_read_model_item_states,
                (SELECT COUNT(*) FROM cima_rm.geom_cluster_state WHERE conversation_id = $1) AS geometry_read_model_cluster_states
            """,
            _uuid.UUID(conversation_id),
        )
        if row is None:
            return {}
        return {k: int(v or 0) for k, v in dict(row).items()}

    async def _append_outbox_event(
        self,
        *,
        schema_name: str,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int:
        row = await self._pool.fetchrow(
            f"""
            INSERT INTO {schema_name}.outbox (topic, message_key, headers_json, payload_json, status, created_at)
            VALUES ($1, $2, $3::jsonb, $4::jsonb, 'NEW', NOW())
            RETURNING outbox_id
            """,
            topic,
            message_key,
            json.dumps(headers_json or {}, ensure_ascii=False),
            json.dumps(payload_json, ensure_ascii=False) if payload_json is not None else None,
        )
        assert row is not None
        return int(row["outbox_id"])

    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int:
        return await self._append_outbox_event(
            schema_name="cima",
            topic=topic,
            message_key=message_key,
            payload_json=payload_json,
            headers_json=headers_json,
        )

    async def append_geom_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int:
        return await self._append_outbox_event(
            schema_name="geom",
            topic=topic,
            message_key=message_key,
            payload_json=payload_json,
            headers_json=headers_json,
        )

    async def _claim_outbox_batch(self, schema_name: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            f"""
            WITH candidate AS (
                SELECT outbox_id
                FROM {schema_name}.outbox
                WHERE status = 'NEW'
                ORDER BY outbox_id ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE {schema_name}.outbox o
            SET status = 'CLAIMED', claimed_at = NOW()
            FROM candidate c
            WHERE o.outbox_id = c.outbox_id
            RETURNING o.outbox_id, o.topic, o.message_key, o.headers_json, o.payload_json, o.created_at
            """,
            limit,
        )
        return [
            {
                "outbox_id": int(row["outbox_id"]),
                "topic": row["topic"],
                "message_key": row["message_key"],
                "headers_json": dict(_parse_jsonb(row["headers_json"], {}) or {}),
                "payload_json": dict(row["payload_json"]) if row["payload_json"] is not None else None,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]

    async def claim_outbox_batch(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._claim_outbox_batch("cima", limit)

    async def claim_geom_outbox_batch(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._claim_outbox_batch("geom", limit)

    async def _mark_outbox_sent(self, schema_name: str, outbox_ids: list[int]) -> None:
        if not outbox_ids:
            return
        await self._pool.execute(
            f"""
            UPDATE {schema_name}.outbox
            SET status = 'SENT', sent_at = NOW(), error = NULL
            WHERE outbox_id = ANY($1::bigint[])
            """,
            outbox_ids,
        )

    async def mark_outbox_sent(self, outbox_ids: list[int]) -> None:
        await self._mark_outbox_sent("cima", outbox_ids)

    async def mark_geom_outbox_sent(self, outbox_ids: list[int]) -> None:
        await self._mark_outbox_sent("geom", outbox_ids)

    async def _mark_outbox_error(self, schema_name: str, outbox_id: int, error: str) -> None:
        await self._pool.execute(
            f"""
            UPDATE {schema_name}.outbox
            SET status = 'ERROR', error = $2
            WHERE outbox_id = $1
            """,
            outbox_id,
            error,
        )

    async def mark_outbox_error(self, outbox_id: int, error: str) -> None:
        await self._mark_outbox_error("cima", outbox_id, error)

    async def mark_geom_outbox_error(self, outbox_id: int, error: str) -> None:
        await self._mark_outbox_error("geom", outbox_id, error)

    async def begin_consumer_effect(
        self,
        *,
        consumer_name: str,
        event_id: str,
        effect_key: str,
    ) -> bool:
        status = await self._pool.execute(
            """
            INSERT INTO cima.consumer_effect (consumer_name, event_id, effect_key, status, started_at)
            VALUES ($1, $2, $3, 'STARTED', NOW())
            ON CONFLICT (consumer_name, event_id, effect_key) DO NOTHING
            """,
            consumer_name,
            event_id,
            effect_key,
        )
        return status.endswith("1")

    async def complete_consumer_effect(
        self,
        *,
        consumer_name: str,
        event_id: str,
        effect_key: str,
        details_json: dict[str, Any] | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE cima.consumer_effect
            SET status = 'SUCCEEDED', completed_at = NOW(), details_json = $4::jsonb
            WHERE consumer_name = $1 AND event_id = $2 AND effect_key = $3
            """,
            consumer_name,
            event_id,
            effect_key,
            json.dumps(details_json or {}, ensure_ascii=False),
        )

    async def append_citem_audit_event(
        self,
        *,
        conversation_id: str,
        citem_id: str,
        event_type: str,
        old_value: str | None = None,
        new_value: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO citem_audit (citem_id, conversation_id, event_type, old_value, new_value, occurred_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            _uuid.UUID(citem_id),
            _uuid.UUID(conversation_id),
            event_type,
            old_value,
            new_value,
        )

    async def load_citem_audit_events(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT citem_id, conversation_id, event_type, old_value, new_value, occurred_at
            FROM citem_audit
            WHERE conversation_id = $1
            ORDER BY occurred_at ASC
            """,
            _uuid.UUID(conversation_id),
        )
        return [{
            "citem_id": str(r["citem_id"]),
            "conversation_id": str(r["conversation_id"]),
            "event_type": r["event_type"],
            "old_value": r["old_value"],
            "new_value": r["new_value"],
            "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] else None,
        } for r in rows]

    async def save_geometry_run(self, run_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO geom.runs (
                run_id, conversation_id, reason, algo_version, n_items, cluster_count, core_count, bridge_count, metrics, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, COALESCE($10::timestamptz, NOW()))
            ON CONFLICT (run_id) DO UPDATE SET
                reason = EXCLUDED.reason,
                algo_version = EXCLUDED.algo_version,
                n_items = EXCLUDED.n_items,
                cluster_count = EXCLUDED.cluster_count,
                core_count = EXCLUDED.core_count,
                bridge_count = EXCLUDED.bridge_count,
                metrics = EXCLUDED.metrics
            """,
            _uuid.UUID(run_json["run_id"]),
            _uuid.UUID(run_json["conversation_id"]),
            run_json.get("reason", "manual"),
            run_json.get("algo_version", "geom_v1"),
            int(run_json.get("n_items", 0)),
            int(run_json.get("cluster_count", 0)),
            int(run_json.get("core_count", 0)),
            int(run_json.get("bridge_count", 0)),
            json.dumps({k: v for k, v in run_json.items() if k not in {"run_id", "conversation_id", "reason", "algo_version", "n_items", "cluster_count", "core_count", "bridge_count", "created_at"}}, ensure_ascii=False),
            _parse_ts(run_json.get("created_at")),
        )

    async def save_geometry_item_state(self, item_state_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO geom.item_state (
                conversation_id, ref_kind, ref_id, run_id, cluster_top1, cluster_top2,
                w1, w2, margin, is_core, is_bridge_candidate, centrality, label, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, COALESCE($14::timestamptz, NOW()))
            ON CONFLICT (conversation_id, ref_kind, ref_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                cluster_top1 = EXCLUDED.cluster_top1,
                cluster_top2 = EXCLUDED.cluster_top2,
                w1 = EXCLUDED.w1,
                w2 = EXCLUDED.w2,
                margin = EXCLUDED.margin,
                is_core = EXCLUDED.is_core,
                is_bridge_candidate = EXCLUDED.is_bridge_candidate,
                centrality = EXCLUDED.centrality,
                label = EXCLUDED.label,
                updated_at = EXCLUDED.updated_at
            """,
            _uuid.UUID(item_state_json["conversation_id"]),
            item_state_json.get("ref_kind", "citem"),
            _uuid.UUID(item_state_json["ref_id"]),
            _uuid.UUID(item_state_json["run_id"]),
            item_state_json.get("cluster_top1", "c_001"),
            item_state_json.get("cluster_top2"),
            float(item_state_json.get("w1", 1.0)),
            float(item_state_json.get("w2")) if item_state_json.get("w2") is not None else None,
            float(item_state_json.get("margin", 1.0)),
            bool(item_state_json.get("is_core", False)),
            bool(item_state_json.get("is_bridge_candidate", False)),
            float(item_state_json.get("centrality", 0.0)) if item_state_json.get("centrality") is not None else None,
            item_state_json.get("label"),
            _parse_ts(item_state_json.get("updated_at")),
        )

    async def save_geometry_cluster_state(self, cluster_state_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO geom.cluster_state (
                conversation_id, cluster_id, run_id, mass, medoid_ref_id, summary_id, label, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8::timestamptz, NOW()))
            ON CONFLICT (conversation_id, cluster_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                mass = EXCLUDED.mass,
                medoid_ref_id = EXCLUDED.medoid_ref_id,
                summary_id = EXCLUDED.summary_id,
                label = EXCLUDED.label,
                updated_at = EXCLUDED.updated_at
            """,
            _uuid.UUID(cluster_state_json["conversation_id"]),
            cluster_state_json.get("cluster_id", "c_001"),
            _uuid.UUID(cluster_state_json["run_id"]),
            float(cluster_state_json.get("mass", 0.0)),
            _uuid.UUID(cluster_state_json["medoid_ref_id"]),
            _uuid.UUID(cluster_state_json["summary_id"]) if cluster_state_json.get("summary_id") else None,
            cluster_state_json.get("label"),
            _parse_ts(cluster_state_json.get("updated_at")),
        )

    async def load_geometry_item_states(self, conversation_id: str, ref_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if ref_ids:
            rows = await self._pool.fetch(
                """
                SELECT conversation_id, ref_kind, ref_id, run_id, cluster_top1, cluster_top2,
                       w1, w2, margin, is_core, is_bridge_candidate, centrality, label, updated_at
                FROM geom.item_state
                WHERE conversation_id = $1 AND ref_id = ANY($2::uuid[])
                ORDER BY updated_at DESC
                """,
                _uuid.UUID(conversation_id),
                [_uuid.UUID(ref_id) for ref_id in ref_ids],
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT conversation_id, ref_kind, ref_id, run_id, cluster_top1, cluster_top2,
                       w1, w2, margin, is_core, is_bridge_candidate, centrality, label, updated_at
                FROM geom.item_state
                WHERE conversation_id = $1
                ORDER BY updated_at DESC
                """,
                _uuid.UUID(conversation_id),
            )
        return [self._row_to_geometry_item_state(row) for row in rows]

    async def load_geometry_cluster_states(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT conversation_id, cluster_id, run_id, mass, medoid_ref_id, summary_id, label, updated_at
            FROM geom.cluster_state
            WHERE conversation_id = $1
            ORDER BY updated_at DESC
            """,
            _uuid.UUID(conversation_id),
        )
        return [self._row_to_geometry_cluster_state(row) for row in rows]

    async def delete_geometry_conversation(self, conversation_id: str) -> None:
        cid = _uuid.UUID(conversation_id)
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM geom.item_state WHERE conversation_id = $1", cid)
            await conn.execute("DELETE FROM geom.cluster_state WHERE conversation_id = $1", cid)
            await conn.execute("DELETE FROM geom.runs WHERE conversation_id = $1", cid)

    async def save_geometry_read_model_run(self, run_json: dict[str, Any]) -> None:
        params = dict(run_json.get("params") or {})
        metrics = dict(run_json.get("metrics") or {})
        await self._pool.execute(
            """
            INSERT INTO cima_rm.geom_run (
                conversation_id, run_id, algo_version, universe_hash,
                k_used, temp, core_q, bridge_percentile, metrics_json, completed_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, COALESCE($10::timestamptz, NOW()))
            ON CONFLICT (conversation_id, run_id) DO UPDATE SET
                algo_version = EXCLUDED.algo_version,
                universe_hash = EXCLUDED.universe_hash,
                k_used = EXCLUDED.k_used,
                temp = EXCLUDED.temp,
                core_q = EXCLUDED.core_q,
                bridge_percentile = EXCLUDED.bridge_percentile,
                metrics_json = EXCLUDED.metrics_json,
                completed_at = EXCLUDED.completed_at
            """,
            _uuid.UUID(run_json["conversation_id"]),
            _uuid.UUID(run_json["run_id"]),
            run_json.get("algo_version", "geom_v1.0"),
            run_json.get("universe_hash", ""),
            int(params.get("k_used", 0)),
            float(params.get("temp", 0.0) or 0.0),
            float(params.get("core_q", 0.0) or 0.0),
            int(params.get("bridge_percentile", 0) or 0),
            json.dumps(metrics, ensure_ascii=False),
            _parse_ts(run_json.get("completed_at")),
        )

    async def save_geometry_read_model_item_state(self, item_state_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima_rm.geom_item_state (
                conversation_id, ref_kind, ref_id, run_id, cluster_top1, cluster_top2,
                w1, w2, margin, is_core, is_bridge_candidate, centrality, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, COALESCE($13::timestamptz, NOW()))
            ON CONFLICT (conversation_id, ref_kind, ref_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                cluster_top1 = EXCLUDED.cluster_top1,
                cluster_top2 = EXCLUDED.cluster_top2,
                w1 = EXCLUDED.w1,
                w2 = EXCLUDED.w2,
                margin = EXCLUDED.margin,
                is_core = EXCLUDED.is_core,
                is_bridge_candidate = EXCLUDED.is_bridge_candidate,
                centrality = EXCLUDED.centrality,
                updated_at = EXCLUDED.updated_at
            """,
            _uuid.UUID(item_state_json["conversation_id"]),
            item_state_json.get("ref_kind", "local_citem"),
            _uuid.UUID(item_state_json["ref_id"]),
            _uuid.UUID(item_state_json["run_id"]),
            item_state_json.get("cluster_top1", "c_001"),
            item_state_json.get("cluster_top2"),
            float(item_state_json.get("w1", 1.0)),
            float(item_state_json.get("w2")) if item_state_json.get("w2") is not None else None,
            float(item_state_json.get("margin", 1.0)),
            bool(item_state_json.get("is_core", False)),
            bool(item_state_json.get("is_bridge_candidate", False)),
            float(item_state_json.get("centrality", 0.0)) if item_state_json.get("centrality") is not None else None,
            _parse_ts(item_state_json.get("updated_at")),
        )

    async def save_geometry_read_model_cluster_state(self, cluster_state_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima_rm.geom_cluster_state (
                conversation_id, cluster_id, run_id, mass, medoid_ref_kind, medoid_ref_id, summary_id, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8::timestamptz, NOW()))
            ON CONFLICT (conversation_id, cluster_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                mass = EXCLUDED.mass,
                medoid_ref_kind = EXCLUDED.medoid_ref_kind,
                medoid_ref_id = EXCLUDED.medoid_ref_id,
                summary_id = EXCLUDED.summary_id,
                updated_at = EXCLUDED.updated_at
            """,
            _uuid.UUID(cluster_state_json["conversation_id"]),
            cluster_state_json.get("cluster_id", "c_001"),
            _uuid.UUID(cluster_state_json["run_id"]),
            float(cluster_state_json.get("mass", 0.0)),
            cluster_state_json.get("medoid_ref_kind", "local_citem"),
            _uuid.UUID(cluster_state_json["medoid_ref_id"]),
            _uuid.UUID(cluster_state_json["summary_id"]) if cluster_state_json.get("summary_id") else None,
            _parse_ts(cluster_state_json.get("updated_at")),
        )

    async def load_geometry_read_model_runs(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT conversation_id, run_id, algo_version, universe_hash, k_used, temp, core_q, bridge_percentile, metrics_json, completed_at
            FROM cima_rm.geom_run
            WHERE conversation_id = $1
            ORDER BY completed_at DESC
            """,
            _uuid.UUID(conversation_id),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({
                "conversation_id": str(row["conversation_id"]),
                "run_id": str(row["run_id"]),
                "algo_version": row["algo_version"],
                "universe_hash": row["universe_hash"],
                "params": {
                    "k_used": int(row["k_used"]),
                    "temp": float(row["temp"]),
                    "core_q": float(row["core_q"]),
                    "bridge_percentile": int(row["bridge_percentile"]),
                },
                "metrics": dict(_parse_jsonb(row["metrics_json"], {}) or {}),
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            })
        return out

    async def load_geometry_read_model_item_states(self, conversation_id: str, ref_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if ref_ids:
            rows = await self._pool.fetch(
                """
                SELECT conversation_id, ref_kind, ref_id, run_id, cluster_top1, cluster_top2,
                       w1, w2, margin, is_core, is_bridge_candidate, centrality, updated_at
                FROM cima_rm.geom_item_state
                WHERE conversation_id = $1 AND ref_id = ANY($2::uuid[])
                ORDER BY updated_at DESC
                """,
                _uuid.UUID(conversation_id),
                [_uuid.UUID(ref_id) for ref_id in ref_ids],
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT conversation_id, ref_kind, ref_id, run_id, cluster_top1, cluster_top2,
                       w1, w2, margin, is_core, is_bridge_candidate, centrality, updated_at
                FROM cima_rm.geom_item_state
                WHERE conversation_id = $1
                ORDER BY updated_at DESC
                """,
                _uuid.UUID(conversation_id),
            )
        return [self._row_to_geometry_item_state(row) for row in rows]

    async def load_geometry_read_model_cluster_states(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT conversation_id, cluster_id, run_id, mass, medoid_ref_kind, medoid_ref_id, summary_id, updated_at
            FROM cima_rm.geom_cluster_state
            WHERE conversation_id = $1
            ORDER BY updated_at DESC
            """,
            _uuid.UUID(conversation_id),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({
                "conversation_id": str(row["conversation_id"]),
                "cluster_id": row["cluster_id"],
                "run_id": str(row["run_id"]),
                "mass": float(row["mass"]),
                "medoid_ref_kind": row["medoid_ref_kind"],
                "medoid_ref_id": str(row["medoid_ref_id"]),
                "summary_id": str(row["summary_id"]) if row["summary_id"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            })
        return out

    async def delete_geometry_read_model_conversation(self, conversation_id: str) -> None:
        cid = _uuid.UUID(conversation_id)
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM cima_rm.geom_item_state WHERE conversation_id = $1", cid)
            await conn.execute("DELETE FROM cima_rm.geom_cluster_state WHERE conversation_id = $1", cid)
            await conn.execute("DELETE FROM cima_rm.geom_run WHERE conversation_id = $1", cid)

    async def delete_geometry_read_model_item_state(self, conversation_id: str, ref_kind: str, ref_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM cima_rm.geom_item_state WHERE conversation_id = $1 AND ref_kind = $2 AND ref_id = $3",
            _uuid.UUID(conversation_id),
            ref_kind,
            _uuid.UUID(ref_id),
        )

    async def delete_geometry_read_model_cluster_state(self, conversation_id: str, cluster_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM cima_rm.geom_cluster_state WHERE conversation_id = $1 AND cluster_id = $2",
            _uuid.UUID(conversation_id),
            cluster_id,
        )

    @staticmethod
    def _row_to_geometry_item_state(row: Any) -> dict[str, Any]:
        return {
            "conversation_id": str(row["conversation_id"]),
            "ref_kind": row["ref_kind"],
            "ref_id": str(row["ref_id"]),
            "run_id": str(row["run_id"]),
            "cluster_top1": row["cluster_top1"],
            "cluster_top2": row["cluster_top2"],
            "w1": float(row["w1"]),
            "w2": float(row["w2"]) if row["w2"] is not None else None,
            "margin": float(row["margin"]),
            "is_core": bool(row["is_core"]),
            "is_bridge_candidate": bool(row["is_bridge_candidate"]),
            "centrality": float(row["centrality"]) if row["centrality"] is not None else None,
            "label": row.get("label") if hasattr(row, 'get') else row["label"] if "label" in row else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    @staticmethod
    def _row_to_geometry_cluster_state(row: Any) -> dict[str, Any]:
        return {
            "conversation_id": str(row["conversation_id"]),
            "cluster_id": row["cluster_id"],
            "run_id": str(row["run_id"]),
            "mass": float(row["mass"]),
            "medoid_ref_id": str(row["medoid_ref_id"]),
            "summary_id": str(row["summary_id"]) if row["summary_id"] else None,
            "label": row.get("label") if hasattr(row, 'get') else row["label"] if "label" in row else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    # ── File registry ─────────────────────────────────────────────────────────

    async def save_file_record(self, record: object) -> None:
        from cima_demo.domain.entities import FileRecord
        assert isinstance(record, FileRecord)
        await self._pool.execute(
            """
            INSERT INTO file_registry
                (file_id, conversation_id, filename, mime_type, size_bytes,
                 content_hash, status, chunk_count, citem_ids, blob_path,
                 ingested_at, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (file_id) DO NOTHING
            """,
            record.file_id,
            record.conversation_id,
            record.filename,
            record.mime_type,
            record.size_bytes,
            record.content_hash,
            record.status,
            record.chunk_count,
            [str(c) for c in record.citem_ids],
            record.blob_path,
            record.ingested_at,
            record.error_message,
        )

    async def update_file_record(
        self,
        file_id: str,
        *,
        status: str,
        chunk_count: int = 0,
        citem_ids: list | None = None,
        error_message: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE file_registry
            SET status        = $2,
                chunk_count   = $3,
                citem_ids     = $4,
                error_message = $5
            WHERE file_id = $1
            """,
            file_id,
            status,
            chunk_count,
            [str(c) for c in (citem_ids or [])],
            error_message,
        )

    async def list_file_records(self, conversation_id: str) -> list:
        from cima_demo.domain.entities import FileRecord
        rows = await self._pool.fetch(
            """
            SELECT file_id, conversation_id, filename, mime_type, size_bytes,
                   content_hash, status, chunk_count, citem_ids, blob_path,
                   ingested_at, error_message
            FROM file_registry
            WHERE conversation_id = $1
            ORDER BY ingested_at DESC
            """,
            conversation_id,
        )
        return [
            FileRecord(
                file_id=str(r["file_id"]),
                conversation_id=str(r["conversation_id"]),
                filename=r["filename"],
                mime_type=r["mime_type"],
                size_bytes=r["size_bytes"],
                content_hash=r["content_hash"],
                status=r["status"],
                chunk_count=r["chunk_count"],
                citem_ids=[str(c) for c in (r["citem_ids"] or [])],
                blob_path=r["blob_path"],
                ingested_at=r["ingested_at"],
                error_message=r["error_message"],
            )
            for r in rows
        ]

    async def get_file_record(self, file_id: str):
        from cima_demo.domain.entities import FileRecord
        row = await self._pool.fetchrow(
            """
            SELECT file_id, conversation_id, filename, mime_type, size_bytes,
                   content_hash, status, chunk_count, citem_ids, blob_path,
                   ingested_at, error_message
            FROM file_registry
            WHERE file_id = $1
            """,
            file_id,
        )
        if row is None:
            return None
        return FileRecord(
            file_id=str(row["file_id"]),
            conversation_id=str(row["conversation_id"]),
            filename=row["filename"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            content_hash=row["content_hash"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            citem_ids=[str(c) for c in (row["citem_ids"] or [])],
            blob_path=row["blob_path"],
            ingested_at=row["ingested_at"],
            error_message=row["error_message"],
        )

    async def save_chunk_record(self, chunk_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.chunk_manifest (
                chunk_id, conversation_id, source_id, file_id, source_span_id,
                chunk_kind, chunk_index, page_num, section_hint,
                normalizer_version, chunker_version, vector_state,
                embedding_model_id, embedding_schema_version, expires_at, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12,
                $13, $14, $15, COALESCE($16::timestamptz, NOW())
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                file_id = EXCLUDED.file_id,
                source_span_id = EXCLUDED.source_span_id,
                page_num = EXCLUDED.page_num,
                section_hint = EXCLUDED.section_hint,
                vector_state = EXCLUDED.vector_state,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                expires_at = EXCLUDED.expires_at
            """,
            _uuid.UUID(chunk_json["chunk_id"]),
            _uuid.UUID(chunk_json["conversation_id"]),
            _uuid.UUID(chunk_json["source_id"]),
            _uuid.UUID(chunk_json["file_id"]) if chunk_json.get("file_id") else None,
            _uuid.UUID(chunk_json["source_span_id"]) if chunk_json.get("source_span_id") else None,
            chunk_json["chunk_kind"],
            int(chunk_json.get("chunk_index", 0)),
            int(chunk_json["page_num"]) if chunk_json.get("page_num") is not None else None,
            chunk_json.get("section_hint"),
            int(chunk_json.get("normalizer_version", 1)),
            int(chunk_json.get("chunker_version", 1)),
            chunk_json.get("vector_state", "NONE"),
            chunk_json.get("embedding_model_id"),
            int(chunk_json["embedding_schema_version"]) if chunk_json.get("embedding_schema_version") is not None else None,
            _parse_ts(chunk_json.get("expires_at")),
            _parse_ts(chunk_json.get("created_at")),
        )

    async def list_chunk_records(self, conversation_id: str, *, source_id: str | None = None) -> list[dict[str, Any]]:
        if source_id is None:
            rows = await self._pool.fetch(
                """
                SELECT chunk_id, conversation_id, source_id, file_id, source_span_id,
                       chunk_kind, chunk_index, page_num, section_hint,
                       normalizer_version, chunker_version, vector_state,
                       embedding_model_id, embedding_schema_version, expires_at, created_at
                FROM cima.chunk_manifest
                WHERE conversation_id = $1
                ORDER BY created_at ASC, chunk_index ASC
                """,
                _uuid.UUID(conversation_id),
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT chunk_id, conversation_id, source_id, file_id, source_span_id,
                       chunk_kind, chunk_index, page_num, section_hint,
                       normalizer_version, chunker_version, vector_state,
                       embedding_model_id, embedding_schema_version, expires_at, created_at
                FROM cima.chunk_manifest
                WHERE conversation_id = $1 AND source_id = $2
                ORDER BY created_at ASC, chunk_index ASC
                """,
                _uuid.UUID(conversation_id),
                _uuid.UUID(source_id),
            )
        return [
            {
                "chunk_id": str(r["chunk_id"]),
                "conversation_id": str(r["conversation_id"]),
                "source_id": str(r["source_id"]),
                "file_id": str(r["file_id"]) if r["file_id"] else None,
                "source_span_id": str(r["source_span_id"]) if r["source_span_id"] else None,
                "chunk_kind": r["chunk_kind"],
                "chunk_index": int(r["chunk_index"]),
                "page_num": int(r["page_num"]) if r["page_num"] is not None else None,
                "section_hint": r["section_hint"],
                "normalizer_version": int(r["normalizer_version"]),
                "chunker_version": int(r["chunker_version"]),
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
                "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]


    async def save_edu_record(self, edu_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.edu_manifest (
                edu_id, conversation_id, source_id, chunk_id, edu_kind,
                span_refs_json, features_json, quality,
                normalizer_version, edu_segmenter_version, created_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6::jsonb, $7::jsonb, $8,
                $9, $10, COALESCE($11::timestamptz, NOW())
            )
            ON CONFLICT (edu_id) DO UPDATE SET
                edu_kind = EXCLUDED.edu_kind,
                span_refs_json = EXCLUDED.span_refs_json,
                features_json = EXCLUDED.features_json,
                quality = EXCLUDED.quality,
                normalizer_version = EXCLUDED.normalizer_version,
                edu_segmenter_version = EXCLUDED.edu_segmenter_version
            """,
            _uuid.UUID(edu_json["edu_id"]),
            _uuid.UUID(edu_json["conversation_id"]),
            _uuid.UUID(edu_json["source_id"]),
            _uuid.UUID(edu_json["chunk_id"]),
            edu_json["edu_kind"],
            json.dumps(edu_json.get("span_refs_json", []), ensure_ascii=False),
            json.dumps(edu_json.get("features_json", {}), ensure_ascii=False),
            float(edu_json.get("quality", 1.0)),
            int(edu_json.get("normalizer_version", 1)),
            int(edu_json.get("edu_segmenter_version", 1)),
            _parse_ts(edu_json.get("created_at")),
        )

    async def list_edu_records(
        self,
        conversation_id: str,
        *,
        chunk_id: str | None = None,
        edu_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["conversation_id = $1"]
        params: list[Any] = [_uuid.UUID(conversation_id)]
        if chunk_id is not None:
            clauses.append(f"chunk_id = ${len(params) + 1}")
            params.append(_uuid.UUID(chunk_id))
        if edu_ids:
            clauses.append(f"edu_id = ANY(${len(params) + 1}::uuid[])")
            params.append([_uuid.UUID(v) for v in edu_ids])
        rows = await self._pool.fetch(
            f"""
            SELECT edu_id, conversation_id, source_id, chunk_id, edu_kind,
                   span_refs_json, features_json, quality,
                   normalizer_version, edu_segmenter_version, created_at
            FROM cima.edu_manifest
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC, edu_id ASC
            """,
            *params,
        )
        return [
            {
                "edu_id": str(r["edu_id"]),
                "conversation_id": str(r["conversation_id"]),
                "source_id": str(r["source_id"]),
                "chunk_id": str(r["chunk_id"]),
                "edu_kind": r["edu_kind"],
                "span_refs_json": dict(_parse_jsonb(r["span_refs_json"], {})) if isinstance(_parse_jsonb(r["span_refs_json"], None), dict) else list(_parse_jsonb(r["span_refs_json"], []) or []),
                "features_json": dict(_parse_jsonb(r["features_json"], {})),
                "quality": float(r["quality"]),
                "normalizer_version": int(r["normalizer_version"]),
                "edu_segmenter_version": int(r["edu_segmenter_version"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    async def save_local_citem_record(self, citem_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.local_citem (
                local_citem_id, semantic_identity_id, conversation_id, type, text, embedding_text,
                meta_json, provenance_json, validity, salience, created_at, updated_at,
                vector_state, embedding_model_id, embedding_schema_version, expires_at,
                is_pinned, was_cited, last_used_at, normalizer_version, citem_builder_version
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7::jsonb, $8::jsonb, $9, $10, COALESCE($11::timestamptz, NOW()), COALESCE($12::timestamptz, NOW()),
                $13, $14, $15, $16,
                $17, $18, $19, $20, $21
            )
            ON CONFLICT (local_citem_id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding_text = EXCLUDED.embedding_text,
                meta_json = EXCLUDED.meta_json,
                provenance_json = EXCLUDED.provenance_json,
                validity = EXCLUDED.validity,
                salience = EXCLUDED.salience,
                updated_at = COALESCE(EXCLUDED.updated_at, NOW()),
                vector_state = EXCLUDED.vector_state,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                expires_at = EXCLUDED.expires_at,
                is_pinned = EXCLUDED.is_pinned,
                was_cited = EXCLUDED.was_cited,
                last_used_at = EXCLUDED.last_used_at,
                normalizer_version = EXCLUDED.normalizer_version,
                citem_builder_version = EXCLUDED.citem_builder_version
            """,
            _uuid.UUID(citem_json["local_citem_id"]),
            _uuid.UUID(citem_json["semantic_identity_id"]),
            _uuid.UUID(citem_json["conversation_id"]),
            citem_json["type"],
            citem_json["text"],
            citem_json["embedding_text"],
            json.dumps(citem_json.get("meta_json", {}), ensure_ascii=False),
            json.dumps(citem_json.get("provenance_json", {}), ensure_ascii=False),
            citem_json.get("validity", "unknown"),
            float(citem_json.get("salience", 0.0)),
            _parse_ts(citem_json.get("created_at")),
            _parse_ts(citem_json.get("updated_at")),
            citem_json.get("vector_state", "NONE"),
            citem_json.get("embedding_model_id"),
            int(citem_json["embedding_schema_version"]) if citem_json.get("embedding_schema_version") is not None else None,
            _parse_ts(citem_json.get("expires_at")),
            bool(citem_json.get("is_pinned", False)),
            bool(citem_json.get("was_cited", False)),
            _parse_ts(citem_json.get("last_used_at")),
            int(citem_json.get("normalizer_version", 1)),
            int(citem_json.get("citem_builder_version", 1)),
        )

    async def list_local_citem_records(
        self,
        conversation_id: str,
        *,
        citem_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["conversation_id = $1"]
        params: list[Any] = [_uuid.UUID(conversation_id)]
        if citem_ids:
            clauses.append(f"local_citem_id = ANY(${len(params) + 1}::uuid[])")
            params.append([_uuid.UUID(v) for v in citem_ids])
        rows = await self._pool.fetch(
            f"""
            SELECT local_citem_id, semantic_identity_id, conversation_id, type, text, embedding_text,
                   meta_json, provenance_json, validity, salience, created_at, updated_at,
                   vector_state, embedding_model_id, embedding_schema_version, expires_at,
                   is_pinned, was_cited, last_used_at, normalizer_version, citem_builder_version
            FROM cima.local_citem
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC, local_citem_id ASC
            """,
            *params,
        )
        return [
            {
                "local_citem_id": str(r["local_citem_id"]),
                "semantic_identity_id": str(r["semantic_identity_id"]),
                "conversation_id": str(r["conversation_id"]),
                "type": r["type"],
                "text": r["text"],
                "embedding_text": r["embedding_text"],
                "meta_json": dict(_parse_jsonb(r["meta_json"], {})),
                "provenance_json": dict(_parse_jsonb(r["provenance_json"], {})),
                "validity": r["validity"],
                "salience": float(r["salience"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
                "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
                "is_pinned": bool(r["is_pinned"]),
                "was_cited": bool(r["was_cited"]),
                "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
                "normalizer_version": int(r["normalizer_version"]),
                "citem_builder_version": int(r["citem_builder_version"]),
            }
            for r in rows
        ]

    async def save_local_citem_evidence(self, evidence_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.local_citem_evidence (
                local_citem_id, source_id, chunk_id, edu_id, ordinal, locator_json
            ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (local_citem_id, ordinal) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                chunk_id = EXCLUDED.chunk_id,
                edu_id = EXCLUDED.edu_id,
                locator_json = EXCLUDED.locator_json
            """,
            _uuid.UUID(evidence_json["local_citem_id"]),
            _uuid.UUID(evidence_json["source_id"]) if evidence_json.get("source_id") else None,
            _uuid.UUID(evidence_json["chunk_id"]) if evidence_json.get("chunk_id") else None,
            _uuid.UUID(evidence_json["edu_id"]) if evidence_json.get("edu_id") else None,
            int(evidence_json.get("ordinal", 0)),
            json.dumps(evidence_json.get("locator_json", {}), ensure_ascii=False),
        )

    async def list_local_citem_evidence(self, local_citem_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT local_citem_id, source_id, chunk_id, edu_id, ordinal, locator_json
            FROM cima.local_citem_evidence
            WHERE local_citem_id = $1
            ORDER BY ordinal ASC
            """,
            _uuid.UUID(local_citem_id),
        )
        return [
            {
                "local_citem_id": str(r["local_citem_id"]),
                "source_id": str(r["source_id"]) if r["source_id"] else None,
                "chunk_id": str(r["chunk_id"]) if r["chunk_id"] else None,
                "edu_id": str(r["edu_id"]) if r["edu_id"] else None,
                "ordinal": int(r["ordinal"]),
                "locator_json": dict(_parse_jsonb(r["locator_json"], {})),
            }
            for r in rows
        ]

    async def save_local_summary_record(self, summary_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.local_summary (
                local_summary_id, conversation_id, level, cluster_id, epoch_no, text, covers_json,
                created_at, updated_at, vector_state, embedding_model_id, embedding_schema_version,
                is_pinned, was_cited, last_used_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7::jsonb,
                COALESCE($8::timestamptz, NOW()), COALESCE($9::timestamptz, NOW()), $10, $11, $12,
                $13, $14, $15
            )
            ON CONFLICT (local_summary_id) DO UPDATE SET
                level = EXCLUDED.level,
                cluster_id = EXCLUDED.cluster_id,
                epoch_no = EXCLUDED.epoch_no,
                text = EXCLUDED.text,
                covers_json = EXCLUDED.covers_json,
                updated_at = COALESCE(EXCLUDED.updated_at, NOW()),
                vector_state = EXCLUDED.vector_state,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                is_pinned = EXCLUDED.is_pinned,
                was_cited = EXCLUDED.was_cited,
                last_used_at = EXCLUDED.last_used_at
            """,
            _uuid.UUID(summary_json["local_summary_id"]),
            _uuid.UUID(summary_json["conversation_id"]),
            summary_json["level"],
            summary_json.get("cluster_id"),
            int(summary_json["epoch_no"]) if summary_json.get("epoch_no") is not None else None,
            summary_json["text"],
            json.dumps(summary_json.get("covers_json", {}), ensure_ascii=False),
            _parse_ts(summary_json.get("created_at")),
            _parse_ts(summary_json.get("updated_at")),
            summary_json.get("vector_state", "NONE"),
            summary_json.get("embedding_model_id"),
            int(summary_json["embedding_schema_version"]) if summary_json.get("embedding_schema_version") is not None else None,
            bool(summary_json.get("is_pinned", False)),
            bool(summary_json.get("was_cited", False)),
            _parse_ts(summary_json.get("last_used_at")),
        )

    async def list_local_summary_records(
        self,
        conversation_id: str,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        cluster_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["conversation_id = $1"]
        params: list[Any] = [_uuid.UUID(conversation_id)]
        if summary_ids:
            clauses.append(f"local_summary_id = ANY(${len(params) + 1}::uuid[])")
            params.append([_uuid.UUID(v) for v in summary_ids])
        if level is not None:
            clauses.append(f"level = ${len(params) + 1}")
            params.append(level)
        if cluster_id is not None:
            clauses.append(f"cluster_id = ${len(params) + 1}")
            params.append(cluster_id)
        rows = await self._pool.fetch(
            f"""
            SELECT local_summary_id, conversation_id, level, cluster_id, epoch_no, text, covers_json,
                   created_at, updated_at, vector_state, embedding_model_id, embedding_schema_version,
                   is_pinned, was_cited, last_used_at
            FROM cima.local_summary
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, local_summary_id ASC
            """,
            *params,
        )
        return [
            {
                "local_summary_id": str(r["local_summary_id"]),
                "conversation_id": str(r["conversation_id"]),
                "level": r["level"],
                "cluster_id": r["cluster_id"],
                "epoch_no": int(r["epoch_no"]) if r["epoch_no"] is not None else None,
                "text": r["text"],
                "covers_json": dict(_parse_jsonb(r["covers_json"], {})),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
                "is_pinned": bool(r["is_pinned"]),
                "was_cited": bool(r["was_cited"]),
                "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
            }
            for r in rows
        ]

    async def save_local_summary_origin(self, origin_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.local_summary_origin (local_summary_id, origin_kind, origin_id, ordinal)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (local_summary_id, origin_kind, origin_id) DO UPDATE SET
                ordinal = EXCLUDED.ordinal
            """,
            _uuid.UUID(origin_json["local_summary_id"]),
            origin_json["origin_kind"],
            _uuid.UUID(origin_json["origin_id"]),
            int(origin_json.get("ordinal", 0)),
        )

    async def list_local_summary_origins(self, local_summary_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT local_summary_id, origin_kind, origin_id, ordinal
            FROM cima.local_summary_origin
            WHERE local_summary_id = $1
            ORDER BY ordinal ASC, origin_kind ASC
            """,
            _uuid.UUID(local_summary_id),
        )
        return [
            {
                "local_summary_id": str(r["local_summary_id"]),
                "origin_kind": r["origin_kind"],
                "origin_id": str(r["origin_id"]),
                "ordinal": int(r["ordinal"]),
            }
            for r in rows
        ]

    async def delete_local_summary_origins(self, local_summary_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM cima.local_summary_origin WHERE local_summary_id = $1",
            _uuid.UUID(local_summary_id),
        )

    async def update_local_summary_vector_state(
        self,
        local_summary_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE cima.local_summary
            SET vector_state = $2,
                embedding_model_id = COALESCE($3, embedding_model_id),
                embedding_schema_version = COALESCE($4, embedding_schema_version),
                updated_at = NOW()
            WHERE local_summary_id = $1
            """,
            _uuid.UUID(local_summary_id),
            vector_state,
            embedding_model_id,
            embedding_schema_version,
        )

    async def update_chunk_vector_state(
        self,
        chunk_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE cima.chunk_manifest
            SET vector_state = $2,
                embedding_model_id = COALESCE($3, embedding_model_id),
                embedding_schema_version = COALESCE($4, embedding_schema_version),
                expires_at = COALESCE($5::timestamptz, expires_at)
            WHERE chunk_id = $1
            """,
            _uuid.UUID(chunk_id),
            vector_state,
            embedding_model_id,
            embedding_schema_version,
            expires_at,
        )

    async def update_local_citem_vector_state(
        self,
        local_citem_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE cima.local_citem
            SET vector_state = $2,
                embedding_model_id = COALESCE($3, embedding_model_id),
                embedding_schema_version = COALESCE($4, embedding_schema_version),
                expires_at = COALESCE($5::timestamptz, expires_at),
                updated_at = NOW()
            WHERE local_citem_id = $1
            """,
            _uuid.UUID(local_citem_id),
            vector_state,
            embedding_model_id,
            embedding_schema_version,
            expires_at,
        )

    async def save_global_citem_record(self, citem_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.global_citem (
                global_citem_id, semantic_identity_id, origin_conversation_id, promotion_origin_local_citem_id,
                type, text, embedding_text, meta_json, provenance_json, validity, salience,
                created_at, updated_at, vector_state, embedding_model_id, embedding_schema_version,
                expires_at, is_pinned, was_cited, last_used_at
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11,
                COALESCE($12::timestamptz, NOW()), COALESCE($13::timestamptz, NOW()), $14, $15, $16,
                $17::timestamptz, $18, $19, $20::timestamptz
            )
            ON CONFLICT (global_citem_id) DO UPDATE SET
                semantic_identity_id = EXCLUDED.semantic_identity_id,
                origin_conversation_id = EXCLUDED.origin_conversation_id,
                promotion_origin_local_citem_id = EXCLUDED.promotion_origin_local_citem_id,
                type = EXCLUDED.type,
                text = EXCLUDED.text,
                embedding_text = EXCLUDED.embedding_text,
                meta_json = EXCLUDED.meta_json,
                provenance_json = EXCLUDED.provenance_json,
                validity = EXCLUDED.validity,
                salience = EXCLUDED.salience,
                updated_at = COALESCE(EXCLUDED.updated_at, NOW()),
                vector_state = EXCLUDED.vector_state,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                expires_at = EXCLUDED.expires_at,
                is_pinned = EXCLUDED.is_pinned,
                was_cited = EXCLUDED.was_cited,
                last_used_at = EXCLUDED.last_used_at
            """,
            _uuid.UUID(citem_json["global_citem_id"]),
            _uuid.UUID(citem_json["semantic_identity_id"]),
            _uuid.UUID(citem_json["origin_conversation_id"]),
            _uuid.UUID(citem_json["promotion_origin_local_citem_id"]),
            citem_json["type"],
            citem_json["text"],
            citem_json["embedding_text"],
            json.dumps(citem_json.get("meta_json", {}), ensure_ascii=False),
            json.dumps(citem_json.get("provenance_json", {}), ensure_ascii=False),
            citem_json.get("validity", "unknown"),
            float(citem_json.get("salience", 0.0) or 0.0),
            _parse_ts(citem_json.get("created_at")),
            _parse_ts(citem_json.get("updated_at")),
            citem_json.get("vector_state", "NONE"),
            citem_json.get("embedding_model_id"),
            int(citem_json["embedding_schema_version"]) if citem_json.get("embedding_schema_version") is not None else None,
            _parse_ts(citem_json.get("expires_at")),
            bool(citem_json.get("is_pinned", False)),
            bool(citem_json.get("was_cited", False)),
            _parse_ts(citem_json.get("last_used_at")),
        )

    async def list_global_citem_records(
        self,
        *,
        global_citem_ids: list[str] | None = None,
        semantic_identity_ids: list[str] | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        if global_citem_ids:
            clauses.append(f"global_citem_id = ANY(${len(params) + 1}::uuid[])")
            params.append([_uuid.UUID(v) for v in global_citem_ids])
        if semantic_identity_ids:
            clauses.append(f"semantic_identity_id = ANY(${len(params) + 1}::uuid[])")
            params.append([_uuid.UUID(v) for v in semantic_identity_ids])
        if origin_conversation_id is not None:
            clauses.append(f"origin_conversation_id = ${len(params) + 1}::uuid")
            params.append(_uuid.UUID(origin_conversation_id))
        rows = await self._pool.fetch(
            f"""
            SELECT global_citem_id, semantic_identity_id, origin_conversation_id, promotion_origin_local_citem_id,
                   type, text, embedding_text, meta_json, provenance_json, validity, salience,
                   created_at, updated_at, vector_state, embedding_model_id, embedding_schema_version,
                   expires_at, is_pinned, was_cited, last_used_at
            FROM cima.global_citem
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC, global_citem_id ASC
            """,
            *params,
        )
        return [
            {
                "global_citem_id": str(r["global_citem_id"]),
                "semantic_identity_id": str(r["semantic_identity_id"]),
                "origin_conversation_id": str(r["origin_conversation_id"]),
                "promotion_origin_local_citem_id": str(r["promotion_origin_local_citem_id"]),
                "type": r["type"],
                "text": r["text"],
                "embedding_text": r["embedding_text"],
                "meta_json": dict(_parse_jsonb(r["meta_json"], {})),
                "provenance_json": dict(_parse_jsonb(r["provenance_json"], {})),
                "validity": r["validity"],
                "salience": float(r["salience"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
                "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
                "is_pinned": bool(r["is_pinned"]),
                "was_cited": bool(r["was_cited"]),
                "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
            }
            for r in rows
        ]

    async def save_global_citem_evidence(self, evidence_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.global_citem_evidence (
                global_citem_id, ordinal, evidence_kind, source_text_snapshot, locator_json
            ) VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (global_citem_id, ordinal) DO UPDATE SET
                evidence_kind = EXCLUDED.evidence_kind,
                source_text_snapshot = EXCLUDED.source_text_snapshot,
                locator_json = EXCLUDED.locator_json
            """,
            _uuid.UUID(evidence_json["global_citem_id"]),
            int(evidence_json.get("ordinal", 0)),
            evidence_json["evidence_kind"],
            evidence_json.get("source_text_snapshot"),
            json.dumps(evidence_json.get("locator_json", {}), ensure_ascii=False),
        )

    async def list_global_citem_evidence(self, global_citem_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT global_citem_id, ordinal, evidence_kind, source_text_snapshot, locator_json
            FROM cima.global_citem_evidence
            WHERE global_citem_id = $1
            ORDER BY ordinal ASC
            """,
            _uuid.UUID(global_citem_id),
        )
        return [
            {
                "global_citem_id": str(r["global_citem_id"]),
                "ordinal": int(r["ordinal"]),
                "evidence_kind": r["evidence_kind"],
                "source_text_snapshot": r["source_text_snapshot"],
                "locator_json": dict(_parse_jsonb(r["locator_json"], {})),
            }
            for r in rows
        ]

    async def save_global_summary_record(self, summary_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.global_summary (
                global_summary_id, level, cluster_id, text, covers_json,
                created_at, updated_at, vector_state, embedding_model_id, embedding_schema_version
            ) VALUES (
                $1, $2, $3, $4, $5::jsonb,
                COALESCE($6::timestamptz, NOW()), COALESCE($7::timestamptz, NOW()), $8, $9, $10
            )
            ON CONFLICT (global_summary_id) DO UPDATE SET
                level = EXCLUDED.level,
                cluster_id = EXCLUDED.cluster_id,
                text = EXCLUDED.text,
                covers_json = EXCLUDED.covers_json,
                updated_at = COALESCE(EXCLUDED.updated_at, NOW()),
                vector_state = EXCLUDED.vector_state,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_schema_version = EXCLUDED.embedding_schema_version
            """,
            _uuid.UUID(summary_json["global_summary_id"]),
            summary_json["level"],
            summary_json.get("cluster_id"),
            summary_json["text"],
            json.dumps(summary_json.get("covers_json", {}), ensure_ascii=False),
            _parse_ts(summary_json.get("created_at")),
            _parse_ts(summary_json.get("updated_at")),
            summary_json.get("vector_state", "NONE"),
            summary_json.get("embedding_model_id"),
            int(summary_json["embedding_schema_version"]) if summary_json.get("embedding_schema_version") is not None else None,
        )

    async def list_global_summary_records(
        self,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        join = ""
        if summary_ids:
            clauses.append(f"gs.global_summary_id = ANY(${len(params) + 1}::uuid[])")
            params.append([_uuid.UUID(v) for v in summary_ids])
        if level is not None:
            clauses.append(f"gs.level = ${len(params) + 1}")
            params.append(level)
        if origin_conversation_id is not None:
            join = " JOIN cima.global_summary_origin gso ON gso.global_summary_id = gs.global_summary_id JOIN cima.global_citem gc ON gc.global_citem_id = gso.origin_id AND gso.origin_kind = 'global_citem' "
            clauses.append(f"gc.origin_conversation_id = ${len(params) + 1}::uuid")
            params.append(_uuid.UUID(origin_conversation_id))
        rows = await self._pool.fetch(
            f"""
            SELECT DISTINCT gs.global_summary_id, gs.level, gs.cluster_id, gs.text, gs.covers_json,
                            gs.created_at, gs.updated_at, gs.vector_state, gs.embedding_model_id, gs.embedding_schema_version
            FROM cima.global_summary gs
            {join}
            WHERE {' AND '.join(clauses)}
            ORDER BY gs.updated_at DESC, gs.global_summary_id ASC
            """,
            *params,
        )
        return [
            {
                "global_summary_id": str(r["global_summary_id"]),
                "level": r["level"],
                "cluster_id": r["cluster_id"],
                "text": r["text"],
                "covers_json": dict(_parse_jsonb(r["covers_json"], {})),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "vector_state": r["vector_state"],
                "embedding_model_id": r["embedding_model_id"],
                "embedding_schema_version": int(r["embedding_schema_version"]) if r["embedding_schema_version"] is not None else None,
            }
            for r in rows
        ]

    async def save_global_summary_origin(self, origin_json: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            INSERT INTO cima.global_summary_origin (global_summary_id, origin_kind, origin_id, ordinal)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (global_summary_id, origin_kind, origin_id) DO UPDATE SET
                ordinal = EXCLUDED.ordinal
            """,
            _uuid.UUID(origin_json["global_summary_id"]),
            origin_json["origin_kind"],
            _uuid.UUID(origin_json["origin_id"]),
            int(origin_json.get("ordinal", 0)),
        )

    async def list_global_summary_origins(self, global_summary_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT global_summary_id, origin_kind, origin_id, ordinal
            FROM cima.global_summary_origin
            WHERE global_summary_id = $1
            ORDER BY ordinal ASC, origin_kind ASC
            """,
            _uuid.UUID(global_summary_id),
        )
        return [
            {
                "global_summary_id": str(r["global_summary_id"]),
                "origin_kind": r["origin_kind"],
                "origin_id": str(r["origin_id"]),
                "ordinal": int(r["ordinal"]),
            }
            for r in rows
        ]

    async def delete_global_summary_origins(self, global_summary_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM cima.global_summary_origin WHERE global_summary_id = $1",
            _uuid.UUID(global_summary_id),
        )

    async def update_global_citem_vector_state(
        self,
        global_citem_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE cima.global_citem
            SET vector_state = $2,
                embedding_model_id = COALESCE($3, embedding_model_id),
                embedding_schema_version = COALESCE($4, embedding_schema_version),
                expires_at = COALESCE($5::timestamptz, expires_at),
                updated_at = NOW()
            WHERE global_citem_id = $1
            """,
            _uuid.UUID(global_citem_id),
            vector_state,
            embedding_model_id,
            embedding_schema_version,
            expires_at,
        )

    async def update_global_summary_vector_state(
        self,
        global_summary_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE cima.global_summary
            SET vector_state = $2,
                embedding_model_id = COALESCE($3, embedding_model_id),
                embedding_schema_version = COALESCE($4, embedding_schema_version),
                updated_at = NOW()
            WHERE global_summary_id = $1
            """,
            _uuid.UUID(global_summary_id),
            vector_state,
            embedding_model_id,
            embedding_schema_version,
        )

    # ── Health ────────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False


# ── Pool factory ──────────────────────────────────────────────────────────────

async def create_pool(dsn: str, min_size: int = 5, max_size: int = 20) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
