from __future__ import annotations

"""Async global-memory witness plane.

This tranche adds the first concrete global-memory path on top of the approved
witness backend:
- eligible local C-items schedule deterministic promotion events;
- promotion events materialise `global_citem` rows with preserved semantic
  identity and explicit evidence;
- promoted globals are indexed into the canonical global Qdrant collection;
- a small deterministic global summary is maintained per origin conversation and
  indexed into the canonical global summary collection.

The implementation stays deterministic and idempotent so it can run entirely in
workers without depending on request-path LLM calls.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cima_demo.domain.entities import CItem
from cima_demo.domain.operations import is_promotion_eligible
from cima_demo.domain.value_objects import PromotionPolicy
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger
from cima_demo.witness_backend.lifecycle_guard import complete_if_conversation_not_active
from cima_demo.witness_backend.events import (
    CItemCreatedData,
    CItemPromotedGlobalData,
    CloudEventEnvelope,
    EventType,
    Producer,
    VectorMeta,
    VectorUpsertedData,
)
from cima_demo.witness_backend.topic_catalog import TOPICS, conversation_key


class GlobalStoreLike(Protocol):
    async def append_outbox_event(
        self,
        *,
        topic: str,
        message_key: str,
        payload_json: dict[str, Any] | None,
        headers_json: dict[str, Any] | None = None,
    ) -> int: ...

    async def list_local_citem_records(
        self,
        conversation_id: str,
        *,
        citem_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def list_local_citem_evidence(self, local_citem_id: str) -> list[dict[str, Any]]: ...

    async def save_global_citem_record(self, citem_json: dict[str, Any]) -> None: ...

    async def list_global_citem_records(
        self,
        *,
        global_citem_ids: list[str] | None = None,
        semantic_identity_ids: list[str] | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def save_global_citem_evidence(self, evidence_json: dict[str, Any]) -> None: ...

    async def update_global_citem_vector_state(
        self,
        global_citem_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...

    async def save_global_summary_record(self, summary_json: dict[str, Any]) -> None: ...

    async def list_global_summary_records(
        self,
        *,
        summary_ids: list[str] | None = None,
        level: str | None = None,
        origin_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def save_global_summary_origin(self, origin_json: dict[str, Any]) -> None: ...

    async def delete_global_summary_origins(self, global_summary_id: str) -> None: ...

    async def update_global_summary_vector_state(
        self,
        global_summary_id: str,
        *,
        vector_state: str,
        embedding_model_id: str | None = None,
        embedding_schema_version: int | None = None,
        expires_at: str | None = None,
    ) -> None: ...


class TokenCounterLike(Protocol):
    def count_text_tokens_sync(self, text: str) -> int: ...


class EmbedderLike(Protocol):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


_TYPE_ORDER = [
    "DECISION",
    "CONSTRAINT",
    "DEFINITION",
    "PLAN_STEP",
    "RISK",
    "FACT",
    "HEDGED_FACT",
    "CONTEXT",
    "ATTRIBUTION",
    "EVALUATION",
    "QUESTION",
    "CODE_ARTIFACT",
]


@dataclass(frozen=True, slots=True)
class BuiltGlobalSummary:
    summary_id: str
    text: str
    token_count: int
    origin_global_citem_ids: list[str]


class DeterministicGlobalSummaryBuilder:
    def __init__(self, *, token_counter: TokenCounterLike) -> None:
        self._counter = token_counter

    def build(self, *, summary_id: str, origin_conversation_id: str, global_rows: list[dict[str, Any]]) -> BuiltGlobalSummary:
        ordered = sorted(
            global_rows,
            key=lambda row: (
                float(row.get("salience", 0.0) or 0.0),
                str(row.get("created_at") or ""),
                str(row.get("global_citem_id") or ""),
            ),
            reverse=True,
        )
        grouped: dict[str, list[str]] = {key: [] for key in _TYPE_ORDER}
        for row in ordered:
            item_type = str(row.get("type") or "FACT")
            grouped.setdefault(item_type, []).append(_one_line(row.get("text") or ""))
        lines = [f"Global memory summary for {origin_conversation_id}"]
        for item_type in _TYPE_ORDER:
            bucket = [text for text in grouped.get(item_type, []) if text]
            if not bucket:
                continue
            lines.append(f"{item_type}: " + " | ".join(bucket[:3]))
        text = "\n".join(lines).strip()
        return BuiltGlobalSummary(
            summary_id=summary_id,
            text=text,
            token_count=max(1, self._counter.count_text_tokens_sync(text)),
            origin_global_citem_ids=[str(row["global_citem_id"]) for row in ordered if row.get("global_citem_id")],
        )


def _one_line(text: str) -> str:
    compact = " ".join(str(text).strip().split())
    if len(compact) <= 180:
        return compact
    return compact[:177].rstrip() + "..."


def _deterministic_event_id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts))


class GlobalPromotionConsumer:
    def __init__(
        self,
        *,
        db: GlobalStoreLike,
        ledger: ConsumerEffectLedger,
        tokenizer: TokenCounterLike,
        embedder: EmbedderLike,
        qdrant_plane: QdrantWitnessPlane,
        embedding_model_id: str,
        embedding_schema_version: int,
        producer: Producer = Producer.CIMA_WORKER,
        promotion_policy: PromotionPolicy | None = None,
    ) -> None:
        self._db = db
        self._ledger = ledger
        self._tokenizer = tokenizer
        self._embedder = embedder
        self._plane = qdrant_plane
        self._embedding_model_id = embedding_model_id
        self._embedding_schema_version = embedding_schema_version
        self._producer = producer
        self._policy = promotion_policy or PromotionPolicy(min_references=1, min_importance=0.85)
        self._summary_builder = DeterministicGlobalSummaryBuilder(token_counter=tokenizer)

    async def handle(self, payload_json: dict[str, Any]) -> None:
        envelope = CloudEventEnvelope.model_validate(payload_json)
        if envelope.type == EventType.MEMORY_CITEM_CREATED:
            await self._handle_citem_created(envelope)
            return
        if envelope.type == EventType.MEMORY_CITEM_PROMOTED_GLOBAL:
            await self._handle_citem_promoted_global(envelope)
            return

    async def _handle_citem_created(self, envelope: CloudEventEnvelope) -> None:
        data = CItemCreatedData.model_validate(envelope.data)
        effect_key = f"global-promotion-schedule:{','.join(sorted(str(v) for v in data.citem_ids))}"
        key = ConsumerEffectKey("global-promotion-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        rows = await self._db.list_local_citem_records(envelope.subject, citem_ids=[str(v) for v in data.citem_ids])
        scheduled = 0
        for row in rows:
            local_id = str(row.get("local_citem_id"))
            evidence = await self._db.list_local_citem_evidence(local_id)
            item = CItem(
                citem_id=local_id,
                conversation_id=envelope.subject,
                content=str(row.get("text") or ""),
                item_type=str(row.get("type") or "FACT"),
                scope="episodic",
                scope_status="active",
                importance=float(row.get("salience", 0.0) or 0.0),
                confidence=1.0,
                validation_label=str(row.get("validity") or "unknown"),
                token_count=max(1, self._tokenizer.count_text_tokens_sync(str(row.get("text") or ""))),
            )
            if not is_promotion_eligible(item, max(1, len(evidence)), self._policy):
                continue
            semantic_identity_id = str(row.get("semantic_identity_id"))
            existing = await self._db.list_global_citem_records(semantic_identity_ids=[semantic_identity_id])
            global_citem_id = existing[0]["global_citem_id"] if existing else str(uuid.uuid5(uuid.NAMESPACE_URL, f"global-citem|{semantic_identity_id}"))
            event = CloudEventEnvelope(
                id=_deterministic_event_id(envelope.subject, EventType.MEMORY_CITEM_PROMOTED_GLOBAL, local_id, global_citem_id),
                type=EventType.MEMORY_CITEM_PROMOTED_GLOBAL,
                source=self._producer,
                subject=envelope.subject,
                dataschema="schemas/cima.memory.citem.promoted_global.v1.json",
                data=CItemPromotedGlobalData(
                    local_citem_id=uuid.UUID(local_id),
                    global_citem_id=uuid.UUID(global_citem_id),
                    semantic_identity_id=uuid.UUID(semantic_identity_id),
                    origin_conversation_id=envelope.subject,
                ).model_dump(mode="json"),
            )
            await self._db.append_outbox_event(
                topic=TOPICS.memory_events,
                message_key=conversation_key(envelope.subject),
                payload_json=event.model_dump(mode="json"),
            )
            scheduled += 1
        await self._ledger.complete(key, details_json={"status": "scheduled", "count": scheduled})

    async def _handle_citem_promoted_global(self, envelope: CloudEventEnvelope) -> None:
        data = CItemPromotedGlobalData.model_validate(envelope.data)
        effect_key = f"global-promote:{data.global_citem_id}:index-v{self._embedding_schema_version}"
        key = ConsumerEffectKey("global-promotion-consumer", str(envelope.id), effect_key)
        if not await self._ledger.begin(key):
            return
        if await complete_if_conversation_not_active(store=self._db, ledger=self._ledger, key=key, conversation_id=envelope.subject):
            return
        local_rows = await self._db.list_local_citem_records(envelope.subject, citem_ids=[str(data.local_citem_id)])
        if not local_rows:
            await self._ledger.complete(key, details_json={"status": "missing_local"})
            return
        local = local_rows[0]
        existing = await self._db.list_global_citem_records(global_citem_ids=[str(data.global_citem_id)])
        global_row = existing[0] if existing else None
        if global_row is None:
            global_row = {
                "global_citem_id": str(data.global_citem_id),
                "semantic_identity_id": str(data.semantic_identity_id),
                "origin_conversation_id": envelope.subject,
                "promotion_origin_local_citem_id": str(data.local_citem_id),
                "type": local["type"],
                "text": local["text"],
                "embedding_text": local.get("embedding_text") or local.get("text") or "",
                "meta_json": dict(local.get("meta_json") or {}),
                "provenance_json": dict(local.get("provenance_json") or {}),
                "validity": local.get("validity") or "unknown",
                "salience": float(local.get("salience", 0.0) or 0.0),
                "created_at": local.get("created_at") or datetime.now(UTC).isoformat(),
                "updated_at": local.get("updated_at") or datetime.now(UTC).isoformat(),
                "vector_state": "NONE",
                "embedding_model_id": None,
                "embedding_schema_version": None,
                "expires_at": None,
                "is_pinned": False,
                "was_cited": False,
                "last_used_at": None,
            }
            await self._db.save_global_citem_record(global_row)
            local_evidence = await self._db.list_local_citem_evidence(str(data.local_citem_id))
            for ordinal, row in enumerate(local_evidence):
                evidence_kind = "chunk_snippet" if row.get("chunk_id") else "source_snippet"
                await self._db.save_global_citem_evidence(
                    {
                        "global_citem_id": str(data.global_citem_id),
                        "ordinal": ordinal,
                        "evidence_kind": evidence_kind,
                        "source_text_snapshot": None,
                        "locator_json": {
                            "source_id": row.get("source_id"),
                            "chunk_id": row.get("chunk_id"),
                            "edu_id": row.get("edu_id"),
                            **dict(row.get("locator_json") or {}),
                        },
                    }
                )
        vector = (await self._embedder.embed_batch([str(global_row.get("embedding_text") or global_row.get("text") or "")]))[0]
        await self._plane.upsert_points(
            collection_name=self._plane.catalog.global_citems,
            points=[
                {
                    "id": str(data.global_citem_id),
                    "payload": {
                        "conversation_id": envelope.subject,
                        "scope": "global",
                        "scope_status": "active",
                        "kind": "citem",
                        "w_scope": "global",
                        "ref_id": str(data.global_citem_id),
                        "citem_id": str(data.global_citem_id),
                        "type": global_row["type"],
                        "item_type": global_row["type"],
                        "content": global_row["text"],
                        "created_at": global_row["created_at"],
                        "created_at_unix": _to_unix(global_row.get("created_at")),
                        "importance": float(global_row.get("salience", 0.0) or 0.0),
                        "confidence": 1.0,
                        "validation_label": global_row.get("validity") or "unknown",
                        "conflict_status": "none",
                        "phase_ingested": "IDLE",
                        "actor": (global_row.get("meta_json") or {}).get("speaker") or "agent",
                        "motivation": (global_row.get("meta_json") or {}).get("source_kind"),
                        "dependency_ids": [],
                        "token_count": max(1, self._tokenizer.count_text_tokens_sync(str(global_row.get("text") or ""))),
                        "semantic_identity_id": str(data.semantic_identity_id),
                    },
                    "vector": vector,
                }
            ],
        )
        await self._db.update_global_citem_vector_state(
            str(data.global_citem_id),
            vector_state="INDEXED",
            embedding_model_id=self._embedding_model_id,
            embedding_schema_version=self._embedding_schema_version,
        )
        await self._emit_vector_upserted(
            conversation_id=envelope.subject,
            ref_kind="global_citem",
            ref_id=str(data.global_citem_id),
            collection=self._plane.catalog.global_citems,
            scope="global",
            item_type=global_row["type"],
        )
        await self._refresh_global_summary(origin_conversation_id=envelope.subject)
        await self._ledger.complete(key, details_json={"status": "promoted", "global_citem_id": str(data.global_citem_id)})

    async def _refresh_global_summary(self, *, origin_conversation_id: str) -> None:
        global_rows = await self._db.list_global_citem_records(origin_conversation_id=origin_conversation_id)
        if not global_rows:
            return
        summary_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"global-summary|{origin_conversation_id}"))
        built = self._summary_builder.build(
            summary_id=summary_id,
            origin_conversation_id=origin_conversation_id,
            global_rows=global_rows,
        )
        summary_record = {
            "global_summary_id": summary_id,
            "level": "MASTER",
            "cluster_id": None,
            "text": built.text,
            "covers_json": {"origin_global_citem_ids": built.origin_global_citem_ids},
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "vector_state": "NONE",
            "embedding_model_id": None,
            "embedding_schema_version": None,
        }
        await self._db.save_global_summary_record(summary_record)
        await self._db.delete_global_summary_origins(summary_id)
        for ordinal, origin_id in enumerate(built.origin_global_citem_ids):
            await self._db.save_global_summary_origin(
                {
                    "global_summary_id": summary_id,
                    "origin_kind": "global_citem",
                    "origin_id": origin_id,
                    "ordinal": ordinal,
                }
            )
        vector = (await self._embedder.embed_batch([built.text]))[0]
        await self._plane.upsert_points(
            collection_name=self._plane.catalog.global_summaries,
            points=[
                {
                    "id": summary_id,
                    "payload": {
                        "conversation_id": origin_conversation_id,
                        "scope": "global",
                        "scope_status": "active",
                        "kind": "summary",
                        "w_scope": "global",
                        "ref_id": summary_id,
                        "citem_id": summary_id,
                        "type": "MASTER",
                        "item_type": "OBSERVATION",
                        "content": built.text,
                        "created_at": summary_record["created_at"],
                        "created_at_unix": _to_unix(summary_record["created_at"]),
                        "importance": 0.7,
                        "confidence": 1.0,
                        "validation_label": "accepted",
                        "conflict_status": "none",
                        "phase_ingested": "IDLE",
                        "actor": "agent",
                        "motivation": "summary:global_master",
                        "dependency_ids": built.origin_global_citem_ids,
                        "token_count": built.token_count,
                    },
                    "vector": vector,
                }
            ],
        )
        await self._db.update_global_summary_vector_state(
            summary_id,
            vector_state="INDEXED",
            embedding_model_id=self._embedding_model_id,
            embedding_schema_version=self._embedding_schema_version,
        )
        await self._emit_vector_upserted(
            conversation_id=origin_conversation_id,
            ref_kind="global_summary",
            ref_id=summary_id,
            collection=self._plane.catalog.global_summaries,
            scope="global",
            item_type="MASTER",
        )

    async def _emit_vector_upserted(
        self,
        *,
        conversation_id: str,
        ref_kind: str,
        ref_id: str,
        collection: str,
        scope: str,
        item_type: str | None,
    ) -> None:
        event_id = _deterministic_event_id(conversation_id, EventType.VECTOR_UPSERTED, ref_kind, ref_id, self._embedding_model_id, str(self._embedding_schema_version))
        payload = VectorUpsertedData(
            ref_kind=ref_kind,  # type: ignore[arg-type]
            ref_id=uuid.UUID(ref_id),
            qdrant_collection=collection,
            vector_state="INDEXED",
            embedding_model_id=self._embedding_model_id,
            embedding_schema_version=self._embedding_schema_version,
            eligible_for_geometry=False,
            meta=VectorMeta(scope=scope, type=item_type),
        )
        envelope = CloudEventEnvelope(
            id=event_id,
            type=EventType.VECTOR_UPSERTED,
            source=self._producer,
            subject=conversation_id,
            dataschema="schemas/cima.vector.upserted.v1.json",
            data=payload.model_dump(mode="json"),
        )
        await self._db.append_outbox_event(
            topic=TOPICS.vector_events,
            message_key=conversation_key(conversation_id),
            payload_json=envelope.model_dump(mode="json"),
        )


def _to_unix(value: str | None) -> float:
    if not value:
        return datetime.now(UTC).timestamp()
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return datetime.now(UTC).timestamp()
