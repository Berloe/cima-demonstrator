from __future__ import annotations

"""Runtime-side EPHEMERAL mirroring for selected context items.

R2.3 / R2.4 need EPHEMERAL to exist in the live runtime/retrieval path, not
only as a persistent GC lane. This helper mirrors recently served context items
into the dedicated ephemeral Qdrant collection, refreshes their witness-plane
lease rows and keeps them explicitly ineligible for geometry.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from cima_demo.domain.ports import EmbeddingPort
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.witness_backend.ephemeral import EphemeralVectorRegistry


_SUMMARY_KINDS = {"summary", "local_summary", "global_summary"}


@dataclass(frozen=True, slots=True)
class EphemeralMirrorStats:
    accepted_items: int
    upserted_points: int
    registered_leases: int
    expires_at: str | None




class EphemeralConversationReader(Protocol):
    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None: ...

class EphemeralRuntimeMirrorPort(Protocol):
    async def mirror_context_items(
        self,
        *,
        conversation_id: str,
        items: list[dict[str, Any]],
    ) -> EphemeralMirrorStats: ...


class EphemeralRuntimeMirror:
    """Project selected runtime items into the EPHEMERAL lane.

    The mirror is intentionally conservative:
    - only selected runtime items are mirrored,
    - payloads remain geometry-ineligible,
    - IDs are deterministic per logical item so repeated turns refresh the same
      lease instead of creating unbounded duplicates,
    - the mirror is best-effort and side-effect free for empty selections.
    """

    def __init__(
        self,
        *,
        plane: QdrantWitnessPlane,
        embedder: EmbeddingPort,
        registry: EphemeralVectorRegistry,
        ttl_seconds: int = 900,
        max_items: int = 12,
        embedding_model_id: str = "tei",
        embedding_schema_version: int = 1,
        conversation_reader: EphemeralConversationReader | None = None,
    ) -> None:
        self._plane = plane
        self._embed = embedder
        self._registry = registry
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._max_items = max(1, int(max_items))
        self._embedding_model_id = embedding_model_id
        self._embedding_schema_version = int(embedding_schema_version)
        self._conversation_reader = conversation_reader

    async def mirror_context_items(
        self,
        *,
        conversation_id: str,
        items: list[dict[str, Any]],
    ) -> EphemeralMirrorStats:
        if not await self._conversation_is_active(conversation_id):
            return EphemeralMirrorStats(accepted_items=0, upserted_points=0, registered_leases=0, expires_at=None)
        now = datetime.now(UTC)
        prepared: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in items:
            item = dict(raw)
            logical_id = str(item.get("ref_id") or "").strip()
            if not logical_id:
                continue
            raw_kind = str(item.get("ref_kind") or "citem").strip() or "citem"
            logical_kind = self._canonical_ref_kind(item)
            dedupe_key = (logical_kind, logical_id)
            if dedupe_key in seen:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            seen.add(dedupe_key)
            w_scope = self._logical_scope(item)
            scope = "global" if w_scope == "global" else "episodic"
            item_type = str(item.get("item_type") or ("SUMMARY" if raw_kind in _SUMMARY_KINDS else "OBSERVATION"))
            ephemeral_id = self._stable_ephemeral_id(
                conversation_id=conversation_id,
                logical_kind=logical_kind,
                logical_id=logical_id,
                content=content,
            )
            prepared.append(
                {
                    "ephemeral_id": ephemeral_id,
                    "logical_ref_kind": logical_kind,
                    "logical_ref_id": logical_id,
                    "payload": {
                        "citem_id": logical_id,
                        "conversation_id": conversation_id,
                        "content": content,
                        "item_type": item_type,
                        "scope": scope,
                        "scope_status": "active",
                        "importance": float(item.get("importance") or 0.6),
                        "confidence": 1.0,
                        "validation_label": str(item.get("validation_label") or "accepted"),
                        "conflict_status": str(item.get("conflict_status") or "none"),
                        "phase_ingested": str(item.get("phase_ingested") or "IDLE"),
                        "actor": str(item.get("actor") or "agent"),
                        "motivation": f"ephemeral:{logical_kind}",
                        "created_at_unix": now.timestamp(),
                        "token_count": max(1, len(content.split())),
                        "dependency_ids": list(item.get("dependency_ids") or []),
                        "kind": "ephemeral",
                        "w_scope": w_scope,
                        "ref_id": logical_id,
                        "origin_ref_kind": logical_kind,
                        "origin_ref_id": logical_id,
                        "vector_state": "EPHEMERAL",
                        "eligible_for_geometry": False,
                    },
                    "text": content,
                    "scope": "global" if w_scope == "global" else "local",
                    "item_type": item_type,
                }
            )
            if len(prepared) >= self._max_items:
                break

        if not prepared:
            return EphemeralMirrorStats(accepted_items=0, upserted_points=0, registered_leases=0, expires_at=None)

        vectors = await self._embed.embed_batch([row["text"] for row in prepared])
        expires_at: str | None = None
        await self._plane.upsert_points(
            collection_name=self._plane.catalog.ephemeral,
            points=[
                {"id": row["ephemeral_id"], "payload": row["payload"], "vector": vector}
                for row, vector in zip(prepared, vectors, strict=False)
            ],
        )
        leases = 0
        for row in prepared:
            lease = await self._registry.register(
                conversation_id=conversation_id,
                origin_ref_kind=row["logical_ref_kind"],
                origin_ref_id=row["logical_ref_id"],
                qdrant_collection=self._plane.catalog.ephemeral,
                embedding_model_id=self._embedding_model_id,
                embedding_schema_version=self._embedding_schema_version,
                ttl_seconds=self._ttl_seconds,
                scope=row["scope"],
                item_type=row["item_type"],
                meta_json={
                    "logical_ref_kind": row["logical_ref_kind"],
                    "logical_ref_id": row["logical_ref_id"],
                    "runtime_origin": "context_view",
                },
                ephemeral_id=row["ephemeral_id"],
                now=now,
            )
            expires_at = lease.expires_at
            leases += 1
        return EphemeralMirrorStats(
            accepted_items=len(prepared),
            upserted_points=len(prepared),
            registered_leases=leases,
            expires_at=expires_at,
        )

    def _canonical_ref_kind(self, item: dict[str, Any]) -> str:
        raw_kind = str(item.get("ref_kind") or "citem").strip() or "citem"
        if raw_kind == "summary":
            return "global_summary" if self._logical_scope(item) == "global" else "local_summary"
        if raw_kind == "citem":
            return "global_citem" if self._logical_scope(item) == "global" else "local_citem"
        return raw_kind

    @staticmethod
    def _logical_scope(item: dict[str, Any]) -> str:
        for key in ("item_resolution_scope", "summary_scope", "citem_resolution_scope", "resolution_scope"):
            value = str(item.get(key) or "").strip().lower()
            if value == "global":
                return "global"
            if value in {"local", "legacy", ""}:
                continue
        return "local"

    async def _conversation_is_active(self, conversation_id: str) -> bool:
        if self._conversation_reader is None:
            return True
        row = await self._conversation_reader.get_conversation(conversation_id)
        if row is None:
            return False
        return str(row.get("status") or "ACTIVE").upper() == "ACTIVE"

    @staticmethod
    def _stable_ephemeral_id(
        *,
        conversation_id: str,
        logical_kind: str,
        logical_id: str,
        content: str,
    ) -> str:
        digest = sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
        token = f"{conversation_id}:{logical_kind}:{logical_id}:{digest}"
        return str(uuid5(NAMESPACE_URL, token))
