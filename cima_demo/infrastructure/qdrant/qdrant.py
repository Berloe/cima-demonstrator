"""QdrantCItemAdapter → CItemStorePort (KIMA_Infrastructure_Layer_v0.6 §3.2)."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - exercised indirectly in environments with qdrant-client installed
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        Fusion,
        FusionQuery,
        HasIdCondition,
        MatchAny,
        MatchValue,
        PointStruct,
        Prefetch,
        SparseVector,
    )
except ModuleNotFoundError:  # pragma: no cover - lightweight test fallback
    from types import SimpleNamespace
    from typing import Any as AsyncQdrantClient

    class MatchValue:
        def __init__(self, *, value: object) -> None:
            self.value = value

    class MatchAny:
        def __init__(self, *, any: list[object]) -> None:
            self.any = any

    class HasIdCondition:
        def __init__(self, *, has_id: list[str]) -> None:
            self.has_id = has_id

    class FieldCondition:
        def __init__(self, *, key: str, match: object | None = None, range: object | None = None) -> None:
            self.key = key
            self.match = match
            self.range = range

    class Filter:
        def __init__(self, *, must: list[object] | None = None, must_not: list[object] | None = None) -> None:
            self.must = must or []
            self.must_not = must_not or []

    class SparseVector:
        def __init__(self, *, indices: list[int], values: list[float]) -> None:
            self.indices = indices
            self.values = values

    class Prefetch:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class Fusion:
        RRF = "rrf"

    class FusionQuery:
        def __init__(self, *, fusion: object) -> None:
            self.fusion = fusion

    class PointStruct:
        def __init__(self, *, id: str, payload: dict[str, object], vector: dict[str, object]) -> None:
            self.id = id
            self.payload = payload
            self.vector = vector

from cima_demo.domain.entities import CItem
from cima_demo.domain.errors import CItemNotFoundError, CItemStoreError, GeometricExpansionError
from cima_demo.domain.ports import CItemStorePort, EmbeddingPort, RelDBPort, SparseEmbeddingPort
from cima_demo.domain.value_objects import CItemFilter, RecallSource, ScoredCItem
from cima_demo.infrastructure.qdrant.setup import ensure_collection

log = logging.getLogger(__name__)


class QdrantCItemAdapter(CItemStorePort):
    """Qdrant-backed C-Item store.

    The demonstrator still exposes the legacy CItemStorePort interface, but the
    underlying physical layout now honours the witness-backend split between
    conversation-scoped and global knowledge. The adapter keeps that split
    internal so existing callers do not need to know collection names.
    """

    _BM25_MODEL = "Qdrant/bm25"

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedding_port: EmbeddingPort,
        collection: str = "citems",
        sparse_embedding_port: SparseEmbeddingPort | None = None,
        dense_dim: int = 768,
        global_collection: str | None = None,
        ephemeral_collection: str | None = None,
        rel_db: RelDBPort | None = None,
    ) -> None:
        self._client = client
        self._embed = embedding_port
        self._collection = collection
        self._global_collection = global_collection or collection
        self._ephemeral_collection = ephemeral_collection or collection
        self._sparse_port = sparse_embedding_port
        self._dense_dim = dense_dim
        self._db = rel_db
        self._bm25: Any = None  # fastembed SparseTextEmbedding (lazy-loaded)

    def _collection_for_scope(self, scope: str, *, ephemeral: bool = False) -> str:
        if ephemeral:
            return self._ephemeral_collection
        return self._global_collection if scope == "global" else self._collection

    async def _retrieve_many(self, collection: str, citem_ids: list[str]) -> list[Any]:
        return await self._client.retrieve(
            collection_name=collection,
            ids=citem_ids,
            with_payload=True,
            with_vectors=False,
        )

    def _search_provenance(self, scope: str | None) -> RecallSource:
        return RecallSource.HYBRID_EPISODIC if scope == "episodic" else RecallSource.HYBRID_GLOBAL

    # ── Sparse encoding ───────────────────────────────────────────────────────

    def _get_bm25(self) -> Any:
        if self._bm25 is None:
            try:
                from fastembed import SparseTextEmbedding
            except ModuleNotFoundError:
                self._bm25 = False
                return self._bm25
            self._bm25 = SparseTextEmbedding(model_name=self._BM25_MODEL)
        return self._bm25

    def _encode_sparse_sync(self, text: str) -> dict[int, float]:
        bm25 = self._get_bm25()
        if not bm25:
            return {}
        result = next(iter(bm25.embed([text])))
        return dict(zip(result.indices.tolist(), result.values.tolist(), strict=False))

    async def _encode_sparse(self, text: str) -> dict[int, float]:
        if self._sparse_port is not None:
            return await self._sparse_port.embed_sparse(text)
        return await asyncio.to_thread(self._encode_sparse_sync, text)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value)
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
            except ValueError:
                pass
        return datetime.now(UTC)

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        parts = str(text or "").split()
        return max(1, len(parts)) if parts else 1

    @staticmethod
    def _annotate_resolution(citem: CItem, *, mode: str, scope: str | None = None) -> CItem:
        setattr(citem, "citem_resolution_mode", mode)
        if scope is not None:
            setattr(citem, "citem_resolution_scope", scope)
        return citem

    def _witness_row_to_citem(self, row: dict[str, Any], *, scope: str) -> CItem:
        meta = dict(row.get("meta_json") or {})
        provenance = dict(row.get("provenance_json") or {})
        row_id = str(
            row.get("local_citem_id")
            or row.get("global_citem_id")
            or row.get("citem_id")
            or ""
        )
        content = str(row.get("text") or "")
        dependency_ids = [str(v) for v in provenance.get("dependency_ids", []) if v]
        actor = str(meta.get("speaker") or meta.get("actor") or "agent")
        return self._annotate_resolution(
            CItem(
                citem_id=row_id,
                conversation_id=str(row.get("conversation_id") or row.get("origin_conversation_id") or ""),
                content=content,
                item_type=str(row.get("type") or "FACT"),
                scope=scope,
                scope_status="active",
                importance=float(row.get("salience", 0.5) or 0.5),
                confidence=float(meta.get("confidence", 1.0) or 1.0),
                validation_label=str(row.get("validity") or meta.get("validation_label") or "unknown"),
                conflict_status=str(meta.get("conflict_status") or "none"),
                phase_ingested=str(meta.get("phase_ingested") or "IDLE"),
                actor=actor,
                motivation=meta.get("source_kind") or meta.get("motivation"),
                created_at=self._coerce_datetime(row.get("created_at") or row.get("updated_at")),
                dependency_ids=dependency_ids,
                token_count=int(meta.get("token_count") or self._estimate_token_count(content)),
                content_hash=meta.get("content_hash"),
                chunk_kind=meta.get("chunk_kind"),
            ),
            mode="witness_first",
            scope="global" if scope == "global" else "local",
        )

    async def _hydrate_rows_from_witness(self, rows: list[Any]) -> dict[str, CItem]:
        if self._db is None or not rows:
            return {}

        local_batches: dict[str, list[str]] = {}
        global_ids: list[str] = []
        for row in rows:
            payload = dict(getattr(row, "payload", None) or {})
            ref_id = str(getattr(row, "id", payload.get("citem_id") or payload.get("ref_id") or ""))
            if not ref_id:
                continue
            witness_scope = str(payload.get("w_scope") or ("global" if payload.get("scope") == "global" else "local"))
            if witness_scope == "global":
                global_ids.append(ref_id)
                continue
            conversation_id = str(payload.get("conversation_id") or "")
            if not conversation_id:
                continue
            local_batches.setdefault(conversation_id, []).append(ref_id)

        hydrated: dict[str, CItem] = {}
        for conversation_id, ids in local_batches.items():
            records = await self._db.list_local_citem_records(conversation_id, citem_ids=list(dict.fromkeys(ids)))
            for record in records:
                citem = self._witness_row_to_citem(record, scope="episodic")
                hydrated[citem.citem_id] = citem
        if global_ids:
            records = await self._db.list_global_citem_records(global_citem_ids=list(dict.fromkeys(global_ids)))
            for record in records:
                citem = self._witness_row_to_citem(record, scope="global")
                hydrated[citem.citem_id] = citem
        return hydrated

    # ── Payload serialization ─────────────────────────────────────────────────

    def _citem_to_payload(self, citem: CItem) -> dict[str, Any]:
        return {
            "citem_id": citem.citem_id,
            "conversation_id": citem.conversation_id,
            "content": citem.content,
            "item_type": str(citem.item_type),
            "scope": citem.scope,
            "scope_status": citem.scope_status,
            "importance": citem.importance,
            "confidence": citem.confidence,
            "validation_label": citem.validation_label,
            "conflict_status": citem.conflict_status,
            "phase_ingested": str(citem.phase_ingested),
            "actor": citem.actor,
            "motivation": citem.motivation,
            "created_at_unix": citem.created_at.timestamp(),
            "dependency_ids": citem.dependency_ids,
            "token_count": citem.token_count,
            **({"archived_at_unix": citem.archived_at_unix} if citem.archived_at_unix is not None else {}),
            **({"summarized_by_node_id": citem.summarized_by_node_id} if citem.summarized_by_node_id is not None else {}),
            **({"content_hash": citem.content_hash} if citem.content_hash is not None else {}),
            **({"chunk_kind": citem.chunk_kind} if citem.chunk_kind is not None else {}),
        }

    def _payload_to_citem(self, payload: dict[str, Any]) -> CItem:
        return self._annotate_resolution(
            CItem(
                citem_id=payload["citem_id"],
                conversation_id=payload["conversation_id"],
                content=payload["content"],
                item_type=payload["item_type"],
                scope=payload["scope"],
                scope_status=payload["scope_status"],
                importance=float(payload.get("importance", 0.5)),
                confidence=float(payload.get("confidence", 1.0)),
                validation_label=payload.get("validation_label"),
                conflict_status=payload.get("conflict_status", "none"),
                phase_ingested=payload.get("phase_ingested", "IDLE"),
                actor=payload.get("actor", "agent"),
                motivation=payload.get("motivation"),
                created_at=datetime.fromtimestamp(float(payload.get("created_at_unix", 0.0)), tz=UTC),
                dependency_ids=list(payload.get("dependency_ids", [])),
                token_count=int(payload.get("token_count", 0)),
                archived_at_unix=payload.get("archived_at_unix"),
                summarized_by_node_id=payload.get("summarized_by_node_id"),
                content_hash=payload.get("content_hash"),
                chunk_kind=payload.get("chunk_kind"),
            ),
            mode="legacy_fallback",
            scope=str(payload.get("w_scope") or payload.get("scope") or "legacy"),
        )

    # ── Filter builder ────────────────────────────────────────────────────────

    def _build_filter(
        self,
        f: CItemFilter,
        extra_exclude_ids: list[str] | None = None,
    ) -> Filter:
        must: list[Any] = []
        must_not: list[Any] = []

        if f.scope is not None:
            must.append(FieldCondition(key="scope", match=MatchValue(value=f.scope)))
        if f.scope_status is not None:
            must.append(FieldCondition(key="scope_status", match=MatchValue(value=f.scope_status)))
        if f.conversation_id is not None:
            must.append(FieldCondition(key="conversation_id", match=MatchValue(value=f.conversation_id)))
        if f.item_types:
            must.append(FieldCondition(key="item_type", match=MatchAny(any=list(f.item_types))))
        if f.actor is not None:
            must.append(FieldCondition(key="actor", match=MatchValue(value=f.actor)))
        if f.conflict_status_in:
            must.append(FieldCondition(key="conflict_status", match=MatchAny(any=list(f.conflict_status_in))))

        exclude = list(f.exclude_ids)
        if extra_exclude_ids:
            exclude.extend(extra_exclude_ids)
        if exclude:
            must_not.append(HasIdCondition(has_id=exclude))  # type: ignore[arg-type]

        return Filter(must=must, must_not=must_not)

    async def _search_collection(
        self,
        *,
        collection: str,
        query_text: str,
        filter: CItemFilter,
        top_k: int,
    ) -> list[ScoredCItem]:
        dense_vec, sparse_vec = await asyncio.gather(
            self._embed.embed(query_text),
            self._encode_sparse(query_text),
        )
        qdrant_filter = self._build_filter(filter)

        prefetch: list[Prefetch] = [
            Prefetch(
                query=dense_vec,
                using="content",
                filter=qdrant_filter,
                limit=top_k * 2,
            ),
        ]
        if sparse_vec:
            prefetch.append(
                Prefetch(
                    query=SparseVector(indices=list(sparse_vec.keys()), values=list(sparse_vec.values())),
                    using="content_sparse",
                    filter=qdrant_filter,
                    limit=top_k * 2,
                )
            )
        else:
            log.debug("search: empty sparse vector for query %r — dense-only", query_text[:60])

        hybrid_coro = self._client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        dense_coro = self._client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="content",
            query_filter=qdrant_filter,
            limit=top_k * 2,
            with_payload=False,
            with_vectors=False,
        )
        results, dense_results = await asyncio.gather(hybrid_coro, dense_coro)
        cosine_map: dict[str, float] = {str(r.id): r.score for r in dense_results.points}
        provenance = self._search_provenance(filter.scope)
        witness_rows = await self._hydrate_rows_from_witness(list(results.points))
        scored_results: list[ScoredCItem] = []
        for r in results.points:
            row_id = str(r.id)
            citem = witness_rows.get(row_id)
            if citem is None:
                citem = self._payload_to_citem(r.payload)  # type: ignore[arg-type]
            scored_results.append(
                ScoredCItem(
                    citem=citem,
                    score=float(r.score),
                    provenance=provenance,
                    dense_score=float(cosine_map.get(row_id, r.score)),
                )
            )
        return scored_results

    @staticmethod
    def _merge_scored_results(*pools: list[ScoredCItem], top_k: int) -> list[ScoredCItem]:
        merged: dict[str, ScoredCItem] = {}
        for pool in pools:
            for scored in pool:
                cid = scored.citem.citem_id
                if cid not in merged or scored.score > merged[cid].score:
                    merged[cid] = scored
        return sorted(merged.values(), key=lambda row: row.score, reverse=True)[:top_k]

    # ── CItemStorePort implementation ─────────────────────────────────────────

    async def save(self, citem: CItem) -> None:
        try:
            dense_vec, sparse_vec = await asyncio.gather(
                self._embed.embed(citem.content),
                self._encode_sparse(citem.content),
            )
            point = PointStruct(
                id=citem.citem_id,
                payload=self._citem_to_payload(citem),
                vector={
                    "content": dense_vec,
                    "content_sparse": SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values()),
                    ),
                },
            )
            await self._client.upsert(collection_name=self._collection_for_scope(citem.scope), points=[point])
        except Exception as exc:
            raise CItemStoreError(f"save failed for {citem.citem_id}: {exc}") from exc

    async def fetch(self, citem_id: str) -> CItem:
        try:
            for collection in (self._collection, self._global_collection):
                results = await self._retrieve_many(collection, [citem_id])
                if results:
                    witness_rows = await self._hydrate_rows_from_witness(results)
                    hydrated = witness_rows.get(str(results[0].id))
                    if hydrated is not None:
                        return hydrated
                    return self._payload_to_citem(results[0].payload)  # type: ignore[arg-type]
        except Exception as exc:
            raise CItemStoreError(f"fetch failed: {exc}") from exc
        raise CItemNotFoundError(f"CItem {citem_id} not found")

    async def fetch_batch(self, citem_ids: list[str]) -> list[CItem]:
        if not citem_ids:
            return []
        try:
            results: list[Any] = []
            seen: set[str] = set()
            for collection in (self._collection, self._global_collection):
                batch = await self._retrieve_many(collection, citem_ids)
                for row in batch:
                    row_id = str(row.id)
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    results.append(row)
            witness_rows = await self._hydrate_rows_from_witness(results)
        except Exception as exc:
            raise CItemStoreError(f"fetch_batch failed: {exc}") from exc
        items: list[CItem] = []
        for row in results:
            row_id = str(row.id)
            items.append(witness_rows.get(row_id) or self._payload_to_citem(row.payload))  # type: ignore[arg-type]
        return items

    async def search(
        self,
        query_text: str,
        filter: CItemFilter,
        top_k: int,
    ) -> list[ScoredCItem]:
        try:
            search_ephemeral = filter.scope != "global" and bool(filter.conversation_id) and self._ephemeral_collection != self._collection
            if filter.scope == "global":
                return await self._search_collection(
                    collection=self._global_collection,
                    query_text=query_text,
                    filter=filter,
                    top_k=top_k,
                )
            if filter.scope == "episodic":
                if search_ephemeral:
                    local_results, ephemeral_results = await asyncio.gather(
                        self._search_collection(
                            collection=self._collection,
                            query_text=query_text,
                            filter=filter,
                            top_k=top_k,
                        ),
                        self._search_collection(
                            collection=self._ephemeral_collection,
                            query_text=query_text,
                            filter=filter,
                            top_k=top_k,
                        ),
                    )
                    return self._merge_scored_results(local_results, ephemeral_results, top_k=top_k)
                return await self._search_collection(
                    collection=self._collection,
                    query_text=query_text,
                    filter=filter,
                    top_k=top_k,
                )
            coros = [
                self._search_collection(
                    collection=self._collection,
                    query_text=query_text,
                    filter=filter,
                    top_k=top_k,
                ),
                self._search_collection(
                    collection=self._global_collection,
                    query_text=query_text,
                    filter=filter,
                    top_k=top_k,
                ),
            ]
            if search_ephemeral:
                coros.append(
                    self._search_collection(
                        collection=self._ephemeral_collection,
                        query_text=query_text,
                        filter=filter,
                        top_k=top_k,
                    )
                )
            pools = await asyncio.gather(*coros)
            return self._merge_scored_results(*pools, top_k=top_k)
        except Exception as exc:
            raise CItemStoreError(f"search failed: {exc}") from exc

    async def fetch_neighbors(
        self,
        seed_ids: list[str],
        conversation_id: str,
        exclude_ids: set[str] | None = None,
        backward_max: int = 500,
    ) -> list[CItem]:
        if not seed_ids:
            return []
        try:
            exclude_set = (exclude_ids or set()) | set(seed_ids)
            seeds = await self.fetch_batch(seed_ids)
            forward_ids: set[str] = set()
            for seed in seeds:
                forward_ids.update(seed.dependency_ids)

            fetchable = list(forward_ids - exclude_set)
            forward_neighbors = await self.fetch_batch(fetchable) if fetchable else []

            candidates = await self._fetch_by_conversation_capped(
                conversation_id,
                scope_status="active",
                max_items=backward_max,
            )
            seed_set = set(seed_ids)
            backward_neighbors = [
                c for c in candidates
                if c.citem_id not in exclude_set and any(dep in seed_set for dep in c.dependency_ids)
            ]

            seen: set[str] = set()
            result: list[CItem] = []
            for item in forward_neighbors + backward_neighbors:
                if item.citem_id not in seen:
                    seen.add(item.citem_id)
                    result.append(item)
            return result
        except CItemStoreError:
            raise
        except Exception as exc:
            raise GeometricExpansionError(f"fetch_neighbors failed: {exc}") from exc

    async def _fetch_by_conversation_capped(
        self,
        conversation_id: str,
        scope_status: str | None = None,
        max_items: int = 500,
    ) -> list[CItem]:
        conditions: list[Any] = [
            FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))
        ]
        if scope_status is not None:
            conditions.append(FieldCondition(key="scope_status", match=MatchValue(value=scope_status)))
        scroll_filter = Filter(must=conditions)
        raw_rows: list[Any] = []
        offset = None
        try:
            while len(raw_rows) < max_items:
                batch_limit = min(200, max_items - len(raw_rows))
                results, next_offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=batch_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                raw_rows.extend(results)
                if next_offset is None or len(raw_rows) >= max_items:
                    break
                offset = next_offset
            witness_rows = await self._hydrate_rows_from_witness(raw_rows)
        except Exception as exc:
            raise CItemStoreError(f"_fetch_by_conversation_capped failed: {exc}") from exc
        items: list[CItem] = []
        for row in raw_rows[:max_items]:
            row_id = str(row.id)
            items.append(witness_rows.get(row_id) or self._payload_to_citem(row.payload))  # type: ignore[arg-type]
        return items

    async def update_field(self, citem_id: str, field: str, value: Any) -> None:
        try:
            payload: dict[str, Any] = {field: value}
            if field == "scope_status" and value == "archived":
                payload["archived_at_unix"] = datetime.now(UTC).timestamp()
            current = await self.fetch(citem_id)
            await self._client.set_payload(
                collection_name=self._collection_for_scope(current.scope),
                payload=payload,
                points=[citem_id],
            )
        except Exception as exc:
            raise CItemStoreError(f"update_field failed for {citem_id}: {exc}") from exc

    async def delete(self, citem_id: str) -> None:
        try:
            for collection in (self._collection, self._global_collection):
                await self._client.delete(collection_name=collection, points_selector=[citem_id])
        except Exception as exc:
            raise CItemStoreError(f"delete failed for {citem_id}: {exc}") from exc

    async def delete_by_conversation(self, conversation_id: str) -> int:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=Filter(
                    must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
                ),
            )
            return 0
        except Exception as exc:
            raise CItemStoreError(f"delete_by_conversation failed: {exc}") from exc

    async def fetch_by_conversation(
        self,
        conversation_id: str,
        scope_status: str | None = None,
    ) -> list[CItem]:
        conditions: list[Any] = [
            FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))
        ]
        if scope_status is not None:
            conditions.append(FieldCondition(key="scope_status", match=MatchValue(value=scope_status)))
        scroll_filter = Filter(must=conditions)
        raw_rows: list[Any] = []
        offset = None
        try:
            while True:
                results, next_offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                raw_rows.extend(results)
                if next_offset is None:
                    break
                offset = next_offset
            witness_rows = await self._hydrate_rows_from_witness(raw_rows)
        except Exception as exc:
            raise CItemStoreError(f"fetch_by_conversation failed: {exc}") from exc
        items: list[CItem] = []
        for row in raw_rows:
            row_id = str(row.id)
            items.append(witness_rows.get(row_id) or self._payload_to_citem(row.payload))  # type: ignore[arg-type]
        return items

    async def exists_by_hash(self, content_hash: str, conversation_id: str) -> bool:
        try:
            results, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                        FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results) > 0
        except Exception:
            return False

    async def fetch_dense_vectors(self, citem_ids: list[str]) -> dict[str, list[float]]:
        if not citem_ids:
            return {}
        try:
            local_results, global_results = await asyncio.gather(
                self._client.retrieve(
                    collection_name=self._collection,
                    ids=citem_ids,
                    with_payload=False,
                    with_vectors=["content"],
                ),
                self._client.retrieve(
                    collection_name=self._global_collection,
                    ids=citem_ids,
                    with_payload=False,
                    with_vectors=["content"],
                ),
            )
            merged = [*local_results, *global_results]
            return {
                str(r.id): r.vector["content"]
                for r in merged
                if r.vector and "content" in r.vector
            }
        except Exception as exc:
            raise CItemStoreError(f"fetch_dense_vectors failed: {exc}") from exc

    async def ping(self) -> bool:
        """Check local/global collections and recreate them if missing."""
        try:
            await self._client.get_collection(self._collection)
            await self._client.get_collection(self._global_collection)
            return True
        except Exception:
            pass
        for collection in (self._collection, self._global_collection, self._ephemeral_collection):
            log.warning("QdrantCItemAdapter: collection '%s' not found — recreating", collection)
            try:
                await ensure_collection(self._client, collection, self._dense_dim)
                log.info("QdrantCItemAdapter: collection '%s' recreated", collection)
            except Exception as exc:
                log.error("QdrantCItemAdapter: failed to recreate collection '%s': %s", collection, exc)
                return False
        return True
