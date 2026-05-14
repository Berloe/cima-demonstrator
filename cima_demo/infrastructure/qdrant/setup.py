"""Qdrant collection setup — run once on first deployment."""
from __future__ import annotations

import contextlib
import logging

try:  # pragma: no cover - real runtime path
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )
except ModuleNotFoundError:  # pragma: no cover - lightweight test fallback
    from typing import Any as AsyncQdrantClient

    class Distance:
        COSINE = "cosine"

    class PayloadSchemaType:
        UUID = "uuid"
        KEYWORD = "keyword"
        FLOAT = "float"

    class SparseIndexParams:
        def __init__(self, *, on_disk: bool) -> None:
            self.on_disk = on_disk

    class SparseVectorParams:
        def __init__(self, *, index: object) -> None:
            self.index = index

    class VectorParams:
        def __init__(self, *, size: int, distance: object) -> None:
            self.size = size
            self.distance = distance

from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog

log = logging.getLogger(__name__)

# conversation_id is always a UUID → UUID index (B-tree, O(log n)) is more
# efficient than KEYWORD (inverted index) for exact-match on high-cardinality UUIDs.
_UUID_FIELDS = ["conversation_id"]
# Low-cardinality string fields — keyword index is appropriate
_KEYWORD_FIELDS = ["scope", "scope_status", "item_type", "conflict_status", "content_hash"]
_FLOAT_FIELDS = ["importance", "created_at_unix"]


async def ensure_collection(
    client: AsyncQdrantClient,
    collection: str,
    dense_dim: int,
) -> None:
    """Create a single Qdrant collection if it doesn't exist, then create payload indexes."""
    try:
        collections_response = await client.get_collections()
        existing = {c.name for c in collections_response.collections}
    except Exception:
        existing = set()

    if collection not in existing:
        log.info("Creating Qdrant collection '%s' (dense_dim=%d)", collection, dense_dim)
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                "content": VectorParams(size=dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "content_sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

    # Create payload indexes (idempotent)
    for field in _UUID_FIELDS:
        with contextlib.suppress(Exception):
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.UUID,
            )

    for field in _KEYWORD_FIELDS:
        with contextlib.suppress(Exception):
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    for field in _FLOAT_FIELDS:
        with contextlib.suppress(Exception):
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.FLOAT,
            )

    log.info("Qdrant collection '%s' ready", collection)


async def ensure_collections(
    client: AsyncQdrantClient,
    catalog: QdrantCollectionCatalog,
    dense_dim: int,
) -> None:
    """Ensure the full witness-backend collection layout exists."""
    for collection in catalog.all():
        await ensure_collection(client, collection, dense_dim)
