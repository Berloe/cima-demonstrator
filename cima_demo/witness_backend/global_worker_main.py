from __future__ import annotations

import argparse
import asyncio
import json

from cima_demo.api.settings import get_settings
from cima_demo.infrastructure.embedding.tei import TEIEmbeddingAdapter
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.infrastructure.tokenizer.llamacpp import LlamaCppTokenizer
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.global_memory import GlobalPromotionConsumer


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Run witness global-memory promotion worker once")
    parser.add_argument("payload", help="CloudEvent JSON payload")
    args = parser.parse_args()

    settings = get_settings()
    pool = await create_pool(settings.postgres_dsn)
    try:
        db = PostgreSQLAdapter(pool)
        tokenizer = LlamaCppTokenizer(base_url=settings.llm_url, timeout_seconds=settings.llm_timeout)
        embedder = TEIEmbeddingAdapter(settings.embedder_url)
        from qdrant_client import AsyncQdrantClient  # pragma: no cover

        client = AsyncQdrantClient(url=settings.qdrant_url)
        plane = QdrantWitnessPlane(
            client=client,
            catalog=QdrantCollectionCatalog.from_settings(settings),
            dense_dim=settings.qdrant_dense_dim,
        )
        consumer = GlobalPromotionConsumer(
            db=db,
            ledger=ConsumerEffectLedger(db),
            tokenizer=tokenizer,
            embedder=embedder,
            qdrant_plane=plane,
            embedding_model_id=settings.embedding_model_id,
            embedding_schema_version=settings.embedding_schema_version,
        )
        await consumer.handle(json.loads(args.payload))
    finally:
        await pool.close()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    main()
