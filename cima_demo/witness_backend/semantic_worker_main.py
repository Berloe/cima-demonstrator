from __future__ import annotations

import asyncio
import json
import logging

from cima_demo.api.settings import get_settings
from cima_demo.infrastructure.embedding.tei import TEIAdapter
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.infrastructure.tokenizer import LlamaCppTokenizerClient
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.semantic_pipeline import MemorySemanticConsumer
from cima_demo.witness_backend.topic_catalog import TOPICS

log = logging.getLogger(__name__)


async def run_semantic_worker() -> None:
    from qdrant_client import AsyncQdrantClient
    from aiokafka import AIOKafkaConsumer

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    tokenizer = LlamaCppTokenizerClient(base_url=settings.llm_url, timeout=settings.llm_timeout)
    embedder = TEIAdapter(base_url=settings.tei_url, timeout=settings.tei_timeout)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    plane = QdrantWitnessPlane(
        client=qdrant,
        catalog=QdrantCollectionCatalog.from_settings(settings),
        dense_dim=await embedder.get_dim(),
    )
    await plane.ensure_ready()

    logic = MemorySemanticConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=tokenizer,
        embedder=embedder,
        qdrant_plane=plane,
        embedding_model_id=settings.tei_url,
        embedding_schema_version=1,
    )

    consumer = AIOKafkaConsumer(
        TOPICS.memory_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="cima-semantic-worker",
        enable_auto_commit=False,
        value_deserializer=lambda data: None if data is None else json.loads(data.decode("utf-8")),
        key_deserializer=lambda data: None if data is None else data.decode("utf-8"),
    )
    await consumer.start()
    log.info("Semantic worker started bootstrap=%s", settings.kafka_bootstrap)
    try:
        async for message in consumer:
            payload = message.value
            if payload is None:
                await consumer.commit()
                continue
            await logic.handle(payload)
            await consumer.commit()
    finally:
        await consumer.stop()
        await tokenizer.aclose()
        await qdrant.close()
        await pool.close()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_semantic_worker())


if __name__ == "__main__":
    main()
