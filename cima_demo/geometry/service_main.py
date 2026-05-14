from __future__ import annotations

"""Minimal geometry service runtime entrypoint.

This is intentionally small: it consumes geometry commands from Kafka, processes
them through the detached geometry worker, appends results to the geometry-owned
outbox and flushes them through the shared internal publisher. It exposes no
business HTTP surface.
"""

import asyncio
import json
import logging

from cima_demo.api.settings import get_settings
from cima_demo.geometry.service import DemoGeometryService
from cima_demo.geometry.worker import GeometryCommandProcessor
from cima_demo.infrastructure.embedding.tei import TEIAdapter
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.qdrant import QdrantCItemAdapter
from cima_demo.infrastructure.qdrant.setup import ensure_collections
from cima_demo.witness_backend.events import CloudEventEnvelope
from cima_demo.witness_backend.outbox_store import DatabaseOutboxStore
from cima_demo.witness_backend.publisher import OutboxPublisher
from cima_demo.witness_backend.topic_catalog import TOPICS

log = logging.getLogger(__name__)


async def run_geometry_service() -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from qdrant_client import AsyncQdrantClient

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    tei = TEIAdapter(base_url=settings.tei_url, timeout=settings.tei_timeout)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    catalog = QdrantCollectionCatalog.from_settings(settings)
    await ensure_collections(qdrant, catalog, settings.tei_embed_dim)
    citem_store = QdrantCItemAdapter(
        client=qdrant,
        embedding_port=tei,
        collection=catalog.local_citems,
        global_collection=catalog.global_citems,
        ephemeral_collection=catalog.ephemeral,
        sparse_embedding_port=None,
        dense_dim=settings.tei_embed_dim,
        rel_db=db,
    )
    service = DemoGeometryService(rel_db=db, citem_store=citem_store, embedding_port=tei, k_max=8)
    processor = GeometryCommandProcessor(service)

    consumer = AIOKafkaConsumer(
        TOPICS.geom_cmd,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="cima-geometry",
        enable_auto_commit=False,
        value_deserializer=lambda data: json.loads(data.decode("utf-8")),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=lambda value: None if value is None else json.dumps(value, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        acks="all",
    )
    geom_store = DatabaseOutboxStore(db, schema_name="geom")
    publisher = OutboxPublisher(store=geom_store, producer=producer)
    await consumer.start()
    await producer.start()
    try:
        async for message in consumer:
            envelope = CloudEventEnvelope.model_validate(message.value)
            outputs = await processor.handle(envelope)
            for topic, key, payload in outputs:
                await db.append_geom_outbox_event(topic=topic, message_key=key, payload_json=payload)
            await publisher.publish_once(limit=max(100, len(outputs) + 1))
            await consumer.commit()
    finally:
        await consumer.stop()
        await producer.stop()
        await tei.aclose()
        await qdrant.close()
        await pool.close()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_geometry_service())


if __name__ == "__main__":
    main()
