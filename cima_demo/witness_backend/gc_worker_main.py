from __future__ import annotations

import asyncio
import json
import logging

from cima_demo.api.settings import get_settings
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.infrastructure.qdrant.catalog import QdrantCollectionCatalog
from cima_demo.infrastructure.qdrant.witness_plane import QdrantWitnessPlane
from cima_demo.geometry.boundary import GeometryCommandPublisher
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.hard_delete import HardDeleteConsumer
from cima_demo.witness_backend.maintenance import MaintenanceConsumer
from cima_demo.witness_backend.topic_catalog import TOPICS

log = logging.getLogger(__name__)


async def run_gc_worker() -> None:
    from aiokafka import AIOKafkaConsumer
    from qdrant_client import AsyncQdrantClient

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    qdrant_plane = QdrantWitnessPlane(
        client=qdrant_client,
        catalog=QdrantCollectionCatalog.from_settings(settings),
        dense_dim=settings.tei_embed_dim,
    )
    ledger = ConsumerEffectLedger(db)
    hard_delete_logic = HardDeleteConsumer(
        store=db,
        citem_store=qdrant_plane,
        ledger=ledger,
        geometry_commands=GeometryCommandPublisher(db),
    )
    maintenance_logic = MaintenanceConsumer(
        store=db,
        qdrant_plane=qdrant_plane,
        ledger=ledger,
        thinning_age_hours=max(1, int(getattr(settings, "gc_thinning_age_hours", 24))),
    )

    consumer = AIOKafkaConsumer(
        TOPICS.conversation_events,
        TOPICS.gc_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="cima-gc-worker",
        enable_auto_commit=False,
        value_deserializer=lambda data: None if data is None else json.loads(data.decode("utf-8")),
        key_deserializer=lambda data: None if data is None else data.decode("utf-8"),
    )
    await consumer.start()
    log.info("GC worker started bootstrap=%s", settings.kafka_bootstrap)
    try:
        async for message in consumer:
            payload = message.value
            if payload is None:
                await consumer.commit()
                continue
            event_type = payload.get("type")
            if event_type == "cima.conversation.hard_delete.requested.v1":
                await hard_delete_logic.handle(payload_json=payload)
            elif event_type in {
                "cima.gc.thinning.requested.v1",
                "cima.gc.ephemeral_expiry.requested.v1",
                "cima.gc.reconcile.requested.v1",
            }:
                await maintenance_logic.handle(payload_json=payload)
            await consumer.commit()
    finally:
        await consumer.stop()
        await qdrant_client.close()
        await pool.close()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_gc_worker())


if __name__ == "__main__":
    main()
