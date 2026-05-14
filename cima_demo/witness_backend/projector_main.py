from __future__ import annotations

import asyncio
import json
import logging

from cima_demo.api.settings import get_settings
from cima_demo.geometry.projector import GeometryReadModelProjector
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.projector_consumer import GeometryReadModelProjectorConsumer
from cima_demo.witness_backend.topic_catalog import TOPICS

log = logging.getLogger(__name__)


async def run_projector() -> None:
    from aiokafka import AIOKafkaConsumer

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    projector = GeometryReadModelProjector(db)
    ledger = ConsumerEffectLedger(db)
    consumer_logic = GeometryReadModelProjectorConsumer(projector=projector, ledger=ledger)

    consumer = AIOKafkaConsumer(
        TOPICS.geom_run,
        TOPICS.geom_item_state,
        TOPICS.geom_cluster_state,
        TOPICS.conversation_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="cima-geom-projector",
        enable_auto_commit=False,
        value_deserializer=lambda data: None if data is None else json.loads(data.decode("utf-8")),
        key_deserializer=lambda data: None if data is None else data.decode("utf-8"),
    )
    await consumer.start()
    log.info("Geometry read-model projector started bootstrap=%s", settings.kafka_bootstrap)
    try:
        async for message in consumer:
            key = message.key or ""
            await consumer_logic.handle(topic=message.topic, message_key=key, payload_json=message.value)
            await consumer.commit()
    finally:
        await consumer.stop()
        await pool.close()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_projector())


if __name__ == "__main__":
    main()
