from __future__ import annotations

import argparse
import asyncio
import json
import logging

from cima_demo.api.settings import get_settings
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.witness_backend.outbox_store import DatabaseOutboxStore
from cima_demo.witness_backend.publisher import OutboxPublisher

log = logging.getLogger(__name__)


async def run_publisher(*, schema_name: str, poll_interval: float = 1.0, batch_size: int = 100) -> None:
    from aiokafka import AIOKafkaProducer

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        value_serializer=lambda value: None if value is None else json.dumps(value, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        acks="all",
    )
    store = DatabaseOutboxStore(db, schema_name=schema_name)
    publisher = OutboxPublisher(store=store, producer=producer)
    await producer.start()
    log.info("Outbox publisher started schema=%s bootstrap=%s", schema_name, settings.kafka_bootstrap)
    try:
        while True:
            report = await publisher.publish_once(limit=batch_size)
            if report.claimed == 0:
                await asyncio.sleep(poll_interval)
    finally:
        await producer.stop()
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish witness-backend outbox rows to Kafka")
    parser.add_argument("--schema", choices=["cima", "geom"], default="cima")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_publisher(schema_name=args.schema, poll_interval=args.poll_interval, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
