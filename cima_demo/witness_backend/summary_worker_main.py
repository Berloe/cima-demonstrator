from __future__ import annotations

import asyncio
import json
import logging

from cima_demo.api.settings import get_settings
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.infrastructure.tokenizer import LlamaCppTokenizerClient
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.summary_pipeline import MemorySummaryConsumer
from cima_demo.witness_backend.topic_catalog import TOPICS

log = logging.getLogger(__name__)


async def run_summary_worker() -> None:
    from aiokafka import AIOKafkaConsumer

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    tokenizer = LlamaCppTokenizerClient(base_url=settings.llm_url, timeout=settings.llm_timeout)
    logic = MemorySummaryConsumer(
        db=db,
        ledger=ConsumerEffectLedger(db),
        tokenizer=tokenizer,
    )

    consumer = AIOKafkaConsumer(
        TOPICS.memory_events,
        TOPICS.summary_cmd,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="cima-summary-worker",
        enable_auto_commit=False,
        value_deserializer=lambda data: None if data is None else json.loads(data.decode("utf-8")),
        key_deserializer=lambda data: None if data is None else data.decode("utf-8"),
    )
    await consumer.start()
    log.info("Summary worker started bootstrap=%s", settings.kafka_bootstrap)
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
        await pool.close()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_summary_worker())


if __name__ == "__main__":
    main()
