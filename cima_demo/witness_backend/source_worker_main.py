from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from cima_demo.api.settings import get_settings
from cima_demo.infrastructure.files.chunker import SemanticChunkerAdapter
from cima_demo.infrastructure.files.processor import FileProcessingAdapter
from cima_demo.infrastructure.postgres.migrations import run_migrations
from cima_demo.infrastructure.postgres.postgres import PostgreSQLAdapter, create_pool
from cima_demo.infrastructure.tokenizer import LlamaCppTokenizerClient
from cima_demo.witness_backend.consumer_effect import ConsumerEffectLedger
from cima_demo.witness_backend.source_ingest import MemorySourceConsumer
from cima_demo.witness_backend.topic_catalog import TOPICS

log = logging.getLogger(__name__)


async def run_source_worker() -> None:
    from aiokafka import AIOKafkaConsumer

    settings = get_settings()
    pool = await create_pool(settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max)
    await run_migrations(pool)
    db = PostgreSQLAdapter(pool)
    tokenizer = LlamaCppTokenizerClient(base_url=settings.llm_url, timeout=settings.llm_timeout)
    chunker = SemanticChunkerAdapter(token_counter=lambda text: tokenizer.count_text_tokens_sync(text))
    logic = MemorySourceConsumer(
        db=db,
        chunker=chunker,
        file_processor=FileProcessingAdapter(),
        ledger=ConsumerEffectLedger(db),
    )

    consumer = AIOKafkaConsumer(
        TOPICS.memory_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="cima-source-worker",
        enable_auto_commit=False,
        value_deserializer=lambda data: None if data is None else json.loads(data.decode("utf-8")),
        key_deserializer=lambda data: None if data is None else data.decode("utf-8"),
    )
    await consumer.start()
    log.info("Source worker started bootstrap=%s", settings.kafka_bootstrap)
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
        await pool.close()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level.upper())
    asyncio.run(run_source_worker())


if __name__ == "__main__":
    main()
