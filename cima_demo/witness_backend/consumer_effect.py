from __future__ import annotations

"""Consumer-effect ledger helpers.

At-least-once Kafka delivery becomes exactly-once domain effect by guarding each
consumer side effect with a unique (consumer_name, event_id, effect_key) row.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ConsumerEffectKey:
    consumer_name: str
    event_id: str
    effect_key: str


class ConsumerEffectStore(Protocol):
    async def begin_consumer_effect(self, *, consumer_name: str, event_id: str, effect_key: str) -> bool: ...

    async def complete_consumer_effect(
        self,
        *,
        consumer_name: str,
        event_id: str,
        effect_key: str,
        details_json: dict[str, Any] | None = None,
    ) -> None: ...


class ConsumerEffectLedger:
    def __init__(self, store: ConsumerEffectStore) -> None:
        self._store = store

    async def begin(self, key: ConsumerEffectKey) -> bool:
        return await self._store.begin_consumer_effect(
            consumer_name=key.consumer_name,
            event_id=key.event_id,
            effect_key=key.effect_key,
        )

    async def complete(self, key: ConsumerEffectKey, *, details_json: dict[str, Any] | None = None) -> None:
        await self._store.complete_consumer_effect(
            consumer_name=key.consumer_name,
            event_id=key.event_id,
            effect_key=key.effect_key,
            details_json=details_json,
        )
