from __future__ import annotations

"""Conversation lifecycle guards for async consumers.

Late events must not resurrect work for conversations already entering DELETING
or DELETED states. Consumers use these helpers after acquiring their consumer
ledger slot so the skip itself becomes an idempotent, durable effect.
"""

from typing import Any

from cima_demo.witness_backend.consumer_effect import ConsumerEffectKey, ConsumerEffectLedger


async def load_conversation_status(store: Any, conversation_id: str) -> str | None:
    getter = getattr(store, "get_conversation", None)
    if getter is None:
        return "ACTIVE"
    row = await getter(conversation_id)
    if row is None:
        return None
    return str(row.get("status") or "ACTIVE")


async def complete_if_conversation_not_active(
    *,
    store: Any,
    ledger: ConsumerEffectLedger,
    key: ConsumerEffectKey,
    conversation_id: str,
) -> bool:
    status = await load_conversation_status(store, conversation_id)
    if status == "ACTIVE":
        return False
    await ledger.complete(
        key,
        details_json={
            "status": "conversation_not_active",
            "conversation_status": status or "MISSING",
        },
    )
    return True
