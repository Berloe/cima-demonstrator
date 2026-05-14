import asyncio

import pytest

from cima_demo.domain.entities import KimaDelta
from cima_demo.domain.value_objects import KimaDeltaType
from cima_demo.infrastructure.events.direct import DirectSSEEventBus


@pytest.mark.asyncio
async def test_done_is_delivered_even_when_queue_is_full() -> None:
    bus = DirectSSEEventBus(queue_maxsize=2)
    sub = bus.subscribe("conv-1")

    await bus.publish(KimaDelta(type=KimaDeltaType.TOKEN, conversation_id="conv-1", token="a"))
    await bus.publish(KimaDelta(type=KimaDeltaType.TOKEN, conversation_id="conv-1", token="b"))
    await bus.publish(KimaDelta(type=KimaDeltaType.DONE, conversation_id="conv-1"))

    seen: list[KimaDelta] = []
    async for item in sub:
        seen.append(item)

    assert seen
    assert seen[-1].type == KimaDeltaType.DONE


@pytest.mark.asyncio
async def test_error_is_prioritized_when_queue_is_full() -> None:
    bus = DirectSSEEventBus(queue_maxsize=2)
    sub = bus.subscribe("conv-2")

    await bus.publish(KimaDelta(type=KimaDeltaType.THOUGHT, conversation_id="conv-2", thought="x"))
    await bus.publish(KimaDelta(type=KimaDeltaType.THOUGHT, conversation_id="conv-2", thought="y"))
    await bus.publish(KimaDelta(
        type=KimaDeltaType.ERROR,
        conversation_id="conv-2",
        error_code="E",
        error_message="boom",
    ))
    await bus.publish(KimaDelta(type=KimaDeltaType.DONE, conversation_id="conv-2"))

    seen: list[KimaDelta] = []
    async for item in sub:
        seen.append(item)

    assert any(item.type == KimaDeltaType.ERROR for item in seen)
    assert seen[-1].type == KimaDeltaType.DONE
