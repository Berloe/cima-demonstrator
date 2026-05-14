from __future__ import annotations

"""Canonical Kafka topic names and key builders for witness backend v1.1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicCatalog:
    memory_events: str = "cima.memory.events.v1"
    vector_events: str = "cima.vector.events.v1"
    conversation_events: str = "cima.conversation.events.v1"
    pin_events: str = "cima.pin.events.v1"
    geom_cmd: str = "cima.geom.cmd.v1"
    geom_run: str = "cima.geom.run.v1"
    geom_item_state: str = "cima.geom.item_state.v1"
    geom_cluster_state: str = "cima.geom.cluster_state.v1"
    summary_cmd: str = "cima.summary.cmd.v1"
    handoff_events: str = "cima.handoff.events.v1"
    gc_events: str = "cima.gc.events.v1"


TOPICS = TopicCatalog()

COMPACTED_TOPICS = frozenset({TOPICS.geom_item_state, TOPICS.geom_cluster_state})


def is_compacted_topic(topic: str) -> bool:
    return topic in COMPACTED_TOPICS


def cleanup_policy_for(topic: str) -> str:
    return "compact,delete" if is_compacted_topic(topic) else "delete"


def conversation_key(conversation_id: str) -> str:
    return conversation_id


def geom_item_state_key(conversation_id: str, ref_kind: str, ref_id: str) -> str:
    return f"{conversation_id}|{ref_kind}|{ref_id}"


def geom_cluster_state_key(conversation_id: str, cluster_id: str) -> str:
    return f"{conversation_id}|{cluster_id}"
