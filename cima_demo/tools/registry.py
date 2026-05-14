"""Minimal tool DTOs kept only for compatibility.

The active CIMA Demonstrator runtime no longer dispatches tools.  This module
survives only because a few value objects and legacy-compatibility tests still
refer to the ToolCall/ToolResult data shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """Serialized tool invocation shape retained for compatibility."""

    tool_call_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolArtifact:
    """Per-item artifact payload retained for compatibility."""

    url: str
    success: bool
    raw_content: str
    canonical_url: str | None = None
    error_message: str | None = None
    evidence_context: str | None = None
    persisted_citem_id: str | None = None


@dataclass
class ToolResult:
    """Serialized tool result shape retained for compatibility."""

    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    summary: str
    error_message: str | None = None
    persisted_citem_id: str | None = None
    state_changed: bool = True
    image_data: str | None = None
    evidence_context: str | None = None
    canonical_url: str | None = None
    artifacts: list[ToolArtifact] = field(default_factory=list)
