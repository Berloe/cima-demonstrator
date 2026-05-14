"""Per-turn dead-end and duplicate detection for tool calls (extracted from engine.py)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cima_demo.tools.registry import ToolCall


@dataclass
class ToolStateGuard:
    """Per-turn dead-end and duplicate detection for tool calls.

    Tracks which queries have been tried for each tool and how many times
    a tool returned STATE:UNCHANGED.  Prevents the model from looping on
    the same dead-end query without making progress.

    exhausted_tools: tools hard-blocked after exceeding their dead-end cap.
    Once exhausted, any call to that tool is rejected regardless of query.
    """
    _tried: dict[str, set[str]] = field(default_factory=dict)
    _dead_end_count: dict[str, int] = field(default_factory=dict)
    exhausted_tools: set[str] = field(default_factory=set)

    # Maximum dead-ends per tool before the tool is hard-blocked for this turn.
    # Primary stop is stall detection — these caps are a last-resort safety valve.
    # memory(action=search) accepts batch queries: one UNCHANGED = source exhausted in 1 call.
    # web covers search+fetch+render variants across multiple queries and URLs.
    _DEAD_END_CAPS: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._DEAD_END_CAPS:
            self._DEAD_END_CAPS = {
                "memory": 2,   # batch queries: allow 2 UNCHANGED before blocking
                "web":    6,   # search+fetch+render across multiple attempts before blocking
            }

    def is_exhausted(self, tool_name: str) -> bool:
        """Return True if this tool has been hard-blocked for the current turn."""
        return tool_name in self.exhausted_tools

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize a query key for duplicate detection."""
        return " ".join(query.lower().split())[:200]

    def is_duplicate(self, tool_name: str, query_key: str) -> bool:
        """Return True if this exact query was already issued for this tool.

        Side-effect: registers the query if it is new.
        """
        key = self._normalize(query_key)
        tried = self._tried.setdefault(tool_name, set())
        if key in tried:
            return True
        tried.add(key)
        return False

    def record_dead_end(self, tool_name: str) -> int:
        """Increment dead-end counter for tool; return new count.

        When the count reaches the cap for this tool, it is added to
        exhausted_tools and will be hard-blocked on any further call.
        """
        count = self._dead_end_count.get(tool_name, 0) + 1
        self._dead_end_count[tool_name] = count
        cap = self._DEAD_END_CAPS.get(tool_name)
        if cap is not None and count >= cap:
            self.exhausted_tools.add(tool_name)
        return count

    def dead_end_count(self, tool_name: str) -> int:
        return self._dead_end_count.get(tool_name, 0)


def _get_query_key(tool_call: "ToolCall") -> str:
    """Extract the canonical query key for deduplication.

    For macro-tools (memory/web), prefixes the action so that different actions
    on the same tool don't collide.  For batch queries, sorts for order-insensitivity.
    """
    p = tool_call.params
    name = tool_call.name

    # Macro-tools: prefix with action for unambiguous dedup
    if name in ("memory", "web"):
        action = p.get("action", "")
        if "queries" in p:
            qs = sorted(str(q).lower().strip() for q in p["queries"] if q)
            return f"{action}:{'|'.join(qs)}"
        if "url" in p:
            return f"{action}:{p['url']}"
        return f"{action}:{json.dumps({k: v for k, v in p.items() if k != 'action'}, sort_keys=True)}"

    # Single-action tools
    if "queries" in p:
        qs = sorted(str(q).lower().strip() for q in p["queries"] if q)
        return "|".join(qs)
    if "query" in p:
        return str(p["query"])
    if "url" in p:
        return str(p["url"])
    return json.dumps(p, sort_keys=True)
