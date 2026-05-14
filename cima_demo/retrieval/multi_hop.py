"""MultiHopAnalyzer — LLM-driven hop-depth estimation and query decomposition.

Design rationale:
- Hop depth cannot be reliably determined by rule-based heuristics: the same
  surface structure ("the X that Y") can represent 1 or 3 hops depending on
  what X and Y denote semantically.
- A single LLM complete() call simultaneously estimates hop_depth AND produces
  the decomposition, avoiding two round-trips.
- Called only when QueryPlanner classifies the query as MULTI_HOP, so the
  overhead is limited to queries that already show structural multi-hop signals.
- Falls back to [query] on any failure, preserving the 2-hop retrieve() path
  (SP-INV-01 pattern: silent degradation, never hard-fails retrieval).
"""
from __future__ import annotations

import json
import logging
import math

from cima_demo.domain.entities import LLMMessage
from cima_demo.domain.ports import LLMPort
from cima_demo.domain.value_objects import MultiHopAnalysis

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a query complexity analyzer for a retrieval-augmented memory system.

Your task: analyze a question and determine how many *reasoning hops* it requires,
where one hop is one traversal of a relationship between two entities or concepts.

Examples of hop counts:
- "Who is Alice?" → 1 hop
- "What did Alice's team build?" → 2 hops  (Alice→team, team→artifact)
- "What was the impact of what Alice's team built?" → 3 hops  (Alice→team→artifact→impact)
- "Who led the team that built the service that caused the incident that affected the client?" → 4 hops

If hop_depth > 2, decompose the question into sub-questions covering ≤2 hops each.
Sub-questions must:
  1. Be self-contained and independently answerable from a memory store.
  2. Be ordered so that earlier answers provide context for later ones.
  3. Together cover all concepts in the original question.

Respond with ONLY valid JSON — no explanation, no markdown fences:
{
  "hop_depth": <integer 1-8>,
  "sub_questions": [<string>, ...]
}

When hop_depth ≤ 2, return "sub_questions": [].
"""

_USER_TEMPLATE = 'Question: "{query}"'

# ── Structured-output schema ──────────────────────────────────────────────────
_MULTI_HOP_RESPONSE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "multi_hop_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "hop_depth":     {"type": "integer"},
                "sub_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["hop_depth", "sub_questions"],
            "additionalProperties": False,
        },
    },
}


class MultiHopAnalyzer:
    """LLM-based multi-hop query analyzer (single complete() call).

    Returns a MultiHopAnalysis with:
    - hop_depth: semantically accurate hop count (not rule-based)
    - sub_questions: decomposed ≤2-hop sub-questions (empty when hop_depth ≤ 2)

    On any failure: returns MultiHopAnalysis(hop_depth=2, sub_questions=[])
    so the caller falls back to the standard 2-hop retrieve() path.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def analyze(self, query: str) -> MultiHopAnalysis:
        """Estimate hop depth and decompose if N > 2. Never raises."""
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=_USER_TEMPLATE.format(query=query)),
        ]
        try:
            raw = await self._llm.complete(
                messages,
                temperature=0.0,
                max_tokens=512,
                response_format=_MULTI_HOP_RESPONSE_FORMAT,
            )
            return self._parse(raw, query)
        except Exception as exc:
            log.warning("MultiHopAnalyzer LLM call failed (%s) — degrading to 2-hop", exc)
            return MultiHopAnalysis(hop_depth=2, sub_questions=[])

    def _parse(self, raw: str, query: str) -> MultiHopAnalysis:
        """Parse LLM JSON response. Falls back on any parse error."""
        text = raw.strip()
        # Strip markdown code fences if the model adds them
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:]).rstrip("`").strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("MultiHopAnalyzer: JSON parse failed (%s) raw=%r", exc, raw[:200])
            return MultiHopAnalysis(hop_depth=2, sub_questions=[])

        hop_depth = int(data.get("hop_depth", 2))
        hop_depth = max(1, min(hop_depth, 8))

        raw_sub = data.get("sub_questions", [])
        if not isinstance(raw_sub, list):
            raw_sub = []

        sub_questions = [str(q).strip() for q in raw_sub if q and str(q).strip()]

        # Validate: if N > 2 but decomposition is missing/empty, degrade gracefully
        if hop_depth > 2 and len(sub_questions) < 2:
            log.warning(
                "MultiHopAnalyzer: hop_depth=%d but only %d sub-questions returned "
                "— falling back to single 2-hop retrieve",
                hop_depth, len(sub_questions),
            )
            return MultiHopAnalysis(hop_depth=hop_depth, sub_questions=[])

        # Enforce the core constraint: N sub-questions = ceil(hop_depth / 2)
        expected_n = math.ceil(hop_depth / 2)
        if len(sub_questions) > expected_n * 2:
            # Truncate excess sub-questions (LLM over-decomposed)
            log.debug(
                "MultiHopAnalyzer: trimming %d → %d sub-questions",
                len(sub_questions), expected_n,
            )
            sub_questions = sub_questions[:expected_n]

        return MultiHopAnalysis(hop_depth=hop_depth, sub_questions=sub_questions)
