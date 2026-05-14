"""SummaryService — context refresh and L1/L2 summarization."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cima_demo.application.stream_manager import StreamManager

from cima_demo.domain.entities import (
    CItem,
    ContextView,
    SummaryNode,
    TaskMemory,
)
from cima_demo.domain.operations import (
    compute_cvs,
    compute_recency_score,
)
from cima_demo.domain.ports import (
    CItemStorePort,
    LLMPort,
    RelDBPort,
)
from cima_demo.domain.value_objects import (
    ForgetParams,
    ItemType,
)

log = logging.getLogger(__name__)

# A-10: maximum L1 summary nodes before L2 AutoPromote (deferred)
_L1_MAX_NODES = 20

# A-4: H_lex failure tracking per conversation — capped at last 20 summarizations
# to compute failure_rate and escalate from WARNING → ERROR when >20%.
_hlex_outcomes: dict[str, list[bool]] = {}  # conversation_id → [passed, ...]

# CCP: type-boost table for eviction ordering.
# Higher value → higher effective CVS → evicted LATER.
# DECISION/CONSTRAINT are listed for reference but handled separately (never evicted).
_EVICTION_TYPE_BOOST: dict[str, float] = {
    ItemType.DECISION:    float("inf"),   # protected — should never reach eviction
    ItemType.CONSTRAINT:  float("inf"),   # protected
    ItemType.FACT:        0.8,
    ItemType.DERIVED:     0.8,            # traceable computation — same durability as FACT
    ItemType.HYPOTHESIS:  0.6,
    ItemType.OBSERVATION: 0.4,            # first evicted
    ItemType.ASSUMPTION:  0.2,            # temporary — evicted early
}
_EVICTION_TYPE_BOOST_DEFAULT = 0.5

# Tool actors whose results may appear as OBSERVATIONs (lower boost if low importance)
_TOOL_ACTORS = frozenset({"web", "memory", "compute", "workspace"})


class SummaryService:
    """Context refresh and L1/L2 summarization.

    Extracted from MemoryService to own:
    refresh_context, _promote_l2, and all internal summary-building helpers.
    """

    def __init__(
        self,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        llm_port: LLMPort,
        stream_manager: "StreamManager",
        forget_params: ForgetParams | None = None,
        lineage_service: Any | None = None,
    ) -> None:
        self._db = rel_db
        self._cstore = citem_store
        self._llm = llm_port
        self._stream = stream_manager
        self._forget_params = forget_params or ForgetParams.default()
        self._lineage = lineage_service

    # ── Context refresh ───────────────────────────────────────────────────────

    async def refresh_context(
        self,
        context_view: ContextView,
        task_memory: TaskMemory,
        conversation_id: str,
        current_goal: str | None = None,
        active_step: str | None = None,
        phase: str | None = None,
        semantic: bool = False,
        force_ids: set[str] | None = None,
    ) -> tuple[ContextView, int, str]:
        """Compress active C-Items via intentional L1 or L2 refresh (CCP v1.0).

        Two-tier operation:
          L1 (default, semantic=False): extractive metadata summary — zero LLM cost.
            Generates a structured index of what was archived. Satisfies CIMA A-4
            (H_lex always reduced: index is shorter than originals).
          L2 (semantic=True): LLM complete() — task-aware semantic summary.
            Used only when explicitly requested by the model via focus_context tool
            or at plan-step/SYNTHESIS milestones. Cost: ~600s on iGPU.

        force_ids: when non-empty, bypass eligibility filtering and force L1 over
          exactly those C-Item IDs (BCM T2 — forced L1 on this-turn evidence).
          DECISION/CONSTRAINT protection still applies.

        Protection (APP-INV-10, A-7.3):
          DECISION and CONSTRAINT items are never included in to_summarize.

        Eviction ordering (CCP type-boost):
          Effective CVS = base_CVS × type_boost, so low-importance OBSERVATIONs
          (including tool results) are evicted first; FACTs next; DECISION/CONSTRAINT
          never evicted.

        Returns (context_view_unchanged, items_summarized, summary_content).
        items_summarized=0 → no compression (OC-03: caller must not emit CONTEXT_REFRESH).
        """
        import time as _time
        _now = _time.time()

        all_active = await self._cstore.fetch_by_conversation(
            conversation_id, scope_status="active"
        )
        if not all_active:
            return context_view, 0, ""

        # Separate protected items (DECISION, CONSTRAINT) — never evicted (APP-INV-10)
        protected_types = frozenset({ItemType.DECISION, ItemType.CONSTRAINT})
        evictable = [c for c in all_active if c.item_type not in protected_types]

        if not evictable:
            return context_view, 0, ""

        if force_ids:
            # BCM T2: bypass eligibility, summarize exactly these C-Items.
            # DECISION/CONSTRAINT protection still applies — filtered above.
            _id_set = set(force_ids)
            to_summarize = [c for c in evictable if c.citem_id in _id_set][:_L1_MAX_NODES]
        else:
            def _effective_cvs(item: CItem) -> float:
                """CVS × type_boost — lower value = evicted first."""
                _ca_unix = (
                    item.created_at.timestamp()
                    if hasattr(item.created_at, "timestamp")
                    else float(item.created_at)
                )
                recency = compute_recency_score(_ca_unix, _now)
                base = compute_cvs(
                    content_relevance=item.importance,
                    recency_score=recency,
                    novelty_score=0.0,
                )
                boost = _EVICTION_TYPE_BOOST.get(item.item_type, _EVICTION_TYPE_BOOST_DEFAULT)
                # Extra penalty for low-importance tool results (ephemeral web content)
                if item.actor in _TOOL_ACTORS and item.importance < 0.3:
                    boost *= 0.5
                return base * boost

            sorted_evictable = sorted(evictable, key=lambda x: (_effective_cvs(x), x.created_at))
            to_summarize = sorted_evictable[:_L1_MAX_NODES]

        if not to_summarize:
            return context_view, 0, ""

        # ── Build summary content ─────────────────────────────────────────────

        if semantic:
            # L2 — LLM complete(): task-aware semantic summary (~600s on iGPU)
            items_text = "\n".join(
                f"- [{item.item_type}|{item.actor}] {item.content}"
                for item in to_summarize
            )
            _goal_line = f"Current task: {current_goal}" if current_goal else ""
            _step_line = f"Active plan step: {active_step}" if active_step else ""
            _phase_line = f"Phase: {phase}" if phase else ""
            _context_block = "\n".join(filter(None, [_goal_line, _step_line, _phase_line]))
            from cima_demo.domain.entities import LLMMessage
            summary_resp = await self._llm.complete(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are compressing memory context for task continuity.\n"
                            + (_context_block + "\n" if _context_block else "")
                            + "\nInstructions:\n"
                            "1. Preserve ALL facts relevant to the current task.\n"
                            "2. For each tool result: state what was found (1 sentence).\n"
                            "3. Note completed sub-tasks: \"Completed: [what] → [result]\".\n"
                            "4. Note open actions: \"Pending: [...]\".\n"
                            "5. End with: \"Status: [current task state]\"\n"
                            "Produce one concise paragraph. Keep numbers, names, dates exact."
                        ),
                    ),
                    LLMMessage(role="user", content=items_text),
                ],
                temperature=0.3,
                max_tokens=300,
            )
            if not summary_resp.strip():
                return context_view, 0, ""
            summary_content = summary_resp.strip()

            # A-4: H_lex proxy check (L2 only — L1 is always shorter by construction)
            def _h_lex(text: str) -> float:
                words = text.lower().split()
                return len(set(words)) / max(len(words), 1)

            h_before = _h_lex(items_text)
            h_after  = _h_lex(summary_content)
            _passed = h_after < h_before
            _outcomes = _hlex_outcomes.get(conversation_id, [])
            _outcomes = (_outcomes[-19:]) + [_passed]
            _hlex_outcomes[conversation_id] = _outcomes
            _failure_rate = _outcomes.count(False) / len(_outcomes)
            if not _passed:
                _log_fn = log.error if _failure_rate > 0.20 else log.warning
                _log_fn(
                    "refresh_context L2: H_lex not reduced (before=%.3f after=%.3f) "
                    "failure_rate=%.0f%% conv=%s",
                    h_before, h_after, _failure_rate * 100, conversation_id,
                )

        else:
            # L1 — extractive metadata summary: zero LLM cost, always H_lex-compliant.
            # Groups items by type and actor; preserves key counts and date range.
            from collections import Counter
            type_counts: Counter[str] = Counter(i.item_type for i in to_summarize)
            actor_tools = [i.actor for i in to_summarize if i.actor in _TOOL_ACTORS]
            tool_summary = f" ({', '.join(sorted(set(actor_tools)))})" if actor_tools else ""
            dates = [
                (i.created_at.timestamp() if hasattr(i.created_at, "timestamp") else float(i.created_at))
                for i in to_summarize
            ]
            _fmt = "%Y-%m-%d %H:%M"
            date_range = (
                f"{datetime.fromtimestamp(min(dates), tz=UTC).strftime(_fmt)} → "
                f"{datetime.fromtimestamp(max(dates), tz=UTC).strftime(_fmt)}"
            ) if dates else ""
            type_lines = "  · ".join(
                f"{count} {itype}{tool_summary if itype == ItemType.OBSERVATION else ''}"
                for itype, count in sorted(type_counts.items())
            )
            goal_hint = f"Focus: {current_goal}" if current_goal else ""
            parts = [
                f"[Compressed {len(to_summarize)} items — {date_range}]",
                f"  · {type_lines}" if type_lines else "",
                goal_hint,
            ]
            summary_content = "\n".join(p for p in parts if p)

        # ── Persist SummaryNode (trazabilidad A-9, INV-08) ───────────────────
        token_count = max(1, len(summary_content) // 4)
        node = SummaryNode(
            conversation_id=conversation_id,
            level=1,
            content=summary_content,
            token_count=token_count,
            origin_citem_ids=[item.citem_id for item in to_summarize],
        )
        await self._db.save_summary(node)
        if self._lineage is not None:
            try:
                await self._lineage.record_summary_resolution(
                    conversation_id=conversation_id,
                    summary_id=node.node_id,
                    summary_text=summary_content,
                    origin_citem_ids=[item.citem_id for item in to_summarize],
                    metadata={"level": 1, "semantic": semantic},
                )
            except Exception:
                log.exception("demo summary resolution failed for %s", node.node_id)

        # Archive in parallel (fire-and-forget gather — non-blocking)
        async def _archive_one(item: CItem) -> None:
            await self._cstore.update_field(item.citem_id, "scope_status", "archived")
            await self._cstore.update_field(item.citem_id, "summarized_by_node_id", node.node_id)

        await asyncio.gather(*(_archive_one(item) for item in to_summarize), return_exceptions=True)

        log.info(
            "refresh_context %s — archived=%d tier=%s node=%s conv=%s",
            "L2(semantic)" if semantic else "L1(extractive)",
            len(to_summarize), "L2" if semantic else "L1",
            node.node_id[:8], conversation_id,
        )

        # A-10: check if parentless L1 node count now exceeds N_max_level → L2 AutoPromote
        try:
            parentless_l1 = await self._db.fetch_nodes_at_level(
                level=1, conversation_id=conversation_id, parentless_only=True,
            )
            if len(parentless_l1) >= _L1_MAX_NODES:
                await self._promote_l2(conversation_id, parentless_l1)
        except Exception:
            log.debug("A-10 L2 check failed (non-fatal) for %s", conversation_id, exc_info=True)

        return context_view, len(to_summarize), summary_content

    # ── Summary pyramid L2 AutoPromote ───────────────────────────────────────

    async def trigger_l2_check(self, conversation_id: str) -> bool:
        """Public entry-point for A-10 L2 AutoPromote check (SPEC-6).

        Checks whether parentless L1 node count >= _L1_MAX_NODES and, if so,
        promotes them into an L2 node.  Called by LifecycleService.run_forget_cycle
        so the promotion fires on the background forget worker schedule, not only
        when refresh_context() is explicitly invoked.

        Returns True if a promotion was triggered, False otherwise.
        """
        try:
            parentless_l1 = await self._db.fetch_nodes_at_level(
                level=1, conversation_id=conversation_id, parentless_only=True,
            )
            if len(parentless_l1) >= _L1_MAX_NODES:
                await self._promote_l2(conversation_id, parentless_l1)
                return True
        except Exception:
            log.debug("trigger_l2_check failed (non-fatal) for %s", conversation_id, exc_info=True)
        return False

    async def _promote_l2(
        self,
        conversation_id: str,
        parentless_l1: list[SummaryNode],
    ) -> None:
        """A-10 L2 AutoPromote: summarize parentless L1 nodes into a single L2 node.

        Called when parentless L1 node count reaches _L1_MAX_NODES.
        The L2 node is the new pyramid top; L1 nodes are linked to it via parent_ids.
        fetch_pyramid_tops() will return L2 (parentless, level=2) instead of the
        now-parented L1 nodes — compression is transparent to the context builder.
        """
        from cima_demo.domain.entities import LLMMessage

        items_text = "\n".join(f"- {node.content}" for node in parentless_l1)
        try:
            summary_resp = await self._llm.complete(
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are summarizing a session memory pyramid. "
                            "Compress the following L1 summaries into a single coherent paragraph "
                            "preserving the most important facts, decisions, and conclusions."
                        ),
                    ),
                    LLMMessage(role="user", content=items_text),
                ],
                temperature=0.3,
                max_tokens=512,
            )
        except Exception as exc:
            log.warning("_promote_l2: LLM complete failed for %s: %s", conversation_id, exc)
            return

        if not summary_resp.strip():
            return

        try:
            token_count = await self._llm.count_tokens(summary_resp)
        except Exception:
            token_count = max(1, len(summary_resp) // 4)

        child_summary_ids = [node.node_id for node in parentless_l1]

        l2_node = SummaryNode(
            conversation_id=conversation_id,
            level=2,
            content=summary_resp,
            token_count=token_count,
            # CIMA lineage is stored only between adjacent levels:
            # L2 -> L1, and each L1 -> L0. Do not persist a transitive
            # L2 -> all L0 closure in origin_citem_ids.
            origin_citem_ids=[],
        )
        await self._db.save_summary(l2_node)
        if self._lineage is not None:
            try:
                await self._lineage.record_summary_resolution(
                    conversation_id=conversation_id,
                    summary_id=l2_node.node_id,
                    summary_text=summary_resp,
                    origin_citem_ids=[],
                    metadata={
                        "level": 2,
                        "child_summary_ids": child_summary_ids,
                        "lineage_policy": "direct_adjacent_levels_only",
                    },
                )
                for child_id in child_summary_ids:
                    await self._db.save_demo_lineage_edge({
                        "edge_id": str(__import__("uuid").uuid4()),
                        "conversation_id": conversation_id,
                        "src_kind": "summary",
                        "src_id": l2_node.node_id,
                        "dst_kind": "summary",
                        "dst_id": child_id,
                        "relation": "SUMMARIZES_LEVEL",
                        "metadata": {
                            "parent_level": 2,
                            "child_level": 1,
                            "lineage_policy": "direct_adjacent_levels_only",
                        },
                    })
            except Exception:
                log.exception("demo summary resolution failed for L2 %s", l2_node.node_id)

        # Link all L1 nodes to the new L2 parent
        for node in parentless_l1:
            await self._db.set_summary_parent(node.node_id, l2_node.node_id)

        log.info(
            "A-10 L2 AutoPromote: %d L1 nodes → L2 node %s for %s",
            len(parentless_l1), l2_node.node_id, conversation_id,
        )
