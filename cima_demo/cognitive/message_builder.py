"""MessageBuilder — constructs LLM message lists for one cognitive iteration.

This module remains only for lightweight prompt-block helpers reused by the
demonstrator runtime.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cima_demo.cognitive.tool_state_guard import _get_query_key
from cima_demo.domain.entities import ContextView, LLMMessage
from cima_demo.domain.value_objects import InputState

if TYPE_CHECKING:
    from cima_demo.domain.entities import Plan
    from cima_demo.cognitive.kernel.state import TurnRuntime

log = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 0  # 0 = no default cap; callers pass budget-derived limits

# persisted_citem_id values that mean the content is NOT in the citem store.
# When persisted_citem_id is anything else (a real UUID, "MULTI_URL", "DEDUP",
# "CROSS_CACHE", …) the full evidence body is in Qdrant and will return via
# retrieval — so the tool message only needs the compact header to avoid
# doubling context pressure.
# H-07 (SPEC-4): "CROSS_CACHE" is intentionally absent so cross-turn cache
# hits render with the same header-only path as same-turn "DEDUP" hits.
_NOT_IN_STORE: frozenset[str] = frozenset({"FAILED", "EMPTY", "NO_CITEM", "LOCATOR"})


def _resolve_output_contract(ts: "TurnRuntime") -> Any | None:
    """Return the best available canonical output contract for the turn."""
    for owner_name in ("task_plan", "task_spec", "task_state"):
        owner = getattr(ts, owner_name, None)
        contract = getattr(owner, "output_contract", None) if owner is not None else None
        if contract is not None:
            return contract
    return getattr(ts, "output_contract", None)


def _render_output_contract_block(contract: Any | None) -> str:
    """Compact prompt block describing the canonical final output shape.

    OutputContract is authoritative when present. Legacy fields such as
    answer_format/answer_unit still exist for backward compatibility, but the
    answer turn should anchor itself on this block to avoid degrading scale,
    rounding, or representation across replans.
    """
    if contract is None:
        return ""

    def _fmt(value: Any) -> str:
        return "None" if value is None else str(value)

    fields = {
        "format": getattr(contract, "format", None),
        "representation": getattr(contract, "representation", None),
        "base_unit": getattr(contract, "base_unit", None),
        "display_scale": getattr(contract, "display_scale", None),
        "rounding_rule": getattr(contract, "rounding_rule", None),
        "precision": getattr(contract, "precision", None),
    }
    non_null = {k: v for k, v in fields.items() if v is not None}
    if not non_null and not getattr(contract, "required_evidence", False):
        return ""
    lines = ["Output contract:"]
    for k, v in non_null.items():
        lines.append(f"  {k}: {v}")
    if getattr(contract, "required_evidence", False):
        lines.append("  required_evidence: true")
    return "\n".join(lines)


def _is_obsolete_compute_gap(gap: str) -> bool:
    """Heuristic filter for stale compute-related readiness gaps."""
    gap_text = " ".join(gap.lower().split())
    markers = (
        "compute",
        "calculation",
        "numeric result",
        "verified numeric",
        "final result",
        "output contract",
        "rounding",
        "display scale",
        "representation",
        "slot",
    )
    return any(marker in gap_text for marker in markers)


def _visible_validation_gaps(ts: "TurnRuntime") -> list[str]:
    gaps = list(getattr(ts, "last_validation_gaps", []) or [])
    if not gaps:
        return []
    if getattr(ts, "has_final_compute_result", False):
        filtered = [gap for gap in gaps if not _is_obsolete_compute_gap(gap)]
        if filtered:
            return filtered
        return []
    # de-duplicate while preserving order for prompt stability
    seen: set[str] = set()
    ordered: list[str] = []
    for gap in gaps:
        if gap not in seen:
            seen.add(gap)
            ordered.append(gap)
    return ordered


def _render_slot_states(task_state: Any) -> str:
    """Compact '## Required Input States' body — one line per slot.

    Rendered regardless of whether a task_plan is also present (PR2 fix:
    task_plan guard previously hid this section entirely).
    """
    lines: list[str] = []
    for s in task_state.slots:
        state = s.state.value if hasattr(s.state, "value") else str(s.state)
        if s.state == InputState.VERIFIED:
            val = str(s.value or "")
            if s.unit:
                val = f"{val} {s.unit}".strip()
            lines.append(f"  [{state}] {s.name} = {val}")
        elif s.state == InputState.LOCATED:
            loc = f" → {s.locator_refs[0]}" if s.locator_refs else ""
            lines.append(f"  [{state}] {s.name}: {s.description}{loc}")
        else:
            src = f" | source={s.required_source}" if s.required_source else ""
            lines.append(f"  [MISSING] {s.name}: {s.description}{src}")
    return "\n".join(lines)


class MessageBuilder:
    """Assembles the LLM message list for one cognitive iteration.

    Pure-ish: no async I/O. All external state passed as arguments.
    Depends only on prompt_factory and llm_vision — injected at construction.
    """

    def __init__(
        self,
        prompt_factory: Callable[..., str],
        llm_vision: bool,
        max_tool_result_chars: int | None = None,
    ) -> None:
        self._prompt_factory = prompt_factory
        self._llm_vision = llm_vision
        self._max_tool_result_chars = max_tool_result_chars

    def derive_query(self, ts: "TurnRuntime", plan: "Plan | None") -> str:
        """Build retrieval query from turn context.

        When a plan has an active step, narrows the query to the step focus.
        context_focus (if set) is a tighter semantic cluster than description —
        used when the description is imperative ("fetch page X") and a noun-phrase
        focus ("X publication date") is more useful for embedding search.
        """
        if plan is not None and plan.active_step is not None:
            step = plan.active_step
            focus = step.context_focus or step.description
            return f"{ts.user_message} {focus}"
        if ts.tool_results_accumulated:
            last = ts.tool_results_accumulated[-1]
            return f"{ts.user_message} {last.summary}"
        return ts.user_message

    def refresh_query(self, ts: "TurnRuntime", plan: "Plan | None") -> str:
        """Query to use when rebuilding context after a refresh.

        CCP: uses the active plan step focus when available — the most
        specific representation of what's being worked on right now.  Falls
        back to the user message so retrieval stays grounded in the original ask.
        """
        if plan is not None and plan.active_step is not None:
            step = plan.active_step
            return step.context_focus or step.description
        return ts.user_message

    def build(
        self,
        ts: "TurnRuntime",
        history: list[LLMMessage],
        context_view: ContextView,
        synthesis: bool = False,
        include_external: bool = True,
        plan: "Plan | None" = None,
        tool_defs: "list[dict[str, Any]] | None" = None,
    ) -> list[LLMMessage]:
        """Build the LLM message list for one cognitive iteration.

        Chronological layout (invariant):
          [0]  system            — stable instructions, rules, tool specs. NO evidence.
          [1…] history           — prior turns from DB, budget-trimmed. Chronological order.
          [n]  user (current)    — single message with two labelled blocks:
                                     <memory_context source="system"> … </memory_context>
                                     <user_message> … </user_message>
                                   memory_context = RAG-retrieved evidence (system-injected,
                                   NOT user input). user_message = verbatim user text.
          [n+1…] assistant/tool  — accumulated tool call/result pairs from this turn only.

        Tool result content (never None):
          failure              → compact error string
          success + evidence_context → evidence_context (web fetch/render summary)
          success + no evidence_context → result.content[:max_tool_result_chars]
          success + no content → "[OK: tool(query) — no content]"
        """
        messages: list[LLMMessage] = []

        # ── Resolve strategy section ──────────────────────────────────────────
        strategy_section: str | None = None
        if ts.strategy_ctx is not None and ts.strategy_ctx.system_prompt_suffix:
            strategy_section = ts.strategy_ctx.system_prompt_suffix

        # ── Derive allowed_tools from provided schemas ────────────────────────
        # Synthesis passes have no tools; tool passes expose only mode-allowed schemas.
        allowed_tools: list[str] | None = None
        if not synthesis and tool_defs:
            allowed_tools = [
                d["function"]["name"]
                for d in tool_defs
                if d.get("function", {}).get("name")
            ]

        # ── Build memory context body ─────────────────────────────────────────
        context_text = context_view.text if hasattr(context_view, "text") else ""
        if context_text.strip():
            context_body = context_text.strip()
        elif history:
            context_body = "(Memory context not yet indexed. Refer to conversation history above.)"
        else:
            context_body = "(No memory context for this query.)"

        # Inject the exact closed citation set.  A numeric range is not enough:
        # summaries/perspective markers can introduce gaps or P# markers, and models
        # tend to extrapolate unseen marker numbers under range-style instructions.
        _ctx_markers = sorted(
            {str(it.get("marker")) for it in (context_view.items if hasattr(context_view, "items") else []) if it.get("marker")},
            key=lambda m: (m[0], int(m[1:]) if m[1:].isdigit() else 0),
        )
        if _ctx_markers:
            _marker_list = ", ".join(f"[{m}]" for m in _ctx_markers)
            context_body += (
                "\n\n## CITATION CONSTRAINT\n"
                f"Allowed citation markers are exactly and only: {_marker_list}. "
                "Use only these exact bracketed markers; do not infer additional markers from the sequence. "
                "If no listed marker supports a claim, omit that claim rather than inventing or substituting a marker."
            )

        if ts.global_drift_detected:
            context_body += (
                f"\n\n## ⚠ Objective Alignment Check\n"
                f"Original request: {ts.user_message}\n"
                f"The current context does not fully cover the original objective. "
                f"Before continuing, verify that your current approach still serves "
                f"the original request. If it has diverged, correct course explicitly."
            )

        # P3: single ## Task section — consolidate plan/spec/state + slots + contract.
        _task_lines: list[str] = []
        if ts.task_plan is not None:
            _task_lines.append(ts.task_plan.context_summary())
        elif ts.task_spec is not None:
            _task_lines.append(ts.task_spec.context_summary())
        elif ts.task_state is not None:
            _task_lines.append(ts.task_state.context_summary())

        if ts.task_state is not None and ts.task_state.slots:
            _task_lines.append("Input states:\n" + _render_slot_states(ts.task_state))

        _output_contract = _resolve_output_contract(ts)
        if _output_contract is not None:
            _contract_block = _render_output_contract_block(_output_contract).strip()
            if _contract_block:
                _task_lines.append(_contract_block)

        if _task_lines:
            context_body += "\n\n## Task\n" + "\n".join(_task_lines)

        if ts.evidence_register is not None and not ts.evidence_register.is_empty:
            context_body += f"\n\n## Evidence Gathered\n{ts.evidence_register.context_summary()}"

        _visible_gaps = _visible_validation_gaps(ts) if synthesis else []
        if synthesis and _visible_gaps:
            gaps_text = "\n".join(f"  - {g}" for g in _visible_gaps)
            context_body += (
                f"\n\n## Unresolved Gaps\n"
                f"The following could not be resolved despite all retrieval attempts:\n"
                f"{gaps_text}\n"
                f"If you cannot produce a complete answer, state clearly what was found "
                f"and what is still missing — do not fabricate."
            )

        # ── Inject compute results for synthesis ──────────────────────────────
        # Only FinalComputeResult is considered verified. Legacy outputs and
        # traces remain visible as intermediate context when needed, but they
        # must not leak into the stronger "Verified Compute Results" section.
        if synthesis and ts.has_final_compute_result and ts.final_compute_result is not None:
            _final_lines = [f"Final: {ts.final_compute_result.rendered_value}"]
            if ts.final_compute_result.source_expression:
                _final_lines.append(f"Source expression: {ts.final_compute_result.source_expression}")
            context_body += (
                "\n\n## Verified Compute Results\n"
                "Only these results satisfy the required output contract. Use them directly in the answer. "
                "Do not recalculate, rescale, or replace them with intermediate arithmetic.\n"
                + "\n".join(_final_lines)
            )
        elif synthesis and ts.compute_traces:
            _trace_lines: list[str] = []
            for _ci, _trace in enumerate(ts.compute_traces, start=1):
                _trace_lines.append(f"C{_ci}: {_trace.rendered_value}")
            context_body += (
                "\n\n## Intermediate Compute Traces\n"
                "These compute outputs are not yet verified final results. "
                "Do not present them as the final answer unless a verified compute result exists.\n"
                + "\n".join(_trace_lines)
            )
        elif synthesis and ts.compute_done and ts.compute_outputs:
            _compute_lines: list[str] = []
            for _ci, (_val, _) in enumerate(ts.compute_outputs, start=1):
                _compute_lines.append(f"C{_ci}: {_val}")
            context_body += (
                "\n\n## Intermediate Compute Outputs (Legacy)\n"
                "These legacy compute outputs are unverified. Do not present them as the final answer unless a "
                "verified compute result exists.\n"
                + "\n".join(_compute_lines)
            )
        # ── [0] System message ────────────────────────────────────────────────
        sys_prompt = self._prompt_factory(strategy_section, include_external, synthesis, allowed_tools)
        messages.append(LLMMessage(role="system", content=sys_prompt))

        # ── [1…] History (prior turns) ────────────────────────────────────────
        messages.extend(history)

        # ── [n] Current user turn — single role="user" message ───────────────
        # Active search locators injected between memory_context and user_message
        _locators_block = ""
        if ts.search_result_locators:
            _loc_sections = ts.search_result_locators[-3:]
            _locators_block = (
                '<search_locators source="this_turn">\n'
                + "\n---\n".join(_loc_sections)
                + "\n</search_locators>\n\n"
            )

        # Active step block — placed OUTSIDE <memory_context> so it reads as a
        # direct protocol directive, not as background context. This gives it
        # higher attention weight and makes rule 2 ("Follow the active step")
        # from the system prompt immediately actionable.
        _active_step_block = ""
        if plan is not None and plan.active_step is not None:
            _step = plan.active_step
            _step_lines = ["## Current Step", f"Active step: {_step.description}"]
            if _step.acceptance_criterion:
                _step_lines.append(f"Done when: {_step.acceptance_criterion}")
            _active_step_block = "\n".join(_step_lines) + "\n\n"

        # Prose violation correction — injected when the model has emitted
        # text in Mode A at least once this turn.  Surfaces as a protocol
        # reminder so the model self-corrects without requiring a new turn.
        _prose_correction = ""
        if not synthesis and getattr(ts, "prose_violation_iters", 0) > 0:
            _prose_correction = (
                "## PROTOCOL REMINDER\n"
                "Your previous response included text output in tool-call mode. "
                "Text output is discarded — only tool calls are processed. "
                "Emit only the tool call. No preamble, no explanation.\n\n"
            )

        user_turn_content = (
            '<memory_context source="system">\n'
            f"{context_body}\n"
            "</memory_context>\n\n"
            + _prose_correction
            + _active_step_block
            + _locators_block
            + "<user_message>\n"
            f"{ts.user_message}\n"
            "</user_message>"
        )
        messages.append(LLMMessage(role="user", content=user_turn_content))

        # ── [n+1…] Tool call / result pairs (this turn only) ─────────────────
        calls   = ts.tool_call_requests
        results = ts.tool_results_accumulated
        sizes   = ts.tool_batch_sizes if ts.tool_batch_sizes else [1] * len(calls)

        # H-08: build aligned batches first, then window by call count.
        # The old approach (start idx = len-10, iterate all sizes) produced
        # misaligned slices and silently dropped gate-rejection messages that
        # fall outside the 10-call window.
        # Fix: always include batches that contain failures (gate rejections)
        # so the model can recover; only truncate successful-only batches.
        _WINDOW_CALLS = 10
        _batches: list[tuple[list, list]] = []
        _bi = 0
        for _bs in sizes:
            _bc = calls[_bi : _bi + _bs]
            _br = results[_bi : _bi + _bs]
            _bi += _bs
            if _bc:
                _batches.append((_bc, _br))

        # Determine which batches to render: last-10-call window + any failures.
        _tail_start = max(0, len(calls) - _WINDOW_CALLS)
        _cumulative = 0
        _render: set[int] = set()
        for _i, (_bc, _br) in enumerate(_batches):
            _batch_start = _cumulative
            _cumulative += len(_bc)
            if _batch_start >= _tail_start:
                _render.add(_i)
            elif any(not r.success for r in _br):
                # Gate rejection outside window — always preserve so model can recover
                _render.add(_i)

        for _i, (batch_calls, batch_results) in enumerate(_batches):
            if _i not in _render:
                continue
            messages.append(LLMMessage(
                role="assistant",
                content="tools history",
                tool_calls=[{
                    "id": c.tool_call_id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.params)},
                } for c in batch_calls],
            ))
            for call, result in zip(batch_calls, batch_results):
                if not result.success:
                    if result.content and result.content.strip():
                        # Gate rejection or structured failure — show full content so model can recover
                        raw_fail = result.content
                        if self._max_tool_result_chars is not None and len(raw_fail) > self._max_tool_result_chars:
                            raw_fail = raw_fail[:self._max_tool_result_chars] + "\n…[truncated]"
                        tool_content: str = raw_fail
                    else:
                        error_hint = (result.error_message or result.summary or "unknown error")[:80]
                        tool_content = f"[FAILED: {call.name}({_get_query_key(call)}) → {error_hint}]"
                elif result.evidence_context:
                    _pid = result.persisted_citem_id
                    if _pid is not None and _pid not in _NOT_IN_STORE:
                        if result.artifacts:
                            # Multi-URL batch: show per-artifact status so the model
                            # knows exactly which URLs succeeded and which failed.
                            # Do NOT just show the first header — that hides failures.
                            _art_lines: list[str] = []
                            for _art in result.artifacts:
                                if not _art.success or _art.persisted_citem_id in ("FAILED",):
                                    _art_lines.append(
                                        f"### {_art.url}\n"
                                        f"[FAILED: {_art.error_message or 'fetch error'}]"
                                    )
                                elif _art.persisted_citem_id in ("EMPTY", "NO_CITEM"):
                                    _art_lines.append(
                                        f"### {_art.url}\n[EMPTY: no extractable content]"
                                    )
                                else:
                                    _art_lines.append(
                                        f"### {_art.url} [indexed — full evidence in memory_context]"
                                    )
                            tool_content = "\n\n".join(_art_lines)
                        else:
                            # Single-URL: body is in the citem store — show only header.
                            _ec = result.evidence_context
                            _header_end = _ec.find("\n[E")
                            _header = _ec[:_header_end] if _header_end > 0 else _ec.split("\n")[0]
                            tool_content = _header + " [indexed — full evidence in memory_context]"
                    elif self._max_tool_result_chars is not None and len(result.evidence_context) > self._max_tool_result_chars:
                        tool_content = result.evidence_context[:self._max_tool_result_chars] + "\n…[truncated]"
                    else:
                        tool_content = result.evidence_context
                elif result.content:
                    raw = result.content
                    if self._max_tool_result_chars is not None and len(raw) > self._max_tool_result_chars:
                        raw = raw[:self._max_tool_result_chars] + "\n…[truncated]"
                    tool_content = raw
                else:
                    tool_content = f"[OK: {call.name}({_get_query_key(call)}) — no content]"

                content_parts = None
                if result.image_data and self._llm_vision:
                    content_parts = [
                        {"type": "text", "text": tool_content},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{result.image_data}",
                        }},
                    ]
                messages.append(LLMMessage(
                    role="tool",
                    content=tool_content,
                    content_parts=content_parts,
                    name=call.name,
                    tool_call_id=call.tool_call_id,
                ))
        return messages
