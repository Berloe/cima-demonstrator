"""Slot-contract parsing helpers (extracted from orchestration/engine.py).

Parses CONSTRAINT/FACT conclusions and optional <task>/<resolve> XML blocks
from model reasoning text into TaskState/TaskSlot domain objects.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from cima_demo.domain.value_objects import InputState, TaskSlot, TaskState

if TYPE_CHECKING:
    from cima_demo.cognitive.kernel.state import TurnRuntime

log = logging.getLogger(__name__)


# CONSTRAINT conclusion content: "SLOT slot_name: description (source: domain.com/path)"
_SLOT_DECL_RE = re.compile(
    r'^SLOT\s+(\S+)\s*:\s*(.+?)(?:\s+\(source:\s*([^)]+)\))?\s*$',
    re.IGNORECASE,
)
# URL extraction from compact locator index produced by web(search)
_LOCATOR_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')

# FACT conclusion content: "SLOT_VALUE:slot_name:value[:unit]"
_SLOT_VALUE_RE = re.compile(
    r'^SLOT_VALUE:([^:]+):(.+?)(?::([^:]+))?$',
)
# Optional inline <task> / <resolve> blocks in reasoning (bonus — not required by model)
_TASK_TAG_RE    = re.compile(r'<task>\s*(.*?)\s*</task>',    re.DOTALL | re.IGNORECASE)
_RESOLVE_TAG_RE = re.compile(r'<resolve>\s*(.*?)\s*</resolve>', re.DOTALL | re.IGNORECASE)


def _extract_urls_from_locator_context(text: str) -> list[str]:
    """Extract unique URLs from a compact search-result locator block.

    Handles the format produced by web(search):
      [L1] https://example.com / Title / Snippet…
      [L2] https://other.com / …
    Trailing punctuation stripped.
    """
    seen: set[str] = set()
    result: list[str] = []
    for m in _LOCATOR_URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?/")
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _update_slots_from_search_locators(ts: "TurnRuntime", locator_text: str) -> int:
    """Mark MISSING slots as LOCATED using URLs found in a web(search) result.

    Backend-driven: does not depend on model CONSTRAINT declarations.
    All MISSING slots receive the same candidate URLs because we cannot
    yet determine which URL is relevant to which slot — that determination
    happens at fetch time (mark_verified).

    Returns number of slots promoted from MISSING → LOCATED.
    """
    if ts.task_state is None or not ts.task_state.missing_slots:
        return 0
    urls = _extract_urls_from_locator_context(locator_text)
    if not urls:
        return 0
    promoted = 0
    for slot in ts.task_state.missing_slots:
        slot.mark_located(urls)
        promoted += 1
        log.debug(
            "TaskSlot LOCATED (backend/search): slot=%s urls=%d conv=%s",
            slot.name, len(urls), ts.conversation_id,
        )
    return promoted


def _try_update_slots_from_evidence(ts: "TurnRuntime") -> int:
    """Backend-driven slot update from fetched web evidence.

    After web(fetch/render) tool results are flushed, scans each result's
    evidence_context (or raw content) for numeric values that correspond to
    unresolved slot names via keyword matching.

    Match algorithm (conservative):
      1. All significant name tokens (>2 chars) of the slot must appear in
         the evidence text (case-insensitive).
      2. At least one number must follow within the same passage.
      3. First match wins; stops scanning that slot after promotion.

    This is the same heuristic as slot_resolver Path 2 but driven by the
    backend after every fetch batch, not by model FACT conclusions.

    Returns number of slots promoted from MISSING/LOCATED → VERIFIED.
    """
    if ts.task_state is None or not ts.task_state.unresolved_slots:
        return 0

    # Build list of (url, text, citem_id) from this turn's web fetch results
    fetched: list[tuple[str, str, str | None]] = []
    for call, result in zip(ts.tool_call_requests, ts.tool_results_accumulated):
        if not (call.name == "web" and result.success):
            continue
        action = (call.params or {}).get("action", "")
        if action not in ("fetch", "render"):
            continue
        text = result.evidence_context or result.content or ""
        if not text.strip():
            continue
        url = str((call.params or {}).get("url", "") or "")
        fetched.append((url, text, result.persisted_citem_id))

    if not fetched:
        return 0

    promoted = 0
    for slot in list(ts.task_state.unresolved_slots):
        name_tokens = [t for t in slot.name.replace("_", " ").split() if len(t) > 2]
        if not name_tokens:
            continue
        for url, text, citem_id in fetched:
            text_lower = text.lower()
            if not all(tok.lower() in text_lower for tok in name_tokens):
                continue

            # Proximity search: find the number closest to the keyword match
            # positions rather than taking the first number in the full text.
            # This prevents distant, unrelated numbers (e.g. page counts, dates,
            # footnote indices) from being picked up over the actual target value.
            tok_positions = [
                text_lower.find(tok.lower())
                for tok in name_tokens
                if tok.lower() in text_lower
            ]
            anchor = max(tok_positions)  # end of last keyword occurrence
            window_start = max(0, anchor - 50)
            window_end   = min(len(text), anchor + 200)
            nums = re.findall(r'[\d,]+(?:\.\d+)?', text[window_start:window_end])
            if not nums:
                # Fallback: any number in full text (last resort)
                nums = re.findall(r'[\d,]+(?:\.\d+)?', text)
            if not nums:
                continue

            val = nums[0].replace(",", "")
            slot.mark_verified(value=val, unit=slot.unit, evidence_id=citem_id)
            log.debug(
                "TaskSlot VERIFIED (backend/scan): slot=%s val=%s url=%s conv=%s",
                slot.name, val, url, ts.conversation_id,
            )
            promoted += 1
            break  # first matching URL wins for this slot
    return promoted


def _auto_create_task_state_from_locks(ts: "TurnRuntime") -> None:
    """Prime a TaskState shell when source_lock is detected.

    Creates an empty TaskState (no slots) so the model can populate semantic
    slots via CONSTRAINT conclusions (e.g. 'SLOT moon_min_perigee_km: ...')
    rather than receiving generic machine-generated slot names like
    'wikipedia_value' that carry no semantic value and cannot drive backend
    extraction.

    The shell signals to the model that slot-tracking is active for this turn.
    The model declares real slots → _update_task_state_from_conclusions fills them.
    The compute gate fires only when all declared slots are resolved.
    """
    if not ts.source_requirements or ts.task_state is not None:
        return
    ts.task_state = TaskState(
        objective=ts.user_message,
        slot_contract_required=False,  # opt-in via declare_slot tool; forced True = deadlock with Mistral tool-call mode
        output_contract=ts.output_contract,
    )
    log.debug(
        "TaskState shell created (source_lock, free compute) conv=%s reqs=%s",
        ts.conversation_id, [r.value for r in ts.source_requirements],
    )


def _parse_slot_declaration(content: str) -> TaskSlot | None:
    """Parse a CONSTRAINT conclusion content into a TaskSlot, or None if not a slot declaration."""
    m = _SLOT_DECL_RE.match(content.strip())
    if not m:
        return None
    name, description, source = m.group(1), m.group(2).strip(), m.group(3)
    # Extract unit hint from description if present (e.g. "distance in km")
    unit_m = re.search(r'\bin\s+([a-zA-Z/]+)\b', description)
    unit = unit_m.group(1) if unit_m else None
    return TaskSlot(name=name, description=description, required_source=source, unit=unit)


def _try_resolve_slot_from_fact(task_state: TaskState, conclusion: dict) -> bool:
    """Try to resolve a pending slot from a FACT/DERIVED conclusion with evidence.

    Two paths:
    1. Explicit: content starts with 'SLOT_VALUE:name:value[:unit]'
    2. Keyword: slot name words appear in the fact content (fuzzy match)

    Returns True if any slot was resolved.
    """
    content = conclusion.get("content", "")
    evidence = conclusion.get("evidence") or []
    evidence_id = evidence[0] if evidence else None
    resolved_any = False

    # Path 1: explicit SLOT_VALUE declaration
    m = _SLOT_VALUE_RE.match(content.strip())
    if m:
        slot_name, value, unit = m.group(1).strip(), m.group(2).strip(), m.group(3)
        slot = task_state.get_slot(slot_name)
        if slot and not slot.resolved:
            slot.mark_verified(value=value, unit=unit, evidence_id=evidence_id)
            log.debug("TaskSlot VERIFIED (explicit): %s = %s %s", slot_name, value, unit or "")
            resolved_any = True
        return resolved_any

    # Path 2: keyword match — check if any unresolved slot's name words appear in content
    content_lower = content.lower()
    for slot in task_state.unresolved_slots:
        # Match if slot name tokens (split on _) all appear in content
        name_tokens = slot.name.replace("_", " ").split()
        if all(tok.lower() in content_lower for tok in name_tokens if len(tok) > 2):
            # Extract a numeric value from the fact if possible
            nums = re.findall(r'[\d,]+(?:\.\d+)?', content)
            if nums:
                val = nums[0].replace(",", "")
                slot.mark_verified(value=val, unit=slot.unit, evidence_id=evidence_id)
                log.debug(
                    "TaskSlot VERIFIED (keyword match): %s = %s (from fact: %.80s)",
                    slot.name, val, content,
                )
                resolved_any = True

    return resolved_any


def _update_task_state_from_conclusions(ts: "TurnRuntime", conclusions: list[dict]) -> None:
    """Parse slot declarations and resolutions from a batch of conclusions."""
    for c in conclusions:
        ctype = c.get("type", "")
        content = c.get("content", "")

        if ctype == "CONSTRAINT":
            slot = _parse_slot_declaration(content)
            if slot:
                if ts.task_state is None:
                    ts.task_state = TaskState(
                        objective=ts.user_message,
                        output_contract=ts.output_contract,
                    )
                # Avoid duplicate slot declarations
                if not ts.task_state.get_slot(slot.name):
                    ts.task_state.slots.append(slot)
                    log.debug("TaskSlot declared: %s (%s)", slot.name, slot.description[:60])

                # Parse output contract if present in conclusion metadata
                if "output_unit" in c:
                    ts.task_state.output_unit = str(c["output_unit"])
                if "output_format" in c:
                    ts.task_state.output_format = str(c["output_format"])

        elif ctype in ("FACT", "DERIVED") and c.get("evidence"):
            if ts.task_state and ts.task_state.unresolved_slots:
                _before = len(ts.task_state.unresolved_slots)
                _try_resolve_slot_from_fact(ts.task_state, c)
                _after = len(ts.task_state.unresolved_slots)
                ts.resolved_slot_count += _before - _after


def _parse_task_from_reasoning(ts: "TurnRuntime", text: str) -> None:
    """Parse optional <task> and <resolve> XML blocks from REASONING text.

    These blocks are emitted by the model in its internal reasoning (pre-response)
    and provide a structured alternative to conclusion-based slot tracking.
    They are optional — the conclusion-based path is the primary mechanism.
    """
    # <task> block: JSON with objective, slots, output_unit, output_format
    for m in _TASK_TAG_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
            if ts.task_state is None:
                ts.task_state = TaskState(
                    objective=data.get("objective", ts.user_message),
                    output_unit=data.get("output_unit"),
                    output_format=data.get("output_format"),
                    output_contract=ts.output_contract,
                )
            for s in data.get("slots", []):
                name = s.get("name", "")
                if name and not ts.task_state.get_slot(name):
                    ts.task_state.slots.append(TaskSlot(
                        name=name,
                        description=s.get("description", ""),
                        required_source=s.get("source"),
                        unit=s.get("unit"),
                    ))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # <resolve> blocks: JSON with slot, value, unit (optional evidence_id)
    if ts.task_state:
        for m in _RESOLVE_TAG_RE.finditer(text):
            try:
                data = json.loads(m.group(1))
                slot_name = data.get("slot", "")
                slot = ts.task_state.get_slot(slot_name)
                if slot and not slot.resolved:
                    slot.resolve(
                        value=str(data.get("value", "")),
                        unit=data.get("unit"),
                        evidence_id=data.get("evidence_id"),
                    )
                    log.debug("TaskSlot resolved via <resolve>: %s = %s", slot_name, data.get("value"))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
