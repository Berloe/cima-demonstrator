"""QueryPlanner + TaskSpecBuilder — pure functions, no LLM call (APP-INV-24)."""
from __future__ import annotations

import re

from cima_demo.domain.operations import classify_query, make_retrieval_plan, _QUANTITATIVE_RE
from cima_demo.domain.value_objects import (
    AnswerSchema,
    ExecutionMode,
    OutputContract,
    QueryType,
    RetrievalPlan,
    TaskSpec,
)

# Patterns for attachment detection (file extensions / content keywords)
_ATTACH_EXTS_RE = re.compile(
    # Unambiguous file-format words and extension literals only.
    # "code" and "script" are deliberately excluded: they appear routinely in
    # questions about programming without implying a file is attached, and
    # would cause ATTACHMENT_REQUIRED to fire falsely on empty workspace.
    r'\b(?:spreadsheet|excel|xlsx?|csv|pdf|image|photo|picture|screenshot'
    r'|png|jpe?g|gif|svg|docx?|pptx?|zip|tar|\.py|\.js|\.ts'
    r'|\.go|\.rs|\.java|attached|attachment|the\s+file|this\s+file)\b',
    re.IGNORECASE,
)

# Named-source patterns for SOURCE_BOUND_QUANT detection
# A named source + quantitative context → SOURCE_BOUND_QUANT
_NAMED_SOURCE_RE = re.compile(
    r'\b(?:wikipedia|according\s+to|from\s+(?:the\s+)?(?:wiki|site|page|article'
    r'|source|url|website)|official|record|published|reference|table|database)\b',
    re.IGNORECASE,
)

# Pure arithmetic constants — no web lookup needed
# Includes Spanish terms (docenas) for multilingual coverage
_PURE_CONST_RE = re.compile(
    r'\b(?:dozen|docenas?|gross|score|pi|\d+\s*(?:x|\*|plus|minus|divided\s+by|times|mod)\s*\d+'
    r'|\d+%\s+of\s+\d+|\d+\s+percent\s+of\s+\d+)\b',
    re.IGNORECASE,
)

# Person/entity + quantity context → SOURCE_BOUND_QUANT
# "Eliud Kipchoge", "marathon", "record" etc. suggest web lookup needed
_ENTITY_QUANT_RE = re.compile(
    r'\b(?:[A-Z][a-z]+ [A-Z][a-z]+|marathon|world record|olympic|championship'
    r'|perigee|apogee|orbit|planet|star|moon|earth|population|gdp|capital'
    r'|president|prime minister|ceo|founded|established)\b',
)

# Property terms for SOURCE_BOUND_QUANT slot name derivation.
# Canonical properties whose names appear reliably in authoritative evidence text.
_SBQ_PROP_RE = re.compile(
    r'\b(speed|velocity|pace|distance|length|height|depth|width|altitude|radius'
    r'|time|duration|period|mass|weight|temperature|pressure|area|volume|energy'
    r'|power|count|number|population|frequency|angle|density|perigee|apogee'
    r'|orbit|distance)\b',
    re.IGNORECASE,
)


def _derive_sbq_slot_names(query: str, output_unit: str | None) -> tuple[str, ...]:
    """Derive one canonical slot name for a SOURCE_BOUND_QUANT query.

    The name is built from the first property keyword in the query plus the
    output unit slug.  Property words (not entity names) are used as tokens
    because they appear reliably in evidence text — the backend keyword-match
    scan in _try_update_slots_from_evidence uses these tokens to locate and
    verify the slot value after a web fetch.

    Examples:
      "Kipchoge's speed in km/h"   → "speed_km_h"
      "Moon's perigee distance km" → "distance_km"
      "marathon time in seconds"   → "time_s"
      no property + unit "km"      → "target_km"
      nothing                      → "target_value"
    """
    prop_m = _SBQ_PROP_RE.search(query)
    prop = prop_m.group(0).lower() if prop_m else None

    unit_slug = ""
    if output_unit:
        # "km/h" → "_km_h", "m/s" → "_m_s", "km" → "_km"
        unit_slug = "_" + re.sub(r'[^a-z0-9]+', '_', output_unit.lower()).strip('_')

    if prop:
        return (f"{prop}{unit_slug}",)
    if unit_slug:
        return (f"target{unit_slug}",)
    return ("target_value",)


# Explicit numeric values present in the prompt — signals PROMPT_CONTAINED_QUANT.
# Matches time expressions (H:MM:SS, HH:MM:SS, H:MM) and numeric measures with units.
# Two or more matches indicate the prompt is self-sufficient for computation.
_EXPLICIT_VALUES_RE = re.compile(
    r'(?:'
    r'\b\d{1,2}:\d{2}:\d{2}\b'                    # HH:MM:SS / H:MM:SS  (e.g. 2:01:39)
    r'|\b\d{1,2}:\d{2}\b'                          # HH:MM / H:MM
    r'|\b\d+(?:[.,]\d+)?\s*'                       # numeric value ...
    r'(?:km/h|m/s|mph|km|kms|mi|miles|m\b|kg|lbs?'
    r'|days?|weeks?|months?|years?|hours?|hr|mins?|sec(?:ond)?s?)\b'  # ... with unit
    r')',
    re.IGNORECASE,
)

# Riddle / direct-answer signals
_RIDDLE_RE = re.compile(
    r'\b(?:riddle|puzzle|what\s+am\s+i|what\s+has|i\s+have|i\s+am\s+not'
    r'|you\s+have|what\s+comes\s+next|next\s+in\s+(?:the\s+)?series)\b',
    re.IGNORECASE,
)

# Queries that ask for a word, term, or name — the expected answer is textual, not numeric.
# Must be checked BEFORE entity-quant classification so that questions like
# "what word describes Fáfnir's characteristic" are not forced into SOURCE_BOUND_QUANT
# just because they contain a proper noun (e.g. "Emily Midkiff").
_TEXTUAL_ANSWER_RE = re.compile(
    r'\bwhat\s+(?:word|term|name|adjective|noun|verb|phrase|concept|type|kind|sort'
    r'|category|genre|species|nickname|title|epithet|label|letter|character'
    r'|colou?rs?)\b'
    r'|\bwhich\s+(?:word|term|name|letter)\b',
    re.IGNORECASE,
)

_BARE_NUMBER_RE = re.compile(
    r'\b(?:bare\s+number|number\s+only|only\s+the\s+number|just\s+the\s+number'
    r'|return\s+only\s+(?:the\s+)?number|respond\s+with\s+(?:the\s+)?number'
    r'|without\s+units?|no\s+units?)\b',
    re.IGNORECASE,
)

_DISPLAY_SCALE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\bper\s+100\s*[ ,]?000\b|\bper\s+100k\b', re.IGNORECASE), 'per_100k'),
    # "in thousands" (explicit) + "how many thousand(s) [of] X" + "nearest thousand"
    (re.compile(
        r'\b(?:in|as|express(?:ed)?\s+in|shown?\s+in)\s+thousands?\b'
        r'|\bhow\s+many\s+thousands?(?:\s+of)?\s+\w'
        r'|\bnearest\s+thousand\b',
        re.IGNORECASE,
    ), 'thousand'),
    # "in millions" + "how many million(s) [of] X" + "nearest million"
    (re.compile(
        r'\b(?:in|as|express(?:ed)?\s+in|shown?\s+in)\s+millions?\b'
        r'|\bhow\s+many\s+millions?(?:\s+of)?\s+\w'
        r'|\bnearest\s+million\b',
        re.IGNORECASE,
    ), 'million'),
    # "in billions" + "how many billion(s) [of] X" + "nearest billion"
    (re.compile(
        r'\b(?:in|as|express(?:ed)?\s+in|shown?\s+in)\s+billions?\b'
        r'|\bhow\s+many\s+billions?(?:\s+of)?\s+\w'
        r'|\bnearest\s+billion\b',
        re.IGNORECASE,
    ), 'billion'),
)

_DECIMAL_PLACES_RE = re.compile(r'\b(\d+)\s+decimal\s+places?\b', re.IGNORECASE)
_SIG_FIGS_RE = re.compile(r'\b(\d+)\s+(?:significant\s+figures?|sig\.?\s*figs?)\b', re.IGNORECASE)
_NEAREST_SIMPLE_RE = re.compile(
    r'\bnearest\s+([a-z0-9 ,/%-]+?)(?:\s+(?:and|but|with|without|if|using|use|do)\b|[?.!,;:]|$)',
    re.IGNORECASE,
)


class QueryPlanner:
    """Maps query text to RetrievalPlan (APP-INV-24 — pure, no I/O).

    Phase 1: rule-based classification.
    Phase 2: lightweight LLM-assisted classification if needed.
    """

    def plan(self, query: str) -> RetrievalPlan:
        """Classify query and return canonical RetrievalPlan."""
        query_type = classify_query(query)
        return make_retrieval_plan(query_type)


class TaskSpecBuilder:
    """Derives TaskSpec from user message — pure, no LLM, no I/O.

    Called once at turn start; result injected into TurnRuntime.
    The TaskSpec drives tool selection, slot contract, compute gate,
    drift monitors, and final-answer validation for the entire turn.
    """

    def build(
        self,
        user_message: str,
        has_attachment: bool = False,
    ) -> TaskSpec:
        """Derive TaskSpec deterministically from user message and context."""
        q = user_message

        # 1. Attachment-required: file/spreadsheet/image/code present or requested
        if has_attachment or _ATTACH_EXTS_RE.search(q):
            return _make_task_spec(
                q,
                mode=ExecutionMode.ATTACHMENT_REQUIRED,
                format='text',
                has_attachment=has_attachment or bool(_ATTACH_EXTS_RE.search(q)),
            )

        # 2. Riddle / direct-answer: no tools needed
        if _RIDDLE_RE.search(q):
            return _make_task_spec(q, mode=ExecutionMode.DIRECT_ANSWER, format='text')

        # 2.5. Textual-answer queries: "what word/term/name/adjective..." → BROWSE_LOOKUP.
        # Must precede entity-quant classification: a question like "what word describes
        # Fáfnir's characteristic" contains a proper noun ("Emily Midkiff") that would
        # otherwise trigger SOURCE_BOUND_QUANT with a numeric contract, producing a
        # spurious slot and blocking synthesis indefinitely.
        if _TEXTUAL_ANSWER_RE.search(q):
            return _make_task_spec(q, mode=ExecutionMode.BROWSE_LOOKUP, format='text')

        is_quantitative = bool(_QUANTITATIVE_RE.search(q))
        has_pure_const = bool(_PURE_CONST_RE.search(q))
        has_entity_quant = bool(_ENTITY_QUANT_RE.search(q))

        # 3. PROMPT_CONTAINED_QUANT: quantitative with ≥2 explicit numeric values in prompt.
        # All inputs are stated directly (e.g. "42.195 km in 2:01:39") — no external fetch needed.
        # Checked BEFORE SOURCE_BOUND_QUANT: explicit values override entity-based lookup trigger.
        _explicit_vals = _EXPLICIT_VALUES_RE.findall(q)
        if is_quantitative and len(_explicit_vals) >= 2 and not has_pure_const:
            _unit = _infer_unit(q)
            return _make_task_spec(
                q,
                mode=ExecutionMode.PROMPT_CONTAINED_QUANT,
                format='number',
                unit=_unit,
                required_evidence=False,
            )

        # 4. SOURCE_BOUND_QUANT: quantitative or entity signal + (named source OR entity/record)
        # _ENTITY_QUANT_RE alone is sufficient — named entities imply a lookup is needed.
        if (is_quantitative or has_entity_quant) and (
            _NAMED_SOURCE_RE.search(q) or has_entity_quant
        ):
            # Pure-const overrides entity in pure arithmetic expressions (e.g. "4 dozen × π")
            if not has_pure_const:
                _unit = _infer_unit(q)
                return _make_task_spec(
                    q,
                    mode=ExecutionMode.SOURCE_BOUND_QUANT,
                    format='number',
                    unit=_unit,
                    required_evidence=True,
                )

        # 5. DIRECT_ARITHMETIC: pure-constant arithmetic — no fetch needed
        # Triggered by _PURE_CONST_RE regardless of _QUANTITATIVE_RE (covers e.g. "4 dozen × 3.14")
        if has_pure_const:
            _unit = _infer_unit(q)
            return _make_task_spec(
                q,
                mode=ExecutionMode.DIRECT_ARITHMETIC,
                format='number',
                unit=_unit,
            )

        # 6. Generic quantitative without named source — keep numeric schema,
        #    but do not require external evidence by default.
        if is_quantitative:
            _unit = _infer_unit(q)
            return _make_task_spec(
                q,
                mode=ExecutionMode.SOURCE_BOUND_QUANT,
                format='number',
                unit=_unit,
                required_evidence=True,
            )

        # 7. classify_query determines MEMORY_RAG vs BROWSE_LOOKUP
        query_type = classify_query(q)
        if query_type in (QueryType.LOCAL_PRECISE, QueryType.GLOBAL_SYNTHETIC):
            # No named source detected: internal RAG first, web if needed
            return _make_task_spec(q, mode=ExecutionMode.MEMORY_RAG, format='text')

        # Default: BROWSE_LOOKUP for multi-hop, procedural, diagnostic
        return _make_task_spec(q, mode=ExecutionMode.BROWSE_LOOKUP, format='text')


def _make_task_spec(
    query: str,
    *,
    mode: ExecutionMode,
    format: str = 'text',
    unit: str | None = None,
    required_evidence: bool = False,
    has_attachment: bool = False,
) -> TaskSpec:
    contract = _infer_output_contract(
        query,
        format=format,
        base_unit=unit,
        required_evidence=required_evidence,
    )
    schema = AnswerSchema(
        format=contract.format,
        unit=contract.base_unit,
        precision=contract.precision,
        required_evidence=contract.required_evidence,
        output_contract=contract,
    )
    # SOURCE_BOUND_QUANT: derive canonical slot names so the orchestrator can
    # pre-populate a TaskState shell with typed targets.  All other modes leave
    # slot_names empty — they either have no slot contract (DIRECT_ARITHMETIC,
    # PROMPT_CONTAINED_QUANT) or are non-quantitative (BROWSE_LOOKUP, MEMORY_RAG).
    slot_names: tuple[str, ...] = ()
    if mode == ExecutionMode.SOURCE_BOUND_QUANT:
        slot_names = _derive_sbq_slot_names(query, contract.base_unit)

    return TaskSpec(
        mode=mode,
        answer_schema=schema,
        has_attachment=has_attachment,
        output_contract=contract,
        slot_names=slot_names,
    )


def _infer_output_contract(
    query: str,
    *,
    format: str,
    base_unit: str | None,
    required_evidence: bool,
) -> OutputContract:
    q = query.strip()
    representation = _infer_representation(q)
    display_scale = _infer_display_scale(q)
    rounding_rule, precision = _infer_rounding_and_precision(q)
    return OutputContract(
        format=format,
        representation=representation,
        base_unit=base_unit,
        display_scale=display_scale,
        rounding_rule=rounding_rule,
        precision=precision,
        required_evidence=required_evidence,
    )


def _infer_representation(query: str) -> str | None:
    q = query.lower()
    if _BARE_NUMBER_RE.search(q):
        return 'bare_number'
    return None


def _infer_display_scale(query: str) -> str | None:
    for pattern, label in _DISPLAY_SCALE_PATTERNS:
        if pattern.search(query):
            return label
    return None


def _infer_rounding_and_precision(query: str) -> tuple[str | None, int | None]:
    q = query.lower()
    match = _DECIMAL_PLACES_RE.search(q)
    if match:
        places = int(match.group(1))
        return f'{places} decimal places', places

    match = _SIG_FIGS_RE.search(q)
    if match:
        figures = int(match.group(1))
        return f'{figures} significant figures', figures

    if re.search(r'\bnearest\s+(?:integer|whole\s+number)\b', q):
        return 'nearest integer', 0

    match = _NEAREST_SIMPLE_RE.search(q)
    if match:
        phrase = ' '.join(match.group(1).replace(',', '').split())
        if phrase:
            return f'nearest {phrase}', None

    return None, None


def _infer_unit(query: str) -> str | None:
    """Heuristically extract expected output unit from query text."""
    q = query.lower()
    if re.search(r'\bkm/?h\b', q):
        return 'km/h'
    if re.search(r'\bmph\b', q):
        return 'mph'
    if re.search(r'\bm/?s\b', q):
        return 'm/s'
    if re.search(r'\bkilomet(?:er|re)s?\b', q):
        return 'km'
    if re.search(r'\bmeters?\b', q):
        return 'm'
    if re.search(r'\bmiles?\b', q):
        return 'miles'
    if re.search(r'\b(?:minutes?|mins?)\b', q):
        return 'min'
    if re.search(r'\bhours?\b', q):
        return 'h'
    if re.search(r'\bpercent(?:age)?|%\b', q):
        return '%'
    return None
