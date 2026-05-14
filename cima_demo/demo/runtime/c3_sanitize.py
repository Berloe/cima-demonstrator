"""C3-SAN-v1 — Deterministic Citation Sanitization Contract.

Guarantees: the published answer contains no markers outside allowed_markers.
Does NOT guarantee: factual correctness, claim support, entailment, semantic quality.

Invariants checked (independent of runtime):
  A  declared_allowed_markers == reconstructed_allowed_markers
  B  no invalid markers remain after sanitize
  C  sanitized_cited ⊆ raw_cited  (no new markers added)
  D  removed markers are exactly the invalid ones
  E  non-citation text is unchanged after normalization
  F  every substantive block has ≥ 1 valid marker
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# Canonical C3-SAN-v1 marker: [S|P][1-9][0-9]*
# S0, S01, X1, source1 are NOT canonical CIMA markers.
_CANONICAL_SINGLE = re.compile(r'\[([SP][1-9][0-9]*)\]')
_CANONICAL_GROUP  = re.compile(r'\[([SP][1-9][0-9]*(?:\s*,\s*[SP][1-9][0-9]*)*)\]')

# Broader pattern to also catch non-canonical lookalikes ([S0], [X1], etc.)
# so they are treated as invalid rather than silently left in the text.
_LOOKALIKE = re.compile(r'\[([A-Za-z][A-Za-z0-9]*(?:\s*,\s*[A-Za-z][A-Za-z0-9]*)*)\]')

# Status codes
NOOP_PASS                      = "NOOP_PASS"
SANITIZED_PASS                 = "SANITIZED_PASS"
SANITIZED_FAIL_UNCITED_BLOCK   = "SANITIZED_FAIL_UNCITED_BLOCK"
FAIL_INVALID_REMAINING         = "FAIL_INVALID_REMAINING"
FAIL_MUTATION                  = "FAIL_MUTATION"
FAIL_ALLOWED_MARKER_MISMATCH   = "FAIL_ALLOWED_MARKER_MISMATCH"

PASSING_STATUSES = {NOOP_PASS, SANITIZED_PASS}

# C3A — Traceable abstention / insufficient-evidence contract.
C3A_NOT_APPLICABLE = "NOT_APPLICABLE"
C3A_PASS = "TRACEABLE_ABSTENTION_PASS"
C3A_FAIL_NOT_TRACEABLE = "TRACEABLE_ABSTENTION_FAIL_NOT_TRACEABLE"
C3A_FAIL_HAS_FACTUAL_BLOCKS = "TRACEABLE_ABSTENTION_FAIL_HAS_FACTUAL_BLOCKS"


@dataclass
class SanitizationResult:
    status: str
    passed: bool
    sanitized_answer: str
    raw_cited_markers: list[str]
    sanitized_cited_markers: list[str]
    removed_invalid_markers: list[str]
    uncited_blocks: list[str]
    failure_reason: str
    report: dict[str, Any] = field(default_factory=dict)


# ── Text helpers ──────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _extract_all_marker_tokens(text: str) -> list[str]:
    """Extract every marker-like token (canonical + lookalikes, including grouped)."""
    tokens: list[str] = []
    for match in _LOOKALIKE.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part:
                tokens.append(part)
    return list(dict.fromkeys(tokens))


def _extract_canonical_markers(text: str) -> list[str]:
    """Extract only canonical C3-SAN-v1 markers (S/P + nonzero digits)."""
    tokens: list[str] = []
    for match in _CANONICAL_GROUP.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part:
                tokens.append(part)
    return list(dict.fromkeys(tokens))


def _normalize_grouped_markers(text: str) -> str:
    """Expand grouped markers: [S1, S2] → [S1][S2]."""
    def _expand(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        return "".join(f"[{p}]" for p in parts if p)
    return _CANONICAL_GROUP.sub(_expand, text)


def _strip_all_markers(text: str) -> str:
    """Remove all marker tokens (canonical + lookalikes)."""
    return _LOOKALIKE.sub("", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace; clean space before punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([.,;:!?])", r"\1", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Block helpers ─────────────────────────────────────────────────────────────

def _split_substantive_blocks(text: str) -> list[str]:
    """Split answer into substantive blocks (paragraphs + bullets)."""
    blocks: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                blocks.append(line)
    return blocks


def _is_substantive(block: str) -> bool:
    """True if the block makes a factual claim (not just framing/meta text)."""
    bare = _strip_all_markers(block)
    bare = re.sub(r"^[\s\-\*•#>]+", "", bare).strip()
    bare = re.sub(r"[*_`#]", "", bare).strip()
    if not bare or len(bare) < 20:
        return False
    if bare.endswith(":") and len(bare) < 160:
        return False
    # Repair/scaffolding text must not be treated as a factual answer block.
    if re.match(r"^here\s+is\s+(the\s+)?(corrected|revised|updated)?\s*(summary|answer)\b", bare, flags=re.IGNORECASE):
        return False
    if re.match(r"^corrected\s+(summary|answer)\s*(with\s+valid\s+citations)?\s*:?$", bare, flags=re.IGNORECASE):
        return False
    if re.match(r"^note\s*:", bare, flags=re.IGNORECASE):
        return False
    # Scope/insufficiency notes are diagnostic, not evidence claims.
    if re.search(
        r"\b(no direct evidence cited|evidence was insufficient|insufficient evidence|not enough evidence|"
        r"limited data|available evidence is limited|evidence is limited|context is limited|"
        r"source material is limited|broader\s+[^.!?]{0,80}\s+remain(?:s)?\s+"
        r"(?:unspecified|unclear|unknown|not covered))\b",
        bare,
        flags=re.IGNORECASE,
    ):
        return False
    # Short framing sentences introducing a meeting/document summary are not
    # evidence-bearing claims for C3.
    if len(bare) < 420 and re.match(
        r"^(this|the|overall,?\s+the)\s+(meeting|document|discussion|source|text)\s+"
        r"(addressed|explored|discussed|highlighted|covered|focused|identified|summarized)\b",
        bare,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _has_valid_marker(block: str, allowed: set[str]) -> bool:
    return any(m in allowed for m in _extract_all_marker_tokens(block))



# ── C3A abstention helpers ───────────────────────────────────────────────────

_ABSTENTION_RE = re.compile(
    r"^\s*(?:"
    r"not\s+enough\s+info(?:rmation)?"
    r"|insufficient\s+(?:evidence|information|context)"
    r"|there\s+is\s+insufficient\s+(?:evidence|information|context)"
    r"|i\s+(?:do\s+not|don['’]?t|cannot|can['’]?t)\s+(?:have|find|determine)"
    r"|cannot\s+determine"
    r"|can['’]?t\s+determine"
    r"|unable\s+to\s+determine"
    r"|the\s+available\s+(?:evidence|context|information)\s+(?:does\s+not|doesn['’]?t)\s+(?:support|establish|provide)"
    r")\b",
    flags=re.IGNORECASE,
)


def is_abstention_block(block: str) -> bool:
    """Return True for a pure insufficient-evidence/abstention block.

    This is intentionally conservative: it accepts short explicit abstentions
    and rejects long mixed blocks that may contain factual claims.
    """
    bare = _normalize_whitespace(_strip_all_markers(block or ""))
    bare = re.sub(r"^[\s\-\*•#>]+", "", bare).strip()
    bare = re.sub(r"[*_`#]", "", bare).strip()
    if not bare:
        return False
    if bare.upper() == "NOT ENOUGH INFO":
        return True
    if len(bare) > 260:
        return False
    return bool(_ABSTENTION_RE.search(bare))


def is_insufficient_evidence_answer(answer: str) -> bool:
    """True iff the visible answer is a pure abstention, not a factual answer."""
    blocks = [b for b in _split_substantive_blocks(answer or "") if _normalize_whitespace(_strip_all_markers(b))]
    if not blocks:
        return False
    return all(is_abstention_block(b) for b in blocks)


def build_c3a_abstention_report(
    *,
    answer: str,
    allowed_markers: set[str],
    context_view_id: str | None = None,
    inspected_markers: list[str] | None = None,
    zoom_attempted: bool = False,
    zoom_out_attempted: bool = False,
    extra_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build C3A traceable-abstention report.

    C3A is separate from factual citation support: it records the evidence scope
    inspected before an abstention. It never claims that inspected markers
    support the abstention as a factual answer.
    """
    answer_type = "insufficient_evidence" if is_insufficient_evidence_answer(answer) else "factual_answer"
    markers = list(dict.fromkeys([str(m) for m in (inspected_markers or []) if str(m)]))
    if not markers:
        markers = sorted(str(m) for m in allowed_markers if str(m))
    blocks = _split_substantive_blocks(answer or "")
    factual_blocks = [
        b for b in blocks
        if _normalize_whitespace(_strip_all_markers(b))
        and not is_abstention_block(b)
        and _is_substantive(b)
    ]
    trace = {
        "context_view_id": context_view_id,
        "inspected_markers": markers,
        "zoom_attempted": bool(zoom_attempted),
        "zoom_out_attempted": bool(zoom_out_attempted),
        "reason": "No factual answer was produced because available evidence was insufficient.",
    }
    if extra_trace:
        trace.update(extra_trace)

    if answer_type != "insufficient_evidence":
        status = C3A_NOT_APPLICABLE
        passed = False
    elif factual_blocks:
        status = C3A_FAIL_HAS_FACTUAL_BLOCKS
        passed = False
    elif not (context_view_id or markers or zoom_attempted or zoom_out_attempted):
        status = C3A_FAIL_NOT_TRACEABLE
        passed = False
    else:
        status = C3A_PASS
        passed = True

    return {
        "schema_version": "cima_demo.c3a_traceable_abstention.v1",
        "checked": answer_type == "insufficient_evidence",
        "applicable": answer_type == "insufficient_evidence",
        "passed": passed,
        "status": status,
        "answer_type": answer_type,
        "factual_citations_required": answer_type != "insufficient_evidence",
        "normal_citation_contract_applicable": answer_type != "insufficient_evidence",
        "insufficiency_trace": trace if answer_type == "insufficient_evidence" else None,
        "factual_block_count": len(factual_blocks),
        "factual_block_preview": [b[:200] for b in factual_blocks[:3]],
    }

# ── Core sanitizer ────────────────────────────────────────────────────────────

def sanitize(raw_answer: str, allowed_markers: set[str]) -> SanitizationResult:
    """Apply C3-SAN-v1 deterministic sanitization.

    Rules:
    - Remove markers not in allowed_markers.
    - Never add or substitute markers.
    - Never change non-citation text.
    - Validate block coverage after removal.
    """
    expanded = _normalize_grouped_markers(raw_answer)
    raw_sha  = _sha256(raw_answer)

    raw_cited = _extract_all_marker_tokens(expanded)
    invalid_raw = [m for m in raw_cited if m not in allowed_markers]
    raw_blocks = _split_substantive_blocks(expanded)
    raw_uncited = [
        b for b in raw_blocks
        if allowed_markers and _is_substantive(b) and not _has_valid_marker(b, allowed_markers)
    ]
    raw_model_passed = bool(allowed_markers) and not invalid_raw and not raw_uncited

    def _make_result(status: str, sanitized: str, failure_reason: str = "") -> SanitizationResult:
        san_cited = _extract_all_marker_tokens(sanitized)
        # D: removed markers must be precisely raw invalid markers.  The sanitizer
        # never removes valid markers intentionally; if that happens, FAIL_MUTATION
        # will be reported by the caller/invariant checks.
        removed   = [m for m in raw_cited if m not in san_cited]
        added     = [m for m in san_cited if m not in raw_cited]
        blocks    = _split_substantive_blocks(sanitized)
        uncited   = [b for b in blocks if _is_substantive(b) and not _has_valid_marker(b, allowed_markers)]
        passed    = status in PASSING_STATUSES
        raw_text_only = _normalize_whitespace(_strip_all_markers(expanded))
        san_text_only = _normalize_whitespace(_strip_all_markers(sanitized))
        report = {
            "schema_version": "cima_demo.citation_contract.v2",
            "method": "deterministic_strip_invalid_markers" if invalid_raw else "noop",
            "status": status,
            "passed": passed,
            "passed_after": passed,
            "raw_model_passed": raw_model_passed,
            "published_passed": passed,
            "raw_answer_sha256": raw_sha,
            "sanitized_answer_sha256": _sha256(sanitized),
            "raw_cited_markers": raw_cited,
            "valid_cited_markers_raw": [m for m in raw_cited if m in allowed_markers],
            "invalid_cited_markers_raw": invalid_raw,
            "sanitized_cited_markers": san_cited,
            "invalid_cited_markers_after_sanitize": [m for m in san_cited if m not in allowed_markers],
            "deterministic_sanitization_applied": bool(invalid_raw),
            "removed_invalid_markers": list(dict.fromkeys(invalid_raw)),
            "added_markers": list(dict.fromkeys(added)),
            "changed_non_citation_text": raw_text_only != san_text_only,
            "raw_answer_block_count": len(raw_blocks),
            "raw_uncited_answer_block_count": len(raw_uncited),
            "uncited_blocks_after_sanitize": len(uncited),
            "uncited_block_preview": [b[:200] for b in uncited[:3]],
        }
        if failure_reason:
            report["failure_reason"] = failure_reason
        return SanitizationResult(
            status=status,
            passed=passed,
            sanitized_answer=sanitized,
            raw_cited_markers=raw_cited,
            sanitized_cited_markers=san_cited,
            removed_invalid_markers=removed,
            uncited_blocks=uncited,
            failure_reason=failure_reason,
            report=report,
        )

    # No invalid markers — check block coverage directly.
    if not invalid_raw:
        blocks  = _split_substantive_blocks(expanded)
        uncited = [b for b in blocks if allowed_markers and _is_substantive(b) and not _has_valid_marker(b, allowed_markers)]
        if uncited:
            return _make_result(SANITIZED_FAIL_UNCITED_BLOCK, expanded, SANITIZED_FAIL_UNCITED_BLOCK)
        return _make_result(NOOP_PASS, expanded)

    # Strip invalid markers and lookalikes.
    def _remove_invalid(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        valid = [p for p in parts if p in allowed_markers]
        return "".join(f"[{p}]" for p in valid) if valid else ""

    sanitized = _LOOKALIKE.sub(_remove_invalid, expanded)
    sanitized = _normalize_whitespace(sanitized)

    # Invariant B — no invalid markers remain.
    invalid_after = [m for m in _extract_all_marker_tokens(sanitized) if m not in allowed_markers]
    if invalid_after:
        return _make_result(FAIL_INVALID_REMAINING, sanitized, FAIL_INVALID_REMAINING)

    # Invariant C — no markers added.
    added = set(_extract_all_marker_tokens(sanitized)) - set(raw_cited)
    if added:
        return _make_result(FAIL_MUTATION, sanitized, f"FAIL_MUTATION: added {sorted(added)}")

    # Invariant D — only invalid raw markers may be removed.
    removed = set(raw_cited) - set(_extract_all_marker_tokens(sanitized))
    removed_valid = sorted(m for m in removed if m in allowed_markers)
    if removed_valid:
        return _make_result(FAIL_MUTATION, sanitized, f"FAIL_MUTATION: removed valid markers {removed_valid}")

    # Invariant E — non-citation text unchanged.
    raw_text_only = _normalize_whitespace(_strip_all_markers(expanded))
    san_text_only = _normalize_whitespace(_strip_all_markers(sanitized))
    if raw_text_only != san_text_only:
        return _make_result(FAIL_MUTATION, sanitized, "FAIL_MUTATION: text content changed")

    # Invariant F — block coverage.
    blocks  = _split_substantive_blocks(sanitized)
    uncited = [b for b in blocks if _is_substantive(b) and not _has_valid_marker(b, allowed_markers)]
    if uncited:
        return _make_result(SANITIZED_FAIL_UNCITED_BLOCK, sanitized, SANITIZED_FAIL_UNCITED_BLOCK)

    return _make_result(SANITIZED_PASS, sanitized)



# ── Publication gate helpers ─────────────────────────────────────────────────

_PUBLICATION_REASON_BY_SAN_STATUS = {
    SANITIZED_FAIL_UNCITED_BLOCK: "uncited_block_after_sanitizer",
    FAIL_INVALID_REMAINING: "invalid_marker_remaining",
    FAIL_MUTATION: "sanitizer_mutation",
    FAIL_ALLOWED_MARKER_MISMATCH: "allowed_marker_mismatch",
}


def build_publication_gate(
    *,
    raw_answer: str,
    published_answer: str,
    published_integrity_passed: bool,
    c3_published_report: dict[str, Any] | None = None,
    c3a_abstention_report: dict[str, Any] | None = None,
    generation_passed: bool = True,
    generation_failure_kind: str | None = None,
    factual_citations_required: bool = True,
    sanitization_applied: bool = False,
) -> dict[str, Any]:
    """Return the explicit CIMA publication-gate decision.

    The gate separates *generated* outputs from *publishable* outputs:
    - publishable: CIMA may expose the answer as satisfying C3/C3A.
    - blocked: CIMA detected an unsupported/invalid output and must not present
      it as a valid published answer.

    This function never repairs evidence and never reclassifies factuality.  It
    only records whether the already-sanitized output satisfied the publication
    contract.
    """
    c3_report = c3_published_report or {}
    c3a_report = c3a_abstention_report or {}
    answer_type = str(c3a_report.get("answer_type") or ("factual_answer" if factual_citations_required else "unknown"))
    status = str(c3_report.get("status") or "")
    raw_answer = raw_answer or ""
    published_answer = published_answer or ""

    blocked_reason: str | None = None
    if not generation_passed:
        blocked_reason = str(generation_failure_kind or "generation_failed")
    elif not published_answer.strip():
        blocked_reason = "empty_generation"
    elif published_integrity_passed:
        blocked_reason = None
    elif answer_type == "insufficient_evidence":
        c3a_status = str(c3a_report.get("status") or "")
        if c3a_status == C3A_FAIL_HAS_FACTUAL_BLOCKS:
            blocked_reason = "abstention_contains_factual_blocks"
        elif c3a_status == C3A_FAIL_NOT_TRACEABLE:
            blocked_reason = "untraceable_abstention"
        else:
            blocked_reason = "abstention_contract_failed"
    elif status in _PUBLICATION_REASON_BY_SAN_STATUS:
        blocked_reason = _PUBLICATION_REASON_BY_SAN_STATUS[status]
        if status == SANITIZED_FAIL_UNCITED_BLOCK and not sanitization_applied:
            blocked_reason = "uncited_block"
    else:
        invalid_after = c3_report.get("invalid_cited_markers_after_sanitize") or []
        uncited_after = c3_report.get("uncited_blocks_after_sanitize") or 0
        if invalid_after:
            blocked_reason = "invalid_marker_remaining"
        elif uncited_after:
            blocked_reason = "uncited_block_after_sanitizer" if sanitization_applied else "uncited_block"
        else:
            blocked_reason = "publication_contract_failed"

    publishable = blocked_reason is None and bool(published_integrity_passed)
    publication_status = "publishable" if publishable else "blocked"
    invalid_published_as_valid = bool(publication_status == "publishable" and not published_integrity_passed)

    return {
        "schema_version": "cima_demo.publication_gate.v1",
        "publication_status": publication_status,
        "publishable": publishable,
        "blocked_by_cima": not publishable,
        "blocked_reason": blocked_reason,
        "invalid_published_as_valid": invalid_published_as_valid,
        "published_integrity_passed": bool(published_integrity_passed),
        "factual_citations_required": bool(factual_citations_required),
        "answer_type": answer_type,
        "sanitization_applied": bool(sanitization_applied),
        "raw_answer_sha256": _sha256(raw_answer),
        "published_answer_sha256": _sha256(published_answer),
        "c3_published_status": status or None,
        "c3a_status": c3a_report.get("status"),
        "generation_passed": bool(generation_passed),
        "generation_failure_kind": generation_failure_kind,
    }

# ── Independent validator (audit-time) ───────────────────────────────────────

def validate_independent(
    *,
    context: dict[str, Any] | None,
    zoom_out: dict[str, Any] | None,
    chat: dict[str, Any],
    citation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct and verify the C3-SAN-v1 contract without trusting runtime fields.

    Invariant A: declared_allowed_markers == reconstructed_allowed_markers.
    """
    reconstructed: set[str] = set()
    if context:
        reconstructed.update(str(m) for m in (context.get("markers") or []) if str(m))
    if zoom_out:
        reconstructed.update(str(m) for m in (zoom_out.get("markers_added") or []) if str(m))

    declared = set(citation_contract.get("allowed_markers") or [])

    if declared != reconstructed:
        return {
            "invariant_a": False,
            "status": FAIL_ALLOWED_MARKER_MISMATCH,
            "declared_allowed_markers": sorted(declared),
            "reconstructed_allowed_markers": sorted(reconstructed),
        }

    published_answer = (chat.get("choices") or [{}])[0].get("message", {}).get("content", "")
    raw_answer = (
        citation_contract.get("raw_model_answer_text")
        or citation_contract.get("raw_answer_text")
        or published_answer
    )
    result = sanitize(str(raw_answer), declared)
    invariant_published_matches = _normalize_whitespace(result.sanitized_answer) == _normalize_whitespace(str(published_answer))
    return {
        "invariant_a": True,
        "invariant_published_matches_sanitized": invariant_published_matches,
        "declared_allowed_markers": sorted(declared),
        **result.report,
        "status": result.report.get("status") if invariant_published_matches else FAIL_MUTATION,
        "passed": bool(result.passed and invariant_published_matches),
    }
