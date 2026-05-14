"""Source-lock detection helpers (extracted from orchestration/engine.py).

Detects explicit source requirements in user messages and tracks whether
fetched URLs satisfy them during a turn.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from cima_demo.domain.value_objects import SourceRequirement


# Meta-turn detection: messages that are system/meta operations (title generation,
# summarization) should not trigger source-lock, even if conversation history
# embedded in the message contains domain keywords.
_META_TURN_RE = re.compile(
    r"^\s*(generate|write|create|produce|provide|give me)\b.{0,120}\b(title|summary|headline)\b",
    re.IGNORECASE | re.DOTALL,
)

# DEBT-02 resolved: _DOMAIN_ALIASES removed from source_lock.py.
# Aliases are now managed by DomainAliasPort / InProcessDomainAliasAdapter.
# Tests and callers that don't inject an alias dict get the built-in defaults
# from InProcessDomainAliasAdapter (imported lazily to avoid circular deps).
def _default_aliases() -> dict[str, str]:
    """Return built-in alias dict for backward compat (lazy import)."""
    from cima_demo.infrastructure.aliases.adapter import _BUILTIN_ALIASES  # noqa: PLC0415
    return _BUILTIN_ALIASES

# URL regex: allow dots (required for domain names) but strip trailing punctuation via rstrip below.
_EXPLICIT_URL_RE = re.compile(r'https?://[^\s)>\]"\']+')


def normalize_url(url: str) -> str:
    """Canonical URL form for requirement matching.

    Strips fragment, strips trailing slash from path, lowercases host.
    Redirection and fragment differences don't break matching.
    """
    try:
        p = urlparse(url)
        return urlunparse((
            p.scheme.lower(),
            p.netloc.lower(),
            p.path.rstrip("/") or "/",
            p.params,
            p.query,
            "",  # drop fragment
        ))
    except Exception:
        return url.lower().rstrip("/")


def _url_satisfies_requirement(req: SourceRequirement, fetched_url: str) -> bool:
    """Return True if fetched_url matches the requirement."""
    if req.kind == "url":
        return normalize_url(fetched_url) == req.normalized
    if req.kind == "domain":
        try:
            hostname = urlparse(fetched_url).hostname or ""
            return hostname == req.normalized or hostname.endswith("." + req.normalized)
        except Exception:
            return False
    return False


def _detect_source_lock(
    user_message: str,
    aliases: dict[str, str] | None = None,
) -> list[SourceRequirement]:
    """Return named-source requirements for this turn.

    Generic: no provider strings in the core.

    aliases: {keyword: domain} mapping from DomainAliasPort.get_aliases().
    When None, the built-in defaults from InProcessDomainAliasAdapter are used
    (backward-compatible: existing callers and tests work without changes).

    Meta-turns (title generation, summarization) return an empty list so
    embedded conversation history does not produce false-positive requirements.
    """
    if _META_TURN_RE.match(user_message):
        return []

    _aliases = aliases if aliases is not None else _default_aliases()
    requirements: list[SourceRequirement] = []
    seen: set[str] = set()

    # Explicit URLs (highest priority — user gave the actual address)
    for m in _EXPLICIT_URL_RE.finditer(user_message):
        url = m.group(0).rstrip(".,;:!?")  # strip trailing punctuation from URLs
        norm = normalize_url(url)
        if norm not in seen:
            requirements.append(SourceRequirement(kind="url", value=url, normalized=norm))
            seen.add(norm)

    # Domain aliases — keyword found in the message
    for keyword, domain in _aliases.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_message, re.IGNORECASE):
            if domain not in seen:
                requirements.append(SourceRequirement(kind="domain", value=keyword, normalized=domain))
                seen.add(domain)

    return requirements
