"""InProcessDomainAliasAdapter → DomainAliasPort (DEBT-02).

Loads domain alias mappings from an environment variable so they can be
changed via ConfigMap without code changes.  Falls back to the five
built-in aliases when the variable is absent or empty.

Environment variable format (KIMA_DOMAIN_ALIASES):
    Comma-separated "keyword=domain" pairs, e.g.:
    KIMA_DOMAIN_ALIASES=wikipedia=wikipedia.org,arxiv=arxiv.org,github=github.com

Built-in defaults (always present unless overridden by the env var):
    wikipedia → wikipedia.org
    arxiv     → arxiv.org
    github    → github.com
    stackoverflow → stackoverflow.com
    pubmed    → pubmed.ncbi.nlm.nih.gov

Set KIMA_DOMAIN_ALIASES=- to disable all aliases (source-lock detects explicit
URLs only).
"""
from __future__ import annotations

import logging
import os

from cima_demo.domain.ports import DomainAliasPort

log = logging.getLogger(__name__)

_BUILTIN_ALIASES: dict[str, str] = {
    "wikipedia":     "wikipedia.org",
    "arxiv":         "arxiv.org",
    "github":        "github.com",
    "stackoverflow": "stackoverflow.com",
    "pubmed":        "pubmed.ncbi.nlm.nih.gov",
}


class InProcessDomainAliasAdapter(DomainAliasPort):
    """Reads domain aliases from KIMA_DOMAIN_ALIASES env var.

    Merges env aliases over the built-in defaults so operators can extend
    or override entries without touching code.

    Pass ``extra: dict[str, str]`` at construction time to add programmatic
    overrides on top of both env and built-ins (useful in tests).
    """

    def __init__(
        self,
        env_var: str = "KIMA_DOMAIN_ALIASES",
        extra: dict[str, str] | None = None,
    ) -> None:
        raw = os.environ.get(env_var, "").strip()

        if raw == "-":
            # Sentinel: operator explicitly disabled all aliases.
            self._aliases: dict[str, str] = {}
            log.info("DomainAliasAdapter: all aliases disabled (KIMA_DOMAIN_ALIASES=-)")
            return

        aliases = dict(_BUILTIN_ALIASES)

        if raw:
            for entry in raw.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if "=" not in entry:
                    log.warning("DomainAliasAdapter: skipping malformed entry %r", entry)
                    continue
                keyword, _, domain = entry.partition("=")
                keyword = keyword.strip().lower()
                domain = domain.strip().lower()
                if keyword and domain:
                    aliases[keyword] = domain
            log.info(
                "DomainAliasAdapter: loaded %d aliases from %s",
                len(aliases), env_var,
            )

        if extra:
            aliases.update(extra)

        self._aliases = aliases

    def get_aliases(self) -> dict[str, str]:
        return self._aliases
