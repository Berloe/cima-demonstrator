"""Public-identity regression guard for the model-visible system prompt.

The CIMA Demonstrator must never present its public assistant identity as the
legacy KIMA agent. The identity line lives outside the tool/answer-mode
conditionals in system_prompt.j2, so it must hold in every render mode. This
test asserts against the *rendered* prompt produced by the real factory
(``_render_system_prompt``), not the raw template file, because that rendered
string is the artifact actually sent to the model.
"""
from __future__ import annotations

from cima_demo.api.app import _render_system_prompt
from cima_demo.branding import PUBLIC_PROJECT_NAME


def test_rendered_system_prompt_uses_cima_public_identity() -> None:
    for synthesis in (False, True):
        rendered = _render_system_prompt(synthesis=synthesis)

        # 1. No legacy public identity may leak to the model output surface
        #    (covers KIMA / Kima / kima and any other casing).
        assert "kima" not in rendered.lower(), (
            f"legacy KIMA identity leaked into the rendered prompt (synthesis={synthesis})"
        )

        # 2. The correct public persona must be present — asserting absence of
        #    KIMA alone would also pass an empty or broken prompt.
        assert "CIMA Demonstrator assistant" in rendered
        assert PUBLIC_PROJECT_NAME in rendered  # stays consistent with branding.py
