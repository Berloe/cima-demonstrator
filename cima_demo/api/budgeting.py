"""Budget helpers for model-side context construction."""
from __future__ import annotations

from typing import Any

from cima_demo.domain.value_objects import ContextBudget


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return max(0, int(default))
    return max(0, parsed)


def build_effective_context_budget(
    *,
    requested_context_tokens: int | None,
    reserve_output_tokens: int | None,
    overhead_tokens: int | None,
    settings: Any,
) -> ContextBudget:
    """Return the prompt/context budget that may be sent to the model.

    `requested_context_tokens` is the user/runtime requested *model window* for
    this turn. The actual selectable memory content must fit inside:

        min(requested_context_tokens, configured_llm_context_window)
        - reserve_output_tokens
        - overhead_tokens

    The returned ContextBudget keeps the existing domain convention where
    `available_for_content == max_tokens - overhead_tokens` by pre-subtracting
    the output reserve from `max_tokens`.
    """
    configured_request = _positive_int(getattr(settings, "context_budget_max", 0), 0)
    requested_window = _positive_int(
        requested_context_tokens if requested_context_tokens is not None else configured_request,
        configured_request,
    )
    model_window = _positive_int(getattr(settings, "llm_context_window", 0), 0)
    if model_window > 0:
        requested_window = min(requested_window, model_window)

    reserve_default = _positive_int(getattr(settings, "llm_max_tokens", 0), 0)
    reserve = _positive_int(
        reserve_output_tokens if reserve_output_tokens is not None else reserve_default,
        reserve_default,
    )
    overhead_default = _positive_int(getattr(settings, "context_budget_overhead", 0), 0)
    overhead = _positive_int(
        overhead_tokens if overhead_tokens is not None else overhead_default,
        overhead_default,
    )

    prompt_budget = max(0, requested_window - reserve)
    # Keep at least the overhead in max_tokens so available_for_content is never
    # negative. If the caller over-reserves, this produces zero selected content.
    max_tokens = max(overhead, prompt_budget)
    return ContextBudget(max_tokens=max_tokens, overhead_tokens=overhead)
