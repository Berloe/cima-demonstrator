"""Tests for TransitionPolicy — verifies all policy decision points."""
from __future__ import annotations

import pytest

from cima_demo.cognitive.kernel.transition import Transition, TransitionPolicy


class TestOnStall:
    def test_synthesize_when_no_slot_pending(self) -> None:
        policy = TransitionPolicy()
        d = policy.on_stall(None, None, slot_pending=False)  # type: ignore[arg-type]
        assert d.transition == Transition.SYNTHESIZE

    def test_act_when_slot_pending(self) -> None:
        policy = TransitionPolicy()
        d = policy.on_stall(None, None, slot_pending=True)  # type: ignore[arg-type]
        assert d.transition == Transition.ACT


class TestOnIterationLimit:
    def test_synthesize_escape_when_tool_results_no_reply(self) -> None:
        policy = TransitionPolicy()
        d = policy.on_iteration_limit(
            has_tool_results=True, has_reply=False, escape_used=False,
        )
        assert d.transition == Transition.SYNTHESIZE

    def test_stall_when_escape_already_used(self) -> None:
        policy = TransitionPolicy()
        d = policy.on_iteration_limit(
            has_tool_results=True, has_reply=False, escape_used=True,
        )
        assert d.transition == Transition.STALL

    def test_stall_when_no_tool_results(self) -> None:
        policy = TransitionPolicy()
        d = policy.on_iteration_limit(
            has_tool_results=False, has_reply=False, escape_used=False,
        )
        assert d.transition == Transition.STALL

    def test_stall_when_reply_exists(self) -> None:
        policy = TransitionPolicy()
        d = policy.on_iteration_limit(
            has_tool_results=True, has_reply=True, escape_used=False,
        )
        assert d.transition == Transition.STALL


class TestOnSynthesisInvalid:
    def test_retry_below_limit(self) -> None:
        policy = TransitionPolicy(max_synthesis_leak_retries=2)
        d = policy.on_synthesis_invalid("tool_call_leak", retry_count=1)
        assert d.transition == Transition.SYNTHESIZE

    def test_stall_at_limit(self) -> None:
        policy = TransitionPolicy(max_synthesis_leak_retries=2)
        d = policy.on_synthesis_invalid("tool_call_leak", retry_count=2)
        assert d.transition == Transition.STALL

    def test_stall_above_limit(self) -> None:
        policy = TransitionPolicy(max_synthesis_leak_retries=2)
        d = policy.on_synthesis_invalid("protocol_tag", retry_count=5)
        assert d.transition == Transition.STALL

    def test_custom_limit(self) -> None:
        policy = TransitionPolicy(max_synthesis_leak_retries=5)
        d = policy.on_synthesis_invalid("tool_call_leak", retry_count=4)
        assert d.transition == Transition.SYNTHESIZE
        d = policy.on_synthesis_invalid("tool_call_leak", retry_count=5)
        assert d.transition == Transition.STALL


class TestPolicyIsInjectable:
    def test_default_policy_frozen(self) -> None:
        policy = TransitionPolicy()
        assert policy.compute_gate_patience == 2
        assert policy.max_replan_per_turn == 2
        assert policy.max_synthesis_leak_retries == 2

    def test_custom_thresholds(self) -> None:
        policy = TransitionPolicy(
            compute_gate_patience=5,
            max_replan_per_turn=4,
            max_synthesis_leak_retries=3,
        )
        assert policy.compute_gate_patience == 5
        assert policy.max_replan_per_turn == 4
        assert policy.max_synthesis_leak_retries == 3
