"""Reproducible scenario harness for the CIMA Demonstrator."""

from __future__ import annotations

from .datasets import HarnessScenario, load_all_scenarios, load_scenario

__all__ = [
    "HarnessScenario",
    "DemoScenarioRunner",
    "ScenarioExecutionResult",
    "load_all_scenarios",
    "load_scenario",
    "run_harness",
]


def __getattr__(name: str):
    if name in {"DemoScenarioRunner", "ScenarioExecutionResult", "run_harness"}:
        from . import scenario_runner as _scenario_runner
        return getattr(_scenario_runner, name)
    raise AttributeError(name)
