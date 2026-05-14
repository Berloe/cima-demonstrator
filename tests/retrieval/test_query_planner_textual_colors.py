from __future__ import annotations

from cima_demo.retrieval.query_planner import TaskSpecBuilder


def test_what_colors_query_is_textual_not_numeric_contract() -> None:
    spec = TaskSpecBuilder().build(
        "What colors are worn by the Oregon Duck, mascot of the University of Oregon athletic teams?"
    )
    assert spec.output_contract.format == "text"
