"""Default contextuality weights for PhasePolicy (CIMA §2.3, SP-INV-01..05).

Edit this file to tune per-phase/per-type contextuality without touching
domain logic. The values are loaded by PhasePolicy at import time.
Format: (phase, item_type, weight) — weight in [0.0, 1.0].
"""
from __future__ import annotations

DEFAULT_CONTEXTUALITY: tuple[tuple[str, str, float], ...] = (
    # ── RECALL phase ──────────────────────────────────────────────────────────
    # RECALL: facts and derived results are most relevant
    ("RECALL", "FACT",        1.0),
    ("RECALL", "DERIVED",     1.0),
    ("RECALL", "OBSERVATION", 0.9),
    ("RECALL", "HYPOTHESIS",  0.8),
    ("RECALL", "DECISION",    0.7),
    ("RECALL", "CONSTRAINT",  0.6),
    ("RECALL", "PROCEDURE",   0.5),
    ("RECALL", "ASSUMPTION",  0.3),
    # PLANNING: decisions and constraints drive the plan
    ("PLANNING", "DECISION",    1.0),
    ("PLANNING", "CONSTRAINT",  1.0),
    ("PLANNING", "PROCEDURE",   0.9),
    ("PLANNING", "FACT",        0.6),
    ("PLANNING", "DERIVED",     0.6),
    ("PLANNING", "HYPOTHESIS",  0.5),
    ("PLANNING", "OBSERVATION", 0.4),
    ("PLANNING", "ASSUMPTION",  0.3),
    # EXECUTION: procedures and constraints govern what to do
    ("EXECUTION", "PROCEDURE",   1.0),
    ("EXECUTION", "CONSTRAINT",  0.8),
    ("EXECUTION", "DECISION",    0.7),
    ("EXECUTION", "OBSERVATION", 0.6),
    ("EXECUTION", "FACT",        0.5),
    ("EXECUTION", "DERIVED",     0.5),
    ("EXECUTION", "HYPOTHESIS",  0.3),
    ("EXECUTION", "ASSUMPTION",  0.2),
    # SYNTHESIS: observations, facts and derived results feed the summary
    ("SYNTHESIS", "OBSERVATION", 1.0),
    ("SYNTHESIS", "FACT",        0.9),
    ("SYNTHESIS", "DERIVED",     0.9),
    ("SYNTHESIS", "HYPOTHESIS",  0.8),
    ("SYNTHESIS", "DECISION",    0.6),
    ("SYNTHESIS", "CONSTRAINT",  0.5),
    ("SYNTHESIS", "PROCEDURE",   0.4),
    ("SYNTHESIS", "ASSUMPTION",  0.2),
    # IDLE: phase-neutral
    ("IDLE", "FACT",        0.5),
    ("IDLE", "DERIVED",     0.5),
    ("IDLE", "OBSERVATION", 0.5),
    ("IDLE", "HYPOTHESIS",  0.5),
    ("IDLE", "DECISION",    0.5),
    ("IDLE", "CONSTRAINT",  0.5),
    ("IDLE", "PROCEDURE",   0.5),
    ("IDLE", "ASSUMPTION",  0.3),
)
