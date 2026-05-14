"""Domain events del orquestador KIMA (ADR-001 v3.4 §D-09, D-23, D-24).

Dos categorías de visibilidad:
  INTERNAL  — solo el kernel las procesa (reducer); nunca van al DomainEventPublisher.
  EXTERNAL  — publicadas al bus externo; dos subtipos de criticidad:
    CRITICAL     — deben entregarse antes de liberar el mutex del turno.
    BEST_EFFORT  — observabilidad; no bloquean el mutex si el bus falla.

Payload: dict[str, Any]. Cada tipo documenta sus claves en su constante _FIELDS.
Construcción siempre vía make_event() para garantizar el envelope correcto.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


# ── Envelope ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainEvent:
    """Evento de dominio del orquestador.

    Invariantes:
      correlation_id == turn_id  (traza completa del turno)
      occurred_at    en UTC
    """
    event_id:         str
    schema_version:   str               # "1.0" — incrementar al cambiar payload breaking
    event_type:       str               # TurnEventType.*
    conversation_id:  str
    turn_id:          str
    iteration:        int               # 0 en Bootstrap/Finalize
    causation_id:     str | None        # event_id del evento que causó este
    correlation_id:   str               # == turn_id
    occurred_at:      datetime
    payload:          dict[str, Any]


def make_event(
    event_type: str,
    conversation_id: str,
    turn_id: str,
    iteration: int,
    payload: dict[str, Any],
    causation_id: str | None = None,
) -> DomainEvent:
    """Construye un DomainEvent con envelope completo."""
    return DomainEvent(
        event_id=str(uuid.uuid4()),
        schema_version="1.0",
        event_type=event_type,
        conversation_id=conversation_id,
        turn_id=turn_id,
        iteration=iteration,
        causation_id=causation_id,
        correlation_id=turn_id,
        occurred_at=datetime.now(UTC),
        payload=payload,
    )


# ── Tipos de evento ───────────────────────────────────────────────────────────

class TurnEventType:
    """Constantes para event_type.

    INTERNAL: procesados solo por el reducer; no van al DomainEventPublisher.
    EXTERNAL / CRITICAL: se publican y bloquean liberación del mutex si fallan.
    EXTERNAL / BEST_EFFORT: se publican pero no bloquean mutex si fallan.
    """

    # ── INTERNAL ──────────────────────────────────────────────────────────────
    # Payload keys: phase (str)
    PHASE_ENTERED         = "PhaseEntered"
    # Payload keys: reason (str), succeeded (bool)
    TURN_TERMINATED       = "TurnTerminated"

    # ── EXTERNAL / BEST_EFFORT ────────────────────────────────────────────────
    # Payload keys: cognitive_phase (str), execution_mode (str | None)
    TURN_BOOTSTRAPPED     = "TurnBootstrapped"
    # Payload keys: context_tokens (int), history_turns (int)
    MONITOR_COMPLETED     = "MonitorCompleted"
    # Payload keys: blockers_count (int), alerts_count (int), impasse (str | None)
    ANALYSIS_COMPLETED    = "AnalysisCompleted"
    # Payload keys: kind (str), blocker_kinds (list[str])
    IMPASSE_RAISED        = "ImpasseRaised"
    # Payload keys: cognitive_mode (str), available_capabilities (list[str])
    DECISION_REQUESTED    = "DecisionRequested"
    # Payload keys: intention_kind (str), action_count (int), has_conclusions (bool)
    DECISION_RECEIVED     = "DecisionReceived"
    # Payload keys: capability_calls_count (int), gated_count (int)
    ACTION_BATCH_COMPILED = "ActionBatchCompiled"
    # Payload keys: capability_name (str), params_hash (str)
    CAPABILITY_DISPATCHED = "CapabilityDispatched"
    # Payload keys: capability_name (str), state_changed (bool), evidence_atoms (int)
    CAPABILITY_COMPLETED  = "CapabilityCompleted"
    # Payload keys: capability_name (str), blocker_kind (str)
    CAPABILITY_BLOCKED    = "CapabilityBlocked"
    # Payload keys: source_url (str), evidence_hash (str)
    EVIDENCE_CAPTURED     = "EvidenceCaptured"
    # Payload keys: types (list[str]), count (int)
    CONCLUSIONS_DECLARED  = "ConclusionsDeclared"
    # Payload keys: slot_name (str)
    SLOT_RESOLVED         = "SlotResolved"
    # Payload keys: item_count (int)
    OVERLAY_UPDATED       = "OverlayUpdated"
    # Payload keys: from_status (str), to_status (str)
    PLAN_TRANSITION_APPLIED = "PlanTransitionApplied"
    # Payload keys: schema_valid (bool)
    ANSWER_CANDIDATE_PRODUCED = "AnswerCandidateProduced"

    # ── EXTERNAL / BEST_EFFORT (continued) ───────────────────────────────────
    # Payload keys: tier (int), tokens_before (int), tokens_after (int), items_summarized (int)
    BUDGET_PRESSURE       = "BudgetPressure"
    # Payload keys: fragment_chars (int), continuation_attempt (int)
    SYNTHESIS_TRUNCATED   = "SynthesisTruncated"

    # ── EXTERNAL / CRITICAL ───────────────────────────────────────────────────
    # Payload keys: outcome (str), duration_ms (int), iterations (int)
    TURN_COMPLETED        = "TurnCompleted"
    # Payload keys: reason (str), recoverable (bool), error_class (str | None)
    TURN_FAILED           = "TurnFailed"
    # Payload keys: reason (str), subtask_count (int)
    # Critical: plan must be persisted before mutex release (T6 — aggressive decomposition)
    BUDGET_IMPASSE        = "BudgetImpasse"


# Conjunto de tipos externos: todo lo que no es INTERNAL.
INTERNAL_EVENT_TYPES: frozenset[str] = frozenset({
    TurnEventType.PHASE_ENTERED,
    TurnEventType.TURN_TERMINATED,
})

# Conjunto de tipos críticos: deben publicarse antes de liberar el mutex.
CRITICAL_EVENT_TYPES: frozenset[str] = frozenset({
    TurnEventType.TURN_COMPLETED,
    TurnEventType.TURN_FAILED,
    TurnEventType.BUDGET_IMPASSE,   # T6: plan must be persisted before next turn starts
})
