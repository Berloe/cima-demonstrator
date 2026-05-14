"""User-facing message strings — single source of truth.

All strings that are directly visible to the end user (error messages,
status messages, recovery prompts) live here so they can be updated in
one place. No i18n framework — plain string constants.
"""
from __future__ import annotations

# ── AnswerFinalizer messages ──────────────────────────────────────────────────
ANSWER_EMPTY = (
    "I was unable to produce a complete answer from the gathered information. "
    "Please try rephrasing your question."
)
ANSWER_NO_EVIDENCE = (
    "I do not have enough evidence in the gathered context to answer reliably."
)
ANSWER_NO_COMPUTE = (
    "I was unable to verify the numeric result: the computation step did not "
    "complete successfully. Please try again."
)

# ── AgentOrchestrator messages ────────────────────────────────────────────────
TURN_TIMEOUT = (
    "La respuesta superó el límite de tiempo ({timeout}s). "
    "Prueba reformular la pregunta o reducir el alcance de la tarea."
)
DECOMPOSITION_PLAN_ACK = (
    "La tarea excede el presupuesto de contexto disponible. "
    "He creado un plan con {subtask_count} subtareas atómicas. "
    "Empezaré con la primera ahora."
)
DECOMPOSITION_FAILED_ACK = (
    "La tarea excede el presupuesto de contexto disponible y no fue posible "
    "descomponerla automáticamente. Por favor, reformula la pregunta en partes más pequeñas."
)

# ── CR-7: file attachment required but none provided ─────────────────────────
ANSWER_FILE_REQUIRED = (
    "This task requires a file attachment. Please attach the file you'd like me to "
    "analyze and I'll process it immediately."
)
