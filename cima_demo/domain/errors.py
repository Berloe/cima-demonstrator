"""Domain exceptions (KIMA_Domain_CIMA_v0.10 §9)."""
from __future__ import annotations


class KimaError(Exception):
    """Base exception for all KIMA errors."""


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLMError(KimaError):
    """Generic LLM error."""

class LLMUnavailableError(LLMError):
    """LLM service unreachable or returning 5xx."""

class LLMContextOverflowError(LLMError):
    """Prompt exceeds model ctx_size."""


# ── Embedding ─────────────────────────────────────────────────────────────────

class EmbeddingError(KimaError):
    """Generic embedding error."""

class EmbeddingUnavailableError(EmbeddingError):
    """TEI embedding service unreachable."""


# ── CItemStore ────────────────────────────────────────────────────────────────

class CItemStoreError(KimaError):
    """Generic CItemStore (Qdrant) error."""

class CItemNotFoundError(CItemStoreError):
    """C-Item not found in the store."""


# ── RelDB ─────────────────────────────────────────────────────────────────────

class RelDBError(KimaError):
    """Generic relational DB error."""


# ── Reranker ──────────────────────────────────────────────────────────────────

class RerankerError(KimaError):
    """Generic reranker error."""

class RerankerUnavailableError(RerankerError):
    """TEI reranker unreachable or returning 5xx. Triggers graceful degradation."""


# ── NLI ───────────────────────────────────────────────────────────────────────

class NLIError(KimaError):
    """Generic NLI classification error."""

class NLIUnavailableError(NLIError):
    """TEI NLI service unreachable or returning 5xx."""


# ── Geometric Expansion ───────────────────────────────────────────────────────

class GeometricExpansionError(KimaError):
    """Error during geometric graph expansion."""


# ── Chunking ──────────────────────────────────────────────────────────────────

class ChunkingError(KimaError):
    """Error during document chunking."""


# ── Sparse Embedding ─────────────────────────────────────────────────────────

class SparseEmbeddingError(KimaError):
    """Error during sparse (SPLADE/BM25) embedding."""


# ── EventBus ──────────────────────────────────────────────────────────────────

class EventBusError(KimaError):
    """Error publishing to or consuming from EventBus."""


# ── Application-level ────────────────────────────────────────────────────────

class TurnInProgressError(KimaError):
    """Turn mutex already held for this conversation."""

class ConflictDetectionError(KimaError):
    """Error during conflict detection (non-fatal, logged)."""

class PlanError(KimaError):
    """Error in plan creation or execution."""

class ToolDispatchError(KimaError):
    """Error dispatching a tool call."""

class IngestError(KimaError):
    """Error ingesting a C-Item."""

class WebSearchError(KimaError):
    """Error calling the web search service."""

class FileProcessingError(KimaError):
    """Error extracting text from a file."""
