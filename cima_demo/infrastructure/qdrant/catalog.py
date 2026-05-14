from __future__ import annotations

"""Canonical Qdrant collection catalog for the witness-backend data plane.

The approved witness backend keeps different semantic roles physically separate.
This module centralises the collection names so the runtime, GC workers and tests
all agree on the same layout.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class QdrantCollectionCatalog:
    local_citems: str = "cima_local_citems"
    local_summaries: str = "cima_local_summaries"
    chunks: str = "cima_chunks"
    global_citems: str = "cima_global_citems"
    global_summaries: str = "cima_global_summaries"
    ephemeral: str = "cima_ephemeral"

    def all(self) -> tuple[str, ...]:
        return (
            self.local_citems,
            self.local_summaries,
            self.chunks,
            self.global_citems,
            self.global_summaries,
            self.ephemeral,
        )

    def local_scoped(self) -> tuple[str, ...]:
        return (
            self.local_citems,
            self.local_summaries,
            self.chunks,
            self.ephemeral,
        )

    def geometry_eligible(self) -> tuple[str, ...]:
        return (self.local_citems, self.local_summaries)

    @classmethod
    def from_settings(cls, settings: object) -> "QdrantCollectionCatalog":
        return cls(
            local_citems=getattr(settings, "qdrant_local_citems_collection", cls.local_citems),
            local_summaries=getattr(settings, "qdrant_local_summaries_collection", cls.local_summaries),
            chunks=getattr(settings, "qdrant_chunks_collection", cls.chunks),
            global_citems=getattr(settings, "qdrant_global_citems_collection", cls.global_citems),
            global_summaries=getattr(settings, "qdrant_global_summaries_collection", cls.global_summaries),
            ephemeral=getattr(settings, "qdrant_ephemeral_collection", cls.ephemeral),
        )


def local_conversation_scoped_collections(catalog: QdrantCollectionCatalog) -> Iterable[str]:
    return catalog.local_scoped()
