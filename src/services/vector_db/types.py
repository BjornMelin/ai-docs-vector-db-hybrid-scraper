"""Lightweight dataclasses representing vector store payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class CollectionSchema:
    """Schema definition for a vector collection."""

    name: str
    vector_size: int
    distance: str = "cosine"
    requires_sparse: bool = False


@dataclass(slots=True)
class TextDocument:
    """Representation of text slated for indexing."""

    id: str
    content: str
    metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class VectorRecord:
    """Dense (and optional sparse) vector payload."""

    id: str
    vector: Sequence[float]
    payload: Mapping[str, object] | None = None
    sparse_vector: Mapping[int, float] | None = None


@dataclass(slots=True)
class VectorMatch:
    """Normalized representation of a vector search result."""

    id: str
    score: float
    payload: Mapping[str, object] | None
    vector: Sequence[float] | None = None
    raw_score: float | None = None
    collection: str | None = None
    normalized_score: float | None = None
