"""Abstract interfaces and helper payloads for vector database adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CollectionSchema:
    """Schema definition for a vector collection.

    Attributes:
        name: Canonical collection identifier.
        vector_size: Dimensionality of the dense embedding vectors.
        distance: Distance metric identifier supported by the backend.
        requires_sparse: Flag indicating whether sparse vectors must be stored.
    """

    name: str
    vector_size: int
    distance: str = "cosine"
    requires_sparse: bool = False


@dataclass(slots=True)
class VectorRecord:
    """Payload representing a single vector insertion/upsert."""

    id: str
    vector: Sequence[float]
    payload: Mapping[str, Any] | None = None
    sparse_vector: Mapping[str, Any] | None = None


@dataclass(slots=True)
class VectorMatch:
    """Normalized representation of a vector search result."""

    id: str
    score: float
    payload: Mapping[str, Any] | None
    vector: Sequence[float] | None = None
    raw_score: float | None = None
    collection: str | None = None
    normalized_score: float | None = None


@dataclass(slots=True)
class TextDocument:
    """Lightweight representation of text to be embedded and indexed."""

    id: str
    content: str
    metadata: Mapping[str, Any] | None = None


class VectorAdapter(ABC):
    """Abstract adapter interface for vector database operations."""

    @abstractmethod
    async def create_collection(self, schema: CollectionSchema) -> None:
        """Create or update a vector collection based on the supplied schema."""

    @abstractmethod
    async def drop_collection(self, name: str, *, missing_ok: bool = True) -> None:
        """Drop a collection, optionally ignoring missing collections."""

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """Return the available collection identifiers."""

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Return True when the supplied collection identifier is present."""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        records: Sequence[VectorRecord],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Insert or update a batch of vectors within a collection."""

    @abstractmethod
    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        """Delete vectors from a collection by identifiers or metadata filters."""

    @abstractmethod
    async def query(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Execute a dense vector similarity search."""

    @abstractmethod
    async def hybrid_query(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float] | None = None,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Execute a hybrid sparse/dense search if supported."""

    def supports_query_groups(self) -> bool:
        """Return True when server-side grouping capabilities are available."""

        return False

    async def query_groups(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        vector: Sequence[float],
        *,
        group_by: str,
        limit: int = 10,
        group_size: int = 1,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[list[VectorMatch], bool]:
        """Attempt a grouped query returning matches and a success flag.

        Base adapters default to a regular similarity search and report that
        grouping was not applied. Concrete implementations should override this
        when the backend supports server-side grouping primitives.
        """

        matches = await self.query(
            collection,
            vector,
            limit=limit,
            filters=filters,
        )
        return matches, False

    @abstractmethod
    async def get_collection_stats(self, name: str) -> Mapping[str, Any]:
        """Return backend-specific statistics for a collection."""

    @abstractmethod
    async def retrieve(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        ids: Sequence[str],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[VectorMatch]:
        """Fetch records by identifier."""

    @abstractmethod
    async def recommend(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        *,
        positive_ids: Sequence[str] | None = None,
        vector: Sequence[float] | None = None,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Return records related to the supplied examples."""

    @abstractmethod
    async def scroll(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        *,
        limit: int = 64,
        offset: str | None = None,
        filters: Mapping[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[VectorMatch], str | None]:
        """Iterate over records using cursor-based pagination."""
