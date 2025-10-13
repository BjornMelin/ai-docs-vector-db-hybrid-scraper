"""Helpers for performing vector retrieval in agent workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from src.models.search import SearchRecord
from src.services.vector_db.service import VectorStoreService


@dataclass(slots=True)
class RetrievalQuery:
    """Parameters describing a retrieval request.

    Attributes:
        collection: Name of the target vector collection.
        text: Natural-language query text.
        top_k: Maximum number of results to return.
        filters: Optional metadata filter for vector search.
    """

    collection: str
    text: str
    top_k: int = 5
    filters: Mapping[str, object] | None = None


@dataclass(slots=True)
class RetrievedDocument:
    """Normalised representation of a retrieved document."""

    id: str
    score: float
    metadata: Mapping[str, object] | None
    raw: SearchRecord | None


class RetrievalHelper:
    """Query the configured vector store service."""

    def __init__(
        self,
        vector_service: VectorStoreService
        | Callable[[], Awaitable[VectorStoreService]],
    ) -> None:
        self._vector_service: VectorStoreService | None = None
        if callable(vector_service):
            self._vector_service_factory = vector_service
        else:
            self._vector_service = vector_service
            self._vector_service_factory = None

    async def _resolve_service(self) -> VectorStoreService:
        if self._vector_service is not None:
            return self._vector_service
        if self._vector_service_factory is None:
            msg = "Vector service resolver is not configured"
            raise RuntimeError(msg)
        service = await self._vector_service_factory()
        self._vector_service = service
        return service

    async def fetch(self, query: RetrievalQuery) -> Sequence[RetrievedDocument]:
        """Execute a dense retrieval query against the configured vector store.

        Args:
            query: Retrieval parameters including collection name, query text,
                document limit, and optional metadata filters.

        Returns:
            Sequence of ``RetrievedDocument`` items sorted by similarity score.
        """

        service = await self._resolve_service()
        records = await service.search_documents(
            query.collection,
            query.text,
            limit=max(1, query.top_k),
            filters=query.filters,
        )
        return [
            RetrievedDocument(
                id=record.id,
                score=record.score,
                metadata=record.metadata or {},
                raw=record,
            )
            for record in records
        ]


__all__ = ["RetrievedDocument", "RetrievalHelper", "RetrievalQuery"]
