"""Helpers for performing vector retrieval in agent workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.contracts.retrieval import SearchRecord
from src.infrastructure.client_manager import ClientManager


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
    """Query the configured vector store through the client manager."""

    def __init__(self, client_manager: ClientManager) -> None:
        self._client_manager = client_manager

    async def fetch(self, query: RetrievalQuery) -> Sequence[RetrievedDocument]:
        """Execute a dense retrieval query against the configured vector store.

        Args:
            query: Retrieval parameters including collection name, query text,
                document limit, and optional metadata filters.

        Returns:
            Sequence of ``RetrievedDocument`` items sorted by similarity score.
        """

        service = await self._client_manager.get_vector_store_service()
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
