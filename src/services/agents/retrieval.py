"""Helpers for performing vector retrieval in agent workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.infrastructure.client_manager import ClientManager
from src.services.vector_db.types import VectorMatch


@dataclass(slots=True)
class RetrievalQuery:
    """Parameters for retrieving supporting documents."""

    collection: str
    text: str
    top_k: int = 5
    filters: Mapping[str, object] | None = None


@dataclass(slots=True)
class RetrievedDocument:
    """Normalised representation of a retrieved document."""

    id: str
    score: float
    payload: Mapping[str, object] | None
    raw: VectorMatch | None


class RetrievalHelper:
    """Perform Qdrant-backed retrieval via the client manager."""

    def __init__(self, client_manager: ClientManager) -> None:
        self._client_manager = client_manager

    async def fetch(self, query: RetrievalQuery) -> Sequence[RetrievedDocument]:
        """Execute a dense retrieval query against the configured vector store."""

        service = await self._client_manager.get_vector_store_service()
        matches = await service.search_documents(
            query.collection,
            query.text,
            limit=max(1, query.top_k),
            filters=query.filters,
        )
        return [
            RetrievedDocument(
                id=match.id,
                score=match.score,
                payload=match.payload,
                raw=match,
            )
            for match in matches
        ]


__all__ = ["RetrievedDocument", "RetrievalHelper", "RetrievalQuery"]
