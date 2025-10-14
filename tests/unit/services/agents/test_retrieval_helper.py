"""Tests for retrieval helper using container-managed vector service."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.services.agents.retrieval import RetrievalHelper, RetrievalQuery
from src.services.vector_db.service import VectorStoreService


class DummyMatch:
    """Vector search result stub."""

    def __init__(self, identifier: str, score: float) -> None:
        self.id = identifier
        self.score = score
        self.metadata = {"title": f"Doc {identifier}"}


class DummyVectorStoreService:
    """Vector store stub providing predictable search responses."""

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[DummyMatch]:
        assert collection == "docs"
        assert query == "what is langgraph"
        assert limit == 3
        assert filters == {"topic": "rag"}
        return [DummyMatch("1", 0.9), DummyMatch("2", 0.7)]


@pytest.mark.asyncio
async def test_fetch_uses_vector_service_directly() -> None:
    """Helper should normalise vector store results without the retired manager."""
    helper = RetrievalHelper(cast(VectorStoreService, DummyVectorStoreService()))
    query = RetrievalQuery(
        collection="docs",
        text="what is langgraph",
        top_k=3,
        filters={"topic": "rag"},
    )

    results = await helper.fetch(query)

    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].metadata == {"title": "Doc 1"}
    assert results[0].raw is not None
