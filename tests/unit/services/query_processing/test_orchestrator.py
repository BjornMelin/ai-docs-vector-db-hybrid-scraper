"""Unit tests for the streamlined SearchOrchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.query_processing.expansion import QueryExpansionResult
from src.services.query_processing.models import SearchRequest
from src.services.query_processing.orchestrator import SearchOrchestrator
from src.services.vector_db.adapter_base import VectorMatch


class VectorServiceStub:
    """Minimal vector service stub with deterministic responses."""

    def __init__(self, collection: str = "docs") -> None:
        self._collection = collection
        self.config = SimpleNamespace(
            qdrant=SimpleNamespace(collection_name=collection)
        )
        self._initialized = False

    async def initialize(self) -> None:  # pragma: no cover - simple stub
        self._initialized = True

    async def cleanup(self) -> None:  # pragma: no cover - simple stub
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    async def search_documents(
        self,
        collection: str,
        query: str,
        **kwargs: Any,
    ) -> list[VectorMatch]:
        assert collection == self._collection
        assert kwargs.get("group_by") == "doc_id"
        match = VectorMatch(
            id="doc-1",
            score=0.9,
            raw_score=0.9,
            payload={
                "content": f"Snippet for {query}",
                "title": "Example",
                "doc_id": "doc-1",
                "_grouping": {"applied": True, "group_id": "doc-1", "rank": 1},
            },
            collection=self._collection,
        )
        return [match]

    async def list_collections(self) -> list[str]:
        return [self._collection]


@pytest.mark.asyncio
async def test_search_returns_results_with_collection_field() -> None:
    service = VectorServiceStub("articles")
    orchestrator = SearchOrchestrator(vector_store_service=service)  # type: ignore
    await orchestrator.initialize()

    request = SearchRequest(
        query="vector databases",
        collection=None,
        limit=3,
        enable_expansion=False,
    )

    result = await orchestrator.search(request)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.collection == "articles"
    assert record.raw_score == pytest.approx(0.9)
    assert record.grouping_applied is True
    assert record.normalized_score is None
    assert result.features_used == []


@pytest.mark.asyncio
async def test_search_uses_list_collections_when_default_missing() -> None:
    service = VectorServiceStub("knowledge")
    service.config = SimpleNamespace()  # no qdrant section
    orchestrator = SearchOrchestrator(vector_store_service=service)  # type: ignore
    await orchestrator.initialize()

    request = SearchRequest(
        query="missing defaults",
        collection=None,
        limit=10,
        enable_expansion=False,
    )

    result = await orchestrator.search(request)

    assert result.records[0].collection == "knowledge"


@pytest.mark.asyncio
async def test_query_expansion_applied_when_enabled(monkeypatch) -> None:
    service = VectorServiceStub("docs")
    orchestrator = SearchOrchestrator(vector_store_service=service)  # type: ignore
    await orchestrator.initialize()

    expansion_result = QueryExpansionResult(
        original_query="python",
        expanded_query="python OR programming",
        expanded_terms=[],
        confidence_score=0.7,
    )
    mock_expand = AsyncMock(return_value=expansion_result)
    monkeypatch.setattr(
        type(orchestrator._expansion_service),  # pylint: disable=protected-access
        "expand_query",
        mock_expand,
    )

    request = SearchRequest(query="python", limit=10, enable_expansion=True)

    result = await orchestrator.search(request)

    assert result.query == "python OR programming"
    assert "query_expansion" in result.features_used
    mock_expand.assert_awaited_once()
