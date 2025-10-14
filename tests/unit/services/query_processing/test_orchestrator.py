"""Unit tests for the streamlined SearchOrchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.contracts.retrieval import SearchRecord
from src.models.search import SearchRequest
from src.services.query_processing.orchestrator import SearchOrchestrator
from src.services.rag import RAGConfig as ServiceRAGConfig
from src.services.vector_db.service import VectorStoreService


class VectorServiceStub:
    """Minimal vector service stub with deterministic responses."""

    def __init__(self, collection: str = "documentation") -> None:
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
    ) -> list[SearchRecord]:
        assert collection == self._collection
        assert kwargs.get("group_by") == "doc_id"
        record = SearchRecord(
            id="doc-1",
            score=0.9,
            raw_score=0.9,
            content=f"Snippet for {query}",
            title="Example",
            metadata={
                "content": f"Snippet for {query}",
                "title": "Example",
                "doc_id": "doc-1",
                "_grouping": {"applied": True, "group_id": "doc-1", "rank": 1},
            },
            collection=self._collection,
            group_id="doc-1",
            group_rank=1,
            grouping_applied=True,
        )
        return [record]

    async def list_collections(self) -> list[str]:
        return [self._collection]


@pytest.mark.asyncio
async def test_search_returns_results_with_collection_field() -> None:
    """Verify search results include collection metadata in returned records."""

    service = VectorServiceStub("articles")
    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, service)
    )
    await orchestrator.initialize()

    request = SearchRequest(
        query="vector databases",
        collection=None,
        limit=3,
        offset=0,
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
    """Verify the orchestrator falls back to list_collections() without a default."""

    service = VectorServiceStub("knowledge")
    service.config = SimpleNamespace()  # no qdrant section
    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, service)
    )
    await orchestrator.initialize()

    request = SearchRequest(
        query="missing defaults",
        collection=None,
        limit=10,
        offset=0,
        enable_expansion=False,
    )

    result = await orchestrator.search(request)

    assert result.records[0].collection == "knowledge"


@pytest.mark.asyncio
async def test_query_expansion_applied_when_enabled() -> None:
    """Verify query expansion is applied when the feature flag is enabled."""

    service = VectorServiceStub()
    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, service)
    )
    await orchestrator.initialize()

    request = SearchRequest(
        query="install widget",
        limit=10,
        offset=0,
        enable_expansion=True,
    )

    result = await orchestrator.search(request)

    assert "setup" in result.query
    assert "widgets" in result.query
    assert "query_expansion" in result.features_used


@pytest.mark.asyncio
async def test_search_with_rag_pipeline(monkeypatch) -> None:
    """Verify the search orchestrator integrates with the LangGraph RAG pipeline."""

    service = VectorServiceStub()
    rag_config = ServiceRAGConfig()

    captured: dict[str, Any] = {}

    class DummyPipeline:
        """Stub pipeline capturing invocation arguments."""

        def __init__(
            self, vector_service: VectorServiceStub
        ) -> None:  # pragma: no cover - simple
            assert vector_service is service

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "answer": "Synthesised result",
                "confidence": 0.42,
                "sources": [{"source_id": "doc-1"}],
            }

    monkeypatch.setattr(
        "src.services.query_processing.orchestrator.LangGraphRAGPipeline",
        DummyPipeline,
    )

    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, service),
        rag_config=rag_config,
    )
    await orchestrator.initialize()

    request = SearchRequest(
        query="vector rag",
        limit=3,
        offset=0,
        enable_expansion=False,
        enable_rag=True,
        rag_top_k=2,
    )

    result = await orchestrator.search(request)

    assert result.generated_answer == "Synthesised result"
    assert result.answer_confidence == pytest.approx(0.42)
    assert result.answer_sources == [{"source_id": "doc-1"}]
    assert "rag_answer_generation" in result.features_used
    assert captured["query"] == "vector rag"
    assert len(captured["prefetched_records"]) == 1
