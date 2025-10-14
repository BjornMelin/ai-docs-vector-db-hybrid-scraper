"""Integration tests for the search orchestrator with LangGraph pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from src.contracts.retrieval import SearchRecord
from src.models.search import SearchRequest
from src.services.query_processing.orchestrator import (
    SearchOrchestrator,
    ServiceRAGConfig,
)
from src.services.rag.langgraph_pipeline import LangGraphRAGPipeline
from src.services.vector_db.service import VectorStoreService


class _StubVectorService:
    """Vector service stub returning deterministic matches."""

    def __init__(self, matches, collection: str) -> None:
        self._matches = list(matches)
        self._collection = collection
        self._initialized = False
        self.config = SimpleNamespace(
            qdrant=SimpleNamespace(collection_name=collection)
        )

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:  # pragma: no cover - symmetry
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        filters: dict | None = None,
        group_by: str | None = None,
        group_size: int | None = None,
        overfetch_multiplier: float | None = None,
        normalize_scores: bool | None = None,
    ):
        assert collection == self._collection
        assert query
        return list(self._matches)[:limit]

    async def list_collections(self) -> list[str]:  # pragma: no cover - defensive
        return [self._collection]


class _StubRAGPipeline:
    """LangGraph pipeline stub capturing invocations."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.invocations = []

    async def run(self, **kwargs) -> dict[str, object] | None:
        self.invocations.append(kwargs)
        return {
            "answer": self.answer,
            "confidence": 0.78,
            "sources": [
                {
                    "source_id": "doc-1",
                    "title": "Stub Source",
                    "url": "https://example.com",
                    "excerpt": "Stub excerpt",
                    "score": 0.91,
                }
            ],
        }


@pytest.mark.asyncio
async def test_orchestrator_generates_rag_answer_with_stub_pipeline() -> None:
    """SearchOrchestrator should delegate to LangGraph pipeline when enabled."""

    matches = [
        SearchRecord(
            id="doc-1",
            score=0.92,
            content="LangGraph integrates retrieval and generation flows.",
            title="LangGraph Overview",
            metadata={
                "content": "LangGraph integrates retrieval and generation flows.",
                "doc_id": "doc-1",
            },
            collection="docs",
        ),
        SearchRecord(
            id="doc-2",
            score=0.84,
            content="FastAPI can front a LangGraph RAG workflow.",
            title="FastAPI & LangGraph",
            metadata={
                "content": "FastAPI can front a LangGraph RAG workflow.",
                "doc_id": "doc-2",
            },
            collection="docs",
        ),
    ]
    stub_vector_service = _StubVectorService(matches, collection="docs")
    rag_pipeline = _StubRAGPipeline("LangGraph provides structured RAG outputs.")

    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, stub_vector_service),
        rag_config=ServiceRAGConfig(
            model="test-model",
            retriever_top_k=2,
            include_sources=True,
            compression_enabled=False,
        ),
    )
    orchestrator._rag_pipeline = cast(LangGraphRAGPipeline, rag_pipeline)  # type: ignore[attr-defined]

    await orchestrator.initialize()

    request = SearchRequest(
        query="How does LangGraph orchestrate RAG?",
        collection="docs",
        limit=2,
        enable_rag=True,
        rag_top_k=2,
        normalize_scores=False,
    )

    response = await orchestrator.search(request)

    assert response.generated_answer == "LangGraph provides structured RAG outputs."
    assert response.answer_confidence == pytest.approx(0.78)
    assert len(response.answer_sources or []) == 1
    assert response.total_results == 2
    assert response.records[0].title == "LangGraph Overview"
    assert rag_pipeline.invocations, "Pipeline should be invoked"
    invocation = rag_pipeline.invocations[0]
    assert invocation["collection"] == "docs"
    assert len(invocation["prefetched_records"]) == 2


@pytest.mark.asyncio
async def test_orchestrator_handles_missing_rag_result() -> None:
    """Orchestrator should return search-only response when pipeline yields nothing."""

    matches = [
        SearchRecord(
            id="doc-1",
            score=0.9,
            content="Doc content",
            title="Doc title",
            metadata={"content": "Doc content"},
            collection="docs",
        )
    ]
    stub_vector_service = _StubVectorService(matches, collection="docs")

    class _NullPipeline(_StubRAGPipeline):
        async def run(self, **kwargs) -> dict[str, object] | None:
            self.invocations.append(kwargs)
            return None

    rag_pipeline = _NullPipeline("unused")

    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, stub_vector_service),
        rag_config=ServiceRAGConfig(
            model="test-model",
            retriever_top_k=1,
            include_sources=True,
            compression_enabled=False,
        ),
    )
    orchestrator._rag_pipeline = cast(LangGraphRAGPipeline, rag_pipeline)  # type: ignore[attr-defined]

    await orchestrator.initialize()

    request = SearchRequest(
        query="Docs only",
        collection="docs",
        limit=1,
        enable_rag=True,
    )

    response = await orchestrator.search(request)

    assert response.generated_answer is None
    assert response.answer_sources is None
    assert response.total_results == 1
