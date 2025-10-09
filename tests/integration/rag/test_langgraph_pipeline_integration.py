"""Integration tests for the LangGraph RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from src.services.query_processing.models import SearchRequest
from src.services.rag import (
    AnswerMetrics,
    LangGraphRAGPipeline,
    RAGConfig,
    RAGGenerator,
    RAGRequest,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import VectorMatch


class _StubVectorService:
    """Minimal vector service stub returning predefined matches."""

    def __init__(self, matches: Sequence[VectorMatch], collection: str) -> None:
        self._matches = list(matches)
        self._collection = collection
        self._initialized = False

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
    ) -> list[VectorMatch]:
        assert collection == self._collection
        assert isinstance(query, str)
        return list(self._matches)[:limit]


class _StubGenerator:
    """Small RAG generator used to observe pipeline integration."""

    def __init__(self, config: RAGConfig, retriever) -> None:
        self._config = config
        self._retriever = retriever
        self._callbacks = []
        self.initialized = False
        self.callback_registered = False

    def register_callbacks(self, callbacks) -> None:  # pragma: no cover - tiny hook
        self._callbacks = list(callbacks)
        self.callback_registered = bool(callbacks)

    async def initialize(self) -> None:
        self.initialized = True

    async def cleanup(self) -> None:
        self.initialized = False

    async def generate_answer(self, request: RAGRequest) -> RAGResult:
        assert self.initialized
        documents = await self._retriever.ainvoke(request.query)
        answer = f"Answer for: {request.query}"
        sources = [
            SourceAttribution(
                source_id=doc.metadata.get("source_id", "unknown"),
                title=str(doc.metadata.get("title", "Untitled")),
                url=doc.metadata.get("url"),
                excerpt=doc.page_content[:60],
                score=doc.metadata.get("score"),
            )
            for doc in documents
        ]
        return RAGResult(
            answer=answer,
            confidence_score=0.87,
            sources=sources,
            generation_time_ms=42.0,
            metrics=AnswerMetrics(
                total_tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
                generation_time_ms=42.0,
            ),
        )

    def get_metrics(self) -> RAGServiceMetrics:
        return RAGServiceMetrics(
            generation_count=1,
            avg_generation_time_ms=42.0,
            total_generation_time_ms=42.0,
        )


@pytest.mark.asyncio
async def test_pipeline_generates_answer_from_vector_matches() -> None:
    """LangGraph pipeline should orchestrate retrieval and generation."""

    matches = [
        VectorMatch(
            id="doc-1",
            score=0.91,
            payload={
                "content": "FastAPI integrates with Pydantic for validation.",
                "title": "FastAPI Validation",
                "url": "https://example.com/fastapi",
            },
        ),
        VectorMatch(
            id="doc-2",
            score=0.88,
            payload={
                "content": "LangGraph orchestrates retrieval and generation stages.",
                "title": "LangGraph Overview",
                "url": "https://example.com/langgraph",
            },
        ),
    ]
    stub_vector_service = _StubVectorService(matches, collection="docs")
    await stub_vector_service.initialize()

    def generator_factory(config: RAGConfig, retriever) -> RAGGenerator:
        return cast(RAGGenerator, _StubGenerator(config, retriever))

    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, stub_vector_service),
        generator_factory=generator_factory,
    )

    request = SearchRequest(
        query="How does FastAPI use Pydantic?",
        collection="docs",
        limit=2,
    )
    rag_config = RAGConfig(model="test-model", compression_enabled=False)

    result = await pipeline.run(
        query=request.query,
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    assert result is not None
    assert result["answer"].startswith("Answer for")
    assert result["confidence"] == pytest.approx(0.87)
    assert len(result["sources"]) == 2
    first_source = result["sources"][0]
    assert first_source["source_id"] == "doc-1"
    assert "FastAPI" in first_source["excerpt"]


@pytest.mark.asyncio
async def test_pipeline_returns_none_when_no_documents() -> None:
    """Pipeline should escalate when retrieval yields no documents."""

    stub_vector_service = _StubVectorService(matches=[], collection="docs")
    await stub_vector_service.initialize()

    def generator_factory(config: RAGConfig, retriever) -> RAGGenerator:
        return cast(RAGGenerator, _StubGenerator(config, retriever))

    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, stub_vector_service),
        generator_factory=generator_factory,
    )

    request = SearchRequest(query="Missing docs", collection="docs", limit=1)
    rag_config = RAGConfig(model="test-model", compression_enabled=False)

    result = await pipeline.run(
        query=request.query,
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    assert result is None
