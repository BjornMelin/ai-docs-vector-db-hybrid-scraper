"""Performance smoke tests for the LangGraph RAG pipeline."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import cast

import pytest

from src.services.monitoring.metrics import set_metrics_registry
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
    """Vector service returning deterministic matches."""

    def __init__(self, matches: Sequence[VectorMatch], collection: str) -> None:
        self._matches = list(matches)
        self._collection = collection
        self._initialized = False

    async def initialize(self) -> None:  # pragma: no cover - symmetry
        self._initialized = True

    async def cleanup(self) -> None:  # pragma: no cover - symmetry
        self._initialized = False

    def is_initialized(self) -> bool:  # pragma: no cover - diagnostic helper
        return self._initialized

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        **_: object,
    ) -> list[VectorMatch]:
        assert collection == self._collection
        assert isinstance(query, str)
        return list(self._matches)[:limit]


class _StubGenerator:
    """Minimal generator used to exercise the pipeline."""

    def __init__(self, retriever) -> None:
        self._retriever = retriever
        self.initialized = False

    def register_callbacks(self, callbacks) -> None:  # pragma: no cover - hook
        assert callbacks is not None

    async def initialize(self) -> None:
        self.initialized = True

    async def cleanup(self) -> None:
        self.initialized = False

    async def generate_answer(self, request: RAGRequest) -> RAGResult:
        assert self.initialized
        documents = await self._retriever.ainvoke(request.query)
        sources = [
            SourceAttribution(
                source_id=doc.metadata.get("source_id", "unknown"),
                title=str(doc.metadata.get("title", "Untitled")),
                url=doc.metadata.get("url"),
                excerpt=doc.page_content[:80],
                score=doc.metadata.get("score"),
            )
            for doc in documents
        ]
        return RAGResult(
            answer=f"Answer for: {request.query}",
            confidence_score=0.9,
            sources=sources,
            generation_time_ms=18.0,
            metrics=AnswerMetrics(
                total_tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
                generation_time_ms=18.0,
            ),
        )

    def get_metrics(self) -> RAGServiceMetrics:
        return RAGServiceMetrics(
            generation_count=1,
            avg_generation_time_ms=18.0,
            total_generation_time_ms=18.0,
        )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_rag_pipeline_average_latency_under_budget() -> None:
    """Average pipeline latency should stay within the micro-benchmark budget."""

    set_metrics_registry(None)

    matches = [
        VectorMatch(
            id="doc-1",
            score=0.95,
            payload={
                "content": "LangGraph orchestrates retrieval and generation stages.",
                "title": "LangGraph Overview",
                "url": "https://example.com/langgraph",
                "score": 0.95,
            },
        ),
        VectorMatch(
            id="doc-2",
            score=0.92,
            payload={
                "content": "FastAPI integrates with Pydantic for validation.",
                "title": "FastAPI Validation",
                "url": "https://example.com/fastapi",
                "score": 0.92,
            },
        ),
    ]

    vector_service = _StubVectorService(matches, collection="docs")
    await vector_service.initialize()

    def generator_factory(config: RAGConfig, retriever) -> RAGGenerator:
        _ = config
        return cast(RAGGenerator, _StubGenerator(retriever))

    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, vector_service),
        generator_factory=generator_factory,
    )

    request = SearchRequest(
        query="How does LangGraph work?", collection="docs", limit=2
    )
    rag_config = RAGConfig(model="test-model", compression_enabled=False)

    # Warm-up run to avoid first-run noise.
    await pipeline.run(
        query=request.query,
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        result = await pipeline.run(
            query=request.query,
            request=request,
            rag_config=rag_config,
            collection="docs",
        )
        assert result is not None
        assert result["answer"].startswith("Answer for")
    duration = time.perf_counter() - start
    average_latency = duration / iterations

    # With fully stubbed dependencies the pipeline should execute within ~20ms.
    assert average_latency < 0.025, (
        f"Average latency {average_latency:.4f}s exceeds 25ms budget"
    )
