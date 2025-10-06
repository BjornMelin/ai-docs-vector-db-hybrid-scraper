"""Service-level integration tests for the LangGraph RAG stack."""

from __future__ import annotations

import time
from typing import cast

import pytest
from prometheus_client import CollectorRegistry

from src.services.monitoring.metrics import (
    MetricsConfig,
    MetricsRegistry,
    set_metrics_registry,
)
from src.services.query_processing.models import SearchRequest
from src.services.query_processing.orchestrator import (
    SearchOrchestrator,
    ServiceRAGConfig,
)
from src.services.rag.generator import RAGGenerator
from src.services.rag.langgraph_pipeline import LangGraphRAGPipeline
from src.services.rag.models import (
    AnswerMetrics,
    RAGConfig,
    RAGRequest,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import VectorMatch


class _StubVectorService:
    """Deterministic vector service for integration tests."""

    def __init__(self, matches: list[VectorMatch], collection: str) -> None:
        self._matches = matches
        self._collection = collection
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    def is_initialized(self) -> bool:
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
        return self._matches[:limit]


class _StubGenerator:
    """Simple generator returning predictable answers."""

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
            answer=f"Answer for {request.query}",
            confidence_score=0.9,
            sources=sources,
            generation_time_ms=21.0,
            metrics=AnswerMetrics(
                total_tokens=None,
                prompt_tokens=None,
                completion_tokens=None,
                generation_time_ms=21.0,
            ),
        )

    def get_metrics(self) -> RAGServiceMetrics:
        return RAGServiceMetrics(
            generation_count=1,
            avg_generation_time_ms=21.0,
            total_generation_time_ms=21.0,
        )


@pytest.mark.asyncio
async def test_services_emit_metrics_for_successful_run() -> None:
    """Running the orchestrator should populate the metrics registry."""

    matches = [
        VectorMatch(
            id="doc-1",
            score=0.93,
            payload={
                "content": "LangGraph orchestrates retrieval and generation stages.",
                "title": "LangGraph Overview",
                "url": "https://example.com/langgraph",
                "score": 0.93,
            },
        )
    ]

    vector_service = _StubVectorService(matches, collection="docs")
    await vector_service.initialize()

    def generator_factory(config: RAGConfig, retriever) -> RAGGenerator:
        _ = config
        return cast(RAGGenerator, _StubGenerator(retriever))

    registry = MetricsRegistry(
        MetricsConfig(namespace="service", enabled=True),
        registry=CollectorRegistry(),
    )
    set_metrics_registry(registry)

    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, vector_service),
        generator_factory=generator_factory,
    )

    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, vector_service),
        rag_config=ServiceRAGConfig(
            model="test-model",
            retriever_top_k=3,
            include_sources=True,
            compression_enabled=False,
        ),
    )
    orchestrator._rag_pipeline = pipeline  # type: ignore[attr-defined]

    await orchestrator.initialize()

    try:
        request = SearchRequest(
            query="Explain LangGraph", collection="docs", limit=1, enable_rag=True
        )
        response = await orchestrator.search(request)
        assert response.generated_answer is not None

        samples = {metric.name: metric for metric in registry.registry.collect()}
        assert "service_rag_stage_latency_seconds" in samples
        assert "service_rag_answers" in samples
    finally:
        await orchestrator.cleanup()
        await vector_service.cleanup()
        set_metrics_registry(None)


@pytest.mark.asyncio
async def test_services_handle_empty_retrieval_quickly() -> None:
    """No documents should still produce a fast response and no answer."""

    vector_service = _StubVectorService(matches=[], collection="docs")
    await vector_service.initialize()

    def generator_factory(config: RAGConfig, retriever) -> RAGGenerator:
        _ = config
        return cast(RAGGenerator, _StubGenerator(retriever))

    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, vector_service),
        generator_factory=generator_factory,
    )

    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, vector_service),
        rag_config=ServiceRAGConfig(
            model="test-model",
            retriever_top_k=3,
            include_sources=True,
            compression_enabled=False,
        ),
    )
    orchestrator._rag_pipeline = pipeline  # type: ignore[attr-defined]

    await orchestrator.initialize()

    try:
        request = SearchRequest(
            query="Unknown topic", collection="docs", limit=1, enable_rag=True
        )
        start = time.perf_counter()
        response = await orchestrator.search(request)
        duration = time.perf_counter() - start

        assert response.generated_answer is None
        assert duration < 0.05
    finally:
        await orchestrator.cleanup()
        await vector_service.cleanup()
