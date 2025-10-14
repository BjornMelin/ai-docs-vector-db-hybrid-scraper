"""Performance smoke tests for the SearchOrchestrator."""

from __future__ import annotations

import asyncio
import time
from typing import cast

import pytest

from src.contracts.retrieval import SearchRecord
from src.models.search import SearchRequest
from src.services.query_processing.orchestrator import (
    SearchOrchestrator,
    ServiceRAGConfig,
)
from src.services.rag import LangGraphRAGPipeline
from src.services.vector_db.service import VectorStoreService


class _StubVectorService:
    """Vector service returning deterministic matches for concurrency tests."""

    def __init__(self, matches: list[SearchRecord], collection: str) -> None:
        self._matches = matches
        self._collection = collection

    async def initialize(self) -> None:  # pragma: no cover - symmetry
        return None

    async def cleanup(self) -> None:  # pragma: no cover - symmetry
        return None

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        **_: object,
    ) -> list[SearchRecord]:
        assert collection == self._collection
        assert isinstance(query, str)
        return self._matches[:limit]


class _StubPipeline:
    """Stub RAG pipeline with predictable latency."""

    def __init__(self) -> None:
        self.invocations = 0

    async def run(self, **_: object) -> dict[str, object]:
        self.invocations += 1
        await asyncio.sleep(0)  # Yield control without adding measurable latency.
        return {
            "answer": "stub-answer",
            "confidence": 0.9,
            "sources": [],
        }


@pytest.mark.performance
@pytest.mark.asyncio
async def test_orchestrator_handles_parallel_queries() -> None:
    """The orchestrator should sustain high concurrency with stubbed pipeline."""
    matches = [
        SearchRecord(
            id=f"doc-{i}",
            score=0.9,
            content=f"Document {i}",
            title=f"Title {i}",
            metadata={"content": f"Document {i}"},
        )
        for i in range(5)
    ]
    vector_service = _StubVectorService(matches, collection="docs")
    orchestrator = SearchOrchestrator(
        vector_store_service=cast(VectorStoreService, vector_service),
        rag_config=ServiceRAGConfig(
            model="test-model",
            retriever_top_k=3,
            include_sources=True,
            compression_enabled=False,
        ),
    )

    stub_pipeline = _StubPipeline()
    orchestrator._rag_pipeline = cast(LangGraphRAGPipeline, stub_pipeline)  # type: ignore[attr-defined]

    await orchestrator.initialize()

    try:
        requests = [
            SearchRequest(
                query=f"parallel query {i}",
                collection="docs",
                limit=3,
                enable_rag=True,
            )
            for i in range(200)
        ]

        start = time.perf_counter()
        responses = await asyncio.gather(
            *[orchestrator.search(request) for request in requests]
        )
        duration = time.perf_counter() - start

        assert len(responses) == len(requests)
        assert all(response.generated_answer == "stub-answer" for response in responses)
        assert stub_pipeline.invocations == len(requests)

        throughput = len(requests) / duration if duration > 0 else float("inf")
        assert throughput >= 200.0, (
            f"Observed throughput {throughput:.1f} req/s below target"
        )
    finally:
        await orchestrator.cleanup()
