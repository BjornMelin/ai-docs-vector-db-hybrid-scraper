"""Tests for the final query processing pipeline wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services.query_processing.models import SearchRequest
from src.services.query_processing.orchestrator import SearchOrchestrator
from src.services.query_processing.pipeline import QueryProcessingPipeline
from src.services.vector_db.adapter_base import VectorMatch


class VectorServiceStub:
    def __init__(self) -> None:
        self.config = type(
            "Cfg", (), {"qdrant": type("Q", (), {"collection_name": "docs"})}
        )()

    async def initialize(self) -> None:  # pragma: no cover - trivial stub
        return

    async def cleanup(self) -> None:  # pragma: no cover - trivial stub
        return

    async def search_documents(self, collection: str, query: str, **kwargs):
        assert collection == "docs"
        assert query
        match = VectorMatch(
            id="id-1",
            score=0.42,
            raw_score=0.42,
            payload={
                "content": "hello",
                "title": "example",
                "_grouping": {"applied": True, "group_id": "id-1", "rank": 1},
            },
            collection="docs",
        )
        return [match]


@pytest.mark.asyncio
async def test_pipeline_process_string_query() -> None:
    orchestrator = SearchOrchestrator(vector_store_service=VectorServiceStub())  # type: ignore
    pipeline = QueryProcessingPipeline(orchestrator)
    await pipeline.initialize()

    response = await pipeline.process("widgets", collection="docs")

    assert len(response.records) == 1
    assert response.records[0].collection == "docs"


@pytest.mark.asyncio
async def test_pipeline_process_request_instance(monkeypatch) -> None:
    orchestrator = SearchOrchestrator(vector_store_service=VectorServiceStub())  # type: ignore
    pipeline = QueryProcessingPipeline(orchestrator)
    await pipeline.initialize()

    spy = AsyncMock(side_effect=orchestrator.search)
    monkeypatch.setattr(orchestrator, "search", spy)

    request = SearchRequest(query="test", collection="docs", limit=10)
    await pipeline.process(request)

    spy.assert_awaited_once()
