"""Modern AI testing utilities for 2025 standards.

This module provides standardized testing utilities for AI/ML components,
following modern testing patterns and best practices.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest


class AITestHelper:
    """Helper class for AI/ML testing patterns."""

    @staticmethod
    def create_mock_embedding_response(
        vectors: list[list[float]] | None = None,
        model: str = "text-embedding-3-small",
        usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized mock embedding response."""
        if vectors is None:
            vectors = [[0.1, 0.2, 0.3] * 512]  # 1536 dimensions

        return {
            "data": [
                {"embedding": vector, "index": i} for i, vector in enumerate(vectors)
            ],
            "model": model,
            "object": "list",
            "usage": usage or {"prompt_tokens": 10, "total_tokens": 10},
        }

    @staticmethod
    def create_mock_search_response(
        results: list[dict[str, Any]] | None = None,
        total: int | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Create a standardized mock search response."""
        if results is None:
            results = []

        return {
            "results": results,
            "_total": total or len(results),
            "query": query or "test query",
            "took": 0.1,
            "max_score": results[0].get("score", 1.0) if results else 0.0,
        }

    @staticmethod
    def create_mock_rag_response(
        answer: str = "This is a test answer",
        sources: list[dict[str, Any]] | None = None,
        confidence: float = 0.9,
    ) -> dict[str, Any]:
        """Create a standardized mock RAG response."""
        if sources is None:
            sources = [
                {
                    "title": "Test Document",
                    "url": "https://example.com/doc",
                    "content": "Test content",
                    "score": 0.8,
                }
            ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }


class AsyncTestHelper:
    """Helper class for async testing patterns."""

    @staticmethod
    async def wait_for_condition(
        condition_func: callable,
        timeout_seconds: float = 5.0,
        interval: float = 0.1,
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        return False

    @staticmethod
    def create_async_mock(
        return_value: Any = None, side_effect: Any = None
    ) -> AsyncMock:
        """Create a properly configured AsyncMock."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock


class PerformanceTestHelper:
    """Helper class for performance testing patterns."""

    @staticmethod
    def measure_execution_time(func: callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        return result, execution_time

    @staticmethod
    async def measure_async_execution_time(
        func: callable, *args, **kwargs
    ) -> tuple[Any, float]:
        """Measure execution time of an async function."""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        return result, execution_time

    @staticmethod
    def assert_performance_threshold(
        execution_time: float,
        max_time: float,
        operation_name: str = "operation",
    ) -> None:
        """Assert that execution time is within acceptable limits."""
        assert execution_time <= max_time, (
            f"{operation_name} took {execution_time:.3f}s, "
            f"exceeding limit of {max_time:.3f}s"
        )


# Convenience functions for common testing patterns
def create_mock_openai_client() -> AsyncMock:
    """Create a mock OpenAI client with standard responses."""
    client = AsyncMock()
    client.embeddings.create.return_value = (
        AITestHelper.create_mock_embedding_response()
    )
    client.chat.completions.create.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 100},
    }
    return client


def create_mock_qdrant_client() -> AsyncMock:
    """Create a mock Qdrant client with standard responses."""
    client = AsyncMock()
    client.search.return_value = AITestHelper.create_mock_search_response()
    client.upsert.return_value = {"status": "completed", "operation_id": "test-op"}
    return client


def create_mock_firecrawl_client() -> AsyncMock:
    """Create a mock Firecrawl client with standard responses."""
    client = AsyncMock()
    client.scrape_url.return_value = {
        "success": True,
        "data": {"content": "Scraped content", "title": "Test Page"},
    }
    return client


# Test fixtures for common patterns
@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Fixture for mock OpenAI client."""
    return create_mock_openai_client()


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Fixture for mock Qdrant client."""
    return create_mock_qdrant_client()


@pytest.fixture
def mock_firecrawl_client() -> AsyncMock:
    """Fixture for mock Firecrawl client."""
    return create_mock_firecrawl_client()


@pytest.fixture
def performance_helper() -> PerformanceTestHelper:
    """Fixture for performance testing helper."""
    return PerformanceTestHelper()


@pytest.fixture
def ai_test_helper() -> AITestHelper:
    """Fixture for AI testing helper."""
    return AITestHelper()
