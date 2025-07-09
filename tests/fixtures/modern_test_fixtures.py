"""Modern test fixtures for comprehensive testing (7/2025 best practices).

This module provides reusable test fixtures following modern testing patterns:
- Async-first design with proper cleanup
- Boundary mocking only (external services)
- Type-safe fixtures with proper annotations
- Property-based testing support with hypothesis
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
import respx
from httpx import AsyncClient
from hypothesis import strategies as st

from src.models.api_contracts import Document, SearchRequest
from src.models.vector_search import SearchResult


# Property-based testing strategies
@st.composite
def document_strategy(draw: st.DrawFn) -> Document:
    """Generate valid Document instances for property testing."""
    return Document(
        content=draw(st.text(min_size=1, max_size=1000)),
        metadata=draw(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=50),
                values=st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
                max_size=10,
            )
        ),
        source=draw(st.text(min_size=1, max_size=200)),
    )


@st.composite
def search_request_strategy(draw: st.DrawFn) -> SearchRequest:
    """Generate valid SearchRequest instances for property testing."""
    return SearchRequest(
        query=draw(st.text(min_size=1, max_size=200)),
        top_k=draw(st.integers(min_value=1, max_value=100)),
        filters=draw(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=50),
                values=st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
                max_size=5,
            )
        ),
    )


# Async HTTP client fixtures
@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient]:
    """Provide async HTTP client with proper cleanup."""
    async with AsyncClient() as client:
        yield client


@pytest_asyncio.fixture
async def mock_http_client() -> AsyncGenerator[respx.MockRouter]:
    """Mock HTTP client using respx for boundary testing."""
    async with respx.mock:
        yield respx


# Database fixtures
@pytest_asyncio.fixture
async def mock_qdrant_client() -> AsyncMock:
    """Mock Qdrant client for vector database testing."""
    client = AsyncMock()
    client.search = AsyncMock(
        return_value=[
            SearchResult(
                id="test-1",
                score=0.95,
                payload={"content": "Test content", "source": "test.txt"},
            )
        ]
    )
    client.upsert = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    return client


@pytest_asyncio.fixture
async def mock_redis_client() -> AsyncMock:
    """Mock Redis client for caching tests."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    client.expire = AsyncMock(return_value=True)
    return client


# Service fixtures
@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock embedding service for testing."""
    service = MagicMock()
    service.embed = MagicMock(return_value=[0.1] * 768)
    service.embed_batch = MagicMock(return_value=[[0.1] * 768] * 10)
    return service


@pytest_asyncio.fixture
async def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client for AI service testing."""
    client = AsyncMock()
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
    )
    return client


# Test data factories
@pytest.fixture
def document_factory() -> type[Document]:
    """Factory for creating test documents."""

    class DocumentFactory:
        @staticmethod
        def create(**kwargs: Any) -> Document:
            defaults = {
                "content": "Test document content",
                "metadata": {"test": True},
                "source": "test_source.txt",
            }
            defaults.update(kwargs)
            return Document(**defaults)

        @staticmethod
        def create_batch(count: int = 10) -> list[Document]:
            return [
                DocumentFactory.create(
                    content=f"Test document {i}", metadata={"index": i}
                )
                for i in range(count)
            ]

    return DocumentFactory


# Performance testing fixtures
@pytest.fixture
def benchmark_data() -> dict[str, Any]:
    """Provide data for performance benchmarks."""
    return {
        "documents": [
            {"content": f"Document {i}" * 100, "metadata": {"id": i}}
            for i in range(1000)
        ],
        "queries": [f"Query {i}" for i in range(100)],
        "vectors": [[0.1] * 768 for _ in range(1000)],
    }
