"""Comprehensive test infrastructure setup for >90% coverage.

This module provides the core testing infrastructure implementing:
- respx for HTTP mocking (no manual mocks)
- pytest-asyncio for async test patterns
- hypothesis for property-based testing
- Modern testing best practices from CLAUDE.md
"""

import asyncio
import functools
import logging
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
import respx
from hypothesis import given, settings, strategies as st

from src.infrastructure.clients.http_client import HTTPClientProvider
from src.services.embeddings.manager import EmbeddingManager


logger = logging.getLogger(__name__)

T = TypeVar("T")
P = TypeVar("P")


# ==============================================================================
# Core Testing Infrastructure
# ==============================================================================


class TestInfrastructure:
    """Central test infrastructure management."""

    def __init__(self):
        """Initialize test infrastructure."""
        self.mocked_services: dict[str, Any] = {}
        self.cleanup_tasks: list[Callable] = []

    @contextmanager
    def boundary_mock(self, service_name: str, mock_instance: Any):
        """Context manager for boundary-level mocking.

        Args:
            service_name: Name of the service to mock
            mock_instance: Mock instance to use

        Yields:
            Mock instance
        """
        self.mocked_services[service_name] = mock_instance
        try:
            yield mock_instance
        finally:
            self.mocked_services.pop(service_name, None)

    async def cleanup(self):
        """Run all cleanup tasks."""
        for task in self.cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
            except Exception as e:
                logger.warning(f"Cleanup task failed: {e}")
        self.cleanup_tasks.clear()


# ==============================================================================
# HTTP Mocking with respx
# ==============================================================================


@pytest.fixture
def respx_mock():
    """Fixture for respx HTTP mocking."""
    with respx.mock() as mock:
        # Add default routes for common services
        mock.route(host="api.openai.com").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        mock.route(host="localhost", port=6333).mock(
            return_value=httpx.Response(200, json={"result": {"status": "ok"}})
        )

        yield mock


@pytest_asyncio.fixture
async def http_client(respx_mock):
    """Fixture for mocked HTTP client."""
    async with httpx.AsyncClient() as client:
        provider = HTTPClientProvider(client)
        yield provider


# ==============================================================================
# Async Test Patterns
# ==============================================================================


@pytest_asyncio.fixture
async def async_test_context():
    """Provide async test context with proper cleanup."""
    infrastructure = TestInfrastructure()

    # Setup
    async with asyncio.TaskGroup() as tg:
        # Any async setup tasks can be added here
        pass

    yield infrastructure

    # Cleanup
    await infrastructure.cleanup()


def async_test(func: Callable) -> Callable:
    """Decorator for async tests with proper error handling."""

    @functools.wraps(func)
    @pytest.mark.asyncio
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Async test failed: {func.__name__}", exc_info=True)
            raise

    return wrapper


# ==============================================================================
# Property-Based Testing with Hypothesis
# ==============================================================================


class PropertyTestStrategies:
    """Hypothesis strategies for property-based testing."""

    @staticmethod
    def valid_embedding_dimensions():
        """Strategy for valid embedding dimensions."""
        return st.sampled_from([384, 768, 1024, 1536, 3072])

    @staticmethod
    def document_content():
        """Strategy for document content."""
        return st.text(min_size=10, max_size=5000)

    @staticmethod
    def metadata_dict():
        """Strategy for metadata dictionaries."""
        return st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        )

    @staticmethod
    def vector_embedding(dimension: int = 1536):
        """Strategy for vector embeddings."""
        return st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
            min_size=dimension,
            max_size=dimension,
        )


# ==============================================================================
# Service Mocking Fixtures
# ==============================================================================


@pytest_asyncio.fixture
async def mock_embedding_service(respx_mock):
    """Mock embedding service with respx."""
    service = EmbeddingManager(
        openai_api_key="test-key",
        model_name="text-embedding-3-small",
        dimension=1536,
    )

    # Mock OpenAI embeddings endpoint
    respx_mock.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )

    await service.initialize()
    yield service
    await service.cleanup()


@pytest_asyncio.fixture
async def mock_vector_db_service():
    """Mock vector database service."""
    with patch("qdrant_client.QdrantClient") as mock_client:
        # Configure mock behaviors
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance

        # Mock common operations
        mock_instance.get_collections.return_value = []
        mock_instance.create_collection.return_value = True
        mock_instance.search.return_value = []

        service = VectorDBService(
            host="localhost",
            port=6333,
            api_key=None,
            collection_name="test_collection",
        )

        yield service


@pytest_asyncio.fixture
async def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock(spec=RedisClient)
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = True
    client.exists.return_value = False
    yield client


# ==============================================================================
# Test Data Factories
# ==============================================================================


class TestDataFactory:
    """Factory for generating test data."""

    @staticmethod
    def create_document(
        doc_id: str = "test-doc-1",
        content: str = "Test document content",
        metadata: dict | None = None,
    ) -> dict:
        """Create a test document."""
        return {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {"source": "test"},
            "embedding": [0.1] * 1536,
        }

    @staticmethod
    def create_search_result(
        doc_id: str = "test-doc-1",
        score: float = 0.95,
        content: str = "Test result content",
    ) -> dict:
        """Create a test search result."""
        return {
            "id": doc_id,
            "score": score,
            "content": content,
            "metadata": {"source": "test"},
        }

    @staticmethod
    def create_api_response(
        status_code: int = 200,
        data: Any = None,
        error: str | None = None,
    ) -> dict:
        """Create a test API response."""
        response = {"status": "success" if status_code < 400 else "error"}
        if data is not None:
            response["data"] = data
        if error:
            response["error"] = error
        return response


# ==============================================================================
# Performance Testing Utilities
# ==============================================================================


class PerformanceTestUtils:
    """Utilities for performance testing."""

    @staticmethod
    @asynccontextmanager
    async def measure_async_time(operation_name: str):
        """Measure async operation time."""
        start_time = asyncio.get_event_loop().time()
        yield
        end_time = asyncio.get_event_loop().time()
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"{operation_name} took {duration_ms:.2f}ms")

    @staticmethod
    def assert_performance(
        duration_ms: float, target_ms: float, tolerance: float = 0.1
    ):
        """Assert performance meets target."""
        max_allowed = target_ms * (1 + tolerance)
        assert duration_ms <= max_allowed, (
            f"Performance {duration_ms:.2f}ms exceeds target "
            f"{target_ms}ms (tolerance: {tolerance * 100}%)"
        )


# ==============================================================================
# Integration Test Helpers
# ==============================================================================


@pytest.fixture
def integration_test_env(tmp_path):
    """Setup integration test environment."""
    # Create temporary directories
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    logs_dir = tmp_path / "logs"

    data_dir.mkdir()
    cache_dir.mkdir()
    logs_dir.mkdir()

    # Create test configuration
    return {
        "data_directory": str(data_dir),
        "cache_directory": str(cache_dir),
        "log_directory": str(logs_dir),
        "debug": True,
    }


# ==============================================================================
# Cleanup and Teardown
# ==============================================================================


@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield

    # Cleanup any remaining async tasks
    tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
    for task in tasks:
        task.cancel()

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Force garbage collection
    import gc

    gc.collect()


# ==============================================================================
# Export commonly used fixtures and utilities
# ==============================================================================

__all__ = [
    "PerformanceTestUtils",
    "PropertyTestStrategies",
    "TestDataFactory",
    "TestInfrastructure",
    "async_test",
    "async_test_context",
    "cleanup_after_test",
    "http_client",
    "integration_test_env",
    "mock_embedding_service",
    "mock_redis_client",
    "mock_vector_db_service",
    "respx_mock",
]
