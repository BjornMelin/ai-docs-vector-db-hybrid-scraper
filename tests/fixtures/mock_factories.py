"""Mock factories for reusable test objects.

This module provides factory functions for creating consistent mock objects
following the boundary-only mocking principle for external services.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class ExternalServiceMockFactory:
    """Factory for creating external service mocks with realistic behavior."""

    @staticmethod
    def create_openai_client(
        embedding_dim: int = 1536,
        model: str = "text-embedding-3-small",
        rate_limit: int = 10000,
    ) -> MagicMock:
        """Create OpenAI client mock with configurable parameters."""
        client = MagicMock()

        # Embedding response
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1] * embedding_dim)]
        embedding_response.model = model
        embedding_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=0, total_tokens=10
        )

        client.embeddings.create = AsyncMock(return_value=embedding_response)

        # Rate limit headers simulation
        client.embeddings.create.headers = {
            "x-ratelimit-limit-requests": str(rate_limit),
            "x-ratelimit-remaining-requests": str(rate_limit - 1),
            "x-ratelimit-reset-requests": "1s",
        }

        # Chat completion for HyDE
        chat_response = MagicMock()
        chat_response.choices = [
            MagicMock(
                message=MagicMock(content="Generated hypothetical document"),
                finish_reason="stop",
            )
        ]
        chat_response.usage = MagicMock(
            prompt_tokens=50, completion_tokens=20, total_tokens=70
        )

        client.chat.completions.create = AsyncMock(return_value=chat_response)

        return client

    @staticmethod
    def create_qdrant_client(
        vector_size: int = 1536,
        distance: str = "Cosine",
        collection_exists: bool = True,
    ) -> MagicMock:
        """Create Qdrant client mock with configurable vector parameters."""
        from tests.conftest import CollectionInfo, CollectionStatus

        client = MagicMock()

        # Collection operations
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.recreate_collection = AsyncMock()

        collection_info = CollectionInfo(
            status=CollectionStatus(status="green"),
            vectors_count=100 if collection_exists else 0,
            points_count=100 if collection_exists else 0,
        )

        client.get_collection = AsyncMock(return_value=collection_info)
        client.collection_exists = AsyncMock(return_value=collection_exists)

        # Point operations with realistic responses
        client.upsert = AsyncMock(return_value=MagicMock(status="completed"))
        client.search = AsyncMock(
            return_value=[
                MagicMock(
                    id=1,
                    score=0.95,
                    payload={
                        "url": "https://example.com/doc1",
                        "title": "Test Document 1",
                        "content": "Test content 1",
                    },
                )
            ]
        )

        client.scroll = AsyncMock(return_value=([], None))
        client.count = AsyncMock(return_value=MagicMock(count=100))
        client.close = AsyncMock()

        return client

    @staticmethod
    def create_redis_client(connected: bool = True) -> MagicMock:
        """Create Redis client mock with configurable connection state."""
        client = MagicMock()

        if connected:
            client.ping = AsyncMock(return_value=True)
            client.get = AsyncMock(return_value=None)
            client.set = AsyncMock(return_value=True)
            client.delete = AsyncMock(return_value=1)
            client.exists = AsyncMock(return_value=0)
            client.expire = AsyncMock(return_value=True)
            client.ttl = AsyncMock(return_value=-2)
        else:
            # Simulate connection failures
            from redis.exceptions import ConnectionError

            client.ping = AsyncMock(side_effect=ConnectionError("Connection failed"))
            client.get = AsyncMock(side_effect=ConnectionError("Connection failed"))

        client.close = AsyncMock()
        client.aclose = AsyncMock()

        return client

    @staticmethod
    def create_httpx_client(status_code: int = 200, content: str = None) -> MagicMock:
        """Create httpx async client mock with configurable responses."""
        client = MagicMock()

        # Default response
        response = MagicMock()
        response.status_code = status_code
        response.text = content or "<html><body>Test content</body></html>"
        response.json = MagicMock(return_value={"status": "ok"})
        response.headers = {"content-type": "text/html"}
        response.raise_for_status = MagicMock()

        # Configure all HTTP methods
        for method in ["get", "post", "put", "delete", "patch"]:
            setattr(client, method, AsyncMock(return_value=response))

        client.aclose = AsyncMock()

        return client


class DataMockFactory:
    """Factory for creating test data objects."""

    @staticmethod
    def create_embedding_points(
        count: int = 5,
        vector_size: int = 1536,
        base_url: str = "https://test.example.com",
    ) -> list[dict[str, Any]]:
        """Create realistic embedding point data."""
        points = []

        for i in range(count):
            point = {
                "id": i + 1,
                "vector": [0.1 * (i + 1)] * vector_size,
                "payload": {
                    "url": f"{base_url}/doc{i + 1}",
                    "title": f"Test Document {i + 1}",
                    "content": f"Test content for document {i + 1}",
                    "chunk_index": 0,
                    "metadata": {
                        "source": "test",
                        "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                        "word_count": 100 + i * 10,
                    },
                },
            }
            points.append(point)

        return points

    @staticmethod
    def create_crawl_response(
        url: str = "https://test.example.com",
        success: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Create realistic crawl response data."""
        if not success:
            return {
                "url": url,
                "status_code": 404,
                "error": "Page not found",
                "html": None,
                "markdown": None,
                "text": None,
            }

        response = {
            "url": url,
            "status_code": 200,
            "html": f"<html><body><h1>Test Page</h1><p>Content for {url}</p></body></html>",
            "cleaned_html": f"<h1>Test Page</h1><p>Content for {url}</p>",
            "markdown": f"# Test Page\n\nContent for {url}",
            "text": f"Test Page\nContent for {url}",
            "links": [f"{url}/link1", f"{url}/link2"],
            "images": [],
            "error": None,
        }

        if include_metadata:
            response["metadata"] = {
                "title": f"Test Page - {url}",
                "description": f"Test description for {url}",
                "keywords": ["test", "example"],
                "language": "en",
                "crawl_time": 0.5,
                "word_count": 20,
            }

        return response


@pytest.fixture
def external_service_factory():
    """Provide external service mock factory."""
    return ExternalServiceMockFactory()


@pytest.fixture
def data_factory():
    """Provide data mock factory."""
    return DataMockFactory()


@pytest.fixture
def mock_openai_factory(external_service_factory):
    """Factory fixture for OpenAI client mocks."""

    def _create(**kwargs):
        return external_service_factory.create_openai_client(**kwargs)

    return _create


@pytest.fixture
def mock_qdrant_factory(external_service_factory):
    """Factory fixture for Qdrant client mocks."""

    def _create(**kwargs):
        return external_service_factory.create_qdrant_client(**kwargs)

    return _create


@pytest.fixture
def mock_redis_factory(external_service_factory):
    """Factory fixture for Redis client mocks."""

    def _create(**kwargs):
        return external_service_factory.create_redis_client(**kwargs)

    return _create
