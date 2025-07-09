"""Comprehensive integration tests for API endpoints.

This module demonstrates:
- FastAPI TestClient usage
- respx for external service mocking
- End-to-end workflow testing
- Performance validation
"""

import asyncio
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st

from src.api.app_factory import create_app
from src.architecture.modes import ApplicationMode
from tests.fixtures.test_infrastructure import (
    PerformanceTestUtils,
    TestDataFactory,
)


@pytest.fixture
def test_app():
    """Create test FastAPI application."""
    return create_app(mode=ApplicationMode.SIMPLE)


@pytest.fixture
def test_client(test_app):
    """Create test client for API testing."""
    with TestClient(test_app) as client:
        yield client


class TestAPIEndpointsIntegration:
    """Integration tests for API endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns correct information."""
        # Act
        response = test_client.get("/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "running"
        assert data["mode"] == "simple"

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        # Act
        response = test_client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "available_services" in data
        assert "timestamp" in data

    @pytest.mark.respx
    def test_document_indexing_workflow(self, test_client, respx_mock):
        """Test complete document indexing workflow."""
        # Arrange
        document = TestDataFactory.create_document(
            doc_id="test-doc-1",
            content="This is a test document for indexing workflow validation.",
            metadata={"source": "test", "category": "integration"},
        )

        # Mock OpenAI embeddings
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        # Mock Qdrant operations
        respx_mock.route(host="localhost", port=6333).mock(
            return_value=httpx.Response(200, json={"result": {"status": "ok"}})
        )

        # Act - Index document
        response = test_client.post("/api/v1/documents", json=document)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == document["id"]
        assert data["status"] == "indexed"

    @pytest.mark.respx
    def test_search_workflow(self, test_client, respx_mock):
        """Test search workflow with mocked services."""
        # Arrange
        query = "test query for search"
        expected_results = [
            TestDataFactory.create_search_result(
                doc_id="doc-1",
                score=0.95,
                content="Highly relevant content",
            ),
            TestDataFactory.create_search_result(
                doc_id="doc-2",
                score=0.85,
                content="Somewhat relevant content",
            ),
        ]

        # Mock embeddings
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.2] * 1536}],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                },
            )
        )

        # Mock vector search
        respx_mock.post(url__regex=r".*search.*").mock(
            return_value=httpx.Response(
                200,
                json={"result": expected_results},
            )
        )

        # Act
        response = test_client.post(
            "/api/v1/search",
            json={"query": query, "limit": 10},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert (
            data["results"][0]["score"]
            > data["results"][1]["score"] @ pytest.mark.respx
        )

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_batch_processing(self, test_client, respx_mock, batch_size):
        """Test batch document processing."""
        # Arrange
        documents = [
            TestDataFactory.create_document(
                doc_id=f"batch-doc-{i}",
                content=f"Batch document {i} content",
            )
            for i in range(batch_size)
        ]

        # Mock embeddings for batch
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536} for _ in range(batch_size)],
                    "usage": {
                        "prompt_tokens": batch_size * 10,
                        "total_tokens": batch_size * 10,
                    },
                },
            )
        )

        # Mock Qdrant batch operations
        respx_mock.route(host="localhost", port=6333).mock(
            return_value=httpx.Response(200, json={"result": {"status": "ok"}})
        )

        # Act
        response = test_client.post(
            "/api/v1/documents/batch", json={"documents": documents}
        )

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["processed"] == batch_size
        assert data["status"] == "success"

    @pytest.mark.respx
    def test_error_handling(self, test_client, respx_mock):
        """Test API error handling for various failure scenarios."""
        # Test 1: External service failure
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(500, json={"error": "Internal server error"})
        )

        response = test_client.post(
            "/api/v1/documents",
            json=TestDataFactory.create_document(doc_id="fail-doc"),
        )
        assert response.status_code == 503
        assert "error" in response.json()

        # Test 2: Invalid request data
        response = test_client.post(
            "/api/v1/documents",
            json={"invalid": "data"},
        )
        assert response.status_code == 422

        # Test 3: Rate limiting
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(429, json={"error": "Rate limit exceeded"})
        )

        response = test_client.post(
            "/api/v1/search",
            json={"query": "test"},
        )
        assert response.status_code == 429

    @pytest.mark.respx
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_app, respx_mock):
        """Test handling of concurrent API requests."""
        # Arrange
        from httpx import AsyncClient

        # Mock services
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        # Create multiple concurrent requests
        async def make_request(client, index):
            return await client.post(
                "/api/v1/search",
                json={"query": f"concurrent query {index}"},
            )

        # Act
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            tasks = [make_request(client, i) for i in range(10)]
            responses = await asyncio.gather(*tasks)

        # Assert
        assert all(r.status_code == 200 for r in responses)
        assert len(responses) == 10

    @pytest.mark.respx
    def test_performance_requirements(self, test_client, respx_mock):
        """Test API performance meets requirements."""
        # Arrange
        perf_utils = PerformanceTestUtils()

        # Mock fast responses
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                },
            )
        )

        # Act & Assert - Test response times
        with perf_utils.measure_time() as timer:
            response = test_client.post(
                "/api/v1/search",
                json={"query": "performance test"},
            )

        assert response.status_code == 200
        assert timer.elapsed_ms < 100  # Should respond within 100ms

    @pytest.mark.respx
    @given(
        query=st.text(min_size=1, max_size=100),
        limit=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=10, deadline=None)
    def test_search_with_property_testing(self, test_client, respx_mock, query, limit):
        """Property-based testing for search endpoint."""
        # Mock services
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        respx_mock.post(url__regex=r".*search.*").mock(
            return_value=httpx.Response(
                200,
                json={"result": []},
            )
        )

        # Act
        response = test_client.post(
            "/api/v1/search",
            json={"query": query, "limit": limit},
        )

        # Assert properties
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= limit
