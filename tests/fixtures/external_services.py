"""Fixtures for mocking external services.

This module provides fixtures for mocking external dependencies like
APIs, databases, and third-party services to ensure test isolation.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses with rate limiting simulation."""
    with patch("openai.AsyncOpenAI") as mock_class:
        client = MagicMock()
        mock_class.return_value = client
        
        # Embedding responses with rate limit headers
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        embedding_response.model = "text-embedding-3-small"
        embedding_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
        
        # Add rate limit simulation
        client.embeddings.create = AsyncMock(return_value=embedding_response)
        client.embeddings.create.headers = {
            "x-ratelimit-limit-requests": "10000",
            "x-ratelimit-remaining-requests": "9999",
            "x-ratelimit-reset-requests": "1s"
        }
        
        yield client


@pytest.fixture
def mock_qdrant_cloud():
    """Mock Qdrant Cloud API with auth and cluster management."""
    with patch("qdrant_client.QdrantClient") as mock_class:
        client = MagicMock()
        mock_class.return_value = client
        
        # Cloud-specific features
        client.get_cluster_info = AsyncMock(return_value={
            "cluster_id": "test-cluster",
            "status": "running",
            "endpoints": {
                "rest": "https://test-cluster.qdrant.io:6333",
                "grpc": "https://test-cluster.qdrant.io:6334"
            }
        })
        
        # Auth
        client.api_key = "test-api-key"
        client.verify_api_key = AsyncMock(return_value=True)
        
        yield client


@pytest.fixture
def mock_redis_sentinel():
    """Mock Redis with Sentinel for HA testing."""
    sentinel = MagicMock()
    master = MagicMock()
    
    # Sentinel discovery
    sentinel.discover_master = MagicMock(return_value=("127.0.0.1", 6379))
    sentinel.discover_slaves = MagicMock(return_value=[("127.0.0.1", 6380)])
    
    # Master operations
    sentinel.master_for = MagicMock(return_value=master)
    
    # Failover simulation
    sentinel.failover = AsyncMock()
    
    return sentinel


@pytest.fixture
def mock_crawl4ai_service():
    """Mock Crawl4AI service with advanced features."""
    service = MagicMock()
    
    async def mock_crawl(url, **kwargs):
        return {
            "url": url,
            "status_code": 200,
            "html": f"<html><body>Content for {url}</body></html>",
            "markdown": f"# Content for {url}",
            "metadata": {
                "title": f"Page: {url}",
                "crawl_time": 0.5,
                "word_count": 100
            },
            "screenshot": b"fake_screenshot" if kwargs.get("screenshot") else None,
            "pdf": b"fake_pdf" if kwargs.get("pdf") else None
        }
    
    service.crawl = mock_crawl
    service.batch_crawl = AsyncMock(side_effect=lambda urls, **kw: [
        mock_crawl(url, **kw) for url in urls
    ])
    
    return service


@pytest.fixture
def mock_webhook_endpoint():
    """Mock webhook endpoint for testing callbacks."""
    webhook_calls = []
    
    async def mock_webhook(url: str, data: dict[str, Any]):
        webhook_calls.append({
            "url": url,
            "data": data,
            "timestamp": "2024-01-01T00:00:00Z"
        })
        return {"status": "accepted", "id": f"webhook-{len(webhook_calls)}"}
    
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=mock_webhook)):
        yield webhook_calls


@pytest_asyncio.fixture
async def mock_elasticsearch():
    """Mock Elasticsearch client for hybrid search testing."""
    from unittest.mock import MagicMock
    
    client = MagicMock()
    
    # Index operations
    client.indices.create = AsyncMock()
    client.indices.delete = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    
    # Document operations
    client.index = AsyncMock(return_value={"_id": "test-id"})
    client.bulk = AsyncMock(return_value={"errors": False})
    
    # Search operations
    client.search = AsyncMock(return_value={
        "hits": {
            "total": {"value": 1},
            "hits": [{
                "_id": "1",
                "_score": 0.95,
                "_source": {
                    "content": "Test document",
                    "embedding": [0.1] * 384
                }
            }]
        }
    })
    
    # KNN search
    client.knn_search = AsyncMock(return_value={
        "hits": {
            "total": {"value": 1},
            "hits": [{
                "_id": "1",
                "_score": 0.95,
                "_source": {"content": "Test document"}
            }]
        }
    })
    
    return client


@pytest.fixture
def mock_monitoring_services():
    """Mock monitoring services (Prometheus, Grafana, etc)."""
    return {
        "prometheus": MagicMock(
            push_metric=AsyncMock(),
            query=AsyncMock(return_value={"status": "success", "data": []})
        ),
        "grafana": MagicMock(
            create_dashboard=AsyncMock(return_value={"id": "test-dash"}),
            create_alert=AsyncMock(return_value={"id": "test-alert"})
        ),
        "opentelemetry": MagicMock(
            trace=MagicMock(),
            metric=MagicMock(),
            log=MagicMock()
        )
    }