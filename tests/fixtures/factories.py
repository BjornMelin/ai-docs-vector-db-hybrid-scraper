"""Unified mock factory fixtures for tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


try:
    from redis.exceptions import ConnectionError as RedisConnectionError
except ImportError:  # pragma: no cover - redis optional for tests
    RedisConnectionError = ConnectionError  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for HTTP mocks
    import respx
except ImportError:  # pragma: no cover - optional dependency for HTTP mocks
    respx = None  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import respx as respx_module


class MockFactory:
    """Factory for building reusable test doubles and data structures."""

    def create_openai_client(
        self,
        embedding_dim: int = 1536,
        model: str = "text-embedding-3-small",
        rate_limit: int = 10000,
    ) -> MagicMock:
        """Return a mocked OpenAI client configured for embeddings tests.

        Args:
            embedding_dim: The dimensionality of the generated embedding.
            model: The embedding model identifier.
            rate_limit: The simulated request limit returned in headers.

        Returns:
            MagicMock: Mocked client with embeddings and responses APIs.
        """

        client = MagicMock()

        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1] * embedding_dim)]
        embedding_response.model = model
        embedding_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
        )

        client.embeddings.create = AsyncMock(return_value=embedding_response)
        client.embeddings.create.headers = {
            "x-ratelimit-limit-requests": str(rate_limit),
            "x-ratelimit-remaining-requests": str(rate_limit - 1),
            "x-ratelimit-reset-requests": "1s",
        }

        responses_payload = MagicMock()
        responses_payload.output_text = "Generated hypothetical document"
        responses_payload.usage = MagicMock(
            input_tokens=50,
            output_tokens=20,
            total_tokens=70,
        )

        responses_mock = MagicMock()
        responses_mock.create = AsyncMock(return_value=responses_payload)
        client.responses = responses_mock

        return client

    def create_qdrant_client(
        self,
        vector_size: int = 1536,
        collection_exists: bool = True,
    ) -> MagicMock:
        """Return a mocked Qdrant client with configurable responses.

        Args:
            vector_size: The expected size of vectors stored in Qdrant.
            collection_exists: Whether the collection should be reported as existing.

        Returns:
            MagicMock: Mocked client with collection and search APIs.
        """

        client = MagicMock()
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()
        client.recreate_collection = AsyncMock()

        client.get_collection = AsyncMock(
            return_value=MagicMock(
                status="green",
                vectors_count=100 if collection_exists else 0,
                points_count=100 if collection_exists else 0,
                vector_size=vector_size,
            )
        )
        client.collection_exists = AsyncMock(return_value=collection_exists)

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

    def create_redis_client(self, connected: bool = True) -> MagicMock:
        """Return a mocked Redis client optionally simulating outages.

        Args:
            connected: When ``False`` the client raises connection failures.

        Returns:
            MagicMock: Redis client mock with async command implementations.
        """

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
            client.ping = AsyncMock(
                side_effect=RedisConnectionError("Connection failed"),
            )
            client.get = AsyncMock(
                side_effect=ConnectionError("Connection failed"),
            )

        client.close = AsyncMock()
        client.aclose = AsyncMock()

        return client

    def create_httpx_client(
        self,
        status_code: int = 200,
        content: str | None = None,
    ) -> MagicMock:
        """Return a mocked ``httpx.AsyncClient`` with canned responses.

        Args:
            status_code: HTTP status to be returned for all requests.
            content: Optional text content for the response body.

        Returns:
            MagicMock: Async client mock covering standard HTTP verbs.
        """

        client = MagicMock()

        response = MagicMock()
        response.status_code = status_code
        response.text = content or "<html><body>Test content</body></html>"
        response.json = MagicMock(return_value={"status": "ok"})
        response.headers = {"content-type": "text/html"}
        response.raise_for_status = MagicMock()

        for method in ["get", "post", "put", "delete", "patch"]:
            setattr(client, method, AsyncMock(return_value=response))

        client.aclose = AsyncMock()

        return client

    def create_redis_sentinel(self) -> MagicMock:
        """Return a mocked Redis sentinel cluster configuration.

        Returns:
            MagicMock: Sentinel mock covering discovery and failover paths.
        """

        sentinel = MagicMock()
        master = MagicMock()

        sentinel.discover_master = MagicMock(return_value=("127.0.0.1", 6379))
        sentinel.discover_slaves = MagicMock(return_value=[("127.0.0.1", 6380)])
        sentinel.master_for = MagicMock(return_value=master)
        sentinel.failover = AsyncMock()

        return sentinel

    def create_crawl4ai_service(self) -> MagicMock:
        """Return a mocked Crawl4AI service with async crawling helpers.

        Returns:
            MagicMock: Service mock exposing crawl and batch_crawl coroutines.
        """

        service = MagicMock()

        async def _crawl(url: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "url": url,
                "status_code": 200,
                "html": f"<html><body>Content for {url}</body></html>",
                "markdown": f"# Content for {url}",
                "metadata": {
                    "title": f"Page: {url}",
                    "crawl_time": 0.5,
                    "word_count": 100,
                },
                "screenshot": b"fake_screenshot" if kwargs.get("screenshot") else None,
                "pdf": b"fake_pdf" if kwargs.get("pdf") else None,
            }

        async def _batch_crawl(urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
            return [await _crawl(url, **kwargs) for url in urls]

        service.crawl = _crawl
        service.batch_crawl = AsyncMock(side_effect=_batch_crawl)

        return service

    @contextmanager
    def mock_webhook_endpoint(self) -> Iterator[list[dict[str, Any]]]:
        """Patch ``httpx.AsyncClient.post`` and capture webhook calls.

        Returns:
            Iterator[list[dict[str, Any]]]: Iterator yielding recorded webhook calls.
        """

        webhook_calls: list[dict[str, Any]] = []

        async def _mock(url: str, data: dict[str, Any]) -> dict[str, Any]:
            webhook_calls.append(
                {"url": url, "data": data, "timestamp": "2024-01-01T00:00:00Z"}
            )
            return {"status": "accepted", "id": f"webhook-{len(webhook_calls)}"}

        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=_mock)):
            yield webhook_calls

    def create_elasticsearch_client(self) -> MagicMock:
        """Return a mocked Elasticsearch client covering CRUD and search.

        Returns:
            MagicMock: Elasticsearch client mock supporting basic operations.
        """

        client = MagicMock()
        client.indices.create = AsyncMock()
        client.indices.delete = AsyncMock()
        client.indices.exists = AsyncMock(return_value=True)
        client.index = AsyncMock(return_value={"_id": "test-id"})
        client.bulk = AsyncMock(return_value={"errors": False})
        client.search = AsyncMock(
            return_value={
                "hits": {
                    "total": {"value": 1},
                    "hits": [
                        {
                            "_id": "1",
                            "_score": 0.95,
                            "_source": {
                                "content": "Test document",
                                "embedding": [0.1] * 384,
                            },
                        }
                    ],
                }
            }
        )
        client.knn_search = AsyncMock(
            return_value={
                "hits": {
                    "total": {"value": 1},
                    "hits": [
                        {
                            "_id": "1",
                            "_score": 0.95,
                            "_source": {"content": "Test document"},
                        }
                    ],
                }
            }
        )

        return client

    def create_monitoring_services(self) -> dict[str, MagicMock]:
        """Return mocked monitoring services such as Prometheus and Grafana.

        Returns:
            dict[str, MagicMock]: Mapping of service names to their mocks.
        """

        return {
            "prometheus": MagicMock(
                push_metric=AsyncMock(),
                query=AsyncMock(return_value={"status": "success", "data": []}),
            ),
            "grafana": MagicMock(
                create_dashboard=AsyncMock(return_value={"id": "test-dash"}),
                create_alert=AsyncMock(return_value={"id": "test-alert"}),
            ),
            "opentelemetry": MagicMock(
                trace=MagicMock(), metric=MagicMock(), log=MagicMock()
            ),
        }

    def build_embedding_points(
        self,
        count: int = 5,
        vector_size: int = 1536,
        base_url: str = "https://test.example.com",
    ) -> list[dict[str, Any]]:
        """Construct a list of embedding points with payload metadata.

        Args:
            count: Number of embedding points to create.
            vector_size: Dimensionality of the embedding vectors.
            base_url: Base URL used to generate document payloads.

        Returns:
            list[dict[str, Any]]: Collection of embedding point dictionaries.
        """

        points: list[dict[str, Any]] = []
        for index in range(count):
            points.append(
                {
                    "id": index + 1,
                    "vector": [0.1 * (index + 1)] * vector_size,
                    "payload": {
                        "url": f"{base_url}/doc{index + 1}",
                        "title": f"Test Document {index + 1}",
                        "content": f"Test content for document {index + 1}",
                        "chunk_index": 0,
                        "metadata": {
                            "source": "test",
                            "timestamp": f"2024-01-{index + 1:02d}T00:00:00Z",
                            "word_count": 100 + index * 10,
                        },
                    },
                }
            )

        return points

    def build_crawl_response(
        self,
        url: str = "https://test.example.com",
        success: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Construct a crawl response payload for ingestion tests.

        Args:
            url: URL associated with the crawl response.
            success: Whether the crawl should be treated as successful.
            include_metadata: Whether to include metadata in the response.

        Returns:
            dict[str, Any]: Crawl response payload with optional metadata.
        """

        if not success:
            return {
                "url": url,
                "status_code": 404,
                "error": "Page not found",
                "html": None,
                "markdown": None,
                "text": None,
            }

        response: dict[str, Any] = {
            "url": url,
            "status_code": 200,
            "html": (
                f"<html><body><h1>Test Page</h1><p>Content for {url}</p></body></html>"
            ),
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

    @asynccontextmanager
    async def mock_http(
        self,
        *,
        assert_all_mocked: bool = False,
    ) -> AsyncIterator[respx_module.Router]:
        """Yield a configured respx router for HTTP interaction tests.

        Args:
            assert_all_mocked: Flag passed to ``respx.mock`` to validate routes.

        Returns:
            AsyncIterator[respx.Router]: Asynchronous iterator yielding the router.
        """

        if respx is None:
            pytest.skip("respx is not installed")

        with respx.mock(assert_all_mocked=assert_all_mocked) as router:  # type: ignore[attr-defined]
            router.post("https://api.openai.com/v1/embeddings").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "embedding": [0.1, 0.2, 0.3],
                                "index": 0,
                            }
                        ]
                    },
                )
            )
            router.get("https://test.example.com").mock(
                return_value=httpx.Response(200, text="Test content"),
            )
            yield router


@pytest.fixture
def mock_factory() -> MockFactory:
    """Provide a unified ``MockFactory`` instance for tests.

    Returns:
        MockFactory: Shared factory for building mocks and test payloads.
    """

    return MockFactory()
