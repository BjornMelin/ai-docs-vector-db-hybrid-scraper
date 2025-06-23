"""Mock factories for creating test doubles and stubs.

This module provides factory functions and classes for creating consistent,
configurable mock objects that simulate the behavior of real system components.
"""

import asyncio
import random
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

from .data_generators import TestDataGenerator


class MockFactory:
    """Factory for creating various types of mock objects."""

    def __init__(self, seed: int | None = None):
        """Initialize the mock factory.

        Args:
            seed: Random seed for consistent mock behavior
        """
        self.data_generator = TestDataGenerator(seed)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def create_mock_response(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        delay: float = 0.0,
    ) -> Mock:
        """Create a mock HTTP response object.

        Args:
            status_code: HTTP status code
            json_data: JSON response data
            headers: Response headers
            delay: Simulated response delay in seconds

        Returns:
            Mock response object
        """
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.headers = headers or {}

        if json_data is not None:
            mock_response.json.return_value = json_data

        # Simulate network delay
        if delay > 0:
            time.sleep(delay)

        return mock_response

    def create_async_mock_with_delay(
        self,
        return_value: Any = None,
        delay_range: tuple = (0.01, 0.1),
        failure_rate: float = 0.0,
        exception_type: type = Exception,
    ) -> AsyncMock:
        """Create an async mock with simulated delay and optional failures.

        Args:
            return_value: Value to return from mock
            delay_range: Min and max delay in seconds
            failure_rate: Probability of raising an exception (0.0 to 1.0)
            exception_type: Type of exception to raise on failure

        Returns:
            Configured AsyncMock
        """

        async def mock_coroutine(*args, **kwargs):
            # Simulate processing delay
            delay = random.uniform(*delay_range)
            await asyncio.sleep(delay)

            # Simulate random failures
            if random.random() < failure_rate:
                raise exception_type("Simulated failure")

            return return_value

        return AsyncMock(side_effect=mock_coroutine)

    def create_stateful_mock(
        self,
        initial_state: dict[str, Any],
        state_transitions: dict[str, Callable] | None = None,
    ) -> Mock:
        """Create a mock that maintains state across calls.

        Args:
            initial_state: Initial state dictionary
            state_transitions: Functions that modify state based on method calls

        Returns:
            Stateful mock object
        """
        state = initial_state.copy()
        transitions = state_transitions or {}

        mock = Mock()
        mock._state = state

        def create_stateful_method(method_name):
            def method(*args, **kwargs):
                if method_name in transitions:
                    transitions[method_name](state, *args, **kwargs)
                return state.copy()

            return method

        # Add common methods
        for method_name in ["get", "set", "update", "delete"]:
            setattr(mock, method_name, create_stateful_method(method_name))

        return mock


def create_mock_vector_db(
    collection_name: str = "test_collection",
    dimension: int = 384,
    documents_count: int = 100,
) -> Mock:
    """Create a mock vector database.

    Args:
        collection_name: Name of the test collection
        dimension: Vector dimension
        documents_count: Number of mock documents in collection

    Returns:
        Mock vector database
    """
    generator = TestDataGenerator()

    # Generate mock documents
    mock_documents = {}
    for _i in range(documents_count):
        doc = generator.generate_document(
            include_embedding=True, embedding_dimension=dimension
        )
        mock_documents[doc["id"]] = doc

    mock_db = Mock()
    mock_db.collection_name = collection_name
    mock_db.dimension = dimension
    mock_db._documents = mock_documents

    def mock_search(query_vector: list[float], limit: int = 10, **kwargs):
        """Mock search implementation."""
        # Simple similarity based on random scoring
        results = []
        doc_ids = list(mock_documents.keys())
        random.shuffle(doc_ids)

        for doc_id in doc_ids[:limit]:
            doc = mock_documents[doc_id]
            score = random.uniform(0.6, 0.95)
            results.append(
                {
                    "id": doc_id,
                    "score": score,
                    "document": doc,
                    "metadata": doc["metadata"],
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def mock_add_documents(documents: list[dict[str, Any]]):
        """Mock add documents implementation."""
        added_ids = []
        for doc in documents:
            doc_id = doc.get("id", str(uuid.uuid4()))
            mock_documents[doc_id] = doc
            added_ids.append(doc_id)
        return {"added_ids": added_ids, "count": len(added_ids)}

    def mock_get_document(doc_id: str):
        """Mock get document implementation."""
        return mock_documents.get(doc_id)

    def mock_delete_document(doc_id: str):
        """Mock delete document implementation."""
        if doc_id in mock_documents:
            del mock_documents[doc_id]
            return True
        return False

    def mock_count():
        """Mock count implementation."""
        return len(mock_documents)

    # Attach methods to mock
    mock_db.search = mock_search
    mock_db.add_documents = mock_add_documents
    mock_db.get_document = mock_get_document
    mock_db.delete_document = mock_delete_document
    mock_db.count = mock_count

    return mock_db


def create_mock_embedding_service(
    model_name: str = "test-model", dimension: int = 384, processing_delay: float = 0.1
) -> AsyncMock:
    """Create a mock embedding service.

    Args:
        model_name: Name of the embedding model
        dimension: Vector dimension
        processing_delay: Simulated processing delay

    Returns:
        Mock embedding service
    """
    generator = TestDataGenerator()

    async def mock_embed_text(text: str) -> list[float]:
        """Mock text embedding."""
        await asyncio.sleep(processing_delay)
        return generator._generate_normalized_vector(dimension)

    async def mock_embed_batch(texts: list[str]) -> list[List[float]]:
        """Mock batch text embedding."""
        await asyncio.sleep(processing_delay * len(texts) * 0.1)  # Batch efficiency
        return [generator._generate_normalized_vector(dimension) for _ in texts]

    def mock_get_model_info():
        """Mock model info."""
        return {
            "model_name": model_name,
            "dimension": dimension,
            "max_tokens": 512,
            "description": f"Mock embedding model with {dimension}d vectors",
        }

    mock_service = AsyncMock()
    mock_service.embed_text = AsyncMock(side_effect=mock_embed_text)
    mock_service.embed_batch = AsyncMock(side_effect=mock_embed_batch)
    mock_service.get_model_info = mock_get_model_info
    mock_service.model_name = model_name
    mock_service.dimension = dimension

    return mock_service


def create_mock_web_scraper(
    success_rate: float = 0.9,
    processing_delay_range: tuple = (0.5, 2.0),
    content_length_range: tuple = (500, 5000),
) -> AsyncMock:
    """Create a mock web scraper.

    Args:
        success_rate: Probability of successful scraping (0.0 to 1.0)
        processing_delay_range: Min and max processing delay in seconds
        content_length_range: Min and max content length for generated content

    Returns:
        Mock web scraper
    """
    generator = TestDataGenerator()

    async def mock_scrape_url(url: str, **kwargs) -> dict[str, Any]:
        """Mock URL scraping."""
        # Simulate processing delay
        delay = random.uniform(*processing_delay_range)
        await asyncio.sleep(delay)

        # Simulate failures
        if random.random() > success_rate:
            raise Exception(f"Failed to scrape URL: {url}")

        # Generate mock scraped content
        content = generator._generate_content(content_length_range)

        return {
            "url": url,
            "title": generator.fake.sentence(nb_words=random.randint(3, 8)).rstrip("."),
            "content": content,
            "metadata": {
                "scraped_at": datetime.utcnow().isoformat(),
                "content_type": "text/html",
                "word_count": len(content.split()),
                "char_count": len(content),
                "language": "en",
                "status_code": 200,
            },
            "links": [generator.fake.url() for _ in range(random.randint(0, 10))],
            "images": [generator.fake.image_url() for _ in range(random.randint(0, 5))],
        }

    async def mock_scrape_batch(urls: list[str], **kwargs) -> list[dict[str, Any]]:
        """Mock batch URL scraping."""
        results = []
        for url in urls:
            try:
                result = await mock_scrape_url(url, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"url": url, "error": str(e), "success": False})
        return results

    def mock_is_scrapable(url: str) -> bool:
        """Mock URL scrapability check."""
        # Simple heuristic - reject some known problematic URLs
        blocked_domains = ["javascript:", "mailto:", "tel:", "ftp:"]
        return not any(url.startswith(domain) for domain in blocked_domains)

    mock_scraper = AsyncMock()
    mock_scraper.scrape_url = AsyncMock(side_effect=mock_scrape_url)
    mock_scraper.scrape_batch = AsyncMock(side_effect=mock_scrape_batch)
    mock_scraper.is_scrapable = mock_is_scrapable
    mock_scraper.success_rate = success_rate

    return mock_scraper


def create_mock_cache_service(hit_rate: float = 0.8, storage_limit: int = 1000) -> Mock:
    """Create a mock cache service.

    Args:
        hit_rate: Cache hit probability (0.0 to 1.0)
        storage_limit: Maximum number of items to store

    Returns:
        Mock cache service
    """
    cache_storage = {}
    access_times = {}

    def mock_get(key: str) -> Any | None:
        """Mock cache get."""
        # Simulate cache misses
        if random.random() > hit_rate:
            return None

        if key in cache_storage:
            access_times[key] = time.time()
            return cache_storage[key]
        return None

    def mock_set(key: str, value: Any, ttl: int | None = None) -> bool:
        """Mock cache set."""
        # Simulate storage limit by evicting oldest items
        if len(cache_storage) >= storage_limit:
            # Remove oldest accessed item
            oldest_key = min(access_times.keys(), key=lambda k: access_times[k])
            del cache_storage[oldest_key]
            del access_times[oldest_key]

        cache_storage[key] = value
        access_times[key] = time.time()
        return True

    def mock_delete(key: str) -> bool:
        """Mock cache delete."""
        if key in cache_storage:
            del cache_storage[key]
            del access_times[key]
            return True
        return False

    def mock_exists(key: str) -> bool:
        """Mock cache exists check."""
        return key in cache_storage

    def mock_clear() -> int:
        """Mock cache clear."""
        count = len(cache_storage)
        cache_storage.clear()
        access_times.clear()
        return count

    def mock_stats() -> dict[str, Any]:
        """Mock cache statistics."""
        return {
            "items_count": len(cache_storage),
            "hit_rate": hit_rate,
            "storage_limit": storage_limit,
            "memory_usage": sum(len(str(v)) for v in cache_storage.values()),
        }

    mock_cache = Mock()
    mock_cache.get = mock_get
    mock_cache.set = mock_set
    mock_cache.delete = mock_delete
    mock_cache.exists = mock_exists
    mock_cache.clear = mock_clear
    mock_cache.stats = mock_stats
    mock_cache._storage = cache_storage  # For testing purposes

    return mock_cache


def create_mock_api_client(
    base_url: str = "https://api.example.com",
    auth_required: bool = True,
    rate_limit: int = 100,
) -> AsyncMock:
    """Create a mock API client.

    Args:
        base_url: Base URL for the API
        auth_required: Whether authentication is required
        rate_limit: Rate limit for requests per minute

    Returns:
        Mock API client
    """
    request_count = 0
    last_reset = time.time()

    async def mock_request(
        method: str,
        endpoint: str,
        params: Dict | None = None,
        json_data: Dict | None = None,
        headers: Dict | None = None,
    ) -> dict[str, Any]:
        """Mock API request."""
        nonlocal request_count, last_reset

        # Reset rate limit counter every minute
        current_time = time.time()
        if current_time - last_reset >= 60:
            request_count = 0
            last_reset = current_time

        # Check rate limit
        if request_count >= rate_limit:
            raise Exception("Rate limit exceeded")

        request_count += 1

        # Simulate authentication check
        if auth_required and (not headers or "Authorization" not in headers):
            raise Exception("Authentication required")

        # Simulate network delay
        await asyncio.sleep(random.uniform(0.1, 0.3))

        # Return mock response based on endpoint
        if endpoint.startswith("/search"):
            return {
                "results": [
                    {
                        "id": str(uuid.uuid4()),
                        "title": f"Result {i}",
                        "score": 0.9 - i * 0.1,
                    }
                    for i in range(random.randint(1, 10))
                ],
                "total": random.randint(10, 100),
                "query": params.get("q", "") if params else "",
            }
        elif endpoint.startswith("/documents"):
            if method.upper() == "POST":
                return {
                    "id": str(uuid.uuid4()),
                    "status": "created",
                    "url": json_data.get("url", "") if json_data else "",
                }
            else:
                return {
                    "documents": [
                        {"id": str(uuid.uuid4()), "title": f"Document {i}"}
                        for i in range(random.randint(1, 5))
                    ]
                }
        else:
            return {"message": "Mock response", "endpoint": endpoint}

    async def mock_get(endpoint: str, **kwargs):
        return await mock_request("GET", endpoint, **kwargs)

    async def mock_post(endpoint: str, **kwargs):
        return await mock_request("POST", endpoint, **kwargs)

    async def mock_put(endpoint: str, **kwargs):
        return await mock_request("PUT", endpoint, **kwargs)

    async def mock_delete(endpoint: str, **kwargs):
        return await mock_request("DELETE", endpoint, **kwargs)

    def mock_get_stats():
        return {
            "base_url": base_url,
            "request_count": request_count,
            "rate_limit": rate_limit,
            "auth_required": auth_required,
        }

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(side_effect=mock_request)
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.put = AsyncMock(side_effect=mock_put)
    mock_client.delete = AsyncMock(side_effect=mock_delete)
    mock_client.get_stats = mock_get_stats
    mock_client.base_url = base_url

    return mock_client
