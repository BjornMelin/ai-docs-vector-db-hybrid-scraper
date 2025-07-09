"""Comprehensive integration tests for external service interactions.

This module demonstrates:
- respx for HTTP mocking (Firecrawl, OpenAI)
- Async patterns with pytest-asyncio
- Realistic service interaction testing
- Error handling and retry logic validation
- Performance and rate limiting tests
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from hypothesis import given, settings, strategies as st

from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.crawling.lightweight_scraper import LightweightScraper
from tests.fixtures.test_infrastructure import (
    PerformanceTestUtils,
    PropertyTestStrategies,
    TestDataFactory,
)


class TestFirecrawlIntegration:
    """Integration tests for Firecrawl service."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_successful_page_scraping(self, respx_mock):
        """Test successful webpage scraping via Firecrawl."""
        # Arrange
        url = "https://example.com/docs"
        expected_content = TestDataFactory.create_document_content(
            title="Example Documentation",
            paragraphs=5,
        )

        respx_mock.post("https://api.firecrawl.dev/v0/scrape").mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "content": expected_content,
                        "markdown": f"# Example Documentation\n\n{expected_content}",
                        "metadata": {
                            "title": "Example Documentation",
                            "description": "Test page",
                            "sourceURL": url,
                        },
                    },
                },
            )
        )

        client = FirecrawlClient(api_key="test-key")

        # Act
        result = await client.scrape_url(url)

        # Assert
        assert result is not None
        assert result["success"] is True
        assert "content" in result["data"]
        assert expected_content in result["data"]["content"]
        assert result["data"]["metadata"]["sourceURL"] == url

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_concurrent_scraping_requests(self, respx_mock):
        """Test concurrent scraping with rate limiting."""
        # Arrange
        urls = [f"https://example.com/page{i}" for i in range(10)]
        call_times = []

        def track_calls(request):
            call_times.append(time.time())
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "content": f"Content for {request.url}",
                        "metadata": {"sourceURL": str(request.url)},
                    },
                },
            )

        respx_mock.post("https://api.firecrawl.dev/v0/scrape").mock(
            side_effect=track_calls
        )

        client = FirecrawlClient(
            api_key="test-key",
            max_concurrent=3,  # Limit concurrent requests
            rate_limit_per_minute=60,
        )

        # Act
        start_time = time.time()
        tasks = [client.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        # Assert
        assert len(results) == 10
        assert all(r["success"] for r in results)
        # With rate limiting, should have controlled request timing
        assert duration < 5.0  # Should complete reasonably fast

        # Check rate limiting worked
        if len(call_times) > 1:
            # Calculate time between calls
            intervals = [
                call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)
            ]
            # At least some intervals should show rate limiting
            assert any(
                interval >= 0.01 for interval in intervals
            )  # Some delay between calls

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_error_handling_and_retry(self, respx_mock):
        """Test error handling and retry logic for failed requests."""
        # Arrange
        url = "https://example.com/flaky"
        attempts = []

        def simulate_flaky_service(request):
            attempts.append(time.time())
            if len(attempts) < 3:
                # Fail first 2 attempts
                return httpx.Response(
                    503,
                    json={"error": "Service temporarily unavailable"},
                )
            # Succeed on 3rd attempt
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {"content": "Success after retries"},
                },
            )

        respx_mock.post("https://api.firecrawl.dev/v0/scrape").mock(
            side_effect=simulate_flaky_service
        )

        client = FirecrawlClient(
            api_key="test-key",
            max_retries=3,
            retry_delay=0.1,  # Short delay for testing
        )

        # Act
        result = await client.scrape_url(url)

        # Assert
        assert len(attempts) == 3
        assert result["success"] is True
        assert result["data"]["content"] == "Success after retries"

        # Verify exponential backoff
        if len(attempts) > 2:
            first_interval = attempts[1] - attempts[0]
            second_interval = attempts[2] - attempts[1]
            assert second_interval > first_interval  # Exponential backoff

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_batch_crawling_with_search(self, respx_mock):
        """Test batch crawling with search functionality."""
        # Arrange
        base_url = "https://docs.example.com"
        search_results = [
            {"url": f"{base_url}/page{i}", "title": f"Page {i}"} for i in range(5)
        ]

        # Mock search endpoint
        respx_mock.post("https://api.firecrawl.dev/v0/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": search_results,
                },
            )
        )

        # Mock scrape endpoints for each result
        for i, result in enumerate(search_results):
            respx_mock.post(
                "https://api.firecrawl.dev/v0/scrape",
                json__contains={"url": result["url"]},
            ).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "content": f"Content for page {i}",
                            "metadata": {"title": result["title"]},
                        },
                    },
                )
            )

        client = FirecrawlClient(api_key="test-key")

        # Act
        # First search for pages
        search_response = await client.search(base_url, query="documentation")

        # Then scrape each result
        scrape_tasks = [
            client.scrape_url(result["url"]) for result in search_response["data"]
        ]
        scrape_results = await asyncio.gather(*scrape_tasks)

        # Assert
        assert search_response["success"] is True
        assert len(search_response["data"]) == 5
        assert len(scrape_results) == 5
        assert all(r["success"] for r in scrape_results)


class TestOpenAIEmbeddingsIntegration:
    """Integration tests for OpenAI embeddings service."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_text_embedding_generation(self, respx_mock):
        """Test single text embedding generation."""
        # Arrange
        text = "This is a test document for embedding generation."
        expected_embedding = [0.1 + i * 0.001 for i in range(1536)]

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": expected_embedding,
                            "index": 0,
                        }
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {
                        "prompt_tokens": 12,
                        "total_tokens": 12,
                    },
                },
            )
        )

        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="text-embedding-3-small",
        )

        # Act
        embedding = await client.create_embedding(text)

        # Assert
        assert len(embedding) == 1536
        assert embedding == expected_embedding
        assert all(isinstance(val, float) for val in embedding)

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_batch_embeddings_with_chunking(self, respx_mock):
        """Test batch embedding generation with automatic chunking."""
        # Arrange
        texts = [f"Document {i}: " + "x" * 1000 for i in range(50)]

        # Mock will be called multiple times due to batching
        call_count = 0

        def mock_batch_response(request):
            nonlocal call_count
            batch_size = len(request.json()["input"])
            call_count += 1

            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.1 + i * 0.01] * 1536,
                            "index": i,
                        }
                        for i in range(batch_size)
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {
                        "prompt_tokens": batch_size * 200,
                        "total_tokens": batch_size * 200,
                    },
                },
            )

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            side_effect=mock_batch_response
        )

        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="text-embedding-3-small",
            max_batch_size=25,  # Force chunking
        )

        # Act
        embeddings = await client.create_embeddings_batch(texts)

        # Assert
        assert len(embeddings) == 50
        assert call_count == 2  # Should be called twice (25 + 25)
        assert all(len(emb) == 1536 for emb in embeddings)

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_rate_limiting_handling(self, respx_mock):
        """Test handling of rate limit responses."""
        # Arrange
        text = "Rate limited text"
        rate_limit_hit = False

        def simulate_rate_limit(request):
            nonlocal rate_limit_hit
            if not rate_limit_hit:
                rate_limit_hit = True
                return httpx.Response(
                    429,
                    json={"error": {"message": "Rate limit exceeded"}},
                    headers={"Retry-After": "1"},
                )
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.5] * 1536, "index": 0}
                    ],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                },
            )

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            side_effect=simulate_rate_limit
        )

        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="text-embedding-3-small",
            handle_rate_limits=True,
        )

        # Act
        start_time = time.time()
        embedding = await client.create_embedding(text)
        duration = time.time() - start_time

        # Assert
        assert rate_limit_hit is True
        assert len(embedding) == 1536
        assert duration >= 0.9  # Should have waited at least the retry-after time

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.parametrize(
        ("model", "dimension"),
        [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
        ],
    )
    async def test_different_embedding_models(self, respx_mock, model, dimension):
        """Test different embedding models and dimensions."""
        # Arrange
        text = "Test text for different models"

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.1] * dimension,
                            "index": 0,
                        }
                    ],
                    "model": model,
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        client = OpenAIEmbeddingClient(api_key="test-key", model=model)

        # Act
        embedding = await client.create_embedding(text)

        # Assert
        assert len(embedding) == dimension


class TestCrawl4AIIntegration:
    """Integration tests for Crawl4AI service."""

    @pytest.mark.asyncio
    async def test_async_webpage_crawling(self):
        """Test async webpage crawling with Crawl4AI."""
        # Arrange
        url = "https://example.com/test"
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        # Setup mock browser behavior
        mock_browser.new_page.return_value = mock_page
        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(
            return_value="<html><body><h1>Test Page</h1><p>Content</p></body></html>"
        )
        mock_page.evaluate = AsyncMock(return_value="Test Page\nContent")

        provider = Crawl4AIProvider()
        provider._browser = mock_browser  # Inject mock browser

        # Act
        result = await provider.crawl_async(url)

        # Assert
        assert result is not None
        assert "Test Page" in result.get("text", "")
        assert result.get("status") == "success"
        mock_page.goto.assert_called_once_with(url)

    @pytest.mark.asyncio
    async def test_javascript_execution(self):
        """Test JavaScript execution during crawling."""
        # Arrange
        url = "https://example.com/dynamic"
        js_code = "document.querySelectorAll('.item').length"

        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_page.return_value = mock_page
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=5)  # 5 items found

        provider = Crawl4AIProvider()
        provider._browser = mock_browser

        # Act
        result = await provider.crawl_with_js(url, js_code)

        # Assert
        assert result == 5
        mock_page.evaluate.assert_called_once_with(js_code)

    @pytest.mark.asyncio
    async def test_concurrent_crawling_with_pool(self):
        """Test concurrent crawling with browser pool management."""
        # Arrange
        urls = [f"https://example.com/page{i}" for i in range(5)]

        # Create mock responses for each URL
        async def create_mock_page(url):
            mock_page = AsyncMock()
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(
                return_value=f"<html><body>Content for {url}</body></html>"
            )
            mock_page.evaluate = AsyncMock(return_value=f"Content for {url}")
            return mock_page

        provider = Crawl4AIProvider(max_concurrent=3)

        # Mock browser pool
        mock_pages = []
        for url in urls:
            page = await create_mock_page(url)
            mock_pages.append(page)

        # Act
        results = await provider.crawl_multiple(urls)

        # Assert
        assert len(results) == 5
        for i, result in enumerate(results):
            assert f"page{i}" in str(result)


class TestLightweightScraperIntegration:
    """Integration tests for lightweight scraper."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_basic_html_scraping(self, respx_mock):
        """Test basic HTML scraping without JavaScript."""
        # Arrange
        url = "https://example.com/simple"
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a paragraph.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """

        respx_mock.get(url).mock(return_value=httpx.Response(200, text=html_content))

        scraper = LightweightScraper()

        # Act
        result = await scraper.scrape(url)

        # Assert
        assert result is not None
        assert "Main Title" in result["text"]
        assert "This is a paragraph." in result["text"]
        assert result["metadata"]["title"] == "Test Page"
        assert result["metadata"]["url"] == url

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_metadata_extraction(self, respx_mock):
        """Test metadata extraction from HTML."""
        # Arrange
        url = "https://example.com/metadata"
        html_content = """
        <html>
            <head>
                <title>Page with Metadata</title>
                <meta name="description" content="Test description">
                <meta name="keywords" content="test, scraping, integration">
                <meta property="og:title" content="Open Graph Title">
                <meta property="og:image" content="https://example.com/image.jpg">
            </head>
            <body>
                <h1>Content</h1>
            </body>
        </html>
        """

        respx_mock.get(url).mock(return_value=httpx.Response(200, text=html_content))

        scraper = LightweightScraper()

        # Act
        result = await scraper.scrape(url)

        # Assert
        metadata = result["metadata"]
        assert metadata["title"] == "Page with Metadata"
        assert metadata["description"] == "Test description"
        assert metadata["keywords"] == "test, scraping, integration"
        assert metadata["og:title"] == "Open Graph Title"
        assert metadata["og:image"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_error_handling(self, respx_mock):
        """Test error handling for various HTTP errors."""
        # Arrange
        test_cases = [
            ("https://example.com/404", 404, "Not Found"),
            ("https://example.com/500", 500, "Internal Server Error"),
            ("https://example.com/timeout", None, "timeout"),
        ]

        for url, status_code, error in test_cases:
            if status_code:
                respx_mock.get(url).mock(
                    return_value=httpx.Response(status_code, text=error)
                )
            else:
                respx_mock.get(url).mock(side_effect=httpx.TimeoutException("timeout"))

        scraper = LightweightScraper()

        # Act & Assert
        for url, status_code, error in test_cases:
            result = await scraper.scrape(url)
            assert result["status"] == "error"
            assert error.lower() in result.get("error", "").lower()


class TestServiceIntegrationScenarios:
    """End-to-end integration scenarios across multiple services."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_complete_document_processing_workflow(self, respx_mock):
        """Test complete workflow: scrape -> extract -> embed."""
        # Arrange
        url = "https://docs.example.com/guide"

        # Mock Firecrawl scraping
        respx_mock.post("https://api.firecrawl.dev/v0/scrape").mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "content": "# Guide\n\nThis is a comprehensive guide.",
                        "metadata": {"title": "User Guide"},
                    },
                },
            )
        )

        # Mock OpenAI embeddings
        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1] * 1536, "index": 0}
                    ],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        # Create service clients
        firecrawl_client = FirecrawlClient(api_key="test-key")
        openai_client = OpenAIEmbeddingClient(api_key="test-key")

        # Act
        # Step 1: Scrape content
        scrape_result = await firecrawl_client.scrape_url(url)
        content = scrape_result["data"]["content"]

        # Step 2: Generate embeddings
        embedding = await openai_client.create_embedding(content)

        # Assert
        assert scrape_result["success"] is True
        assert len(content) > 0
        assert len(embedding) == 1536

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.hypothesis
    @given(
        urls=st.lists(
            st.from_regex(r"https://[a-z]+\.com/[a-z]+", fullmatch=True),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=5, deadline=None)
    async def test_property_based_batch_processing(self, respx_mock, urls):
        """Property-based test for batch processing workflows."""
        # Arrange - Mock all URLs
        for url in urls:
            respx_mock.post(
                "https://api.firecrawl.dev/v0/scrape",
                json__contains={"url": url},
            ).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {"content": f"Content for {url}"},
                    },
                )
            )

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1] * 1536, "index": i}
                        for i in range(len(urls))
                    ],
                    "usage": {
                        "prompt_tokens": len(urls) * 10,
                        "total_tokens": len(urls) * 10,
                    },
                },
            )
        )

        firecrawl_client = FirecrawlClient(api_key="test-key")
        openai_client = OpenAIEmbeddingClient(api_key="test-key")

        # Act
        # Scrape all URLs
        scrape_tasks = [firecrawl_client.scrape_url(url) for url in urls]
        scrape_results = await asyncio.gather(*scrape_tasks)

        # Extract content
        contents = [r["data"]["content"] for r in scrape_results if r["success"]]

        # Generate embeddings
        if contents:
            embeddings = await openai_client.create_embeddings_batch(contents)

            # Assert
            assert len(scrape_results) == len(urls)
            assert all(r["success"] for r in scrape_results)
            assert len(embeddings) == len(contents)
            assert all(len(emb) == 1536 for emb in embeddings)
