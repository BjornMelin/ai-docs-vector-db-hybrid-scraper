"""Comprehensive tests for Crawl4AI browser adapter."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Crawl4AIConfig
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def basic_crawl4ai_config():
    """Basic Crawl4AI configuration."""
    return Crawl4AIConfig(
        max_concurrent_crawls=5,
        headless=True,
        browser_type="chromium",
        page_timeout=30.0,
    )


@pytest.fixture
def mock_crawl4ai_config():
    """Mock Crawl4AI configuration."""
    return MagicMock(spec=Crawl4AIConfig)


class TestCrawl4AIAdapterInit:
    """Test Crawl4AIAdapter initialization."""

    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    def test_init_with_basic_crawl4ai_config(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test initialization with basic configuration."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        assert adapter._provider == mock_provider
        assert adapter._initialized is False
        mock_provider_class.assert_called_once_with(
            config=basic_crawl4ai_config, rate_limiter=None
        )

    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    def test_init_with_crawl4ai_config(self, mock_provider_class, mock_crawl4ai_config):
        """Test initialization with mock configuration."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        Crawl4AIAdapter(mock_crawl4ai_config)

        mock_provider_class.assert_called_once_with(
            config=mock_crawl4ai_config, rate_limiter=None
        )


class TestCrawl4AIAdapterInitialization:
    """Test adapter initialization process."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_initialize_success(self, mock_provider_class, basic_crawl4ai_config):
        """Test successful initialization."""
        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        await adapter.initialize()

        assert adapter._initialized is True
        mock_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_initialize_already_initialized(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test that re-initialization is skipped."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        await adapter.initialize()

        # Should not call initialize again
        mock_provider.initialize.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_initialize_provider_failure(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test initialization failure from provider."""
        mock_provider = AsyncMock()
        mock_provider.initialize.side_effect = Exception("Provider init failed")
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        with pytest.raises(
            CrawlServiceError, match="Failed to initialize Crawl4AI adapter"
        ):
            await adapter.initialize()

        assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_cleanup_success(self, mock_provider_class, basic_crawl4ai_config):
        """Test successful cleanup."""
        mock_provider = AsyncMock()
        mock_provider.cleanup = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        await adapter.cleanup()

        mock_provider.cleanup.assert_called_once()
        assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_cleanup_with_error(self, mock_provider_class, basic_crawl4ai_config):
        """Test cleanup continues even if provider cleanup fails."""
        mock_provider = AsyncMock()
        mock_provider.cleanup.side_effect = Exception("Cleanup failed")
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Should not raise exception
        await adapter.cleanup()

        assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_cleanup_no_provider(
        self, _mock_provider_class, basic_crawl4ai_config
    ):
        """Test cleanup when provider is None."""
        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._provider = None

        # Should not raise exception
        await adapter.cleanup()


class TestCrawl4AIAdapterScraping:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_not_initialized(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping when adapter is not initialized."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_success_basic(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test successful basic scraping."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "success": True,
            "content": "Test content",
            "html": "<div>Test content</div>",
            "title": "Test Page",
            "metadata": {"url": "https://example.com"},
            "links": [{"text": "Link", "href": "https://example.com/link"}],
            "structured_data": {"type": "article"},
        }
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["content"] == "Test content"
        assert result["html"] == "<div>Test content</div>"
        assert result["title"] == "Test Page"
        assert result["metadata"]["extraction_method"] == "crawl4ai"
        assert "processing_time_ms" in result["metadata"]
        assert result["links"] == [{"text": "Link", "href": "https://example.com/link"}]
        assert result["structured_data"] == {"type": "article"}

        # Verify provider was called correctly
        mock_provider.scrape_url.assert_called_once_with(
            url="https://example.com",
            formats=["markdown"],
            extraction_type="markdown",
            wait_for=None,
            js_code=None,
        )

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_with_wait_selector(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping with wait selector."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {"success": True, "content": "Content"}
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        await adapter.scrape(
            "https://example.com",
            wait_for_selector=".dynamic-content",
        )

        mock_provider.scrape_url.assert_called_once_with(
            url="https://example.com",
            formats=["markdown"],
            extraction_type="markdown",
            wait_for=".dynamic-content",
            js_code=None,
        )

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_with_js_code(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping with JavaScript code."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {"success": True, "content": "Content"}
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        js_code = "document.querySelector('.more-button').click();"

        await adapter.scrape(
            "https://example.com",
            js_code=js_code,
        )

        mock_provider.scrape_url.assert_called_once_with(
            url="https://example.com",
            formats=["markdown"],
            extraction_type="markdown",
            wait_for=None,
            js_code=js_code,
        )

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_with_timeout(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping with custom timeout."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {"success": True, "content": "Content"}
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        await adapter.scrape(
            "https://example.com",
            timeout=60000,
        )

        # Timeout is passed to the provider call
        mock_provider.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_provider_failure(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping when provider returns failure."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "success": False,
            "error": "Provider error",
            "content": "",
        }
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape("https://example.com")

        assert result["success"] is False
        assert result["error"] == "Provider error"
        assert result["content"] == ""
        assert result["metadata"]["extraction_method"] == "crawl4ai"

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_provider_exception(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping when provider raises exception."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.side_effect = Exception("Provider exception")
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape("https://example.com")

        assert result["success"] is False
        assert "Provider exception" in result["error"]
        assert result["content"] == ""
        assert result["metadata"]["extraction_method"] == "crawl4ai"

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_partial_provider_response(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test scraping with partial provider response."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "success": True,
            "content": "Content only",
            # Missing some optional fields
        }
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape("https://example.com")

        assert result["success"] is True
        assert result["content"] == "Content only"
        assert result["html"] == ""  # Default for missing field
        assert result["title"] == ""  # Default for missing field
        assert result["links"] == []  # Default for missing field
        assert result["structured_data"] == {}  # Default for missing field

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_metadata_enrichment(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test that adapter enriches metadata properly."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "success": True,
            "content": "Content",
            "metadata": {"original": "data"},
        }
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape(
            "https://example.com",
            wait_for_selector=".test",
            js_code="console.log('test');",
        )

        metadata = result["metadata"]
        assert metadata["extraction_method"] == "crawl4ai"
        assert metadata["js_executed"] is True
        assert metadata["wait_selector"] == ".test"
        assert metadata["original"] == "data"  # Original metadata preserved
        assert "processing_time_ms" in metadata


class TestCrawl4AIAdapterBulkCrawling:
    """Test bulk crawling functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_crawl_bulk_not_initialized(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test bulk crawling when adapter is not initialized."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.crawl_bulk(["https://example.com"])

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_crawl_bulk_success(self, mock_provider_class, basic_crawl4ai_config):
        """Test successful bulk crawling."""
        mock_provider = AsyncMock()
        mock_provider.crawl_bulk.return_value = [
            {
                "success": True,
                "url": "https://example.com/1",
                "content": "Content 1",
                "html": "<div>Content 1</div>",
                "title": "Page 1",
                "metadata": {"page": 1},
                "links": [],
                "structured_data": {},
            },
            {
                "success": True,
                "url": "https://example.com/2",
                "content": "Content 2",
                "html": "<div>Content 2</div>",
                "title": "Page 2",
                "metadata": {"page": 2},
                "links": [],
                "structured_data": {},
            },
        ]
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        urls = ["https://example.com/1", "https://example.com/2"]
        results = await adapter.crawl_bulk(urls)

        assert len(results) == 2

        for i, result in enumerate(results, 1):
            assert result["success"] is True
            assert result["url"] == f"https://example.com/{i}"
            assert result["content"] == f"Content {i}"
            assert result["metadata"]["extraction_method"] == "crawl4ai_bulk"

        mock_provider.crawl_bulk.assert_called_once_with(urls, "markdown")

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_crawl_bulk_with_failures(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test bulk crawling with some failures."""
        mock_provider = AsyncMock()
        mock_provider.crawl_bulk.return_value = [
            {
                "success": True,
                "url": "https://example.com/1",
                "content": "Content 1",
                "metadata": {},
            },
            {
                "success": False,
                "url": "https://example.com/2",
                "error": "Failed to load",
            },
        ]
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        urls = ["https://example.com/1", "https://example.com/2"]
        results = await adapter.crawl_bulk(urls)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"] == "Failed to load"
        assert results[1]["metadata"]["extraction_method"] == "crawl4ai_bulk"

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_crawl_bulk_custom_extraction_type(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test bulk crawling with custom extraction type."""
        mock_provider = AsyncMock()
        mock_provider.crawl_bulk.return_value = []
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        await adapter.crawl_bulk(["https://example.com"], extraction_type="html")

        mock_provider.crawl_bulk.assert_called_once_with(
            ["https://example.com"], "html"
        )


class TestCrawl4AIAdapterCapabilities:
    """Test capabilities and metadata functionality."""

    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    def test_get_capabilities(self, mock_provider_class, basic_crawl4ai_config):
        """Test getting adapter capabilities."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        capabilities = adapter.get_capabilities()

        assert capabilities["name"] == "crawl4ai"
        assert "description" in capabilities
        assert "advantages" in capabilities
        assert "limitations" in capabilities
        assert "best_for" in capabilities
        assert "performance" in capabilities
        assert capabilities["javascript_support"] == "basic"
        assert capabilities["dynamic_content"] == "limited"
        assert capabilities["authentication"] is False
        assert capabilities["cost"] == 0

    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    def test_capabilities_advantages_and_limitations(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test that capabilities include expected advantages and limitations."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        capabilities = adapter.get_capabilities()

        # Check advantages
        advantages = capabilities["advantages"]
        assert "4-6x faster" in " ".join(advantages)
        assert "Zero cost" in advantages
        assert "parallel processing" in " ".join(advantages).lower()

        # Check limitations
        limitations = capabilities["limitations"]
        assert any("JavaScript" in limitation for limitation in limitations)
        assert any("AI" in limitation for limitation in limitations)

        # Check best use cases
        best_for = capabilities["best_for"]
        assert "Documentation sites" in best_for
        assert "Static content" in best_for
        assert "Bulk crawling" in best_for


class TestCrawl4AIAdapterHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_health_check_not_initialized(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test health check when adapter is not initialized."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        health = await adapter.health_check()

        assert health["healthy"] is False
        assert health["status"] == "not_initialized"
        assert "not initialized" in health["message"]

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_health_check_success(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test successful health check."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Mock successful scrape
        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": True, "content": "test"}

            health = await adapter.health_check()

            assert health["healthy"] is True
            assert health["status"] == "operational"
            assert "Health check passed" in health["message"]
            assert "response_time_ms" in health
            assert health["test_url"] == "https://httpbin.org/html"
            assert "capabilities" in health

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_health_check_failure(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test health check when scraping fails."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Mock failed scrape
        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": False, "error": "Scrape failed"}

            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "degraded"
            assert "Scrape failed" in health["message"]

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_health_check_timeout(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test health check timeout."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Mock timeout
        async def timeout_scrape(*_args, **_kwargs):
            await asyncio.sleep(15)  # Longer than timeout

        with patch.object(adapter, "scrape", side_effect=timeout_scrape):
            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "timeout"
            assert health["response_time_ms"] == 10000

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_health_check_exception(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test health check exception handling."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Mock exception
        with patch.object(adapter, "scrape", side_effect=Exception("Test error")):
            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "error"
            assert "Test error" in health["message"]


class TestCrawl4AIAdapterPerformanceMetrics:
    """Test performance metrics functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_get_performance_metrics_with_provider_metrics(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test getting performance metrics when provider has metrics."""
        mock_provider = AsyncMock()
        mock_provider.metrics = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "avg_response_time": 0.5,
            "cache_hits": 20,
            "cache_misses": 80,
        }
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        metrics = await adapter.get_performance_metrics()

        assert metrics == mock_provider.metrics
        assert metrics["total_requests"] == 100
        assert metrics["successful_requests"] == 95
        assert metrics["avg_response_time"] == 0.5

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_get_performance_metrics_without_provider_metrics(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test getting performance metrics when provider doesn't have metrics."""
        mock_provider = AsyncMock()
        # Provider doesn't have metrics attribute
        del mock_provider.metrics
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        metrics = await adapter.get_performance_metrics()

        # Should return default metrics
        expected_defaults = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        assert metrics == expected_defaults


class TestCrawl4AIAdapterTimeTracking:
    """Test time tracking in scraping operations."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_time_tracking(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test that scraping properly tracks processing time."""
        mock_provider = AsyncMock()

        # Add delay to provider call
        async def delayed_scrape(*_args, **_kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return {"success": True, "content": "test"}

        mock_provider.scrape_url = delayed_scrape
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        start_time = time.time()
        result = await adapter.scrape("https://example.com")
        end_time = time.time()

        assert result["success"] is True
        processing_time = result["metadata"]["processing_time_ms"]

        # Should be roughly 100ms (allowing for some variance)
        assert 50 <= processing_time <= 200

        # Should be consistent with actual elapsed time
        actual_elapsed = (end_time - start_time) * 1000
        assert abs(processing_time - actual_elapsed) < 50  # Within 50ms

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_time_tracking_on_error(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test time tracking when scraping fails."""
        mock_provider = AsyncMock()

        # Add delay before failure
        async def delayed_failure(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            raise Exception("Provider failed")

        mock_provider.scrape_url = delayed_failure
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape("https://example.com")

        assert result["success"] is False
        processing_time = result["metadata"]["processing_time_ms"]
        assert processing_time >= 100  # Should still track time on error


class TestCrawl4AIAdapterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_empty_url(self, mock_provider_class, basic_crawl4ai_config):
        """Test scraping with empty URL."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {"success": True, "content": ""}
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape("")

        assert result["url"] == ""
        mock_provider.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_scrape_none_values(self, mock_provider_class, basic_crawl4ai_config):
        """Test scraping with None values for optional parameters."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {"success": True, "content": "test"}
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        result = await adapter.scrape(
            "https://example.com",
            wait_for_selector=None,
            js_code=None,
        )

        assert result["success"] is True

        # Check provider was called with None values passed through
        call_args = mock_provider.scrape_url.call_args[1]
        assert call_args["wait_for"] is None
        assert call_args["js_code"] is None

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_crawl_bulk_empty_urls(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test bulk crawling with empty URL list."""
        mock_provider = AsyncMock()
        mock_provider.crawl_bulk.return_value = []
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        results = await adapter.crawl_bulk([])

        assert results == []
        mock_provider.crawl_bulk.assert_called_once_with([], "markdown")

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_provider_returns_none(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test handling when provider returns None."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = None
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Should handle None response gracefully
        result = await adapter.scrape("https://example.com")

        # Should treat None as unsuccessful
        assert result["success"] is False


class TestCrawl4AIAdapterIntegration:
    """Integration tests for Crawl4AI adapter."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_full_workflow(self, mock_provider_class, basic_crawl4ai_config):
        """Test complete workflow from initialization to cleanup."""
        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.cleanup = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "success": True,
            "content": "Test content",
            "html": "<div>Test</div>",
            "title": "Test Page",
            "metadata": {"extraction_time": 0.5},
        }
        mock_provider.crawl_bulk.return_value = [
            {"success": True, "url": "https://example.com/1", "content": "Content 1"},
            {"success": True, "url": "https://example.com/2", "content": "Content 2"},
        ]
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)

        # Initialize
        await adapter.initialize()
        assert adapter._initialized is True

        # Single scrape
        result = await adapter.scrape("https://example.com")
        assert result["success"] is True
        assert result["content"] == "Test content"

        # Bulk crawl
        bulk_results = await adapter.crawl_bulk(
            ["https://example.com/1", "https://example.com/2"]
        )
        assert len(bulk_results) == 2
        assert all(r["success"] for r in bulk_results)

        # Health check
        health = await adapter.health_check()
        assert health["healthy"] is True

        # Performance metrics
        mock_provider.metrics = {"total_requests": 3}
        metrics = await adapter.get_performance_metrics()
        assert metrics["total_requests"] == 3

        # Capabilities
        capabilities = adapter.get_capabilities()
        assert capabilities["name"] == "crawl4ai"

        # Cleanup
        await adapter.cleanup()
        assert adapter._initialized is False

        # Verify all provider methods were called
        mock_provider.initialize.assert_called_once()
        mock_provider.scrape_url.assert_called()
        mock_provider.crawl_bulk.assert_called_once()
        mock_provider.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_error_recovery_scenarios(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test adapter behavior under various error conditions."""
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Test scenario 1: Provider returns malformed response
        mock_provider.scrape_url.return_value = {"invalid": "response"}
        result = await adapter.scrape("https://example.com")
        assert result["success"] is False  # Should handle gracefully

        # Test scenario 2: Provider timeout
        async def timeout_provider(*_args, **_kwargs):
            await asyncio.sleep(2)
            return {"success": True, "content": "delayed"}

        mock_provider.scrape_url = timeout_provider
        result = await adapter.scrape("https://example.com")
        assert result["success"] is True  # Should complete successfully

        # Test scenario 3: Memory/resource constraints simulation
        # Reset mock to AsyncMock before setting side_effect
        mock_provider.scrape_url = AsyncMock()
        mock_provider.scrape_url.side_effect = MemoryError("Out of memory")
        result = await adapter.scrape("https://example.com")
        assert result["success"] is False
        assert "Out of memory" in result["error"]

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIProvider")
    async def test_concurrent_operations(
        self, mock_provider_class, basic_crawl4ai_config
    ):
        """Test adapter behavior with concurrent operations."""
        mock_provider = AsyncMock()

        # Simulate varying response times
        async def variable_response(url, **_kwargs):
            delay = 0.1 if "slow" in url else 0.01
            await asyncio.sleep(delay)
            return {"success": True, "content": f"Content for {url}"}

        mock_provider.scrape_url = variable_response
        mock_provider_class.return_value = mock_provider

        adapter = Crawl4AIAdapter(basic_crawl4ai_config)
        adapter._initialized = True

        # Run multiple scrapes concurrently
        urls = [
            "https://fast.example.com",
            "https://slow.example.com",
            "https://medium.example.com",
        ]

        tasks = [adapter.scrape(url) for url in urls]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r["success"] for r in results)

        # Fast URL should complete before slow URL
        fast_time = results[0]["metadata"]["processing_time_ms"]
        slow_time = results[1]["metadata"]["processing_time_ms"]
        assert slow_time > fast_time
