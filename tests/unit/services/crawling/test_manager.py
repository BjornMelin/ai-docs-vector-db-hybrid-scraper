"""Tests for crawling manager module."""

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.crawling.base import CrawlProvider
from src.services.crawling.manager import CrawlManager
from src.services.errors import CrawlServiceError


class MockCrawlProvider(CrawlProvider):
    """Mock crawl provider for testing."""

    def __init__(self, name: str = "mock", should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.initialized = False
        self.cleanup_called = False
        self.scrape_calls = []
        self.crawl_calls = []

    async def initialize(self) -> None:
        if self.should_fail:
            raise Exception(f"Failed to initialize {self.name}")
        self.initialized = True

    async def cleanup(self) -> None:
        self.cleanup_called = True
        self.initialized = False

    async def scrape_url(
        self, url: str, formats: list[str] | None = None
    ) -> dict[str, Any]:
        self.scrape_calls.append({"url": url, "formats": formats})
        if self.should_fail:
            return {"success": False, "error": f"{self.name} failed", "url": url}
        return {
            "success": True,
            "url": url,
            "content": f"Content from {self.name}",
            "metadata": {"provider": self.name},
        }

    async def crawl_site(
        self, url: str, max_pages: int = 50, formats: list[str] | None = None
    ) -> dict[str, Any]:
        self.crawl_calls.append(
            {"url": url, "max_pages": max_pages, "formats": formats}
        )
        if self.should_fail:
            return {
                "success": False,
                "error": f"{self.name} crawl failed",
                "pages": [],
                "total": 0,
            }

        pages = [
            {"url": f"{url}/page{i}", "content": f"Page {i}"}
            for i in range(min(max_pages, 3))
        ]
        return {
            "success": True,
            "pages": pages,
            "total": len(pages),
        }


class TestCrawlManager:
    """Test the CrawlManager class."""

    def test_init(self):
        """Test CrawlManager initialization."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        assert manager.config == config
        assert manager.providers == {}
        assert manager._initialized is False
        assert manager.rate_limiter is None

    def test_init_with_rate_limiter(self):
        """Test CrawlManager initialization with rate limiter."""
        config = MagicMock(spec=UnifiedConfig)
        rate_limiter = MagicMock()
        manager = CrawlManager(config, rate_limiter)

        assert manager.rate_limiter == rate_limiter

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization with both providers."""
        config = MagicMock(spec=UnifiedConfig)
        config.performance = MagicMock()
        config.performance.default_rate_limits = {
            "crawl4ai": {"max_calls": 50, "time_window": 1}
        }
        config.performance.max_concurrent_requests = 10
        config.performance.request_timeout = 30
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = "test_key"

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            mock_firecrawl_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance

            manager = CrawlManager(config)
            await manager.initialize()

            assert manager._initialized is True
            assert len(manager.providers) == 2
            assert "crawl4ai" in manager.providers
            assert "firecrawl" in manager.providers

            mock_crawl4ai_instance.initialize.assert_called_once()
            mock_firecrawl_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_only(self):
        """Test initialization with only Crawl4AI (no Firecrawl API key)."""
        config = MagicMock(spec=UnifiedConfig)
        config.performance = MagicMock()
        config.performance.default_rate_limits = {
            "crawl4ai": {"max_calls": 50, "time_window": 1}
        }
        config.performance.max_concurrent_requests = 10
        config.performance.request_timeout = 30
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = None

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            manager = CrawlManager(config)
            await manager.initialize()

            assert manager._initialized is True
            assert len(manager.providers) == 1
            assert "crawl4ai" in manager.providers
            assert "firecrawl" not in manager.providers

    @pytest.mark.asyncio
    async def test_initialize_failure_both_providers(self):
        """Test initialization failure when both providers fail."""
        config = MagicMock(spec=UnifiedConfig)
        config.performance = MagicMock()
        config.performance.default_rate_limits = {
            "crawl4ai": {"max_calls": 50, "time_window": 1}
        }
        config.performance.max_concurrent_requests = 10
        config.performance.request_timeout = 30
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = "test_key"

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            mock_crawl4ai.side_effect = Exception("Crawl4AI init failed")
            mock_firecrawl.side_effect = Exception("Firecrawl init failed")

            manager = CrawlManager(config)

            with pytest.raises(
                CrawlServiceError, match="No crawling providers available"
            ):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that multiple initialization calls are safe."""
        config = MagicMock(spec=UnifiedConfig)
        config.performance = MagicMock()
        config.performance.default_rate_limits = {
            "crawl4ai": {"max_calls": 50, "time_window": 1}
        }
        config.performance.max_concurrent_requests = 10
        config.performance.request_timeout = 30
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = None

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            manager = CrawlManager(config)

            # Initialize twice
            await manager.initialize()
            await manager.initialize()

            # Should only initialize once
            mock_crawl4ai.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        # Add mock providers
        provider1 = AsyncMock()
        provider2 = AsyncMock()
        manager.providers = {"provider1": provider1, "provider2": provider2}
        manager._initialized = True

        await manager.cleanup()

        provider1.cleanup.assert_called_once()
        provider2.cleanup.assert_called_once()
        assert manager.providers == {}
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self):
        """Test cleanup with provider error."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        # Provider that fails cleanup
        failing_provider = AsyncMock()
        failing_provider.cleanup.side_effect = Exception("Cleanup failed")

        working_provider = AsyncMock()

        manager.providers = {"failing": failing_provider, "working": working_provider}
        manager._initialized = True

        # Should not raise exception, just log error
        await manager.cleanup()

        failing_provider.cleanup.assert_called_once()
        working_provider.cleanup.assert_called_once()
        assert manager.providers == {}
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self):
        """Test scrape_url when manager not initialized."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await manager.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self):
        """Test successful URL scraping."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        provider = MockCrawlProvider("crawl4ai")
        manager.providers = {"crawl4ai": provider}
        manager._initialized = True

        result = await manager.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["provider"] == "crawl4ai"
        assert result["url"] == "https://example.com"
        assert len(provider.scrape_calls) == 1

    @pytest.mark.asyncio
    async def test_scrape_url_with_preferred_provider(self):
        """Test scraping with preferred provider."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        crawl4ai = MockCrawlProvider("crawl4ai")
        firecrawl = MockCrawlProvider("firecrawl")
        manager.providers = {"crawl4ai": crawl4ai, "firecrawl": firecrawl}
        manager._initialized = True

        result = await manager.scrape_url(
            "https://example.com", preferred_provider="firecrawl"
        )

        assert result["success"] is True
        assert result["provider"] == "firecrawl"
        assert len(firecrawl.scrape_calls) == 1
        assert len(crawl4ai.scrape_calls) == 0

    @pytest.mark.asyncio
    async def test_scrape_url_fallback(self):
        """Test provider fallback on failure."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        failing_provider = MockCrawlProvider("crawl4ai", should_fail=True)
        working_provider = MockCrawlProvider("firecrawl")
        manager.providers = {
            "crawl4ai": failing_provider,
            "firecrawl": working_provider,
        }
        manager._initialized = True

        result = await manager.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["provider"] == "firecrawl"
        assert len(failing_provider.scrape_calls) == 1
        assert len(working_provider.scrape_calls) == 1

    @pytest.mark.asyncio
    async def test_scrape_url_all_providers_fail(self):
        """Test when all providers fail."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        provider1 = MockCrawlProvider("crawl4ai", should_fail=True)
        provider2 = MockCrawlProvider("firecrawl", should_fail=True)
        manager.providers = {"crawl4ai": provider1, "firecrawl": provider2}
        manager._initialized = True

        result = await manager.scrape_url("https://example.com")

        assert result["success"] is False
        assert "All providers failed" in result["error"]
        assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_scrape_url_with_formats(self):
        """Test scraping with specific formats."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        provider = MockCrawlProvider("crawl4ai")
        manager.providers = {"crawl4ai": provider}
        manager._initialized = True

        await manager.scrape_url("https://example.com", formats=["html", "markdown"])

        assert len(provider.scrape_calls) == 1
        assert provider.scrape_calls[0]["formats"] == ["html", "markdown"]

    @pytest.mark.asyncio
    async def test_crawl_site_not_initialized(self):
        """Test crawl_site when manager not initialized."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await manager.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_site_success(self):
        """Test successful site crawling."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        provider = MockCrawlProvider("crawl4ai")
        manager.providers = {"crawl4ai": provider}
        manager._initialized = True

        result = await manager.crawl_site("https://example.com", max_pages=25)

        assert result["success"] is True
        assert result["provider"] == "crawl4ai"
        assert result["total"] == 3  # MockCrawlProvider returns 3 pages
        assert len(provider.crawl_calls) == 1
        assert provider.crawl_calls[0]["max_pages"] == 25

    @pytest.mark.asyncio
    async def test_crawl_site_prefers_crawl4ai(self):
        """Test that site crawling prefers Crawl4AI when available."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "firecrawl"  # Config prefers firecrawl

        manager = CrawlManager(config)
        crawl4ai = MockCrawlProvider("crawl4ai")
        firecrawl = MockCrawlProvider("firecrawl")
        manager.providers = {"crawl4ai": crawl4ai, "firecrawl": firecrawl}
        manager._initialized = True

        result = await manager.crawl_site("https://example.com")

        # Should use crawl4ai despite config preference
        assert result["provider"] == "crawl4ai"
        assert len(crawl4ai.crawl_calls) == 1
        assert len(firecrawl.crawl_calls) == 0

    @pytest.mark.asyncio
    async def test_crawl_site_with_fallback(self):
        """Test site crawling with fallback on provider failure."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)

        # Create a failing provider that raises an exception
        class FailingCrawlProvider(MockCrawlProvider):
            async def crawl_site(
                self, url: str, max_pages: int = 50, formats: list[str] | None = None
            ) -> dict[str, Any]:
                self.crawl_calls.append(
                    {"url": url, "max_pages": max_pages, "formats": formats}
                )
                raise Exception("Crawl4AI provider failed")

        failing_provider = FailingCrawlProvider("crawl4ai", should_fail=True)
        working_provider = MockCrawlProvider("firecrawl")
        manager.providers = {
            "crawl4ai": failing_provider,
            "firecrawl": working_provider,
        }
        manager._initialized = True

        result = await manager.crawl_site("https://example.com")

        assert result["success"] is True
        assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_crawl_site_all_fail(self):
        """Test site crawling when all providers fail."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        provider1 = MockCrawlProvider("crawl4ai", should_fail=True)
        provider2 = MockCrawlProvider("firecrawl", should_fail=True)
        manager.providers = {"crawl4ai": provider1, "firecrawl": provider2}
        manager._initialized = True

        result = await manager.crawl_site("https://example.com")

        assert result["success"] is False
        assert result["provider"] == "crawl4ai"  # Primary provider name
        assert result["total"] == 0

    def test_get_provider_info(self):
        """Test getting provider information."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = "test_key"

        manager = CrawlManager(config)
        provider1 = MockCrawlProvider("crawl4ai")
        provider2 = MockCrawlProvider("firecrawl")
        manager.providers = {"crawl4ai": provider1, "firecrawl": provider2}

        info = manager.get_provider_info()

        assert "crawl4ai" in info
        assert "firecrawl" in info
        assert info["crawl4ai"]["is_preferred"] is True
        assert info["firecrawl"]["is_preferred"] is False
        assert info["firecrawl"]["has_api_key"] is True

    @pytest.mark.asyncio
    async def test_map_url_not_initialized(self):
        """Test map_url when manager not initialized."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await manager.map_url("https://example.com")

    @pytest.mark.asyncio
    async def test_map_url_no_firecrawl(self):
        """Test map_url when Firecrawl provider not available."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)
        provider = MockCrawlProvider("crawl4ai")
        manager.providers = {"crawl4ai": provider}
        manager._initialized = True

        result = await manager.map_url("https://example.com")

        assert result["success"] is False
        assert "URL mapping requires Firecrawl provider" in result["error"]
        assert result["urls"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_map_url_with_firecrawl(self):
        """Test map_url with Firecrawl provider."""
        config = MagicMock(spec=UnifiedConfig)
        manager = CrawlManager(config)

        # Mock Firecrawl provider with map_url method
        firecrawl_provider = AsyncMock()
        firecrawl_provider.map_url.return_value = {
            "success": True,
            "urls": ["https://example.com/page1", "https://example.com/page2"],
            "total": 2,
        }

        manager.providers = {"firecrawl": firecrawl_provider}
        manager._initialized = True

        result = await manager.map_url("https://example.com", include_subdomains=True)

        assert result["success"] is True
        assert result["total"] == 2
        firecrawl_provider.map_url.assert_called_once_with("https://example.com", True)

    @pytest.mark.asyncio
    async def test_provider_initialization_config(self):
        """Test provider initialization with correct configuration."""
        config = MagicMock(spec=UnifiedConfig)
        config.performance = MagicMock()
        config.performance.default_rate_limits = {
            "crawl4ai": {"max_calls": 100, "time_window": 1}
        }
        config.performance.max_concurrent_requests = 20
        config.performance.request_timeout = 45
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = "test_api_key"

        rate_limiter = MagicMock()

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            mock_firecrawl_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance

            manager = CrawlManager(config, rate_limiter)
            await manager.initialize()

            # Check Crawl4AI configuration
            crawl4ai_call_args = mock_crawl4ai.call_args
            assert crawl4ai_call_args[1]["config"]["max_concurrent"] == 20
            assert crawl4ai_call_args[1]["config"]["rate_limit"] == 100
            assert (
                crawl4ai_call_args[1]["config"]["page_timeout"] == 45000
            )  # converted to ms
            assert crawl4ai_call_args[1]["rate_limiter"] == rate_limiter

            # Check Firecrawl configuration
            firecrawl_call_args = mock_firecrawl.call_args
            assert firecrawl_call_args[1]["api_key"] == "test_api_key"
            assert firecrawl_call_args[1]["rate_limiter"] == rate_limiter

    @pytest.mark.asyncio
    async def test_provider_order_logic(self):
        """Test provider selection order logic."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "firecrawl"

        manager = CrawlManager(config)
        crawl4ai = MockCrawlProvider("crawl4ai")
        firecrawl = MockCrawlProvider("firecrawl")
        manager.providers = {"crawl4ai": crawl4ai, "firecrawl": firecrawl}
        manager._initialized = True

        # Test with no preferred provider (should use config preference)
        await manager.scrape_url("https://example.com")
        assert len(firecrawl.scrape_calls) == 1
        assert len(crawl4ai.scrape_calls) == 0

        # Reset
        firecrawl.scrape_calls.clear()
        crawl4ai.scrape_calls.clear()

        # Test with explicit preferred provider
        await manager.scrape_url("https://example.com", preferred_provider="crawl4ai")
        assert len(crawl4ai.scrape_calls) == 1
        assert len(firecrawl.scrape_calls) == 0

    @pytest.mark.asyncio
    async def test_invalid_preferred_provider(self):
        """Test handling of invalid preferred provider."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        manager = CrawlManager(config)
        provider = MockCrawlProvider("crawl4ai")
        manager.providers = {"crawl4ai": provider}
        manager._initialized = True

        # Should fall back to available providers
        result = await manager.scrape_url(
            "https://example.com", preferred_provider="nonexistent"
        )

        assert result["success"] is True
        assert result["provider"] == "crawl4ai"
