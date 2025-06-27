class TestError(Exception):
    """Custom exception for this module."""

    pass


"""Tests for crawling manager module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Config
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
            raise TestError(f"Failed to initialize {self.name}")
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
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        assert manager.config == config
        assert manager._unified_browser_manager is None
        assert manager._initialized is False
        assert manager.rate_limiter is None

    def test_init_with_rate_limiter(self):
        """Test CrawlManager initialization with rate limiter."""
        config = MagicMock(spec=Config)
        rate_limiter = MagicMock()
        manager = CrawlManager(config, rate_limiter)

        assert manager.rate_limiter == rate_limiter

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization with UnifiedBrowserManager."""
        config = MagicMock(spec=Config)

        with patch(
            "src.services.browser.unified_manager.UnifiedBrowserManager"
        ) as mock_unified_manager:
            mock_unified_instance = AsyncMock()
            mock_unified_manager.return_value = mock_unified_instance

            manager = CrawlManager(config)
            await manager.initialize()

            assert manager._initialized is True
            assert manager._unified_browser_manager is not None
            mock_unified_manager.assert_called_once_with(config)
            mock_unified_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_only(self):
        """Test initialization always uses UnifiedBrowserManager."""
        config = MagicMock(spec=Config)

        with patch(
            "src.services.browser.unified_manager.UnifiedBrowserManager"
        ) as mock_unified_manager:
            mock_unified_instance = AsyncMock()
            mock_unified_manager.return_value = mock_unified_instance

            manager = CrawlManager(config)
            await manager.initialize()

            assert manager._initialized is True
            assert manager._unified_browser_manager is not None
            # UnifiedBrowserManager handles all tiers internally

    @pytest.mark.asyncio
    async def test_initialize_failure_both_providers(self):
        """Test initialization failure when UnifiedBrowserManager fails."""
        config = MagicMock(spec=Config)
        config.crawl4ai = MagicMock()  # Add Crawl4AI config
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = "test_key"

        with patch(
            "src.services.browser.unified_manager.UnifiedBrowserManager"
        ) as mock_ubm:
            mock_ubm.side_effect = Exception("UnifiedBrowserManager init failed")

            manager = CrawlManager(config)

            with pytest.raises(
                CrawlServiceError, match="Failed to initialize crawl manager"
            ):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that multiple initialization calls are safe."""
        config = MagicMock(spec=Config)
        config.crawl4ai = MagicMock()  # Add Crawl4AI config
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = None

        with patch(
            "src.services.browser.unified_manager.UnifiedBrowserManager"
        ) as mock_unified_manager:
            mock_unified_manager_instance = AsyncMock()
            mock_unified_manager.return_value = mock_unified_manager_instance

            manager = CrawlManager(config)

            # Initialize twice
            await manager.initialize()
            await manager.initialize()

            # Should only initialize once
            mock_unified_manager.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        # Add mock UnifiedBrowserManager
        mock_unified_manager = AsyncMock()
        manager._unified_browser_manager = mock_unified_manager
        manager._initialized = True

        await manager.cleanup()

        mock_unified_manager.cleanup.assert_called_once()
        assert manager._unified_browser_manager is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self):
        """Test cleanup with UnifiedBrowserManager error."""
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        # UnifiedBrowserManager that fails cleanup
        failing_unified_manager = AsyncMock()
        failing_unified_manager.cleanup.side_effect = Exception("Cleanup failed")

        manager._unified_browser_manager = failing_unified_manager
        manager._initialized = True

        # Should not raise exception, just log error
        await manager.cleanup()

        failing_unified_manager.cleanup.assert_called_once()
        assert manager._unified_browser_manager is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self):
        """Test scrape_url when manager not initialized."""
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await manager.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self):
        """Test successful URL scraping with UnifiedBrowserManager."""
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        # Mock UnifiedBrowserManager
        mock_unified_manager = AsyncMock()

        # Mock scrape response as a simple object with required attributes
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.content = "Mock content"
        mock_response.url = "https://example.com"
        mock_response.title = "Mock Title"
        mock_response.metadata = {}
        mock_response.tier_used = "crawl4ai"
        mock_response.execution_time_ms = 100
        mock_response.quality_score = 0.9
        mock_response.error = None
        mock_response.fallback_attempted = False
        mock_response.failed_tiers = []

        mock_unified_manager.scrape.return_value = mock_response

        manager._unified_browser_manager = mock_unified_manager
        manager._initialized = True

        result = await manager.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["tier_used"] == "crawl4ai"
        assert result["url"] == "https://example.com"
        assert result["content"] == "Mock content"
        mock_unified_manager.scrape.assert_called_once()

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_scrape_url_with_preferred_provider(self):
        """Test scraping with preferred provider."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_scrape_url_fallback(self):
        """Test provider fallback on failure."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_scrape_url_all_providers_fail(self):
        """Test when all providers fail."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete API - scrape_url no longer accepts formats parameter"
    )
    @pytest.mark.asyncio
    async def test_scrape_url_with_formats(self):
        """Test scraping with specific formats."""
        config = MagicMock(spec=Config)
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
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await manager.crawl_site("https://example.com")

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_crawl_site_success(self):
        """Test successful site crawling."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_crawl_site_prefers_crawl4ai(self):
        """Test that site crawling prefers Crawl4AI when available."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_crawl_site_with_fallback(self):
        """Test site crawling with fallback on provider failure."""
        config = MagicMock(spec=Config)
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
                raise TestError("Crawl4AI provider failed")

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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_crawl_site_all_fail(self):
        """Test site crawling when all providers fail."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    def test_get_provider_info(self):
        """Test getting provider information."""
        config = MagicMock(spec=Config)
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
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)

        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await manager.map_url("https://example.com")

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_map_url_no_firecrawl(self):
        """Test map_url when Firecrawl provider not available."""
        config = MagicMock(spec=Config)
        manager = CrawlManager(config)
        provider = MockCrawlProvider("crawl4ai")
        manager.providers = {"crawl4ai": provider}
        manager._initialized = True

        result = await manager.map_url("https://example.com")

        assert result["success"] is False
        assert "URL mapping requires Firecrawl provider" in result["error"]
        assert result["urls"] == []
        assert result["total"] == 0

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_map_url_with_firecrawl(self):
        """Test map_url with Firecrawl provider."""
        config = MagicMock(spec=Config)
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
    async def test_unified_browser_manager_initialization_config(self):
        """Test UnifiedBrowserManager initialization with correct configuration."""
        config = MagicMock(spec=Config)
        config.crawl4ai = MagicMock()  # Mock Crawl4AI config
        config.firecrawl = MagicMock()
        config.firecrawl.api_key = "test_api_key"

        rate_limiter = MagicMock()

        with patch(
            "src.services.browser.unified_manager.UnifiedBrowserManager"
        ) as mock_unified_manager:
            mock_unified_manager_instance = AsyncMock()
            mock_unified_manager.return_value = mock_unified_manager_instance

            manager = CrawlManager(config, rate_limiter)
            await manager.initialize()

            # Check that UnifiedBrowserManager is called with correct config
            mock_unified_manager.assert_called_once_with(config)
            mock_unified_manager_instance.initialize.assert_called_once()

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_provider_order_logic(self):
        """Test provider selection order logic."""
        config = MagicMock(spec=Config)
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

    @pytest.mark.skip(
        reason="Test uses obsolete provider-based architecture - needs update for UnifiedBrowserManager"
    )
    @pytest.mark.asyncio
    async def test_invalid_preferred_provider(self):
        """Test handling of invalid preferred provider."""
        config = MagicMock(spec=Config)
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
