"""Integration tests for CrawlManager with enhanced Crawl4AI."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.crawling.manager import CrawlManager


class TestCrawlManagerIntegration:
    """Test CrawlManager with enhanced providers."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        # Set up nested attributes properly
        performance = MagicMock()
        performance.max_concurrent_requests = 10
        performance.request_timeout = 30.0
        config.performance = performance

        firecrawl = MagicMock()
        firecrawl.api_key = None
        config.firecrawl = firecrawl

        return config

    @pytest.fixture
    def config_with_firecrawl(self):
        """Create test configuration with Firecrawl."""
        config = MagicMock(spec=UnifiedConfig)
        config.crawl_provider = "crawl4ai"

        # Set up nested attributes properly
        performance = MagicMock()
        performance.max_concurrent_requests = 10
        performance.request_timeout = 30.0
        config.performance = performance

        firecrawl = MagicMock()
        firecrawl.api_key = "test-firecrawl-key"
        config.firecrawl = firecrawl

        return config

    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_primary(self, config):
        """Test Crawl4AI initialized as primary provider."""
        manager = CrawlManager(config)

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            mock_provider = AsyncMock()
            mock_crawl4ai.return_value = mock_provider

            await manager.initialize()

            assert "crawl4ai" in manager.providers
            assert manager._initialized
            mock_crawl4ai.assert_called_once()
            mock_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_both_providers(self, config_with_firecrawl):
        """Test initialization with both providers."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance
            mock_firecrawl.return_value = mock_firecrawl_instance

            await manager.initialize()

            assert "crawl4ai" in manager.providers
            assert "firecrawl" in manager.providers
            assert len(manager.providers) == 2

            # Verify Crawl4AI initialized first (primary)
            mock_crawl4ai.assert_called_once()
            mock_firecrawl.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_prefers_crawl4ai(self, config_with_firecrawl):
        """Test URL scraping prefers Crawl4AI."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            # Setup providers
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance
            mock_firecrawl.return_value = mock_firecrawl_instance

            # Mock successful Crawl4AI response
            mock_crawl4ai_instance.scrape_url.return_value = {
                "success": True,
                "content": "Crawl4AI content",
                "url": "https://example.com",
                "metadata": {},
            }

            await manager.initialize()

            result = await manager.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "Crawl4AI content"
            assert result["provider"] == "crawl4ai"

            # Verify Crawl4AI was called, not Firecrawl
            mock_crawl4ai_instance.scrape_url.assert_called_once()
            mock_firecrawl_instance.scrape_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_fallback_to_firecrawl(self, config_with_firecrawl):
        """Test fallback to Firecrawl when Crawl4AI fails."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            # Setup providers
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance
            mock_firecrawl.return_value = mock_firecrawl_instance

            # Mock Crawl4AI failure
            mock_crawl4ai_instance.scrape_url.return_value = {
                "success": False,
                "error": "Crawl4AI failed",
            }

            # Mock Firecrawl success
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": True,
                "content": "Firecrawl content",
                "url": "https://example.com",
                "metadata": {},
            }

            await manager.initialize()

            result = await manager.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "Firecrawl content"
            assert result["provider"] == "firecrawl"

            # Verify both were called
            mock_crawl4ai_instance.scrape_url.assert_called_once()
            mock_firecrawl_instance.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_site_uses_crawl4ai(self, config_with_firecrawl):
        """Test site crawling uses Crawl4AI by default."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            # Setup providers
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance
            mock_firecrawl.return_value = mock_firecrawl_instance

            # Mock Crawl4AI site crawl
            mock_crawl4ai_instance.crawl_site.return_value = {
                "success": True,
                "pages": [
                    {"url": "https://example.com", "content": "Main page"},
                    {"url": "https://example.com/page1", "content": "Page 1"},
                ],
                "total": 2,
            }

            await manager.initialize()

            result = await manager.crawl_site("https://example.com", max_pages=10)

            assert result["success"] is True
            assert result["total"] == 2
            assert result["provider"] == "crawl4ai"

            # Verify Crawl4AI was called
            mock_crawl4ai_instance.crawl_site.assert_called_once_with(
                "https://example.com", 10, None
            )
            mock_firecrawl_instance.crawl_site.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_provider_selection(self, config_with_firecrawl):
        """Test explicit provider selection."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            # Setup providers
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance
            mock_firecrawl.return_value = mock_firecrawl_instance

            # Mock Firecrawl response
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": True,
                "content": "Firecrawl content",
                "url": "https://example.com",
                "metadata": {},
            }

            await manager.initialize()

            # Explicitly request Firecrawl
            result = await manager.scrape_url(
                "https://example.com", preferred_provider="firecrawl"
            )

            assert result["success"] is True
            assert result["provider"] == "firecrawl"

            # Verify only Firecrawl was called
            mock_firecrawl_instance.scrape_url.assert_called_once()
            mock_crawl4ai_instance.scrape_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_provider_info(self, config_with_firecrawl):
        """Test provider information retrieval."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            mock_crawl4ai.return_value = AsyncMock()
            mock_firecrawl.return_value = AsyncMock()

            await manager.initialize()

            info = manager.get_provider_info()

            assert "crawl4ai" in info
            assert "firecrawl" in info
            assert info["crawl4ai"]["is_preferred"] is True
            assert info["firecrawl"]["is_preferred"] is False
            assert info["firecrawl"]["has_api_key"] is True

    @pytest.mark.asyncio
    async def test_rate_limiting_passed_to_providers(self, config):
        """Test rate limiting is passed to providers."""
        rate_limiter = AsyncMock()
        manager = CrawlManager(config, rate_limiter=rate_limiter)

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            # Create proper async mock for provider
            mock_provider = AsyncMock()
            mock_crawl4ai.return_value = mock_provider

            await manager.initialize()

            # Verify rate limiter was passed to provider
            mock_crawl4ai.assert_called_once()
            call_args = mock_crawl4ai.call_args
            assert call_args.kwargs["rate_limiter"] == rate_limiter

    @pytest.mark.asyncio
    async def test_cleanup(self, config_with_firecrawl):
        """Test manager cleanup."""
        manager = CrawlManager(config_with_firecrawl)

        with (
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
        ):
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance
            mock_firecrawl.return_value = mock_firecrawl_instance

            await manager.initialize()
            assert manager._initialized

            await manager.cleanup()

            assert not manager._initialized
            assert len(manager.providers) == 0
            mock_crawl4ai_instance.cleanup.assert_called_once()
            mock_firecrawl_instance.cleanup.assert_called_once()
