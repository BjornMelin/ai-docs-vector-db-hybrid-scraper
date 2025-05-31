"""Comprehensive tests for crawl manager with provider abstraction."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.crawling.manager import CrawlManager
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock unified configuration."""
    config = MagicMock(spec=UnifiedConfig)
    config.crawl_provider = "crawl4ai"
    
    # Mock firecrawl config
    config.firecrawl = MagicMock()
    config.firecrawl.api_key = "fc-test-key"
    
    # Mock performance config
    config.performance = MagicMock()
    config.performance.default_rate_limits = {
        "crawl4ai": {"max_calls": 50, "time_window": 1}
    }
    config.performance.max_concurrent_requests = 5
    config.performance.request_timeout = 30
    return config


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    limiter = AsyncMock()
    limiter.acquire = AsyncMock()
    limiter.release = AsyncMock()
    return limiter


@pytest.fixture
def crawl_manager(mock_config, mock_rate_limiter):
    """Create crawl manager for testing."""
    return CrawlManager(config=mock_config, rate_limiter=mock_rate_limiter)


@pytest.fixture
def mock_crawl4ai_provider():
    """Create mock Crawl4AI provider."""
    provider = AsyncMock()
    provider.initialize = AsyncMock()
    provider.cleanup = AsyncMock()
    provider.scrape_url = AsyncMock(
        return_value={
            "success": True,
            "content": "Scraped content from Crawl4AI",
            "metadata": {"provider": "crawl4ai", "response_time": 1.2},
            "url": "https://example.com",
        }
    )
    provider.crawl_site = AsyncMock(
        return_value={
            "success": True,
            "pages": [
                {"url": "https://example.com", "content": "Page 1"},
                {"url": "https://example.com/page2", "content": "Page 2"},
            ],
            "total": 2,
        }
    )
    return provider


@pytest.fixture
def mock_firecrawl_provider():
    """Create mock Firecrawl provider."""
    provider = AsyncMock()
    provider.initialize = AsyncMock()
    provider.cleanup = AsyncMock()
    provider.scrape_url = AsyncMock(
        return_value={
            "success": True,
            "content": "Scraped content from Firecrawl",
            "metadata": {"provider": "firecrawl", "response_time": 0.8},
            "url": "https://example.com",
        }
    )
    provider.crawl_site = AsyncMock(
        return_value={
            "success": True,
            "pages": [
                {"url": "https://example.com", "content": "FC Page 1"},
                {"url": "https://example.com/about", "content": "FC Page 2"},
            ],
            "total": 2,
        }
    )
    provider.map_url = AsyncMock(
        return_value={
            "success": True,
            "urls": ["https://example.com", "https://example.com/about"],
            "total": 2,
        }
    )
    return provider


class TestCrawlManagerInitialization:
    """Test crawl manager initialization."""

    def test_manager_initialization(
        self, crawl_manager, mock_config, mock_rate_limiter
    ):
        """Test basic manager initialization."""
        assert crawl_manager.config == mock_config
        assert crawl_manager.rate_limiter == mock_rate_limiter
        assert crawl_manager.providers == {}
        assert crawl_manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success_with_both_providers(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test successful initialization with both providers."""
        with (
            patch(
                "src.services.crawling.manager.Crawl4AIProvider"
            ) as mock_crawl4ai_class,
            patch(
                "src.services.crawling.manager.FirecrawlProvider"
            ) as mock_firecrawl_class,
        ):
            mock_crawl4ai_class.return_value = mock_crawl4ai_provider
            mock_firecrawl_class.return_value = mock_firecrawl_provider

            await crawl_manager.initialize()

            assert crawl_manager._initialized is True
            assert len(crawl_manager.providers) == 2
            assert "crawl4ai" in crawl_manager.providers
            assert "firecrawl" in crawl_manager.providers

            # Verify providers were initialized
            mock_crawl4ai_provider.initialize.assert_called_once()
            mock_firecrawl_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_only(
        self, crawl_manager, mock_crawl4ai_provider, mock_config
    ):
        """Test initialization with only Crawl4AI (no Firecrawl API key)."""
        mock_config.firecrawl.api_key = None

        with patch(
            "src.services.crawling.manager.Crawl4AIProvider"
        ) as mock_crawl4ai_class:
            mock_crawl4ai_class.return_value = mock_crawl4ai_provider

            await crawl_manager.initialize()

            assert crawl_manager._initialized is True
            assert len(crawl_manager.providers) == 1
            assert "crawl4ai" in crawl_manager.providers
            assert "firecrawl" not in crawl_manager.providers

    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_failure_firecrawl_success(
        self, crawl_manager, mock_firecrawl_provider
    ):
        """Test initialization when Crawl4AI fails but Firecrawl succeeds."""
        with (
            patch(
                "src.services.crawling.manager.Crawl4AIProvider"
            ) as mock_crawl4ai_class,
            patch(
                "src.services.crawling.manager.FirecrawlProvider"
            ) as mock_firecrawl_class,
        ):
            # Mock Crawl4AI initialization failure
            mock_crawl4ai_class.side_effect = Exception("Crawl4AI init failed")
            mock_firecrawl_class.return_value = mock_firecrawl_provider

            await crawl_manager.initialize()

            assert crawl_manager._initialized is True
            assert len(crawl_manager.providers) == 1
            assert "firecrawl" in crawl_manager.providers
            assert "crawl4ai" not in crawl_manager.providers

    @pytest.mark.asyncio
    async def test_initialize_no_providers_available(self, crawl_manager, mock_config):
        """Test initialization when no providers are available."""
        mock_config.firecrawl.api_key = None

        with patch(
            "src.services.crawling.manager.Crawl4AIProvider"
        ) as mock_crawl4ai_class:
            # Mock Crawl4AI initialization failure
            mock_crawl4ai_class.side_effect = Exception("No providers available")

            with pytest.raises(
                CrawlServiceError, match="No crawling providers available"
            ):
                await crawl_manager.initialize()

            assert crawl_manager._initialized is False
            assert len(crawl_manager.providers) == 0

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test that initialization is idempotent."""
        with (
            patch(
                "src.services.crawling.manager.Crawl4AIProvider"
            ) as mock_crawl4ai_class,
            patch(
                "src.services.crawling.manager.FirecrawlProvider"
            ) as mock_firecrawl_class,
        ):
            mock_crawl4ai_class.return_value = mock_crawl4ai_provider
            mock_firecrawl_class.return_value = mock_firecrawl_provider

            await crawl_manager.initialize()
            await crawl_manager.initialize()  # Second call

            # Should only initialize once
            mock_crawl4ai_provider.initialize.assert_called_once()
            mock_firecrawl_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test manager cleanup."""
        # Set up initialized state
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        await crawl_manager.cleanup()

        # Verify all providers were cleaned up
        mock_crawl4ai_provider.cleanup.assert_called_once()
        mock_firecrawl_provider.cleanup.assert_called_once()

        assert crawl_manager._initialized is False
        assert len(crawl_manager.providers) == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_provider_error(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test cleanup when a provider raises an error."""
        # Set up initialized state
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        # Mock cleanup failure for one provider
        mock_crawl4ai_provider.cleanup.side_effect = Exception("Cleanup failed")

        await crawl_manager.cleanup()

        # Should still clean up other providers and reset state
        mock_crawl4ai_provider.cleanup.assert_called_once()
        mock_firecrawl_provider.cleanup.assert_called_once()
        assert crawl_manager._initialized is False
        assert len(crawl_manager.providers) == 0


class TestSingleURLScraping:
    """Test single URL scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self, crawl_manager):
        """Test scraping when manager not initialized."""
        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await crawl_manager.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success_default_provider(
        self, crawl_manager, mock_crawl4ai_provider, mock_config
    ):
        """Test successful URL scraping with default provider."""
        # Set up initialized state
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}
        mock_config.crawl_provider = "crawl4ai"

        result = await crawl_manager.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["content"] == "Scraped content from Crawl4AI"
        assert result["provider"] == "crawl4ai"
        mock_crawl4ai_provider.scrape_url.assert_called_once_with(
            "https://example.com", None
        )

    @pytest.mark.asyncio
    async def test_scrape_url_with_formats(self, crawl_manager, mock_crawl4ai_provider):
        """Test URL scraping with specific formats."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        formats = ["markdown", "html"]
        await crawl_manager.scrape_url("https://example.com", formats=formats)

        mock_crawl4ai_provider.scrape_url.assert_called_once_with(
            "https://example.com", formats
        )

    @pytest.mark.asyncio
    async def test_scrape_url_preferred_provider(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test URL scraping with preferred provider."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        result = await crawl_manager.scrape_url(
            "https://example.com", preferred_provider="firecrawl"
        )

        assert result["provider"] == "firecrawl"
        mock_firecrawl_provider.scrape_url.assert_called_once()
        mock_crawl4ai_provider.scrape_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_preferred_provider_not_available(
        self, crawl_manager, mock_crawl4ai_provider
    ):
        """Test URL scraping when preferred provider not available."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        result = await crawl_manager.scrape_url(
            "https://example.com", preferred_provider="nonexistent"
        )

        # Should fall back to available provider
        assert result["provider"] == "crawl4ai"
        mock_crawl4ai_provider.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_provider_fallback(
        self,
        crawl_manager,
        mock_crawl4ai_provider,
        mock_firecrawl_provider,
        mock_config,
    ):
        """Test URL scraping with provider fallback."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }
        mock_config.crawl_provider = "crawl4ai"

        # Mock first provider failure
        mock_crawl4ai_provider.scrape_url.return_value = {
            "success": False,
            "error": "Provider failed",
        }

        result = await crawl_manager.scrape_url("https://example.com")

        # Should try both providers
        mock_crawl4ai_provider.scrape_url.assert_called_once()
        mock_firecrawl_provider.scrape_url.assert_called_once()
        assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_scrape_url_provider_exception_fallback(
        self,
        crawl_manager,
        mock_crawl4ai_provider,
        mock_firecrawl_provider,
        mock_config,
    ):
        """Test URL scraping with provider exception and fallback."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }
        mock_config.crawl_provider = "crawl4ai"

        # Mock first provider exception
        mock_crawl4ai_provider.scrape_url.side_effect = Exception("Provider crashed")

        result = await crawl_manager.scrape_url("https://example.com")

        # Should try both providers
        mock_crawl4ai_provider.scrape_url.assert_called_once()
        mock_firecrawl_provider.scrape_url.assert_called_once()
        assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_scrape_url_all_providers_fail(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test URL scraping when all providers fail."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        # Mock all providers failing
        mock_crawl4ai_provider.scrape_url.side_effect = Exception("Crawl4AI failed")
        mock_firecrawl_provider.scrape_url.side_effect = Exception("Firecrawl failed")

        result = await crawl_manager.scrape_url("https://example.com")

        assert result["success"] is False
        assert "All providers failed" in result["error"]
        assert result["content"] == ""
        assert result["url"] == "https://example.com"


class TestSiteCrawling:
    """Test site crawling functionality."""

    @pytest.mark.asyncio
    async def test_crawl_site_not_initialized(self, crawl_manager):
        """Test site crawling when manager not initialized."""
        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await crawl_manager.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_site_success_default_provider(
        self, crawl_manager, mock_crawl4ai_provider
    ):
        """Test successful site crawling with default provider."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        result = await crawl_manager.crawl_site("https://example.com", max_pages=10)

        assert result["success"] is True
        assert result["total"] == 2
        assert len(result["pages"]) == 2
        assert result["provider"] == "crawl4ai"
        mock_crawl4ai_provider.crawl_site.assert_called_once_with(
            "https://example.com", 10, None
        )

    @pytest.mark.asyncio
    async def test_crawl_site_with_formats(self, crawl_manager, mock_crawl4ai_provider):
        """Test site crawling with specific formats."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        formats = ["markdown", "text"]
        await crawl_manager.crawl_site(
            "https://example.com", max_pages=5, formats=formats
        )

        mock_crawl4ai_provider.crawl_site.assert_called_once_with(
            "https://example.com", 5, formats
        )

    @pytest.mark.asyncio
    async def test_crawl_site_preferred_provider(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test site crawling with preferred provider."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        result = await crawl_manager.crawl_site(
            "https://example.com", preferred_provider="firecrawl"
        )

        assert result["provider"] == "firecrawl"
        mock_firecrawl_provider.crawl_site.assert_called_once()
        mock_crawl4ai_provider.crawl_site.assert_not_called()

    @pytest.mark.asyncio
    async def test_crawl_site_auto_prefer_crawl4ai(
        self,
        crawl_manager,
        mock_crawl4ai_provider,
        mock_firecrawl_provider,
        mock_config,
    ):
        """Test that site crawling auto-prefers Crawl4AI when available."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }
        mock_config.crawl_provider = "firecrawl"  # Config prefers Firecrawl

        result = await crawl_manager.crawl_site("https://example.com")

        # Should prefer Crawl4AI for site crawling despite config
        assert result["provider"] == "crawl4ai"
        mock_crawl4ai_provider.crawl_site.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_site_fallback_on_failure(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test site crawling with fallback when primary provider fails."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        # Mock primary provider failure
        mock_crawl4ai_provider.crawl_site.side_effect = Exception("Crawl failed")

        result = await crawl_manager.crawl_site("https://example.com")

        # Should try both providers
        mock_crawl4ai_provider.crawl_site.assert_called_once()
        mock_firecrawl_provider.crawl_site.assert_called_once()
        assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_crawl_site_all_providers_fail(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test site crawling when all providers fail."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        # Mock all providers failing
        mock_crawl4ai_provider.crawl_site.side_effect = Exception("Crawl4AI failed")
        mock_firecrawl_provider.crawl_site.side_effect = Exception("Firecrawl failed")

        result = await crawl_manager.crawl_site("https://example.com")

        assert result["success"] is False
        assert "Crawl4AI failed" in result["error"]
        assert result["pages"] == []
        assert result["total"] == 0
        assert result["provider"] == "crawl4ai"  # Primary provider name

    @pytest.mark.asyncio
    async def test_crawl_site_single_provider_no_fallback(
        self, crawl_manager, mock_crawl4ai_provider
    ):
        """Test site crawling with single provider (no fallback available)."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        # Mock provider failure
        mock_crawl4ai_provider.crawl_site.side_effect = Exception(
            "Only provider failed"
        )

        result = await crawl_manager.crawl_site("https://example.com")

        assert result["success"] is False
        assert result["pages"] == []
        assert result["total"] == 0


class TestURLMapping:
    """Test URL mapping functionality."""

    @pytest.mark.asyncio
    async def test_map_url_not_initialized(self, crawl_manager):
        """Test URL mapping when manager not initialized."""
        with pytest.raises(CrawlServiceError, match="Manager not initialized"):
            await crawl_manager.map_url("https://example.com")

    @pytest.mark.asyncio
    async def test_map_url_no_firecrawl_provider(
        self, crawl_manager, mock_crawl4ai_provider
    ):
        """Test URL mapping when Firecrawl provider not available."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        result = await crawl_manager.map_url("https://example.com")

        assert result["success"] is False
        assert "URL mapping requires Firecrawl provider" in result["error"]
        assert result["urls"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_map_url_success(self, crawl_manager, mock_firecrawl_provider):
        """Test successful URL mapping."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"firecrawl": mock_firecrawl_provider}

        result = await crawl_manager.map_url("https://example.com")

        assert result["success"] is True
        assert len(result["urls"]) == 2
        assert result["total"] == 2
        mock_firecrawl_provider.map_url.assert_called_once_with(
            "https://example.com", False
        )

    @pytest.mark.asyncio
    async def test_map_url_with_subdomains(
        self, crawl_manager, mock_firecrawl_provider
    ):
        """Test URL mapping with subdomains."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"firecrawl": mock_firecrawl_provider}

        await crawl_manager.map_url("https://example.com", include_subdomains=True)

        mock_firecrawl_provider.map_url.assert_called_once_with(
            "https://example.com", True
        )


class TestProviderInfo:
    """Test provider information functionality."""

    def test_get_provider_info_empty(self, crawl_manager):
        """Test getting provider info when no providers available."""
        info = crawl_manager.get_provider_info()
        assert info == {}

    def test_get_provider_info_crawl4ai_only(
        self, crawl_manager, mock_crawl4ai_provider, mock_config
    ):
        """Test getting provider info with only Crawl4AI."""
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}
        mock_config.crawl_provider = "crawl4ai"
        mock_config.firecrawl.api_key = None

        info = crawl_manager.get_provider_info()

        assert "crawl4ai" in info
        assert info["crawl4ai"]["type"] == "AsyncMock"  # Mock class name
        assert info["crawl4ai"]["available"] is True
        assert info["crawl4ai"]["is_preferred"] is True
        assert info["crawl4ai"]["has_api_key"] is False  # Not Firecrawl

    def test_get_provider_info_both_providers(
        self,
        crawl_manager,
        mock_crawl4ai_provider,
        mock_firecrawl_provider,
        mock_config,
    ):
        """Test getting provider info with both providers."""
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }
        mock_config.crawl_provider = "firecrawl"
        mock_config.firecrawl.api_key = "test-key"

        info = crawl_manager.get_provider_info()

        assert len(info) == 2
        assert info["crawl4ai"]["is_preferred"] is False
        assert info["firecrawl"]["is_preferred"] is True
        assert info["firecrawl"]["has_api_key"] is True


class TestProviderOrder:
    """Test provider order and selection logic."""

    @pytest.mark.asyncio
    async def test_provider_order_with_preferred_provider(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test provider order when preferred provider is specified."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        # Mock first provider failing so we can see the order
        mock_firecrawl_provider.scrape_url.return_value = {"success": False}

        await crawl_manager.scrape_url(
            "https://example.com", preferred_provider="firecrawl"
        )

        # Should try firecrawl first, then crawl4ai
        mock_firecrawl_provider.scrape_url.assert_called_once()
        mock_crawl4ai_provider.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_order_with_config_preference(
        self,
        crawl_manager,
        mock_crawl4ai_provider,
        mock_firecrawl_provider,
        mock_config,
    ):
        """Test provider order using config preference."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }
        mock_config.crawl_provider = "firecrawl"

        # Mock first provider failing
        mock_firecrawl_provider.scrape_url.return_value = {"success": False}

        await crawl_manager.scrape_url("https://example.com")

        # Should try config preference first
        mock_firecrawl_provider.scrape_url.assert_called_once()
        mock_crawl4ai_provider.scrape_url.assert_called_once()


class TestConcurrencyAndRateLimiting:
    """Test concurrency control and rate limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_scraping(
        self, crawl_manager, mock_crawl4ai_provider, mock_rate_limiter
    ):
        """Test concurrent URL scraping."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]

        # Create concurrent scraping tasks
        tasks = [crawl_manager.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 3
        assert all(result["success"] for result in results)
        assert mock_crawl4ai_provider.scrape_url.call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(
        self, crawl_manager, mock_crawl4ai_provider, mock_rate_limiter
    ):
        """Test rate limiter integration."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        # Mock provider to use rate limiter
        async def mock_scrape_with_rate_limit(url, formats=None):
            await mock_rate_limiter.acquire()
            try:
                return {
                    "success": True,
                    "content": f"Content for {url}",
                    "url": url,
                }
            finally:
                mock_rate_limiter.release()

        mock_crawl4ai_provider.scrape_url.side_effect = mock_scrape_with_rate_limit

        await crawl_manager.scrape_url("https://example.com")

        # Rate limiter should have been used
        mock_rate_limiter.acquire.assert_called()
        mock_rate_limiter.release.assert_called()


class TestErrorHandling:
    """Test comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_provider_configuration_errors(self, crawl_manager, mock_config):
        """Test handling of provider configuration errors."""
        # Mock configuration that would cause provider init to fail
        mock_config.performance.default_rate_limits = {}  # Missing rate limits

        with patch(
            "src.services.crawling.manager.Crawl4AIProvider"
        ) as mock_crawl4ai_class:
            mock_crawl4ai_class.side_effect = ValueError("Invalid config")

            # Should handle the error gracefully if Firecrawl is available
            with patch(
                "src.services.crawling.manager.FirecrawlProvider"
            ) as mock_firecrawl_class:
                mock_firecrawl_provider = AsyncMock()
                mock_firecrawl_provider.initialize = AsyncMock()
                mock_firecrawl_class.return_value = mock_firecrawl_provider

                await crawl_manager.initialize()

                # Should initialize successfully with just Firecrawl
                assert crawl_manager._initialized is True
                assert len(crawl_manager.providers) == 1

    @pytest.mark.asyncio
    async def test_partial_provider_failure_during_operation(
        self, crawl_manager, mock_crawl4ai_provider, mock_firecrawl_provider
    ):
        """Test handling when providers fail during operation."""
        crawl_manager._initialized = True
        crawl_manager.providers = {
            "crawl4ai": mock_crawl4ai_provider,
            "firecrawl": mock_firecrawl_provider,
        }

        # Simulate provider becoming unavailable during operation
        mock_crawl4ai_provider.scrape_url.side_effect = ConnectionError(
            "Provider disconnected"
        )

        result = await crawl_manager.scrape_url("https://example.com")

        # Should fall back to working provider
        assert result["success"] is True
        assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_network_timeout_handling(
        self, crawl_manager, mock_crawl4ai_provider
    ):
        """Test handling of network timeouts."""
        crawl_manager._initialized = True
        crawl_manager.providers = {"crawl4ai": mock_crawl4ai_provider}

        # Mock timeout error
        mock_crawl4ai_provider.scrape_url.side_effect = TimeoutError(
            "Request timed out"
        )

        result = await crawl_manager.scrape_url("https://example.com")

        assert result["success"] is False
        assert "All providers failed" in result["error"]
        assert "Request timed out" in result["error"]
