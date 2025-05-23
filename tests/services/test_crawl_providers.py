"""Tests for crawl providers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.config import APIConfig
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.crawling.firecrawl_provider import FirecrawlProvider
from src.services.crawling.manager import CrawlManager


class TestFirecrawlProvider:
    """Test Firecrawl provider."""

    @pytest.fixture
    def firecrawl_provider(self):
        """Create Firecrawl provider instance."""
        return FirecrawlProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_initialize(self, firecrawl_provider):
        """Test provider initialization."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            await firecrawl_provider.initialize()

            assert firecrawl_provider._initialized
            mock_app.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, firecrawl_provider):
        """Test successful URL scraping."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock successful response
            mock_instance.scrape_url.return_value = {
                "success": True,
                "markdown": "# Test Content",
                "html": "<h1>Test Content</h1>",
                "metadata": {"title": "Test"},
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "# Test Content"
            assert result["html"] == "<h1>Test Content</h1>"
            assert result["metadata"]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self, firecrawl_provider):
        """Test failed URL scraping."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock failed response
            mock_instance.scrape_url.return_value = {
                "success": False,
                "error": "Failed to scrape",
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert result["error"] == "Failed to scrape"

    @pytest.mark.asyncio
    async def test_crawl_site(self, firecrawl_provider):
        """Test site crawling."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock crawl start
            mock_instance.async_crawl_url.return_value = {"id": "crawl-123"}

            # Mock crawl status - completed
            mock_instance.check_crawl_status.return_value = {
                "status": "completed",
                "data": [
                    {
                        "url": "https://example.com/page1",
                        "markdown": "Page 1",
                        "html": "<p>Page 1</p>",
                        "metadata": {},
                    },
                    {
                        "url": "https://example.com/page2",
                        "markdown": "Page 2",
                        "html": "<p>Page 2</p>",
                        "metadata": {},
                    },
                ],
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.crawl_site(
                "https://example.com",
                max_pages=10,
            )

            assert result["success"] is True
            assert result["total"] == 2
            assert len(result["pages"]) == 2
            assert result["crawl_id"] == "crawl-123"

    @pytest.mark.asyncio
    async def test_map_url(self, firecrawl_provider):
        """Test URL mapping."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock map response
            mock_instance.map_url.return_value = {
                "success": True,
                "links": [
                    "https://example.com/page1",
                    "https://example.com/page2",
                ],
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.map_url("https://example.com")

            assert result["success"] is True
            assert result["total"] == 2
            assert len(result["urls"]) == 2


class TestCrawl4AIProvider:
    """Test Crawl4AI provider."""

    @pytest.fixture
    def crawl4ai_provider(self):
        """Create Crawl4AI provider instance."""
        return Crawl4AIProvider()

    @pytest.mark.asyncio
    async def test_initialize(self, crawl4ai_provider):
        """Test provider initialization."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            await crawl4ai_provider.initialize()

            assert crawl4ai_provider._initialized
            mock_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, crawl4ai_provider):
        """Test successful URL scraping."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock crawl result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.markdown = "# Test Content"
            mock_result.html = "<h1>Test Content</h1>"
            mock_result.title = "Test Page"
            mock_instance.crawl.return_value = mock_result

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "# Test Content"
            assert result["metadata"]["title"] == "Test Page"


class TestCrawlManager:
    """Test crawl manager."""

    @pytest.fixture
    def api_config(self):
        """Create test API config."""
        return APIConfig(
            firecrawl_api_key="test-key",
            preferred_crawl_provider="firecrawl",
        )

    @pytest.fixture
    def crawl_manager(self, api_config):
        """Create crawl manager instance."""
        return CrawlManager(api_config)

    @pytest.mark.asyncio
    async def test_initialize(self, crawl_manager):
        """Test manager initialization."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            await crawl_manager.initialize()

            assert crawl_manager._initialized
            assert len(crawl_manager.providers) == 2
            assert "firecrawl" in crawl_manager.providers
            assert "crawl4ai" in crawl_manager.providers

    @pytest.mark.asyncio
    async def test_scrape_url_with_fallback(self, crawl_manager):
        """Test URL scraping with fallback."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # First provider fails
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": False,
                "error": "API error",
            }

            # Second provider succeeds
            mock_crawl4ai_instance.scrape_url.return_value = {
                "success": True,
                "content": "Fallback content",
            }

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "Fallback content"
            assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_url_all_fail(self, crawl_manager):
        """Test URL scraping when all providers fail."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Both providers fail
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": False,
                "error": "API error",
            }
            mock_crawl4ai_instance.scrape_url.return_value = {
                "success": False,
                "error": "Crawl failed",
            }

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_url("https://example.com")

            assert result["success"] is False
            assert "All providers failed" in result["error"]

    @pytest.mark.asyncio
    async def test_get_provider_info(self, crawl_manager):
        """Test getting provider information."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            await crawl_manager.initialize()

            info = crawl_manager.get_provider_info()

            assert "firecrawl" in info
            assert "crawl4ai" in info
            assert info["firecrawl"]["is_preferred"] is True
            assert info["firecrawl"]["has_api_key"] is True

    @pytest.mark.asyncio
    async def test_map_url_requires_firecrawl(self, crawl_manager):
        """Test URL mapping requires Firecrawl."""
        # Create manager without Firecrawl
        config = APIConfig(firecrawl_api_key=None)
        manager = CrawlManager(config)

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            await manager.initialize()

            result = await manager.map_url("https://example.com")

            assert result["success"] is False
            assert "requires Firecrawl" in result["error"]
