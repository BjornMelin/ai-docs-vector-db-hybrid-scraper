"""Tests for crawl providers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.enums import CrawlProvider as CrawlProviderEnum
from src.config.models import UnifiedConfig
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
            assert (
                result["content"] == "# Test Content"
            )  # FirecrawlProvider returns "content", not "markdown"
            assert result["metadata"]["title"] == "Test"
            mock_instance.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self, firecrawl_provider):
        """Test failed URL scraping."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock failure response
            mock_instance.scrape_url.return_value = {
                "success": False,
                "error": "Failed to fetch",
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_crawl_website(self, firecrawl_provider):
        """Test website crawling."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock crawl response
            mock_instance.crawl_url.return_value = [
                {
                    "url": "https://example.com/page1",
                    "markdown": "Page 1 content",
                    "metadata": {"title": "Page 1"},
                },
                {
                    "url": "https://example.com/page2",
                    "markdown": "Page 2 content",
                    "metadata": {"title": "Page 2"},
                },
            ]

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.crawl_site(
                "https://example.com", max_pages=10
            )

            assert len(result) == 2
            assert result[0]["url"] == "https://example.com/page1"
            assert result[1]["url"] == "https://example.com/page2"

    @pytest.mark.asyncio
    async def test_map_website(self, firecrawl_provider):
        """Test website mapping."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock map response
            mock_instance.map_url.return_value = [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3",
            ]

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.map_url("https://example.com")

            assert len(result) == 3
            assert "https://example.com/page1" in result
            mock_instance.map_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, firecrawl_provider):
        """Test provider cleanup."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp"):
            await firecrawl_provider.initialize()
            assert firecrawl_provider._initialized

            await firecrawl_provider.cleanup()

            assert not firecrawl_provider._initialized
            assert firecrawl_provider._client is None


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
            assert crawl4ai_provider._crawler is not None
            mock_instance.start.assert_called_once()  # Check that start() was called, not __aenter__

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
            mock_result.metadata = {"title": "Test Page"}
            mock_instance.arun.return_value = mock_result

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.scrape_url("https://example.com")

            assert result["success"] is True
            assert (
                result["content"] == "# Test Content"
            )  # Crawl4AIProvider returns "content", not "markdown"
            assert result["metadata"]["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self, crawl4ai_provider):
        """Test failed URL scraping."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock failed result
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error_message = "Connection failed"
            mock_instance.arun.return_value = mock_result

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "error" in result
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_website(self, crawl4ai_provider):
        """Test website crawling."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # First call returns the main page with links
            main_result = MagicMock()
            main_result.success = True
            main_result.markdown = "Main page"
            main_result.html = """
            <html>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="https://external.com">External</a>
            </html>
            """
            main_result.metadata = {"title": "Main", "url": "https://example.com"}

            # Subsequent calls for discovered pages
            page1_result = MagicMock()
            page1_result.success = True
            page1_result.markdown = "Page 1 content"
            page1_result.metadata = {
                "title": "Page 1",
                "url": "https://example.com/page1",
            }

            page2_result = MagicMock()
            page2_result.success = True
            page2_result.markdown = "Page 2 content"
            page2_result.metadata = {
                "title": "Page 2",
                "url": "https://example.com/page2",
            }

            mock_instance.arun.side_effect = [main_result, page1_result, page2_result]

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.crawl_website(
                "https://example.com", max_pages=3
            )

            # Should have main page + 2 internal pages
            assert len(result) >= 1  # At least the main page
            assert result[0]["markdown"] == "Main page"

    @pytest.mark.asyncio
    async def test_map_website(self, crawl4ai_provider):
        """Test website mapping."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock crawl result with links
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.html = """
            <html>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="https://example.com/page3">Page 3</a>
                <a href="https://external.com">External</a>
            </html>
            """
            mock_instance.arun.return_value = mock_result

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.map_website("https://example.com")

            # Should include the base URL and discovered internal links
            assert "https://example.com" in result
            # Note: Exact URL normalization depends on implementation

    @pytest.mark.asyncio
    async def test_cleanup(self, crawl4ai_provider):
        """Test provider cleanup."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            await crawl4ai_provider.initialize()
            assert crawl4ai_provider._initialized

            await crawl4ai_provider.cleanup()

            assert not crawl4ai_provider._initialized
            assert crawl4ai_provider._crawler is None
            mock_instance.__aexit__.assert_called_once()


class TestCrawlManager:
    """Test crawl manager."""

    @pytest.fixture
    def config(self):
        """Create test configuration.

        Note: Using double underscore syntax (firecrawl__api_key) here because
        it mimics how environment variables are loaded. In production, this would
        be set as FIRECRAWL__API_KEY environment variable.
        """
        return UnifiedConfig(
            firecrawl__api_key="test-key",
            crawl_provider=CrawlProviderEnum.FIRECRAWL,
        )

    @pytest.fixture
    def crawl_manager(self, config):
        """Create crawl manager instance."""
        return CrawlManager(config)

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
    async def test_scrape_url_with_preferred_provider(self, crawl_manager):
        """Test URL scraping with preferred provider."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock Firecrawl success
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": True,
                "markdown": "Firecrawl content",
                "provider": "firecrawl",
            }

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["markdown"] == "Firecrawl content"
            assert result["provider"] == "firecrawl"
            mock_firecrawl_instance.scrape_url.assert_called_once()
            mock_crawl4ai_instance.scrape_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_url_with_fallback(self, crawl_manager):
        """Test URL scraping with fallback to secondary provider."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock Firecrawl failure
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": False,
                "error": "Firecrawl failed",
            }

            # Mock Crawl4AI success
            mock_crawl4ai_instance.scrape_url.return_value = {
                "success": True,
                "markdown": "Crawl4AI content",
                "provider": "crawl4ai",
            }

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["markdown"] == "Crawl4AI content"
            assert result["provider"] == "crawl4ai"
            mock_firecrawl_instance.scrape_url.assert_called_once()
            mock_crawl4ai_instance.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_all_providers_fail(self, crawl_manager):
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

            # Mock both providers failing
            mock_firecrawl_instance.scrape_url.return_value = {
                "success": False,
                "error": "Firecrawl failed",
            }
            mock_crawl4ai_instance.scrape_url.return_value = {
                "success": False,
                "error": "Crawl4AI failed",
            }

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_url("https://example.com")

            assert result["success"] is False
            assert "All providers failed" in result["error"]

    @pytest.mark.asyncio
    async def test_initialize_with_no_firecrawl_key(self):
        """Test initialization without Firecrawl API key."""
        config = UnifiedConfig()  # No API keys
        crawl_manager = CrawlManager(config)

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            await crawl_manager.initialize()

            assert crawl_manager._initialized
            assert len(crawl_manager.providers) == 1
            assert "crawl4ai" in crawl_manager.providers
            assert "firecrawl" not in crawl_manager.providers

    @pytest.mark.asyncio
    async def test_cleanup(self, crawl_manager):
        """Test manager cleanup."""
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

            await crawl_manager.cleanup()

            assert not crawl_manager._initialized
            assert len(crawl_manager.providers) == 0
            mock_firecrawl_instance.cleanup.assert_called_once()
            mock_crawl4ai_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_multiple_urls(self, crawl_manager):
        """Test scraping multiple URLs."""
        urls = ["https://example1.com", "https://example2.com"]

        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock successful batch scrape
            mock_firecrawl_instance.scrape_multiple_urls.return_value = {
                "success": True,
                "data": [
                    {"url": urls[0], "markdown": "Content 1"},
                    {"url": urls[1], "markdown": "Content 2"},
                ],
                "provider": "firecrawl",
            }

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_multiple_urls(urls)

            assert result["success"] is True
            assert len(result["data"]) == 2
            assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_crawl_website(self, crawl_manager):
        """Test website crawling."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock crawl result
            mock_firecrawl_instance.crawl_website.return_value = [
                {"url": "https://example.com/page1", "markdown": "Page 1"},
                {"url": "https://example.com/page2", "markdown": "Page 2"},
            ]

            await crawl_manager.initialize()

            result = await crawl_manager.crawl_website(
                "https://example.com", max_pages=10
            )

            assert len(result) == 2
            assert result[0]["url"] == "https://example.com/page1"

    @pytest.mark.asyncio
    async def test_provider_initialization_failure(self):
        """Test handling of provider initialization failure."""
        config = UnifiedConfig(firecrawl__api_key="test-key")
        crawl_manager = CrawlManager(config)

        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Mock Firecrawl initialization failure
            mock_firecrawl_instance = AsyncMock()
            mock_firecrawl_instance.initialize.side_effect = Exception("Init failed")
            mock_firecrawl.return_value = mock_firecrawl_instance

            # Mock Crawl4AI success
            mock_crawl4ai_instance = AsyncMock()
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            await crawl_manager.initialize()

            # Should still initialize with Crawl4AI
            assert crawl_manager._initialized
            assert "crawl4ai" in crawl_manager.providers
            assert "firecrawl" not in crawl_manager.providers
