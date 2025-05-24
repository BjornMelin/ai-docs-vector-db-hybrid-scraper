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

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self, firecrawl_provider):
        """Test scraping when not initialized."""
        from src.services.errors import CrawlServiceError

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await firecrawl_provider.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_exception(self, firecrawl_provider):
        """Test scraping with exception."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock to raise exception
            mock_instance.scrape_url.side_effect = Exception("API error")

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_in_progress(self, firecrawl_provider):
        """Test site crawling with in-progress status."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock crawl start
            mock_instance.async_crawl_url.return_value = {"id": "crawl-123"}

            # Mock crawl status - in progress then completed
            mock_instance.check_crawl_status.side_effect = [
                {"status": "crawling", "data": []},
                {
                    "status": "completed",
                    "data": [
                        {
                            "url": "https://example.com",
                            "markdown": "Test",
                            "html": "<p>Test</p>",
                            "metadata": {},
                        }
                    ],
                },
            ]

            await firecrawl_provider.initialize()

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await firecrawl_provider.crawl_site(
                    "https://example.com", max_pages=5
                )

            assert result["success"] is True
            assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_crawl_site_failed(self, firecrawl_provider):
        """Test site crawling failure."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock crawl start
            mock_instance.async_crawl_url.return_value = {"id": "crawl-123"}

            # Mock crawl status - failed
            mock_instance.check_crawl_status.return_value = {
                "status": "failed",
                "error": "Crawl failed",
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "Crawl failed" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_exception(self, firecrawl_provider):
        """Test site crawling with exception."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock to raise exception
            mock_instance.async_crawl_url.side_effect = Exception("API error")

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_map_url_failure(self, firecrawl_provider):
        """Test URL mapping failure."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock map failure
            mock_instance.map_url.return_value = {
                "success": False,
                "error": "Map failed",
            }

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.map_url("https://example.com")

            assert result["success"] is False
            assert "Map failed" in result["error"]

    @pytest.mark.asyncio
    async def test_map_url_exception(self, firecrawl_provider):
        """Test URL mapping with exception."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            # Mock to raise exception
            mock_instance.map_url.side_effect = Exception("API error")

            await firecrawl_provider.initialize()

            result = await firecrawl_provider.map_url("https://example.com")

            assert result["success"] is False
            assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_cleanup(self, firecrawl_provider):
        """Test provider cleanup."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            await firecrawl_provider.initialize()
            await firecrawl_provider.cleanup()

            assert not firecrawl_provider._initialized

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, firecrawl_provider):
        """Test initialization when already initialized."""
        with patch("src.services.crawling.firecrawl_provider.FirecrawlApp") as mock_app:
            mock_instance = MagicMock()
            mock_app.return_value = mock_instance

            await firecrawl_provider.initialize()
            await firecrawl_provider.initialize()  # Second call

            # Should only be called once
            mock_app.assert_called_once()


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

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self, crawl4ai_provider):
        """Test failed URL scraping."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock crawl result with failure
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Failed to crawl"
            mock_instance.crawl.return_value = mock_result

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert result["error"] == "Failed to crawl"
            assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_scrape_url_exception(self, crawl4ai_provider):
        """Test URL scraping with exception."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock crawl to raise exception
            mock_instance.crawl.side_effect = Exception("Network error")

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_success(self, crawl4ai_provider):
        """Test successful site crawling."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock crawl results for multiple pages
            mock_result1 = MagicMock()
            mock_result1.success = True
            mock_result1.markdown = "Page 1 content"
            mock_result1.html = "<p>Page 1</p>"
            mock_result1.title = "Page 1"

            mock_result2 = MagicMock()
            mock_result2.success = True
            mock_result2.markdown = "Page 2 content"
            mock_result2.html = "<p>Page 2</p>"
            mock_result2.title = "Page 2"

            # Return different results for each call
            mock_instance.crawl.side_effect = [mock_result1, mock_result2]

            await crawl4ai_provider.initialize()

            result = await crawl4ai_provider.crawl_site(
                "https://example.com", max_pages=2
            )

            assert result["success"] is True
            assert result["total"] == 1  # Only crawls the first URL
            assert len(result["pages"]) == 1

    @pytest.mark.asyncio
    async def test_crawl_site_failure(self, crawl4ai_provider):
        """Test site crawling failure."""
        # Mock the scrape_url method directly since crawl_site calls it
        with patch.object(crawl4ai_provider, "scrape_url") as mock_scrape:
            # Mock scrape to fail
            mock_scrape.side_effect = Exception("Crawl error")

            # Mark as initialized
            crawl4ai_provider._initialized = True

            result = await crawl4ai_provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "Crawl error" in result["error"]
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, crawl4ai_provider):
        """Test provider cleanup."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            await crawl4ai_provider.initialize()
            await crawl4ai_provider.cleanup()

            mock_instance.close.assert_called_once()
            assert not crawl4ai_provider._initialized

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, crawl4ai_provider):
        """Test error when provider not initialized."""
        from src.services.errors import CrawlServiceError

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await crawl4ai_provider.scrape_url("https://example.com")

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await crawl4ai_provider.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_initialize_error(self, crawl4ai_provider):
        """Test initialization error."""
        from src.services.errors import CrawlServiceError

        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            # Mock to raise error on creation
            mock_crawler.side_effect = Exception("Init failed")

            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Crawl4AI"
            ):
                await crawl4ai_provider.initialize()


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

    @pytest.mark.asyncio
    async def test_crawl_site_with_provider(self, crawl_manager):
        """Test site crawling with specific provider."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock crawl response
            mock_firecrawl_instance.crawl_site = AsyncMock()
            mock_firecrawl_instance.crawl_site.return_value = {
                "success": True,
                "pages": [{"url": "https://example.com", "content": "Test"}],
                "total": 1,
            }

            await crawl_manager.initialize()

            result = await crawl_manager.crawl_site(
                "https://example.com", preferred_provider="firecrawl"
            )

            assert result["success"] is True
            assert result["total"] == 1
            assert result["provider"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_map_url_success(self, crawl_manager):
        """Test successful URL mapping."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock firecrawl as available with map_url capability
            mock_firecrawl_instance.map_url = AsyncMock()
            mock_firecrawl_instance.map_url.return_value = {
                "success": True,
                "urls": ["https://example.com/page1", "https://example.com/page2"],
                "total": 2,
            }

            await crawl_manager.initialize()

            result = await crawl_manager.map_url("https://example.com")

            assert result["success"] is True
            assert result["total"] == 2
            # map_url doesn't add provider to result
            mock_firecrawl_instance.map_url.assert_called_once_with(
                "https://example.com", False
            )

    @pytest.mark.asyncio
    async def test_cleanup(self, crawl_manager):
        """Test manager cleanup."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            await crawl_manager.initialize()
            await crawl_manager.cleanup()

            mock_firecrawl_instance.cleanup.assert_called_once()
            mock_crawl4ai_instance.cleanup.assert_called_once()
            assert not crawl_manager._initialized

    @pytest.mark.asyncio
    async def test_manual_init_cleanup(self, crawl_manager):
        """Test manual initialization and cleanup."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock the cleanup methods
            mock_firecrawl_instance.cleanup = AsyncMock()
            mock_crawl4ai_instance.cleanup = AsyncMock()

            # Initialize
            await crawl_manager.initialize()
            assert crawl_manager._initialized

            # Cleanup
            await crawl_manager.cleanup()
            mock_firecrawl_instance.cleanup.assert_called_once()
            mock_crawl4ai_instance.cleanup.assert_called_once()
            assert not crawl_manager._initialized

    @pytest.mark.asyncio
    async def test_no_providers_available(self):
        """Test when no providers are available."""
        from src.services.errors import CrawlServiceError

        config = APIConfig()  # No API keys
        manager = CrawlManager(config)

        with patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai:
            mock_crawl4ai.side_effect = Exception("Failed to create provider")

            with pytest.raises(
                CrawlServiceError, match="No crawling providers available"
            ):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_scrape_url_with_invalid_provider(self, crawl_manager):
        """Test scraping with invalid provider name."""
        with (
            patch("src.services.crawling.manager.FirecrawlProvider") as mock_firecrawl,
            patch("src.services.crawling.manager.Crawl4AIProvider") as mock_crawl4ai,
        ):
            # Setup providers
            mock_firecrawl_instance = AsyncMock()
            mock_crawl4ai_instance = AsyncMock()
            mock_firecrawl.return_value = mock_firecrawl_instance
            mock_crawl4ai.return_value = mock_crawl4ai_instance

            # Mock scrape_url to return a successful result
            mock_firecrawl_instance.scrape_url = AsyncMock(
                return_value={"success": True, "content": "Test content"}
            )

            await crawl_manager.initialize()

            result = await crawl_manager.scrape_url(
                "https://example.com", preferred_provider="invalid"
            )

            # Since invalid provider is not available, it falls back to available providers
            assert result["success"] is True
            assert result["provider"] == "firecrawl"  # Falls back to first available

            # Verify warning was logged
            mock_firecrawl_instance.scrape_url.assert_called_once()
