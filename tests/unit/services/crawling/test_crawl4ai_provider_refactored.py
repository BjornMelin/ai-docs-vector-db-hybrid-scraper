"""Tests for refactored Crawl4AI provider implementation."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from src.config import Crawl4AIConfig
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.errors import CrawlServiceError


class TestCrawl4AIProvider:
    """Test the refactored Crawl4AI provider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            viewport={"width": 1920, "height": 1080},
            max_concurrent_crawls=5,
            page_timeout=30.0,
            enable_streaming=False,  # Disable streaming for tests
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        return Crawl4AIProvider(config)

    @pytest.fixture
    def mock_crawler_result(self):
        """Create mock crawler result."""
        result = Mock()
        result.success = True
        result.markdown = "# Test Content"
        result.html = "<h1>Test Content</h1>"
        result.metadata = {"title": "Test Page"}
        result.extracted_content = None
        result.links = []
        result.media = {}
        return result

    async def test_initialization(self, provider):
        """Test provider initialization."""
        assert not provider._initialized
        assert provider.config.browser_type == "chromium"
        assert provider.config.viewport["width"] == 1920

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_initialize_success(self, mock_crawler_class, provider):
        """Test successful initialization."""
        mock_crawler = AsyncMock()
        mock_crawler.start = AsyncMock()  # Mock the start method
        mock_crawler_class.return_value = mock_crawler

        await provider.initialize()

        assert provider._initialized
        assert provider._crawler is not None
        mock_crawler_class.assert_called_once_with(config=provider.browser_config)
        mock_crawler.start.assert_called_once()

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_cleanup(self, mock_crawler_class, provider):
        """Test cleanup after initialization."""
        mock_crawler = AsyncMock()
        mock_crawler.start = AsyncMock()
        mock_crawler.close = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        await provider.initialize()
        await provider.cleanup()

        assert not provider._initialized
        assert provider._crawler is None
        mock_crawler.close.assert_called_once()

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_scrape_url_success(
        self, mock_crawler_class, provider, mock_crawler_result
    ):
        """Test successful URL scraping."""
        mock_crawler = AsyncMock()
        mock_crawler.start = AsyncMock()
        mock_crawler.arun.return_value = mock_crawler_result
        mock_crawler_class.return_value = mock_crawler

        await provider.initialize()

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["content"] == "# Test Content"
        assert result["html"] == "<h1>Test Content</h1>"
        assert result["provider"] == "crawl4ai"
        mock_crawler.arun.assert_called_once()

    async def test_scrape_url_not_initialized(self, provider):
        """Test scraping without initialization."""
        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_scrape_url_failure(self, mock_crawler_class, provider):
        """Test URL scraping failure."""
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Page not found"
        mock_crawler.arun.return_value = mock_result
        mock_crawler_class.return_value = mock_crawler

        await provider.initialize()

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is False
        assert result["error"] == "Page not found"
        assert result["provider"] == "crawl4ai"

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_scrape_url_exception(self, mock_crawler_class, provider):
        """Test URL scraping with exception."""
        mock_crawler = AsyncMock()
        mock_crawler.arun.side_effect = Exception("Network error")
        mock_crawler_class.return_value = mock_crawler

        await provider.initialize()

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is False
        assert "Network error" in result["error"]
        assert result["provider"] == "crawl4ai"

    def test_create_extraction_strategy_structured(self, provider):
        """Test structured extraction strategy creation."""
        strategy = provider._create_extraction_strategy("structured")
        assert strategy is not None
        assert hasattr(strategy, "schema")

    def test_create_extraction_strategy_llm(self, provider):
        """Test LLM extraction strategy creation."""
        strategy = provider._create_extraction_strategy("llm")
        assert strategy is not None

    def test_create_extraction_strategy_markdown(self, provider):
        """Test markdown extraction strategy (returns None)."""
        strategy = provider._create_extraction_strategy("markdown")
        assert strategy is None

    def test_create_run_config(self, provider):
        """Test run configuration creation."""
        config = provider._create_run_config(
            wait_for=".content", js_code="console.log('test')", extraction_strategy=None
        )

        assert config.wait_for == ".content"
        assert config.js_code == "console.log('test')"
        assert config.cache_mode == "enabled"
        assert config.page_timeout == 30000  # 30 seconds in milliseconds

    def test_build_success_result(self, provider, mock_crawler_result):
        """Test success result building."""
        result = provider._build_success_result(
            "https://example.com", mock_crawler_result, "markdown"
        )

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["content"] == "# Test Content"
        assert result["metadata"]["extraction_type"] == "markdown"

    def test_build_error_result(self, provider):
        """Test error result building."""
        result = provider._build_error_result(
            "https://example.com", "Test error", "markdown"
        )

        assert result["success"] is False
        assert result["url"] == "https://example.com"
        assert result["error"] == "Test error"
        assert "error_context" in result
