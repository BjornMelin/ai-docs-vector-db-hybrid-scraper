"""Tests for Crawl4AI provider implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import Crawl4AIConfig
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.errors import CrawlServiceError


class TestCrawl4AIConfig:
    """Tests for Crawl4AIConfig validation and defaults."""

    def test_default_configuration(self):
        """Test default Crawl4AI configuration values."""

        config = Crawl4AIConfig()

        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.max_concurrent_crawls == 10
        assert config.page_timeout == 30.0
        assert config.remove_scripts is True
        assert config.remove_styles is True

    def test_custom_configuration(self):
        """Test custom Crawl4AI configuration values."""

        config = Crawl4AIConfig(
            browser_type="firefox",
            headless=False,
            max_concurrent_crawls=20,
            page_timeout=60.0,
            remove_scripts=False,
            remove_styles=False,
        )

        assert config.browser_type == "firefox"
        assert config.headless is False
        assert config.max_concurrent_crawls == 20
        assert config.page_timeout == 60.0
        assert config.remove_scripts is False
        assert config.remove_styles is False

    def test_config_validation_max_concurrent(self):
        """Test configuration validation for max_concurrent_crawls."""

        # Should accept valid values
        config = Crawl4AIConfig(max_concurrent_crawls=1)
        assert config.max_concurrent_crawls == 1

        config = Crawl4AIConfig(max_concurrent_crawls=50)
        assert config.max_concurrent_crawls == 50

    def test_config_validation_timeout(self):
        """Test configuration validation for page_timeout."""

        # Should accept valid positive timeout
        config = Crawl4AIConfig(page_timeout=15.5)
        assert config.page_timeout == 15.5


class TestCrawl4AIProviderInitialization:
    """Tests for provider initialization and cleanup."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""

        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            max_concurrent_crawls=5,
            page_timeout=30.0,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""

        return Crawl4AIProvider(config)

    def test_provider_initialization(self, provider, config):
        """Test provider initializes with correct configuration."""

        assert provider.config == config
        assert provider._initialized is False
        assert hasattr(provider, "js_executor")
        assert hasattr(provider, "doc_extractor")

    @pytest.mark.asyncio
    async def test_initialize_marks_ready(self, provider):
        """Test that initialize() marks provider as ready."""

        assert provider._initialized is False

        await provider.initialize()

        assert provider._initialized is True

    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self, provider):
        """Test that cleanup() resets provider state."""

        await provider.initialize()
        assert provider._initialized is True

        await provider.cleanup()

        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_operations_require_initialization(self, provider):
        """Test that operations fail when provider not initialized."""

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_bulk(["https://example.com"])

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_site("https://example.com")


class TestCrawl4AIProviderScraping:
    """Tests for single URL scraping functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""

        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            max_concurrent_crawls=5,
            page_timeout=30.0,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""

        return Crawl4AIProvider(config)

    @pytest.fixture
    def mock_crawl_result(self):
        """Create mock successful crawl result."""

        result = Mock()
        result.success = True
        result.markdown = "# Test Content\n\nThis is test content."
        result.html = "<h1>Test Content</h1><p>This is test content.</p>"
        result.metadata = {"title": "Test Page"}
        result.extracted_content = None
        result.links = ["https://example.com/link1", "https://example.com/link2"]
        result.media = {}
        result.url = "https://example.com"
        result.error_message = None
        return result

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, provider, mock_crawl_result):
        """Test successful URL scraping."""

        provider._initialized = True

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            # Return the raw results object (will be normalized by provider)
            mock_run_many.return_value = (mock_crawl_result, "direct")

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["url"] == "https://example.com"
            assert "content" in result
            assert result["content"] == "# Test Content\n\nThis is test content."

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self, provider):
        """Test URL scraping failure handling."""

        provider._initialized = True

        mock_fail_result = Mock()
        mock_fail_result.success = False
        mock_fail_result.error_message = "Connection timeout"
        mock_fail_result.url = "https://example.com"

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = (mock_fail_result, "direct")

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "error" in result
            assert "Connection timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_with_wait_for(self, provider, mock_crawl_result):
        """Test URL scraping with wait_for selector."""

        provider._initialized = True

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = (mock_crawl_result, "direct")

            result = await provider.scrape_url(
                "https://example.com",
                wait_for=".content-loaded",
            )

            assert result["success"] is True
            # Verify run config was prepared with wait_for
            mock_run_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_with_js_code(self, provider, mock_crawl_result):
        """Test URL scraping with JavaScript code."""

        provider._initialized = True

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = (mock_crawl_result, "direct")

            result = await provider.scrape_url(
                "https://example.com",
                js_code="console.log('test')",
            )

            assert result["success"] is True
            mock_run_many.assert_called_once()


class TestCrawl4AIProviderStreaming:
    """Tests for streaming scraping functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""

        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            max_concurrent_crawls=5,
            page_timeout=30.0,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""

        return Crawl4AIProvider(config)

    @pytest.mark.asyncio
    async def test_scrape_url_stream(self, provider):
        """Test streaming URL scraping."""

        provider._initialized = True

        # Create mock streaming results
        mock_result1 = Mock()
        mock_result1.success = False  # Intermediate chunk
        mock_result1.markdown = "Partial..."

        mock_result2 = Mock()
        mock_result2.success = True
        mock_result2.markdown = "# Complete Content"
        mock_result2.html = "<h1>Complete Content</h1>"
        mock_result2.metadata = {"title": "Test"}
        mock_result2.url = "https://example.com"

        async def mock_async_gen():
            """Mock async generator for streaming."""
            yield mock_result1
            yield mock_result2

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = (mock_async_gen(), "direct")

            results = []
            async for chunk in provider.scrape_url_stream("https://example.com"):
                results.append(chunk)

            assert len(results) == 2
            assert results[1]["success"] is True


class TestCrawl4AIProviderBulkCrawling:
    """Tests for bulk crawling functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""

        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            max_concurrent_crawls=10,
            page_timeout=30.0,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""

        return Crawl4AIProvider(config)

    @pytest.mark.asyncio
    async def test_crawl_bulk_success(self, provider):
        """Test successful bulk crawling."""

        provider._initialized = True

        # Create mock results for multiple URLs (returned as tuple)
        mock_results = []
        for i in range(3):
            result = Mock()
            result.success = True
            result.markdown = f"# Content {i}"
            result.html = f"<h1>Content {i}</h1>"
            result.metadata = {"title": f"Page {i}"}
            result.url = f"https://example.com/page{i}"
            result.extracted_content = None
            result.links = []
            result.media = {}
            mock_results.append(result)

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            # Return tuple of results (will be normalized)
            mock_run_many.return_value = (tuple(mock_results), "direct")

            urls = [f"https://example.com/page{i}" for i in range(3)]
            results = await provider.crawl_bulk(urls)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["success"] is True
                assert result["url"] == f"https://example.com/page{i}"

    @pytest.mark.asyncio
    async def test_crawl_bulk_mixed_results(self, provider):
        """Test bulk crawling with mixed success/failure results."""

        provider._initialized = True

        mock_result1 = Mock()
        mock_result1.success = True
        mock_result1.markdown = "# Success"
        mock_result1.html = "<h1>Success</h1>"
        mock_result1.metadata = {"title": "Success"}
        mock_result1.url = "https://example.com/success"
        mock_result1.extracted_content = None
        mock_result1.links = []
        mock_result1.media = {}

        mock_result2 = Mock()
        mock_result2.success = False
        mock_result2.error_message = "Failed to load"
        mock_result2.url = "https://example.com/fail"

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = ((mock_result1, mock_result2), "direct")

            urls = ["https://example.com/success", "https://example.com/fail"]
            results = await provider.crawl_bulk(urls)

            # Only successful results are returned
            assert len(results) == 1
            assert results[0]["success"] is True


class TestCrawl4AIProviderSiteCrawling:
    """Tests for site crawling functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""

        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            max_concurrent_crawls=10,
            page_timeout=30.0,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""

        return Crawl4AIProvider(config)

    @pytest.mark.asyncio
    async def test_crawl_site_basic(self, provider):
        """Test basic site crawling."""

        provider._initialized = True

        # Mock a single page result
        mock_result = Mock()
        mock_result.success = True
        mock_result.markdown = "# Homepage"
        mock_result.html = "<h1>Homepage</h1>"
        mock_result.metadata = {"title": "Home"}
        mock_result.url = "https://example.com"
        mock_result.extracted_content = None
        mock_result.links = []
        mock_result.media = {}

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = ((mock_result,), "direct")

            result = await provider.crawl_site("https://example.com", max_pages=1)

            assert result["success"] is True
            assert result["total"] >= 1
            assert "pages" in result

    @pytest.mark.asyncio
    async def test_crawl_site_respects_max_pages(self, provider):
        """Test that site crawling respects max_pages limit."""

        provider._initialized = True

        # Create results with links to more pages
        def create_result(url, links):
            """Helper to create mock result."""
            result = Mock()
            result.success = True
            result.markdown = "# Content"
            result.html = "<h1>Content</h1>"
            result.metadata = {"title": "Page"}
            result.url = url
            result.extracted_content = None
            result.links = links
            result.media = {}
            return result

        mock_results = [
            create_result(
                "https://example.com",
                [
                    {"href": "https://example.com/page1"},
                    {"href": "https://example.com/page2"},
                ],
            )
        ]

        with patch.object(
            provider, "_run_many", new_callable=AsyncMock
        ) as mock_run_many:
            mock_run_many.return_value = (tuple(mock_results), "direct")

            result = await provider.crawl_site("https://example.com", max_pages=1)

            assert result["success"] is True
            # Should not exceed max_pages
            assert result["total"] <= 1


class TestCrawl4AIProviderErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""

        return Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            max_concurrent_crawls=5,
            page_timeout=30.0,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""

        return Crawl4AIProvider(config)

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, provider):
        """Test error when scraping without initialization."""

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_bulk_crawl_not_initialized(self, provider):
        """Test error when bulk crawling without initialization."""

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_bulk(["https://example.com"])

    @pytest.mark.asyncio
    async def test_site_crawl_not_initialized(self, provider):
        """Test error when site crawling without initialization."""

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_streaming_not_initialized(self, provider):
        """Test error when streaming without initialization."""

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            async for _ in provider.scrape_url_stream("https://example.com"):
                pass


class TestCrawl4AIProviderConfiguration:
    """Tests for configuration handling."""

    def test_provider_stores_config(self):
        """Test that provider correctly stores configuration."""

        config = Crawl4AIConfig(
            browser_type="firefox",
            headless=False,
            max_concurrent_crawls=15,
            page_timeout=45.0,
        )
        provider = Crawl4AIProvider(config)

        assert provider.config == config
        assert provider.config.browser_type == "firefox"
        assert provider.config.headless is False
        assert provider.config.max_concurrent_crawls == 15
        assert provider.config.page_timeout == 45.0

    def test_provider_initializes_executors(self):
        """Test that provider initializes JavaScript and documentation extractors."""

        config = Crawl4AIConfig()
        provider = Crawl4AIProvider(config)

        assert provider.js_executor is not None
        assert provider.doc_extractor is not None
