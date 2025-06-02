"""Comprehensive tests for Crawl4AI provider with Pydantic configuration."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import Crawl4AIConfig
from src.services.base import BaseService
from src.services.crawling.base import CrawlProvider
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.crawling.crawl4ai_provider import CrawlBenchmark
from src.services.crawling.crawl4ai_provider import CrawlCache
from src.services.crawling.crawl4ai_provider import DocumentationExtractor
from src.services.crawling.crawl4ai_provider import JavaScriptExecutor
from src.services.errors import CrawlServiceError


@pytest.fixture
def basic_config():
    """Create basic Crawl4AI configuration."""
    return Crawl4AIConfig(
        browser_type="chromium",
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        max_concurrent_crawls=5,
        page_timeout=30.0,
    )


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    return MagicMock()


class TestJavaScriptExecutor:
    """Test JavaScript executor functionality."""

    def test_init(self):
        """Test JavaScriptExecutor initialization."""
        executor = JavaScriptExecutor()
        assert hasattr(executor, "common_patterns")
        assert "spa_navigation" in executor.common_patterns
        assert "infinite_scroll" in executor.common_patterns
        assert "click_show_more" in executor.common_patterns

    def test_get_js_for_site_python_docs(self):
        """Test JavaScript selection for Python docs."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://docs.python.org/3/")
        assert js == executor.common_patterns["spa_navigation"]

    def test_get_js_for_site_react_docs(self):
        """Test JavaScript selection for React docs."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://reactjs.org/docs")
        assert js == executor.common_patterns["spa_navigation"]

    def test_get_js_for_site_stackoverflow(self):
        """Test JavaScript selection for Stack Overflow."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://stackoverflow.com/questions")
        assert js == executor.common_patterns["infinite_scroll"]

    def test_get_js_for_site_mdn(self):
        """Test JavaScript selection for MDN."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://developer.mozilla.org/docs")
        assert js == executor.common_patterns["click_show_more"]

    def test_get_js_for_site_unknown(self):
        """Test JavaScript selection for unknown site."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://unknown-site.com")
        assert js is None


class TestDocumentationExtractor:
    """Test documentation extraction functionality."""

    def test_init(self):
        """Test DocumentationExtractor initialization."""
        extractor = DocumentationExtractor()
        assert hasattr(extractor, "selectors")
        assert "content" in extractor.selectors
        assert "code" in extractor.selectors
        assert "nav" in extractor.selectors
        assert "metadata" in extractor.selectors

    def test_create_extraction_schema_general(self):
        """Test schema creation for general documentation."""
        extractor = DocumentationExtractor()
        schema = extractor.create_extraction_schema("general")

        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

    def test_create_extraction_schema_api_reference(self):
        """Test schema creation for API reference."""
        extractor = DocumentationExtractor()
        schema = extractor.create_extraction_schema("api_reference")

        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema
        assert "endpoints" in schema
        assert "parameters" in schema
        assert "responses" in schema
        assert "examples" in schema

    def test_create_extraction_schema_tutorial(self):
        """Test schema creation for tutorials."""
        extractor = DocumentationExtractor()
        schema = extractor.create_extraction_schema("tutorial")

        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema
        assert "steps" in schema
        assert "code_examples" in schema
        assert "prerequisites" in schema
        assert "objectives" in schema

    def test_create_extraction_schema_guide(self):
        """Test schema creation for guides."""
        extractor = DocumentationExtractor()
        schema = extractor.create_extraction_schema("guide")

        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema
        assert "sections" in schema
        assert "callouts" in schema
        assert "related" in schema


class TestCrawl4AIProvider:
    """Test Crawl4AIProvider class with Pydantic configuration."""

    def test_init_basic(self, basic_config):
        """Test basic initialization with Pydantic config."""
        provider = Crawl4AIProvider(basic_config)

        assert isinstance(provider, BaseService)
        assert isinstance(provider, CrawlProvider)
        assert provider.config == basic_config
        assert provider._initialized is False
        assert provider._crawler is None
        assert provider.max_concurrent == 5

    def test_init_with_rate_limiter(self, basic_config, mock_rate_limiter):
        """Test initialization with custom rate limiter."""
        provider = Crawl4AIProvider(basic_config, mock_rate_limiter)
        assert provider.rate_limiter == mock_rate_limiter

    def test_browser_config_creation(self, basic_config):
        """Test browser configuration from Pydantic model."""
        provider = Crawl4AIProvider(basic_config)

        browser_config = provider.browser_config
        assert browser_config.browser_type == "chromium"
        assert browser_config.headless is True
        assert browser_config.viewport_width == 1920
        assert browser_config.viewport_height == 1080

    def test_semaphore_creation(self, basic_config):
        """Test semaphore creation based on config."""
        provider = Crawl4AIProvider(basic_config)
        assert provider.semaphore._value == 5  # max_concurrent_crawls

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_initialize_success(self, mock_crawler_class, basic_config):
        """Test successful provider initialization."""
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        assert provider._initialized is True
        assert provider._crawler == mock_crawler
        mock_crawler.start.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_initialize_already_initialized(self, mock_crawler_class, basic_config):
        """Test initialization when already initialized."""
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        provider._initialized = True

        await provider.initialize()

        # Should not create new crawler
        mock_crawler_class.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_initialize_failure(self, mock_crawler_class, basic_config):
        """Test initialization failure."""
        mock_crawler_class.side_effect = Exception("Crawler creation failed")

        provider = Crawl4AIProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Failed to initialize Crawl4AI"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_cleanup_success(self, basic_config):
        """Test successful cleanup."""
        provider = Crawl4AIProvider(basic_config)
        mock_crawler = AsyncMock()
        provider._crawler = mock_crawler
        provider._initialized = True

        await provider.cleanup()

        mock_crawler.close.assert_called_once()
        assert provider._crawler is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_no_crawler(self, basic_config):
        """Test cleanup when no crawler exists."""
        provider = Crawl4AIProvider(basic_config)

        # Should not raise exception
        await provider.cleanup()

        assert provider._crawler is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_crawler_error(self, basic_config):
        """Test cleanup when crawler close fails."""
        provider = Crawl4AIProvider(basic_config)
        mock_crawler = AsyncMock()
        mock_crawler.close.side_effect = Exception("Close failed")
        provider._crawler = mock_crawler
        provider._initialized = True

        # Should not raise exception even if close fails
        await provider.cleanup()

        # Should still reset state even if close fails
        assert provider._crawler is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self, basic_config):
        """Test scraping when provider not initialized."""
        provider = Crawl4AIProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_scrape_url_success(self, mock_crawler_class, basic_config):
        """Test successful URL scraping."""
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Content"
        mock_result.html = "<h1>Test Content</h1>"
        mock_result.metadata = {"title": "Test Page"}
        mock_result.extracted_content = {"key": "value"}
        mock_result.links = [{"href": "https://example.com/link"}]
        mock_result.media = {"images": []}

        mock_crawler.arun.return_value = mock_result
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["content"] == "# Test Content"
        assert result["html"] == "<h1>Test Content</h1>"
        assert result["title"] == "Test Page"
        assert result["metadata"]["title"] == "Test Page"
        assert result["structured_data"] == {"key": "value"}
        assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_scrape_url_failure(self, mock_crawler_class, basic_config):
        """Test URL scraping failure."""
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Crawl failed"

        mock_crawler.arun.return_value = mock_result
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is False
        assert result["error"] == "Crawl failed"
        assert result["content"] == ""
        assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_scrape_url_exception(self, mock_crawler_class, basic_config):
        """Test URL scraping exception handling."""
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_crawler.arun.side_effect = Exception("Network error")
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is False
        assert "Network error" in result["error"]
        assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_bulk_success(self, mock_crawler_class, basic_config):
        """Test bulk crawling success."""
        # Setup mock crawler with AsyncMock
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        # Setup mock results
        urls = ["https://example.com/1", "https://example.com/2"]

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        # Mock scrape_url to return success
        with patch.object(provider, 'scrape_url') as mock_scrape:
            mock_scrape.side_effect = [
                {"success": True, "url": urls[0], "content": "Content 1"},
                {"success": True, "url": urls[1], "content": "Content 2"},
            ]

            results = await provider.crawl_bulk(urls)

            assert len(results) == 2
            assert all(r["success"] for r in results)
            assert mock_scrape.call_count == 2

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_bulk_with_failures(self, mock_crawler_class, basic_config):
        """Test bulk crawling with some failures."""
        # Setup mock crawler with AsyncMock
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        urls = ["https://example.com/1", "https://example.com/2"]

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        # Mock scrape_url to have one success and one exception
        with patch.object(provider, 'scrape_url') as mock_scrape:
            mock_scrape.side_effect = [
                {"success": True, "url": urls[0], "content": "Content 1"},
                Exception("Network error"),
            ]

            results = await provider.crawl_bulk(urls)

            # Should only return successful results
            assert len(results) == 1
            assert results[0]["success"] is True

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_site_success(self, mock_crawler_class, basic_config):
        """Test site crawling success."""
        # Setup mock crawler with AsyncMock
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        # Mock crawl_bulk to return pages - need to handle multiple calls
        with patch.object(provider, 'crawl_bulk') as mock_crawl_bulk:
            # First call: ["https://example.com"] -> returns home page with link
            # Second call: ["https://example.com/about"] -> returns about page
            mock_crawl_bulk.side_effect = [
                [  # First call result
                    {
                        "success": True,
                        "url": "https://example.com",
                        "content": "Home page content",
                        "html": "<html></html>",
                        "metadata": {"title": "Home"},
                        "title": "Home",
                        "links": [{"href": "https://example.com/about"}],
                    }
                ],
                [  # Second call result
                    {
                        "success": True,
                        "url": "https://example.com/about",
                        "content": "About page content",
                        "html": "<html><h1>About</h1></html>",
                        "metadata": {"title": "About"},
                        "title": "About",
                        "links": [],
                    }
                ]
            ]

            result = await provider.crawl_site("https://example.com", max_pages=2)

            assert result["success"] is True
            assert len(result["pages"]) == 2
            assert result["total"] == 2
            assert result["provider"] == "crawl4ai"
            assert result["pages"][0]["url"] == "https://example.com"
            assert result["pages"][1]["url"] == "https://example.com/about"

    @pytest.mark.asyncio
    async def test_crawl_site_not_initialized(self, basic_config):
        """Test site crawling when not initialized."""
        provider = Crawl4AIProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_site("https://example.com")

    @pytest.mark.asyncio
    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_site_exception(self, mock_crawler_class, basic_config):
        """Test site crawling exception handling."""
        # Setup mock crawler with AsyncMock
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        provider = Crawl4AIProvider(basic_config)
        await provider.initialize()

        # Mock crawl_bulk to raise exception
        with patch.object(provider, 'crawl_bulk') as mock_crawl_bulk:
            mock_crawl_bulk.side_effect = Exception("Bulk crawl failed")

            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "Bulk crawl failed" in result["error"]
            assert result["provider"] == "crawl4ai"

    def test_config_timeout_conversion(self, basic_config):
        """Test that page timeout is correctly converted from seconds to milliseconds."""
        Crawl4AIProvider(basic_config)

        # The config stores timeout in seconds (30.0)
        # Provider should convert to milliseconds for internal use
        assert basic_config.page_timeout == 30.0


class TestCrawlCache:
    """Test caching functionality."""

    def test_init(self):
        """Test CrawlCache initialization."""
        mock_cache_manager = MagicMock()
        cache = CrawlCache(mock_cache_manager)

        assert cache.cache == mock_cache_manager
        assert cache.ttl == 86400  # 24 hours

    def test_calculate_ttl_api_docs(self):
        """Test TTL calculation for API docs."""
        mock_cache_manager = MagicMock()
        cache = CrawlCache(mock_cache_manager)

        result = {"url": "https://example.com/api/docs"}
        ttl = cache.calculate_ttl(result)

        assert ttl == 604800  # 7 days

    def test_calculate_ttl_blog_posts(self):
        """Test TTL calculation for blog posts."""
        mock_cache_manager = MagicMock()
        cache = CrawlCache(mock_cache_manager)

        result = {"url": "https://example.com/blog/post"}
        ttl = cache.calculate_ttl(result)

        assert ttl == 2592000  # 30 days

    def test_calculate_ttl_default(self):
        """Test TTL calculation for other content."""
        mock_cache_manager = MagicMock()
        cache = CrawlCache(mock_cache_manager)

        result = {"url": "https://example.com/docs"}
        ttl = cache.calculate_ttl(result)

        assert ttl == 259200  # 3 days

    @pytest.mark.asyncio
    async def test_get_or_crawl_cache_hit(self):
        """Test cache hit scenario."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = {"cached": "result"}

        cache = CrawlCache(mock_cache_manager)
        mock_crawler = AsyncMock()

        result = await cache.get_or_crawl("https://example.com", mock_crawler)

        assert result == {"cached": "result"}
        mock_crawler.scrape_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_crawl_cache_miss(self):
        """Test cache miss scenario."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = None

        cache = CrawlCache(mock_cache_manager)
        mock_crawler = AsyncMock()
        mock_crawler.scrape_url.return_value = {"success": True, "url": "https://example.com"}

        result = await cache.get_or_crawl("https://example.com", mock_crawler)

        assert result["success"] is True
        mock_crawler.scrape_url.assert_called_once_with("https://example.com")
        mock_cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_crawl_force_refresh(self):
        """Test forced refresh scenario."""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get.return_value = {"cached": "result"}

        cache = CrawlCache(mock_cache_manager)
        mock_crawler = AsyncMock()
        mock_crawler.scrape_url.return_value = {"success": True, "url": "https://example.com"}

        result = await cache.get_or_crawl("https://example.com", mock_crawler, force_refresh=True)

        assert result["success"] is True
        mock_crawler.scrape_url.assert_called_once_with("https://example.com")
        mock_cache_manager.get.assert_not_called()


class TestCrawlBenchmark:
    """Test benchmarking functionality."""

    def test_init(self):
        """Test CrawlBenchmark initialization."""
        mock_crawl4ai = MagicMock()
        mock_firecrawl = MagicMock()

        benchmark = CrawlBenchmark(mock_crawl4ai, mock_firecrawl)

        assert benchmark.crawl4ai == mock_crawl4ai
        assert benchmark.firecrawl == mock_firecrawl

    @pytest.mark.asyncio
    async def test_run_comparison_success(self):
        """Test benchmark comparison with successful results."""
        mock_crawl4ai = AsyncMock()
        mock_crawl4ai.scrape_url.return_value = {"success": True}

        mock_firecrawl = AsyncMock()
        mock_firecrawl.scrape_url.return_value = {"success": True}

        benchmark = CrawlBenchmark(mock_crawl4ai, mock_firecrawl)

        with patch("time.time", side_effect=[0, 0.1, 0.2, 0.3]):  # Mock timing
            results = await benchmark.run_comparison(["https://example.com"])

        assert "crawl4ai" in results
        assert "firecrawl" in results
        assert results["crawl4ai"]["success"] == 1
        assert results["firecrawl"]["success"] == 1

    @pytest.mark.asyncio
    async def test_run_comparison_with_failures(self):
        """Test benchmark comparison with failures."""
        mock_crawl4ai = AsyncMock()
        mock_crawl4ai.scrape_url.side_effect = Exception("Crawl4AI error")

        mock_firecrawl = AsyncMock()
        mock_firecrawl.scrape_url.return_value = {"success": False}

        benchmark = CrawlBenchmark(mock_crawl4ai, mock_firecrawl)

        results = await benchmark.run_comparison(["https://example.com"])

        assert results["crawl4ai"]["failed"] == 1
        assert results["firecrawl"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_run_comparison_no_firecrawl(self):
        """Test benchmark comparison without Firecrawl."""
        mock_crawl4ai = AsyncMock()
        mock_crawl4ai.scrape_url.return_value = {"success": True}

        benchmark = CrawlBenchmark(mock_crawl4ai, None)

        results = await benchmark.run_comparison(["https://example.com"])

        assert results["crawl4ai"]["success"] == 1
        assert results["firecrawl"]["success"] == 0
        assert results["firecrawl"]["failed"] == 0


class TestPydanticConfigIntegration:
    """Test integration with Pydantic configuration models."""

    def test_config_validation(self):
        """Test that Pydantic config validation works."""
        # Valid config
        config = Crawl4AIConfig(
            browser_type="chromium",
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            max_concurrent_crawls=5,
            page_timeout=30.0,
        )

        provider = Crawl4AIProvider(config)
        assert provider.config == config

    def test_config_field_access(self, basic_config):
        """Test accessing Pydantic config fields."""
        provider = Crawl4AIProvider(basic_config)

        assert provider.config.browser_type == "chromium"
        assert provider.config.headless is True
        assert provider.config.viewport_width == 1920
        assert provider.config.viewport_height == 1080
        assert provider.config.max_concurrent_crawls == 5
        assert provider.config.page_timeout == 30.0

    def test_config_defaults(self):
        """Test Pydantic config defaults."""
        config = Crawl4AIConfig()  # Use all defaults
        provider = Crawl4AIProvider(config)

        assert provider.config.browser_type == "chromium"
        assert provider.config.headless is True
        assert provider.config.viewport_width == 1920
        assert provider.config.viewport_height == 1080
        assert provider.config.max_concurrent_crawls == 10
        assert provider.config.page_timeout == 30.0

    def test_config_browser_config_mapping(self, basic_config):
        """Test that Pydantic config fields map correctly to browser config."""
        provider = Crawl4AIProvider(basic_config)
        browser_config = provider.browser_config

        # Verify mapping from Pydantic config to browser config
        assert browser_config.browser_type == basic_config.browser_type
        assert browser_config.headless == basic_config.headless
        assert browser_config.viewport_width == basic_config.viewport_width
        assert browser_config.viewport_height == basic_config.viewport_height
