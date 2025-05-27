"""Comprehensive tests for enhanced Crawl4AI provider with advanced features."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.crawling.crawl4ai_provider import CrawlBenchmark
from src.services.crawling.crawl4ai_provider import CrawlCache
from src.services.crawling.crawl4ai_provider import DocumentationExtractor
from src.services.crawling.crawl4ai_provider import JavaScriptExecutor


class TestJavaScriptExecutor:
    """Test JavaScript execution patterns."""

    def test_get_js_for_site_spa(self):
        """Test JavaScript for SPA sites."""
        executor = JavaScriptExecutor()

        # Test React site
        js = executor.get_js_for_site("https://react.dev/learn")
        assert js is not None
        assert "MutationObserver" in js
        assert "content-loaded" in js

        # Test Python docs
        js = executor.get_js_for_site("https://docs.python.org/3/")
        assert js is not None
        assert "MutationObserver" in js

    def test_get_js_for_site_infinite_scroll(self):
        """Test JavaScript for infinite scroll sites."""
        executor = JavaScriptExecutor()

        js = executor.get_js_for_site("https://stackoverflow.com/questions")
        assert js is not None
        assert "scrollTo" in js
        assert "document.body.scrollHeight" in js

    def test_get_js_for_site_click_show_more(self):
        """Test JavaScript for show more buttons."""
        executor = JavaScriptExecutor()

        js = executor.get_js_for_site("https://developer.mozilla.org/en-US/")
        assert js is not None
        assert "show-more" in js
        assert "button.click()" in js

    def test_get_js_for_unknown_site(self):
        """Test no JavaScript for unknown sites."""
        executor = JavaScriptExecutor()

        js = executor.get_js_for_site("https://unknown-site.com")
        assert js is None


class TestDocumentationExtractor:
    """Test documentation extraction strategies."""

    def test_create_extraction_schema_general(self):
        """Test general extraction schema."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema()
        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema
        assert isinstance(schema["content"], list)

    def test_create_extraction_schema_api_reference(self):
        """Test API reference extraction schema."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("api_reference")
        assert "endpoints" in schema
        assert "parameters" in schema
        assert "responses" in schema
        assert "examples" in schema

    def test_create_extraction_schema_tutorial(self):
        """Test tutorial extraction schema."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("tutorial")
        assert "steps" in schema
        assert "code_examples" in schema
        assert "prerequisites" in schema
        assert "objectives" in schema

    def test_create_extraction_schema_guide(self):
        """Test guide extraction schema."""
        extractor = DocumentationExtractor()

        schema = extractor.create_extraction_schema("guide")
        assert "sections" in schema
        assert "content" in schema
        assert "callouts" in schema
        assert "related" in schema


class TestEnhancedCrawl4AIProvider:
    """Test enhanced Crawl4AI provider features."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "max_concurrent": 5,
            "rate_limit": 30,
            "headless": True,
            "browser": "chromium",
            "page_timeout": 15000,
        }

    @pytest.fixture
    def provider(self, config):
        """Create enhanced provider instance."""
        return Crawl4AIProvider(config=config)

    @pytest.mark.asyncio
    async def test_initialize_with_config(self, provider):
        """Test initialization with custom configuration."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            await provider.initialize()

            assert provider._initialized
            assert provider.max_concurrent == 5
            assert provider.browser_config.browser_type == "chromium"
            assert provider.browser_config.headless is True

    @pytest.mark.asyncio
    async def test_scrape_url_with_javascript(self, provider):
        """Test scraping with JavaScript execution."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.markdown = "# React Documentation"
            mock_result.html = "<h1>React Documentation</h1>"
            mock_result.metadata = {"title": "React"}
            mock_result.links = []
            mock_result.media = {}
            mock_result.extracted_content = None
            mock_instance.arun = AsyncMock(return_value=mock_result)

            await provider.initialize()

            # Test with React URL (should get SPA JavaScript)
            result = await provider.scrape_url("https://react.dev/learn")

            assert result["success"] is True
            assert result["content"] == "# React Documentation"
            assert result["provider"] == "crawl4ai"

            # Verify JavaScript was applied
            call_args = mock_instance.arun.call_args
            assert call_args is not None
            config = call_args.kwargs.get("config")
            assert config is not None
            assert config.js_code is not None

    @pytest.mark.asyncio
    async def test_scrape_url_with_structured_extraction(self, provider):
        """Test scraping with structured extraction."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock result with structured data
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.markdown = "# API Reference"
            mock_result.html = "<h1>API Reference</h1>"
            mock_result.metadata = {"title": "API Docs"}
            mock_result.links = []
            mock_result.media = {}
            mock_result.extracted_content = {
                "endpoints": ["/api/users", "/api/posts"],
                "parameters": ["id", "name"],
            }
            mock_instance.arun = AsyncMock(return_value=mock_result)

            await provider.initialize()

            result = await provider.scrape_url(
                "https://example.com/api", extraction_type="structured"
            )

            assert result["success"] is True
            assert result["structured_data"]["endpoints"] == [
                "/api/users",
                "/api/posts",
            ]
            assert result["metadata"]["has_structured_data"] is True

    @pytest.mark.asyncio
    async def test_crawl_bulk_concurrent(self, provider):
        """Test bulk crawling with concurrency."""
        urls = [f"https://example.com/page{i}" for i in range(10)]

        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock results
            results = []
            for i, _url in enumerate(urls):
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.markdown = f"Content {i}"
                mock_result.html = f"<p>Content {i}</p>"
                mock_result.metadata = {"title": f"Page {i}"}
                mock_result.links = []
                mock_result.media = {}
                mock_result.extracted_content = None
                results.append(mock_result)

            mock_instance.arun = AsyncMock(side_effect=results)

            await provider.initialize()

            # Test bulk crawling
            crawl_results = await provider.crawl_bulk(urls)

            assert len(crawl_results) == 10
            assert all(r["success"] for r in crawl_results)
            assert crawl_results[0]["content"] == "Content 0"
            assert crawl_results[9]["content"] == "Content 9"

    @pytest.mark.asyncio
    async def test_crawl_site_with_depth(self, provider):
        """Test site crawling with recursive URL discovery."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock results with links
            page1 = MagicMock()
            page1.success = True
            page1.markdown = "Main page"
            page1.html = "<html><body>Main</body></html>"
            page1.metadata = {"title": "Main"}
            page1.links = [
                {"href": "https://example.com/page2"},
                {"href": "https://example.com/page3"},
            ]
            page1.media = {}
            page1.extracted_content = None

            page2 = MagicMock()
            page2.success = True
            page2.markdown = "Page 2"
            page2.html = "<html><body>Page 2</body></html>"
            page2.metadata = {"title": "Page 2"}
            page2.links = []
            page2.media = {}
            page2.extracted_content = None

            mock_instance.arun = AsyncMock(side_effect=[page1, page2])

            await provider.initialize()

            result = await provider.crawl_site("https://example.com", max_pages=2)

            assert result["success"] is True
            assert result["total"] == 2
            assert len(result["pages"]) == 2
            assert result["pages"][0]["content"] == "Main page"
            assert result["pages"][1]["content"] == "Page 2"

    @pytest.mark.asyncio
    async def test_rate_limiting(self, provider):
        """Test rate limiting functionality."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler:
            mock_instance = AsyncMock()
            mock_crawler.return_value = mock_instance

            # Mock result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.markdown = "Content"
            mock_result.html = "<p>Content</p>"
            mock_result.metadata = {}
            mock_result.links = []
            mock_result.media = {}
            mock_result.extracted_content = None
            mock_instance.arun = AsyncMock(return_value=mock_result)

            await provider.initialize()

            # Test rate limiting by making concurrent requests
            urls = [f"https://example.com/{i}" for i in range(3)]

            results = await provider.crawl_bulk(urls)

            assert len(results) == 3
            assert all(r["success"] for r in results)
            # Rate limiter should have been called


class TestCrawlCache:
    """Test caching functionality."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache

    @pytest.fixture
    def crawl_cache(self, mock_cache_manager):
        """Create cache instance."""
        return CrawlCache(mock_cache_manager)

    @pytest.mark.asyncio
    async def test_cache_miss(self, crawl_cache, mock_cache_manager):
        """Test cache miss scenario."""
        mock_crawler = AsyncMock()
        mock_crawler.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "url": "https://example.com",
                "content": "Test content",
            }
        )

        result = await crawl_cache.get_or_crawl("https://example.com", mock_crawler)

        assert result["success"] is True
        assert result["content"] == "Test content"
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit(self, crawl_cache, mock_cache_manager):
        """Test cache hit scenario."""
        cached_data = {
            "success": True,
            "url": "https://example.com",
            "content": "Cached content",
        }
        mock_cache_manager.get = AsyncMock(return_value=cached_data)

        mock_crawler = AsyncMock()

        result = await crawl_cache.get_or_crawl("https://example.com", mock_crawler)

        assert result["content"] == "Cached content"
        mock_crawler.scrape_url.assert_not_called()

    def test_calculate_ttl_api_docs(self, crawl_cache):
        """Test TTL calculation for API docs."""
        result = {"url": "https://example.com/api/reference"}
        ttl = crawl_cache.calculate_ttl(result)
        assert ttl == 604800  # 7 days

    def test_calculate_ttl_blog(self, crawl_cache):
        """Test TTL calculation for blog posts."""
        result = {"url": "https://example.com/blog/post"}
        ttl = crawl_cache.calculate_ttl(result)
        assert ttl == 2592000  # 30 days

    def test_calculate_ttl_default(self, crawl_cache):
        """Test default TTL calculation."""
        result = {"url": "https://example.com/docs"}
        ttl = crawl_cache.calculate_ttl(result)
        assert ttl == 259200  # 3 days


class TestCrawlBenchmark:
    """Test benchmarking functionality."""

    @pytest.mark.asyncio
    async def test_benchmark_comparison(self):
        """Test benchmark comparison between crawlers."""
        # Mock Crawl4AI
        mock_crawl4ai = AsyncMock()
        mock_crawl4ai.scrape_url = AsyncMock(
            side_effect=[
                {"success": True},
                {"success": True},
                {"success": False},
            ]
        )

        # Mock Firecrawl
        mock_firecrawl = AsyncMock()
        mock_firecrawl.scrape_url = AsyncMock(
            side_effect=[
                {"success": True},
                {"success": False},
                {"success": False},
            ]
        )

        benchmark = CrawlBenchmark(mock_crawl4ai, mock_firecrawl)

        urls = ["https://example1.com", "https://example2.com", "https://example3.com"]
        results = await benchmark.run_comparison(urls)

        # Check Crawl4AI results
        assert results["crawl4ai"]["success"] == 2
        assert results["crawl4ai"]["failed"] == 1
        assert "avg_time" in results["crawl4ai"]
        assert "p95_time" in results["crawl4ai"]

        # Check Firecrawl results
        assert results["firecrawl"]["success"] == 1
        assert results["firecrawl"]["failed"] == 2
        assert "avg_time" in results["firecrawl"]
