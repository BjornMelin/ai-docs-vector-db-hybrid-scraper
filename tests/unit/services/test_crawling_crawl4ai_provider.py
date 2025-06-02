"""Tests for Crawl4AI provider module."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.base import BaseService
from src.services.crawling.base import CrawlProvider
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.crawling.crawl4ai_provider import CrawlBenchmark
from src.services.crawling.crawl4ai_provider import CrawlCache
from src.services.crawling.crawl4ai_provider import DocumentationExtractor
from src.services.crawling.crawl4ai_provider import JavaScriptExecutor
from src.services.errors import CrawlServiceError


class TestJavaScriptExecutor:
    """Test the JavaScriptExecutor helper class."""

    def test_init(self):
        """Test JavaScriptExecutor initialization."""
        executor = JavaScriptExecutor()
        assert hasattr(executor, "common_patterns")
        assert "spa_navigation" in executor.common_patterns
        assert "infinite_scroll" in executor.common_patterns
        assert "click_show_more" in executor.common_patterns

    def test_get_js_for_site_python_docs(self):
        """Test getting JavaScript for Python docs."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://docs.python.org/3/")
        assert js == executor.common_patterns["spa_navigation"]

    def test_get_js_for_site_react_docs(self):
        """Test getting JavaScript for React docs."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://reactjs.org/docs")
        assert js == executor.common_patterns["spa_navigation"]

    def test_get_js_for_site_stackoverflow(self):
        """Test getting JavaScript for Stack Overflow."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://stackoverflow.com/questions")
        assert js == executor.common_patterns["infinite_scroll"]

    def test_get_js_for_site_mdn(self):
        """Test getting JavaScript for MDN."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://developer.mozilla.org/docs")
        assert js == executor.common_patterns["click_show_more"]

    def test_get_js_for_site_unknown(self):
        """Test getting JavaScript for unknown site."""
        executor = JavaScriptExecutor()
        js = executor.get_js_for_site("https://unknown-site.com")
        assert js is None

    def test_common_patterns_structure(self):
        """Test that common patterns have expected structure."""
        executor = JavaScriptExecutor()

        for _pattern_name, pattern_code in executor.common_patterns.items():
            assert isinstance(pattern_code, str)
            assert len(pattern_code.strip()) > 0
            # Each pattern should be valid JavaScript-like syntax
            assert "await" in pattern_code or "setTimeout" in pattern_code


class TestDocumentationExtractor:
    """Test the DocumentationExtractor helper class."""

    def test_init(self):
        """Test DocumentationExtractor initialization."""
        extractor = DocumentationExtractor()
        assert hasattr(extractor, "selectors")
        assert "content" in extractor.selectors
        assert "code" in extractor.selectors
        assert "nav" in extractor.selectors
        assert "metadata" in extractor.selectors

    def test_selectors_structure(self):
        """Test that selectors have expected structure."""
        extractor = DocumentationExtractor()

        # Content selectors should be a list
        assert isinstance(extractor.selectors["content"], list)
        assert len(extractor.selectors["content"]) > 0

        # Code selectors should be a list
        assert isinstance(extractor.selectors["code"], list)
        assert len(extractor.selectors["code"]) > 0

        # Metadata should be a dict
        assert isinstance(extractor.selectors["metadata"], dict)
        assert "title" in extractor.selectors["metadata"]

    def test_create_extraction_schema_general(self):
        """Test creating general extraction schema."""
        extractor = DocumentationExtractor()
        schema = extractor.create_extraction_schema("general")

        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema

    def test_create_extraction_schema_api_reference(self):
        """Test creating API reference extraction schema."""
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
        """Test creating tutorial extraction schema."""
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
        """Test creating guide extraction schema."""
        extractor = DocumentationExtractor()
        schema = extractor.create_extraction_schema("guide")

        assert "title" in schema
        assert "content" in schema
        assert "code_blocks" in schema
        assert "sections" in schema
        assert "callouts" in schema
        assert "related" in schema


class TestCrawl4AIProvider:
    """Test the Crawl4AIProvider class."""

    def test_init_default_config(self):
        """Test Crawl4AIProvider initialization with default config."""
        provider = Crawl4AIProvider()

        assert isinstance(provider, BaseService)
        assert isinstance(provider, CrawlProvider)
        assert provider.config == {}
        assert provider._initialized is False
        assert provider._crawler is None

    def test_init_with_config(self):
        """Test Crawl4AIProvider initialization with custom config."""
        config = {
            "browser": "firefox",
            "headless": False,
            "rate_limit": 100,
            "max_concurrent": 20,
        }
        provider = Crawl4AIProvider(config)

        assert provider.config == config
        assert provider.max_concurrent == 20

    def test_init_with_rate_limiter(self):
        """Test initialization with custom rate limiter."""
        rate_limiter = MagicMock()
        provider = Crawl4AIProvider(rate_limiter=rate_limiter)

        assert provider.rate_limiter == rate_limiter

    def test_browser_config_creation(self):
        """Test browser configuration creation."""
        config = {
            "browser": "firefox",
            "headless": False,
            "viewport_width": 1366,
            "viewport_height": 768,
            "user_agent": "Custom/1.0",
        }
        provider = Crawl4AIProvider(config)

        browser_config = provider.browser_config
        assert browser_config.browser_type == "firefox"
        assert browser_config.headless is False
        assert browser_config.viewport_width == 1366
        assert browser_config.viewport_height == 768
        assert browser_config.user_agent == "Custom/1.0"

    def test_semaphore_creation(self):
        """Test semaphore creation for concurrent crawling."""
        config = {"max_concurrent": 5}
        provider = Crawl4AIProvider(config)

        assert isinstance(provider.semaphore, asyncio.Semaphore)
        # Semaphore value is internal, but we can test it was created

    def test_helpers_initialization(self):
        """Test that helper classes are initialized."""
        provider = Crawl4AIProvider()

        assert isinstance(provider.js_executor, JavaScriptExecutor)
        assert isinstance(provider.doc_extractor, DocumentationExtractor)

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful provider initialization."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value = mock_crawler

            provider = Crawl4AIProvider()
            await provider.initialize()

            assert provider._initialized is True
            assert provider._crawler == mock_crawler
            mock_crawler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test provider initialization failure."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler_class:
            mock_crawler_class.side_effect = Exception("Init failed")

            provider = Crawl4AIProvider()

            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Crawl4AI"
            ):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that multiple initialization calls are safe."""
        with patch(
            "src.services.crawling.crawl4ai_provider.AsyncWebCrawler"
        ) as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value = mock_crawler

            provider = Crawl4AIProvider()

            await provider.initialize()
            await provider.initialize()  # Second call

            # Should only create crawler once
            mock_crawler_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test provider cleanup."""
        provider = Crawl4AIProvider()
        mock_crawler = AsyncMock()
        provider._crawler = mock_crawler
        provider._initialized = True

        await provider.cleanup()

        mock_crawler.close.assert_called_once()
        assert provider._crawler is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_no_crawler(self):
        """Test cleanup when no crawler exists."""
        provider = Crawl4AIProvider()

        # Should not raise exception
        await provider.cleanup()
        assert provider._crawler is None

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self):
        """Test scraping when provider not initialized."""
        provider = Crawl4AIProvider()

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self):
        """Test successful URL scraping."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        # Mock crawler and result
        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Content"
        mock_result.html = "<h1>Test Content</h1>"
        mock_result.metadata = {"title": "Test Page"}
        mock_result.extracted_content = {}
        mock_result.links = [{"href": "https://example.com/link"}]
        mock_result.media = {"images": []}

        mock_crawler.arun.return_value = mock_result
        provider._crawler = mock_crawler

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["content"] == "# Test Content"
        assert result["html"] == "<h1>Test Content</h1>"
        assert result["title"] == "Test Page"
        assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self):
        """Test URL scraping failure."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Page not found"

        mock_crawler.arun.return_value = mock_result
        provider._crawler = mock_crawler

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is False
        assert result["error"] == "Page not found"
        assert result["url"] == "https://example.com"
        assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_url_with_rate_limiting(self):
        """Test URL scraping with rate limiting."""
        rate_limiter = AsyncMock()
        provider = Crawl4AIProvider(rate_limiter=rate_limiter)
        provider._initialized = True

        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "Content"
        mock_result.html = "<p>Content</p>"
        mock_result.metadata = {}
        mock_result.extracted_content = {}
        mock_result.links = []
        mock_result.media = {}

        mock_crawler.arun.return_value = mock_result
        provider._crawler = mock_crawler

        await provider.scrape_url("https://example.com")

        rate_limiter.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_url_with_custom_js(self):
        """Test URL scraping with custom JavaScript."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "Content"
        mock_result.html = "<p>Content</p>"
        mock_result.metadata = {}
        mock_result.extracted_content = {}
        mock_result.links = []
        mock_result.media = {}

        mock_crawler.arun.return_value = mock_result
        provider._crawler = mock_crawler

        custom_js = "console.log('custom');"
        await provider.scrape_url("https://example.com", js_code=custom_js)

        # Check that custom JS was passed to crawler
        call_args = mock_crawler.arun.call_args
        config = call_args[1]["config"]
        assert config.js_code == custom_js

    @pytest.mark.asyncio
    async def test_scrape_url_with_structured_extraction(self):
        """Test URL scraping with structured extraction."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "Content"
        mock_result.html = "<p>Content</p>"
        mock_result.metadata = {}
        mock_result.extracted_content = {"title": "Extracted Title"}
        mock_result.links = []
        mock_result.media = {}

        mock_crawler.arun.return_value = mock_result
        provider._crawler = mock_crawler

        result = await provider.scrape_url(
            "https://example.com", extraction_type="structured"
        )

        assert result["structured_data"] == {"title": "Extracted Title"}
        assert result["metadata"]["has_structured_data"] is True

    @pytest.mark.asyncio
    async def test_scrape_url_exception_handling(self):
        """Test exception handling during scraping."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        mock_crawler = AsyncMock()
        mock_crawler.arun.side_effect = Exception("Network error")
        provider._crawler = mock_crawler

        result = await provider.scrape_url("https://example.com")

        assert result["success"] is False
        assert "Network error" in result["error"]
        assert "error_context" in result

    @pytest.mark.asyncio
    async def test_crawl_bulk(self):
        """Test bulk URL crawling."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        # Mock successful scraping
        with patch.object(provider, "scrape_url") as mock_scrape:
            mock_scrape.side_effect = [
                {
                    "success": True,
                    "url": "https://example.com/1",
                    "content": "Content 1",
                },
                {
                    "success": True,
                    "url": "https://example.com/2",
                    "content": "Content 2",
                },
                {"success": False, "url": "https://example.com/3", "error": "Failed"},
            ]

            urls = [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
            ]
            results = await provider.crawl_bulk(urls)

            # crawl_bulk only returns successful results (based on implementation)
            assert len(results) == 3  # All results are returned
            successful_results = [r for r in results if r.get("success", False)]
            assert len(successful_results) == 2

    @pytest.mark.asyncio
    async def test_crawl_site_not_initialized(self):
        """Test site crawling when provider not initialized."""
        provider = Crawl4AIProvider()

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_site_success(self):
        """Test successful site crawling."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        # Mock crawl_bulk to return pages with links
        with patch.object(provider, "crawl_bulk") as mock_crawl_bulk:
            # First call returns initial page with links, subsequent calls return pages for those links
            mock_crawl_bulk.side_effect = [
                [
                    {
                        "success": True,
                        "url": "https://example.com",
                        "content": "Page content",
                        "html": "<p>Page content</p>",
                        "metadata": {"title": "Home"},
                        "title": "Home Page",
                        "links": [{"href": "https://example.com/about"}],
                    }
                ],
                [
                    {
                        "success": True,
                        "url": "https://example.com/about",
                        "content": "About content",
                        "html": "<p>About content</p>",
                        "metadata": {"title": "About"},
                        "title": "About Page",
                        "links": [],
                    }
                ],
            ]

            result = await provider.crawl_site("https://example.com", max_pages=5)

            assert result["success"] is True
            assert result["total"] == 2
            assert len(result["pages"]) == 2
            assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_crawl_site_memory_optimization(self):
        """Test memory optimization during large site crawls."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        # Mock crawl_bulk to return pages in batches
        with patch.object(provider, "crawl_bulk") as mock_crawl_bulk:
            # Simulate batch processing - return 10 pages per batch call
            def batch_side_effect(urls):
                batch_pages = []
                for url in urls[:10]:  # Process up to 10 URLs per batch
                    batch_pages.append(
                        {
                            "success": True,
                            "url": url,
                            "content": f"Content for {url}",
                            "html": f"<p>Content for {url}</p>",
                            "metadata": {},
                            "title": f"Page {url}",
                            "links": [],
                        }
                    )
                return batch_pages

            mock_crawl_bulk.side_effect = batch_side_effect

            result = await provider.crawl_site("https://example.com", max_pages=50)

            # Should respect max_pages limit
            assert result["total"] <= 50

    @pytest.mark.asyncio
    async def test_crawl_site_exception_handling(self):
        """Test exception handling during site crawling."""
        provider = Crawl4AIProvider()
        provider._initialized = True

        with patch.object(provider, "crawl_bulk") as mock_crawl_bulk:
            mock_crawl_bulk.side_effect = Exception("Crawl failed")

            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "Crawl failed" in result["error"]
            assert "error_context" in result


class TestCrawlCache:
    """Test the CrawlCache helper class."""

    def test_init(self):
        """Test CrawlCache initialization."""
        cache_manager = MagicMock()
        cache = CrawlCache(cache_manager)

        assert cache.cache == cache_manager
        assert cache.ttl == 86400  # 24 hours

    @pytest.mark.asyncio
    async def test_get_or_crawl_cache_hit(self):
        """Test getting from cache when data exists."""
        cache_manager = AsyncMock()
        cached_data = {"success": True, "content": "Cached content"}
        cache_manager.get.return_value = cached_data

        cache = CrawlCache(cache_manager)
        crawler = AsyncMock()

        result = await cache.get_or_crawl("https://example.com", crawler)

        assert result == cached_data
        cache_manager.get.assert_called_once()
        crawler.scrape_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_crawl_cache_miss(self):
        """Test crawling when cache miss occurs."""
        cache_manager = AsyncMock()
        cache_manager.get.return_value = None  # Cache miss

        crawl_result = {
            "success": True,
            "content": "Fresh content",
            "url": "https://example.com",
        }
        crawler = AsyncMock()
        crawler.scrape_url.return_value = crawl_result

        cache = CrawlCache(cache_manager)

        result = await cache.get_or_crawl("https://example.com", crawler)

        assert result == crawl_result
        crawler.scrape_url.assert_called_once_with("https://example.com")
        cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_crawl_force_refresh(self):
        """Test force refresh bypasses cache."""
        cache_manager = AsyncMock()
        crawl_result = {
            "success": True,
            "content": "Fresh content",
            "url": "https://example.com",
        }
        crawler = AsyncMock()
        crawler.scrape_url.return_value = crawl_result

        cache = CrawlCache(cache_manager)

        result = await cache.get_or_crawl(
            "https://example.com", crawler, force_refresh=True
        )

        assert result == crawl_result
        cache_manager.get.assert_not_called()
        crawler.scrape_url.assert_called_once()

    def test_calculate_ttl_api_docs(self):
        """Test TTL calculation for API documentation."""
        cache = CrawlCache(MagicMock())
        result = {"url": "https://example.com/api/docs"}

        ttl = cache.calculate_ttl(result)
        assert ttl == 604800  # 7 days

    def test_calculate_ttl_blog(self):
        """Test TTL calculation for blog posts."""
        cache = CrawlCache(MagicMock())
        result = {"url": "https://example.com/blog/post"}

        ttl = cache.calculate_ttl(result)
        assert ttl == 2592000  # 30 days

    def test_calculate_ttl_other(self):
        """Test TTL calculation for other content."""
        cache = CrawlCache(MagicMock())
        result = {"url": "https://example.com/tutorials/guide"}

        ttl = cache.calculate_ttl(result)
        assert ttl == 259200  # 3 days


class TestCrawlBenchmark:
    """Test the CrawlBenchmark helper class."""

    def test_init(self):
        """Test CrawlBenchmark initialization."""
        crawl4ai = MagicMock()
        firecrawl = MagicMock()

        benchmark = CrawlBenchmark(crawl4ai, firecrawl)

        assert benchmark.crawl4ai == crawl4ai
        assert benchmark.firecrawl == firecrawl

    def test_init_without_firecrawl(self):
        """Test initialization without Firecrawl provider."""
        crawl4ai = MagicMock()

        benchmark = CrawlBenchmark(crawl4ai)

        assert benchmark.crawl4ai == crawl4ai
        assert benchmark.firecrawl is None

    @pytest.mark.asyncio
    async def test_run_comparison_crawl4ai_only(self):
        """Test benchmark with only Crawl4AI provider."""
        crawl4ai = AsyncMock()
        crawl4ai.scrape_url.return_value = {"success": True}

        benchmark = CrawlBenchmark(crawl4ai)

        results = await benchmark.run_comparison(["https://example.com"])

        assert "crawl4ai" in results
        assert "firecrawl" in results
        assert results["crawl4ai"]["success"] == 1
        assert results["crawl4ai"]["failed"] == 0
        assert results["firecrawl"]["success"] == 0
        assert results["firecrawl"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_comparison_both_providers(self):
        """Test benchmark with both providers."""
        crawl4ai = AsyncMock()
        crawl4ai.scrape_url.return_value = {"success": True}

        firecrawl = AsyncMock()
        firecrawl.scrape_url.return_value = {"success": True}

        benchmark = CrawlBenchmark(crawl4ai, firecrawl)

        results = await benchmark.run_comparison(["https://example.com"])

        assert results["crawl4ai"]["success"] == 1
        assert results["firecrawl"]["success"] == 1
        assert "avg_time" in results["crawl4ai"]
        assert "avg_time" in results["firecrawl"]

    @pytest.mark.asyncio
    async def test_run_comparison_with_failures(self):
        """Test benchmark handling provider failures."""
        crawl4ai = AsyncMock()
        crawl4ai.scrape_url.side_effect = Exception("Crawl4AI failed")

        firecrawl = AsyncMock()
        firecrawl.scrape_url.return_value = {"success": False}

        benchmark = CrawlBenchmark(crawl4ai, firecrawl)

        results = await benchmark.run_comparison(["https://example.com"])

        assert results["crawl4ai"]["failed"] == 1
        assert results["firecrawl"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_timing_statistics(self):
        """Test that timing statistics are calculated correctly."""

        async def slow_scrape(url):
            await asyncio.sleep(0.01)  # Small delay for timing
            return {"success": True}

        crawl4ai = AsyncMock()
        crawl4ai.scrape_url = slow_scrape

        benchmark = CrawlBenchmark(crawl4ai)

        results = await benchmark.run_comparison(["https://example.com"])

        crawl4ai_stats = results["crawl4ai"]
        assert "avg_time" in crawl4ai_stats
        assert "p95_time" in crawl4ai_stats
        assert "min_time" in crawl4ai_stats
        assert "max_time" in crawl4ai_stats
        assert crawl4ai_stats["avg_time"] > 0
