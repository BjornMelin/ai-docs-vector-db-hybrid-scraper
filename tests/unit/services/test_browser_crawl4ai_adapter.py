"""Comprehensive tests for Crawl4AI adapter."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = {
        "max_concurrent": 5,
        "headless": True,
        "browser": "chromium",
        "page_timeout": 30000,
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": "Mozilla/5.0 (Test Browser)",
        "headers": {"Accept": "text/html"},
        "cookies": [],
        "verbose": False,
    }
    return config


@pytest.fixture
def adapter(mock_config):
    """Create Crawl4AIAdapter instance for testing."""
    return Crawl4AIAdapter(mock_config)


class TestCrawl4AIAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_initialization(self, adapter, mock_config):
        """Test basic adapter initialization."""
        assert adapter.config == mock_config
        assert adapter._provider is None
        assert adapter._initialized is False
        assert adapter.name == "crawl4ai"
        assert adapter._available is True  # Should detect if crawl4ai available

    def test_adapter_unavailable(self):
        """Test adapter when crawl4ai is not installed."""
        with patch.dict("sys.modules", {"crawl4ai": None}):
            adapter = Crawl4AIAdapter({})
            assert adapter._available is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, adapter):
        """Test successful adapter initialization."""
        mock_provider = AsyncMock()

        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIProvider",
            return_value=mock_provider,
        ):
            await adapter.initialize()

        assert adapter._initialized is True
        assert adapter._provider is mock_provider
        mock_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, adapter):
        """Test initialization failure handling."""
        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIProvider",
            side_effect=Exception("Init failed"),
        ):
            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Crawl4AI"
            ):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, adapter):
        """Test that initialization is idempotent."""
        mock_provider = AsyncMock()

        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIProvider",
            return_value=mock_provider,
        ):
            await adapter.initialize()
            await adapter.initialize()  # Second call

        # Should only initialize provider once
        mock_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, adapter):
        """Test adapter cleanup."""
        mock_provider = AsyncMock()
        adapter._provider = mock_provider
        adapter._initialized = True

        await adapter.cleanup()

        mock_provider.cleanup.assert_called_once()
        assert adapter._provider is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self, adapter):
        """Test cleanup when not initialized."""
        # Should not raise error
        await adapter.cleanup()
        assert adapter._initialized is False


class TestScrapeOperation:
    """Test main scrape functionality."""

    @pytest.mark.asyncio
    async def test_scrape_not_available(self, adapter):
        """Test scraping when adapter not available."""
        adapter._available = False

        with pytest.raises(CrawlServiceError, match="Crawl4AI not available"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, adapter):
        """Test scraping when adapter not initialized."""
        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_success_basic(self, adapter):
        """Test successful basic scraping."""
        url = "https://example.com"
        mock_result = {
            "url": url,
            "content": "# Example Content\nThis is test content.",
            "html": "<h1>Example Content</h1><p>This is test content.</p>",
            "title": "Example Page",
            "success": True,
            "status_code": 200,
            "response_headers": {"content-type": "text/html"},
            "metadata": {"extraction_time": 1.5},
            "links": ["https://example.com/link1", "https://example.com/link2"],
            "structured_data": {"type": "WebPage"},
        }

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = mock_result

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["success"] is True
        assert result["content"] == mock_result["content"]
        assert result["title"] == mock_result["title"]
        assert result["url"] == url
        assert result["status_code"] == 200
        assert result["metadata"]["extraction_method"] == "crawl4ai"
        assert result["metadata"]["extraction_time"] == 1.5
        assert len(result["links"]) == 2
        assert result["structured_data"]["type"] == "WebPage"

        mock_provider.scrape_url.assert_called_once_with(
            url=url,
            wait_for_selector=None,
            js_code=None,
            timeout=30000,
            extract_links=True,
            extract_structured_data=True,
        )

    @pytest.mark.asyncio
    async def test_scrape_with_wait_selector(self, adapter):
        """Test scraping with wait selector."""
        url = "https://example.com"
        wait_selector = ".content-loaded"

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "content": "Content",
            "success": True,
            "html": "<div>Content</div>",
        }

        adapter._provider = mock_provider
        adapter._initialized = True

        await adapter.scrape(url, wait_for_selector=wait_selector)

        call_args = mock_provider.scrape_url.call_args
        assert call_args.kwargs["wait_for_selector"] == wait_selector

    @pytest.mark.asyncio
    async def test_scrape_with_js_code(self, adapter):
        """Test scraping with JavaScript code execution."""
        url = "https://example.com"
        js_code = """
        document.querySelector('#load-more').click();
        await new Promise(r => setTimeout(r, 1000));
        """

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "content": "Extended content",
            "success": True,
            "html": "<div>Extended content</div>",
        }

        adapter._provider = mock_provider
        adapter._initialized = True

        await adapter.scrape(url, js_code=js_code)

        call_args = mock_provider.scrape_url.call_args
        assert call_args.kwargs["js_code"] == js_code

    @pytest.mark.asyncio
    async def test_scrape_with_custom_timeout(self, adapter):
        """Test scraping with custom timeout."""
        url = "https://example.com"
        timeout = 60000

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "content": "Content",
            "success": True,
            "html": "<div>Content</div>",
        }

        adapter._provider = mock_provider
        adapter._initialized = True

        await adapter.scrape(url, timeout=timeout)

        call_args = mock_provider.scrape_url.call_args
        assert call_args.kwargs["timeout"] == 60000

    @pytest.mark.asyncio
    async def test_scrape_failure_network_error(self, adapter):
        """Test handling of network errors."""
        url = "https://example.com"

        mock_provider = AsyncMock()
        mock_provider.scrape_url.side_effect = Exception(
            "Network error: Connection refused"
        )

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["success"] is False
        assert "Network error" in result["error"]
        assert result["url"] == url
        assert result["content"] == ""
        assert result["metadata"]["extraction_method"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_failure_timeout(self, adapter):
        """Test handling of timeout errors."""
        url = "https://slow-site.com"

        mock_provider = AsyncMock()
        mock_provider.scrape_url.side_effect = TimeoutError("Page load timeout")

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_scrape_partial_success(self, adapter):
        """Test handling of partial success (e.g., 404 pages)."""
        url = "https://example.com/notfound"
        mock_result = {
            "url": url,
            "content": "404 - Page Not Found",
            "html": "<h1>404</h1>",
            "title": "Not Found",
            "success": True,  # Crawl4AI might return success for 404
            "status_code": 404,
            "response_headers": {},
            "metadata": {},
            "links": [],
            "structured_data": {},
        }

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = mock_result

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        # Should still return the content but indicate the status
        assert result["success"] is True
        assert result["status_code"] == 404
        assert "404" in result["content"]

    @pytest.mark.asyncio
    async def test_scrape_handles_missing_fields(self, adapter):
        """Test graceful handling of missing fields in response."""
        url = "https://example.com"
        # Minimal response with missing fields
        mock_result = {
            "content": "Basic content",
            "success": True,
        }

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = mock_result

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        # Should provide defaults for missing fields
        assert result["success"] is True
        assert result["content"] == "Basic content"
        assert result["title"] == ""
        assert result["html"] == ""
        assert result["status_code"] is None
        assert result["links"] == []
        assert result["structured_data"] == {}


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_not_available(self, adapter):
        """Test health check when adapter not available."""
        adapter._available = False

        health = await adapter.health_check()

        assert health["status"] == "unavailable"
        assert health["healthy"] is False
        assert health["available"] is False

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, adapter):
        """Test health check when not initialized."""
        health = await adapter.health_check()

        assert health["status"] == "not_initialized"
        assert health["healthy"] is False
        assert health["available"] is True

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter):
        """Test successful health check."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "content": "Test content",
            "success": True,
            "html": "<div>Test</div>",
        }

        adapter._provider = mock_provider
        adapter._initialized = True

        health = await adapter.health_check()

        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert health["available"] is True
        assert "response_time_ms" in health
        assert health["response_time_ms"] < 15000  # Within timeout

        # Should have tried to scrape a test URL
        mock_provider.scrape_url.assert_called_once()
        call_args = mock_provider.scrape_url.call_args
        assert "example.com" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, adapter):
        """Test health check timeout handling."""
        mock_provider = AsyncMock()

        async def slow_scrape(*args, **kwargs):
            await asyncio.sleep(16)  # Longer than health check timeout

        mock_provider.scrape_url.side_effect = slow_scrape

        adapter._provider = mock_provider
        adapter._initialized = True

        health = await adapter.health_check()

        assert health["status"] == "timeout"
        assert health["healthy"] is False
        assert health["response_time_ms"] == 15000

    @pytest.mark.asyncio
    async def test_health_check_error(self, adapter):
        """Test health check error handling."""
        mock_provider = AsyncMock()
        mock_provider.scrape_url.side_effect = Exception("Health check failed")

        adapter._provider = mock_provider
        adapter._initialized = True

        health = await adapter.health_check()

        assert health["status"] == "error"
        assert health["healthy"] is False
        assert "Health check failed" in health["error"]


class TestCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self, adapter):
        """Test capability information."""
        capabilities = adapter.get_capabilities()

        assert capabilities["name"] == "crawl4ai"
        assert capabilities["browser_based"] is True
        assert "Fastest performance" in capabilities["advantages"][0]
        assert capabilities["performance"]["avg_speed"] == "0.3s per page"
        assert capabilities["performance"]["success_rate"] == "98%+"

        # Check all expected fields
        assert "advantages" in capabilities
        assert "limitations" in capabilities
        assert "best_for" in capabilities
        assert isinstance(capabilities["advantages"], list)
        assert isinstance(capabilities["limitations"], list)
        assert isinstance(capabilities["best_for"], list)


class TestAdvancedFeatures:
    """Test advanced Crawl4AI features."""

    @pytest.mark.asyncio
    async def test_scrape_with_complex_js(self, adapter):
        """Test scraping with complex JavaScript interactions."""
        url = "https://spa-app.com"
        js_code = """
        // Wait for React app to load
        await new Promise(r => {
            const checkLoaded = setInterval(() => {
                if (document.querySelector('[data-app-loaded="true"]')) {
                    clearInterval(checkLoaded);
                    r();
                }
            }, 100);
        });
        
        // Click through pagination
        const pages = document.querySelectorAll('.pagination-button');
        for (const page of pages) {
            page.click();
            await new Promise(r => setTimeout(r, 500));
        }
        """

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "content": "All paginated content",
            "success": True,
            "html": "<div>All pages content</div>",
            "metadata": {"pages_loaded": 5},
        }

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url, js_code=js_code)

        assert result["success"] is True
        assert result["metadata"]["pages_loaded"] == 5

    @pytest.mark.asyncio
    async def test_scrape_extract_structured_data(self, adapter):
        """Test extraction of structured data."""
        url = "https://example.com/product"
        mock_result = {
            "content": "Product page",
            "success": True,
            "html": "<div>Product</div>",
            "structured_data": {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Test Product",
                "price": "99.99",
                "currency": "USD",
                "availability": "InStock",
            },
        }

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = mock_result

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["structured_data"]["@type"] == "Product"
        assert result["structured_data"]["price"] == "99.99"

    @pytest.mark.asyncio
    async def test_scrape_extract_links(self, adapter):
        """Test link extraction functionality."""
        url = "https://example.com"
        mock_result = {
            "content": "Page with links",
            "success": True,
            "html": "<div>Content</div>",
            "links": [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://external.com/resource",
                "/relative/path",
                "#anchor",
            ],
        }

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = mock_result

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url, extract_links=True)

        assert len(result["links"]) == 5
        assert "https://example.com/page1" in result["links"]
        assert "/relative/path" in result["links"]

    @pytest.mark.asyncio
    async def test_concurrent_scrapes(self, adapter):
        """Test concurrent scraping capability."""
        urls = [f"https://example.com/page{i}" for i in range(5)]

        mock_provider = AsyncMock()

        async def mock_scrape(url, **kwargs):
            # Simulate some work
            await asyncio.sleep(0.1)
            return {
                "url": url,
                "content": f"Content for {url}",
                "success": True,
                "html": f"<div>Content for {url}</div>",
            }

        mock_provider.scrape_url.side_effect = mock_scrape

        adapter._provider = mock_provider
        adapter._initialized = True
        adapter.config["max_concurrent"] = 3  # Limit concurrency

        # Scrape all URLs concurrently
        tasks = [adapter.scrape(url) for url in urls]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r["success"] for r in results)
        assert all(r["url"] in urls for r in results)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handle_javascript_error(self, adapter):
        """Test handling of JavaScript execution errors."""
        url = "https://example.com"
        js_code = "invalid.javascript.code();"

        mock_provider = AsyncMock()
        mock_provider.scrape_url.side_effect = Exception(
            "JavaScript execution failed: invalid is not defined"
        )

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url, js_code=js_code)

        assert result["success"] is False
        assert "JavaScript execution failed" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_blocked_content(self, adapter):
        """Test handling of blocked/protected content."""
        url = "https://protected-site.com"

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = {
            "content": "Access Denied",
            "success": True,
            "status_code": 403,
            "html": "<h1>403 Forbidden</h1>",
        }

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["success"] is True  # Crawl succeeded even if access denied
        assert result["status_code"] == 403
        assert "Access Denied" in result["content"]

    @pytest.mark.asyncio
    async def test_handle_redirect_chain(self, adapter):
        """Test handling of redirect chains."""
        initial_url = "https://example.com/old-page"
        final_url = "https://example.com/new-page"

        mock_result = {
            "url": final_url,  # Final URL after redirects
            "content": "Redirected content",
            "success": True,
            "html": "<div>New page</div>",
            "metadata": {
                "redirect_chain": [
                    initial_url,
                    "https://example.com/intermediate",
                    final_url,
                ],
            },
        }

        mock_provider = AsyncMock()
        mock_provider.scrape_url.return_value = mock_result

        adapter._provider = mock_provider
        adapter._initialized = True

        result = await adapter.scrape(initial_url)

        assert result["success"] is True
        assert result["url"] == final_url
        assert len(result["metadata"]["redirect_chain"]) == 3


class TestConfigurationOptions:
    """Test various configuration options."""

    def test_custom_viewport_config(self):
        """Test custom viewport configuration."""
        config = {
            "viewport": {"width": 1366, "height": 768},
            "headless": True,
        }
        adapter = Crawl4AIAdapter(config)
        assert adapter.config["viewport"]["width"] == 1366
        assert adapter.config["viewport"]["height"] == 768

    def test_custom_headers_config(self):
        """Test custom headers configuration."""
        config = {
            "headers": {
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
            },
        }
        adapter = Crawl4AIAdapter(config)
        assert adapter.config["headers"]["Accept-Language"] == "en-US,en;q=0.9"
        assert adapter.config["headers"]["DNT"] == "1"

    def test_cookies_config(self):
        """Test cookies configuration."""
        config = {
            "cookies": [
                {"name": "session", "value": "abc123", "domain": ".example.com"},
                {"name": "auth", "value": "token", "domain": ".example.com"},
            ],
        }
        adapter = Crawl4AIAdapter(config)
        assert len(adapter.config["cookies"]) == 2
        assert adapter.config["cookies"][0]["name"] == "session"
