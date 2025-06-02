"""Comprehensive tests for the refactored Crawl4AI provider.

This test file covers the Crawl4AI provider with the new viewport dict structure
and improved configuration approach.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.models import Crawl4AIConfig
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.errors import CrawlServiceError


class TestCrawl4AIProviderRefactored:
    """Test the refactored Crawl4AI provider with improved config structure."""

    @pytest.fixture
    def mock_crawler(self):
        """Create a mock AsyncWebCrawler."""
        crawler = Mock()
        crawler.arun = AsyncMock()
        crawler.aclose = AsyncMock()
        return crawler

    @pytest.fixture
    def default_config(self):
        """Create a default Crawl4AI configuration."""
        return Crawl4AIConfig()

    @pytest.fixture
    def custom_config(self):
        """Create a custom Crawl4AI configuration with viewport dict."""
        return Crawl4AIConfig(
            browser_type="firefox",
            headless=False,
            viewport={"width": 1280, "height": 720},
            max_concurrent_crawls=5,
            page_timeout=45.0,
            remove_scripts=False,
        )

    @pytest.fixture
    def provider(self, default_config):
        """Create a Crawl4AI provider with default config."""
        return Crawl4AIProvider(default_config)

    def test_provider_initialization_default_config(self, provider, default_config):
        """Test provider initialization with default configuration."""
        assert provider.config == default_config
        assert provider.config.browser_type == "chromium"
        assert provider.config.headless is True
        assert provider.config.viewport == {"width": 1920, "height": 1080}
        assert provider.config.max_concurrent_crawls == 10
        assert provider.config.page_timeout == 30.0

    def test_provider_initialization_custom_config(self, custom_config):
        """Test provider initialization with custom configuration."""
        provider = Crawl4AIProvider(custom_config)
        
        assert provider.config == custom_config
        assert provider.config.browser_type == "firefox"
        assert provider.config.headless is False
        assert provider.config.viewport == {"width": 1280, "height": 720}
        assert provider.config.max_concurrent_crawls == 5
        assert provider.config.page_timeout == 45.0
        assert provider.config.remove_scripts is False

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_single_url_success(self, mock_crawler_class, provider, mock_crawler):
        """Test successful crawling of a single URL."""
        mock_crawler_class.return_value = mock_crawler
        
        # Mock successful crawl result
        mock_result = Mock()
        mock_result.success = True
        mock_result.cleaned_html = "<html><body><h1>Test Page</h1></body></html>"
        mock_result.markdown = "# Test Page"
        mock_result.extracted_content = "Test Page"
        mock_result.links = {"internal": ["http://example.com/page2"], "external": []}
        mock_result.metadata = {"title": "Test Page", "description": "A test page"}
        
        mock_crawler.arun.return_value = mock_result

        url = "http://example.com"
        result = await provider.crawl(url)

        # Verify crawler was called with correct parameters
        mock_crawler_class.assert_called_once()
        mock_crawler.arun.assert_called_once_with(
            url=url,
            word_threshold=5,
            only_text=False,
            remove_overlay_elements=True,
            process_iframes=True,
            delay_before_return_html=2.0,
            override_navigator=False,
        )
        
        # Verify result structure
        assert result["success"] is True
        assert result["url"] == url
        assert result["content"] == "Test Page"
        assert result["html"] == "<html><body><h1>Test Page</h1></body></html>"
        assert result["markdown"] == "# Test Page"
        assert result["links"]["internal"] == ["http://example.com/page2"]
        assert result["metadata"]["title"] == "Test Page"

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_single_url_failure(self, mock_crawler_class, provider, mock_crawler):
        """Test crawling failure for a single URL."""
        mock_crawler_class.return_value = mock_crawler
        
        # Mock failed crawl result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Page not found"
        
        mock_crawler.arun.return_value = mock_result

        url = "http://example.com/notfound"
        
        with pytest.raises(CrawlServiceError, match="Failed to crawl http://example.com/notfound: Page not found"):
            await provider.crawl(url)

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_single_url_exception(self, mock_crawler_class, provider, mock_crawler):
        """Test exception handling during single URL crawling."""
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.arun.side_effect = Exception("Network error")

        url = "http://example.com"
        
        with pytest.raises(CrawlError, match="Failed to crawl http://example.com: Network error"):
            await provider.crawl(url)

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_multiple_urls_success(self, mock_crawler_class, provider, mock_crawler):
        """Test successful crawling of multiple URLs."""
        mock_crawler_class.return_value = mock_crawler
        
        # Mock successful crawl results
        def mock_arun(url, **kwargs):
            result = Mock()
            result.success = True
            result.cleaned_html = f"<html><body><h1>Page {url[-1]}</h1></body></html>"
            result.markdown = f"# Page {url[-1]}"
            result.extracted_content = f"Page {url[-1]} content"
            result.links = {"internal": [], "external": []}
            result.metadata = {"title": f"Page {url[-1]}"}
            return result
        
        mock_crawler.arun.side_effect = mock_arun

        urls = ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3"]
        results = await provider.crawl_multiple(urls)

        # Verify crawler was called for each URL
        assert mock_crawler.arun.call_count == 3
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results, 1):
            assert result["success"] is True
            assert result["url"] == f"http://example.com/page{i}"
            assert f"Page {i}" in result["content"]

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_multiple_urls_partial_failure(self, mock_crawler_class, provider, mock_crawler):
        """Test crawling multiple URLs with some failures."""
        mock_crawler_class.return_value = mock_crawler
        
        # Mock mixed results (success, failure, success)
        def mock_arun(url, **kwargs):
            result = Mock()
            if "page2" in url:
                result.success = False
                result.error_message = "Page not found"
            else:
                result.success = True
                result.cleaned_html = f"<html><body><h1>{url}</h1></body></html>"
                result.markdown = f"# {url}"
                result.extracted_content = f"{url} content"
                result.links = {"internal": [], "external": []}
                result.metadata = {"title": url}
            return result
        
        mock_crawler.arun.side_effect = mock_arun

        urls = ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3"]
        results = await provider.crawl_multiple(urls, ignore_errors=True)

        # Verify results
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"] == "Page not found"
        assert results[2]["success"] is True

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawl_multiple_urls_concurrency_limit(self, mock_crawler_class, custom_config, mock_crawler):
        """Test that crawling respects concurrency limits."""
        provider = Crawl4AIProvider(custom_config)  # max_concurrent_crawls = 5
        mock_crawler_class.return_value = mock_crawler
        
        # Create a semaphore to track concurrent calls
        call_tracker = asyncio.Semaphore(0)
        max_concurrent = 0
        current_concurrent = 0
        
        async def mock_arun(url, **kwargs):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            
            # Simulate some async work
            await asyncio.sleep(0.01)
            
            current_concurrent -= 1
            
            result = Mock()
            result.success = True
            result.cleaned_html = f"<html><body>{url}</body></html>"
            result.markdown = url
            result.extracted_content = url
            result.links = {"internal": [], "external": []}
            result.metadata = {"title": url}
            return result
        
        mock_crawler.arun.side_effect = mock_arun

        # Create more URLs than the concurrency limit
        urls = [f"http://example.com/page{i}" for i in range(10)]
        results = await provider.crawl_multiple(urls)

        # Verify concurrency was limited
        assert max_concurrent <= custom_config.max_concurrent_crawls
        assert len(results) == 10
        assert all(result["success"] for result in results)

    def test_crawl_options_configuration(self, provider):
        """Test that crawl options are properly configured."""
        options = provider._get_crawl_options()
        
        # Test default options
        expected_options = {
            "word_threshold": 5,
            "only_text": False,
            "remove_overlay_elements": True,
            "process_iframes": True,
            "delay_before_return_html": 2.0,
            "override_navigator": False,
        }
        
        assert options == expected_options

    def test_crawl_options_with_custom_config(self, custom_config):
        """Test crawl options with custom configuration."""
        provider = Crawl4AIProvider(custom_config)
        options = provider._get_crawl_options()
        
        # Should still use default crawl options regardless of config
        expected_options = {
            "word_threshold": 5,
            "only_text": False,
            "remove_overlay_elements": True,
            "process_iframes": True,
            "delay_before_return_html": 2.0,
            "override_navigator": False,
        }
        
        assert options == expected_options

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawler_cleanup(self, mock_crawler_class, provider, mock_crawler):
        """Test that crawler is properly cleaned up."""
        mock_crawler_class.return_value = mock_crawler
        
        # Mock successful crawl
        mock_result = Mock()
        mock_result.success = True
        mock_result.cleaned_html = "<html></html>"
        mock_result.markdown = "content"
        mock_result.extracted_content = "content"
        mock_result.links = {"internal": [], "external": []}
        mock_result.metadata = {}
        
        mock_crawler.arun.return_value = mock_result

        await provider.crawl("http://example.com")

        # Verify cleanup was called
        mock_crawler.aclose.assert_called_once()

    @patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler")
    async def test_crawler_cleanup_on_exception(self, mock_crawler_class, provider, mock_crawler):
        """Test that crawler is cleaned up even when exceptions occur."""
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.arun.side_effect = Exception("Network error")

        with pytest.raises(CrawlError):
            await provider.crawl("http://example.com")

        # Verify cleanup was still called
        mock_crawler.aclose.assert_called_once()

    def test_viewport_configuration_integration(self):
        """Test that viewport configuration is properly handled."""
        # Test different viewport configurations
        configs = [
            {"width": 1920, "height": 1080},
            {"width": 1280, "height": 720},
            {"width": 800, "height": 600},
        ]
        
        for viewport in configs:
            config = Crawl4AIConfig(viewport=viewport)
            provider = Crawl4AIProvider(config)
            assert provider.config.viewport == viewport

    def test_browser_type_validation(self):
        """Test that different browser types are properly handled."""
        browser_types = ["chromium", "firefox", "webkit"]
        
        for browser_type in browser_types:
            config = Crawl4AIConfig(browser_type=browser_type)
            provider = Crawl4AIProvider(config)
            assert provider.config.browser_type == browser_type

    async def test_error_handling_robustness(self, provider):
        """Test comprehensive error handling scenarios."""
        with patch("src.services.crawling.crawl4ai_provider.AsyncWebCrawler") as mock_crawler_class:
            mock_crawler = Mock()
            mock_crawler_class.return_value = mock_crawler
            
            # Test various error scenarios
            error_scenarios = [
                Exception("Connection timeout"),
                Exception("DNS resolution failed"),
                Exception("SSL certificate error"),
                Exception("HTTP 404 Not Found"),
            ]
            
            for error in error_scenarios:
                mock_crawler.arun.side_effect = error
                
                with pytest.raises(CrawlError):
                    await provider.crawl("http://example.com")
                
                # Ensure cleanup is always called
                mock_crawler.aclose.assert_called()

    def test_configuration_field_validation(self):
        """Test that configuration fields are properly validated."""
        # Test valid configurations
        valid_configs = [
            {"max_concurrent_crawls": 1},
            {"max_concurrent_crawls": 25},
            {"max_concurrent_crawls": 50},
            {"page_timeout": 10.0},
            {"page_timeout": 120.0},
        ]
        
        for config_data in valid_configs:
            config = Crawl4AIConfig(**config_data)
            provider = Crawl4AIProvider(config)
            assert provider.config is not None

        # Test invalid configurations
        with pytest.raises(Exception):  # ValidationError from Pydantic
            Crawl4AIConfig(max_concurrent_crawls=0)
            
        with pytest.raises(Exception):  # ValidationError from Pydantic
            Crawl4AIConfig(max_concurrent_crawls=100)  # > 50 limit
            
        with pytest.raises(Exception):  # ValidationError from Pydantic
            Crawl4AIConfig(page_timeout=0)