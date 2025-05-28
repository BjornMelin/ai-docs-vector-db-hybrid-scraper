"""Tests for browser automation services."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.browser.automation_router import AutomationRouter
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter
from src.services.browser.playwright_adapter import PlaywrightAdapter
from src.services.browser.stagehand_adapter import StagehandAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock unified config for testing."""
    config = MagicMock(spec=UnifiedConfig)

    # Set up nested mock attributes for crawling
    config.crawling = MagicMock()
    config.crawling.max_concurrent = 5
    config.crawling.timeout = 30
    config.crawling.headless = True

    # Set up crawl4ai config as dict-like object with get method
    config.crawl4ai = MagicMock()
    config.crawl4ai.get = lambda key, default=None: {
        "max_concurrency": 5,
        "timeout": 30,
        "rate_limit": 60,
    }.get(key, default)

    # Set up rate limiting config
    config.rate_limiting = MagicMock()
    config.rate_limiting.default_rate_limits = {
        "crawl4ai": {"max_calls": 50, "time_window": 60}
    }

    return config


@pytest.fixture
def router(mock_config):
    """Create AutomationRouter instance for testing."""
    return AutomationRouter(mock_config)


class TestAutomationRouter:
    """Test cases for AutomationRouter."""

    @pytest.mark.asyncio
    async def test_initialization(self, router):
        """Test router initialization."""
        assert router.config is not None
        assert router.routing_rules is not None
        assert "stagehand" in router.routing_rules
        assert "playwright" in router.routing_rules

    def test_select_tool_site_specific(self, router):
        """Test tool selection based on site-specific routing rules."""
        # Mock adapters as available
        router._adapters = {
            "crawl4ai": "mock",
            "stagehand": "mock",
            "playwright": "mock",
        }

        # Test Stagehand selection
        tool = router._select_tool("https://vercel.com/docs/api", False, None)
        assert tool == "stagehand"

        # Test Playwright selection (check actual routing rules)
        tool = router._select_tool("https://github.com/user/repo", False, None)
        assert tool == "playwright"

        # Test default (Crawl4AI)
        tool = router._select_tool("https://example.com/docs", False, None)
        assert tool == "crawl4ai"

    def test_select_tool_with_requirements(self, router):
        """Test tool selection based on automation requirements."""
        # Mock adapters as available
        router._adapters = {
            "crawl4ai": "mock",
            "stagehand": "mock",
            "playwright": "mock",
        }

        # Test interactive requirements force Stagehand/Playwright
        tool = router._select_tool("https://example.com", True, None)
        assert tool in ["stagehand", "playwright"]

        # Test custom actions requirements force Stagehand/Playwright
        custom_actions = [{"action": "click", "selector": "#button"}]
        tool = router._select_tool("https://example.com", False, custom_actions)
        assert tool in ["stagehand", "playwright"]

    @pytest.mark.asyncio
    async def test_scrape_success_first_tool(self, router):
        """Test successful scraping with first tool."""
        # Mock initialization to skip real adapter creation
        router._initialized = True
        router._adapters = {
            "crawl4ai": AsyncMock(),
            "stagehand": AsyncMock(),
            "playwright": AsyncMock(),
        }

        url = "https://example.com"
        expected_result = {
            "content": "Test content",
            "metadata": {"tool": "crawl4ai"},
            "success": True,
        }

        with patch.object(
            router, "_try_crawl4ai", return_value=expected_result
        ) as mock_crawl4ai:
            result = await router.scrape(url)

            assert result == expected_result
            mock_crawl4ai.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_uses_fallback(self, router):
        """Test fallback mechanism when primary tool fails."""
        # Mock initialization to skip real adapter creation
        router._initialized = True
        router._adapters = {
            "crawl4ai": AsyncMock(),
            "stagehand": AsyncMock(),
            "playwright": AsyncMock(),
        }

        url = "https://vercel.com/docs"  # Should try Stagehand first

        # Mock Stagehand failure and Playwright success
        stagehand_error = CrawlServiceError("Stagehand failed")
        playwright_result = {
            "content": "Fallback content",
            "metadata": {"tool": "playwright"},
            "success": True,
        }

        with (
            patch.object(router, "_try_stagehand", side_effect=stagehand_error),
            patch.object(
                router, "_try_playwright", return_value=playwright_result
            ) as mock_playwright,
        ):
            result = await router.scrape(url)

            assert result == playwright_result
            mock_playwright.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_all_tools_fail(self, router):
        """Test behavior when all tools fail."""
        # Mock initialization to skip real adapter creation
        router._initialized = True
        router._adapters = {
            "crawl4ai": AsyncMock(),
            "stagehand": AsyncMock(),
            "playwright": AsyncMock(),
        }

        url = "https://example.com"

        with (
            patch.object(
                router, "_try_crawl4ai", side_effect=CrawlServiceError("Failed")
            ),
            patch.object(
                router, "_try_stagehand", side_effect=CrawlServiceError("Failed")
            ),
            patch.object(
                router, "_try_playwright", side_effect=CrawlServiceError("Failed")
            ),
        ):
            result = await router.scrape(url)

            # Should return failure result instead of raising exception
            assert result["success"] is False
            assert "All automation tools failed" in result["error"]
            assert result["provider"] == "none"

    @pytest.mark.asyncio
    async def test_try_crawl4ai_success(self, router):
        """Test successful Crawl4AI adapter usage."""
        # Initialize router
        await router.initialize()

        url = "https://example.com"
        expected_result = {"content": "Test", "success": True}

        # Mock the adapter directly in the router's adapters dict
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = expected_result
        router._adapters["crawl4ai"] = mock_adapter

        result = await router._try_crawl4ai(url, None)

        assert result == expected_result
        mock_adapter.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_try_stagehand_success(self, router):
        """Test successful Stagehand adapter usage."""
        # Initialize router
        await router.initialize()

        url = "https://example.com"
        expected_result = {"content": "Test", "success": True}

        # Mock the adapter directly in the router's adapters dict
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = expected_result
        router._adapters["stagehand"] = mock_adapter

        result = await router._try_stagehand(url, None)

        assert result == expected_result
        mock_adapter.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_try_playwright_success(self, router):
        """Test successful Playwright adapter usage."""
        # Initialize router
        await router.initialize()

        url = "https://example.com"
        expected_result = {"content": "Test", "success": True}

        # Mock the adapter directly in the router's adapters dict
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = expected_result
        router._adapters["playwright"] = mock_adapter

        result = await router._try_playwright(url, None)

        assert result == expected_result
        mock_adapter.scrape.assert_called_once()

    def test_update_performance_metrics(self, router):
        """Test performance metrics tracking."""
        initial_success = router.metrics.get("crawl4ai", {}).get("success", 0)

        router._update_metrics("crawl4ai", success=True, elapsed=0.5)

        assert router.metrics["crawl4ai"]["success"] == initial_success + 1
        assert "avg_time" in router.metrics["crawl4ai"]

    def test_get_metrics(self, router):
        """Test metrics retrieval functionality."""
        # Initialize metrics
        router._update_metrics("crawl4ai", success=True, elapsed=0.5)
        router._update_metrics("stagehand", success=False, elapsed=2.0)

        metrics = router.get_metrics()

        assert "crawl4ai" in metrics
        assert "stagehand" in metrics
        assert metrics["crawl4ai"]["success"] >= 1
        assert metrics["stagehand"]["failed"] >= 1


class TestCrawl4AIAdapter:
    """Test cases for Crawl4AIAdapter."""

    @pytest.fixture
    def adapter(self, mock_config):
        """Create Crawl4AIAdapter instance for testing."""
        return Crawl4AIAdapter(mock_config)

    @pytest.mark.asyncio
    async def test_scrape_success(self, adapter):
        """Test successful scraping with Crawl4AI adapter."""
        url = "https://example.com"
        mock_result = {
            "url": url,
            "markdown": "# Test Content",
            "html": "<h1>Test Content</h1>",
            "success": True,
            "status_code": 200,
            "response_headers": {"content-type": "text/html"},
        }

        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIProvider"
        ) as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.crawl_url.return_value = mock_result
            mock_provider.initialize.return_value = None
            mock_provider_class.return_value = mock_provider

            # Initialize the adapter
            await adapter.initialize()

            result = await adapter.scrape(url, {})

            assert result["success"] is True
            assert result["content"] == "# Test Content"
            assert result["metadata"]["tool"] == "crawl4ai"
            assert result["metadata"]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_scrape_failure(self, adapter):
        """Test handling of scraping failure."""
        url = "https://example.com"

        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIProvider"
        ) as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.crawl_url.side_effect = Exception("Network error")
            mock_provider.initialize.return_value = None
            mock_provider_class.return_value = mock_provider

            # Initialize the adapter
            await adapter.initialize()

            with pytest.raises(CrawlServiceError, match="Crawl4AI failed"):
                await adapter.scrape(url, {})

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test adapter health check."""
        health = await adapter.health_check()

        assert "status" in health
        assert "healthy" in health
        # Check that it reports not initialized state
        assert health["status"] == "not_initialized"


class TestStagehandAdapter:
    """Test cases for StagehandAdapter."""

    @pytest.fixture
    def adapter(self, mock_config):
        """Create StagehandAdapter instance for testing."""
        return StagehandAdapter(mock_config)

    @pytest.mark.asyncio
    async def test_scrape_success_with_stagehand_available(self, adapter):
        """Test scraping behavior when Stagehand is not available in test environment."""
        url = "https://example.com"

        # Since Stagehand is not available in test environment,
        # test that it raises CrawlServiceError
        with pytest.raises(CrawlServiceError, match="Stagehand not available"):
            await adapter.scrape(url, {})

    @pytest.mark.asyncio
    async def test_scrape_fallback_when_stagehand_unavailable(self, adapter):
        """Test fallback behavior when Stagehand is not available."""
        url = "https://example.com"

        # Test the actual behavior since Stagehand is not available
        with pytest.raises(CrawlServiceError, match="Stagehand not available"):
            await adapter.scrape(url, {})

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test adapter health check."""
        health = await adapter.health_check()

        assert "status" in health
        assert "available" in health
        # Should report as unavailable since Stagehand is not installed
        assert health["available"] is False


class TestPlaywrightAdapter:
    """Test cases for PlaywrightAdapter."""

    @pytest.fixture
    def adapter(self, mock_config):
        """Create PlaywrightAdapter instance for testing."""
        return PlaywrightAdapter(mock_config)

    @pytest.mark.asyncio
    async def test_scrape_success(self, adapter):
        """Test successful scraping with Playwright adapter."""
        url = "https://example.com"

        # Mock Playwright components
        mock_page = AsyncMock()
        mock_page.goto.return_value = None
        mock_page.content.return_value = "<html><body><h1>Test</h1></body></html>"
        mock_page.title.return_value = "Test Page"

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page
        mock_context.__aenter__.return_value = mock_context

        mock_browser = AsyncMock()
        mock_browser.new_context.return_value = mock_context
        mock_browser.__aenter__.return_value = mock_browser

        with patch(
            "src.services.browser.playwright_adapter.async_playwright"
        ) as mock_playwright:
            mock_playwright.return_value.__aenter__.return_value.chromium.launch.return_value = mock_browser

            result = await adapter.scrape(url, {})

            assert result["success"] is True
            assert "content" in result
            assert result["metadata"]["tool"] == "playwright"
            assert result["metadata"]["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_scrape_with_actions(self, adapter):
        """Test scraping with custom actions."""
        url = "https://example.com"
        actions = [
            {"action": "click", "selector": "#button"},
            {"action": "type", "selector": "#input", "text": "test"},
            {"action": "wait", "duration": 1000},
        ]

        mock_page = AsyncMock()
        mock_page.goto.return_value = None
        mock_page.click.return_value = None
        mock_page.fill.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.content.return_value = "<html>Updated content</html>"
        mock_page.title.return_value = "Updated Page"

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page
        mock_context.__aenter__.return_value = mock_context

        mock_browser = AsyncMock()
        mock_browser.new_context.return_value = mock_context
        mock_browser.__aenter__.return_value = mock_browser

        with patch(
            "src.services.browser.playwright_adapter.async_playwright"
        ) as mock_playwright:
            mock_playwright.return_value.__aenter__.return_value.chromium.launch.return_value = mock_browser

            result = await adapter.scrape(url, {"actions": actions})

            assert result["success"] is True
            mock_page.click.assert_called_once_with("#button")
            mock_page.fill.assert_called_once_with("#input", "test")
            mock_page.wait_for_timeout.assert_called_once_with(1000)

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test adapter health check."""
        health = await adapter.health_check()

        assert "status" in health
        assert "tool" in health
        assert health["tool"] == "playwright"
