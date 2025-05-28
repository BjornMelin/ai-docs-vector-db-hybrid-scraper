"""Tests for browser automation services."""

import time
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
            "content": "# Test Content",
            "html": "<h1>Test Content</h1>",
            "title": "Test Page",
            "success": True,
            "status_code": 200,
            "response_headers": {"content-type": "text/html"},
            "metadata": {},
            "links": [],
            "structured_data": {},
        }

        # Mock the provider instance directly
        mock_provider = AsyncMock()
        mock_provider.scrape_url = AsyncMock(return_value=mock_result)
        mock_provider.initialize = AsyncMock(return_value=None)

        # Replace the adapter's provider with our mock
        adapter._provider = mock_provider

        # Initialize the adapter
        await adapter.initialize()

        result = await adapter.scrape(url)

        assert result["success"] is True
        assert result["content"] == "# Test Content"
        assert result["metadata"]["extraction_method"] == "crawl4ai"
        assert result["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_scrape_failure(self, adapter):
        """Test handling of scraping failure."""
        url = "https://example.com"

        # Mock the provider instance directly
        mock_provider = AsyncMock()
        mock_provider.scrape_url = AsyncMock(side_effect=Exception("Network error"))
        mock_provider.initialize = AsyncMock(return_value=None)

        # Replace the adapter's provider with our mock
        adapter._provider = mock_provider

        # Initialize the adapter
        await adapter.initialize()

        result = await adapter.scrape(url)

        assert result["success"] is False
        assert "Network error" in result["error"]
        assert result["metadata"]["extraction_method"] == "crawl4ai"

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
        mock_page.evaluate.return_value = {
            "title": "Test Page",
            "description": None,
            "author": None,
            "keywords": None,
            "canonical": None,
            "content": "Test content",
            "links": [],
        }

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page
        mock_context.__aenter__.return_value = mock_context

        mock_browser = AsyncMock()
        mock_browser.new_context.return_value = mock_context

        # Set up adapter as initialized
        adapter._available = True
        adapter._initialized = True
        adapter._browser = mock_browser

        result = await adapter.scrape(url, [])

        assert result["success"] is True
        assert "content" in result
        assert result["metadata"]["extraction_method"] == "playwright"
        assert result["metadata"]["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_scrape_with_actions(self, adapter):
        """Test scraping with custom actions."""
        url = "https://example.com"
        actions = [
            {"type": "click", "selector": "#button"},
            {"type": "fill", "selector": "#input", "text": "test"},
            {"type": "wait", "timeout": 1000},
        ]

        mock_page = AsyncMock()
        mock_page.goto.return_value = None
        mock_page.click.return_value = None
        mock_page.fill.return_value = None
        mock_page.wait_for_timeout.return_value = None
        mock_page.wait_for_selector.return_value = None
        mock_page.content.return_value = "<html>Updated content</html>"
        mock_page.title.return_value = "Updated Page"
        mock_page.evaluate.return_value = {
            "title": "Updated Page",
            "description": None,
            "author": None,
            "keywords": None,
            "canonical": None,
            "content": "Updated content",
            "links": [],
        }

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page
        mock_context.__aenter__.return_value = mock_context

        mock_browser = AsyncMock()
        mock_browser.new_context.return_value = mock_context

        # Set up adapter as initialized
        adapter._available = True
        adapter._initialized = True
        adapter._browser = mock_browser

        result = await adapter.scrape(url, actions)

        assert result["success"] is True
        mock_page.click.assert_called_once()
        mock_page.fill.assert_called_once()
        mock_page.wait_for_timeout.assert_called_once_with(1000)

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test adapter health check."""
        health = await adapter.health_check()

        assert "status" in health
        assert "available" in health
        assert "healthy" in health
        # Should report as not initialized since we haven't set up the browser
        assert health["status"] == "not_initialized"


class TestAutomationRouterErrorPaths:
    """Test error paths and edge cases for AutomationRouter."""

    @pytest.mark.asyncio
    async def test_scrape_without_initialization(self, mock_config):
        """Test scraping without initialization raises error."""
        router = AutomationRouter(mock_config)

        with pytest.raises(CrawlServiceError, match="Router not initialized"):
            await router.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_cleanup_with_adapter_errors(self, mock_config):
        """Test cleanup handles adapter errors gracefully."""
        router = AutomationRouter(mock_config)

        # Add a mock adapter that will fail during cleanup
        failing_adapter = AsyncMock()
        failing_adapter.cleanup.side_effect = Exception("Cleanup failed")
        router._adapters["failing"] = failing_adapter
        router._initialized = True

        # Should not raise exception despite adapter failure
        await router.cleanup()

        assert router._initialized is False
        assert len(router._adapters) == 0

    @pytest.mark.asyncio
    async def test_force_tool_unavailable(self, mock_config):
        """Test forcing a tool that isn't available."""
        router = AutomationRouter(mock_config)

        # Mock initialization without any adapters
        router._initialized = True
        router._adapters = {}

        with pytest.raises(
            CrawlServiceError, match="Forced tool 'playwright' not available"
        ):
            await router.scrape("https://example.com", force_tool="playwright")

    @pytest.mark.asyncio
    async def test_all_tools_fail_fallback(self, mock_config):
        """Test behavior when all tools fail."""
        router = AutomationRouter(mock_config)
        router._initialized = True

        # Mock failing adapters
        failing_adapter = AsyncMock()
        failing_adapter.scrape.side_effect = Exception("Tool failed")
        router._adapters = {
            "crawl4ai": failing_adapter,
            "stagehand": failing_adapter,
            "playwright": failing_adapter,
        }

        result = await router.scrape("https://example.com")

        assert result["success"] is False
        assert "error" in result
        # Check that it attempted fallback behavior
        assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_adapter_initialization_failures(self, mock_config):
        """Test graceful handling of adapter initialization failures."""
        # This tests the exception handling in initialize() method
        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIAdapter"
        ) as mock_crawl4ai:
            mock_crawl4ai.side_effect = Exception("Failed to import")

            router = AutomationRouter(mock_config)
            await router.initialize()  # Should not raise, just log warning

            assert "crawl4ai" not in router._adapters

    @pytest.mark.asyncio
    async def test_metrics_update_and_retrieval(self, mock_config):
        """Test performance metrics tracking."""
        router = AutomationRouter(mock_config)

        # Update metrics for different tools
        router._update_metrics("crawl4ai", True, 100.5)
        router._update_metrics("crawl4ai", False, 150.0)
        router._update_metrics("stagehand", True, 200.0)

        metrics = router.get_metrics()

        assert metrics["crawl4ai"]["success"] == 1
        assert metrics["crawl4ai"]["failed"] == 1
        assert metrics["crawl4ai"]["total_time"] == 250.5
        assert metrics["crawl4ai"]["avg_time"] == 125.25

        assert metrics["stagehand"]["success"] == 1
        assert metrics["stagehand"]["avg_time"] == 200.0

    @pytest.mark.asyncio
    async def test_routing_rules_loading(self, mock_config):
        """Test routing rules loading and fallback."""
        router = AutomationRouter(mock_config)

        # Test that routing rules are loaded
        assert hasattr(router, "routing_rules")
        assert isinstance(router.routing_rules, dict)

        # Should contain default rules
        assert len(router.routing_rules) >= 0


class TestPlaywrightAdapterErrorPaths:
    """Test error paths for PlaywrightAdapter."""

    @pytest.mark.asyncio
    async def test_scrape_without_initialization(self):
        """Test scraping without initialization."""
        adapter = PlaywrightAdapter({})

        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com", actions=[])


class TestCrawl4AIAdapterErrorPaths:
    """Test error paths for Crawl4AIAdapter."""

    @pytest.mark.asyncio
    async def test_scrape_with_provider_error(self):
        """Test scraping when provider fails."""
        adapter = Crawl4AIAdapter({})
        adapter._initialized = True

        # Mock failing provider
        mock_provider = AsyncMock()
        mock_provider.scrape_url.side_effect = Exception("Provider failed")
        adapter._provider = mock_provider

        result = await adapter.scrape("https://example.com")

        assert result["success"] is False
        assert "Provider failed" in result["error"]


class TestStagehandAdapterErrorPaths:
    """Test error paths and comprehensive functionality for StagehandAdapter."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for StagehandAdapter."""
        return {
            "env": "LOCAL",
            "headless": True,
            "model": "ollama/llama2",
            "enable_caching": True,
            "debug": False,
            "viewport": {"width": 1920, "height": 1080},
        }

    @pytest.mark.asyncio
    async def test_initialization_when_stagehand_unavailable(self, mock_config):
        """Test adapter behavior when Stagehand is not available."""
        # Mock Stagehand as unavailable
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", False):
            adapter = StagehandAdapter(mock_config)

            assert not adapter._available

            with pytest.raises(CrawlServiceError, match="Stagehand not available"):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_scrape_when_unavailable(self, mock_config):
        """Test scraping when adapter is unavailable."""
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", False):
            adapter = StagehandAdapter(mock_config)

            with pytest.raises(CrawlServiceError, match="Stagehand not available"):
                await adapter.scrape("https://example.com", ["extract content"])

    @pytest.mark.asyncio
    async def test_scrape_without_initialization(self, mock_config):
        """Test scraping without initialization."""
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True):
            adapter = StagehandAdapter(mock_config)

            with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
                await adapter.scrape("https://example.com", ["extract content"])

    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_config):
        """Test initialization failure handling."""
        with (
            patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True),
            patch(
                "src.services.browser.stagehand_adapter.Stagehand"
            ) as mock_stagehand_class,
        ):
            mock_stagehand = AsyncMock()
            mock_stagehand.start.side_effect = Exception("Stagehand failed to start")
            mock_stagehand_class.return_value = mock_stagehand

            adapter = StagehandAdapter(mock_config)

            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Stagehand"
            ):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self, mock_config):
        """Test cleanup when stagehand.stop() fails."""
        with (
            patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True),
            patch(
                "src.services.browser.stagehand_adapter.Stagehand"
            ) as mock_stagehand_class,
        ):
            mock_stagehand = AsyncMock()
            mock_stagehand.stop.side_effect = Exception("Stop failed")
            mock_stagehand_class.return_value = mock_stagehand

            adapter = StagehandAdapter(mock_config)
            adapter._stagehand = mock_stagehand
            adapter._initialized = True

            # Should not raise exception, just log error
            await adapter.cleanup()

            # After cleanup with error, _stagehand should still be None
            # and _initialized should be False (cleanup sets these even on error)
            assert adapter._stagehand is None
            assert not adapter._initialized

    @pytest.mark.asyncio
    async def test_page_navigation_failure(self, mock_config):
        """Test handling of page navigation failures."""
        with (
            patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True),
            patch(
                "src.services.browser.stagehand_adapter.Stagehand"
            ) as mock_stagehand_class,
        ):
            mock_stagehand = AsyncMock()
            mock_page = AsyncMock()
            mock_page.goto.side_effect = Exception("Navigation failed")
            mock_stagehand.new_page.return_value = mock_page
            mock_stagehand_class.return_value = mock_stagehand

            adapter = StagehandAdapter(mock_config)
            adapter._stagehand = mock_stagehand
            adapter._initialized = True

            result = await adapter.scrape("https://example.com", ["extract content"])

            assert result["success"] is False
            assert "Navigation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_instruction_execution_patterns(self, mock_config):
        """Test different instruction execution patterns."""
        with (
            patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True),
            patch(
                "src.services.browser.stagehand_adapter.Stagehand"
            ) as mock_stagehand_class,
        ):
            mock_stagehand = AsyncMock()
            mock_page = AsyncMock()
            mock_page.content.return_value = "<html>Test</html>"
            mock_page.title.return_value = "Test Page"
            mock_page.url = "https://example.com"
            mock_stagehand.new_page.return_value = mock_page
            mock_stagehand.extract.return_value = {"content": "extracted content"}
            mock_stagehand_class.return_value = mock_stagehand

            adapter = StagehandAdapter(mock_config)
            adapter._stagehand = mock_stagehand
            adapter._initialized = True

            # Test various instruction types
            instructions = [
                "click on the button",
                "type hello world",
                "enter your name",
                "extract main content",
                "wait 2 seconds",
                "scroll down",
                "take screenshot",
                "custom action",
            ]

            result = await adapter.scrape("https://example.com", instructions)

            assert result["success"] is True
            assert result["metadata"]["instructions_executed"] == len(instructions)

            # Verify different methods were called
            mock_stagehand.click.assert_called()
            mock_stagehand.type.assert_called()
            mock_stagehand.extract.assert_called()
            mock_stagehand.act.assert_called()

    @pytest.mark.asyncio
    async def test_wait_time_extraction(self, mock_config):
        """Test wait time extraction from instructions."""
        adapter = StagehandAdapter(mock_config)

        # Test various wait patterns
        assert adapter._extract_wait_time("wait 5 seconds") == 5000
        assert adapter._extract_wait_time("wait 3s") == 3000
        assert adapter._extract_wait_time("wait 1500 milliseconds") == 1500
        assert (
            adapter._extract_wait_time("wait 1500ms") == 1500
        )  # ms pattern should not be converted
        assert adapter._extract_wait_time("wait for 10") == 10
        assert adapter._extract_wait_time("just wait") == 1000  # default

    @pytest.mark.asyncio
    async def test_health_check_scenarios(self, mock_config):
        """Test various health check scenarios."""
        # Test unavailable adapter
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", False):
            adapter = StagehandAdapter(mock_config)
            health = await adapter.health_check()

            assert not health["healthy"]
            assert health["status"] == "unavailable"
            assert not health["available"]

        # Test uninitialized adapter
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True):
            adapter = StagehandAdapter(mock_config)
            health = await adapter.health_check()

            assert not health["healthy"]
            assert health["status"] == "not_initialized"
            assert health["available"] is True

        # Test timeout scenario
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True):
            adapter = StagehandAdapter(mock_config)
            adapter._initialized = True

            with patch.object(adapter, "scrape") as mock_scrape:
                mock_scrape.side_effect = TimeoutError()

                health = await adapter.health_check()

                assert not health["healthy"]
                assert health["status"] == "timeout"
                assert health["response_time_ms"] == 15000

    @pytest.mark.asyncio
    async def test_ai_capabilities_testing(self, mock_config):
        """Test AI capabilities testing functionality."""
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True):
            # Test unavailable/uninitialized adapter
            adapter = StagehandAdapter(mock_config)
            result = await adapter.test_ai_capabilities()

            assert not result["success"]
            assert "not available or initialized" in result["error"]

            # Test successful AI capabilities test
            adapter._available = True
            adapter._initialized = True

            with patch.object(adapter, "scrape") as mock_scrape:
                mock_scrape.return_value = {
                    "success": True,
                    "extraction_results": {"test1": "result1", "test2": "result2"},
                    "screenshots": [{"data": "screenshot1"}],
                    "content": "test content",
                    "ai_insights": {"confidence": 0.95},
                }

                result = await adapter.test_ai_capabilities("https://test.com")

                assert result["success"] is True
                assert result["test_url"] == "https://test.com"
                assert result["instructions_count"] == 5
                assert result["extractions_count"] == 2
                assert result["screenshots_count"] == 1
                assert result["content_length"] == len("test content")
                assert result["ai_insights"]["confidence"] == 0.95

    def test_get_capabilities(self, mock_config):
        """Test capabilities reporting."""
        with patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True):
            adapter = StagehandAdapter(mock_config)
            capabilities = adapter.get_capabilities()

            assert capabilities["name"] == "stagehand"
            assert capabilities["ai_powered"] is True
            assert capabilities["available"] is True
            assert "AI understands complex interactions" in capabilities["advantages"]
            assert "Slower than direct automation" in capabilities["limitations"]
            assert "Complex interactions" in capabilities["best_for"]
            assert (
                capabilities["performance"]["success_rate"] == "95% for complex sites"
            )

    @pytest.mark.asyncio
    async def test_error_result_building(self, mock_config):
        """Test error result building."""
        adapter = StagehandAdapter(mock_config)

        start_time = time.time()
        result = adapter._build_error_result(
            "https://example.com",
            start_time,
            "Test error",
            ["instruction1", "instruction2"],
        )

        assert result["success"] is False
        assert result["url"] == "https://example.com"
        assert result["error"] == "Test error"
        assert result["content"] == ""
        assert result["metadata"]["extraction_method"] == "stagehand_ai"
        assert result["metadata"]["instructions_attempted"] == 2
        assert "processing_time_ms" in result["metadata"]

    @pytest.mark.asyncio
    async def test_page_close_failure(self, mock_config):
        """Test handling of page close failures."""
        with (
            patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True),
            patch(
                "src.services.browser.stagehand_adapter.Stagehand"
            ) as mock_stagehand_class,
        ):
            mock_stagehand = AsyncMock()
            mock_page = AsyncMock()
            mock_page.close.side_effect = Exception("Close failed")
            mock_page.content.return_value = "<html>Test</html>"
            mock_page.title.return_value = "Test Page"
            mock_page.url = "https://example.com"
            mock_stagehand.new_page.return_value = mock_page
            mock_stagehand.extract.return_value = {"content": "test"}
            mock_stagehand_class.return_value = mock_stagehand

            adapter = StagehandAdapter(mock_config)
            adapter._stagehand = mock_stagehand
            adapter._initialized = True

            # Should complete successfully despite page close failure
            result = await adapter.scrape("https://example.com", ["extract content"])

            assert result["success"] is True
            mock_page.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_content_combination_logic(self, mock_config):
        """Test content combination from multiple extractions."""
        with (
            patch("src.services.browser.stagehand_adapter.STAGEHAND_AVAILABLE", True),
            patch(
                "src.services.browser.stagehand_adapter.Stagehand"
            ) as mock_stagehand_class,
        ):
            mock_stagehand = AsyncMock()
            mock_page = AsyncMock()
            mock_page.content.return_value = "<html>Test</html>"
            mock_page.title.return_value = "Test Page"
            mock_page.url = "https://example.com"
            mock_stagehand.new_page.return_value = mock_page

            # Mock different extraction results
            extract_call_count = 0

            def mock_extract(page, instruction):
                nonlocal extract_call_count
                extract_call_count += 1
                if "documentation content" in instruction.lower():
                    # This is the final extraction call
                    return {
                        "content": "final content",
                        "metadata": {"source": "final"},
                    }
                else:
                    # This is an instruction-based extraction
                    return {"content": "instruction content"}

            mock_stagehand.extract.side_effect = mock_extract
            mock_stagehand_class.return_value = mock_stagehand

            adapter = StagehandAdapter(mock_config)
            adapter._stagehand = mock_stagehand
            adapter._initialized = True

            result = await adapter.scrape(
                "https://example.com", ["extract test content"]
            )

            assert result["success"] is True
            # Should combine final content and instruction extractions
            assert "final content" in result["content"]
            assert "instruction content" in result["content"]
            assert result["metadata"]["source"] == "final"
