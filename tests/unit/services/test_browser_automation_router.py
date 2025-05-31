"""Comprehensive tests for browser automation router."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.browser.automation_router import AutomationRouter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock unified config for testing."""
    config = MagicMock(spec=UnifiedConfig)

    # Set up nested performance config
    config.performance = MagicMock()
    config.performance.max_concurrent_requests = 5
    config.performance.request_timeout = 30

    return config


@pytest.fixture
def routing_rules():
    """Sample routing rules for testing."""
    return {
        "browser_use": [
            "vercel.com",
            "clerk.com",
            "supabase.com",
            "react.dev",
            "nextjs.org",
            "docs.anthropic.com",
        ],
        "playwright": [
            "github.com",
            "stackoverflow.com",
            "discord.com",
            "slack.com",
            "notion.so",
        ],
    }


@pytest.fixture
def router(mock_config):
    """Create AutomationRouter instance for testing."""
    return AutomationRouter(mock_config)


class TestAutomationRouterInitialization:
    """Test router initialization and configuration."""

    def test_router_initialization(self, router, mock_config):
        """Test basic router initialization."""
        assert router.config == mock_config
        assert router.logger is not None
        assert isinstance(router._adapters, dict)
        assert router._initialized is False
        assert router.routing_rules is not None
        assert isinstance(router.metrics, dict)

    def test_metrics_initialization(self, router):
        """Test metrics dictionary initialization."""
        expected_tools = ["crawl4ai", "browser_use", "playwright"]
        for tool in expected_tools:
            assert tool in router.metrics
            assert router.metrics[tool]["success"] == 0
            assert router.metrics[tool]["failed"] == 0
            assert router.metrics[tool]["avg_time"] == 0.0
            assert router.metrics[tool]["total_time"] == 0.0

    def test_load_routing_rules_from_file(self, router, routing_rules, tmp_path):
        """Test loading routing rules from configuration file."""
        # Create a temporary config file
        config_file = tmp_path / "browser-routing-rules.json"
        config_data = {"routing_rules": routing_rules}
        config_file.write_text(json.dumps(config_data))

        with patch.object(Path, "__new__", return_value=config_file.parent):
            rules = router._load_routing_rules()

        # Should load rules from file
        assert "browser_use" in rules
        assert "playwright" in rules

    def test_load_routing_rules_file_not_found(self, router):
        """Test fallback when routing rules file not found."""
        with patch("pathlib.Path.exists", return_value=False):
            rules = router._load_routing_rules()

        # Should return default rules
        assert isinstance(rules, dict)
        assert "browser_use" in rules
        assert "playwright" in rules

    def test_load_routing_rules_file_error(self, router):
        """Test fallback when routing rules file has error."""
        with patch("builtins.open", side_effect=Exception("File error")):
            rules = router._load_routing_rules()

        # Should return default rules
        assert isinstance(rules, dict)

    def test_get_default_routing_rules(self, router):
        """Test default routing rules structure."""
        rules = router._get_default_routing_rules()

        assert "browser_use" in rules
        assert "playwright" in rules
        assert isinstance(rules["browser_use"], list)
        assert isinstance(rules["playwright"], list)
        assert len(rules["browser_use"]) > 0
        assert len(rules["playwright"]) > 0


class TestAutomationRouterInitializeCleanup:
    """Test router initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_success_all_adapters(self, router):
        """Test successful initialization of all adapters."""
        # Mock adapter classes
        mock_crawl4ai = AsyncMock()
        mock_browser_use = AsyncMock()
        mock_playwright = AsyncMock()

        with (
            patch(
                "src.services.browser.automation_router.Crawl4AIAdapter",
                return_value=mock_crawl4ai,
            ),
            patch(
                "src.services.browser.automation_router.BrowserUseAdapter",
                return_value=mock_browser_use,
            ),
            patch(
                "src.services.browser.automation_router.PlaywrightAdapter",
                return_value=mock_playwright,
            ),
        ):
            await router.initialize()

        assert router._initialized is True
        assert len(router._adapters) == 3
        assert "crawl4ai" in router._adapters
        assert "browser_use" in router._adapters
        assert "playwright" in router._adapters

        # Verify all adapters were initialized
        mock_crawl4ai.initialize.assert_called_once()
        mock_browser_use.initialize.assert_called_once()
        mock_playwright.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_partial_adapter_failure(self, router):
        """Test initialization when some adapters fail."""
        mock_crawl4ai = AsyncMock()

        with (
            patch(
                "src.services.browser.automation_router.Crawl4AIAdapter",
                return_value=mock_crawl4ai,
            ),
            patch(
                "src.services.browser.automation_router.BrowserUseAdapter",
                side_effect=Exception("Import failed"),
            ),
            patch(
                "src.services.browser.automation_router.PlaywrightAdapter",
                side_effect=Exception("Import failed"),
            ),
        ):
            await router.initialize()

        assert router._initialized is True
        assert len(router._adapters) == 1
        assert "crawl4ai" in router._adapters
        assert "browser_use" not in router._adapters
        assert "playwright" not in router._adapters

    @pytest.mark.asyncio
    async def test_initialize_no_adapters_available(self, router):
        """Test initialization fails when no adapters available."""
        with (
            patch(
                "src.services.browser.automation_router.Crawl4AIAdapter",
                side_effect=Exception("Failed"),
            ),
            patch(
                "src.services.browser.automation_router.BrowserUseAdapter",
                side_effect=Exception("Failed"),
            ),
            patch(
                "src.services.browser.automation_router.PlaywrightAdapter",
                side_effect=Exception("Failed"),
            ),
        ):
            with pytest.raises(
                CrawlServiceError, match="No automation adapters available"
            ):
                await router.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, router):
        """Test initialization is idempotent."""
        mock_adapter = AsyncMock()

        with patch(
            "src.services.browser.automation_router.Crawl4AIAdapter",
            return_value=mock_adapter,
        ):
            await router.initialize()
            first_adapters = router._adapters.copy()

            # Initialize again
            await router.initialize()

        # Should not re-initialize
        assert router._adapters == first_adapters
        mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_all_adapters(self, router):
        """Test cleanup of all adapters."""
        # Set up mock adapters
        mock_adapters = {
            "crawl4ai": AsyncMock(),
            "browser_use": AsyncMock(),
            "playwright": AsyncMock(),
        }
        router._adapters = mock_adapters
        router._initialized = True

        await router.cleanup()

        # Verify all adapters cleaned up
        for adapter in mock_adapters.values():
            adapter.cleanup.assert_called_once()

        assert len(router._adapters) == 0
        assert router._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_with_adapter_errors(self, router):
        """Test cleanup continues even if some adapters fail."""
        # Set up mock adapters with one failing
        mock_adapters = {
            "crawl4ai": AsyncMock(),
            "browser_use": AsyncMock(),
            "playwright": AsyncMock(),
        }
        mock_adapters["browser_use"].cleanup.side_effect = Exception("Cleanup failed")

        router._adapters = mock_adapters
        router._initialized = True

        # Should not raise exception
        await router.cleanup()

        # All adapters should have cleanup attempted
        for adapter in mock_adapters.values():
            adapter.cleanup.assert_called_once()

        assert len(router._adapters) == 0
        assert router._initialized is False


class TestToolSelection:
    """Test tool selection logic."""

    def test_select_tool_site_specific_browser_use(self, router):
        """Test selection based on browser-use routing rules."""
        router._adapters = {
            "crawl4ai": "mock",
            "browser_use": "mock",
            "playwright": "mock",
        }

        urls = [
            "https://vercel.com/docs",
            "https://clerk.com/api",
            "https://supabase.com/docs",
            "https://react.dev/learn",
            "https://nextjs.org/docs",
            "https://docs.anthropic.com/claude",
        ]

        for url in urls:
            tool = router._select_tool(url, False, None)
            assert tool == "browser_use"

    def test_select_tool_site_specific_playwright(self, router):
        """Test selection based on playwright routing rules."""
        router._adapters = {
            "crawl4ai": "mock",
            "browser_use": "mock",
            "playwright": "mock",
        }

        urls = [
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions",
            "https://discord.com/channels",
            "https://slack.com/workspace",
            "https://notion.so/page",
        ]

        for url in urls:
            tool = router._select_tool(url, False, None)
            assert tool == "playwright"

    def test_select_tool_default_crawl4ai(self, router):
        """Test default selection is crawl4ai."""
        router._adapters = {
            "crawl4ai": "mock",
            "browser_use": "mock",
            "playwright": "mock",
        }

        # URLs not in routing rules
        urls = [
            "https://example.com/docs",
            "https://randomsite.org/api",
            "https://unknown.dev/guide",
        ]

        for url in urls:
            tool = router._select_tool(url, False, None)
            assert tool == "crawl4ai"

    def test_select_tool_interaction_required(self, router):
        """Test tool selection when interaction is required."""
        router._adapters = {
            "crawl4ai": "mock",
            "browser_use": "mock",
            "playwright": "mock",
        }

        # Interaction required should prefer browser-use or playwright
        tool = router._select_tool(
            "https://example.com", interaction_required=True, custom_actions=None
        )
        assert tool in ["browser_use", "playwright"]

    def test_select_tool_custom_actions(self, router):
        """Test tool selection with custom actions."""
        router._adapters = {
            "crawl4ai": "mock",
            "browser_use": "mock",
            "playwright": "mock",
        }

        custom_actions = [{"type": "click", "selector": "#button"}]
        tool = router._select_tool("https://example.com", False, custom_actions)
        assert tool in ["browser_use", "playwright"]

    def test_select_tool_js_patterns(self, router):
        """Test JavaScript pattern detection."""
        router._adapters = {"crawl4ai": "mock", "browser_use": "mock"}

        js_urls = [
            "https://example.com/spa/dashboard",
            "https://myreactapp.com/",
            "https://vue-app.io/admin",
            "https://angular-site.com/",
            "https://example.com/app/settings",
        ]

        for url in js_urls:
            tool = router._select_tool(url, False, None)
            assert tool == "browser_use"

    def test_select_tool_missing_preferred_adapter(self, router):
        """Test selection when preferred adapter not available."""
        # Only crawl4ai available
        router._adapters = {"crawl4ai": "mock"}

        # Even though this should use browser-use, it's not available
        tool = router._select_tool("https://vercel.com/docs", False, None)
        assert tool == "crawl4ai"

    def test_get_default_tool_priority(self, router):
        """Test default tool selection priority."""
        # Test with all adapters
        router._adapters = {
            "crawl4ai": "mock",
            "browser_use": "mock",
            "playwright": "mock",
        }
        assert router._get_default_tool() == "crawl4ai"

        # Test without crawl4ai
        router._adapters = {"browser_use": "mock", "playwright": "mock"}
        assert router._get_default_tool() == "playwright"

        # Test with only browser-use
        router._adapters = {"browser_use": "mock"}
        assert router._get_default_tool() == "browser_use"


class TestScrapeOperation:
    """Test main scrape operation."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, router):
        """Test scraping fails when router not initialized."""
        with pytest.raises(CrawlServiceError, match="Router not initialized"):
            await router.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_success_crawl4ai(self, router):
        """Test successful scraping with Crawl4AI."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        expected_result = {
            "content": "Test content",
            "success": True,
            "metadata": {"tool": "crawl4ai"},
        }

        with patch.object(router, "_try_crawl4ai", return_value=expected_result):
            result = await router.scrape("https://example.com")

        assert result["success"] is True
        assert result["content"] == "Test content"
        assert result["provider"] == "crawl4ai"
        assert "automation_time_ms" in result

    @pytest.mark.asyncio
    async def test_scrape_force_tool(self, router):
        """Test scraping with forced tool selection."""
        router._initialized = True
        router._adapters = {
            "crawl4ai": AsyncMock(),
            "browser_use": AsyncMock(),
            "playwright": AsyncMock(),
        }

        expected_result = {"content": "Playwright content", "success": True}

        with patch.object(router, "_try_playwright", return_value=expected_result):
            result = await router.scrape("https://example.com", force_tool="playwright")

        assert result["provider"] == "playwright"

    @pytest.mark.asyncio
    async def test_scrape_force_unavailable_tool(self, router):
        """Test forcing unavailable tool raises error."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        with pytest.raises(
            CrawlServiceError, match="Forced tool 'playwright' not available"
        ):
            await router.scrape("https://example.com", force_tool="playwright")

    @pytest.mark.asyncio
    async def test_scrape_metrics_update_success(self, router):
        """Test metrics are updated on successful scrape."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        initial_success = router.metrics["crawl4ai"]["success"]

        with patch.object(router, "_try_crawl4ai", return_value={"success": True}):
            await router.scrape("https://example.com")

        assert router.metrics["crawl4ai"]["success"] == initial_success + 1
        assert router.metrics["crawl4ai"]["total_time"] > 0
        assert router.metrics["crawl4ai"]["avg_time"] > 0

    @pytest.mark.asyncio
    async def test_scrape_metrics_update_failure(self, router):
        """Test metrics are updated on failed scrape."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        initial_failed = router.metrics["crawl4ai"]["failed"]

        with (
            patch.object(router, "_try_crawl4ai", side_effect=Exception("Failed")),
            patch.object(router, "_fallback_scrape", return_value={"success": False}),
        ):
            await router.scrape("https://example.com")

        assert router.metrics["crawl4ai"]["failed"] == initial_failed + 1


class TestFallbackMechanism:
    """Test fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_to_next_tool(self, router):
        """Test fallback to next tool in hierarchy."""
        router._initialized = True
        router._adapters = {
            "crawl4ai": AsyncMock(),
            "browser_use": AsyncMock(),
            "playwright": AsyncMock(),
        }

        fallback_result = {
            "content": "Fallback content",
            "success": True,
        }

        with (
            patch.object(router, "_try_crawl4ai", side_effect=Exception("Failed")),
            patch.object(router, "_try_browser_use", return_value=fallback_result),
        ):
            result = await router.scrape("https://example.com")

        assert result["success"] is True
        assert result["provider"] == "browser_use"
        assert result["fallback_from"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_fallback_all_tools_fail(self, router):
        """Test result when all tools fail."""
        router._initialized = True
        router._adapters = {
            "crawl4ai": AsyncMock(),
            "browser_use": AsyncMock(),
            "playwright": AsyncMock(),
        }

        with (
            patch.object(router, "_try_crawl4ai", side_effect=Exception("Failed")),
            patch.object(router, "_try_browser_use", side_effect=Exception("Failed")),
            patch.object(router, "_try_playwright", side_effect=Exception("Failed")),
        ):
            result = await router.scrape("https://example.com")

        assert result["success"] is False
        assert "All automation tools failed" in result["error"]
        assert result["provider"] == "none"
        assert "failed_tools" in result

    @pytest.mark.asyncio
    async def test_fallback_order_browser_use(self, router):
        """Test fallback order for browser-use."""
        deployment = await router._fallback_scrape(
            "https://example.com", "browser_use", None, 30000
        )

        # When browser-use fails, should try playwright then crawl4ai
        # (actual order depends on availability)
        assert isinstance(deployment, dict)


class TestAdapterMethods:
    """Test individual adapter try methods."""

    @pytest.mark.asyncio
    async def test_try_crawl4ai_basic(self, router):
        """Test basic Crawl4AI execution."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "content": "Crawl4AI result",
            "success": True,
        }
        router._adapters = {"crawl4ai": mock_adapter}

        result = await router._try_crawl4ai("https://example.com")

        assert result["success"] is True
        mock_adapter.scrape.assert_called_once()

        # Check default wait selector was provided
        call_args = mock_adapter.scrape.call_args
        assert "wait_for_selector" in call_args.kwargs
        assert call_args.kwargs["wait_for_selector"] == ".content, main, article"

    @pytest.mark.asyncio
    async def test_try_crawl4ai_with_custom_actions(self, router):
        """Test Crawl4AI with custom actions converted to JS."""
        mock_adapter = AsyncMock()
        router._adapters = {"crawl4ai": mock_adapter}

        custom_actions = [
            {"type": "click", "selector": "#button"},
            {"type": "wait", "timeout": 1000},
            {"type": "type", "selector": "input", "text": "test"},
        ]

        await router._try_crawl4ai("https://example.com", custom_actions)

        # Verify JS code was generated
        call_args = mock_adapter.scrape.call_args
        js_code = call_args.kwargs["js_code"]
        assert "querySelector('#button')?.click()" in js_code
        assert "setTimeout(r, 1000)" in js_code
        assert "querySelector('input').value = 'test'" in js_code

    @pytest.mark.asyncio
    async def test_try_browser_use_basic(self, router):
        """Test basic browser-use execution."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "content": "Browser-use result",
            "success": True,
        }
        router._adapters = {"browser_use": mock_adapter}

        result = await router._try_browser_use("https://example.com")

        assert result["success"] is True

        # Check default task was provided
        call_args = mock_adapter.scrape.call_args
        assert "task" in call_args.kwargs
        assert "Extract all documentation content" in call_args.kwargs["task"]

    @pytest.mark.asyncio
    async def test_try_browser_use_with_custom_actions(self, router):
        """Test browser-use with custom actions converted to task."""
        mock_adapter = AsyncMock()
        router._adapters = {"browser_use": mock_adapter}

        custom_actions = [
            {"type": "click", "selector": "#expand"},
            {"type": "scroll", "direction": "bottom"},
            {"type": "extract"},
        ]

        await router._try_browser_use("https://example.com", custom_actions)

        # Verify task was generated from actions
        call_args = mock_adapter.scrape.call_args
        task = call_args.kwargs["task"]
        assert "click on element with selector '#expand'" in task
        assert "scroll to the bottom of the page" in task
        assert "extract all visible content" in task

    @pytest.mark.asyncio
    async def test_try_playwright_basic(self, router):
        """Test basic Playwright execution."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "content": "Playwright result",
            "success": True,
        }
        router._adapters = {"playwright": mock_adapter}

        result = await router._try_playwright("https://example.com")

        assert result["success"] is True

        # Check empty actions list was provided
        call_args = mock_adapter.scrape.call_args
        assert call_args.kwargs["actions"] == []

    @pytest.mark.asyncio
    async def test_try_playwright_with_actions(self, router):
        """Test Playwright with custom actions."""
        mock_adapter = AsyncMock()
        router._adapters = {"playwright": mock_adapter}

        custom_actions = [
            {"type": "click", "selector": "#button"},
            {"type": "fill", "selector": "input", "text": "test"},
        ]

        await router._try_playwright("https://example.com", custom_actions)

        # Actions should be passed directly
        call_args = mock_adapter.scrape.call_args
        assert call_args.kwargs["actions"] == custom_actions


class TestHelperMethods:
    """Test helper methods."""

    def test_get_basic_js(self, router):
        """Test basic JavaScript generation."""
        js_code = router._get_basic_js("https://example.com")

        assert "setTimeout" in js_code
        assert "aria-expanded" in js_code
        assert "show more" in js_code
        assert "load more" in js_code
        assert "scrollTo" in js_code

    def test_convert_actions_to_js_empty(self, router):
        """Test JS conversion with empty actions."""
        js_code = router._convert_actions_to_js([])
        assert js_code == ""

    def test_convert_actions_to_js_all_types(self, router):
        """Test JS conversion for all action types."""
        actions = [
            {"type": "click", "selector": "#btn"},
            {"type": "type", "selector": "input", "text": "hello"},
            {"type": "wait", "timeout": 2000},
            {"type": "scroll"},
            {"type": "evaluate", "script": "console.log('test');"},
        ]

        js_code = router._convert_actions_to_js(actions)

        assert "querySelector('#btn')?.click()" in js_code
        assert "querySelector('input').value = 'hello'" in js_code
        assert "setTimeout(r, 2000)" in js_code
        assert "scrollTo(0, document.body.scrollHeight)" in js_code
        assert "console.log('test');" in js_code

    def test_convert_to_task_empty(self, router):
        """Test task conversion with empty actions."""
        task = router._convert_to_task([])
        assert "Navigate to the page and extract all documentation content" in task

    def test_convert_to_task_all_types(self, router):
        """Test task conversion for all action types."""
        actions = [
            {"type": "click", "selector": "#menu"},
            {"type": "type", "selector": ".search", "text": "API docs"},
            {"type": "wait", "timeout": 3000},
            {"type": "scroll"},
            {"type": "extract"},
            {"type": "expand"},
        ]

        task = router._convert_to_task(actions)

        assert "Navigate to the page, then" in task
        assert "click on element with selector '#menu'" in task
        assert "type 'API docs' in element with selector '.search'" in task
        assert "wait for 3000 milliseconds" in task
        assert "scroll to the bottom of the page" in task
        assert "extract all visible content from the page" in task
        assert "expand any collapsed sections or menus" in task

    def test_update_metrics(self, router):
        """Test metrics update calculation."""
        tool = "crawl4ai"

        # Test success metric
        router._update_metrics(tool, success=True, elapsed=0.5)
        assert router.metrics[tool]["success"] == 1
        assert router.metrics[tool]["total_time"] == 0.5
        assert router.metrics[tool]["avg_time"] == 0.5

        # Test failure metric
        router._update_metrics(tool, success=False, elapsed=1.0)
        assert router.metrics[tool]["failed"] == 1
        assert router.metrics[tool]["total_time"] == 1.5
        assert router.metrics[tool]["avg_time"] == 0.75  # (0.5 + 1.0) / 2

    def test_get_metrics(self, router):
        """Test metrics retrieval with calculated fields."""
        # Set up some metrics
        router._update_metrics("crawl4ai", True, 1.0)
        router._update_metrics("crawl4ai", True, 2.0)
        router._update_metrics("crawl4ai", False, 0.5)
        router._adapters = {"crawl4ai": "mock"}

        metrics = router.get_metrics()

        assert metrics["crawl4ai"]["success"] == 2
        assert metrics["crawl4ai"]["failed"] == 1
        assert metrics["crawl4ai"]["total_attempts"] == 3
        assert metrics["crawl4ai"]["success_rate"] == 2 / 3
        assert metrics["crawl4ai"]["available"] is True
        assert metrics["browser_use"]["available"] is False

    def test_get_recommended_tool_base_selection(self, router):
        """Test basic tool recommendation."""
        router._adapters = {"crawl4ai": "mock", "browser_use": "mock"}

        # Should use routing rules
        tool = router.get_recommended_tool("https://vercel.com/docs")
        assert tool == "browser_use"

        tool = router.get_recommended_tool("https://example.com")
        assert tool == "crawl4ai"

    def test_get_recommended_tool_performance_based(self, router):
        """Test performance-based tool recommendation."""
        router._adapters = {"crawl4ai": "mock", "browser_use": "mock"}

        # Set up metrics showing crawl4ai has poor success rate
        for _ in range(10):
            router._update_metrics("crawl4ai", False, 1.0)  # All failures
            router._update_metrics("browser_use", True, 2.0)  # All successes

        # Even though crawl4ai would be default, should recommend browser-use
        tool = router.get_recommended_tool("https://example.com")
        assert tool == "browser_use"

    def test_get_recommended_tool_insufficient_data(self, router):
        """Test recommendation with insufficient performance data."""
        router._adapters = {"crawl4ai": "mock"}

        # Less than 5 attempts, should use base recommendation
        router._update_metrics("crawl4ai", True, 1.0)

        tool = router.get_recommended_tool("https://example.com")
        assert tool == "crawl4ai"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_scrape_with_very_long_timeout(self, router):
        """Test scraping with maximum timeout."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        with patch.object(router, "_try_crawl4ai", return_value={"success": True}):
            result = await router.scrape(
                "https://example.com",
                timeout=30000,  # Maximum timeout
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_scrapes(self, router):
        """Test concurrent scraping operations."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        async def mock_scrape(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return {"success": True, "content": f"Result for {args[0]}"}

        with patch.object(router, "_try_crawl4ai", side_effect=mock_scrape):
            urls = [f"https://example{i}.com" for i in range(5)]
            tasks = [router.scrape(url) for url in urls]
            results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r["success"] for r in results)

    def test_check_routing_rules_case_insensitive(self, router):
        """Test routing rules are case insensitive."""
        router._adapters = {"browser_use": "mock"}
        router.routing_rules = {"browser_use": ["Example.COM"]}

        # Should match regardless of case
        tool = router._check_routing_rules("example.com")
        assert tool == "browser_use"

        tool = router._check_routing_rules("EXAMPLE.COM")
        assert tool == "browser_use"

    def test_check_routing_rules_subdomain_matching(self, router):
        """Test routing rules match subdomains."""
        router._adapters = {"playwright": "mock"}
        router.routing_rules = {"playwright": ["github.com"]}

        # Should match subdomains
        domains = ["github.com", "api.github.com", "docs.github.com"]
        for domain in domains:
            tool = router._check_routing_rules(domain)
            assert tool == "playwright"
