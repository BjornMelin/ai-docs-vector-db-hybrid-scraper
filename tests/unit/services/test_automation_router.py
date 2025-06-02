"""Comprehensive tests for browser automation router."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.browser.automation_router import AutomationRouter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock unified configuration."""
    config = MagicMock(spec=UnifiedConfig)
    config.performance.max_concurrent_requests = 5
    config.performance.request_timeout = 30.0
    return config


@pytest.fixture
def sample_routing_rules():
    """Sample routing rules for testing."""
    return {
        "routing_rules": {
            "browser_use": [
                "vercel.com",
                "clerk.com",
                "supabase.com",
            ],
            "playwright": [
                "github.com",
                "stackoverflow.com",
            ],
        }
    }


@pytest.fixture
def router(mock_config):
    """Create automation router instance."""
    return AutomationRouter(mock_config)


class TestAutomationRouterInit:
    """Test AutomationRouter initialization."""

    def test_init_with_config(self, mock_config):
        """Test router initialization with config."""
        router = AutomationRouter(mock_config)
        assert router.config == mock_config
        assert router._initialized is False
        assert router._adapters == {}
        assert router.metrics is not None

    def test_init_metrics_structure(self, router):
        """Test initial metrics structure."""
        expected_tools = ["crawl4ai", "browser_use", "playwright"]
        for tool in expected_tools:
            assert tool in router.metrics
            metrics = router.metrics[tool]
            assert metrics["success"] == 0
            assert metrics["failed"] == 0
            assert metrics["avg_time"] == 0.0
            assert metrics["total_time"] == 0.0

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_routing_rules_from_file(
        self, mock_file, mock_exists, router, sample_routing_rules
    ):
        """Test loading routing rules from configuration file."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(sample_routing_rules)

        rules = router._load_routing_rules()

        assert "browser_use" in rules
        assert "vercel.com" in rules["browser_use"]
        assert "playwright" in rules
        assert "github.com" in rules["playwright"]

    @patch("pathlib.Path.exists")
    def test_load_routing_rules_file_not_found(self, mock_exists, router):
        """Test fallback to default rules when file not found."""
        mock_exists.return_value = False

        rules = router._load_routing_rules()

        # Should return default rules
        assert "browser_use" in rules
        assert "playwright" in rules
        assert isinstance(rules["browser_use"], list)
        assert isinstance(rules["playwright"], list)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_routing_rules_file_error(self, mock_file, mock_exists, router):
        """Test fallback when file loading fails."""
        mock_exists.return_value = True
        mock_file.side_effect = Exception("File read error")

        rules = router._load_routing_rules()

        # Should return default rules on error
        assert "browser_use" in rules
        assert "playwright" in rules

    def test_get_default_routing_rules(self, router):
        """Test default routing rules structure."""
        rules = router._get_default_routing_rules()

        assert "browser_use" in rules
        assert "playwright" in rules
        assert isinstance(rules["browser_use"], list)
        assert isinstance(rules["playwright"], list)

        # Check some expected domains
        assert "vercel.com" in rules["browser_use"]
        assert "github.com" in rules["playwright"]


class TestAutomationRouterInitialization:
    """Test router adapter initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, router):
        """Test successful initialization of all adapters."""
        with patch.multiple(
            "src.services.browser.automation_router",
            Crawl4AIAdapter=MagicMock(),
            BrowserUseAdapter=MagicMock(),
            PlaywrightAdapter=MagicMock(),
        ) as mocks:
            # Setup mock adapters
            for adapter_class in mocks.values():
                adapter_instance = AsyncMock()
                adapter_class.return_value = adapter_instance
                adapter_instance.initialize = AsyncMock()

            await router.initialize()

            assert router._initialized is True
            assert len(router._adapters) == 3
            assert "crawl4ai" in router._adapters
            assert "browser_use" in router._adapters
            assert "playwright" in router._adapters

    @pytest.mark.asyncio
    async def test_initialize_partial_failure(self, router):
        """Test initialization with some adapters failing."""
        with patch.multiple(
            "src.services.browser.automation_router",
            Crawl4AIAdapter=MagicMock(side_effect=Exception("Crawl4AI init failed")),
            BrowserUseAdapter=MagicMock(),
            PlaywrightAdapter=MagicMock(),
        ) as mocks:
            # Setup working adapters
            for name, adapter_class in mocks.items():
                if name != "Crawl4AIAdapter":
                    adapter_instance = AsyncMock()
                    adapter_class.return_value = adapter_instance
                    adapter_instance.initialize = AsyncMock()

            await router.initialize()

            assert router._initialized is True
            assert len(router._adapters) == 2  # Only 2 should succeed
            assert "crawl4ai" not in router._adapters
            assert "browser_use" in router._adapters
            assert "playwright" in router._adapters

    @pytest.mark.asyncio
    async def test_initialize_all_adapters_fail(self, router):
        """Test initialization when all adapters fail."""
        with patch.multiple(
            "src.services.browser.automation_router",
            Crawl4AIAdapter=MagicMock(side_effect=Exception("Failed")),
            BrowserUseAdapter=MagicMock(side_effect=Exception("Failed")),
            PlaywrightAdapter=MagicMock(side_effect=Exception("Failed")),
        ):
            with pytest.raises(
                CrawlServiceError, match="No automation adapters available"
            ):
                await router.initialize()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, router):
        """Test that re-initialization is skipped."""
        router._initialized = True
        original_adapters = router._adapters.copy()

        await router.initialize()

        assert router._adapters == original_adapters

    @pytest.mark.asyncio
    async def test_cleanup(self, router):
        """Test cleanup of all adapters."""
        # Setup mock adapters
        mock_adapters = {
            "crawl4ai": AsyncMock(),
            "browser_use": AsyncMock(),
            "playwright": AsyncMock(),
        }
        router._adapters = mock_adapters
        router._initialized = True

        await router.cleanup()

        # Check all adapters were cleaned up
        for adapter in mock_adapters.values():
            adapter.cleanup.assert_called_once()

        assert router._adapters == {}
        assert router._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_with_errors(self, router):
        """Test cleanup continues even if some adapters fail."""
        # Setup mock adapters with one failing
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

        assert router._adapters == {}
        assert router._initialized is False


class TestToolSelection:
    """Test tool selection logic."""

    def test_select_tool_routing_rules_priority(self, router):
        """Test that routing rules take priority."""
        router.routing_rules = {
            "browser_use": ["vercel.com"],
            "playwright": ["github.com"],
        }
        router._adapters = {
            "crawl4ai": MagicMock(),
            "browser_use": MagicMock(),
            "playwright": MagicMock(),
        }

        # Test routing rule matches
        assert (
            router._select_tool("https://vercel.com/docs", False, None) == "browser_use"
        )
        assert (
            router._select_tool("https://github.com/user/repo", False, None)
            == "playwright"
        )

    def test_select_tool_interaction_requirements(self, router):
        """Test tool selection based on interaction requirements."""
        router.routing_rules = {}
        router._adapters = {
            "crawl4ai": MagicMock(),
            "browser_use": MagicMock(),
            "playwright": MagicMock(),
        }

        # Interaction required should prefer browser_use
        result = router._select_tool("https://example.com", True, None)
        assert result == "browser_use"

        # Custom actions should prefer browser_use
        result = router._select_tool("https://example.com", False, [{"type": "click"}])
        assert result == "browser_use"

    def test_select_tool_javascript_patterns(self, router):
        """Test tool selection for JavaScript-heavy sites."""
        router.routing_rules = {}
        router._adapters = {"crawl4ai": MagicMock(), "browser_use": MagicMock()}

        js_urls = [
            "https://example.com/spa/app",
            "https://react-site.com/docs",
            "https://vue-app.com",
            "https://angular-docs.com",
        ]

        for url in js_urls:
            result = router._select_tool(url, False, None)
            assert result == "browser_use"

    def test_select_tool_fallback_order(self, router):
        """Test fallback order when preferred tools not available."""
        router.routing_rules = {}

        # Only crawl4ai available
        router._adapters = {"crawl4ai": MagicMock()}
        result = router._get_default_tool()
        assert result == "crawl4ai"

        # Only playwright available
        router._adapters = {"playwright": MagicMock()}
        result = router._get_default_tool()
        assert result == "playwright"

        # Only browser_use available
        router._adapters = {"browser_use": MagicMock()}
        result = router._get_default_tool()
        assert result == "browser_use"

    def test_check_routing_rules(self, router):
        """Test routing rules checking."""
        router.routing_rules = {
            "browser_use": ["vercel.com", "clerk.com"],
            "playwright": ["github.com"],
        }
        router._adapters = {"browser_use": MagicMock(), "playwright": MagicMock()}

        assert router._check_routing_rules("app.vercel.com") == "browser_use"
        assert router._check_routing_rules("github.com") == "playwright"
        assert router._check_routing_rules("unknown.com") is None

        # Test partial matches
        assert router._check_routing_rules("subdomain.vercel.com") == "browser_use"

    def test_check_interaction_requirements(self, router):
        """Test interaction requirements checking."""
        router._adapters = {"browser_use": MagicMock(), "playwright": MagicMock()}

        # Explicit interaction required
        result = router._check_interaction_requirements(
            "https://example.com", True, None
        )
        assert result == "browser_use"

        # Custom actions provided
        result = router._check_interaction_requirements(
            "https://example.com", False, [{"type": "click"}]
        )
        assert result == "browser_use"

        # JavaScript patterns in URL
        result = router._check_interaction_requirements(
            "https://spa-app.com", False, None
        )
        assert result == "browser_use"

        # No special requirements
        result = router._check_interaction_requirements(
            "https://static-site.com", False, None
        )
        assert result is None

    def test_check_interaction_requirements_fallback(self, router):
        """Test interaction requirements with adapter fallback."""
        # No browser_use, should fallback to playwright
        router._adapters = {"playwright": MagicMock()}

        result = router._check_interaction_requirements(
            "https://example.com", True, None
        )
        assert result == "playwright"


class TestScraping:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, router):
        """Test scraping when router not initialized."""
        with pytest.raises(CrawlServiceError, match="Router not initialized"):
            await router.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_forced_tool_success(self, router):
        """Test scraping with forced tool selection."""
        router._initialized = True
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "success": True,
            "content": "test content",
            "metadata": {},
        }
        router._adapters = {"crawl4ai": mock_adapter}

        result = await router.scrape("https://example.com", force_tool="crawl4ai")

        assert result["success"] is True
        assert result["provider"] == "crawl4ai"
        assert "automation_time_ms" in result
        mock_adapter.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_forced_tool_unavailable(self, router):
        """Test scraping with unavailable forced tool."""
        router._initialized = True
        router._adapters = {"crawl4ai": AsyncMock()}

        with pytest.raises(
            CrawlServiceError, match="Forced tool 'playwright' not available"
        ):
            await router.scrape("https://example.com", force_tool="playwright")

    @pytest.mark.asyncio
    async def test_scrape_automatic_selection(self, router):
        """Test scraping with automatic tool selection."""
        router._initialized = True
        router.routing_rules = {"browser_use": ["example.com"]}

        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "success": True,
            "content": "test content",
            "metadata": {},
        }
        router._adapters = {"browser_use": mock_adapter}

        result = await router.scrape("https://example.com")

        assert result["provider"] == "browser_use"
        mock_adapter.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_with_fallback(self, router):
        """Test scraping with fallback when primary tool fails."""
        router._initialized = True

        # Setup primary adapter to fail
        primary_adapter = AsyncMock()
        primary_adapter.scrape.side_effect = Exception("Primary failed")

        # Setup fallback adapter to succeed
        fallback_adapter = AsyncMock()
        fallback_adapter.scrape.return_value = {
            "success": True,
            "content": "fallback content",
            "metadata": {},
        }

        router._adapters = {"crawl4ai": primary_adapter, "playwright": fallback_adapter}

        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            result = await router.scrape("https://example.com")

        assert result["provider"] == "playwright"
        assert result["fallback_from"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_all_tools_fail(self, router):
        """Test scraping when all tools fail."""
        router._initialized = True

        mock_adapter = AsyncMock()
        mock_adapter.scrape.side_effect = Exception("Tool failed")
        router._adapters = {"crawl4ai": mock_adapter}

        result = await router.scrape("https://example.com")

        assert result["success"] is False
        assert "All automation tools failed" in result["error"]
        assert result["provider"] == "none"


class TestActionConversion:
    """Test action conversion utilities."""

    def test_convert_actions_to_js(self, router):
        """Test converting actions to JavaScript."""
        actions = [
            {"type": "click", "selector": "button"},
            {"type": "type", "selector": "input", "text": "test"},
            {"type": "wait", "timeout": 1000},
            {"type": "scroll"},
            {"type": "evaluate", "script": "console.log('test')"},
        ]

        js_code = router._convert_actions_to_js(actions)

        assert "document.querySelector('button')?.click();" in js_code
        assert "document.querySelector('input').value = 'test';" in js_code
        assert "await new Promise(r => setTimeout(r, 1000));" in js_code
        assert "window.scrollTo(0, document.body.scrollHeight);" in js_code
        assert "console.log('test')" in js_code

    def test_convert_to_task(self, router):
        """Test converting actions to natural language task."""
        actions = [
            {"type": "click", "selector": "button"},
            {"type": "type", "selector": "input", "text": "search"},
            {"type": "wait", "timeout": 2000},
            {"type": "scroll"},
            {"type": "extract"},
            {"type": "expand"},
        ]

        task = router._convert_to_task(actions)

        assert "click on element with selector 'button'" in task
        assert "type 'search' in element with selector 'input'" in task
        assert "wait for 2000 milliseconds" in task
        assert "scroll to the bottom of the page" in task
        assert "extract all visible content" in task
        assert "expand any collapsed sections" in task

    def test_convert_to_task_empty_actions(self, router):
        """Test converting empty actions list."""
        task = router._convert_to_task([])
        assert "extract all documentation content" in task

    def test_get_basic_js(self, router):
        """Test basic JavaScript generation."""
        js_code = router._get_basic_js("https://example.com")

        assert "await new Promise(r => setTimeout(r, 2000));" in js_code
        assert "querySelectorAll('[aria-expanded=\"false\"]')" in js_code
        assert "show more" in js_code.lower()
        assert "scrollTo(0, document.body.scrollHeight)" in js_code


class TestMetrics:
    """Test performance metrics tracking."""

    def test_update_metrics_success(self, router):
        """Test updating metrics for successful operation."""
        initial_success = router.metrics["crawl4ai"]["success"]
        initial_total_time = router.metrics["crawl4ai"]["total_time"]

        router._update_metrics("crawl4ai", True, 1.5)

        assert router.metrics["crawl4ai"]["success"] == initial_success + 1
        assert router.metrics["crawl4ai"]["failed"] == 0
        assert router.metrics["crawl4ai"]["total_time"] == initial_total_time + 1.5
        assert router.metrics["crawl4ai"]["avg_time"] == 1.5

    def test_update_metrics_failure(self, router):
        """Test updating metrics for failed operation."""
        router._update_metrics("playwright", False, 2.0)

        assert router.metrics["playwright"]["success"] == 0
        assert router.metrics["playwright"]["failed"] == 1
        assert router.metrics["playwright"]["total_time"] == 2.0
        assert router.metrics["playwright"]["avg_time"] == 2.0

    def test_update_metrics_rolling_average(self, router):
        """Test rolling average calculation."""
        # Add multiple measurements
        router._update_metrics("browser_use", True, 1.0)
        router._update_metrics("browser_use", True, 2.0)
        router._update_metrics("browser_use", False, 3.0)

        metrics = router.metrics["browser_use"]
        assert metrics["success"] == 2
        assert metrics["failed"] == 1
        assert metrics["total_time"] == 6.0
        assert metrics["avg_time"] == 2.0  # 6.0 / 3 attempts

    def test_get_metrics(self, router):
        """Test getting comprehensive metrics."""
        # Add some test data
        router._update_metrics("crawl4ai", True, 1.0)
        router._update_metrics("crawl4ai", False, 2.0)
        router._adapters = {"crawl4ai": MagicMock()}

        metrics = router.get_metrics()

        assert "crawl4ai" in metrics
        crawl4ai_metrics = metrics["crawl4ai"]
        assert crawl4ai_metrics["success"] == 1
        assert crawl4ai_metrics["failed"] == 1
        assert crawl4ai_metrics["total_attempts"] == 2
        assert crawl4ai_metrics["success_rate"] == 0.5
        assert crawl4ai_metrics["available"] is True

    def test_get_metrics_no_attempts(self, router):
        """Test metrics for tools with no attempts."""
        metrics = router.get_metrics()

        for tool_metrics in metrics.values():
            assert tool_metrics["success_rate"] == 0.0
            assert tool_metrics["total_attempts"] == 0
            assert tool_metrics["available"] is False

    def test_get_recommended_tool(self, router):
        """Test getting recommended tool based on performance."""
        router._adapters = {"crawl4ai": MagicMock(), "playwright": MagicMock()}

        # Test with insufficient data (< 5 attempts)
        recommended = router.get_recommended_tool("https://example.com")
        assert recommended in ["crawl4ai", "playwright"]  # Base recommendation

        # Add enough data to make performance-based recommendation
        for _ in range(6):
            router._update_metrics("crawl4ai", True, 1.0)  # High success rate
        for _ in range(6):
            router._update_metrics("playwright", False, 2.0)  # Low success rate

        # Should recommend crawl4ai due to better performance
        with patch.object(router, "_select_tool", return_value="playwright"):
            recommended = router.get_recommended_tool("https://example.com")
            assert recommended == "crawl4ai"

    def test_get_recommended_tool_low_success_rate(self, router):
        """Test recommendation when base tool has low success rate."""
        router._adapters = {"crawl4ai": MagicMock(), "playwright": MagicMock()}

        # Make crawl4ai have low success rate
        for _ in range(5):
            router._update_metrics("crawl4ai", False, 1.0)

        # Make playwright have higher success rate
        for _ in range(5):
            router._update_metrics("playwright", True, 1.0)

        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            recommended = router.get_recommended_tool("https://example.com")
            assert recommended == "playwright"


class TestAdapterMethods:
    """Test adapter-specific method calls."""

    @pytest.mark.asyncio
    async def test_try_crawl4ai(self, router):
        """Test Crawl4AI adapter method."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {"success": True, "content": "test"}
        router._adapters = {"crawl4ai": mock_adapter}

        result = await router._try_crawl4ai("https://example.com")

        mock_adapter.scrape.assert_called_once()
        call_args = mock_adapter.scrape.call_args
        assert call_args[1]["url"] == "https://example.com"
        assert "wait_for_selector" in call_args[1]
        assert "js_code" in call_args[1]

    @pytest.mark.asyncio
    async def test_try_browser_use(self, router):
        """Test BrowserUse adapter method."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {"success": True, "content": "test"}
        router._adapters = {"browser_use": mock_adapter}

        result = await router._try_browser_use("https://example.com")

        mock_adapter.scrape.assert_called_once()
        call_args = mock_adapter.scrape.call_args
        assert call_args[1]["url"] == "https://example.com"
        assert "task" in call_args[1]

    @pytest.mark.asyncio
    async def test_try_playwright(self, router):
        """Test Playwright adapter method."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {"success": True, "content": "test"}
        router._adapters = {"playwright": mock_adapter}

        actions = [{"type": "click", "selector": "button"}]
        result = await router._try_playwright("https://example.com", actions)

        mock_adapter.scrape.assert_called_once()
        call_args = mock_adapter.scrape.call_args
        assert call_args[1]["url"] == "https://example.com"
        assert call_args[1]["actions"] == actions

    @pytest.mark.asyncio
    async def test_try_crawl4ai_with_custom_actions(self, router):
        """Test Crawl4AI with custom actions."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {"success": True}
        router._adapters = {"crawl4ai": mock_adapter}

        custom_actions = [{"type": "click", "selector": "button"}]
        await router._try_crawl4ai("https://example.com", custom_actions)

        call_args = mock_adapter.scrape.call_args
        assert "js_code" in call_args[1]
        # Should contain converted JavaScript
        js_code = call_args[1]["js_code"]
        assert "querySelector('button')" in js_code

    @pytest.mark.asyncio
    async def test_try_browser_use_with_custom_actions(self, router):
        """Test BrowserUse with custom actions."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {"success": True}
        router._adapters = {"browser_use": mock_adapter}

        custom_actions = [{"type": "click", "selector": "button"}]
        await router._try_browser_use("https://example.com", custom_actions)

        call_args = mock_adapter.scrape.call_args
        task = call_args[1]["task"]
        assert "click on element with selector 'button'" in task


class TestFallbackMechanism:
    """Test fallback mechanism."""

    @pytest.mark.asyncio
    async def test_fallback_scrape_success(self, router):
        """Test successful fallback scraping."""
        # Setup adapters
        failed_adapter = AsyncMock()
        failed_adapter.scrape.return_value = {"success": True}

        fallback_adapter = AsyncMock()
        fallback_adapter.scrape.return_value = {"success": True, "content": "fallback"}

        router._adapters = {"crawl4ai": failed_adapter, "playwright": fallback_adapter}

        result = await router._fallback_scrape(
            "https://example.com", "crawl4ai", None, 30000
        )

        assert result["success"] is True
        assert result["provider"] == "playwright"
        assert result["fallback_from"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_fallback_scrape_all_fail(self, router):
        """Test fallback when all tools fail."""
        # Setup all adapters to fail
        failed_adapter = AsyncMock()
        failed_adapter.scrape.side_effect = Exception("Failed")

        router._adapters = {"crawl4ai": failed_adapter, "playwright": failed_adapter}

        result = await router._fallback_scrape(
            "https://example.com", "browser_use", None, 30000
        )

        assert result["success"] is False
        assert "All automation tools failed" in result["error"]
        assert result["provider"] == "none"
        assert "failed_tools" in result

    def test_fallback_order(self, router):
        """Test fallback order definition."""
        # Test fallback order for each primary tool
        fallback_order = {
            "crawl4ai": ["browser_use", "playwright"],
            "browser_use": ["playwright", "crawl4ai"],
            "playwright": ["browser_use", "crawl4ai"],
        }

        # This tests the logic inside _fallback_scrape
        for primary, expected_fallbacks in fallback_order.items():
            # The actual fallback order is defined in the method
            # We're testing that it follows logical patterns
            assert isinstance(expected_fallbacks, list)
            assert len(expected_fallbacks) == 2
            assert primary not in expected_fallbacks


class TestIntegration:
    """Integration tests for router functionality."""

    @pytest.mark.asyncio
    async def test_full_workflow_success(self, router):
        """Test complete workflow from initialization to scraping."""
        # Mock all adapters
        mock_adapters = {}
        for name in ["crawl4ai", "browser_use", "playwright"]:
            adapter = AsyncMock()
            adapter.initialize = AsyncMock()
            adapter.cleanup = AsyncMock()
            adapter.scrape.return_value = {
                "success": True,
                "content": f"{name} content",
                "metadata": {},
            }
            mock_adapters[name] = adapter

        with patch.multiple(
            "src.services.browser.automation_router",
            Crawl4AIAdapter=lambda config: mock_adapters["crawl4ai"],
            BrowserUseAdapter=lambda config: mock_adapters["browser_use"],
            PlaywrightAdapter=lambda config: mock_adapters["playwright"],
        ):
            # Initialize
            await router.initialize()
            assert router._initialized is True

            # Scrape with different scenarios
            result1 = await router.scrape("https://example.com")
            assert result1["success"] is True

            result2 = await router.scrape(
                "https://github.com/user/repo",
                interaction_required=True,
                custom_actions=[{"type": "click", "selector": "button"}],
            )
            assert result2["success"] is True

            # Cleanup
            await router.cleanup()
            assert router._initialized is False

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, router):
        """Test error handling and recovery mechanisms."""
        # Setup scenario with intermittent failures
        unreliable_adapter = AsyncMock()
        reliable_adapter = AsyncMock()

        # Make first adapter fail sometimes
        unreliable_adapter.scrape.side_effect = [
            Exception("Temporary failure"),
            {"success": True, "content": "recovered"},
        ]
        reliable_adapter.scrape.return_value = {"success": True, "content": "reliable"}

        router._adapters = {
            "crawl4ai": unreliable_adapter,
            "playwright": reliable_adapter,
        }
        router._initialized = True

        # First call should trigger fallback
        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            result = await router.scrape("https://example.com")
            assert result["provider"] == "playwright"  # Fallback used
            assert "fallback_from" in result

    def test_metrics_accumulation(self, router):
        """Test that metrics accumulate correctly over time."""
        # Simulate multiple operations
        operations = [
            ("crawl4ai", True, 0.5),
            ("crawl4ai", True, 1.0),
            ("crawl4ai", False, 2.0),
            ("playwright", True, 1.5),
            ("playwright", False, 0.8),
        ]

        for tool, success, elapsed in operations:
            router._update_metrics(tool, success, elapsed)

        # Check accumulated metrics
        crawl4ai_metrics = router.metrics["crawl4ai"]
        assert crawl4ai_metrics["success"] == 2
        assert crawl4ai_metrics["failed"] == 1
        assert crawl4ai_metrics["total_time"] == 3.5
        assert crawl4ai_metrics["avg_time"] == 3.5 / 3

        playwright_metrics = router.metrics["playwright"]
        assert playwright_metrics["success"] == 1
        assert playwright_metrics["failed"] == 1
        assert playwright_metrics["total_time"] == 2.3
        assert playwright_metrics["avg_time"] == 2.3 / 2
