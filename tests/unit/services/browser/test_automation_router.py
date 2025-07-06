"""Comprehensive tests for AutomationRouter with Pydantic configuration."""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from src.config import (
    BrowserUseConfig,
    Config,
    Crawl4AIConfig,
    PlaywrightConfig
)
from src.services.browser.automation_router import AutomationRouter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_unified_config():
    """Create mock unified configuration with all browser configs."""
    config = MagicMock(spec=Config)

    # Mock browser configurations
    config.crawl4ai = MagicMock(spec=Crawl4AIConfig)
    config.browser_use = MagicMock(spec=BrowserUseConfig)
    config.playwright = MagicMock(spec=PlaywrightConfig)

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
def router(mock_unified_config):
    """Create automation router instance."""
    return AutomationRouter(mock_unified_config)


class TestAutomationRouterInit:
    """Test AutomationRouter initialization."""

    def test_init_with_config(self, mock_unified_config):
        """Test initialization with unified config."""
        router = AutomationRouter(mock_unified_config)

        assert router.config == mock_unified_config
        assert router._adapters == {}
        assert router._initialized is False
        assert isinstance(router.metrics, dict)
        assert "crawl4ai" in router.metrics
        assert "browser_use" in router.metrics
        assert "playwright" in router.metrics

    def test_metrics_structure(self, router):
        """Test metrics structure initialization."""
        for tool in ["crawl4ai", "browser_use", "playwright"]:
            assert tool in router.metrics
            assert "success" in router.metrics[tool]
            assert "failed" in router.metrics[tool]
            assert "avg_time" in router.metrics[tool]
            assert "_total_time" in router.metrics[tool]
            assert router.metrics[tool]["success"] == 0
            assert router.metrics[tool]["failed"] == 0

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_routing_rules_from_file(
        self, _mock_exists, mock_file, sample_routing_rules
    ):
        """Test loading routing rules from file."""
        mock_file.return_value.read.return_value = json.dumps(sample_routing_rules)

        with patch("json.load", return_value=sample_routing_rules):
            router = AutomationRouter(MagicMock())

            assert router.routing_rules == sample_routing_rules["routing_rules"]

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_routing_rules_file_not_found(self, _mock_exists):
        """Test routing rules fallback when file not found."""
        router = AutomationRouter(MagicMock())

        # Should use default rules
        assert "browser_use" in router.routing_rules
        assert "playwright" in router.routing_rules
        assert "vercel.com" in router.routing_rules["browser_use"]

    @patch("builtins.open", side_effect=Exception("File read error"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_routing_rules_file_error(self, _mock_exists, _mock_file):
        """Test routing rules fallback on file error."""
        router = AutomationRouter(MagicMock())

        # Should use default rules
        assert "browser_use" in router.routing_rules
        assert "playwright" in router.routing_rules

    def test_get_default_routing_rules(self, router):
        """Test default routing rules structure."""
        default_rules = router._get_default_routing_rules()

        assert "browser_use" in default_rules
        assert "playwright" in default_rules
        assert isinstance(default_rules["browser_use"], list)
        assert isinstance(default_rules["playwright"], list)
        assert "vercel.com" in default_rules["browser_use"]
        assert "github.com" in default_rules["playwright"]


class TestAutomationRouterInitialization:
    """Test adapter initialization process."""

    @pytest.mark.asyncio
    @patch("src.services.browser.crawl4ai_adapter.Crawl4AIAdapter")
    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_success(self, mock_adapter_class, router):
        """Test successful Crawl4AI adapter initialization."""
        mock_adapter = AsyncMock()
        mock_adapter_class.return_value = mock_adapter

        await router.initialize()

        assert router._initialized is True
        assert "crawl4ai" in router._adapters
        assert router._adapters["crawl4ai"] == mock_adapter
        mock_adapter_class.assert_called_once_with(router.config.crawl4ai)
        mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BrowserUseAdapter")
    @pytest.mark.asyncio
    async def test_initialize_browser_use_success(self, mock_adapter_class, router):
        """Test successful BrowserUse adapter initialization."""
        mock_adapter = AsyncMock()
        mock_adapter_class.return_value = mock_adapter

        await router.initialize()

        assert "browser_use" in router._adapters
        assert router._adapters["browser_use"] == mock_adapter
        mock_adapter_class.assert_called_once_with(router.config.browser_use)
        mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PlaywrightAdapter")
    @pytest.mark.asyncio
    async def test_initialize_playwright_success(self, mock_adapter_class, router):
        """Test successful Playwright adapter initialization."""
        mock_adapter = AsyncMock()
        mock_adapter_class.return_value = mock_adapter

        await router.initialize()

        assert "playwright" in router._adapters
        assert router._adapters["playwright"] == mock_adapter
        mock_adapter_class.assert_called_once_with(router.config.playwright)
        mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_crawl4ai_failure(self, router):
        """Test Crawl4AI adapter initialization failure."""
        with (
            patch("src.services.browser.crawl4ai_adapter.Crawl4AIAdapter") as mock_c4ai,
            patch(
                "src.services.browser.browser_use_adapter.BrowserUseAdapter"
            ) as mock_bu,
            patch(
                "src.services.browser.playwright_adapter.PlaywrightAdapter"
            ) as mock_pw,
        ):
            # Crawl4AI fails, others succeed
            mock_c4ai.side_effect = Exception("Crawl4AI init failed")

            # Setup successful adapters
            for mock_class in [mock_bu, mock_pw]:
                mock_adapter = AsyncMock()
                mock_class.return_value = mock_adapter

            await router.initialize()

            # Should still initialize successfully without Crawl4AI
            assert router._initialized is True
            assert "crawl4ai" not in router._adapters
            assert "browser_use" in router._adapters
            assert "playwright" in router._adapters

    @pytest.mark.asyncio
    async def test_initialize_browser_use_failure(self, router):
        """Test BrowserUse adapter initialization failure."""
        with (
            patch("src.services.browser.crawl4ai_adapter.Crawl4AIAdapter") as mock_c4ai,
            patch(
                "src.services.browser.browser_use_adapter.BrowserUseAdapter"
            ) as mock_bu,
            patch(
                "src.services.browser.playwright_adapter.PlaywrightAdapter"
            ) as mock_pw,
        ):
            # BrowserUse fails, others succeed
            mock_bu.side_effect = Exception("BrowserUse init failed")

            # Setup successful adapters
            for mock_class in [mock_c4ai, mock_pw]:
                mock_adapter = AsyncMock()
                mock_class.return_value = mock_adapter

            await router.initialize()

            # Should still initialize successfully without BrowserUse
            assert router._initialized is True
            assert "browser_use" not in router._adapters
            assert "crawl4ai" in router._adapters
            assert "playwright" in router._adapters

    @pytest.mark.asyncio
    async def test_initialize_playwright_failure(self, router):
        """Test Playwright adapter initialization failure."""
        with (
            patch("src.services.browser.crawl4ai_adapter.Crawl4AIAdapter") as mock_c4ai,
            patch(
                "src.services.browser.browser_use_adapter.BrowserUseAdapter"
            ) as mock_bu,
            patch(
                "src.services.browser.playwright_adapter.PlaywrightAdapter"
            ) as mock_pw,
        ):
            # Playwright fails, others succeed
            mock_pw.side_effect = Exception("Playwright init failed")

            # Setup successful adapters
            for mock_class in [mock_c4ai, mock_bu]:
                mock_adapter = AsyncMock()
                mock_class.return_value = mock_adapter

            await router.initialize()

            # Should still initialize successfully without Playwright
            assert router._initialized is True
            assert "playwright" not in router._adapters
            assert "crawl4ai" in router._adapters
            assert "browser_use" in router._adapters

    @pytest.mark.asyncio
    async def test_initialize_all_adapters_fail(self, router):
        """Test initialization when all adapters fail."""
        with (
            patch("src.services.browser.crawl4ai_adapter.Crawl4AIAdapter") as mock_c4ai,
            patch(
                "src.services.browser.browser_use_adapter.BrowserUseAdapter"
            ) as mock_bu,
            patch(
                "src.services.browser.playwright_adapter.PlaywrightAdapter"
            ) as mock_pw,
        ):
            mock_c4ai.side_effect = Exception("Crawl4AI failed")
            mock_bu.side_effect = Exception("BrowserUse failed")
            mock_pw.side_effect = Exception("Playwright failed")

            with pytest.raises(
                CrawlServiceError, match="No automation adapters available"
            ):
                await router.initialize()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, router):
        """Test initialization when already initialized."""
        router._initialized = True

        with patch(
            "src.services.browser.crawl4ai_adapter.Crawl4AIAdapter"
        ) as mock_adapter:
            await router.initialize()

            # Should not create new adapters
            mock_adapter.assert_not_called()


class TestAutomationRouterCleanup:
    """Test cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self, router):
        """Test successful cleanup of all adapters."""
        # Setup mock adapters
        mock_adapters = {}
        for name in ["crawl4ai", "browser_use", "playwright"]:
            mock_adapter = AsyncMock()
            mock_adapters[name] = mock_adapter
            router._adapters[name] = mock_adapter

        router._initialized = True

        await router.cleanup()

        # Verify all adapters were cleaned up
        for mock_adapter in mock_adapters.values():
            mock_adapter.cleanup.assert_called_once()

        assert router._adapters == {}
        assert router._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_adapter_error(self, router):
        """Test cleanup when adapter cleanup fails."""
        mock_adapter = AsyncMock()
        mock_adapter.cleanup.side_effect = Exception("Cleanup failed")
        router._adapters["crawl4ai"] = mock_adapter
        router._initialized = True

        # Should not raise exception even if cleanup fails
        await router.cleanup()

        assert router._adapters == {}
        assert router._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_no_adapters(self, router):
        """Test cleanup when no adapters exist."""
        router._initialized = True

        # Should not raise exception
        await router.cleanup()

        assert router._adapters == {}
        assert router._initialized is False


class TestAutomationRouterScraping:
    """Test scraping functionality and tool selection."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, router):
        """Test scraping when router not initialized."""
        with pytest.raises(CrawlServiceError, match="Router not initialized"):
            await router.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_force_tool_success(self, router):
        """Test scraping with forced tool selection."""
        # Setup mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "success": True,
            "content": "Test content",
            "url": "https://example.com",
        }
        router._adapters["crawl4ai"] = mock_adapter
        router._initialized = True

        result = await router.scrape("https://example.com", force_tool="crawl4ai")

        assert result["success"] is True
        assert result["provider"] == "crawl4ai"
        assert "automation_time_ms" in result
        mock_adapter.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_force_tool_unavailable(self, router):
        """Test scraping with unavailable forced tool."""
        router._initialized = True

        with pytest.raises(
            CrawlServiceError, match="Forced tool 'crawl4ai' not available"
        ):
            await router.scrape("https://example.com", force_tool="crawl4ai")

    @pytest.mark.asyncio
    async def test_scrape_crawl4ai_success(self, router):
        """Test successful scraping with Crawl4AI."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "success": True,
            "content": "Test content",
            "html": "<div>Test</div>",
            "title": "Test Page",
            "metadata": {},
        }
        router._adapters["crawl4ai"] = mock_adapter
        router._initialized = True

        # Mock tool selection to return crawl4ai
        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            result = await router.scrape("https://example.com")

            assert result["success"] is True
            assert result["provider"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_browser_use_success(self, router):
        """Test successful scraping with BrowserUse."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "success": True,
            "content": "Test content",
            "url": "https://example.com",
        }
        router._adapters["browser_use"] = mock_adapter
        router._initialized = True

        # Mock tool selection to return browser_use
        with patch.object(router, "_select_tool", return_value="browser_use"):
            result = await router.scrape("https://example.com")

            assert result["success"] is True
            assert result["provider"] == "browser_use"

    @pytest.mark.asyncio
    async def test_scrape_playwright_success(self, router):
        """Test successful scraping with Playwright."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {
            "success": True,
            "content": "Test content",
            "url": "https://example.com",
        }
        router._adapters["playwright"] = mock_adapter
        router._initialized = True

        # Mock tool selection to return playwright
        with patch.object(router, "_select_tool", return_value="playwright"):
            result = await router.scrape("https://example.com")

            assert result["success"] is True
            assert result["provider"] == "playwright"

    @pytest.mark.asyncio
    async def test_scrape_with_custom_actions(self, router):
        """Test scraping with custom actions."""
        mock_adapter = AsyncMock()
        mock_adapter.scrape.return_value = {"success": True, "content": "Test"}
        router._adapters["crawl4ai"] = mock_adapter
        router._initialized = True

        custom_actions = [{"type": "click", "selector": ".button"}]

        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            await router.scrape("https://example.com", custom_actions=custom_actions)

            # Should pass the actions to the try_crawl4ai method
            mock_adapter.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_with_fallback(self, router):
        """Test scraping with fallback when primary tool fails."""
        # Primary tool fails
        mock_crawl4ai = AsyncMock()
        mock_crawl4ai.scrape.side_effect = Exception("Crawl4AI failed")
        router._adapters["crawl4ai"] = mock_crawl4ai

        # Fallback tool succeeds
        mock_playwright = AsyncMock()
        mock_playwright.scrape.return_value = {
            "success": True,
            "content": "Fallback content",
            "url": "https://example.com",
        }
        router._adapters["playwright"] = mock_playwright
        router._initialized = True

        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            result = await router.scrape("https://example.com")

            assert result["success"] is True
            assert result["provider"] == "playwright"
            assert result["fallback_from"] == "crawl4ai"

    @pytest.mark.asyncio
    async def test_scrape_all_tools_fail(self, router):
        """Test scraping when all tools fail."""
        # Setup failing adapters
        for name in ["crawl4ai", "browser_use", "playwright"]:
            mock_adapter = AsyncMock()
            mock_adapter.scrape.side_effect = Exception(f"{name} failed")
            router._adapters[name] = mock_adapter

        router._initialized = True

        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            result = await router.scrape("https://example.com")

            assert result["success"] is False
            assert "All automation tools failed" in result["error"]
            assert result["provider"] == "none"
            assert "failed_tools" in result


class TestToolSelection:
    """Test tool selection logic."""

    @pytest.mark.asyncio
    async def test_select_tool_routing_rules(self, router):
        """Test tool selection based on routing rules."""
        router.routing_rules = {
            "browser_use": ["vercel.com"],
            "playwright": ["github.com"],
        }
        router._adapters = {"browser_use": MagicMock(), "playwright": MagicMock()}

        # Test vercel.com -> browser_use
        tool = await router._select_tool("https://vercel.com/docs", False, None)
        assert tool == "browser_use"

        # Test github.com -> playwright
        tool = await router._select_tool("https://github.com/user/repo", False, None)
        assert tool == "playwright"

    @pytest.mark.asyncio
    async def test_select_tool_interaction_required(self, router):
        """Test tool selection when interaction is required."""
        router._adapters = {"browser_use": MagicMock(), "playwright": MagicMock()}

        # Should prefer browser_use for interaction
        tool = await router._select_tool("https://example.com", True, None)
        assert tool == "browser_use"

    @pytest.mark.asyncio
    async def test_select_tool_custom_actions(self, router):
        """Test tool selection with custom actions."""
        router._adapters = {"browser_use": MagicMock(), "playwright": MagicMock()}

        actions = [{"type": "click", "selector": ".button"}]
        tool = await router._select_tool("https://example.com", False, actions)
        assert tool == "browser_use"

    @pytest.mark.asyncio
    async def test_select_tool_spa_patterns(self, router):
        """Test tool selection for SPA patterns."""
        router._adapters = {"browser_use": MagicMock(), "crawl4ai": MagicMock()}

        # Should prefer browser_use for SPAs
        tool = await router._select_tool("https://example.com/spa/app", False, None)
        assert tool == "browser_use"

    @pytest.mark.asyncio
    async def test_select_tool_default_crawl4ai(self, router):
        """Test default tool selection when crawl4ai available."""
        router._adapters = {"crawl4ai": MagicMock()}

        tool = await router._select_tool("https://example.com", False, None)
        assert tool == "crawl4ai"

    @pytest.mark.asyncio
    async def test_select_tool_default_playwright(self, router):
        """Test default tool selection when only playwright available."""
        router._adapters = {"playwright": MagicMock()}

        tool = await router._select_tool("https://example.com", False, None)
        assert tool == "playwright"

    @pytest.mark.asyncio
    async def test_select_tool_default_browser_use(self, router):
        """Test default tool selection when only browser_use available."""
        router._adapters = {"browser_use": MagicMock()}

        tool = await router._select_tool("https://example.com", False, None)
        assert tool == "browser_use"


class TestActionConversion:
    """Test action conversion methods."""

    def test_convert_actions_to_js(self, router):
        """Test converting actions to JavaScript."""
        actions = [
            {"type": "click", "selector": ".button"},
            {"type": "type", "selector": "input", "text": "test"},
            {"type": "wait", "timeout": 1000},
            {"type": "scroll"},
            {"type": "evaluate", "script": "console.log('test')"},
        ]

        js_code = router._convert_actions_to_js(actions)

        assert "document.querySelector('.button')?.click();" in js_code
        assert "document.querySelector('input').value = 'test';" in js_code
        assert "await new Promise(r => setTimeout(r, 1000));" in js_code
        assert "window.scrollTo(0, document.body.scrollHeight);" in js_code
        assert "console.log('test')" in js_code

    def test_convert_to_task(self, router):
        """Test converting actions to natural language task."""
        actions = [
            {"type": "click", "selector": ".button"},
            {"type": "type", "selector": "input", "text": "test"},
            {"type": "wait", "timeout": 1000},
            {"type": "scroll"},
            {"type": "extract"},
            {"type": "expand"},
        ]

        task = router._convert_to_task(actions)

        assert "click on element with selector '.button'" in task
        assert "type 'test' in element with selector 'input'" in task
        assert "wait for 1000 milliseconds" in task
        assert "scroll to the bottom of the page" in task
        assert "extract all visible content" in task
        assert "expand any collapsed sections" in task

    def test_get_basic_js(self, router):
        """Test basic JavaScript generation."""
        js_code = router._get_basic_js("https://example.com")

        assert "await new Promise(r => setTimeout(r, 2000));" in js_code
        assert "aria-expanded" in js_code
        assert "show more" in js_code
        assert "window.scrollTo" in js_code


class TestMetrics:
    """Test metrics functionality."""

    def test_update_metrics_success(self, router):
        """Test updating metrics for successful operation."""
        router._update_metrics("crawl4ai", True, 1.5)

        metrics = router.metrics["crawl4ai"]
        assert metrics["success"] == 1
        assert metrics["failed"] == 0
        assert metrics["_total_time"] == 1.5
        assert metrics["avg_time"] == 1.5

    def test_update_metrics_failure(self, router):
        """Test updating metrics for failed operation."""
        router._update_metrics("crawl4ai", False, 2.0)

        metrics = router.metrics["crawl4ai"]
        assert metrics["success"] == 0
        assert metrics["failed"] == 1
        assert metrics["_total_time"] == 2.0
        assert metrics["avg_time"] == 2.0

    def test_update_metrics_multiple(self, router):
        """Test updating metrics multiple times."""
        router._update_metrics("crawl4ai", True, 1.0)
        router._update_metrics("crawl4ai", False, 2.0)
        router._update_metrics("crawl4ai", True, 3.0)

        metrics = router.metrics["crawl4ai"]
        assert metrics["success"] == 2
        assert metrics["failed"] == 1
        assert metrics["_total_time"] == 6.0
        assert metrics["avg_time"] == 2.0

    def test_get_metrics(self, router):
        """Test getting formatted metrics."""
        router._update_metrics("crawl4ai", True, 1.0)
        router._update_metrics("crawl4ai", False, 2.0)

        formatted_metrics = router.get_metrics()

        assert "crawl4ai" in formatted_metrics
        crawl4ai_metrics = formatted_metrics["crawl4ai"]
        assert crawl4ai_metrics["success"] == 1
        assert crawl4ai_metrics["failed"] == 1
        assert crawl4ai_metrics["success_rate"] == 0.5
        assert crawl4ai_metrics["_total_attempts"] == 2
        assert crawl4ai_metrics["available"] is False  # No adapters in this test

    @pytest.mark.asyncio
    async def test_get_recommended_tool(self, router):
        """Test tool recommendation based on metrics."""
        # Setup adapters
        router._adapters = {"crawl4ai": MagicMock(), "playwright": MagicMock()}

        # Not enough data, should use base recommendation
        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            tool = await router.get_recommended_tool("https://example.com")
            assert tool == "crawl4ai"

        # Add low success rate data
        for _ in range(10):
            router._update_metrics("crawl4ai", False, 1.0)

        # Should recommend different tool due to low success rate
        with patch.object(router, "_select_tool", return_value="crawl4ai"):
            # Add better metrics for playwright
            for _ in range(5):
                router._update_metrics("playwright", True, 1.0)

            tool = await router.get_recommended_tool("https://example.com")
            assert tool == "playwright"


class TestConfigIntegration:
    """Test integration with Pydantic configurations."""

    def test_config_passed_to_adapters(self, mock_unified_config):
        """Test that correct configs are passed to adapters."""
        router = AutomationRouter(mock_unified_config)

        # Verify router has access to all configs
        assert router.config.crawl4ai == mock_unified_config.crawl4ai
        assert router.config.browser_use == mock_unified_config.browser_use
        assert router.config.playwright == mock_unified_config.playwright

    @pytest.mark.asyncio
    async def test_adapter_initialization_with_configs(self, mock_unified_config):
        """Test that adapters are initialized with correct Pydantic configs."""
        router = AutomationRouter(mock_unified_config)

        with (
            patch("src.services.browser.crawl4ai_adapter.Crawl4AIAdapter") as mock_c4ai,
            patch(
                "src.services.browser.browser_use_adapter.BrowserUseAdapter"
            ) as mock_bu,
            patch(
                "src.services.browser.playwright_adapter.PlaywrightAdapter"
            ) as mock_pw,
        ):
            # Setup mocks
            for mock_class in [mock_c4ai, mock_bu, mock_pw]:
                mock_adapter = AsyncMock()
                mock_class.return_value = mock_adapter

            await router.initialize()

            # Verify correct configs were passed
            mock_c4ai.assert_called_once_with(mock_unified_config.crawl4ai)
            mock_bu.assert_called_once_with(mock_unified_config.browser_use)
            mock_pw.assert_called_once_with(mock_unified_config.playwright)
