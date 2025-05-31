"""Additional tests for browser automation coverage."""

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from playwright.async_api import Error as PlaywrightError
from src.config.models import UnifiedConfig
from src.services.browser.action_schemas import ClickAction
from src.services.browser.action_schemas import EvaluateAction
from src.services.browser.action_schemas import HoverAction
from src.services.browser.action_schemas import ScreenshotAction
from src.services.browser.action_schemas import ScrollAction
from src.services.browser.action_schemas import SelectAction
from src.services.browser.action_schemas import TypeAction
from src.services.browser.action_schemas import WaitAction
from src.services.browser.automation_router import AutomationRouter
from src.services.browser.browser_use_adapter import BrowserUseAdapter
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter
from src.services.browser.playwright_adapter import PlaywrightAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    config = MagicMock(spec=UnifiedConfig)

    # Set up crawling config
    config.crawling = MagicMock()
    config.crawling.max_concurrent = 5
    config.crawling.timeout = 30
    config.crawling.headless = True
    config.crawling.javascript_enabled = True
    config.crawling.user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # Set up crawl4ai config as dict
    config.crawl4ai = {
        "max_concurrency": 10,
        "timeout": 30,
        "rate_limit": 60,
        "headless": True,
        "javascript_enabled": True,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    # Set up performance config
    config.performance = MagicMock()
    config.performance.enable_monitoring = True

    return config


class TestCrawl4AIAdapter:
    """Test Crawl4AI adapter coverage."""

    @pytest.fixture
    def adapter(self, mock_config):
        """Create Crawl4AI adapter."""
        return Crawl4AIAdapter(mock_config)

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, adapter):
        """Test initializing already initialized adapter."""
        adapter._initialized = True

        await adapter.initialize()  # Should return early

        assert adapter._initialized

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, adapter):
        """Test scraping when not initialized."""
        adapter._initialized = False

        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_with_actions(self, adapter):
        """Test scraping with browser actions."""
        adapter._initialized = True

        # Mock the provider
        mock_result = {
            "content": "Test content",
            "title": "Test Page",
            "success": True,
        }
        adapter._provider.scrape_url = AsyncMock(return_value=mock_result)

        actions = [
            ClickAction(selector=".button"),
            WaitAction(timeout=1000),
        ]

        result = await adapter.scrape(
            "https://example.com",
            actions=actions,
        )

        assert result["content"] == "Test content"
        adapter._provider.scrape_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_error_handling(self, adapter):
        """Test scrape error handling."""
        adapter._initialized = True
        adapter._provider.scrape_url = AsyncMock(side_effect=Exception("Scrape failed"))

        with pytest.raises(CrawlServiceError, match="Crawl4AI scraping failed"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, adapter):
        """Test cleanup error handling."""
        adapter._provider = MagicMock()
        adapter._provider.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))

        await adapter.cleanup()  # Should not raise, just log error

    @pytest.mark.asyncio
    async def test_execute_browser_actions(self, adapter):
        """Test executing browser actions."""
        adapter._initialized = True

        # Mock provider with action support
        adapter._provider.scrape_url = AsyncMock(
            return_value={"content": "Result", "success": True}
        )

        # Test various action types
        actions = [
            ClickAction(selector="#btn"),
            TypeAction(selector="#input", text="test"),
            WaitAction(timeout=1000),
            ScrollAction(direction="down"),
            EvaluateAction(script="return true;"),
        ]

        result = await adapter.scrape("https://example.com", actions=actions)

        assert result["success"]
        # Verify js_code was generated from actions
        call_args = adapter._provider.scrape_url.call_args[1]
        assert "js_code" in call_args or "extraction_config" in call_args


class TestPlaywrightAdapter:
    """Test Playwright adapter coverage."""

    @pytest.fixture
    def adapter(self, mock_config):
        """Create Playwright adapter."""
        return PlaywrightAdapter(mock_config)

    @pytest.mark.asyncio
    async def test_initialize_browser(self, adapter):
        """Test browser initialization."""
        with patch("playwright.async_api.async_playwright") as mock_playwright:
            mock_browser = MagicMock()
            mock_playwright.return_value.__aenter__.return_value.chromium.launch = (
                AsyncMock(return_value=mock_browser)
            )

            await adapter.initialize()

            assert adapter._browser is not None

    @pytest.mark.asyncio
    async def test_cleanup_no_browser(self, adapter):
        """Test cleanup when no browser exists."""
        adapter._browser = None
        adapter._context = None
        adapter._playwright = None

        await adapter.cleanup()  # Should not raise

    @pytest.mark.asyncio
    async def test_scrape_with_page_actions(self, adapter):
        """Test scraping with page-level actions."""
        adapter._initialized = True

        # Mock page
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>Test Content</html>")
        mock_page.title = AsyncMock(return_value="Test Title")
        mock_page.close = AsyncMock()

        # Mock context
        adapter._context = MagicMock()
        adapter._context.new_page = AsyncMock(return_value=mock_page)

        actions = [
            WaitAction(timeout=2000),
        ]

        result = await adapter.scrape(
            "https://example.com", actions=actions, wait_for_selector=".content"
        )

        assert "Test Content" in result["html"]
        assert result["title"] == "Test Title"
        mock_page.wait_for_selector.assert_called()

    @pytest.mark.asyncio
    async def test_execute_action_handlers(self, adapter):
        """Test individual action handlers."""
        mock_page = MagicMock()

        # Test click action
        mock_page.click = AsyncMock()
        action = ClickAction(selector="#btn")
        await adapter._execute_action(mock_page, action)
        mock_page.click.assert_called_once_with("#btn")

        # Test type action
        mock_page.fill = AsyncMock()
        action = TypeAction(selector="#input", text="test")
        await adapter._execute_action(mock_page, action)
        mock_page.fill.assert_called_once_with("#input", "test")

        # Test select action
        mock_page.select_option = AsyncMock()
        action = SelectAction(selector="#dropdown", value="opt1")
        await adapter._execute_action(mock_page, action)
        mock_page.select_option.assert_called_once_with("#dropdown", "opt1")

        # Test hover action
        mock_page.hover = AsyncMock()
        action = HoverAction(selector="#menu")
        await adapter._execute_action(mock_page, action)
        mock_page.hover.assert_called_once_with("#menu")

        # Test screenshot action
        mock_page.screenshot = AsyncMock(return_value=b"image_data")
        action = ScreenshotAction()
        await adapter._execute_action(mock_page, action)
        mock_page.screenshot.assert_called_once()

        # Test wait action
        action = WaitAction(timeout=500)
        await adapter._execute_action(mock_page, action)  # Should complete

    @pytest.mark.asyncio
    async def test_scrape_error_recovery(self, adapter):
        """Test error recovery during scraping."""
        adapter._initialized = True

        # Mock context that fails on first page creation
        mock_page = MagicMock()
        mock_page.goto = AsyncMock(side_effect=PlaywrightError("Navigation failed"))
        mock_page.close = AsyncMock()

        adapter._context = MagicMock()
        adapter._context.new_page = AsyncMock(return_value=mock_page)

        with pytest.raises(CrawlServiceError, match="Playwright scraping failed"):
            await adapter.scrape("https://example.com")

        mock_page.close.assert_called_once()


class TestBrowserUseAdapter:
    """Test BrowserUse adapter coverage."""

    @pytest.fixture
    def adapter(self, mock_config):
        """Create BrowserUse adapter."""
        # Add LLM provider config
        mock_config.llm_provider = "openai"
        mock_config.openai = MagicMock()
        mock_config.openai.api_key = "test-key"
        return BrowserUseAdapter(mock_config)

    @pytest.mark.asyncio
    async def test_initialize_with_anthropic(self, mock_config):
        """Test initialization with Anthropic provider."""
        mock_config.llm_provider = "anthropic"
        mock_config.anthropic = MagicMock()
        mock_config.anthropic.api_key = "test-key"

        adapter = BrowserUseAdapter(mock_config)

        with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
            await adapter.initialize()
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_google(self, mock_config):
        """Test initialization with Google provider."""
        mock_config.llm_provider = "google"
        mock_config.google = MagicMock()
        mock_config.google.api_key = "test-key"

        adapter = BrowserUseAdapter(mock_config)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_google:
            await adapter.initialize()
            mock_google.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_agent(self, adapter):
        """Test cleanup with active agent."""
        mock_agent = MagicMock()
        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        mock_agent.browser = mock_browser

        adapter._agent = mock_agent

        await adapter.cleanup()

        mock_browser.close.assert_called_once()
        assert adapter._agent is None

    @pytest.mark.asyncio
    async def test_convert_actions_to_task(self, adapter):
        """Test converting actions to natural language task."""
        actions = [
            ClickAction(selector="#submit"),
            TypeAction(selector="#email", text="test@example.com"),
            SelectAction(selector="#country", value="USA"),
            ScrollAction(direction="down"),
            WaitAction(timeout=2000),
        ]

        task = adapter._convert_actions_to_task(actions)

        assert "click on #submit" in task
        assert "type 'test@example.com' into #email" in task
        assert "select 'USA' from #country" in task
        assert "scroll down" in task
        assert "wait 2 seconds" in task

    @pytest.mark.asyncio
    async def test_scrape_with_retries(self, adapter):
        """Test scraping with retry logic."""
        adapter._initialized = True

        # Mock agent that fails twice then succeeds
        mock_agent = MagicMock()
        mock_agent.execute_task = AsyncMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                "Success! Content extracted.",
            ]
        )
        adapter._agent = mock_agent

        # Mock page content
        with patch.object(
            adapter, "_get_page_content", return_value=("HTML", "Title", "URL")
        ):
            result = await adapter.scrape("https://example.com", max_retries=3)

        assert result["success"]
        assert result["content"] == "Success! Content extracted."
        assert mock_agent.execute_task.call_count == 3

    @pytest.mark.asyncio
    async def test_scrape_max_retries_exceeded(self, adapter):
        """Test scraping when max retries exceeded."""
        adapter._initialized = True

        # Mock agent that always fails
        mock_agent = MagicMock()
        mock_agent.execute_task = AsyncMock(side_effect=Exception("Always fails"))
        adapter._agent = mock_agent

        with pytest.raises(CrawlServiceError, match="Max retries"):
            await adapter.scrape("https://example.com", max_retries=2)

        assert mock_agent.execute_task.call_count == 3  # Initial + 2 retries


class TestAutomationRouterCoverage:
    """Test automation router additional coverage."""

    @pytest.fixture
    def router(self, mock_config):
        """Create router instance."""
        return AutomationRouter(mock_config)

    @pytest.mark.asyncio
    async def test_select_adapter_by_url_pattern(self, router):
        """Test adapter selection based on URL patterns."""
        # Load routing rules
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(
                    {
                        "patterns": {
                            "spa_sites": ["react.dev", "angular.io"],
                            "static_sites": ["docs.python.org"],
                            "complex_sites": ["github.com"],
                        },
                        "fallback_order": ["crawl4ai", "browser_use", "playwright"],
                    }
                )
            )
            router._load_routing_rules()

        # Test SPA site
        adapter = router._select_adapter("https://react.dev/docs")
        assert adapter == "browser_use"

        # Test static site
        adapter = router._select_adapter("https://docs.python.org/3/")
        assert adapter == "crawl4ai"

        # Test complex site
        adapter = router._select_adapter("https://github.com/login")
        assert adapter == "playwright"

    @pytest.mark.asyncio
    async def test_fallback_chain_all_fail(self, router):
        """Test fallback chain when all adapters fail."""
        # Mock all adapters to fail
        mock_adapters = {
            "crawl4ai": MagicMock(),
            "browser_use": MagicMock(),
            "playwright": MagicMock(),
        }

        for adapter in mock_adapters.values():
            adapter.initialize = AsyncMock()
            adapter.scrape = AsyncMock(side_effect=CrawlServiceError("Failed"))
            adapter.cleanup = AsyncMock()

        router._adapters = mock_adapters
        router._get_adapter = lambda name: mock_adapters.get(name)

        with pytest.raises(CrawlServiceError, match="All adapters failed"):
            await router.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_with_performance_monitoring(self, router, mock_config):
        """Test scraping with performance monitoring enabled."""
        mock_config.performance.enable_monitoring = True

        # Mock successful adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()
        mock_adapter.scrape = AsyncMock(
            return_value={"content": "Test", "success": True}
        )
        mock_adapter.cleanup = AsyncMock()

        router._adapters["crawl4ai"] = mock_adapter

        result = await router.scrape("https://example.com")

        assert result["success"]
        assert "performance" in result
        assert "total_time" in result["performance"]
        assert "adapter" in result["performance"]

    @pytest.mark.asyncio
    async def test_cleanup_all_adapters(self, router):
        """Test cleaning up all adapters."""
        # Create mock adapters
        mock_adapters = {}
        for name in ["crawl4ai", "browser_use", "playwright"]:
            adapter = MagicMock()
            adapter.cleanup = AsyncMock()
            mock_adapters[name] = adapter

        router._adapters = mock_adapters

        await router.cleanup()

        for adapter in mock_adapters.values():
            adapter.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_scraping(self, router):
        """Test concurrent scraping requests."""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Make scrape method delay to test concurrency
        async def delayed_scrape(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"content": f"Result for {args[0]}", "success": True}

        mock_adapter.scrape = delayed_scrape
        mock_adapter.cleanup = AsyncMock()

        router._adapters["crawl4ai"] = mock_adapter

        # Run multiple concurrent scrapes
        urls = [f"https://example{i}.com" for i in range(5)]
        tasks = [router.scrape(url) for url in urls]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r["success"] for r in results)
        assert mock_adapter.initialize.call_count >= 1
