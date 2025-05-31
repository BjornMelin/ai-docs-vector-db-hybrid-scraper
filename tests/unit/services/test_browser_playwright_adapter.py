"""Comprehensive tests for Playwright adapter."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.browser.action_schemas import ClickAction
from src.services.browser.action_schemas import DragAndDropAction
from src.services.browser.action_schemas import EvaluateAction
from src.services.browser.action_schemas import FillAction
from src.services.browser.action_schemas import HoverAction
from src.services.browser.action_schemas import PressAction
from src.services.browser.action_schemas import ScreenshotAction
from src.services.browser.action_schemas import ScrollAction
from src.services.browser.action_schemas import SelectAction
from src.services.browser.action_schemas import TypeAction
from src.services.browser.action_schemas import WaitAction
from src.services.browser.action_schemas import WaitForLoadStateAction
from src.services.browser.action_schemas import WaitForSelectorAction
from src.services.browser.playwright_adapter import PlaywrightAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        "browser": "chromium",
        "headless": True,
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": "Mozilla/5.0 (Playwright Test)",
        "locale": "en-US",
        "timezone": "America/New_York",
        "permissions": ["geolocation"],
        "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
    }


@pytest.fixture
def adapter(mock_config):
    """Create PlaywrightAdapter instance for testing."""
    return PlaywrightAdapter(mock_config)


@pytest.fixture
def mock_page():
    """Create mock Playwright page."""
    page = AsyncMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    page.title = AsyncMock(return_value="Test Page")
    page.url = "https://example.com"
    page.evaluate = AsyncMock(return_value={"content": "Test content"})
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.type = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"screenshot_data")
    page.hover = AsyncMock()
    page.select_option = AsyncMock()
    page.press = AsyncMock()
    page.drag_and_drop = AsyncMock()
    page.mouse = MagicMock()
    page.mouse.wheel = AsyncMock()
    return page


@pytest.fixture
def mock_context():
    """Create mock browser context."""
    context = AsyncMock()
    context.new_page = AsyncMock()
    context.close = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=context)
    context.__aexit__ = AsyncMock(return_value=None)
    return context


@pytest.fixture
def mock_browser():
    """Create mock browser instance."""
    browser = AsyncMock()
    browser.new_context = AsyncMock()
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_playwright():
    """Create mock playwright instance."""
    playwright = MagicMock()
    playwright.chromium = MagicMock()
    playwright.firefox = MagicMock()
    playwright.webkit = MagicMock()
    playwright.__aenter__ = AsyncMock(return_value=playwright)
    playwright.__aexit__ = AsyncMock(return_value=None)
    return playwright


class TestPlaywrightAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_initialization(self, adapter, mock_config):
        """Test basic adapter initialization."""
        assert adapter.config == mock_config
        assert adapter._browser is None
        assert adapter._initialized is False
        assert adapter.name == "playwright"
        assert adapter._available is True

    def test_adapter_unavailable(self):
        """Test adapter when playwright is not installed."""
        with patch.dict("sys.modules", {"playwright": None}):
            adapter = PlaywrightAdapter({})
            assert adapter._available is False

    def test_browser_type_selection(self):
        """Test different browser type configurations."""
        browsers = ["chromium", "firefox", "webkit"]
        for browser in browsers:
            config = {"browser": browser}
            adapter = PlaywrightAdapter(config)
            assert adapter.config["browser"] == browser

    @pytest.mark.asyncio
    async def test_initialize_success(self, adapter, mock_browser, mock_playwright):
        """Test successful adapter initialization."""
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

        with patch(
            "playwright.async_api.async_playwright", return_value=mock_playwright
        ):
            await adapter.initialize()

        assert adapter._initialized is True
        assert adapter._browser is mock_browser
        assert adapter._playwright is mock_playwright

        mock_playwright.chromium.launch.assert_called_once_with(
            headless=True, args=["--disable-blink-features=AutomationControlled"]
        )

    @pytest.mark.asyncio
    async def test_initialize_different_browsers(self, mock_browser, mock_playwright):
        """Test initialization with different browser types."""
        browsers = {
            "chromium": mock_playwright.chromium,
            "firefox": mock_playwright.firefox,
            "webkit": mock_playwright.webkit,
        }

        for browser_type, browser_attr in browsers.items():
            adapter = PlaywrightAdapter({"browser": browser_type, "headless": True})
            browser_attr.launch = AsyncMock(return_value=mock_browser)

            with patch(
                "playwright.async_api.async_playwright", return_value=mock_playwright
            ):
                await adapter.initialize()

            browser_attr.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, adapter, mock_playwright):
        """Test initialization failure handling."""
        mock_playwright.chromium.launch = AsyncMock(
            side_effect=Exception("Launch failed")
        )

        with patch(
            "playwright.async_api.async_playwright", return_value=mock_playwright
        ):
            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Playwright"
            ):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, adapter, mock_browser, mock_playwright):
        """Test adapter cleanup."""
        adapter._browser = mock_browser
        adapter._playwright = mock_playwright
        adapter._initialized = True

        await adapter.cleanup()

        mock_browser.close.assert_called_once()
        assert adapter._browser is None
        assert adapter._playwright is None
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

        with pytest.raises(CrawlServiceError, match="Playwright not available"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, adapter):
        """Test scraping when adapter not initialized."""
        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_success_basic(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test successful basic scraping."""
        url = "https://example.com"

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # Mock page evaluate response
        mock_page.evaluate.return_value = {
            "title": "Example Page",
            "description": "Test description",
            "author": "Test Author",
            "keywords": "test, example",
            "canonical": "https://example.com",
            "content": "This is the main content of the page.",
            "links": [
                {"href": "https://example.com/page1", "text": "Page 1"},
                {"href": "https://example.com/page2", "text": "Page 2"},
            ],
        }

        adapter._browser = mock_browser
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["success"] is True
        assert result["content"] == "This is the main content of the page."
        assert result["url"] == url
        assert result["title"] == "Example Page"
        assert result["html"] == "<html><body>Test</body></html>"
        assert result["metadata"]["extraction_method"] == "playwright"
        assert result["metadata"]["title"] == "Example Page"
        assert result["metadata"]["description"] == "Test description"
        assert result["metadata"]["author"] == "Test Author"
        assert len(result["links"]) == 2

        mock_page.goto.assert_called_once_with(
            url, wait_until="networkidle", timeout=30000
        )

    @pytest.mark.asyncio
    async def test_scrape_with_actions(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test scraping with custom actions."""
        url = "https://example.com"
        actions = [
            {"type": "click", "selector": "#expand-button"},
            {"type": "wait", "timeout": 1000},
            {"type": "fill", "selector": "#search", "text": "test query"},
            {"type": "press", "key": "Enter"},
            {"type": "wait_for_selector", "selector": ".results"},
        ]

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = {"content": "Search results", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        result = await adapter.scrape(url, actions=actions)

        assert result["success"] is True

        # Verify actions were executed
        mock_page.click.assert_called_once_with("#expand-button")
        mock_page.wait_for_timeout.assert_called_once_with(1000)
        mock_page.fill.assert_called_once_with("#search", "test query")
        mock_page.press.assert_called_once_with("Enter")
        mock_page.wait_for_selector.assert_called_once_with(".results", timeout=5000)

    @pytest.mark.asyncio
    async def test_scrape_with_timeout(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test scraping with custom timeout."""
        url = "https://example.com"
        timeout = 60000

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = {"content": "Content", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        await adapter.scrape(url, timeout=timeout)

        mock_page.goto.assert_called_once_with(
            url, wait_until="networkidle", timeout=60000
        )

    @pytest.mark.asyncio
    async def test_scrape_error_handling(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test error handling during scrape."""
        url = "https://error-site.com"

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.side_effect = Exception(
            "Navigation failed: net::ERR_CONNECTION_REFUSED"
        )

        adapter._browser = mock_browser
        adapter._initialized = True

        result = await adapter.scrape(url)

        assert result["success"] is False
        assert "Navigation failed" in result["error"]
        assert result["url"] == url


class TestActionExecution:
    """Test individual action execution."""

    @pytest.mark.asyncio
    async def test_execute_click_action(self, adapter, mock_page):
        """Test click action execution."""
        action = ClickAction(selector="#button")

        await adapter._execute_action(mock_page, action)

        mock_page.click.assert_called_once_with("#button")

    @pytest.mark.asyncio
    async def test_execute_fill_action(self, adapter, mock_page):
        """Test fill action execution."""
        action = FillAction(selector="input[name='email']", text="test@example.com")

        await adapter._execute_action(mock_page, action)

        mock_page.fill.assert_called_once_with(
            "input[name='email']", "test@example.com"
        )

    @pytest.mark.asyncio
    async def test_execute_type_action(self, adapter, mock_page):
        """Test type action execution."""
        action = TypeAction(selector=".search", text="search query")

        await adapter._execute_action(mock_page, action)

        mock_page.type.assert_called_once_with(".search", "search query")

    @pytest.mark.asyncio
    async def test_execute_wait_action(self, adapter, mock_page):
        """Test wait action execution."""
        action = WaitAction(timeout=2000)

        await adapter._execute_action(mock_page, action)

        mock_page.wait_for_timeout.assert_called_once_with(2000)

    @pytest.mark.asyncio
    async def test_execute_wait_for_selector_action(self, adapter, mock_page):
        """Test wait for selector action execution."""
        action = WaitForSelectorAction(selector=".loaded", timeout=10000)

        await adapter._execute_action(mock_page, action)

        mock_page.wait_for_selector.assert_called_once_with(".loaded", timeout=10000)

    @pytest.mark.asyncio
    async def test_execute_wait_for_load_state_action(self, adapter, mock_page):
        """Test wait for load state action execution."""
        action = WaitForLoadStateAction(state="domcontentloaded")

        await adapter._execute_action(mock_page, action)

        mock_page.wait_for_load_state.assert_called_once_with("domcontentloaded")

    @pytest.mark.asyncio
    async def test_execute_scroll_action_bottom(self, adapter, mock_page):
        """Test scroll to bottom action."""
        action = ScrollAction(direction="bottom")

        await adapter._execute_action(mock_page, action)

        mock_page.evaluate.assert_called_once_with(
            "window.scrollTo(0, document.body.scrollHeight)"
        )

    @pytest.mark.asyncio
    async def test_execute_scroll_action_top(self, adapter, mock_page):
        """Test scroll to top action."""
        action = ScrollAction(direction="top")

        await adapter._execute_action(mock_page, action)

        mock_page.evaluate.assert_called_once_with("window.scrollTo(0, 0)")

    @pytest.mark.asyncio
    async def test_execute_scroll_action_position(self, adapter, mock_page):
        """Test scroll to position action."""
        action = ScrollAction(direction="position", y=500)

        await adapter._execute_action(mock_page, action)

        mock_page.evaluate.assert_called_once_with("window.scrollTo(0, 500)")

    @pytest.mark.asyncio
    async def test_execute_screenshot_action(self, adapter, mock_page):
        """Test screenshot action execution."""
        action = ScreenshotAction(path="/tmp/screenshot.png", full_page=True)

        await adapter._execute_action(mock_page, action)

        mock_page.screenshot.assert_called_once_with(
            path="/tmp/screenshot.png", full_page=True
        )

    @pytest.mark.asyncio
    async def test_execute_evaluate_action(self, adapter, mock_page):
        """Test evaluate action execution."""
        action = EvaluateAction(script="document.querySelectorAll('.item').length")

        await adapter._execute_action(mock_page, action)

        mock_page.evaluate.assert_called_once_with(
            "document.querySelectorAll('.item').length"
        )

    @pytest.mark.asyncio
    async def test_execute_hover_action(self, adapter, mock_page):
        """Test hover action execution."""
        action = HoverAction(selector=".tooltip-trigger")

        await adapter._execute_action(mock_page, action)

        mock_page.hover.assert_called_once_with(".tooltip-trigger")

    @pytest.mark.asyncio
    async def test_execute_select_action(self, adapter, mock_page):
        """Test select action execution."""
        action = SelectAction(selector="select#country", value="US")

        await adapter._execute_action(mock_page, action)

        mock_page.select_option.assert_called_once_with("select#country", "US")

    @pytest.mark.asyncio
    async def test_execute_press_action(self, adapter, mock_page):
        """Test press action execution."""
        action = PressAction(key="Enter", selector="input#search")

        await adapter._execute_action(mock_page, action)

        mock_page.press.assert_called_once_with("input#search", "Enter")

    @pytest.mark.asyncio
    async def test_execute_press_action_no_selector(self, adapter, mock_page):
        """Test press action without selector."""
        action = PressAction(key="Escape")

        await adapter._execute_action(mock_page, action)

        mock_page.press.assert_called_once_with("body", "Escape")

    @pytest.mark.asyncio
    async def test_execute_drag_and_drop_action(self, adapter, mock_page):
        """Test drag and drop action execution."""
        action = DragAndDropAction(source="#draggable", target="#droppable")

        await adapter._execute_action(mock_page, action)

        mock_page.drag_and_drop.assert_called_once_with("#draggable", "#droppable")

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, adapter, mock_page):
        """Test handling of unknown action type."""
        # Create a mock action with unknown type
        action = MagicMock()
        action.type = "unknown_action"

        with pytest.raises(ValueError, match="Unknown action type: unknown_action"):
            await adapter._execute_action(mock_page, action)


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
        adapter._browser = AsyncMock()
        adapter._initialized = True

        mock_result = {
            "success": True,
            "content": "Test page loaded",
        }

        with patch.object(adapter, "scrape", return_value=mock_result):
            health = await adapter.health_check()

        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert health["available"] is True
        assert "response_time_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, adapter):
        """Test health check timeout handling."""
        adapter._browser = AsyncMock()
        adapter._initialized = True

        async def slow_scrape(*args, **kwargs):
            await asyncio.sleep(16)  # Longer than timeout

        with patch.object(adapter, "scrape", side_effect=slow_scrape):
            health = await adapter.health_check()

        assert health["status"] == "timeout"
        assert health["healthy"] is False
        assert health["response_time_ms"] == 15000

    @pytest.mark.asyncio
    async def test_health_check_error(self, adapter):
        """Test health check error handling."""
        adapter._browser = AsyncMock()
        adapter._initialized = True

        with patch.object(
            adapter, "scrape", side_effect=Exception("Health check failed")
        ):
            health = await adapter.health_check()

        assert health["status"] == "error"
        assert health["healthy"] is False
        assert "Health check failed" in health["error"]


class TestCapabilities:
    """Test capability reporting."""

    def test_get_capabilities(self, adapter):
        """Test capability information."""
        capabilities = adapter.get_capabilities()

        assert capabilities["name"] == "playwright"
        assert capabilities["full_control"] is True
        assert "Complete programmatic control" in capabilities["advantages"][0]
        assert capabilities["performance"]["avg_speed"] == "0.5s per page"
        assert capabilities["performance"]["success_rate"] == "99%+"

        # Check all expected fields
        assert "advantages" in capabilities
        assert "limitations" in capabilities
        assert "best_for" in capabilities


class TestAdvancedFeatures:
    """Test advanced Playwright features."""

    @pytest.mark.asyncio
    async def test_scrape_with_custom_viewport(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test scraping with custom viewport."""
        adapter.config["viewport"] = {"width": 1366, "height": 768}

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = {"content": "Content", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        await adapter.scrape("https://example.com")

        # Verify context created with viewport
        context_args = mock_browser.new_context.call_args.kwargs
        assert context_args["viewport"] == {"width": 1366, "height": 768}

    @pytest.mark.asyncio
    async def test_scrape_with_custom_headers(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test scraping with custom headers."""
        adapter.config["extra_http_headers"] = {
            "Accept-Language": "fr-FR,fr;q=0.9",
            "X-Custom-Header": "test",
        }

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = {"content": "Content", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        await adapter.scrape("https://example.com")

        # Verify context created with headers
        context_args = mock_browser.new_context.call_args.kwargs
        assert context_args["extra_http_headers"]["Accept-Language"] == "fr-FR,fr;q=0.9"
        assert context_args["extra_http_headers"]["X-Custom-Header"] == "test"

    @pytest.mark.asyncio
    async def test_scrape_with_permissions(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test scraping with browser permissions."""
        adapter.config["permissions"] = ["geolocation", "notifications"]

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = {"content": "Content", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        await adapter.scrape("https://example.com")

        # Verify context created with permissions
        context_args = mock_browser.new_context.call_args.kwargs
        assert "geolocation" in context_args["permissions"]
        assert "notifications" in context_args["permissions"]

    @pytest.mark.asyncio
    async def test_complex_action_workflow(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test complex multi-action workflow."""
        url = "https://complex-app.com"
        actions = [
            {"type": "wait_for_load_state", "state": "networkidle"},
            {"type": "click", "selector": "#cookie-accept"},
            {"type": "wait", "timeout": 500},
            {"type": "hover", "selector": "#menu"},
            {"type": "click", "selector": "#menu .dropdown-item"},
            {"type": "wait_for_selector", "selector": ".form-container"},
            {"type": "fill", "selector": "#name", "text": "John Doe"},
            {"type": "fill", "selector": "#email", "text": "john@example.com"},
            {"type": "select", "selector": "#country", "value": "US"},
            {"type": "click", "selector": "#terms"},
            {"type": "screenshot", "path": "/tmp/form.png"},
            {"type": "click", "selector": "#submit"},
            {"type": "wait_for_selector", "selector": ".success-message"},
        ]

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = {"content": "Success!", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        result = await adapter.scrape(url, actions=actions)

        assert result["success"] is True

        # Verify all actions were executed
        assert mock_page.wait_for_load_state.called
        assert mock_page.click.call_count >= 3
        assert mock_page.fill.call_count == 2
        assert mock_page.hover.called
        assert mock_page.select_option.called
        assert mock_page.screenshot.called

    @pytest.mark.asyncio
    async def test_javascript_execution_in_scrape(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test JavaScript execution during scraping."""
        url = "https://spa-app.com"
        actions = [
            {
                "type": "evaluate",
                "script": "localStorage.setItem('token', 'test-token')",
            },
            {
                "type": "evaluate",
                "script": "document.querySelector('#dynamic-content').click()",
            },
            {"type": "wait", "timeout": 1000},
            {
                "type": "evaluate",
                "script": "return document.querySelector('.result').textContent",
            },
        ]

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # Set up evaluate return values
        evaluate_returns = [None, None, None, "Dynamic result text"]
        mock_page.evaluate.side_effect = evaluate_returns

        adapter._browser = mock_browser
        adapter._initialized = True

        result = await adapter.scrape(url, actions=actions)

        # Verify JavaScript was executed
        assert mock_page.evaluate.call_count >= 4  # 3 from actions + extraction


class TestErrorScenarios:
    """Test various error scenarios."""

    @pytest.mark.asyncio
    async def test_handle_navigation_timeout(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test handling of navigation timeout."""
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.side_effect = TimeoutError("Navigation timeout")

        adapter._browser = mock_browser
        adapter._initialized = True

        result = await adapter.scrape("https://slow-site.com")

        assert result["success"] is False
        assert "Navigation timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_selector_not_found(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test handling of selector not found errors."""
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.click.side_effect = Exception("Element not found: #missing-button")
        mock_page.evaluate.return_value = {"content": "Partial content", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        actions = [{"type": "click", "selector": "#missing-button"}]

        # Should handle error and return partial result
        result = await adapter.scrape("https://example.com", actions=actions)

        assert result["success"] is False
        assert "Element not found" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_context_closed(
        self, adapter, mock_browser, mock_context, mock_page
    ):
        """Test handling of context closed errors."""
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_context.__aexit__.side_effect = Exception("Context already closed")
        mock_page.evaluate.return_value = {"content": "Content", "links": []}

        adapter._browser = mock_browser
        adapter._initialized = True

        # Should still return result even if context cleanup fails
        result = await adapter.scrape("https://example.com")

        # Result might be successful if error only in cleanup
        assert "content" in result.lower() or "error" in result
