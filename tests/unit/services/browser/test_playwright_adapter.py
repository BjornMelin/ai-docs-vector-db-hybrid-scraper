"""Comprehensive tests for Playwright browser adapter."""

import asyncio  # noqa: PLC0415
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.config import PlaywrightConfig
from src.services.browser.playwright_adapter import PlaywrightAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def basic_config():
    """Basic configuration for Playwright adapter."""
    return PlaywrightConfig(
        browser="chromium",
        headless=True,
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (compatible; Test/1.0)",
    )


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock(spec=PlaywrightConfig)
    config.browser = "chromium"
    config.headless = True
    config.viewport = {"width": 1920, "height": 1080}
    config.user_agent = "Mozilla/5.0 (compatible; Test/1.0)"
    return config


class TestPlaywrightAdapterInit:
    """Test PlaywrightAdapter initialization."""

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    def test_init_with_basic_config(self, basic_config):
        """Test initialization with basic configuration."""
        adapter = PlaywrightAdapter(basic_config)

        assert adapter._available is True
        assert adapter.config.browser == "chromium"
        assert adapter.config.headless is True
        assert adapter.config.viewport == {"width": 1920, "height": 1080}
        assert "Mozilla/5.0" in adapter.config.user_agent
        assert adapter._initialized is False

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    def test_init_with_mock_config(self, mock_config):
        """Test initialization with mock configuration."""
        adapter = PlaywrightAdapter(mock_config)

        assert adapter.config.browser == "chromium"
        assert adapter.config.headless is True
        assert adapter.config.viewport == {"width": 1920, "height": 1080}

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = PlaywrightConfig()
        adapter = PlaywrightAdapter(config)

        assert adapter.config.browser == "chromium"
        assert adapter.config.headless is True
        assert adapter.config.viewport == {"width": 1920, "height": 1080}
        assert "Mozilla/5.0" in adapter.config.user_agent

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", False)
    def test_init_playwright_unavailable(self, basic_config):
        """Test initialization when Playwright is not available."""
        adapter = PlaywrightAdapter(basic_config)

        assert adapter._available is False

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    def test_init_custom_browser_types(self, _basic_config):
        """Test initialization with different browser types."""
        for browser in ["chromium", "firefox", "webkit"]:
            config = PlaywrightConfig(
                browser=browser,
                headless=True,
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (compatible; Test/1.0)",
            )
            adapter = PlaywrightAdapter(config)
            assert adapter.config.browser == browser


class TestPlaywrightAdapterInitialization:
    """Test adapter initialization process."""

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_initialize_success(self, basic_config):
        """Test successful initialization."""
        adapter = PlaywrightAdapter(basic_config)

        # Mock Playwright components
        mock_playwright = AsyncMock()
        mock_browser_launcher = AsyncMock()
        mock_browser = AsyncMock()

        mock_browser_launcher.launch.return_value = mock_browser
        mock_playwright.chromium = mock_browser_launcher

        with patch(
            "src.services.browser.playwright_adapter.async_playwright"
        ) as mock_async_pw:
            mock_pw_instance = AsyncMock()
            mock_pw_instance.start.return_value = mock_playwright
            mock_async_pw.return_value = mock_pw_instance

            await adapter.initialize()

            assert adapter._initialized is True
            assert adapter._playwright == mock_playwright
            assert adapter._browser == mock_browser

            # Verify browser was launched with correct options
            mock_browser_launcher.launch.assert_called_once()
            launch_args = mock_browser_launcher.launch.call_args
            assert launch_args[1]["headless"] is True
            assert "--no-sandbox" in launch_args[1]["args"]

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", False)
    async def test_initialize_playwright_unavailable(self, basic_config):
        """Test initialization when Playwright is unavailable."""
        adapter = PlaywrightAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="Playwright not available"):
            await adapter.initialize()

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_initialize_already_initialized(self, basic_config):
        """Test re-initialization is skipped."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._initialized = True
        original_browser = adapter._browser

        await adapter.initialize()

        assert adapter._browser == original_browser

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_initialize_browser_launch_failure(self, basic_config):
        """Test initialization failure during browser launch."""
        adapter = PlaywrightAdapter(basic_config)

        mock_playwright = AsyncMock()
        mock_browser_launcher = AsyncMock()
        mock_browser_launcher.launch.side_effect = Exception("Browser launch failed")
        mock_playwright.chromium = mock_browser_launcher

        with patch(
            "src.services.browser.playwright_adapter.async_playwright"
        ) as mock_async_pw:
            mock_pw_instance = AsyncMock()
            mock_pw_instance.start.return_value = mock_playwright
            mock_async_pw.return_value = mock_pw_instance

            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Playwright"
            ):
                await adapter.initialize()

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_initialize_different_browsers(self):
        """Test initialization with different browser types."""
        browsers = ["chromium", "firefox", "webkit"]

        for browser_type in browsers:
            config = PlaywrightConfig(
                browser=browser_type,
                headless=True,
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (compatible; Test/1.0)",
            )
            adapter = PlaywrightAdapter(config)

            mock_playwright = AsyncMock()
            mock_browser_launcher = AsyncMock()
            mock_browser = AsyncMock()
            mock_browser_launcher.launch.return_value = mock_browser
            setattr(mock_playwright, browser_type, mock_browser_launcher)

            with patch(
                "src.services.browser.playwright_adapter.async_playwright"
            ) as mock_async_pw:
                mock_pw_instance = AsyncMock()
                mock_pw_instance.start.return_value = mock_playwright
                mock_async_pw.return_value = mock_pw_instance

                await adapter.initialize()

                assert adapter._initialized is True
                mock_browser_launcher.launch.assert_called_once()


class TestPlaywrightAdapterCleanup:
    """Test adapter cleanup process."""

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_cleanup_success(self, basic_config):
        """Test successful cleanup."""
        adapter = PlaywrightAdapter(basic_config)

        # Setup initialized state
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()
        adapter._browser = mock_browser
        adapter._playwright = mock_playwright
        adapter._initialized = True

        await adapter.cleanup()

        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        assert adapter._browser is None
        assert adapter._playwright is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_cleanup_browser_close_error(self, basic_config):
        """Test cleanup when browser close fails."""
        adapter = PlaywrightAdapter(basic_config)

        # Setup initialized state with failing browser
        mock_browser = AsyncMock()
        mock_browser.close.side_effect = Exception("Close failed")
        mock_playwright = AsyncMock()
        adapter._browser = mock_browser
        adapter._playwright = mock_playwright
        adapter._initialized = True

        # Should not raise exception but cleanup is incomplete due to error
        await adapter.cleanup()

        # Browser cleanup failed so references remain
        assert adapter._browser is not None
        assert adapter._playwright is not None
        assert adapter._initialized is True

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_cleanup_playwright_stop_error(self, basic_config):
        """Test cleanup when playwright stop fails."""
        adapter = PlaywrightAdapter(basic_config)

        # Setup initialized state with failing playwright
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()
        mock_playwright.stop.side_effect = Exception("Stop failed")
        adapter._browser = mock_browser
        adapter._playwright = mock_playwright
        adapter._initialized = True

        # Should not raise exception but cleanup is incomplete due to error
        await adapter.cleanup()

        # Browser cleanup succeeded but playwright stop failed
        assert adapter._browser is None
        assert adapter._playwright is not None
        assert adapter._initialized is True

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_cleanup_not_initialized(self, basic_config):
        """Test cleanup when not initialized."""
        adapter = PlaywrightAdapter(basic_config)

        # Should not raise exception
        await adapter.cleanup()

        assert adapter._browser is None
        assert adapter._playwright is None
        assert adapter._initialized is False


class TestPlaywrightAdapterScraping:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", False)
    async def test_scrape_playwright_unavailable(self, basic_config):
        """Test scraping when Playwright is unavailable."""
        adapter = PlaywrightAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="Playwright not available"):
            await adapter.scrape("https://example.com", [])

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_scrape_not_initialized(self, basic_config):
        """Test scraping when adapter is not initialized."""
        adapter = PlaywrightAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com", [])

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_scrape_success_basic(self, basic_config):
        """Test successful basic scraping."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Setup mocks
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.inner_text.return_value = "Test content"
        mock_page.inner_html.return_value = "<div>Test content</div>"
        mock_page.title.return_value = "Test Page"
        mock_page.evaluate.return_value = {
            "title": "Test Page",
            "description": "Test description",
            "links": [],
            "headings": [],
        }

        adapter._browser = mock_browser

        result = await adapter.scrape("https://example.com", [])

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["content"] == "Test content"
        assert result["html"] == "<div>Test content</div>"
        assert result["title"] == "Test Page"
        assert "metadata" in result
        assert result["metadata"]["extraction_method"] == "playwright"

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_scrape_with_actions(self, basic_config):
        """Test scraping with actions."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Setup mocks
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.wait_for_selector = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.inner_text.return_value = "Updated content"
        mock_page.inner_html.return_value = "<div>Updated content</div>"
        mock_page.title.return_value = "Updated Page"
        mock_page.evaluate.return_value = {
            "title": "Updated Page",
            "links": [],
            "headings": [],
        }

        adapter._browser = mock_browser

        actions = [
            {"type": "click", "selector": "button"},
            {"type": "fill", "selector": "input", "text": "test"},
        ]

        result = await adapter.scrape("https://example.com", actions, timeout=15000)

        assert result["success"] is True
        assert len(result["action_results"]) == 2
        assert all(action["success"] for action in result["action_results"])

        # Verify actions were executed
        mock_page.click.assert_called_once_with("button")
        mock_page.fill.assert_called_once_with("input", "test")

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_scrape_invalid_actions(self, basic_config):
        """Test scraping with invalid actions."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Setup minimal mocks
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto = AsyncMock()

        adapter._browser = mock_browser

        # Invalid action (missing required field)
        invalid_actions = [{"type": "click"}]  # Missing selector

        with patch(
            "src.services.browser.playwright_adapter.validate_actions"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid action", [])

            result = await adapter.scrape("https://example.com", invalid_actions)

            assert result["success"] is False
            assert "Invalid actions" in result["error"]

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_scrape_navigation_failure(self, basic_config):
        """Test scraping when navigation fails."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Setup mocks
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.side_effect = Exception("Navigation failed")

        adapter._browser = mock_browser

        result = await adapter.scrape("https://example.com", [])

        assert result["success"] is False
        assert "Navigation failed" in result["error"]
        assert result["metadata"]["extraction_method"] == "playwright"

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_scrape_action_failure_continues(self, basic_config):
        """Test that scraping continues when individual actions fail."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Setup mocks
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.wait_for_selector = AsyncMock()
        mock_page.click.side_effect = Exception("Click failed")  # First action fails
        mock_page.fill = AsyncMock()  # Second action succeeds
        mock_page.inner_text.return_value = "Content"
        mock_page.inner_html.return_value = "<div>Content</div>"
        mock_page.title.return_value = "Page"
        mock_page.evaluate.return_value = {"title": "Page", "links": [], "headings": []}

        adapter._browser = mock_browser

        actions = [
            {"type": "click", "selector": "button"},
            {"type": "fill", "selector": "input", "text": "test"},
        ]

        result = await adapter.scrape("https://example.com", actions)

        assert result["success"] is True
        assert len(result["action_results"]) == 2
        assert result["action_results"][0]["success"] is False
        assert result["action_results"][1]["success"] is True
        assert result["metadata"]["successful_actions"] == 1


class TestCapabilitiesAndHealth:
    """Test capabilities and health check functionality."""

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    def test_get_capabilities(self, basic_config):
        """Test getting adapter capabilities."""
        adapter = PlaywrightAdapter(basic_config)

        capabilities = adapter.get_capabilities()

        assert capabilities["name"] == "playwright"
        assert "description" in capabilities
        assert "advantages" in capabilities
        assert "limitations" in capabilities
        assert "best_for" in capabilities
        assert "performance" in capabilities
        assert capabilities["javascript_support"] == "complete"
        assert capabilities["cost"] == 0
        assert capabilities["available"] is True
        assert "browsers" in capabilities

    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", False)
    def test_get_capabilities_unavailable(self, basic_config):
        """Test capabilities when Playwright unavailable."""
        adapter = PlaywrightAdapter(basic_config)

        capabilities = adapter.get_capabilities()

        assert capabilities["available"] is False

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", False)
    async def test_health_check_unavailable(self, basic_config):
        """Test health check when Playwright unavailable."""
        adapter = PlaywrightAdapter(basic_config)

        health = await adapter.health_check()

        assert health["healthy"] is False
        assert health["status"] == "unavailable"
        assert health["available"] is False

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_health_check_not_initialized(self, basic_config):
        """Test health check when not initialized."""
        adapter = PlaywrightAdapter(basic_config)

        health = await adapter.health_check()

        assert health["healthy"] is False
        assert health["status"] == "not_initialized"
        assert health["available"] is True

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_health_check_success(self, basic_config):
        """Test successful health check."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._initialized = True

        # Mock successful scrape
        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": True, "content": "test"}

            health = await adapter.health_check()

            assert health["healthy"] is True
            assert health["status"] == "operational"
            assert health["available"] is True
            assert "response_time_ms" in health
            assert health["test_url"] == "https://httpbin.org/html"

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_health_check_timeout(self, basic_config):
        """Test health check timeout."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._initialized = True

        # Mock timeout
        async def timeout_scrape(*_args, **_kwargs):
            await asyncio.sleep(20)  # Longer than timeout

        with patch.object(adapter, "scrape", side_effect=timeout_scrape):
            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "timeout"
            assert health["response_time_ms"] == 15000

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_health_check_error(self, basic_config):
        """Test health check error handling."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._initialized = True

        # Mock error
        with patch.object(adapter, "scrape", side_effect=Exception("Test error")):
            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "error"
            assert "Test error" in health["message"]


class TestComplexInteractionTesting:
    """Test complex interaction testing functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_test_complex_interaction_success(self, basic_config):
        """Test successful complex interaction test."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        mock_result = {
            "success": True,
            "content": "Test content",
            "metadata": {"successful_actions": 6},
            "performance": {"loadTime": 1500},
        }

        with patch.object(adapter, "scrape", return_value=mock_result):
            result = await adapter.test_complex_interaction()

            assert result["success"] is True
            assert result["test_url"] == "https://example.com"
            assert result["actions_count"] == 6
            assert result["successful_actions"] == 6
            assert result["content_length"] == len("Test content")
            assert "execution_time_ms" in result

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_test_complex_interaction_not_available(self, basic_config):
        """Test complex interaction when adapter not available."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = False

        result = await adapter.test_complex_interaction()

        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_test_complex_interaction_error(self, basic_config):
        """Test complex interaction error handling."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        with patch.object(adapter, "scrape", side_effect=Exception("Test failed")):
            result = await adapter.test_complex_interaction()

            assert result["success"] is False
            assert "Test failed" in result["error"]
            assert "execution_time_ms" in result


class TestIntegration:
    """Integration tests for PlaywrightAdapter."""

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_full_workflow(self, basic_config):
        """Test complete workflow from initialization to cleanup."""
        adapter = PlaywrightAdapter(basic_config)

        # Mock Playwright components
        mock_playwright = AsyncMock()
        mock_browser_launcher = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        # Setup chain of mocks
        mock_browser_launcher.launch.return_value = mock_browser
        mock_playwright.chromium = mock_browser_launcher
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # Setup page responses
        mock_page.goto = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.wait_for_selector = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.inner_text.return_value = "Page content"
        mock_page.inner_html.return_value = "<div>Page content</div>"
        mock_page.title.return_value = "Page Title"
        mock_page.evaluate.return_value = {
            "title": "Page Title",
            "description": "Page description",
            "links": [],
            "headings": [],
        }

        with patch(
            "src.services.browser.playwright_adapter.async_playwright"
        ) as mock_async_pw:
            mock_pw_instance = AsyncMock()
            mock_pw_instance.start.return_value = mock_playwright
            mock_async_pw.return_value = mock_pw_instance

            # Initialize
            await adapter.initialize()
            assert adapter._initialized is True

            # Scrape with actions
            actions = [
                {"type": "click", "selector": "button"},
                {"type": "wait", "timeout": 1000},
                {"type": "screenshot", "path": "test.png"},
            ]

            result = await adapter.scrape("https://example.com", actions)

            assert result["success"] is True
            assert result["content"] == "Page content"
            assert len(result["action_results"]) == 3

            # Health check
            health = await adapter.health_check()
            assert health["healthy"] is True

            # Cleanup
            await adapter.cleanup()
            assert adapter._initialized is False
            mock_browser.close.assert_called_once()
            mock_playwright.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.playwright_adapter.PLAYWRIGHT_AVAILABLE", True)
    async def test_error_resilience(self, basic_config):
        """Test adapter resilience to various errors."""
        adapter = PlaywrightAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Test with various failing scenarios
        scenarios = [
            {
                "action": {"type": "click", "selector": "nonexistent"},
                "error": "Element not found",
            },
            {
                "action": {"type": "fill", "selector": "readonly", "text": "test"},
                "error": "Read-only element",
            },
            {
                "action": {"type": "evaluate", "script": "invalid.script"},
                "error": "JavaScript error",
            },
        ]

        for scenario in scenarios:
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            mock_page.goto = AsyncMock()
            mock_page.url = "https://example.com"

            # Make the specific action fail
            if scenario["action"]["type"] == "click":
                mock_page.click.side_effect = Exception(scenario["error"])
            elif scenario["action"]["type"] == "fill":
                mock_page.fill.side_effect = Exception(scenario["error"])
            elif scenario["action"]["type"] == "evaluate":
                mock_page.evaluate.side_effect = Exception(scenario["error"])

            # Setup content extraction to succeed
            mock_page.inner_text.return_value = "Content"
            mock_page.inner_html.return_value = "<div>Content</div>"
            mock_page.title.return_value = "Title"
            mock_page.evaluate.return_value = {
                "title": "Title",
                "links": [],
                "headings": [],
            }

            adapter._browser = mock_browser

            result = await adapter.scrape("https://example.com", [scenario["action"]])

            # Scraping should still succeed overall
            assert result["success"] is True
            # But the failing action should be recorded
            assert len(result["action_results"]) == 1
            assert result["action_results"][0]["success"] is False
            assert scenario["error"] in result["action_results"][0]["error"]
