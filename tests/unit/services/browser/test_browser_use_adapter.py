"""Comprehensive tests for BrowserUse AI-powered browser adapter."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import BrowserUseConfig
from src.services.browser.browser_use_adapter import BrowserUseAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def basic_config():
    """Basic configuration for BrowserUse adapter."""
    return BrowserUseConfig(
        llm_provider="openai",
        model="gpt-4o-mini",
        headless=True,
        timeout=30000,
        max_retries=3,
        max_steps=20,
        disable_security=False,
        generate_gif=False,
    )


@pytest.fixture
def anthropic_config():
    """Configuration for Anthropic provider."""
    return BrowserUseConfig(
        llm_provider="anthropic",
        model="claude-3-haiku-20240307",
        headless=True,
        timeout=30000,
    )


@pytest.fixture
def gemini_config():
    """Configuration for Gemini provider."""
    return BrowserUseConfig(
        llm_provider="gemini",
        model="gemini-pro",
        headless=True,
        timeout=30000,
    )


class TestBrowserUseAdapterInit:
    """Test BrowserUseAdapter initialization."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_with_basic_config(self, basic_config):
        """Test initialization with basic configuration."""
        adapter = BrowserUseAdapter(basic_config)

        assert adapter._available is True
        assert adapter.config.llm_provider == "openai"
        assert adapter.config.model == "gpt-4o-mini"
        assert adapter.config.headless is True
        assert adapter.config.timeout == 30000
        assert adapter.config.max_retries == 3
        assert adapter.config.max_steps == 20
        assert adapter.config.disable_security is False
        assert adapter.config.generate_gif is False
        assert adapter._initialized is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    def test_init_browser_use_unavailable(self, basic_config):
        """Test initialization when browser-use is not available."""
        adapter = BrowserUseAdapter(basic_config)

        assert adapter._available is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = BrowserUseConfig()
        adapter = BrowserUseAdapter(config)

        assert adapter.config.llm_provider == "openai"
        assert adapter.config.model == "gpt-4o-mini"
        assert adapter.config.headless is True
        assert adapter.config.timeout == 30000
        assert adapter.config.max_retries == 3
        assert adapter.config.max_steps == 20
        assert adapter.config.disable_security is False
        assert adapter.config.generate_gif is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_custom_llm_providers(self):
        """Test initialization with different LLM providers."""
        providers = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-opus-20240229"),
            ("gemini", "gemini-1.5-pro"),
        ]

        for provider, model in providers:
            config = BrowserUseConfig(llm_provider=provider, model=model)
            adapter = BrowserUseAdapter(config)
            assert adapter.config.llm_provider == provider
            assert adapter.config.model == model

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_custom_browser_settings(self):
        """Test initialization with custom browser settings."""
        config = BrowserUseConfig(
            headless=False,
            disable_security=True,
            generate_gif=True,
            timeout=60000,
            max_retries=5,
            max_steps=50,
        )
        adapter = BrowserUseAdapter(config)

        assert adapter.config.headless is False
        assert adapter.config.disable_security is True
        assert adapter.config.generate_gif is True
        assert adapter.config.timeout == 60000
        assert adapter.config.max_retries == 5
        assert adapter.config.max_steps == 50


class TestBrowserUseAdapterLLMSetup:
    """Test LLM configuration setup."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.ChatOpenAI")
    def test_setup_llm_config_openai(self, mock_chat_openai, basic_config):
        """Test OpenAI LLM setup."""
        adapter = BrowserUseAdapter(basic_config)

        # Mock environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            adapter._setup_llm_config()

            mock_chat_openai.assert_called_once_with(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key="test-api-key",
            )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_setup_llm_config_openai_no_api_key(self, basic_config):
        """Test OpenAI setup without API key."""
        adapter = BrowserUseAdapter(basic_config)

        # Clear environment variable
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(CrawlServiceError, match="OPENAI_API_KEY"),
        ):
            adapter._setup_llm_config()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.ChatAnthropic")
    def test_setup_llm_config_anthropic(self, mock_chat_anthropic, anthropic_config):
        """Test Anthropic LLM setup."""
        adapter = BrowserUseAdapter(anthropic_config)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}):
            adapter._setup_llm_config()

            mock_chat_anthropic.assert_called_once_with(
                model="claude-3-haiku-20240307",
                temperature=0.1,
                api_key="test-api-key",
            )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.ChatGoogleGenerativeAI")
    def test_setup_llm_config_gemini(self, mock_chat_gemini, gemini_config):
        """Test Gemini LLM setup."""
        adapter = BrowserUseAdapter(gemini_config)

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
            adapter._setup_llm_config()

            mock_chat_gemini.assert_called_once_with(
                model="gemini-pro",
                temperature=0.1,
                google_api_key="test-api-key",
            )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_setup_llm_config_unsupported_provider(self):
        """Test unsupported LLM provider."""
        config = BrowserUseConfig(llm_provider="unsupported")
        adapter = BrowserUseAdapter(config)

        with pytest.raises(CrawlServiceError, match="Unsupported LLM provider"):
            adapter._setup_llm_config()


class TestBrowserUseAdapterInitialization:
    """Test BrowserUse adapter initialization process."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    async def test_initialize_browser_use_unavailable(self, basic_config):
        """Test initialization when browser-use is not available."""
        adapter = BrowserUseAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="browser-use not available"):
            await adapter.initialize()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Browser")
    @patch("src.services.browser.browser_use_adapter.BrowserConfig")
    @patch("src.services.browser.browser_use_adapter.ChatOpenAI")
    async def test_initialize_success(
        self, mock_chat_openai, mock_browser_config, mock_browser, basic_config
    ):
        """Test successful initialization."""
        adapter = BrowserUseAdapter(basic_config)

        # Mock LLM setup
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock browser setup
        mock_browser_instance = MagicMock()
        mock_browser.return_value = mock_browser_instance

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            await adapter.initialize()

            # Verify LLM setup
            mock_chat_openai.assert_called_once()

            # Verify browser config
            mock_browser_config.assert_called_once_with(
                headless=True,
                disable_security=False,
            )

            # Verify browser creation
            mock_browser.assert_called_once()

            assert adapter._initialized is True
            assert adapter.llm_config == mock_llm
            assert adapter._browser == mock_browser_instance

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.ChatOpenAI")
    async def test_initialize_already_initialized(self, mock_chat_openai, basic_config):
        """Test initialization when already initialized."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        await adapter.initialize()

        # Should not setup LLM again
        mock_chat_openai.assert_not_called()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.ChatOpenAI")
    async def test_initialize_llm_setup_failure(self, mock_chat_openai, basic_config):
        """Test initialization failure during LLM setup."""
        adapter = BrowserUseAdapter(basic_config)

        # Mock LLM setup failure
        mock_chat_openai.side_effect = Exception("LLM setup failed")

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            pytest.raises(CrawlServiceError, match="Failed to initialize"),
        ):
            await adapter.initialize()


class TestBrowserUseAdapterScraping:
    """Test BrowserUse scraping functionality."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_not_initialized(self, basic_config):
        """Test scraping when not initialized."""
        adapter = BrowserUseAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="not initialized"):
            await adapter.scrape("https://example.com", "Extract title")

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Agent")
    async def test_scrape_simple_success(self, mock_agent_class, basic_config):
        """Test successful simple scraping."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = None
        mock_agent_class.return_value = mock_agent

        # Mock browser context
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value="<html><title>Test</title></html>")
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.url = "https://example.com"
        mock_context.current_page = mock_page
        adapter._browser.context = mock_context

        result = await adapter.scrape("https://example.com", "Extract title")

        assert result["success"] is True
        assert "Test Page" in result["content"]  # Title should be in content
        assert result["html"] == "<html><title>Test</title></html>"
        assert result["title"] == "Test Page"
        assert result["metadata"]["title"] == "Test Page"
        assert result["url"] == "https://example.com"

        # Verify agent was created and run
        mock_agent_class.assert_called_once()
        # Check the call arguments
        call_args = mock_agent_class.call_args[1]
        assert "Extract title" in call_args["task"]
        assert call_args["llm"] == adapter.llm_config
        assert call_args["browser"] == adapter._browser
        assert call_args["max_steps"] == adapter.config.max_steps
        assert call_args["generate_gif"] == adapter.config.generate_gif
        mock_agent.run.assert_called_once()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Agent")
    async def test_scrape_with_instructions(self, mock_agent_class, basic_config):
        """Test scraping with action instructions."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = None
        mock_agent_class.return_value = mock_agent

        # Mock browser context
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value="<html>Content</html>")
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_context.current_page = mock_page
        adapter._browser.context = mock_context

        instructions = [
            {"action": "click", "selector": "button"},
            {"action": "type", "selector": "input", "text": "test"},
        ]

        await adapter.scrape(
            "https://example.com", "Extract data", instructions=instructions
        )

        # Check that task was properly formatted
        mock_agent_class.assert_called_once()
        actual_task = mock_agent_class.call_args[1]["task"]

        # Check that the task contains the formatted instructions
        assert "Extract data" in actual_task
        assert "Please perform these actions:" in actual_task
        assert "1. Click on element: button" in actual_task
        assert "2. Type 'test' into element: input" in actual_task

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Agent")
    async def test_scrape_with_retry(self, mock_agent_class, basic_config):
        """Test scraping with retry on failure."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock agent that fails first time, succeeds second
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = [
            Exception("First attempt failed"),
            None,  # Success on second attempt
        ]
        mock_agent_class.return_value = mock_agent

        # Mock browser context
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value="<html>Success</html>")
        mock_page.title = AsyncMock(return_value="Success")
        mock_page.url = "https://example.com"
        mock_context.current_page = mock_page
        adapter._browser.context = mock_context

        result = await adapter.scrape("https://example.com", "Extract data")

        assert result["success"] is True
        assert mock_agent.run.call_count == 2  # Called twice due to retry

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Agent")
    async def test_scrape_all_retries_failed(self, mock_agent_class, basic_config):
        """Test scraping when all retries fail."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock agent that always fails
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("Always fails")
        mock_agent_class.return_value = mock_agent

        result = await adapter.scrape("https://example.com", "Extract data")

        assert result["success"] is False
        assert "Always fails" in result["error"]
        assert mock_agent.run.call_count == 3  # max_retries = 3 _total attempts

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Agent")
    async def test_scrape_timeout(self, mock_agent_class, basic_config):
        """Test scraping with timeout."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock agent that takes too long
        async def slow_run():
            await asyncio.sleep(5)  # Longer than timeout

        mock_agent = AsyncMock()
        mock_agent.run = slow_run
        mock_agent_class.return_value = mock_agent

        # Use short timeout
        adapter.config.timeout = 100  # 100ms

        result = await adapter.scrape("https://example.com", "Extract data")

        assert result["success"] is False
        assert "timeout" in result["error"]


class TestBrowserUseAdapterUtilities:
    """Test utility methods."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_cleanup(self, basic_config):
        """Test cleanup process."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        # Mock browser
        mock_browser = AsyncMock()
        adapter._browser = mock_browser

        await adapter.cleanup()

        mock_browser.close.assert_called_once()
        assert adapter._browser is None
        assert adapter._initialized is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_cleanup_with_error(self, basic_config):
        """Test cleanup with error during close."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        # Mock browser that raises error on close
        mock_browser = AsyncMock()
        mock_browser.close.side_effect = Exception("Close failed")
        adapter._browser = mock_browser

        # Should not raise exception
        await adapter.cleanup()

        # State should still be reset
        assert adapter._browser is None
        assert adapter._initialized is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.Agent")
    async def test_health_check_healthy(self, mock_agent_class, basic_config):
        """Test health check when healthy."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock successful scrape
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value="<html>Test</html>")
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.url = "https://httpbin.org/html"
        mock_context.current_page = mock_page
        adapter._browser.context = mock_context

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = None
        mock_agent_class.return_value = mock_agent

        result = await adapter.health_check()

        assert result["healthy"] is True
        assert result["available"] is True
        assert result["initialized"] is True
        assert "browser-use" in result["adapter"]

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    async def test_health_check_unavailable(self, basic_config):
        """Test health check when browser-use unavailable."""
        adapter = BrowserUseAdapter(basic_config)

        result = await adapter.health_check()

        assert result["healthy"] is False
        assert result["available"] is False
        assert result["initialized"] is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_capabilities(self, basic_config):
        """Test capabilities method."""
        adapter = BrowserUseAdapter(basic_config)

        caps = adapter.capabilities()

        assert caps["ai_powered"] is True
        assert caps["natural_language_tasks"] is True
        assert caps["self_correcting"] is True
        assert caps["max_retries"] == 3
        assert caps["supported_providers"] == ["openai", "anthropic", "gemini"]

    def test_format_instructions_to_task(self, basic_config):
        """Test instruction formatting."""
        adapter = BrowserUseAdapter(basic_config)

        instructions = [
            {"action": "click", "selector": "button"},
            {"action": "type", "selector": "input", "text": "hello"},
            {"action": "scroll", "direction": "down"},
            {"action": "wait", "timeout": 1000},
        ]

        task = adapter._format_instructions_to_task("Base task", instructions)

        expected = (
            "Base task\n\n"
            "Please perform these actions:\n"
            "1. Click on element: button\n"
            "2. Type 'hello' into element: input\n"
            "3. Scroll down\n"
            "4. Wait for 1000ms"
        )
        assert task == expected

    def test_format_instructions_unsupported_action(self, basic_config):
        """Test formatting with unsupported action."""
        adapter = BrowserUseAdapter(basic_config)

        instructions = [
            {"action": "unsupported", "some": "data"},
        ]

        task = adapter._format_instructions_to_task("Base task", instructions)

        expected = (
            "Base task\n\n"
            "Please perform these actions:\n"
            "1. unsupported (parameters: {'some': 'data'})"
        )
        assert task == expected


class TestBrowserUseAdapterIntegration:
    """Integration tests for BrowserUse adapter."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch("src.services.browser.browser_use_adapter.BrowserConfig")
    @patch("src.services.browser.browser_use_adapter.Browser")
    @patch("src.services.browser.browser_use_adapter.Agent")
    @patch("src.services.browser.browser_use_adapter.ChatOpenAI")
    async def test_full_scraping_flow(
        self,
        mock_chat_openai,
        mock_agent_class,
        mock_browser_class,
        mock_browser_config_class,
        basic_config,
    ):
        """Test complete scraping flow from initialization to cleanup."""
        adapter = BrowserUseAdapter(basic_config)

        # Mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock BrowserConfig
        mock_browser_config = MagicMock()
        mock_browser_config_class.return_value = mock_browser_config

        # Mock browser
        mock_browser = AsyncMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value="<html>Test Content</html>")
        mock_page.title = AsyncMock(return_value="Test Title")
        mock_page.url = "https://example.com"
        mock_context.current_page = mock_page
        mock_browser.context = mock_context
        mock_browser_class.return_value = mock_browser

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = None
        mock_agent_class.return_value = mock_agent

        # Full flow
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Initialize
            await adapter.initialize()
            assert adapter._initialized is True

            # Scrape
            result = await adapter.scrape("https://example.com", "Extract all content")
            assert result["success"] is True
            assert "Test Title" in result["content"]

            # Health check
            health = await adapter.health_check()
            assert health["healthy"] is True

            # Cleanup
            await adapter.cleanup()
            assert adapter._initialized is False
            mock_browser.close.assert_called_once()
