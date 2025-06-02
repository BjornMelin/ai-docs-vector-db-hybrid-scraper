"""Comprehensive tests for BrowserUse AI-powered browser adapter."""

import asyncio
import os
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.browser.browser_use_adapter import BrowserUseAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def basic_config():
    """Basic configuration for BrowserUse adapter."""
    return {
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "headless": True,
        "timeout": 30000,
        "max_retries": 3,
        "max_steps": 20,
        "disable_security": False,
        "generate_gif": False,
    }


@pytest.fixture
def anthropic_config():
    """Configuration for Anthropic provider."""
    return {
        "llm_provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "headless": True,
        "timeout": 30000,
    }


@pytest.fixture
def gemini_config():
    """Configuration for Gemini provider."""
    return {
        "llm_provider": "gemini",
        "model": "gemini-pro",
        "headless": True,
        "timeout": 30000,
    }


class TestBrowserUseAdapterInit:
    """Test BrowserUseAdapter initialization."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_with_basic_config(self, basic_config):
        """Test initialization with basic configuration."""
        adapter = BrowserUseAdapter(basic_config)

        assert adapter._available is True
        assert adapter.llm_provider == "openai"
        assert adapter.model == "gpt-4o-mini"
        assert adapter.headless is True
        assert adapter.timeout == 30000
        assert adapter.max_retries == 3
        assert adapter.max_steps == 20
        assert adapter.disable_security is False
        assert adapter.generate_gif is False
        assert adapter._initialized is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    def test_init_browser_use_unavailable(self, basic_config):
        """Test initialization when browser-use is not available."""
        adapter = BrowserUseAdapter(basic_config)

        assert adapter._available is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = {}
        adapter = BrowserUseAdapter(config)

        assert adapter.llm_provider == "openai"
        assert adapter.model == "gpt-4o-mini"
        assert adapter.headless is True
        assert adapter.timeout == 30000
        assert adapter.max_retries == 3
        assert adapter.max_steps == 20
        assert adapter.disable_security is False
        assert adapter.generate_gif is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_custom_llm_providers(self):
        """Test initialization with different LLM providers."""
        providers = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-opus-20240229"),
            ("gemini", "gemini-1.5-pro"),
        ]

        for provider, model in providers:
            config = {"llm_provider": provider, "model": model}
            adapter = BrowserUseAdapter(config)
            assert adapter.llm_provider == provider
            assert adapter.model == model

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_init_custom_browser_settings(self):
        """Test initialization with custom browser settings."""
        config = {
            "headless": False,
            "timeout": 60000,
            "max_retries": 5,
            "max_steps": 30,
            "disable_security": True,
            "generate_gif": True,
        }

        adapter = BrowserUseAdapter(config)

        assert adapter.headless is False
        assert adapter.timeout == 60000
        assert adapter.max_retries == 5
        assert adapter.max_steps == 30
        assert adapter.disable_security is True
        assert adapter.generate_gif is True


class TestBrowserUseAdapterLLMSetup:
    """Test LLM configuration setup."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_setup_llm_config_openai(self, basic_config):
        """Test OpenAI LLM configuration setup."""
        adapter = BrowserUseAdapter(basic_config)

        with patch(
            "src.services.browser.browser_use_adapter.ChatOpenAI"
        ) as mock_openai:
            mock_llm = MagicMock()
            mock_openai.return_value = mock_llm

            llm_config = adapter._setup_llm_config()

            assert llm_config == mock_llm
            mock_openai.assert_called_once_with(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key="test-openai-key",
            )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_setup_llm_config_anthropic(self, anthropic_config):
        """Test Anthropic LLM configuration setup."""
        adapter = BrowserUseAdapter(anthropic_config)

        with patch(
            "src.services.browser.browser_use_adapter.ChatAnthropic"
        ) as mock_anthropic:
            mock_llm = MagicMock()
            mock_anthropic.return_value = mock_llm

            llm_config = adapter._setup_llm_config()

            assert llm_config == mock_llm
            mock_anthropic.assert_called_once_with(
                model="claude-3-haiku-20240307",
                temperature=0.1,
                api_key="test-anthropic-key",
            )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"})
    def test_setup_llm_config_gemini(self, gemini_config):
        """Test Gemini LLM configuration setup."""
        adapter = BrowserUseAdapter(gemini_config)

        with patch(
            "src.services.browser.browser_use_adapter.ChatGoogleGenerativeAI"
        ) as mock_gemini:
            mock_llm = MagicMock()
            mock_gemini.return_value = mock_llm

            llm_config = adapter._setup_llm_config()

            assert llm_config == mock_llm
            mock_gemini.assert_called_once_with(
                model="gemini-pro",
                temperature=0.1,
                google_api_key="test-google-key",
            )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_setup_llm_config_missing_openai_key(self, basic_config):
        """Test LLM setup fails when OpenAI API key is missing."""
        adapter = BrowserUseAdapter(basic_config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                CrawlServiceError, match="OPENAI_API_KEY environment variable required"
            ):
                adapter._setup_llm_config()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_setup_llm_config_missing_anthropic_key(self, anthropic_config):
        """Test LLM setup fails when Anthropic API key is missing."""
        adapter = BrowserUseAdapter(anthropic_config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                CrawlServiceError,
                match="ANTHROPIC_API_KEY environment variable required",
            ):
                adapter._setup_llm_config()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_setup_llm_config_missing_google_key(self, gemini_config):
        """Test LLM setup fails when Google API key is missing."""
        adapter = BrowserUseAdapter(gemini_config)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                CrawlServiceError, match="GOOGLE_API_KEY environment variable required"
            ):
                adapter._setup_llm_config()

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_setup_llm_config_unsupported_provider(self, basic_config):
        """Test LLM setup fails for unsupported provider."""
        config = {**basic_config, "llm_provider": "unsupported"}
        adapter = BrowserUseAdapter(config)

        with pytest.raises(
            CrawlServiceError, match="Unsupported LLM provider: unsupported"
        ):
            adapter._setup_llm_config()


class TestBrowserUseAdapterInitialization:
    """Test adapter initialization process."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    async def test_initialize_browser_use_unavailable(self, basic_config):
        """Test initialization when browser-use is unavailable."""
        adapter = BrowserUseAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="browser-use not available"):
            await adapter.initialize()

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_initialize_success(self, basic_config):
        """Test successful initialization."""
        adapter = BrowserUseAdapter(basic_config)

        with (
            patch("src.services.browser.browser_use_adapter.ChatOpenAI") as mock_openai,
            patch(
                "src.services.browser.browser_use_adapter.BrowserConfig"
            ) as mock_browser_config,
            patch(
                "src.services.browser.browser_use_adapter.Browser"
            ) as mock_browser_class,
        ):
            mock_llm = MagicMock()
            mock_openai.return_value = mock_llm

            mock_config = MagicMock()
            mock_browser_config.return_value = mock_config

            mock_browser = MagicMock()
            mock_browser_class.return_value = mock_browser

            await adapter.initialize()

            assert adapter._initialized is True
            assert adapter.llm_config == mock_llm
            assert adapter._browser == mock_browser

            # Verify browser config was created correctly
            mock_browser_config.assert_called_once_with(
                headless=True,
                disable_security=False,
            )

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_initialize_already_initialized(self, basic_config):
        """Test that re-initialization is skipped."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True
        original_llm_config = adapter.llm_config

        await adapter.initialize()

        assert adapter.llm_config == original_llm_config

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_initialize_llm_setup_failure(self, basic_config):
        """Test initialization failure during LLM setup."""
        adapter = BrowserUseAdapter(basic_config)

        with patch(
            "src.services.browser.browser_use_adapter.ChatOpenAI",
            side_effect=Exception("LLM setup failed"),
        ):
            with pytest.raises(
                CrawlServiceError, match="Failed to initialize browser-use"
            ):
                await adapter.initialize()

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_initialize_browser_setup_failure(self, basic_config):
        """Test initialization failure during browser setup."""
        adapter = BrowserUseAdapter(basic_config)

        with (
            patch("src.services.browser.browser_use_adapter.ChatOpenAI"),
            patch(
                "src.services.browser.browser_use_adapter.Browser",
                side_effect=Exception("Browser setup failed"),
            ),
        ):
            with pytest.raises(
                CrawlServiceError, match="Failed to initialize browser-use"
            ):
                await adapter.initialize()


class TestBrowserUseAdapterCleanup:
    """Test adapter cleanup process."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_cleanup_success(self, basic_config):
        """Test successful cleanup."""
        adapter = BrowserUseAdapter(basic_config)

        # Setup initialized state
        mock_browser = AsyncMock()
        adapter._browser = mock_browser
        adapter._initialized = True

        await adapter.cleanup()

        mock_browser.close.assert_called_once()
        assert adapter._browser is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_cleanup_browser_close_error(self, basic_config):
        """Test cleanup when browser close fails."""
        adapter = BrowserUseAdapter(basic_config)

        # Setup initialized state with failing browser
        mock_browser = AsyncMock()
        mock_browser.close.side_effect = Exception("Close failed")
        adapter._browser = mock_browser
        adapter._initialized = True

        # Should not raise exception
        await adapter.cleanup()

        assert adapter._browser is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_cleanup_no_browser(self, basic_config):
        """Test cleanup when browser is None."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._browser = None
        adapter._initialized = True

        # Should not raise exception
        await adapter.cleanup()

        assert adapter._browser is None
        assert adapter._initialized is False


class TestBrowserUseAdapterScraping:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    async def test_scrape_browser_use_unavailable(self, basic_config):
        """Test scraping when browser-use is unavailable."""
        adapter = BrowserUseAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="browser-use not available"):
            await adapter.scrape("https://example.com", "Extract content")

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_not_initialized(self, basic_config):
        """Test scraping when adapter is not initialized."""
        adapter = BrowserUseAdapter(basic_config)

        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com", "Extract content")

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_success_basic(self, basic_config):
        """Test successful basic scraping."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        # Setup mocks
        mock_llm = MagicMock()
        mock_browser = MagicMock()
        adapter.llm_config = mock_llm
        adapter._browser = mock_browser

        # Mock agent execution
        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run.return_value = "Extracted content from the page"

            result = await adapter.scrape("https://example.com", "Extract all content")

            assert result["success"] is True
            assert result["url"] == "https://example.com"
            assert result["content"] == "Extracted content from the page"
            assert result["metadata"]["extraction_method"] == "browser_use_ai"
            assert result["metadata"]["llm_provider"] == "openai"
            assert result["metadata"]["model_used"] == "gpt-4o-mini"
            assert "processing_time_ms" in result["metadata"]
            assert result["ai_insights"]["task_completed"] is True

            # Verify agent was created correctly
            mock_agent_class.assert_called_once()
            agent_kwargs = mock_agent_class.call_args[1]
            assert (
                "Navigate to https://example.com and Extract all content"
                in agent_kwargs["task"]
            )
            assert agent_kwargs["llm"] == mock_llm
            assert agent_kwargs["browser"] == mock_browser
            assert agent_kwargs["max_steps"] == 20

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_with_custom_timeout(self, basic_config):
        """Test scraping with custom timeout."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run.return_value = "Content"

            with patch("asyncio.wait_for") as mock_wait_for:
                mock_wait_for.return_value = "Content"

                await adapter.scrape("https://example.com", "Extract", timeout=60000)

                # Should call wait_for with timeout in seconds
                mock_wait_for.assert_called_once()
                assert mock_wait_for.call_args[1]["timeout"] == 60.0

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_with_retries(self, basic_config):
        """Test scraping with retry logic."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            # First two attempts timeout, third succeeds
            mock_agent.run.side_effect = [
                TimeoutError(),
                TimeoutError(),
                "Success on retry",
            ]

            with (
                patch(
                    "asyncio.wait_for",
                    side_effect=[
                        TimeoutError(),
                        TimeoutError(),
                        "Success on retry",
                    ],
                ) as mock_wait_for,
                patch("asyncio.sleep") as mock_sleep,
            ):
                result = await adapter.scrape("https://example.com", "Extract")

                assert result["success"] is True
                assert result["content"] == "Success on retry"
                assert result["metadata"]["retries_used"] == 2

                # Should have called sleep for exponential backoff
                assert mock_sleep.call_count == 2
                mock_sleep.assert_any_call(2)  # 2^1
                mock_sleep.assert_any_call(4)  # 2^2

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_max_retries_exceeded(self, basic_config):
        """Test scraping when max retries are exceeded."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            # All attempts fail
            with patch("asyncio.wait_for", side_effect=TimeoutError()):
                result = await adapter.scrape("https://example.com", "Extract")

                assert result["success"] is False
                assert "Failed after 3 retries" in result["error"]
                assert result["metadata"]["extraction_method"] == "browser_use_ai"

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_exception_handling(self, basic_config):
        """Test scraping exception handling."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            # All attempts raise exception
            with patch("asyncio.wait_for", side_effect=Exception("Agent error")):
                result = await adapter.scrape("https://example.com", "Extract")

                assert result["success"] is False
                assert "Failed after 3 retries" in result["error"]

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_enhanced_task_generation(self, basic_config):
        """Test that enhanced task is generated correctly."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run.return_value = "Content"

            await adapter.scrape("https://example.com", "Click the submit button")

            # Verify enhanced task contains expected elements
            agent_kwargs = mock_agent_class.call_args[1]
            task = agent_kwargs["task"]

            assert "Navigate to https://example.com" in task
            assert "Click the submit button" in task
            assert "Wait for the page to fully load" in task
            assert "Handle any cookie banners" in task
            assert "Extract all relevant content" in task
            assert "comprehensive structured content" in task


class TestBrowserUseAdapterTaskConversion:
    """Test task conversion utilities."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_convert_instructions_to_task_single_instruction(self, basic_config):
        """Test converting single instruction to task."""
        adapter = BrowserUseAdapter(basic_config)

        instructions = ["Click the login button"]
        task = adapter._convert_instructions_to_task(instructions)

        assert "Navigate to the page, then Click the login button." in task
        assert "extract all documentation content" in task

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_convert_instructions_to_task_multiple_instructions(self, basic_config):
        """Test converting multiple instructions to task."""
        adapter = BrowserUseAdapter(basic_config)

        instructions = [
            "Fill in the search box",
            "Click the search button",
            "Wait for results to load",
        ]
        task = adapter._convert_instructions_to_task(instructions)

        expected_parts = [
            "Fill in the search box.",
            "Click the search button.",
            "Wait for results to load.",
        ]

        for part in expected_parts:
            assert part in task

        assert "Navigate to the page, then" in task
        assert "Finally, extract all documentation content" in task

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_convert_instructions_to_task_empty_list(self, basic_config):
        """Test converting empty instruction list."""
        adapter = BrowserUseAdapter(basic_config)

        task = adapter._convert_instructions_to_task([])

        assert (
            task
            == "Navigate to the page and extract all documentation content including code examples."
        )

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_convert_instructions_to_task_whitespace_handling(self, basic_config):
        """Test handling of whitespace in instructions."""
        adapter = BrowserUseAdapter(basic_config)

        instructions = [
            "  Fill form  ",
            "",  # Empty instruction
            "Submit form",
            "   ",  # Whitespace only
        ]
        task = adapter._convert_instructions_to_task(instructions)

        # Should only include non-empty instructions with proper punctuation
        assert "Fill form." in task
        assert "Submit form." in task
        # Empty/whitespace instructions should be filtered out

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_convert_instructions_to_task_punctuation_handling(self, basic_config):
        """Test proper punctuation handling in instructions."""
        adapter = BrowserUseAdapter(basic_config)

        instructions = [
            "Click button",  # No punctuation
            "Fill form!",  # Exclamation
            "Wait.",  # Period
            "Submit?",  # Question mark
        ]
        task = adapter._convert_instructions_to_task(instructions)

        # Should preserve existing punctuation and add periods where missing
        assert "Click button." in task
        assert "Fill form!" in task
        assert "Wait." in task
        assert "Submit?" in task

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_scrape_with_instructions(self, basic_config):
        """Test scraping with instruction list compatibility method."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run.return_value = "Content"

            instructions = ["Click login", "Enter credentials"]
            result = await adapter.scrape_with_instructions(
                "https://example.com", instructions
            )

            assert result["success"] is True

            # Verify the task was converted properly
            agent_kwargs = mock_agent_class.call_args[1]
            task = agent_kwargs["task"]
            assert "Click login." in task
            assert "Enter credentials." in task


class TestBrowserUseAdapterResultBuilding:
    """Test result building functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_build_success_result(self, basic_config):
        """Test building successful result."""
        adapter = BrowserUseAdapter(basic_config)

        start_time = time.time() - 1.5  # 1.5 seconds ago
        agent_result = "Extracted content from the page"
        task = "Extract all content from the documentation page"
        retry_count = 1

        result = await adapter._build_success_result(
            "https://example.com", start_time, agent_result, task, retry_count
        )

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["content"] == "Extracted content from the page"
        assert result["html"] == ""  # Not provided by browser-use
        assert result["title"] == ""  # Not provided by browser-use
        assert result["screenshots"] == []

        metadata = result["metadata"]
        assert metadata["extraction_method"] == "browser_use_ai"
        assert metadata["llm_provider"] == "openai"
        assert metadata["model_used"] == "gpt-4o-mini"
        assert metadata["max_steps"] == 20
        assert metadata["retries_used"] == 1
        assert 1400 <= metadata["processing_time_ms"] <= 1600  # Around 1.5 seconds
        assert "Extract all content" in metadata["task_description"]

        ai_insights = result["ai_insights"]
        assert ai_insights["task_completed"] is True
        assert ai_insights["extraction_confidence"] == "high"

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_build_error_result(self, basic_config):
        """Test building error result."""
        adapter = BrowserUseAdapter(basic_config)

        start_time = time.time() - 2.0  # 2 seconds ago
        error = "Browser automation failed"
        task = "Complex extraction task"

        result = adapter._build_error_result(
            "https://example.com", start_time, error, task
        )

        assert result["success"] is False
        assert result["url"] == "https://example.com"
        assert result["error"] == error
        assert result["content"] == ""

        metadata = result["metadata"]
        assert metadata["extraction_method"] == "browser_use_ai"
        assert metadata["llm_provider"] == "openai"
        assert metadata["model_used"] == "gpt-4o-mini"
        assert 1900 <= metadata["processing_time_ms"] <= 2100  # Around 2 seconds
        assert "Complex extraction" in metadata["task_description"]

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_build_success_result_with_none_agent_result(self, basic_config):
        """Test building result when agent returns None."""
        adapter = BrowserUseAdapter(basic_config)

        result = await adapter._build_success_result(
            "https://example.com", time.time(), None, "task", 0
        )

        assert result["success"] is True
        assert result["content"] == ""  # Should handle None gracefully

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_build_success_result_task_truncation(self, basic_config):
        """Test task description truncation in result metadata."""
        adapter = BrowserUseAdapter(basic_config)

        long_task = "A" * 150  # Longer than 100 characters

        result = await adapter._build_success_result(
            "https://example.com", time.time(), "content", long_task, 0
        )

        task_description = result["metadata"]["task_description"]
        assert len(task_description) == 103  # 100 chars + "..."
        assert task_description.endswith("...")


class TestBrowserUseAdapterCapabilities:
    """Test capabilities and metadata functionality."""

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_get_capabilities(self, basic_config):
        """Test getting adapter capabilities."""
        adapter = BrowserUseAdapter(basic_config)

        capabilities = adapter.get_capabilities()

        assert capabilities["name"] == "browser_use"
        assert "AI-powered browser automation" in capabilities["description"]
        assert "advantages" in capabilities
        assert "limitations" in capabilities
        assert "best_for" in capabilities
        assert "performance" in capabilities
        assert capabilities["javascript_support"] == "excellent"
        assert capabilities["dynamic_content"] == "excellent"
        assert capabilities["authentication"] == "good"
        assert capabilities["cost"] == "api_usage_based"
        assert capabilities["ai_powered"] is True
        assert capabilities["available"] is True
        assert capabilities["llm_provider"] == "openai"
        assert capabilities["model"] == "gpt-4o-mini"

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    def test_get_capabilities_unavailable(self, basic_config):
        """Test capabilities when browser-use unavailable."""
        adapter = BrowserUseAdapter(basic_config)

        capabilities = adapter.get_capabilities()

        assert capabilities["available"] is False

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_capabilities_advantages_and_limitations(self, basic_config):
        """Test that capabilities include expected advantages and limitations."""
        adapter = BrowserUseAdapter(basic_config)
        capabilities = adapter.get_capabilities()

        # Check advantages
        advantages = capabilities["advantages"]
        assert any("Python-native" in advantage for advantage in advantages)
        assert any("Multi-LLM" in advantage for advantage in advantages)
        assert any("Self-correcting" in advantage for advantage in advantages)

        # Check limitations
        limitations = capabilities["limitations"]
        assert any("Slower" in limitation for limitation in limitations)
        assert any("API keys" in limitation for limitation in limitations)

        # Check best use cases
        best_for = capabilities["best_for"]
        assert any("Complex interactions" in use_case for use_case in best_for)
        assert any("Dynamic SPAs" in use_case for use_case in best_for)

    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    def test_capabilities_performance_metrics(self, basic_config):
        """Test performance metrics in capabilities."""
        adapter = BrowserUseAdapter(basic_config)
        capabilities = adapter.get_capabilities()

        performance = capabilities["performance"]
        assert "avg_speed" in performance
        assert "concurrency" in performance
        assert "success_rate" in performance
        assert "1.8s" in performance["avg_speed"]
        assert "96%" in performance["success_rate"]


class TestBrowserUseAdapterHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", False)
    async def test_health_check_unavailable(self, basic_config):
        """Test health check when browser-use unavailable."""
        adapter = BrowserUseAdapter(basic_config)

        health = await adapter.health_check()

        assert health["healthy"] is False
        assert health["status"] == "unavailable"
        assert health["available"] is False
        assert "not installed" in health["message"]

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_health_check_not_initialized(self, basic_config):
        """Test health check when not initialized."""
        adapter = BrowserUseAdapter(basic_config)

        health = await adapter.health_check()

        assert health["healthy"] is False
        assert health["status"] == "not_initialized"
        assert health["available"] is True
        assert "not initialized" in health["message"]

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_health_check_success(self, basic_config):
        """Test successful health check."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        # Mock successful scrape
        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": True, "content": "test"}

            health = await adapter.health_check()

            assert health["healthy"] is True
            assert health["status"] == "operational"
            assert health["available"] is True
            assert "Health check passed" in health["message"]
            assert "response_time_ms" in health
            assert health["test_url"] == "https://httpbin.org/html"
            assert "capabilities" in health

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_health_check_failure(self, basic_config):
        """Test health check when scraping fails."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        # Mock failed scrape
        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": False, "error": "Scrape failed"}

            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "degraded"
            assert "Scrape failed" in health["message"]

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_health_check_timeout(self, basic_config):
        """Test health check timeout."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        # Mock timeout
        async def timeout_scrape(*args, **kwargs):
            await asyncio.sleep(20)  # Longer than timeout

        with patch.object(adapter, "scrape", side_effect=timeout_scrape):
            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "timeout"
            assert health["response_time_ms"] == 15000

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_health_check_exception(self, basic_config):
        """Test health check exception handling."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._initialized = True

        # Mock exception
        with patch.object(adapter, "scrape", side_effect=Exception("Test error")):
            health = await adapter.health_check()

            assert health["healthy"] is False
            assert health["status"] == "error"
            assert "Test error" in health["message"]


class TestBrowserUseAdapterAICapabilityTesting:
    """Test AI capability testing functionality."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_test_ai_capabilities_success(self, basic_config):
        """Test successful AI capability test."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        mock_result = {
            "success": True,
            "content": "Test content extracted successfully",
            "ai_insights": {"task_completed": True, "extraction_confidence": "high"},
            "metadata": {"processing_time_ms": 2500},
        }

        with patch.object(adapter, "scrape", return_value=mock_result):
            result = await adapter.test_ai_capabilities()

            assert result["success"] is True
            assert result["test_url"] == "https://example.com"
            assert "Navigate to the page and perform" in result["task_description"]
            assert result["content_length"] == len(
                "Test content extracted successfully"
            )
            assert result["ai_insights"] == mock_result["ai_insights"]
            assert "execution_time_ms" in result

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_test_ai_capabilities_custom_url(self, basic_config):
        """Test AI capability test with custom URL."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": True, "content": "custom"}

            result = await adapter.test_ai_capabilities("https://custom.example.com")

            assert result["test_url"] == "https://custom.example.com"
            mock_scrape.assert_called_once()
            # Verify custom URL was passed to scrape
            assert mock_scrape.call_args[0][0] == "https://custom.example.com"

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_test_ai_capabilities_not_available(self, basic_config):
        """Test AI capability test when adapter not available."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = False

        result = await adapter.test_ai_capabilities()

        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_test_ai_capabilities_error(self, basic_config):
        """Test AI capability test error handling."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        with patch.object(adapter, "scrape", side_effect=Exception("AI test failed")):
            result = await adapter.test_ai_capabilities()

            assert result["success"] is False
            assert "AI test failed" in result["error"]
            assert "execution_time_ms" in result

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_test_ai_capabilities_comprehensive_task(self, basic_config):
        """Test that AI capability test uses comprehensive task."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True

        with patch.object(adapter, "scrape") as mock_scrape:
            mock_scrape.return_value = {"success": True, "content": "test"}

            await adapter.test_ai_capabilities()

            # Verify comprehensive task was used
            task = mock_scrape.call_args[0][1]
            expected_elements = [
                "Wait for the page to fully load",
                "Extract the main heading and title",
                "Find and note any links present",
                "Extract all visible text content",
                "Identify any interactive elements",
                "Provide a summary of the page structure",
            ]

            for element in expected_elements:
                assert element in task


class TestBrowserUseAdapterIntegration:
    """Integration tests for BrowserUse adapter."""

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_full_workflow(self, basic_config):
        """Test complete workflow from initialization to cleanup."""
        adapter = BrowserUseAdapter(basic_config)

        # Mock dependencies
        mock_llm = MagicMock()
        mock_browser_config = MagicMock()
        mock_browser = AsyncMock()

        with (
            patch(
                "src.services.browser.browser_use_adapter.ChatOpenAI",
                return_value=mock_llm,
            ),
            patch(
                "src.services.browser.browser_use_adapter.BrowserConfig",
                return_value=mock_browser_config,
            ),
            patch(
                "src.services.browser.browser_use_adapter.Browser",
                return_value=mock_browser,
            ),
            patch("src.services.browser.browser_use_adapter.Agent") as mock_agent_class,
        ):
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run.return_value = "Extracted content successfully"

            # Initialize
            await adapter.initialize()
            assert adapter._initialized is True

            # Scrape
            result = await adapter.scrape("https://example.com", "Extract all content")
            assert result["success"] is True
            assert result["content"] == "Extracted content successfully"

            # Scrape with instructions
            instructions_result = await adapter.scrape_with_instructions(
                "https://example.com", ["Click button", "Extract data"]
            )
            assert instructions_result["success"] is True

            # Health check
            health = await adapter.health_check()
            assert health["healthy"] is True

            # Test AI capabilities
            ai_test = await adapter.test_ai_capabilities()
            assert ai_test["success"] is True

            # Get capabilities
            capabilities = adapter.get_capabilities()
            assert capabilities["name"] == "browser_use"

            # Cleanup
            await adapter.cleanup()
            assert adapter._initialized is False

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_error_resilience(self, basic_config):
        """Test adapter resilience to various errors."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Test various error scenarios
        error_scenarios = [
            (TimeoutError(), "timeout"),
            (Exception("Network error"), "network"),
            (Exception("LLM API error"), "api"),
            (MemoryError("Out of memory"), "memory"),
        ]

        for error, scenario_type in error_scenarios:
            with patch(
                "src.services.browser.browser_use_adapter.Agent"
            ) as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent_class.return_value = mock_agent

                # All attempts fail with the specific error
                with patch("asyncio.wait_for", side_effect=error):
                    result = await adapter.scrape(
                        "https://example.com", f"Test {scenario_type}"
                    )

                    assert result["success"] is False
                    assert "Failed after" in result["error"]
                    assert result["metadata"]["extraction_method"] == "browser_use_ai"

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_multiple_llm_providers(self):
        """Test adapter with different LLM providers."""
        providers = [
            ("openai", {"OPENAI_API_KEY": "test-openai-key"}),
            ("anthropic", {"ANTHROPIC_API_KEY": "test-anthropic-key"}),
            ("gemini", {"GOOGLE_API_KEY": "test-google-key"}),
        ]

        for provider, env_vars in providers:
            config = {"llm_provider": provider, "model": "test-model"}
            adapter = BrowserUseAdapter(config)

            with patch.dict(os.environ, env_vars):
                # Test LLM setup
                if provider == "openai":
                    with patch(
                        "src.services.browser.browser_use_adapter.ChatOpenAI"
                    ) as mock_llm:
                        adapter._setup_llm_config()
                        mock_llm.assert_called_once()
                elif provider == "anthropic":
                    with patch(
                        "src.services.browser.browser_use_adapter.ChatAnthropic"
                    ) as mock_llm:
                        adapter._setup_llm_config()
                        mock_llm.assert_called_once()
                elif provider == "gemini":
                    with patch(
                        "src.services.browser.browser_use_adapter.ChatGoogleGenerativeAI"
                    ) as mock_llm:
                        adapter._setup_llm_config()
                        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.browser.browser_use_adapter.BROWSER_USE_AVAILABLE", True)
    async def test_concurrent_scraping_operations(self, basic_config):
        """Test adapter behavior with concurrent scraping operations."""
        adapter = BrowserUseAdapter(basic_config)
        adapter._available = True
        adapter._initialized = True
        adapter.llm_config = MagicMock()
        adapter._browser = MagicMock()

        # Mock different response times for different URLs
        async def variable_agent_response(task, **kwargs):
            if "slow" in task:
                await asyncio.sleep(0.2)
                return "Slow response"
            else:
                await asyncio.sleep(0.05)
                return "Fast response"

        with patch(
            "src.services.browser.browser_use_adapter.Agent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.run = variable_agent_response

            # Run multiple scrapes concurrently
            tasks = [
                adapter.scrape("https://fast.example.com", "Quick task"),
                adapter.scrape("https://slow.example.com", "Slow complex task"),
                adapter.scrape("https://medium.example.com", "Medium task"),
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(r["success"] for r in results)

            # Fast task should complete before slow task
            fast_time = results[0]["metadata"]["processing_time_ms"]
            slow_time = results[1]["metadata"]["processing_time_ms"]
            assert slow_time > fast_time
