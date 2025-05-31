"""Comprehensive tests for browser-use adapter."""

import asyncio
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.browser.browser_use_adapter import BrowserUseAdapter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "headless": True,
        "timeout": 30000,
        "max_retries": 3,
        "max_steps": 20,
        "disable_security": False,
        "generate_gif": False,
        "save_conversation_path": None,
    }


@pytest.fixture
def adapter(mock_config):
    """Create BrowserUseAdapter instance for testing."""
    return BrowserUseAdapter(mock_config)


class TestBrowserUseAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_initialization(self, adapter, mock_config):
        """Test basic adapter initialization."""
        assert adapter.config == mock_config
        assert adapter._browser is None
        assert adapter._initialized is False
        assert adapter.name == "browser_use"
        assert adapter._llm is None

    def test_adapter_unavailable(self):
        """Test adapter when browser-use is not installed."""
        with patch.dict("sys.modules", {"browser_use": None}):
            adapter = BrowserUseAdapter({})
            assert adapter._available is False

    def test_llm_provider_validation(self):
        """Test LLM provider validation."""
        # Valid providers
        valid_providers = ["openai", "anthropic", "gemini", "azure_openai", "ollama"]
        for provider in valid_providers:
            config = {"llm_provider": provider, "model": "test-model"}
            adapter = BrowserUseAdapter(config)
            assert adapter.config["llm_provider"] == provider

        # Invalid provider
        config = {"llm_provider": "invalid", "model": "test-model"}
        adapter = BrowserUseAdapter(config)
        with pytest.raises(CrawlServiceError, match="Unsupported LLM provider"):
            adapter._setup_llm_config()

    @pytest.mark.asyncio
    async def test_initialize_success(self, adapter):
        """Test successful adapter initialization."""
        mock_browser = AsyncMock()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch(
                "src.services.browser.browser_use_adapter.Browser",
                return_value=mock_browser,
            ),
            patch.object(adapter, "_setup_llm_config", return_value=MagicMock()),
        ):
            await adapter.initialize()

        assert adapter._initialized is True
        assert adapter._browser is mock_browser

    @pytest.mark.asyncio
    async def test_initialize_failure(self, adapter):
        """Test initialization failure handling."""
        with patch(
            "src.services.browser.browser_use_adapter.Browser",
            side_effect=Exception("Init failed"),
        ):
            with pytest.raises(
                CrawlServiceError, match="Failed to initialize browser-use"
            ):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, adapter):
        """Test adapter cleanup."""
        mock_browser = AsyncMock()
        adapter._browser = mock_browser
        adapter._initialized = True

        await adapter.cleanup()

        mock_browser.close.assert_called_once()
        assert adapter._browser is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self, adapter):
        """Test cleanup error handling."""
        mock_browser = AsyncMock()
        mock_browser.close.side_effect = Exception("Close failed")
        adapter._browser = mock_browser
        adapter._initialized = True

        # Should not raise exception
        await adapter.cleanup()

        assert adapter._browser is None
        assert adapter._initialized is False


class TestLLMConfiguration:
    """Test LLM configuration for different providers."""

    def test_setup_llm_openai(self, adapter):
        """Test OpenAI LLM setup."""
        adapter.config["llm_provider"] = "openai"
        adapter.config["model"] = "gpt-4o"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch(
                "src.services.browser.browser_use_adapter.ChatOpenAI"
            ) as mock_openai:
                llm = adapter._setup_llm_config()

        mock_openai.assert_called_once_with(
            model="gpt-4o",
            api_key="sk-test",
            temperature=0.0,
        )
        assert llm is not None

    def test_setup_llm_openai_missing_key(self, adapter):
        """Test OpenAI setup with missing API key."""
        adapter.config["llm_provider"] = "openai"

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                CrawlServiceError, match="OPENAI_API_KEY environment variable required"
            ):
                adapter._setup_llm_config()

    def test_setup_llm_anthropic(self, adapter):
        """Test Anthropic LLM setup."""
        adapter.config["llm_provider"] = "anthropic"
        adapter.config["model"] = "claude-3-sonnet-20241022"

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch(
                "src.services.browser.browser_use_adapter.ChatAnthropic"
            ) as mock_anthropic:
                llm = adapter._setup_llm_config()

        mock_anthropic.assert_called_once_with(
            model="claude-3-sonnet-20241022",
            api_key="test-key",
            temperature=0.0,
        )

    def test_setup_llm_gemini(self, adapter):
        """Test Gemini LLM setup."""
        adapter.config["llm_provider"] = "gemini"
        adapter.config["model"] = "gemini-2.0-flash-exp"

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch(
                "src.services.browser.browser_use_adapter.ChatGoogleGenerativeAI"
            ) as mock_gemini:
                llm = adapter._setup_llm_config()

        mock_gemini.assert_called_once_with(
            model="gemini-2.0-flash-exp",
            google_api_key="test-key",
            temperature=0.0,
        )

    def test_setup_llm_azure_openai(self, adapter):
        """Test Azure OpenAI LLM setup."""
        adapter.config["llm_provider"] = "azure_openai"
        adapter.config["model"] = "gpt-4"

        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4-deployment",
            "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
        }

        with patch.dict(os.environ, env_vars):
            with patch(
                "src.services.browser.browser_use_adapter.AzureChatOpenAI"
            ) as mock_azure:
                llm = adapter._setup_llm_config()

        mock_azure.assert_called_once_with(
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4-deployment",
            api_version="2024-02-15-preview",
            temperature=0.0,
        )

    def test_setup_llm_ollama(self, adapter):
        """Test Ollama LLM setup."""
        adapter.config["llm_provider"] = "ollama"
        adapter.config["model"] = "llama2"

        with patch(
            "src.services.browser.browser_use_adapter.ChatOllama"
        ) as mock_ollama:
            llm = adapter._setup_llm_config()

        mock_ollama.assert_called_once_with(
            model="llama2",
            temperature=0.0,
        )


class TestScrapeOperation:
    """Test main scrape functionality."""

    @pytest.mark.asyncio
    async def test_scrape_not_available(self, adapter):
        """Test scraping when adapter not available."""
        adapter._available = False

        with pytest.raises(CrawlServiceError, match="browser-use not available"):
            await adapter.scrape("https://example.com", "Extract content")

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, adapter):
        """Test scraping when adapter not initialized."""
        with pytest.raises(CrawlServiceError, match="Adapter not initialized"):
            await adapter.scrape("https://example.com", "Extract content")

    @pytest.mark.asyncio
    async def test_scrape_success_basic(self, adapter):
        """Test successful basic scraping."""
        url = "https://example.com"
        task = "Extract all documentation content"

        mock_agent = AsyncMock()
        mock_agent.run.return_value = (
            "# Documentation\n\nThis is the extracted content from the page."
        )

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task)

        assert result["success"] is True
        assert "Documentation" in result["content"]
        assert result["url"] == url
        assert result["metadata"]["extraction_method"] == "browser_use_ai"
        assert result["metadata"]["llm_provider"] == "openai"
        assert result["metadata"]["model_used"] == "gpt-4o-mini"
        assert result["metadata"]["task"] == task
        assert "execution_time_ms" in result["metadata"]

        mock_agent.run.assert_called_once_with(task)

    @pytest.mark.asyncio
    async def test_scrape_with_instructions(self, adapter):
        """Test scraping with instruction list."""
        url = "https://example.com"
        instructions = [
            "Click on the login button",
            "Fill in username field with 'test@example.com'",
            "Fill in password field",
            "Click submit",
            "Extract dashboard content",
        ]

        mock_agent = AsyncMock()
        mock_agent.run.return_value = "Dashboard content successfully extracted"

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, instructions)

        # Check that instructions were converted to task
        expected_task = adapter._convert_instructions_to_task(instructions)
        mock_agent.run.assert_called_once_with(expected_task)

        assert result["success"] is True
        assert "Dashboard content" in result["content"]

    @pytest.mark.asyncio
    async def test_scrape_with_timeout(self, adapter):
        """Test scraping with custom timeout."""
        url = "https://example.com"
        task = "Extract content"
        timeout = 60000

        mock_agent = AsyncMock()

        async def slow_run(task):
            await asyncio.sleep(0.1)  # Simulate work
            return "Content extracted"

        mock_agent.run.side_effect = slow_run

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task, timeout=timeout)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scrape_timeout_exceeded(self, adapter):
        """Test handling of timeout errors."""
        url = "https://slow-site.com"
        task = "Extract content"

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = TimeoutError("Task timed out")

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task, timeout=1000)

        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_scrape_with_retries(self, adapter):
        """Test retry mechanism on failure."""
        url = "https://example.com"
        task = "Extract content"

        mock_agent = AsyncMock()
        # Fail twice, succeed on third attempt
        mock_agent.run.side_effect = [
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            "Success on third attempt!",
        ]

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True
        adapter.config["max_retries"] = 3

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task)

        assert result["success"] is True
        assert "Success on third attempt" in result["content"]
        assert mock_agent.run.call_count == 3

    @pytest.mark.asyncio
    async def test_scrape_max_retries_exceeded(self, adapter):
        """Test failure after max retries exceeded."""
        url = "https://example.com"
        task = "Extract content"

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("Persistent failure")

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True
        adapter.config["max_retries"] = 2

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task)

        assert result["success"] is False
        assert "Persistent failure" in result["error"]
        assert mock_agent.run.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_scrape_ai_insights(self, adapter):
        """Test extraction of AI insights from response."""
        url = "https://example.com"
        task = "Navigate to the pricing page and extract pricing tiers"

        mock_agent = AsyncMock()
        mock_agent.run.return_value = """
        I successfully navigated to the pricing page. Here are the pricing tiers:
        
        - Basic: $10/month
        - Pro: $50/month
        - Enterprise: Custom pricing
        
        The page also contains comparison features and a FAQ section.
        """

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task)

        assert result["success"] is True
        assert "$10/month" in result["content"]
        assert "ai_insights" in result
        assert result["ai_insights"]["task_completed"] is True
        assert "successfully navigated" in result["ai_insights"]["summary"]


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
        adapter._llm = MagicMock()
        adapter._initialized = True

        mock_result = {
            "success": True,
            "content": "Test page loaded",
            "ai_insights": {"task_completed": True},
        }

        with patch.object(adapter, "scrape", return_value=mock_result):
            health = await adapter.health_check()

        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert health["available"] is True
        assert "response_time_ms" in health
        assert health["response_time_ms"] < 15000

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, adapter):
        """Test health check timeout handling."""
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch.object(adapter, "scrape", side_effect=TimeoutError()):
            health = await adapter.health_check()

        assert health["status"] == "timeout"
        assert health["healthy"] is False
        assert health["response_time_ms"] == 15000

    @pytest.mark.asyncio
    async def test_health_check_error(self, adapter):
        """Test health check error handling."""
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        with patch.object(
            adapter, "scrape", side_effect=Exception("Health check failed")
        ):
            health = await adapter.health_check()

        assert health["status"] == "error"
        assert health["healthy"] is False
        assert "Health check failed" in health["error"]


class TestAICapabilities:
    """Test AI capability testing functionality."""

    @pytest.mark.asyncio
    async def test_ai_capabilities_not_available(self, adapter):
        """Test AI capabilities when adapter not available."""
        adapter._available = False

        result = await adapter.test_ai_capabilities()

        assert result["success"] is False
        assert "not available or initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_ai_capabilities_success(self, adapter):
        """Test successful AI capabilities test."""
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        mock_result = {
            "success": True,
            "content": "Successfully tested AI capabilities",
            "ai_insights": {
                "task_completed": True,
                "capabilities_tested": [
                    "navigation",
                    "interaction",
                    "extraction",
                    "decision_making",
                ],
            },
        }

        with patch.object(adapter, "scrape", return_value=mock_result):
            result = await adapter.test_ai_capabilities("https://test.com")

        assert result["success"] is True
        assert result["test_url"] == "https://test.com"
        assert "Navigate to the page and perform" in result["task_description"]
        assert result["content_length"] > 0
        assert len(result["ai_insights"]["capabilities_tested"]) == 4


class TestHelperMethods:
    """Test helper methods."""

    def test_convert_instructions_empty(self, adapter):
        """Test instruction conversion with empty list."""
        task = adapter._convert_instructions_to_task([])
        assert "Navigate to the page and extract all documentation content" in task

    def test_convert_instructions_single(self, adapter):
        """Test instruction conversion with single instruction."""
        instructions = ["Click on the download button"]
        task = adapter._convert_instructions_to_task(instructions)

        assert "Navigate to the page, then" in task
        assert "Click on the download button" in task
        assert "extract all documentation content" in task

    def test_convert_instructions_multiple(self, adapter):
        """Test instruction conversion with multiple instructions."""
        instructions = [
            "Wait for page to load",
            "Click on 'Sign In'",
            "Type 'user@example.com' in email field",
            "Type password",
            "Press Enter",
            "Extract account dashboard",
        ]

        task = adapter._convert_instructions_to_task(instructions)

        assert "Navigate to the page, then" in task
        for instruction in instructions:
            assert instruction in task
        assert "extract all documentation content" in task

    def test_get_capabilities(self, adapter):
        """Test capability reporting."""
        capabilities = adapter.get_capabilities()

        assert capabilities["name"] == "browser_use"
        assert capabilities["ai_powered"] is True
        assert capabilities["llm_provider"] == "openai"
        assert capabilities["model"] == "gpt-4o-mini"

        # Check advantages
        assert any("Python-native" in adv for adv in capabilities["advantages"])
        assert any("Multi-LLM provider" in adv for adv in capabilities["advantages"])
        assert any("Self-correcting AI" in adv for adv in capabilities["advantages"])

        # Check limitations
        assert "Slower than direct automation" in capabilities["limitations"]

        # Check performance metrics
        assert capabilities["performance"]["avg_speed"] == "1.8s per page"
        assert "96%" in capabilities["performance"]["success_rate"]


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handle_browser_crash(self, adapter):
        """Test handling of browser crash."""
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("Browser process crashed")

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape("https://example.com", "Extract content")

        assert result["success"] is False
        assert "Browser process crashed" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_llm_error(self, adapter):
        """Test handling of LLM errors."""
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("LLM API rate limit exceeded")

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape("https://example.com", "Extract content")

        assert result["success"] is False
        assert "LLM API rate limit exceeded" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_invalid_task(self, adapter):
        """Test handling of invalid task format."""
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        # Test with None task
        with pytest.raises(CrawlServiceError, match="Task is required"):
            await adapter.scrape("https://example.com", None)

        # Test with empty task
        with pytest.raises(CrawlServiceError, match="Task is required"):
            await adapter.scrape("https://example.com", "")


class TestAdvancedFeatures:
    """Test advanced browser-use features."""

    @pytest.mark.asyncio
    async def test_conversation_saving(self, adapter):
        """Test conversation saving feature."""
        adapter.config["save_conversation_path"] = "/tmp/conversations"
        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True

        mock_agent = AsyncMock()
        mock_agent.run.return_value = "Content extracted"
        mock_agent.conversation_history = [
            {"role": "user", "content": "Navigate to example.com"},
            {"role": "assistant", "content": "Navigating to the page"},
            {"role": "user", "content": "Extract main content"},
            {"role": "assistant", "content": "Content extracted"},
        ]

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape("https://example.com", "Extract content")

        assert result["success"] is True
        # In real implementation, would verify conversation was saved

    @pytest.mark.asyncio
    async def test_custom_browser_config(self, adapter):
        """Test custom browser configuration."""
        adapter.config.update(
            {
                "disable_security": True,
                "generate_gif": True,
                "headless": False,
            }
        )

        mock_browser_class = MagicMock()
        mock_browser_instance = AsyncMock()
        mock_browser_class.return_value = mock_browser_instance

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch(
                "src.services.browser.browser_use_adapter.Browser", mock_browser_class
            ),
            patch.object(adapter, "_setup_llm_config", return_value=MagicMock()),
        ):
            await adapter.initialize()

        mock_browser_class.assert_called_once()
        config_used = mock_browser_class.call_args.kwargs.get("config", {})

        # Verify browser config parameters
        assert config_used.get("disable_security") is True
        assert config_used.get("generate_gif") is True
        assert config_used.get("headless") is False

    @pytest.mark.asyncio
    async def test_multi_step_task_execution(self, adapter):
        """Test complex multi-step task execution."""
        url = "https://complex-app.com"
        task = """
        1. Navigate to the login page
        2. Enter credentials (use test@example.com)
        3. Navigate to settings
        4. Change theme to dark mode
        5. Save settings
        6. Extract confirmation message
        """

        mock_agent = AsyncMock()
        mock_agent.run.return_value = """
        I completed all the requested steps:
        1. ✓ Navigated to login page
        2. ✓ Entered credentials for test@example.com
        3. ✓ Found and clicked settings link
        4. ✓ Changed theme preference to dark mode
        5. ✓ Clicked save button
        6. ✓ Extracted confirmation: "Settings saved successfully!"
        
        The dark mode is now active across the application.
        """

        adapter._browser = AsyncMock()
        adapter._llm = MagicMock()
        adapter._initialized = True
        adapter.config["max_steps"] = 30  # Allow more steps for complex task

        with patch(
            "src.services.browser.browser_use_adapter.Agent", return_value=mock_agent
        ):
            result = await adapter.scrape(url, task)

        assert result["success"] is True
        assert "Settings saved successfully" in result["content"]
        assert all(
            step in result["content"] for step in ["login", "settings", "dark mode"]
        )
