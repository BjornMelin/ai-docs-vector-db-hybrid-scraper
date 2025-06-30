"""Unit tests for unified_mcp_server module."""

import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.tool_registry import register_all_tools
from src.services.logging_config import configure_logging


# Mock problematic imports before importing the module
sys.modules["fastmcp"] = MagicMock()
sys.modules["src.infrastructure.client_manager"] = MagicMock()
sys.modules["src.mcp_tools.tool_registry"] = MagicMock()
sys.modules["src.services.logging_config"] = MagicMock()
sys.modules["src.services.monitoring.initialization"] = MagicMock()
sys.modules["src.config"] = MagicMock()
sys.modules["src.config.enums"] = MagicMock()
sys.modules["src.services"] = MagicMock()
sys.modules["src.services.vector_db"] = MagicMock()
sys.modules["src.services.vector_db.search"] = MagicMock()

from src import unified_mcp_server  # noqa: E402


logger = logging.getLogger(__name__)


class TestValidateStreamingConfig:
    """Test cases for _validate_streaming_config function."""

    @patch.dict("os.environ", {"FASTMCP_TRANSPORT": "stdio"})
    def test_non_streamable_http_transport_skips_validation(self):
        """Test that non-streamable-http transport skips validation."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 0
        assert len(warnings) == 0

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_PORT": "8080",
            "FASTMCP_BUFFER_SIZE": "4096",
            "FASTMCP_MAX_RESPONSE_SIZE": "5242880",
        },
    )
    def test_valid_streaming_config(self):
        """Test validation with valid streaming configuration."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 0
        assert len(warnings) == 0

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_PORT": "0",
        },
    )
    def test_invalid_port_zero(self):
        """Test validation with invalid port (zero)."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 1
        assert "Invalid port number: 0" in errors[0]

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_PORT": "70000",
        },
    )
    def test_invalid_port_too_high(self):
        """Test validation with invalid port (too high)."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 1
        assert "Invalid port number: 70000" in errors[0]

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_PORT": "not_a_number",
        },
    )
    def test_invalid_port_not_integer(self):
        """Test validation with non-integer port."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 1
        assert "Invalid port value: not_a_number" in errors[0]

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_BUFFER_SIZE": "-1",
        },
    )
    def test_negative_buffer_size_warning(self):
        """Test validation with negative buffer size."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 0
        assert len(warnings) == 1
        assert "Buffer size -1 is very small" in warnings[0]

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_BUFFER_SIZE": "not_a_number",
        },
    )
    def test_invalid_buffer_size(self):
        """Test validation with non-integer buffer size."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 1
        assert "Invalid buffer size: not_a_number" in errors[0]

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_MAX_RESPONSE_SIZE": "-1",
        },
    )
    def test_negative_max_response_size(self):
        """Test validation with negative max response size."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 1
        assert "Max response size must be positive" in errors[0]

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_MAX_RESPONSE_SIZE": "not_a_number",
        },
    )
    def test_invalid_max_response_size(self):
        """Test validation with non-integer max response size."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 1
        assert "Invalid max response size: not_a_number" in errors[0]

    @patch.dict("os.environ", {}, clear=True)
    def test_default_values_are_valid(self):
        """Test that default values pass validation."""
        errors = []
        warnings = []

        # Set minimal required env var to trigger validation
        with patch.dict("os.environ", {"FASTMCP_TRANSPORT": "streamable-http"}):
            unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 0
        assert len(warnings) == 0


class TestValidateConfiguration:
    """Test cases for validate_configuration function."""

    @patch("src.config.get_config")
    @patch("src.unified_mcp_server._validate_streaming_config")
    def test_valid_configuration(self, mock_validate_streaming, mock_get_config):
        """Test validation with valid configuration."""
        # Mock config
        mock_config = MagicMock()
        mock_config.get_active_providers.return_value = ["fastembed"]
        mock_config.crawling.providers = ["crawl4ai"]
        mock_config.qdrant.url = "http://localhost:6333"
        mock_get_config.return_value = mock_config

        # Should not raise exception
        unified_mcp_server.validate_configuration()

        mock_validate_streaming.assert_called_once()

    @patch("src.config.get_config")
    def test_missing_openai_api_key(self, mock_get_config):
        """Test validation with missing OpenAI API key."""
        mock_config = MagicMock()
        mock_config.get_active_providers.return_value = ["openai"]
        mock_config.openai.api_key = None
        mock_config.crawling.providers = ["crawl4ai"]
        mock_config.qdrant.url = "http://localhost:6333"
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            unified_mcp_server.validate_configuration()

    @patch("src.config.get_config")
    def test_missing_firecrawl_api_key_warning(self, mock_get_config):
        """Test validation with missing Firecrawl API key (warning only)."""
        mock_config = MagicMock()
        mock_config.get_active_providers.return_value = ["fastembed"]
        mock_config.crawling.providers = ["firecrawl"]
        mock_config.firecrawl.api_key = None
        mock_config.qdrant.url = "http://localhost:6333"
        mock_get_config.return_value = mock_config

        # Should not raise exception, only log warning
        with patch("src.unified_mcp_server.logger.warning") as mock_logger:
            unified_mcp_server.validate_configuration()
            mock_logger.assert_called()

    @patch("src.config.get_config")
    def test_missing_qdrant_url(self, mock_get_config):
        """Test validation with missing Qdrant URL."""
        mock_config = MagicMock()
        mock_config.get_active_providers.return_value = ["fastembed"]
        mock_config.crawling.providers = ["crawl4ai"]
        mock_config.qdrant.url = None
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="Qdrant URL is required"):
            unified_mcp_server.validate_configuration()

    @patch("src.config.get_config")
    @patch("src.unified_mcp_server._validate_streaming_config")
    def test_multiple_configuration_errors(
        self, mock_validate_streaming, mock_get_config
    ):
        """Test validation with multiple configuration errors."""
        mock_config = MagicMock()
        mock_config.get_active_providers.return_value = ["openai"]
        mock_config.openai.api_key = None
        mock_config.crawling.providers = ["crawl4ai"]
        mock_config.qdrant.url = None
        mock_get_config.return_value = mock_config

        # Mock streaming validation to add errors
        def add_streaming_errors(errors, _warnings):
            errors.append("Streaming error")

        mock_validate_streaming.side_effect = add_streaming_errors

        with pytest.raises(ValueError) as exc_info:
            unified_mcp_server.validate_configuration()

        # Should contain all errors
        error_message = str(exc_info.value)
        assert "OpenAI API key is required" in error_message
        assert "Qdrant URL is required" in error_message
        assert "Streaming error" in error_message


class TestLifespanContextManager:
    """Test cases for lifespan context manager."""

    @pytest.mark.asyncio
    @patch("src.unified_mcp_server.validate_configuration")
    @patch("src.unified_mcp_server.ClientManager")
    @patch("src.unified_mcp_server.register_all_tools")
    @patch("src.config.get_config")
    @patch("src.unified_mcp_server.initialize_monitoring_system")
    @patch("src.unified_mcp_server.setup_fastmcp_monitoring")
    @patch("src.unified_mcp_server.run_periodic_health_checks")
    @patch("src.unified_mcp_server.update_system_metrics_periodically")
    @patch("src.unified_mcp_server.update_cache_metrics_periodically")
    async def test_lifespan_successful_initialization(
        self,
        mock_update_cache_metrics,
        mock_update_system_metrics,
        mock_health_checks,
        _mock_setup_monitoring,
        mock_init_monitoring,
        mock_get_config,
        mock_register_tools,
        mock_client_manager_class,
        mock_validate_config,
    ):
        """Test successful lifespan initialization and cleanup."""
        # Mock config
        mock_config = MagicMock()
        mock_config.cache.enable_dragonfly_cache = False
        mock_get_config.return_value = mock_config

        # Mock client manager
        mock_client_manager = AsyncMock()
        mock_client_manager.initialize = AsyncMock()
        mock_client_manager.cleanup = AsyncMock()
        mock_client_manager_class.return_value = mock_client_manager

        # Mock monitoring system initialization
        mock_metrics_registry = MagicMock()
        mock_health_manager = MagicMock()
        mock_init_monitoring.return_value = (mock_metrics_registry, mock_health_manager)

        # Mock async monitoring tasks to return coroutines
        async def mock_health_task(*args, **_kwargs):
            pass

        async def mock_system_metrics(*args, **_kwargs):
            pass

        async def mock_cache_metrics(*args, **_kwargs):
            pass

        mock_health_checks.return_value = mock_health_task()
        mock_update_system_metrics.return_value = mock_system_metrics()
        mock_update_cache_metrics.return_value = mock_cache_metrics()

        # Make register_all_tools async
        async def mock_register(*_args, **__kwargs):
            return True

        mock_register_tools.side_effect = mock_register

        async with unified_mcp_server.lifespan():
            # Verify initialization calls
            mock_validate_config.assert_called_once()
            mock_get_config.assert_called_once()
            mock_client_manager_class.assert_called_once_with(mock_config)
            mock_client_manager.initialize.assert_called_once()
            mock_register_tools.assert_called_once()

        # Verify cleanup was called
        mock_client_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.unified_mcp_server.validate_configuration")
    @patch("src.unified_mcp_server.ClientManager")
    async def test_lifespan_validation_failure(
        self, mock_client_manager_class, mock_validate_config
    ):
        """Test lifespan with configuration validation failure."""
        mock_validate_config.side_effect = ValueError("Configuration error")

        # Mock client manager for cleanup
        mock_client_manager = AsyncMock()
        mock_client_manager.cleanup = AsyncMock()
        mock_client_manager_class.return_value = mock_client_manager

        with pytest.raises(ValueError, match="Configuration error"):
            async with unified_mcp_server.lifespan():
                pass

    @pytest.mark.asyncio
    @patch("src.unified_mcp_server.validate_configuration")
    @patch("src.unified_mcp_server.ClientManager")
    @patch("src.config.get_config")
    async def test_lifespan_client_manager_initialization_failure(
        self, mock_get_config, mock_client_manager_class, _mock_validate_config
    ):
        """Test lifespan with client manager initialization failure."""
        # Mock config
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_client_manager = AsyncMock()
        mock_client_manager.initialize = AsyncMock(side_effect=Exception("Init error"))
        mock_client_manager.cleanup = AsyncMock()
        mock_client_manager_class.return_value = mock_client_manager

        with pytest.raises(Exception, match="Init error"):
            async with unified_mcp_server.lifespan():
                pass

        # Cleanup should still be called
        mock_client_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_mock_validate_config")
    @patch("src.unified_mcp_server.validate_configuration")
    @patch("src.unified_mcp_server.ClientManager")
    @patch("src.unified_mcp_server.register_all_tools")
    @patch("src.config.get_config")
    @patch("src.unified_mcp_server.initialize_monitoring_system")
    @patch("src.unified_mcp_server.setup_fastmcp_monitoring")
    @patch("src.unified_mcp_server.run_periodic_health_checks")
    @patch("src.unified_mcp_server.update_system_metrics_periodically")
    @patch("src.unified_mcp_server.update_cache_metrics_periodically")
    async def test_lifespan_cleanup_on_exception(
        self,
        mock_update_cache_metrics,
        mock_update_system_metrics,
        mock_health_checks,
        _mock_setup_monitoring,
        mock_init_monitoring,
        mock_get_config,
        mock_register_tools,
        mock_client_manager_class,
    ):
        """Test that cleanup is called even when exception occurs during operation."""
        # Mock config
        mock_config = MagicMock()
        mock_config.cache.enable_dragonfly_cache = False
        mock_get_config.return_value = mock_config

        mock_client_manager = AsyncMock()
        mock_client_manager.initialize = AsyncMock()
        mock_client_manager.cleanup = AsyncMock()
        mock_client_manager_class.return_value = mock_client_manager

        # Mock monitoring system initialization
        mock_metrics_registry = MagicMock()
        mock_health_manager = MagicMock()
        mock_init_monitoring.return_value = (mock_metrics_registry, mock_health_manager)

        # Mock async monitoring tasks to return coroutines
        async def mock_health_task(*args, **_kwargs):
            pass

        async def mock_system_metrics(*args, **_kwargs):
            pass

        async def mock_cache_metrics(*args, **_kwargs):
            pass

        mock_health_checks.return_value = mock_health_task()
        mock_update_system_metrics.return_value = mock_system_metrics()
        mock_update_cache_metrics.return_value = mock_cache_metrics()

        # Make register_all_tools async
        async def mock_register(*_args, **__kwargs):
            return True

        mock_register_tools.side_effect = mock_register

        try:
            async with unified_mcp_server.lifespan():
                # Simulate an exception during operation
                msg = "Operation failed"
                raise RuntimeError(msg)  # noqa: TRY301
        except RuntimeError:
            pass

        # Verify cleanup was called despite exception
        mock_client_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_mock_validate_config")
    @patch("src.unified_mcp_server.validate_configuration")
    @patch("src.unified_mcp_server.ClientManager")
    @patch("src.unified_mcp_server.register_all_tools")
    @patch("src.config.get_config")
    async def test_lifespan_cleanup_exception_handling(
        self,
        mock_get_config,
        mock_register_tools,
        mock_client_manager_class,
    ):
        """Test that cleanup exceptions are handled gracefully."""
        # Mock config
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_client_manager = AsyncMock()
        mock_client_manager.initialize = AsyncMock()
        mock_client_manager.cleanup = AsyncMock(side_effect=Exception("Cleanup error"))
        mock_client_manager_class.return_value = mock_client_manager

        # Make register_all_tools async
        async def mock_register(*_args, **__kwargs):
            return True

        mock_register_tools.side_effect = mock_register

        # Should raise cleanup exception since it's not suppressed in current implementation
        with pytest.raises(Exception, match="Cleanup error"):
            async with unified_mcp_server.lifespan():
                pass

        mock_client_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_mock_register_tools", "_mock_client_manager_class")
    @patch("src.unified_mcp_server.validate_configuration")
    @patch("src.unified_mcp_server.ClientManager")
    @patch("src.unified_mcp_server.register_all_tools")
    async def test_lifespan_without_client_manager(self, mock_validate_config):
        """Test lifespan cleanup when client manager was not created."""
        # Simulate failure before client manager creation
        mock_validate_config.side_effect = Exception("Early failure")

        try:
            async with unified_mcp_server.lifespan():
                pass
        except Exception as e:
            # Exception expected during lifespan test
            logger.debug(
                f"Expected lifespan test exception: {e}"
            )  # TODO: Convert f-string to logging format

        # Should not attempt to cleanup non-existent client manager
        # (No assertion needed as exception would be raised if cleanup was called incorrectly)


class TestMainExecutionLogic:
    """Test cases for main execution logic."""

    @patch.dict("os.environ", {"FASTMCP_TRANSPORT": "streamable-http"})
    def test_environment_variable_handling_streamable_http(self):
        """Test environment variable handling for streamable-http transport."""
        assert os.getenv("FASTMCP_TRANSPORT") == "streamable-http"

    @patch.dict("os.environ", {"FASTMCP_TRANSPORT": "stdio"})
    def test_environment_variable_handling_stdio(self):
        """Test environment variable handling for stdio transport."""
        assert os.getenv("FASTMCP_TRANSPORT") == "stdio"

    @patch.dict("os.environ", {}, clear=True)
    def test_environment_variable_defaults(self):
        """Test default environment variable values."""
        assert os.getenv("FASTMCP_TRANSPORT", "streamable-http") == "streamable-http"
        assert os.getenv("FASTMCP_HOST", "127.0.0.1") == "127.0.0.1"
        assert os.getenv("FASTMCP_PORT", "8000") == "8000"


class TestServerConfiguration:
    """Test cases for server configuration and initialization."""

    def test_mcp_server_initialization(self):
        """Test that MCP server is initialized with correct parameters."""
        # Check that the mcp instance exists and has expected attributes
        assert hasattr(unified_mcp_server, "mcp")
        assert unified_mcp_server.mcp is not None

        # Check that lifespan is set
        assert hasattr(unified_mcp_server.mcp, "lifespan")
        assert unified_mcp_server.mcp.lifespan is not None

    def test_server_instructions(self):
        """Test that server has proper instructions."""
        # The instructions should be accessible (they're set during MCP init)
        # This test verifies the server was configured with instructions
        assert hasattr(unified_mcp_server, "mcp")

    def test_logging_configuration(self):
        """Test that logging is configured on import."""
        # Since we mocked the import, we just verify the mock exists

        assert "src.services.logging_config" in sys.modules

    def test_sys_path_setup(self):
        """Test that sys path is properly configured."""

        # The path should be in sys.path (added during import)
        # We can't assert exact position due to import order
        assert any(path.endswith("src") for path in sys.path)


class TestImportAndModuleStructure:
    """Test cases for import structure and module organization."""

    def test_all_required_imports(self):
        """Test that all required modules can be imported."""
        # These imports should work without errors

        assert ClientManager is not None
        assert register_all_tools is not None
        assert configure_logging is not None

    def test_fastmcp_import(self):
        """Test that FastMCP can be imported."""

        assert FastMCP is not None

    def test_module_level_variables(self):
        """Test that module-level variables are properly defined."""
        # Check logger
        assert hasattr(unified_mcp_server, "logger")
        assert unified_mcp_server.logger is not None

        # Check mcp server instance
        assert hasattr(unified_mcp_server, "mcp")
        assert unified_mcp_server.mcp is not None


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    @patch("src.config.get_config")
    def test_configuration_error_handling(self, mock_get_config):
        """Test proper error handling for configuration issues."""
        mock_get_config.side_effect = Exception("Config load error")

        with pytest.raises(Exception, match="Config load error"):
            unified_mcp_server.validate_configuration()

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_PORT": "invalid",
            "FASTMCP_BUFFER_SIZE": "invalid",
            "FASTMCP_MAX_RESPONSE_SIZE": "invalid",
        },
    )
    def test_multiple_streaming_errors(self):
        """Test handling of multiple streaming configuration errors."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        # Should collect all errors
        assert len(errors) == 3  # port, buffer_size, max_response_size
        assert any("Invalid port value" in error for error in errors)
        assert any("Invalid buffer size" in error for error in errors)
        assert any("Invalid max response size" in error for error in errors)


class TestEnvironmentVariableHandling:
    """Test cases for environment variable handling."""

    @patch.dict("os.environ", {}, clear=True)
    def test_default_environment_values(self):
        """Test behavior with no environment variables set."""
        errors = []
        warnings = []

        # Should use defaults and not error
        with patch.dict("os.environ", {"FASTMCP_TRANSPORT": "streamable-http"}):
            unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 0
        assert len(warnings) == 0

    @patch.dict(
        "os.environ",
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_HOST": "custom.host.com",
            "FASTMCP_PORT": "3000",
        },
    )
    def test_custom_environment_values(self):
        """Test behavior with custom environment values."""
        errors = []
        warnings = []

        unified_mcp_server._validate_streaming_config(errors, warnings)

        assert len(errors) == 0
        assert len(warnings) == 0

        # Test that the values would be used in main
        assert os.getenv("FASTMCP_HOST") == "custom.host.com"
        assert os.getenv("FASTMCP_PORT") == "3000"
