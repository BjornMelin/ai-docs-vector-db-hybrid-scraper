"""Tests for configuration validation in unified MCP server."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest


class TestValidateConfiguration:
    """Test configuration validation functionality."""

    @pytest.fixture
    def mock_config_base(self):
        """Create base mock configuration."""
        config = Mock()
        config.get_active_providers = Mock(return_value=[])
        config.openai = Mock(api_key=None)
        config.firecrawl = Mock(api_key=None)
        config.crawling = Mock(providers=[])
        config.qdrant = Mock(url="http://localhost:6333")
        return config

    def test_validate_configuration_valid(self, mock_config_base):
        """Test validate_configuration with valid configuration."""
        # Import needs to be inside the test to avoid module conflicts
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            from unified_mcp_server import validate_configuration

            # Should not raise any exception
            validate_configuration()

    def test_validate_configuration_missing_openai_key(self, mock_config_base):
        """Test validate_configuration with missing OpenAI API key."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        # Configure to require OpenAI key
        mock_config_base.get_active_providers.return_value = ["openai"]
        mock_config_base.openai.api_key = None

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            from unified_mcp_server import validate_configuration

            # Should raise ValueError
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                validate_configuration()

    def test_validate_configuration_missing_qdrant_url(self, mock_config_base):
        """Test validate_configuration with missing Qdrant URL."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        # Configure with no Qdrant URL
        mock_config_base.qdrant.url = None

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            from unified_mcp_server import validate_configuration

            # Should raise ValueError
            with pytest.raises(ValueError, match="Qdrant URL is required"):
                validate_configuration()

    def test_validate_configuration_firecrawl_warning(self, mock_config_base):
        """Test validate_configuration with missing Firecrawl API key (warning only)."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        # Configure to use Firecrawl without API key
        mock_config_base.crawling.providers = ["firecrawl"]
        mock_config_base.firecrawl.api_key = None

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            with patch("unified_mcp_server.logger") as mock_logger:
                from unified_mcp_server import validate_configuration

                # Should not raise exception but log warning
                validate_configuration()

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "Firecrawl API key not set" in warning_msg

    def test_validate_configuration_multiple_errors(self, mock_config_base):
        """Test validate_configuration with multiple errors."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        # Configure multiple errors
        mock_config_base.get_active_providers.return_value = ["openai"]
        mock_config_base.openai.api_key = None
        mock_config_base.qdrant.url = None

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            from unified_mcp_server import validate_configuration

            # Should raise ValueError with both errors
            with pytest.raises(ValueError) as exc_info:
                validate_configuration()

            error_msg = str(exc_info.value)
            assert "OpenAI API key is required" in error_msg
            assert "Qdrant URL is required" in error_msg

    def test_validate_configuration_openai_with_key(self, mock_config_base):
        """Test validate_configuration with OpenAI provider and valid key."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        # Configure with OpenAI provider and key
        mock_config_base.get_active_providers.return_value = ["openai"]
        mock_config_base.openai.api_key = "sk-test-key"

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            from unified_mcp_server import validate_configuration

            # Should not raise exception
            validate_configuration()

    def test_validate_configuration_logs_success(self, mock_config_base):
        """Test validate_configuration logs success message."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

        with patch("unified_mcp_server.get_config") as mock_get_config:
            mock_get_config.return_value = mock_config_base

            with patch("unified_mcp_server.logger") as mock_logger:
                from unified_mcp_server import validate_configuration

                validate_configuration()

                # Verify success was logged
                mock_logger.info.assert_called_with("Configuration validation passed")
