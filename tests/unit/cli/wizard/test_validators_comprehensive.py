"""Comprehensive tests for WizardValidator with modern validation patterns.

This module tests the WizardValidator class with focus on real-time validation,
user-friendly error messages, and Rich console integration.
"""

from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from src.cli.wizard.validators import WizardValidator


class TestWizardValidatorCore:
    """Core functionality tests for WizardValidator."""

    def test_init(self):
        """Test WizardValidator initialization."""
        validator = WizardValidator()

        assert validator.validation_cache == {}
        assert hasattr(validator, "validate_api_key")
        assert hasattr(validator, "validate_url")
        assert hasattr(validator, "validate_port")

    def test_validation_cache_mechanism(self):
        """Test validation caching for performance."""
        validator = WizardValidator()

        # Validate same API key twice
        result1 = validator.validate_api_key(
            "openai", "sk-test123456789012345678901234567890"
        )
        result2 = validator.validate_api_key(
            "openai", "sk-test123456789012345678901234567890"
        )

        assert result1 == result2
        # Cache should be used for repeated validations


class TestAPIKeyValidation:
    """Comprehensive API key validation tests."""

    def test_validate_openai_api_key_valid(self):
        """Test validation of valid OpenAI API key."""
        validator = WizardValidator()

        valid_keys = [
            "sk-1234567890123456789012345678901234567890",
            "sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "sk-proj-1234567890123456789012345678901234567890",
        ]

        for key in valid_keys:
            is_valid, error = validator.validate_api_key("openai", key)
            assert is_valid is True
            assert error is None

    def test_validate_openai_api_key_invalid(self):
        """Test validation of invalid OpenAI API keys."""
        validator = WizardValidator()

        invalid_keys = [
            "sk-short",  # Too short
            "invalid-format",  # Wrong format
            "sk-",  # Empty after prefix
            "",  # Empty string
            "pk-1234567890123456789012345678901234567890",  # Wrong prefix
        ]

        for key in invalid_keys:
            is_valid, error = validator.validate_api_key("openai", key)
            assert is_valid is False
            assert error is not None
            assert isinstance(error, str)
            assert len(error) > 0

    def test_validate_firecrawl_api_key_valid(self):
        """Test validation of valid Firecrawl API key."""
        validator = WizardValidator()

        valid_keys = [
            "fc-1234567890123456789012345678901234567890",
            "fc-abcdefghijklmnopqrstuvwxyz1234567890",
        ]

        for key in valid_keys:
            is_valid, error = validator.validate_api_key("firecrawl", key)
            assert is_valid is True
            assert error is None

    def test_validate_firecrawl_api_key_invalid(self):
        """Test validation of invalid Firecrawl API keys."""
        validator = WizardValidator()

        invalid_keys = [
            "fc-short",  # Too short
            "sk-1234567890123456789012345678901234567890",  # Wrong prefix
            "fc-",  # Empty after prefix
        ]

        for key in invalid_keys:
            is_valid, error = validator.validate_api_key("firecrawl", key)
            assert is_valid is False
            assert error is not None

    def test_validate_anthropic_api_key_valid(self):
        """Test validation of valid Anthropic API key."""
        validator = WizardValidator()

        valid_keys = [
            "sk-ant-1234567890123456789012345678901234567890",
            "sk-ant-abcdefghijklmnopqrstuvwxyz1234567890",
        ]

        for key in valid_keys:
            is_valid, error = validator.validate_api_key("anthropic", key)
            assert is_valid is True
            assert error is None

    def test_validate_anthropic_api_key_invalid(self):
        """Test validation of invalid Anthropic API keys."""
        validator = WizardValidator()

        invalid_keys = [
            "sk-ant-short",  # Too short
            "sk-1234567890123456789012345678901234567890",  # Wrong prefix
            "sk-ant-",  # Empty after prefix
        ]

        for key in invalid_keys:
            is_valid, error = validator.validate_api_key("anthropic", key)
            assert is_valid is False
            assert error is not None

    def test_validate_unknown_provider_api_key(self):
        """Test validation of API key for unknown provider."""
        validator = WizardValidator()

        # Should apply generic validation rules
        is_valid, error = validator.validate_api_key("unknown", "1234567890")
        assert is_valid is True
        assert error is None

        # Too short should fail
        is_valid, error = validator.validate_api_key("unknown", "short")
        assert is_valid is False
        assert "too short" in error

    def test_validate_empty_api_key(self):
        """Test validation of empty API key."""
        validator = WizardValidator()

        is_valid, error = validator.validate_api_key("openai", "")
        assert is_valid is False
        assert "cannot be empty" in error


class TestURLValidation:
    """Comprehensive URL validation tests."""

    def test_validate_url_valid_http(self):
        """Test validation of valid HTTP URLs."""
        validator = WizardValidator()

        valid_urls = [
            "http://localhost:6333",
            "http://127.0.0.1:6333",
            "http://example.com",
            "http://example.com:8080",
            "http://example.com/path",
            "http://subdomain.example.com",
        ]

        for url in valid_urls:
            is_valid, error = validator.validate_url(url)
            assert is_valid is True, f"URL {url} should be valid"
            assert error is None

    def test_validate_url_valid_https(self):
        """Test validation of valid HTTPS URLs."""
        validator = WizardValidator()

        valid_urls = [
            "https://localhost:6333",
            "https://example.com",
            "https://api.example.com:443",
            "https://cloud.qdrant.io",
            "https://example.com/path/to/resource",
        ]

        for url in valid_urls:
            is_valid, error = validator.validate_url(url)
            assert is_valid is True, f"URL {url} should be valid"
            assert error is None

    def test_validate_url_invalid_format(self):
        """Test validation of invalid URL formats."""
        validator = WizardValidator()

        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Wrong protocol
            "http://",  # Incomplete
            "://example.com",  # Missing protocol
            "http:example.com",  # Wrong format
            "",  # Empty
        ]

        for url in invalid_urls:
            is_valid, error = validator.validate_url(url)
            assert is_valid is False, f"URL {url} should be invalid"
            assert error is not None

    def test_validate_url_localhost_allowed(self):
        """Test URL validation with localhost allowed."""
        validator = WizardValidator()

        localhost_urls = [
            "http://localhost:6333",
            "http://127.0.0.1:6333",
            "https://localhost",
        ]

        for url in localhost_urls:
            is_valid, error = validator.validate_url(url, allow_localhost=True)
            assert is_valid is True
            assert error is None

    def test_validate_url_localhost_not_allowed(self):
        """Test URL validation with localhost not allowed."""
        validator = WizardValidator()

        localhost_urls = [
            "http://localhost:6333",
            "http://127.0.0.1:6333",
            "https://localhost",
        ]

        for url in localhost_urls:
            is_valid, error = validator.validate_url(url, allow_localhost=False)
            assert is_valid is False
            assert "Localhost URLs not allowed" in error

    def test_validate_empty_url(self):
        """Test validation of empty URL."""
        validator = WizardValidator()

        is_valid, error = validator.validate_url("")
        assert is_valid is False
        assert "cannot be empty" in error


class TestPortValidation:
    """Comprehensive port validation tests."""

    def test_validate_port_valid_string(self):
        """Test validation of valid port numbers as strings."""
        validator = WizardValidator()

        valid_ports = ["6333", "8080", "9000", "65535"]  # Exclude ports < 1024

        for port in valid_ports:
            is_valid, error = validator.validate_port(port)
            assert is_valid is True, f"Port {port} should be valid"
            assert error is None

    def test_validate_port_valid_integer(self):
        """Test validation of valid port numbers as integers."""
        validator = WizardValidator()

        valid_ports = [6333, 8080, 9000, 65535]  # Exclude ports < 1024

        for port in valid_ports:
            is_valid, error = validator.validate_port(port)
            assert is_valid is True, f"Port {port} should be valid"
            assert error is None

    def test_validate_port_privileged_ports(self):
        """Test validation of privileged ports (< 1024)."""
        validator = WizardValidator()

        privileged_ports = [80, 443, 22, 21]

        for port in privileged_ports:
            is_valid, error = validator.validate_port(port)
            assert is_valid is False, f"Port {port} should be invalid (privileged)"
            assert "root privileges" in error

    def test_validate_port_invalid_range(self):
        """Test validation of ports outside valid range."""
        validator = WizardValidator()

        invalid_ports = [0, -1, 65536, 100000]

        for port in invalid_ports:
            is_valid, error = validator.validate_port(port)
            assert is_valid is False, f"Port {port} should be invalid"
            assert error is not None

    def test_validate_port_invalid_format(self):
        """Test validation of invalid port formats."""
        validator = WizardValidator()

        invalid_ports = ["not-a-number", "", "80.5", "abc"]

        for port in invalid_ports:
            is_valid, error = validator.validate_port(port)
            assert is_valid is False, f"Port {port} should be invalid"
            assert error is not None


class TestConfigurationValidation:
    """Tests for full configuration validation."""

    def test_validate_and_show_errors_valid_config(self, _rich_output_capturer):
        """Test validation of valid configuration."""
        validator = WizardValidator()

        valid_config = {
            "qdrant": {"host": "localhost", "port": 6333, "timeout": 30},
            "openai": {
                "api_key": "sk-1234567890123456789012345678901234567890",
                "model": "text-embedding-3-small",
            },
        }

        with patch("src.config.core.Config") as mock_config:
            # Mock successful validation
            mock_config.return_value = MagicMock()

            result = validator.validate_and_show_errors(valid_config)

            assert result is True

    def test_validate_and_show_errors_invalid_config(self, rich_output_capturer):
        """Test validation of invalid configuration with error display."""
        validator = WizardValidator()
        validator.console = rich_output_capturer.console

        invalid_config = {
            "qdrant": {
                "host": "",  # Invalid empty host
                "port": "invalid",  # Invalid port
            }
        }

        with patch("src.config.core.Config") as mock_config:
            # Mock validation error
            mock_config.side_effect = ValidationError(
                [
                    {
                        "loc": ("qdrant", "host"),
                        "msg": "field required",
                        "type": "value_error.missing",
                    }
                ],
                MagicMock,
            )

            result = validator.validate_and_show_errors(invalid_config)

            assert result is False
            rich_output_capturer.assert_contains("❌ Configuration Validation Errors")

    def test_show_validation_summary(self, rich_output_capturer):
        """Test validation summary display."""
        validator = WizardValidator()
        validator.console = rich_output_capturer.console

        # Mock config object
        mock_config = MagicMock()
        mock_config.qdrant.host = "localhost"
        mock_config.qdrant.port = 6333
        mock_config.openai.model = "text-embedding-3-small"

        validator.show_validation_summary(mock_config)

        rich_output_capturer.assert_panel_title("✅ Configuration Valid")
        rich_output_capturer.assert_contains("Configuration Summary")

    def test_validate_file_path_existing(self, tmp_path):
        """Test file path validation for existing files."""
        validator = WizardValidator()

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        is_valid, error = validator.validate_file_path(str(test_file), must_exist=True)
        assert is_valid is True
        assert error is None

    def test_validate_file_path_nonexistent_required(self, tmp_path):
        """Test file path validation for required but non-existent files."""
        validator = WizardValidator()

        nonexistent_file = tmp_path / "nonexistent.txt"

        is_valid, error = validator.validate_file_path(
            str(nonexistent_file), must_exist=True
        )
        assert is_valid is False
        assert "does not exist" in error

    def test_validate_file_path_new_file(self, tmp_path):
        """Test file path validation for new files."""
        validator = WizardValidator()

        new_file = tmp_path / "new_file.txt"

        is_valid, error = validator.validate_file_path(str(new_file), must_exist=False)
        assert is_valid is True
        assert error is None

    def test_validate_directory_path_existing(self, tmp_path):
        """Test directory path validation for existing directories."""
        validator = WizardValidator()

        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        is_valid, error = validator.validate_directory_path(
            str(test_dir), must_exist=True
        )
        assert is_valid is True
        assert error is None

    def test_validate_directory_path_nonexistent_required(self, tmp_path):
        """Test directory path validation for required but non-existent directories."""
        validator = WizardValidator()

        nonexistent_dir = tmp_path / "nonexistent_dir"

        is_valid, error = validator.validate_directory_path(
            str(nonexistent_dir), must_exist=True
        )
        assert is_valid is False
        assert "does not exist" in error

    def test_validate_json_string_valid(self):
        """Test JSON string validation with valid JSON."""
        validator = WizardValidator()

        valid_json = '{"key": "value", "number": 42}'

        is_valid, error, parsed = validator.validate_json_string(valid_json)
        assert is_valid is True
        assert error is None
        assert parsed == {"key": "value", "number": 42}

    def test_validate_json_string_invalid(self):
        """Test JSON string validation with invalid JSON."""
        validator = WizardValidator()

        invalid_json = '{"key": "value", "number":}'  # Missing value

        is_valid, error, parsed = validator.validate_json_string(invalid_json)
        assert is_valid is False
        assert error is not None
        assert parsed is None


class TestValidatorPerformance:
    """Performance and caching tests for WizardValidator."""

    def test_validation_caching_performance(self):
        """Test that validation caching improves performance."""
        validator = WizardValidator()

        # First validation - should be slower (no cache)
        api_key = "sk-1234567890123456789012345678901234567890"

        # Validate multiple times - subsequent calls should use cache
        for _ in range(5):
            is_valid, error = validator.validate_api_key("openai", api_key)
            assert is_valid is True
            assert error is None

    def test_large_configuration_validation(self):
        """Test validation of large configuration objects."""
        validator = WizardValidator()

        # Create a large configuration
        large_config = {
            "qdrant": {"host": "localhost", "port": 6333},
            "openai": {"api_key": "sk-" + "1" * 50, "model": "text-embedding-3-small"},
            "cache": {"enabled": True, "ttl": 3600},
            "browser": {"headless": True, "timeout": 30},
            "performance": {"max_workers": 4, "batch_size": 100},
        }

        # Add many additional fields
        for i in range(100):
            large_config[f"field_{i}"] = f"value_{i}"

        with patch("src.config.core.Config") as mock_config:
            mock_config.return_value = MagicMock()

            result = validator.validate_and_show_errors(large_config)
            # Should handle large configs without issues
            assert result is True


class TestValidatorIntegration:
    """Integration tests for WizardValidator with real scenarios."""

    def test_real_world_configuration_scenarios(self):
        """Test validator with real-world configuration scenarios."""
        validator = WizardValidator()

        scenarios = [
            # Local development
            {
                "qdrant": {"host": "localhost", "port": 6333},
                "openai": {"api_key": "sk-" + "1" * 40},
            },
            # Cloud deployment
            {
                "qdrant": {"url": "https://cloud.qdrant.io", "api_key": "qdrant-key"},
                "openai": {"api_key": "sk-" + "2" * 40},
            },
            # Minimal configuration
            {
                "qdrant": {"host": "localhost", "port": 6333},
            },
        ]

        for scenario in scenarios:
            with patch("src.config.core.Config") as mock_config:
                mock_config.return_value = MagicMock()

                result = validator.validate_and_show_errors(scenario)
                assert result is True

    def test_validation_error_message_quality(self, rich_output_capturer):
        """Test that validation error messages are helpful and user-friendly."""
        validator = WizardValidator()
        validator.console = rich_output_capturer.console

        # Test API key validation message
        is_valid, error = validator.validate_api_key("openai", "invalid-key")
        assert is_valid is False
        assert "OpenAI API key must start with 'sk-'" in error

        # Test URL validation message
        is_valid, error = validator.validate_url("not-a-url")
        assert is_valid is False
        assert "Invalid URL format" in error

        # Test port validation message
        is_valid, error = validator.validate_port("99999")
        assert is_valid is False
        assert "must be between" in error

    def test_rich_console_integration(self, rich_output_capturer):
        """Test Rich console integration for all validation methods."""
        validator = WizardValidator()
        validator.console = rich_output_capturer.console

        # Test validation summary display
        mock_config = MagicMock()
        mock_config.qdrant.host = "localhost"
        mock_config.openai.model = "text-embedding-3-small"

        validator.show_validation_summary(mock_config)

        # Verify Rich formatting
        output = rich_output_capturer.get_output()
        assert len(output) > 0
        assert "✅" in output  # Should have success indicator

        # Test error display
        rich_output_capturer.reset()

        with patch("src.config.core.Config") as mock_config_class:
            mock_config_class.side_effect = ValidationError(
                [{"loc": ("test",), "msg": "test error", "type": "value_error"}],
                MagicMock,
            )

            validator.validate_and_show_errors({"test": "invalid"})

            error_output = rich_output_capturer.get_output()
            assert "❌" in error_output  # Should have error indicator
