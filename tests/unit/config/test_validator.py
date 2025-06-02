"""Unit tests for configuration validator module."""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

from src.config.models import UnifiedConfig
from src.config.validator import ConfigValidator


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    def test_validate_env_var_format_valid(self):
        """Test validation of valid environment variable formats."""
        valid_names = [
            "AI_DOCS_TEST",
            "AI_DOCS__URL",
            "DEBUG_MODE",
            "API_KEY_1",
            "TEST_ENV_VAR",
            "A",
            "ABC_DEF_GHI",
        ]

        for name in valid_names:
            assert ConfigValidator.validate_env_var_format(name), (
                f"Should be valid: {name}"
            )

    def test_validate_env_var_format_invalid(self):
        """Test validation of invalid environment variable formats."""
        invalid_names = [
            "lowercase",
            "mixed_Case",
            "123_INVALID",
            "AI-DOCS-TEST",  # hyphen instead of underscore
            "AI.DOCS.TEST",  # dots
            "AI DOCS TEST",  # spaces
            "",  # empty
            "_STARTS_WITH_UNDERSCORE",
        ]

        for name in invalid_names:
            assert not ConfigValidator.validate_env_var_format(name), (
                f"Should be invalid: {name}"
            )

    def test_validate_env_var_format_with_pattern(self):
        """Test validation with expected pattern."""
        # Test pattern matching
        pattern = r"^AI_DOCS__.*"

        assert ConfigValidator.validate_env_var_format("AI_DOCS__TEST", pattern)
        assert not ConfigValidator.validate_env_var_format("OTHER__TEST", pattern)
        assert not ConfigValidator.validate_env_var_format(
            "AI_DOCS_TEST", pattern
        )  # Missing double underscore

    def test_validate_url_valid_http(self):
        """Test validation of valid HTTP URLs."""
        valid_urls = [
            "http://localhost:6333",
            "https://example.com",
            "http://192.168.1.1:8080",
            "https://api.example.com/v1",
            "http://test.local",
        ]

        for url in valid_urls:
            is_valid, error = ConfigValidator.validate_url(url)
            assert is_valid, f"Should be valid: {url} - Error: {error}"
            assert error == ""

    def test_validate_url_invalid_schemes(self):
        """Test validation of URLs with invalid schemes."""
        invalid_urls = [
            "ftp://example.com",
            "ws://example.com",
            "file:///path/to/file",
            "javascript:alert('xss')",
        ]

        for url in invalid_urls:
            is_valid, error = ConfigValidator.validate_url(url)
            assert not is_valid, f"Should be invalid: {url}"
            assert "Invalid URL scheme" in error

    def test_validate_url_missing_netloc(self):
        """Test validation of URLs missing network location."""
        invalid_urls = [
            "http://",
            "https://",
            "http:///path",
        ]

        for url in invalid_urls:
            is_valid, error = ConfigValidator.validate_url(url)
            assert not is_valid, f"Should be invalid: {url}"
            assert "missing network location" in error

    def test_validate_url_custom_schemes(self):
        """Test validation with custom allowed schemes."""
        redis_urls = [
            "redis://localhost:6379",
            "rediss://secure.redis.com:6380",
        ]

        for url in redis_urls:
            is_valid, error = ConfigValidator.validate_url(
                url, schemes=["redis", "rediss"]
            )
            assert is_valid, f"Should be valid with custom schemes: {url}"

    def test_validate_url_malformed(self):
        """Test validation of malformed URLs."""
        malformed_urls = [
            "not-a-url",
            "://missing-scheme",
            "http:missing-slashes",
        ]

        for url in malformed_urls:
            is_valid, error = ConfigValidator.validate_url(url)
            assert not is_valid, f"Should be invalid: {url}"
            assert error != ""

    def test_validate_api_key_openai_valid(self):
        """Test validation of valid OpenAI API keys."""
        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef12345678",
            "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890",
            "sk-" + "a" * 40,  # Minimum length
        ]

        for key in valid_keys:
            is_valid, error = ConfigValidator.validate_api_key(key, "openai")
            assert is_valid, f"Should be valid OpenAI key: {key} - Error: {error}"
            assert error == ""

    def test_validate_api_key_openai_invalid(self):
        """Test validation of invalid OpenAI API keys."""
        invalid_keys = [
            "",  # Empty
            "sk-",  # Too short
            "sk-short",  # Too short
            "not-sk-prefix",  # Wrong prefix
            "api-key-123456789",  # Wrong prefix
        ]

        for key in invalid_keys:
            is_valid, error = ConfigValidator.validate_api_key(key, "openai")
            assert not is_valid, f"Should be invalid OpenAI key: {key}"
            assert error != ""

    def test_validate_api_key_firecrawl_valid(self):
        """Test validation of valid Firecrawl API keys."""
        valid_keys = [
            "fc-1234567890abcdef1234",
            "firecrawl-" + "x" * 30,
            "a" * 25,  # Minimum length
        ]

        for key in valid_keys:
            is_valid, error = ConfigValidator.validate_api_key(key, "firecrawl")
            assert is_valid, f"Should be valid Firecrawl key: {key} - Error: {error}"
            assert error == ""

    def test_validate_api_key_firecrawl_invalid(self):
        """Test validation of invalid Firecrawl API keys."""
        invalid_keys = [
            "",  # Empty
            "short",  # Too short
            "a" * 10,  # Too short
        ]

        for key in invalid_keys:
            is_valid, error = ConfigValidator.validate_api_key(key, "firecrawl")
            assert not is_valid, f"Should be invalid Firecrawl key: {key}"
            assert error != ""

    def test_validate_api_key_qdrant(self):
        """Test validation of Qdrant API keys."""
        # Empty keys are treated as invalid by the validation function
        is_valid, error = ConfigValidator.validate_api_key("", "qdrant")
        assert not is_valid
        assert "API key is empty" in error

        is_valid, error = ConfigValidator.validate_api_key(None, "qdrant")
        assert not is_valid
        assert "API key is empty" in error

        # Valid key (longer than 10 chars)
        is_valid, error = ConfigValidator.validate_api_key(
            "qdrant-key-123456", "qdrant"
        )
        assert is_valid
        assert error == ""

        # Too short (if provided)
        is_valid, error = ConfigValidator.validate_api_key("short", "qdrant")
        assert not is_valid
        assert "too short" in error

    def test_validate_env_var_value_boolean(self):
        """Test validation and conversion of boolean environment variables."""
        true_values = ["true", "1", "yes", "on", "TRUE", "Yes", "ON"]
        false_values = ["false", "0", "no", "off", "FALSE", "No", "OFF"]

        for value in true_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, bool
            )
            assert is_valid, f"Should be valid boolean: {value}"
            assert converted is True
            assert error == ""

        for value in false_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, bool
            )
            assert is_valid, f"Should be valid boolean: {value}"
            assert converted is False
            assert error == ""

    def test_validate_env_var_value_boolean_invalid(self):
        """Test validation of invalid boolean values."""
        invalid_values = ["maybe", "2", "true1", "false0", ""]

        for value in invalid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, bool
            )
            assert not is_valid, f"Should be invalid boolean: {value}"
            assert converted is None
            assert "Invalid boolean value" in error

    def test_validate_env_var_value_integer(self):
        """Test validation and conversion of integer environment variables."""
        valid_values = ["0", "123", "-456", "1000000"]

        for value in valid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, int
            )
            assert is_valid, f"Should be valid integer: {value}"
            assert converted == int(value)
            assert error == ""

    def test_validate_env_var_value_integer_invalid(self):
        """Test validation of invalid integer values."""
        invalid_values = ["abc", "123.45", "1e10", ""]

        for value in invalid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, int
            )
            assert not is_valid, f"Should be invalid integer: {value}"
            assert converted is None
            assert "Failed to convert" in error

    def test_validate_env_var_value_float(self):
        """Test validation and conversion of float environment variables."""
        valid_values = ["0.0", "123.45", "-456.78", "1e10", "1.0e-5"]

        for value in valid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, float
            )
            assert is_valid, f"Should be valid float: {value}"
            assert converted == float(value)
            assert error == ""

    def test_validate_env_var_value_float_invalid(self):
        """Test validation of invalid float values."""
        invalid_values = ["abc", "123.45.67", ""]

        for value in invalid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, float
            )
            assert not is_valid, f"Should be invalid float: {value}"
            assert converted is None
            assert "Failed to convert" in error

    def test_validate_env_var_value_list(self):
        """Test validation and conversion of list environment variables."""
        valid_values = [
            ('["a", "b", "c"]', ["a", "b", "c"]),
            ("[1, 2, 3]", [1, 2, 3]),
            ("[]", []),
        ]

        for value, expected in valid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, list
            )
            assert is_valid, f"Should be valid list: {value}"
            assert converted == expected
            assert error == ""

    def test_validate_env_var_value_list_invalid(self):
        """Test validation of invalid list values."""
        invalid_values = [
            '{"key": "value"}',  # Dict instead of list
            '"string"',  # String instead of list
            "invalid json",
            "",
        ]

        for value in invalid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, list
            )
            assert not is_valid, f"Should be invalid list: {value}"
            assert converted is None

    def test_validate_env_var_value_dict(self):
        """Test validation and conversion of dict environment variables."""
        valid_values = [
            ('{"key": "value"}', {"key": "value"}),
            ('{"num": 123}', {"num": 123}),
            ("{}", {}),
        ]

        for value, expected in valid_values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, dict
            )
            assert is_valid, f"Should be valid dict: {value}"
            assert converted == expected
            assert error == ""

    def test_validate_env_var_value_string(self):
        """Test validation of string environment variables."""
        values = ["simple", "with spaces", "123", "true", ""]

        for value in values:
            is_valid, converted, error = ConfigValidator.validate_env_var_value(
                "TEST_VAR", value, str
            )
            assert is_valid, f"Should be valid string: {value}"
            assert converted == value
            assert error == ""

    def test_check_env_vars_with_prefix(self):
        """Test checking environment variables with specific prefix."""
        # Clear any existing AI_DOCS__ variables first, then set our test ones
        test_env = {
            k: v for k, v in os.environ.items() if not k.startswith("AI_DOCS__")
        }
        test_env.update(
            {
                "AI_DOCS__DEBUG": "true",
                "AI_DOCS__API_KEY": "sk-test",
                "AI_DOCS__PLACEHOLDER": "your-api-key",
                "AI_DOCS__EMPTY": "",
                "AI_DOCS__WHITESPACE": " value ",
                "OTHER_VAR": "ignored",
                "invalid_format": "also ignored",
            }
        )

        with patch.dict(os.environ, test_env, clear=True):
            results = ConfigValidator.check_env_vars("AI_DOCS__")

            # Should only include AI_DOCS__ prefixed variables
            assert len(results) == 5
            assert "AI_DOCS__DEBUG" in results
            assert "AI_DOCS__API_KEY" in results
            assert "OTHER_VAR" not in results
            assert "invalid_format" not in results

    def test_check_env_vars_validation_issues(self):
        """Test detection of environment variable issues."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__VALID": "good-value",
                "AI_DOCS__EMPTY": "",
                "AI_DOCS__PLACEHOLDER": "your-api-key",
                "AI_DOCS__WHITESPACE": " value ",
                "AI_DOCS__XXX": "xxx-placeholder",
                "invalid_name": "bad-format",
            },
            clear=False,
        ):
            results = ConfigValidator.check_env_vars("AI_DOCS__")

            # Check specific issues
            assert len(results["AI_DOCS__VALID"]["issues"]) == 0
            assert "Empty value" in results["AI_DOCS__EMPTY"]["issues"]
            assert "placeholder" in results["AI_DOCS__PLACEHOLDER"]["issues"][0]
            assert "whitespace" in results["AI_DOCS__WHITESPACE"]["issues"][0]
            assert "placeholder" in results["AI_DOCS__XXX"]["issues"][0]

            # Invalid format should not be included
            assert "invalid_name" not in results

    @patch("qdrant_client.QdrantClient")
    def test_validate_config_connections_qdrant_success(self, mock_client_class):
        """Test successful Qdrant connection validation."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client_class.return_value = mock_client

        config = UnifiedConfig(qdrant={"url": "http://localhost:6333", "api_key": None})

        results = ConfigValidator.validate_config_connections(config)

        assert "qdrant" in results
        assert results["qdrant"]["connected"] is True
        assert results["qdrant"]["error"] is None

    @patch("qdrant_client.QdrantClient")
    def test_validate_config_connections_qdrant_auth_failure(self, mock_client_class):
        """Test Qdrant authentication failure."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        mock_client.get_collections.side_effect = UnexpectedResponse(
            status_code=401,
            reason_phrase="Unauthorized",
            content=b"Unauthorized",
            headers={},
        )
        mock_client_class.return_value = mock_client

        config = UnifiedConfig(
            qdrant={"url": "http://localhost:6333", "api_key": "wrong"}
        )

        results = ConfigValidator.validate_config_connections(config)

        assert results["qdrant"]["connected"] is False
        assert "Authentication failed" in results["qdrant"]["error"]

    @patch("redis.from_url")
    def test_validate_config_connections_redis_success(self, mock_redis):
        """Test successful Redis connection validation."""
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        config = UnifiedConfig(
            cache={
                "enable_dragonfly_cache": True,
                "dragonfly_url": "redis://localhost:6379",
            }
        )

        results = ConfigValidator.validate_config_connections(config)

        assert "redis" in results
        assert results["redis"]["connected"] is True
        assert results["redis"]["error"] is None

    @patch("redis.from_url")
    def test_validate_config_connections_redis_failure(self, mock_redis):
        """Test Redis connection failure."""
        import redis

        mock_redis.side_effect = redis.ConnectionError("Connection refused")

        config = UnifiedConfig(
            cache={
                "enable_dragonfly_cache": True,
                "dragonfly_url": "redis://localhost:6379",
            }
        )

        results = ConfigValidator.validate_config_connections(config)

        assert results["redis"]["connected"] is False
        assert "Connection refused" in results["redis"]["error"]

    @patch("openai.OpenAI")
    def test_validate_config_connections_openai_success(self, mock_openai_class):
        """Test successful OpenAI connection validation."""
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_client.models.list.return_value = mock_models
        mock_openai_class.return_value = mock_client

        config = UnifiedConfig(
            embedding_provider="openai",
            openai={"api_key": "sk-test"},
        )

        results = ConfigValidator.validate_config_connections(config)

        assert "openai" in results
        assert results["openai"]["connected"] is True
        assert results["openai"]["error"] is None

    def test_validate_config_connections_skip_disabled(self):
        """Test that disabled services are skipped."""
        config = UnifiedConfig(
            embedding_provider="fastembed",  # Not OpenAI
            cache={"enable_dragonfly_cache": False},  # DragonflyDB disabled
            openai={"api_key": None},  # No API key
        )

        results = ConfigValidator.validate_config_connections(config)

        # Should only check Qdrant (always checked)
        assert "qdrant" in results
        assert "redis" not in results
        assert "openai" not in results

    def test_generate_validation_report_basic(self):
        """Test basic validation report generation."""
        config = UnifiedConfig(environment="development", debug=True)

        with patch.object(ConfigValidator, "check_env_vars", return_value={}):
            with patch.object(
                ConfigValidator, "validate_config_connections", return_value={}
            ):
                report = ConfigValidator.generate_validation_report(config)

        assert isinstance(report, str)
        assert "Configuration Validation Report" in report
        assert "Environment: development" in report
        assert "Debug Mode: True" in report

    def test_generate_validation_report_with_issues(self):
        """Test validation report with configuration issues."""
        # Create config with validation issues
        config = UnifiedConfig(
            environment="production",
            debug=True,  # Invalid for production
            openai={"api_key": ""},  # Empty API key
        )

        with patch.object(ConfigValidator, "check_env_vars", return_value={}):
            with patch.object(
                ConfigValidator, "validate_config_connections", return_value={}
            ):
                report = ConfigValidator.generate_validation_report(config)

        assert (
            "⚠️  Configuration issues found:" in report
            or "❌ Validation failed:" in report
        )

    def test_generate_validation_report_with_env_vars(self):
        """Test validation report with environment variables."""
        config = UnifiedConfig()

        env_vars = {
            "AI_DOCS__DEBUG": {"value": "true", "issues": []},
            "AI_DOCS__INVALID": {"value": "bad", "issues": ["Invalid value"]},
        }

        with patch.object(ConfigValidator, "check_env_vars", return_value=env_vars):
            with patch.object(
                ConfigValidator, "validate_config_connections", return_value={}
            ):
                report = ConfigValidator.generate_validation_report(config)

        assert "Environment Variables:" in report
        assert "AI_DOCS__DEBUG" in report
        assert "AI_DOCS__INVALID" in report

    def test_generate_validation_report_with_connections(self):
        """Test validation report with service connections."""
        config = UnifiedConfig()

        connections = {
            "qdrant": {"connected": True, "error": None},
            "redis": {"connected": False, "error": "Connection refused"},
        }

        with patch.object(ConfigValidator, "check_env_vars", return_value={}):
            with patch.object(
                ConfigValidator, "validate_config_connections", return_value=connections
            ):
                report = ConfigValidator.generate_validation_report(config)

        assert "Service Connections:" in report
        assert "✅ Qdrant" in report
        assert "❌ Redis" in report
        assert "Connection refused" in report
