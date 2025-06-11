"""Comprehensive tests for configuration validators.

This test file covers the validation functions used throughout the configuration system
including API key validators, URL validators, and configuration validators.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.validators import ConfigValidator
from src.config.validators import validate_api_key_common
from src.config.validators import validate_chunk_sizes
from src.config.validators import validate_model_benchmark_consistency
from src.config.validators import validate_rate_limit_config
from src.config.validators import validate_scoring_weights
from src.config.validators import validate_url_format


class TestValidateApiKeyCommon:
    """Test the validate_api_key_common function."""

    def test_validate_api_key_common_valid_openai(self):
        """Test valid OpenAI API key."""
        key = "sk-abcdef1234567890abcdef1234567890abcdef12"
        result = validate_api_key_common(key, "sk-", "OpenAI")
        assert result == key

    def test_validate_api_key_common_valid_test_key(self):
        """Test valid OpenAI test API key."""
        key = "sk-test-abcdef123"
        result = validate_api_key_common(key, "sk-", "OpenAI")
        assert result == key

    def test_validate_api_key_common_none_value(self):
        """Test None API key value."""
        result = validate_api_key_common(None, "sk-", "OpenAI")
        assert result is None

    def test_validate_api_key_common_empty_string(self):
        """Test empty string API key."""
        result = validate_api_key_common("", "sk-", "OpenAI")
        assert result is None

    def test_validate_api_key_common_wrong_prefix(self):
        """Test API key with wrong prefix."""
        with pytest.raises(ValueError, match="API key must start with"):
            validate_api_key_common("pk-wrongprefix", "sk-", "OpenAI")

    def test_validate_api_key_common_too_short(self):
        """Test API key that's too short."""
        with pytest.raises(ValueError, match="appears to be too short"):
            validate_api_key_common("sk-short", "sk-", "OpenAI")

    def test_validate_api_key_common_too_long(self):
        """Test API key that's too long."""
        long_key = "sk-" + "a" * 300
        with pytest.raises(ValueError, match="appears to be too long"):
            validate_api_key_common(long_key, "sk-", "OpenAI")

    def test_validate_api_key_common_non_ascii(self):
        """Test API key with non-ASCII characters."""
        with pytest.raises(ValueError, match="non-ASCII characters"):
            validate_api_key_common("sk-ñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoñoño", "sk-", "OpenAI")

    def test_validate_api_key_common_invalid_characters(self):
        """Test API key with invalid characters."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_api_key_common("sk-invalid@characters!", "sk-", "OpenAI")

    def test_validate_api_key_common_custom_prefix(self):
        """Test API key with custom prefix."""
        key = "fc-abcdef1234567890"
        result = validate_api_key_common(key, "fc-", "Firecrawl")
        assert result == key

    def test_validate_api_key_common_test_key_too_short(self):
        """Test test API key that's too short."""
        with pytest.raises(ValueError, match="test API key contains invalid characters"):
            validate_api_key_common("sk-test", "sk-", "OpenAI")

    def test_validate_api_key_common_test_key_invalid_chars(self):
        """Test test API key with invalid characters."""
        with pytest.raises(ValueError, match="test API key contains invalid characters"):
            validate_api_key_common("sk-test@invalid", "sk-", "OpenAI")


class TestValidateUrlFormat:
    """Test the validate_url_format function."""

    def test_validate_url_format_http(self):
        """Test valid HTTP URL."""
        url = "http://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_validate_url_format_https(self):
        """Test valid HTTPS URL."""
        url = "https://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_validate_url_format_trailing_slash_removed(self):
        """Test that trailing slash is removed."""
        url = "https://example.com/"
        result = validate_url_format(url)
        assert result == "https://example.com"

    def test_validate_url_format_invalid_scheme(self):
        """Test URL with invalid scheme."""
        with pytest.raises(ValueError, match="URL must start with http"):
            validate_url_format("ftp://example.com")

    def test_validate_url_format_no_scheme(self):
        """Test URL without scheme."""
        with pytest.raises(ValueError, match="URL must start with http"):
            validate_url_format("example.com")


class TestValidateChunkSizes:
    """Test the validate_chunk_sizes function."""

    def test_validate_chunk_sizes_valid(self):
        """Test valid chunk sizes."""
        # Should not raise any exception
        validate_chunk_sizes(1000, 200, 100, 2000)

    def test_validate_chunk_sizes_overlap_too_large(self):
        """Test chunk overlap that's too large."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            validate_chunk_sizes(1000, 1000, 100, 2000)

    def test_validate_chunk_sizes_overlap_larger_than_size(self):
        """Test chunk overlap larger than chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            validate_chunk_sizes(1000, 1200, 100, 2000)

    def test_validate_chunk_sizes_min_larger_than_max(self):
        """Test min chunk size larger than max."""
        with pytest.raises(ValueError, match="min_chunk_size must be less than max_chunk_size"):
            validate_chunk_sizes(1000, 200, 2000, 1500)

    def test_validate_chunk_sizes_chunk_size_too_large(self):
        """Test chunk size exceeding maximum."""
        with pytest.raises(ValueError, match="chunk_size cannot exceed max_chunk_size"):
            validate_chunk_sizes(3000, 200, 100, 2000)


class TestValidateRateLimitConfig:
    """Test the validate_rate_limit_config function."""

    def test_validate_rate_limit_config_valid(self):
        """Test valid rate limit configuration."""
        config = {
            "openai": {
                "max_calls": 100,
                "time_window": 60
            },
            "firecrawl": {
                "max_calls": 50,
                "time_window": 60
            }
        }
        result = validate_rate_limit_config(config)
        assert result == config

    def test_validate_rate_limit_config_invalid_structure(self):
        """Test rate limit config with invalid structure."""
        config = {
            "openai": "not_a_dict"
        }
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_rate_limit_config(config)

    def test_validate_rate_limit_config_missing_keys(self):
        """Test rate limit config with missing required keys."""
        config = {
            "openai": {
                "max_calls": 100
                # missing time_window
            }
        }
        with pytest.raises(ValueError, match="must contain keys"):
            validate_rate_limit_config(config)

    def test_validate_rate_limit_config_zero_max_calls(self):
        """Test rate limit config with zero max_calls."""
        config = {
            "openai": {
                "max_calls": 0,
                "time_window": 60
            }
        }
        with pytest.raises(ValueError, match="max_calls .* must be positive"):
            validate_rate_limit_config(config)

    def test_validate_rate_limit_config_negative_time_window(self):
        """Test rate limit config with negative time_window."""
        config = {
            "openai": {
                "max_calls": 100,
                "time_window": -60
            }
        }
        with pytest.raises(ValueError, match="time_window .* must be positive"):
            validate_rate_limit_config(config)


class TestValidateScoringWeights:
    """Test the validate_scoring_weights function."""

    def test_validate_scoring_weights_valid(self):
        """Test valid scoring weights."""
        # Should not raise any exception
        validate_scoring_weights(0.5, 0.3, 0.2)

    def test_validate_scoring_weights_sum_to_one(self):
        """Test scoring weights that sum to exactly 1.0."""
        validate_scoring_weights(0.33, 0.33, 0.34)

    def test_validate_scoring_weights_invalid_sum(self):
        """Test scoring weights that don't sum to 1.0."""
        with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
            validate_scoring_weights(0.3, 0.3, 0.3)

    def test_validate_scoring_weights_sum_too_low(self):
        """Test scoring weights that sum too low."""
        with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
            validate_scoring_weights(0.2, 0.2, 0.2)

    def test_validate_scoring_weights_sum_too_high(self):
        """Test scoring weights that sum too high."""
        with pytest.raises(ValueError, match="Scoring weights must sum to 1.0"):
            validate_scoring_weights(0.5, 0.5, 0.5)


class TestValidateModelBenchmarkConsistency:
    """Test the validate_model_benchmark_consistency function."""

    def test_validate_model_benchmark_consistency_valid(self):
        """Test valid model benchmark consistency."""
        key = "gpt-4"
        model_name = "gpt-4"
        result = validate_model_benchmark_consistency(key, model_name)
        assert result == key

    def test_validate_model_benchmark_consistency_mismatch(self):
        """Test model benchmark consistency with mismatch."""
        with pytest.raises(ValueError, match="Dictionary key .* does not match ModelBenchmark.model_name"):
            validate_model_benchmark_consistency("gpt-4", "gpt-3.5-turbo")


class TestConfigValidator:
    """Test the ConfigValidator class methods."""

    def test_validate_env_var_format_valid(self):
        """Test valid environment variable format."""
        assert ConfigValidator.validate_env_var_format("AI_DOCS__CONFIG") is True
        assert ConfigValidator.validate_env_var_format("DATABASE_URL") is True
        assert ConfigValidator.validate_env_var_format("TEST_123") is True

    def test_validate_env_var_format_invalid(self):
        """Test invalid environment variable format."""
        assert ConfigValidator.validate_env_var_format("lowercase") is False
        assert ConfigValidator.validate_env_var_format("123_START_WITH_NUMBER") is False
        assert ConfigValidator.validate_env_var_format("INVALID-DASH") is False
        assert ConfigValidator.validate_env_var_format("INVALID SPACE") is False

    def test_validate_env_var_format_with_pattern(self):
        """Test environment variable format with expected pattern."""
        pattern = r"^AI_DOCS__.*"
        assert ConfigValidator.validate_env_var_format("AI_DOCS__CONFIG", pattern) is True
        assert ConfigValidator.validate_env_var_format("OTHER_VAR", pattern) is False

    def test_validate_url_valid(self):
        """Test valid URL validation."""
        is_valid, error = ConfigValidator.validate_url("https://example.com")
        assert is_valid is True
        assert error == ""

    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid scheme."""
        is_valid, error = ConfigValidator.validate_url("ftp://example.com")
        assert is_valid is False
        assert "Invalid URL scheme" in error

    def test_validate_url_missing_netloc(self):
        """Test URL validation with missing network location."""
        is_valid, error = ConfigValidator.validate_url("https://")
        assert is_valid is False
        assert "missing network location" in error

    def test_validate_url_custom_schemes(self):
        """Test URL validation with custom allowed schemes."""
        is_valid, error = ConfigValidator.validate_url("ftp://example.com", schemes=["ftp"])
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_openai_valid(self):
        """Test valid OpenAI API key validation."""
        is_valid, error = ConfigValidator.validate_api_key("sk-abcdef1234567890abcdef1234567890abcdef12", "openai")
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_openai_test_key(self):
        """Test OpenAI test API key validation."""
        is_valid, error = ConfigValidator.validate_api_key("sk-test-abcdef123", "openai")
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_openai_invalid_prefix(self):
        """Test OpenAI API key with invalid prefix."""
        is_valid, error = ConfigValidator.validate_api_key("pk-wrongprefix", "openai")
        assert is_valid is False
        assert "must start with 'sk-'" in error

    def test_validate_api_key_openai_too_short(self):
        """Test OpenAI API key that's too short."""
        is_valid, error = ConfigValidator.validate_api_key("sk-short", "openai")
        assert is_valid is False
        assert "too short" in error

    def test_validate_api_key_firecrawl_valid(self):
        """Test valid Firecrawl API key validation."""
        is_valid, error = ConfigValidator.validate_api_key("fc-abcdef1234567890abcdef", "firecrawl")
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_firecrawl_too_short(self):
        """Test Firecrawl API key that's too short."""
        is_valid, error = ConfigValidator.validate_api_key("fc-short", "firecrawl")
        assert is_valid is False
        assert "too short" in error

    def test_validate_api_key_empty(self):
        """Test empty API key validation."""
        is_valid, error = ConfigValidator.validate_api_key("", "openai")
        assert is_valid is False
        assert "empty" in error

    def test_validate_api_key_qdrant_optional(self):
        """Test Qdrant API key validation (optional)."""
        is_valid, error = ConfigValidator.validate_api_key("valid_qdrant_key", "qdrant")
        assert is_valid is True
        assert error == ""

    def test_validate_env_var_value_boolean_true(self):
        """Test environment variable value conversion to boolean (true)."""
        for value in ["true", "1", "yes", "on", "TRUE", "YES"]:
            is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", value, bool)
            assert is_valid is True
            assert converted is True
            assert error == ""

    def test_validate_env_var_value_boolean_false(self):
        """Test environment variable value conversion to boolean (false)."""
        for value in ["false", "0", "no", "off", "FALSE", "NO"]:
            is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", value, bool)
            assert is_valid is True
            assert converted is False
            assert error == ""

    def test_validate_env_var_value_boolean_invalid(self):
        """Test environment variable value conversion to boolean (invalid)."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", "maybe", bool)
        assert is_valid is False
        assert converted is None
        assert "Invalid boolean value" in error

    def test_validate_env_var_value_integer(self):
        """Test environment variable value conversion to integer."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", "123", int)
        assert is_valid is True
        assert converted == 123
        assert error == ""

    def test_validate_env_var_value_integer_invalid(self):
        """Test environment variable value conversion to integer (invalid)."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", "abc", int)
        assert is_valid is False
        assert converted is None
        assert "Failed to convert" in error

    def test_validate_env_var_value_float(self):
        """Test environment variable value conversion to float."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", "123.45", float)
        assert is_valid is True
        assert converted == 123.45
        assert error == ""

    def test_validate_env_var_value_list(self):
        """Test environment variable value conversion to list."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", '["a", "b", "c"]', list)
        assert is_valid is True
        assert converted == ["a", "b", "c"]
        assert error == ""

    def test_validate_env_var_value_dict(self):
        """Test environment variable value conversion to dict."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", '{"key": "value"}', dict)
        assert is_valid is True
        assert converted == {"key": "value"}
        assert error == ""

    def test_validate_env_var_value_json_invalid(self):
        """Test environment variable value conversion with invalid JSON."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", '{"invalid": json}', dict)
        assert is_valid is False
        assert converted is None
        assert "Failed to convert" in error

    def test_validate_env_var_value_string(self):
        """Test environment variable value as string."""
        is_valid, converted, error = ConfigValidator.validate_env_var_value("TEST_VAR", "test_value", str)
        assert is_valid is True
        assert converted == "test_value"
        assert error == ""

    @patch.dict(os.environ, {
        "AI_DOCS__TEST_VAR": "test_value",
        "AI_DOCS__EMPTY_VAR": "",
        "AI_DOCS__WHITESPACE_VAR": " trimme ",
        "AI_DOCS__PLACEHOLDER_VAR": "your-api-key-here",
        "OTHER_PREFIX": "ignored"
    })
    def test_check_env_vars(self):
        """Test checking environment variables."""
        results = ConfigValidator.check_env_vars("AI_DOCS__")
        
        # Should find AI_DOCS__ prefixed vars
        assert "AI_DOCS__TEST_VAR" in results
        assert "AI_DOCS__EMPTY_VAR" in results
        assert "AI_DOCS__WHITESPACE_VAR" in results
        assert "AI_DOCS__PLACEHOLDER_VAR" in results
        
        # Should not find other prefixed vars
        assert "OTHER_PREFIX" not in results
        
        # Check specific issues
        assert results["AI_DOCS__TEST_VAR"]["valid_format"] is True
        assert len(results["AI_DOCS__TEST_VAR"]["issues"]) == 0
        
        assert "Empty value" in results["AI_DOCS__EMPTY_VAR"]["issues"]
        assert "Value has leading/trailing whitespace" in results["AI_DOCS__WHITESPACE_VAR"]["issues"]
        assert "Value appears to be a placeholder" in results["AI_DOCS__PLACEHOLDER_VAR"]["issues"]

    @patch('src.utils.health_checks.ServiceHealthChecker.perform_all_health_checks')
    def test_validate_config_connections(self, mock_health_check):
        """Test config connections validation."""
        # Mock health check results
        mock_health_check.return_value = {
            "qdrant": {"connected": True, "error": None},
            "database": {"connected": False, "error": "Connection refused"}
        }
        
        mock_config = MagicMock()
        results = ConfigValidator.validate_config_connections(mock_config)
        
        assert "qdrant" in results
        assert "database" in results
        assert results["qdrant"]["connected"] is True
        assert results["database"]["connected"] is False
        assert results["database"]["error"] == "Connection refused"

    def test_generate_validation_report(self):
        """Test validation report generation."""
        # Create mock config
        mock_config = MagicMock()
        mock_config.environment = "development"
        mock_config.debug = True
        mock_config.log_level = "DEBUG"
        mock_config.embedding_provider = "openai"
        mock_config.crawl_provider = "firecrawl"
        mock_config.cache.enable_caching = True
        mock_config.validate_completeness.return_value = []
        
        with patch.object(ConfigValidator, 'check_env_vars', return_value={}), \
             patch.object(ConfigValidator, 'validate_config_connections', return_value={}):
            
            report = ConfigValidator.generate_validation_report(mock_config)
            
            assert isinstance(report, str)
            assert "Configuration Validation Report" in report
            assert "Environment: development" in report
            assert "Debug Mode: True" in report
            assert "Log Level: DEBUG" in report
            assert "Embedding Provider: openai" in report


class TestValidatorIntegration:
    """Integration tests for validator functions."""

    def test_comprehensive_validation_flow(self):
        """Test comprehensive validation flow."""
        # Test API key validation
        assert validate_api_key_common("sk-test123456789", "sk-", "OpenAI") == "sk-test123456789"
        
        # Test URL validation
        assert validate_url_format("https://example.com/") == "https://example.com"
        
        # Test chunk sizes
        validate_chunk_sizes(1000, 200, 100, 2000)  # Should not raise
        
        # Test scoring weights
        validate_scoring_weights(0.5, 0.3, 0.2)  # Should not raise
        
        # Test model consistency
        assert validate_model_benchmark_consistency("gpt-4", "gpt-4") == "gpt-4"

    def test_validator_error_propagation(self):
        """Test that validator errors are properly propagated."""
        # API key validation error
        with pytest.raises(ValueError):
            validate_api_key_common("invalid-key", "sk-", "OpenAI")
        
        # URL validation error
        with pytest.raises(ValueError):
            validate_url_format("invalid-url")
        
        # Chunk size validation error
        with pytest.raises(ValueError):
            validate_chunk_sizes(1000, 1200, 100, 2000)
        
        # Scoring weights validation error
        with pytest.raises(ValueError):
            validate_scoring_weights(0.5, 0.5, 0.5)

    def test_config_validator_methods_consistency(self):
        """Test ConfigValidator methods work consistently."""
        validator = ConfigValidator()
        
        # Environment variable format
        assert validator.validate_env_var_format("VALID_VAR") is True
        assert validator.validate_env_var_format("invalid_var") is False
        
        # URL validation
        is_valid, _ = validator.validate_url("https://example.com")
        assert is_valid is True
        
        is_valid, _ = validator.validate_url("invalid-url")
        assert is_valid is False
        
        # API key validation
        is_valid, _ = validator.validate_api_key("sk-validkeyhere123456789012345678901234567890", "openai")
        assert is_valid is True
        
        is_valid, _ = validator.validate_api_key("invalid", "openai")
        assert is_valid is False

    def test_rate_limit_config_edge_cases(self):
        """Test rate limit config validation edge cases."""
        # Empty config should be valid
        result = validate_rate_limit_config({})
        assert result == {}
        
        # Single provider config
        config = {"openai": {"max_calls": 1, "time_window": 1}}
        result = validate_rate_limit_config(config)
        assert result == config
        
        # Multiple providers
        config = {
            "openai": {"max_calls": 100, "time_window": 60},
            "firecrawl": {"max_calls": 50, "time_window": 60},
            "custom": {"max_calls": 200, "time_window": 120}
        }
        result = validate_rate_limit_config(config)
        assert result == config

    def test_env_var_value_conversion_edge_cases(self):
        """Test environment variable value conversion edge cases."""
        validator = ConfigValidator()
        
        # Test various boolean representations
        for true_val in ["True", "TRUE", "tRuE", "1", "YES", "yes", "On", "ON"]:
            is_valid, converted, _ = validator.validate_env_var_value("TEST", true_val, bool)
            assert is_valid is True
            assert converted is True
        
        for false_val in ["False", "FALSE", "fAlSe", "0", "NO", "no", "Off", "OFF"]:
            is_valid, converted, _ = validator.validate_env_var_value("TEST", false_val, bool)
            assert is_valid is True
            assert converted is False
        
        # Test numeric edge cases
        is_valid, converted, _ = validator.validate_env_var_value("TEST", "0", int)
        assert is_valid is True
        assert converted == 0
        
        is_valid, converted, _ = validator.validate_env_var_value("TEST", "-123", int)
        assert is_valid is True
        assert converted == -123
        
        is_valid, converted, _ = validator.validate_env_var_value("TEST", "0.0", float)
        assert is_valid is True
        assert converted == 0.0