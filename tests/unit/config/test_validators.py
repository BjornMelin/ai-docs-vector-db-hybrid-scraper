"""Tests for configuration validators.

This test file covers the validation functions used throughout the configuration system
including API key validators, URL validators, and configuration validators.
"""

import pytest
from src.config.validators import ConfigValidator
from src.config.validators import validate_api_key_common
from src.config.validators import validate_chunk_sizes
from src.config.validators import validate_url_format


class TestValidateApiKeyCommon:
    """Test the validate_api_key_common function."""

    def test_validate_api_key_common_valid_openai(self):
        """Test valid OpenAI API key."""
        key = "sk-abcdef1234567890abcdef1234567890abcdef12"
        result = validate_api_key_common(key, "sk-", "OpenAI")
        assert result == key

    def test_validate_api_key_common_none_value(self):
        """Test None API key value."""
        result = validate_api_key_common(None, "sk-", "OpenAI")
        assert result is None

    def test_validate_api_key_common_wrong_prefix(self):
        """Test API key with wrong prefix."""
        with pytest.raises(ValueError, match="API key must start with"):
            validate_api_key_common("pk-wrongprefix", "sk-", "OpenAI")


class TestValidateUrlFormat:
    """Test the validate_url_format function."""

    def test_validate_url_format_https(self):
        """Test valid HTTPS URL."""
        url = "https://example.com"
        result = validate_url_format(url)
        assert result == url

    def test_validate_url_format_invalid_scheme(self):
        """Test URL with invalid scheme."""
        with pytest.raises(ValueError, match="URL must start with http"):
            validate_url_format("ftp://example.com")


class TestValidateChunkSizes:
    """Test the validate_chunk_sizes function."""

    def test_validate_chunk_sizes_valid(self):
        """Test valid chunk sizes."""
        validate_chunk_sizes(1000, 200, 100, 2000)

    def test_validate_chunk_sizes_overlap_too_large(self):
        """Test chunk overlap that's too large."""
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            validate_chunk_sizes(1000, 1000, 100, 2000)


class TestConfigValidator:
    """Test the ConfigValidator class methods."""

    def test_validate_env_var_format_valid(self):
        """Test valid environment variable format."""
        assert ConfigValidator.validate_env_var_format("AI_DOCS__CONFIG") is True
        assert ConfigValidator.validate_env_var_format("DATABASE_URL") is True

    def test_validate_url_valid(self):
        """Test valid URL validation."""
        is_valid, error = ConfigValidator.validate_url("https://example.com")
        assert is_valid is True
        assert error == ""

    def test_validate_api_key_openai_valid(self):
        """Test valid OpenAI API key validation."""
        is_valid, error = ConfigValidator.validate_api_key(
            "sk-abcdef1234567890abcdef1234567890abcdef12", "openai"
        )
        assert is_valid is True
        assert error == ""
