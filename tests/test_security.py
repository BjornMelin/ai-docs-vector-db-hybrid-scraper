#!/usr/bin/env python3
"""Tests for security utilities."""

import pytest
from src.security import APIKeyValidator
from src.security import SecurityError
from src.security import SecurityValidator


class TestSecurityValidator:
    """Test security validation utilities."""

    def test_validate_url_valid(self):
        """Test valid URL validation."""
        validator = SecurityValidator.from_unified_config()
        valid_urls = [
            "https://example.com",
            "http://docs.python.org",
            "https://api.github.com/repos",
            "https://openai.com/api/docs",
        ]

        for url in valid_urls:
            result = validator.validate_url(url)
            assert result == url

    def test_validate_url_invalid_scheme(self):
        """Test invalid URL scheme rejection."""
        validator = SecurityValidator.from_unified_config()
        invalid_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "ftp://example.com",
        ]

        for url in invalid_urls:
            with pytest.raises(SecurityError):
                validator.validate_url(url)

    def test_validate_url_dangerous_patterns(self):
        """Test dangerous pattern rejection."""
        validator = SecurityValidator.from_unified_config()
        dangerous_urls = [
            "http://localhost:8080",
            "https://127.0.0.1",
            "http://192.168.1.1",
            "http://10.0.0.1",
            "https://172.16.0.1",
        ]

        for url in dangerous_urls:
            with pytest.raises(SecurityError):
                validator.validate_url(url)

    def test_validate_url_too_long(self):
        """Test URL length limit."""
        validator = SecurityValidator.from_unified_config()
        long_url = "https://example.com/" + "a" * 2100

        with pytest.raises(SecurityError):
            validator.validate_url(long_url)

    def test_validate_collection_name_valid(self):
        """Test valid collection name validation."""
        validator = SecurityValidator.from_unified_config()
        valid_names = ["documentation", "my-docs", "test_collection", "docs123", "a"]

        for name in valid_names:
            result = validator.validate_collection_name(name)
            assert result == name

    def test_validate_collection_name_invalid(self):
        """Test invalid collection name rejection."""
        validator = SecurityValidator.from_unified_config()
        invalid_names = [
            "",
            "   ",
            "docs with spaces",
            "docs@special",
            "docs/slash",
            "a" * 70,  # Too long
        ]

        for name in invalid_names:
            with pytest.raises(SecurityError):
                validator.validate_collection_name(name)

    def test_validate_query_string_valid(self):
        """Test valid query string validation."""
        validator = SecurityValidator.from_unified_config()
        valid_queries = [
            "how to install python",
            "API documentation",
            "error handling best practices",
        ]

        for query in valid_queries:
            result = validator.validate_query_string(query)
            assert len(result) > 0
            # Should not contain dangerous characters
            assert "<" not in result
            assert ">" not in result

    def test_validate_query_string_sanitization(self):
        """Test query string sanitization."""
        validator = SecurityValidator.from_unified_config()
        dangerous_query = "search<script>alert('xss')</script>"
        result = validator.validate_query_string(dangerous_query)

        # Should remove dangerous characters
        assert "<script>" not in result
        assert "alert" in result  # Content should remain

    def test_validate_query_string_too_long(self):
        """Test query length limit."""
        validator = SecurityValidator.from_unified_config()
        long_query = "a" * 1100

        with pytest.raises(SecurityError):
            validator.validate_query_string(long_query)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        validator = SecurityValidator.from_unified_config()
        test_cases = [
            ("normal.txt", "normal.txt"),
            ("file with spaces.txt", "file with spaces.txt"),
            ("../../../etc/passwd", "passwd"),
            ('file<>:"/\\|?*.txt', "file_________.txt"),
            ("", "safe_filename"),
            ("a" * 300, "a" * 255),  # Length limit
        ]

        for input_name, _ in test_cases:
            result = validator.sanitize_filename(input_name)
            assert len(result) <= 255
            # Should not contain dangerous characters
            assert "/" not in result
            assert "\\" not in result


class TestAPIKeyValidator:
    """Test API key validation utilities."""

    def test_validate_openai_key_valid(self):
        """Test valid OpenAI API key format."""
        valid_keys = [
            "sk-" + "a" * 48,  # Standard format
            "sk-" + "x" * 50,  # Slightly longer
        ]

        for key in valid_keys:
            assert APIKeyValidator.validate_openai_key(key) is True

    def test_validate_openai_key_invalid(self):
        """Test invalid OpenAI API key format."""
        invalid_keys = [
            "",
            "invalid",
            "sk-short",
            "ak-" + "a" * 48,  # Wrong prefix
            None,
        ]

        for key in invalid_keys:
            assert APIKeyValidator.validate_openai_key(key) is False

    def test_mask_api_key(self):
        """Test API key masking for logging."""
        test_cases = [
            ("sk-1234567890abcdef1234567890abcdef1234567890abcdef", "sk-1********cdef"),
            ("short", "************"),
            ("", "************"),
            (None, "************"),
        ]

        for key, _ in test_cases:
            result = APIKeyValidator.mask_api_key(key)
            assert len(result) >= 12  # Minimum masked length
            if key and len(key) >= 8:
                assert result.startswith(key[:4])
                assert result.endswith(key[-4:])

    def test_validate_required_env_vars_missing(self, monkeypatch):
        """Test missing required environment variables."""
        # Remove the required env var
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(SecurityError, match="OPENAI_API_KEY"):
            APIKeyValidator.validate_required_env_vars()

    def test_validate_required_env_vars_invalid_format(self, monkeypatch):
        """Test invalid API key format."""
        monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")

        with pytest.raises(SecurityError, match="Invalid OPENAI_API_KEY format"):
            APIKeyValidator.validate_required_env_vars()

    def test_validate_required_env_vars_valid(self, monkeypatch):
        """Test valid environment variable validation."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "a" * 48)
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        result = APIKeyValidator.validate_required_env_vars()

        assert "OPENAI_API_KEY" in result
        assert "QDRANT_URL" in result
        assert result["OPENAI_API_KEY"] == "sk-" + "a" * 48
