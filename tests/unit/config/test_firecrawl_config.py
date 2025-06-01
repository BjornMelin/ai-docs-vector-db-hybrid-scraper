"""Test FirecrawlConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import FirecrawlConfig


class TestFirecrawlConfig:
    """Test FirecrawlConfig model validation and behavior."""

    def test_default_values(self):
        """Test FirecrawlConfig with default values."""
        config = FirecrawlConfig()

        assert config.api_key is None
        assert config.api_url == "https://api.firecrawl.dev"
        assert config.timeout == 30.0

    def test_custom_values(self):
        """Test FirecrawlConfig with custom values."""
        config = FirecrawlConfig(
            api_key="fc-test123456",
            api_url="https://custom.firecrawl.api",
            timeout=60.0,
        )

        assert config.api_key == "fc-test123456"
        assert config.api_url == "https://custom.firecrawl.api"
        assert config.timeout == 60.0

    def test_api_key_validation_valid(self):
        """Test valid API key formats."""
        # Valid API key with fc- prefix
        config1 = FirecrawlConfig(api_key="fc-abcdef123456")
        assert config1.api_key == "fc-abcdef123456"

        # Valid with underscores and hyphens
        config2 = FirecrawlConfig(api_key="fc-test_key-123")
        assert config2.api_key == "fc-test_key-123"

        # None is valid (optional)
        config3 = FirecrawlConfig(api_key=None)
        assert config3.api_key is None

        # Empty string becomes None
        config4 = FirecrawlConfig(api_key="")
        assert config4.api_key is None

        # Whitespace-only becomes None
        config5 = FirecrawlConfig(api_key="   ")
        assert config5.api_key is None

    def test_api_key_validation_invalid_prefix(self):
        """Test API key with invalid prefix."""
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="sk-1234567890")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "must start with 'fc-'" in str(errors[0]["msg"])

    def test_api_key_validation_too_short(self):
        """Test API key that's too short."""
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-abc")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "too short" in str(errors[0]["msg"])

    def test_api_key_validation_too_long(self):
        """Test API key that's too long."""
        long_key = "fc-" + "a" * 200
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key=long_key)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "too long" in str(errors[0]["msg"])

    def test_api_key_validation_invalid_characters(self):
        """Test API key with invalid characters."""
        # Special characters not allowed (except underscore and hyphen)
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-test@key#123")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "invalid characters" in str(errors[0]["msg"])

        # Non-ASCII characters
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-test™key123")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "non-ASCII characters" in str(errors[0]["msg"])

    def test_timeout_constraints(self):
        """Test timeout must be positive."""
        # Valid timeout
        config = FirecrawlConfig(timeout=120.0)
        assert config.timeout == 120.0

        # Invalid: zero timeout
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(timeout=0.0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: negative timeout
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(timeout=-10.0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout",)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-test123", unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = FirecrawlConfig(
            api_key="fc-myapikey123", api_url="https://api.custom.com", timeout=45.0
        )

        # Test model_dump
        data = config.model_dump()
        assert data["api_key"] == "fc-myapikey123"
        assert data["api_url"] == "https://api.custom.com"
        assert data["timeout"] == 45.0

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"api_key":"fc-myapikey123"' in json_str
        assert '"api_url":"https://api.custom.com"' in json_str
        assert '"timeout":45.0' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = FirecrawlConfig(api_key="fc-original123", timeout=30.0)

        updated = original.model_copy(
            update={"api_key": "fc-updated456", "timeout": 60.0}
        )

        assert original.api_key == "fc-original123"
        assert original.timeout == 30.0
        assert updated.api_key == "fc-updated456"
        assert updated.timeout == 60.0

    def test_type_validation(self):
        """Test type validation for fields."""
        # Test string field
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_url=123)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_url",)

        # Test float field
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(timeout="30 seconds")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout",)
