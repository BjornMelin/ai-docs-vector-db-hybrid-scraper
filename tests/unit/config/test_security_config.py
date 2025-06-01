"""Test SecurityConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import SecurityConfig


class TestSecurityConfig:
    """Test SecurityConfig model validation and behavior."""

    def test_default_values(self):
        """Test SecurityConfig with default values."""
        config = SecurityConfig()

        # Domain lists
        assert config.allowed_domains == []
        assert config.blocked_domains == []

        # API security
        assert config.require_api_keys is True
        assert config.api_key_header == "X-API-Key"

        # Rate limiting
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 100

    def test_custom_values(self):
        """Test SecurityConfig with custom values."""
        config = SecurityConfig(
            allowed_domains=["example.com", "docs.example.com"],
            blocked_domains=["malicious.com", "spam.net"],
            require_api_keys=False,
            api_key_header="Authorization",
            enable_rate_limiting=False,
            rate_limit_requests=200,
        )

        assert config.allowed_domains == ["example.com", "docs.example.com"]
        assert config.blocked_domains == ["malicious.com", "spam.net"]
        assert config.require_api_keys is False
        assert config.api_key_header == "Authorization"
        assert config.enable_rate_limiting is False
        assert config.rate_limit_requests == 200

    def test_domain_lists(self):
        """Test domain list fields."""
        # Empty lists are valid
        config1 = SecurityConfig(allowed_domains=[], blocked_domains=[])
        assert config1.allowed_domains == []
        assert config1.blocked_domains == []

        # Single domain
        config2 = SecurityConfig(
            allowed_domains=["example.com"], blocked_domains=["bad.com"]
        )
        assert len(config2.allowed_domains) == 1
        assert len(config2.blocked_domains) == 1

        # Multiple domains
        config3 = SecurityConfig(
            allowed_domains=[
                "docs.python.org",
                "docs.djangoproject.com",
                "fastapi.tiangolo.com",
            ],
            blocked_domains=["spam1.com", "spam2.com", "malware.net"],
        )
        assert len(config3.allowed_domains) == 3
        assert len(config3.blocked_domains) == 3

        # Wildcard domains
        config4 = SecurityConfig(
            allowed_domains=["*.example.com", "docs.*", "*.github.io"],
            blocked_domains=["*.spam.com", "phishing.*"],
        )
        assert "*.example.com" in config4.allowed_domains
        assert "*.spam.com" in config4.blocked_domains

    def test_api_key_header(self):
        """Test API key header field."""
        # Standard headers
        headers = ["X-API-Key", "Authorization", "API-Key", "X-Auth-Token"]

        for header in headers:
            config = SecurityConfig(api_key_header=header)
            assert config.api_key_header == header

        # Custom header
        config = SecurityConfig(api_key_header="X-Custom-Auth")
        assert config.api_key_header == "X-Custom-Auth"

    def test_rate_limit_requests_constraint(self):
        """Test rate_limit_requests must be positive."""
        # Valid values
        config1 = SecurityConfig(rate_limit_requests=1)
        assert config1.rate_limit_requests == 1

        config2 = SecurityConfig(rate_limit_requests=10000)
        assert config2.rate_limit_requests == 10000

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(rate_limit_requests=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("rate_limit_requests",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: negative
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(rate_limit_requests=-100)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("rate_limit_requests",)

    def test_boolean_fields(self):
        """Test boolean field validation."""
        # All true
        config1 = SecurityConfig(require_api_keys=True, enable_rate_limiting=True)
        assert config1.require_api_keys is True
        assert config1.enable_rate_limiting is True

        # All false
        config2 = SecurityConfig(require_api_keys=False, enable_rate_limiting=False)
        assert config2.require_api_keys is False
        assert config2.enable_rate_limiting is False

        # Mixed
        config3 = SecurityConfig(require_api_keys=True, enable_rate_limiting=False)
        assert config3.require_api_keys is True
        assert config3.enable_rate_limiting is False

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(allowed_domains=["example.com"], unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = SecurityConfig(
            allowed_domains=["api.example.com", "docs.example.com"],
            blocked_domains=["malware.com"],
            require_api_keys=False,
            api_key_header="Bearer-Token",
            enable_rate_limiting=True,
            rate_limit_requests=500,
        )

        # Test model_dump
        data = config.model_dump()
        assert data["allowed_domains"] == ["api.example.com", "docs.example.com"]
        assert data["blocked_domains"] == ["malware.com"]
        assert data["require_api_keys"] is False
        assert data["api_key_header"] == "Bearer-Token"
        assert data["enable_rate_limiting"] is True
        assert data["rate_limit_requests"] == 500

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"allowed_domains":["api.example.com","docs.example.com"]' in json_str
        assert '"require_api_keys":false' in json_str
        assert '"api_key_header":"Bearer-Token"' in json_str
        assert '"rate_limit_requests":500' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = SecurityConfig(
            allowed_domains=["example.com"], rate_limit_requests=100
        )

        updated = original.model_copy(
            update={
                "allowed_domains": ["example.com", "api.example.com"],
                "blocked_domains": ["spam.com"],
                "rate_limit_requests": 200,
            }
        )

        assert original.allowed_domains == ["example.com"]
        assert original.blocked_domains == []
        assert original.rate_limit_requests == 100
        assert updated.allowed_domains == ["example.com", "api.example.com"]
        assert updated.blocked_domains == ["spam.com"]
        assert updated.rate_limit_requests == 200

    def test_type_validation(self):
        """Test type validation for fields."""
        # Test list field with wrong type
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(allowed_domains="example.com")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("allowed_domains",)

        # Test boolean field with wrong type
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(
                require_api_keys={"value": True}
            )  # Dict can't coerce to bool

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("require_api_keys",)

        # Test int field with wrong type (string that can't convert)
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(rate_limit_requests="one hundred")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("rate_limit_requests",)

    def test_security_profiles(self):
        """Test different security profile configurations."""
        # Open profile (for development)
        open_profile = SecurityConfig(
            allowed_domains=[],  # Allow all
            blocked_domains=[],  # Block none
            require_api_keys=False,
            enable_rate_limiting=False,
        )
        assert open_profile.require_api_keys is False
        assert open_profile.enable_rate_limiting is False

        # Strict profile (for production)
        strict_profile = SecurityConfig(
            allowed_domains=["api.company.com", "docs.company.com"],
            blocked_domains=["localhost", "127.0.0.1", "*.local"],
            require_api_keys=True,
            api_key_header="X-Secure-Token",
            enable_rate_limiting=True,
            rate_limit_requests=50,
        )
        assert len(strict_profile.allowed_domains) == 2
        assert "localhost" in strict_profile.blocked_domains
        assert strict_profile.rate_limit_requests == 50

        # API-only profile
        api_profile = SecurityConfig(
            allowed_domains=["api.service.com"],
            blocked_domains=["*"],  # Block everything else
            require_api_keys=True,
            enable_rate_limiting=True,
            rate_limit_requests=1000,  # Higher limit for APIs
        )
        assert api_profile.allowed_domains == ["api.service.com"]
        assert api_profile.rate_limit_requests == 1000

    def test_domain_list_duplicates(self):
        """Test handling of duplicate domains in lists."""
        # Lists can contain duplicates (no automatic deduplication)
        config = SecurityConfig(
            allowed_domains=["example.com", "example.com", "api.example.com"],
            blocked_domains=["spam.com", "spam.com"],
        )

        assert len(config.allowed_domains) == 3
        assert config.allowed_domains.count("example.com") == 2
        assert len(config.blocked_domains) == 2
        assert config.blocked_domains.count("spam.com") == 2
