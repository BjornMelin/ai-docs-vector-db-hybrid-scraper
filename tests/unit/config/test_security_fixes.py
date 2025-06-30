"""Tests for security configuration and settings.

Tests the security configuration system including validation,
rate limiting, and security controls.
"""

from src.config.settings import (
    SecurityConfig,
    Settings,
)


class TestSecurityConfig:
    """Test suite for security configuration."""

    def test_default_security_config(self):
        """Test default security configuration."""
        config = SecurityConfig()

        assert config.allowed_domains == ["*"]
        assert config.blocked_domains == []
        assert config.require_api_keys is True
        assert config.api_key_header == "X-API-Key"
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 100
        assert config.rate_limit_requests_per_minute == 60

    def test_query_validation_settings(self):
        """Test query validation settings."""
        config = SecurityConfig()

        assert config.max_query_length == 1000
        assert config.max_url_length == 2048

    def test_custom_security_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            allowed_domains=["example.com", "docs.example.com"],
            blocked_domains=["malicious.com"],
            require_api_keys=False,
            max_query_length=500,
            rate_limit_requests_per_minute=120,
        )

        assert config.allowed_domains == ["example.com", "docs.example.com"]
        assert config.blocked_domains == ["malicious.com"]
        assert config.require_api_keys is False
        assert config.max_query_length == 500
        assert config.rate_limit_requests_per_minute == 120

    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        config = SecurityConfig(
            enable_rate_limiting=True,
            rate_limit_requests=200,
            rate_limit_requests_per_minute=100,
        )

        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 200
        assert config.rate_limit_requests_per_minute == 100

    def test_api_key_configuration(self):
        """Test API key configuration."""
        config = SecurityConfig(require_api_keys=True, api_key_header="Authorization")

        assert config.require_api_keys is True
        assert config.api_key_header == "Authorization"

    def test_domain_restrictions(self):
        """Test domain restriction configuration."""
        config = SecurityConfig(
            allowed_domains=["trusted.com", "api.trusted.com"],
            blocked_domains=["blocked.com", "spam.com"],
        )

        assert len(config.allowed_domains) == 2
        assert "trusted.com" in config.allowed_domains
        assert "api.trusted.com" in config.allowed_domains

        assert len(config.blocked_domains) == 2
        assert "blocked.com" in config.blocked_domains
        assert "spam.com" in config.blocked_domains

    def test_validation_limits(self):
        """Test validation limits configuration."""
        config = SecurityConfig(max_query_length=2000, max_url_length=4096)

        assert config.max_query_length == 2000
        assert config.max_url_length == 4096


class TestSecurityIntegration:
    """Test security integration with main settings."""

    def test_security_in_settings(self):
        """Test security configuration in main settings."""
        settings = Settings()

        assert hasattr(settings, "security")
        assert isinstance(settings.security, SecurityConfig)
        assert settings.security.require_api_keys is True

    def test_security_configuration_override(self):
        """Test overriding security configuration in settings."""
        settings = Settings()

        # Update security settings
        settings.security.require_api_keys = False
        settings.security.max_query_length = 2000
        settings.security.allowed_domains = ["custom.com"]

        assert settings.security.require_api_keys is False
        assert settings.security.max_query_length == 2000
        assert settings.security.allowed_domains == ["custom.com"]

    def test_rate_limiting_integration(self):
        """Test rate limiting integration."""
        settings = Settings()

        # Rate limiting should be enabled by default
        assert settings.security.enable_rate_limiting is True
        assert settings.security.rate_limit_requests > 0
        assert settings.security.rate_limit_requests_per_minute > 0

    def test_security_validation_constraints(self):
        """Test security validation constraints."""
        # Valid configuration
        config = SecurityConfig(max_query_length=100, max_url_length=200)
        assert config.max_query_length == 100
        assert config.max_url_length == 200

    def test_security_defaults_in_different_environments(self):
        """Test security defaults in different environments."""
        from src.config.settings import Environment

        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        prod_settings = Settings(environment=Environment.PRODUCTION)

        # Security should be consistent across environments
        assert (
            dev_settings.security.require_api_keys
            == prod_settings.security.require_api_keys
        )
        assert (
            dev_settings.security.enable_rate_limiting
            == prod_settings.security.enable_rate_limiting
        )


class TestSecurityHelpers:
    """Test security helper functions and utilities."""

    def test_get_security_config(self):
        """Test getting security configuration."""
        from src.config.settings import get_security_config, reset_settings

        # Reset to ensure clean state
        reset_settings()

        security_config = get_security_config()
        assert isinstance(security_config, SecurityConfig)
        assert security_config.require_api_keys is True

    def test_security_config_immutability(self):
        """Test that security config maintains expected behavior."""
        config1 = SecurityConfig()
        config2 = SecurityConfig()

        # Should have same defaults
        assert config1.require_api_keys == config2.require_api_keys
        assert config1.max_query_length == config2.max_query_length

        # Should be independent instances
        config1.require_api_keys = False
        assert config2.require_api_keys is True  # Should not be affected

    def test_security_config_serialization(self):
        """Test security config serialization."""
        config = SecurityConfig(
            allowed_domains=["test.com"], require_api_keys=False, max_query_length=500
        )

        # Should be able to serialize to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["allowed_domains"] == ["test.com"]
        assert config_dict["require_api_keys"] is False
        assert config_dict["max_query_length"] == 500

    def test_security_policy_combinations(self):
        """Test different security policy combinations."""
        # Strict security policy
        strict_config = SecurityConfig(
            allowed_domains=["trusted.com"],
            require_api_keys=True,
            enable_rate_limiting=True,
            rate_limit_requests=50,
            max_query_length=500,
        )

        assert strict_config.allowed_domains == ["trusted.com"]
        assert strict_config.require_api_keys is True
        assert strict_config.rate_limit_requests == 50

        # Permissive security policy
        permissive_config = SecurityConfig(
            allowed_domains=["*"],
            require_api_keys=False,
            enable_rate_limiting=False,
            max_query_length=10000,
        )

        assert permissive_config.allowed_domains == ["*"]
        assert permissive_config.require_api_keys is False
        assert permissive_config.enable_rate_limiting is False
        assert permissive_config.max_query_length == 10000
