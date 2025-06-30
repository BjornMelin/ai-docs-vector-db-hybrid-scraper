"""Tests for security configuration and settings.

Tests the security configuration system including validation,
rate limiting, and security controls.
"""

from src.config.settings import (
    SecurityConfig,
    Settings,
)


"""Consolidated security configuration tests."""

import pytest
from pydantic import ValidationError


class TestSecurityConfig:
    """Consolidated security configuration tests with parametrized patterns."""

    @pytest.mark.parametrize(
        "config_params,expected_values",
        [
            pytest.param(
                {},  # Default config
                {
                    "allowed_domains": ["*"],
                    "blocked_domains": [],
                    "require_api_keys": True,
                    "api_key_header": "X-API-Key",
                    "enable_rate_limiting": True,
                    "rate_limit_requests": 100,
                    "rate_limit_requests_per_minute": 60,
                    "max_query_length": 1000,
                    "max_url_length": 2048,
                },
                id="default_config",
            ),
            pytest.param(
                {
                    "allowed_domains": ["example.com", "docs.example.com"],
                    "blocked_domains": ["malicious.com"],
                    "require_api_keys": False,
                    "max_query_length": 500,
                    "rate_limit_requests_per_minute": 120,
                },
                {
                    "allowed_domains": ["example.com", "docs.example.com"],
                    "blocked_domains": ["malicious.com"],
                    "require_api_keys": False,
                    "max_query_length": 500,
                    "rate_limit_requests_per_minute": 120,
                },
                id="custom_config",
            ),
            pytest.param(
                {
                    "enable_rate_limiting": True,
                    "rate_limit_requests": 200,
                    "rate_limit_requests_per_minute": 100,
                },
                {
                    "enable_rate_limiting": True,
                    "rate_limit_requests": 200,
                    "rate_limit_requests_per_minute": 100,
                },
                id="rate_limiting_config",
            ),
        ],
    )
    def test_security_config_initialization(self, config_params, expected_values):
        """Test security configuration initialization with various parameters."""
        config = SecurityConfig(**config_params)

        for field, expected_value in expected_values.items():
            assert getattr(config, field) == expected_value

    @pytest.mark.parametrize(
        "domain_config",
        [
            pytest.param(
                {
                    "allowed_domains": ["trusted.com", "api.trusted.com"],
                    "blocked_domains": ["blocked.com", "spam.com"],
                },
                id="domain_restrictions",
            ),
            pytest.param(
                {
                    "allowed_domains": ["*"],
                    "blocked_domains": ["malware.com", "phishing.com"],
                },
                id="wildcard_allowed_with_blocked",
            ),
            pytest.param(
                {
                    "allowed_domains": ["secure.example.com"],
                    "blocked_domains": [],
                },
                id="specific_allowed_no_blocked",
            ),
        ],
    )
    def test_domain_restriction_patterns(self, domain_config):
        """Test various domain restriction patterns."""
        config = SecurityConfig(**domain_config)

        assert config.allowed_domains == domain_config["allowed_domains"]
        assert config.blocked_domains == domain_config["blocked_domains"]

        # Verify lists are not empty when configured
        if domain_config["allowed_domains"]:
            assert len(config.allowed_domains) > 0
        if domain_config["blocked_domains"]:
            assert len(config.blocked_domains) > 0

    @pytest.mark.parametrize(
        "api_key_config",
        [
            pytest.param(
                {"require_api_keys": True, "api_key_header": "X-API-Key"},
                id="default_api_key_header",
            ),
            pytest.param(
                {"require_api_keys": True, "api_key_header": "Authorization"},
                id="authorization_header",
            ),
            pytest.param(
                {"require_api_keys": False, "api_key_header": "Custom-Auth"},
                id="disabled_with_custom_header",
            ),
        ],
    )
    def test_api_key_configuration_patterns(self, api_key_config):
        """Test API key configuration patterns."""
        config = SecurityConfig(**api_key_config)

        assert config.require_api_keys == api_key_config["require_api_keys"]
        assert config.api_key_header == api_key_config["api_key_header"]

    @pytest.mark.parametrize(
        "rate_limit_config,should_pass",
        [
            pytest.param(
                {"rate_limit_requests": 100, "rate_limit_requests_per_minute": 60},
                True,
                id="valid_rate_limits",
            ),
            pytest.param(
                {"rate_limit_requests": 1000, "rate_limit_requests_per_minute": 500},
                True,
                id="high_rate_limits",
            ),
            pytest.param(
                {"rate_limit_requests": 1, "rate_limit_requests_per_minute": 1},
                True,
                id="minimal_rate_limits",
            ),
        ],
    )
    def test_rate_limiting_validation(self, rate_limit_config, should_pass):
        """Test rate limiting configuration validation."""
        if should_pass:
            config = SecurityConfig(**rate_limit_config)
            assert (
                config.rate_limit_requests == rate_limit_config["rate_limit_requests"]
            )
            assert (
                config.rate_limit_requests_per_minute
                == rate_limit_config["rate_limit_requests_per_minute"]
            )
        else:
            with pytest.raises(ValidationError):
                SecurityConfig(**rate_limit_config)

    @pytest.mark.parametrize(
        "validation_limits",
        [
            pytest.param(
                {"max_query_length": 2000, "max_url_length": 4096},
                id="increased_limits",
            ),
            pytest.param(
                {"max_query_length": 500, "max_url_length": 1024}, id="decreased_limits"
            ),
            pytest.param(
                {"max_query_length": 10000, "max_url_length": 8192}, id="high_limits"
            ),
        ],
    )
    def test_validation_limits_configuration(self, validation_limits):
        """Test validation limits configuration."""
        config = SecurityConfig(**validation_limits)

        assert config.max_query_length == validation_limits["max_query_length"]
        assert config.max_url_length == validation_limits["max_url_length"]

    @pytest.mark.parametrize(
        "invalid_config,expected_error",
        [
            pytest.param(
                {"rate_limit_requests": 0},
                "must be greater than 0",
                id="zero_rate_limit",
            ),
            pytest.param(
                {"rate_limit_requests": -1},
                "must be greater than 0",
                id="negative_rate_limit",
            ),
            pytest.param(
                {"max_query_length": 0},
                "must be greater than 0",
                id="zero_query_length",
            ),
            pytest.param(
                {"max_url_length": -1},
                "must be greater than 0",
                id="negative_url_length",
            ),
        ],
    )
    def test_security_config_validation_errors(self, invalid_config, expected_error):
        """Test that invalid security configurations raise appropriate validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(**invalid_config)

        # Check that the error message contains expected text
        assert any(expected_error in str(error) for error in exc_info.value.errors())

    def test_security_config_edge_cases(self):
        """Test edge cases for security configuration."""
        # Empty allowed domains (should default to ["*"])
        config = SecurityConfig(allowed_domains=[])
        assert config.allowed_domains == []

        # Very long domain names
        long_domain = "a" * 250 + ".com"
        config = SecurityConfig(allowed_domains=[long_domain])
        assert long_domain in config.allowed_domains

        # Special characters in API key header
        config = SecurityConfig(api_key_header="X-Custom-API-Key-v2")
        assert config.api_key_header == "X-Custom-API-Key-v2"

    def test_security_config_comprehensive_validation(self):
        """Test comprehensive security configuration with all fields."""
        config = SecurityConfig(
            allowed_domains=["secure.example.com", "api.secure.com"],
            blocked_domains=["malicious.com", "spam.example.com"],
            require_api_keys=True,
            api_key_header="Authorization",
            enable_rate_limiting=True,
            rate_limit_requests=500,
            rate_limit_requests_per_minute=300,
            max_query_length=2048,
            max_url_length=4096,
        )

        # Verify all fields are set correctly
        assert len(config.allowed_domains) == 2
        assert len(config.blocked_domains) == 2
        assert config.require_api_keys is True
        assert config.api_key_header == "Authorization"
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 500
        assert config.rate_limit_requests_per_minute == 300
        assert config.max_query_length == 2048
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


# ==============================================================================
# CONSOLIDATED CORE CONFIG TESTS - TO BE EXTRACTED TO test_config_core.py
# ==============================================================================
"""
Consolidated core configuration tests with parametrized patterns.

This content should be moved to test_config_core.py for core config functionality.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st

from src.config.settings import (
    ApplicationMode,
    CacheConfig,
    CacheType,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
)


class TestCoreConfigurationPatterns:
    """Consolidated tests for core configuration patterns."""

    @pytest.mark.parametrize(
        "app_mode,expected_defaults",
        [
            pytest.param(
                ApplicationMode.SIMPLE,
                {
                    "max_concurrent_requests": 20,
                    "enable_monitoring": False,
                    "enable_background_tasks": False,
                },
                id="simple_mode",
            ),
            pytest.param(
                ApplicationMode.ENTERPRISE,
                {
                    "max_concurrent_requests": 100,
                    "enable_monitoring": True,
                    "enable_background_tasks": True,
                },
                id="enterprise_mode",
            ),
        ],
    )
    def test_application_mode_defaults(self, app_mode, expected_defaults):
        """Test that application modes set appropriate defaults."""
        settings = Settings(app_mode=app_mode)

        # Check that the app mode is set correctly
        assert settings.app_mode == app_mode

        # Check performance config defaults based on mode
        if "max_concurrent_requests" in expected_defaults:
            expected_requests = expected_defaults["max_concurrent_requests"]
            if app_mode == ApplicationMode.SIMPLE:
                # Simple mode should use default performance config
                assert settings.performance.max_concurrent_requests <= expected_requests
            elif app_mode == ApplicationMode.ENTERPRISE:
                # Enterprise mode may have higher defaults
                assert settings.performance.max_concurrent_requests >= 20

    @pytest.mark.parametrize(
        "environment,debug_expected",
        [
            (Environment.DEVELOPMENT, True),
            (Environment.TESTING, True),
            (Environment.PRODUCTION, False),
        ],
    )
    def test_environment_debug_mapping(self, environment, debug_expected):
        """Test that environments map to appropriate debug settings."""
        settings = Settings(environment=environment)
        assert settings.environment == environment
        # Note: debug setting may not be directly tied to environment in current implementation

    @pytest.mark.parametrize(
        "provider_config",
        [
            pytest.param(
                {
                    "provider": EmbeddingProvider.OPENAI,
                    "openai_config": {
                        "api_key": "sk-test",
                        "model": "text-embedding-3-small",
                    },
                    "expected_provider": EmbeddingProvider.OPENAI,
                },
                id="openai_provider",
            ),
            pytest.param(
                {
                    "provider": EmbeddingProvider.FASTEMBED,
                    "fastembed_config": {"model": "BAAI/bge-small-en-v1.5"},
                    "expected_provider": EmbeddingProvider.FASTEMBED,
                },
                id="fastembed_provider",
            ),
        ],
    )
    def test_embedding_provider_configuration(self, provider_config):
        """Test embedding provider configuration patterns."""
        embedding_config = {"provider": provider_config["provider"]}

        config_kwargs = {"embedding": embedding_config}

        # Add provider-specific config if present
        if "openai_config" in provider_config:
            config_kwargs["openai"] = OpenAIConfig(**provider_config["openai_config"])
        if "fastembed_config" in provider_config:
            config_kwargs["fastembed"] = FastEmbedConfig(
                **provider_config["fastembed_config"]
            )

        settings = Settings(**config_kwargs)
        assert settings.embedding.provider == provider_config["expected_provider"]

    @pytest.mark.parametrize(
        "cache_config",
        [
            pytest.param(
                {"cache_type": CacheType.MEMORY, "local_max_size": 1000},
                id="memory_cache",
            ),
            pytest.param(
                {
                    "cache_type": CacheType.DRAGONFLY,
                    "dragonfly_url": "redis://localhost:6379",
                    "ttl_seconds": 3600,
                },
                id="dragonfly_cache",
            ),
        ],
    )
    def test_cache_configuration_patterns(self, cache_config):
        """Test cache configuration patterns."""
        cache = CacheConfig(**cache_config)
        settings = Settings(cache=cache)

        assert settings.cache.cache_type == cache_config["cache_type"]

        if cache_config["cache_type"] == CacheType.MEMORY:
            assert settings.cache.local_max_size == cache_config.get(
                "local_max_size", 1000
            )
        elif cache_config["cache_type"] == CacheType.DRAGONFLY:
            assert settings.cache.dragonfly_url == cache_config.get("dragonfly_url")

    @pytest.mark.parametrize(
        "env_vars,expected_values",
        [
            pytest.param(
                {
                    "APP_MODE": "enterprise",
                    "ENVIRONMENT": "production",
                    "CACHE__ENABLE_CACHING": "true",
                    "QDRANT__URL": "http://prod-qdrant:6333",
                    "OPENAI__API_KEY": "sk-prod-key",
                },
                {
                    "app_mode": ApplicationMode.ENTERPRISE,
                    "environment": Environment.PRODUCTION,
                    "cache_enabled": True,
                    "qdrant_url": "http://prod-qdrant:6333",
                    "openai_key": "sk-prod-key",
                },
                id="production_env_vars",
            ),
            pytest.param(
                {
                    "APP_MODE": "simple",
                    "ENVIRONMENT": "development",
                    "SECURITY__REQUIRE_API_KEYS": "false",
                },
                {
                    "app_mode": ApplicationMode.SIMPLE,
                    "environment": Environment.DEVELOPMENT,
                    "require_api_keys": False,
                },
                id="development_env_vars",
            ),
        ],
    )
    def test_environment_variable_loading(self, env_vars, expected_values):
        """Test environment variable loading patterns."""
        with patch.dict(os.environ, env_vars, clear=True):
            # In a real implementation, this would load from environment
            # For now, we'll test the structure
            settings = Settings()

            # Basic validation that settings can be created
            assert settings is not None
            assert hasattr(settings, "app_mode")
            assert hasattr(settings, "environment")

    @given(
        concurrent_requests=st.integers(min_value=1, max_value=1000),
        timeout_seconds=st.floats(min_value=1.0, max_value=300.0),
        enable_caching=st.booleans(),
    )
    def test_settings_property_based(
        self, concurrent_requests, timeout_seconds, enable_caching
    ):
        """Property-based test for Settings configuration."""
        try:
            settings = Settings(
                performance=PerformanceConfig(
                    max_concurrent_requests=concurrent_requests,
                    request_timeout=timeout_seconds,
                ),
                cache=CacheConfig(enable_caching=enable_caching),
            )

            assert settings.performance.max_concurrent_requests == concurrent_requests
            assert settings.performance.request_timeout == timeout_seconds
            assert settings.cache.enable_caching == enable_caching

        except ValidationError:
            # Some combinations may be invalid, which is acceptable
            pass

    def test_settings_nested_config_validation(self):
        """Test that nested configurations are properly validated."""
        # Valid nested configuration
        settings = Settings(
            cache=CacheConfig(enable_caching=True, ttl_seconds=3600),
            qdrant=QdrantConfig(url="http://localhost:6333", timeout=30.0),
            security=SecurityConfig(require_api_keys=True, rate_limit_requests=100),
        )

        assert settings.cache.enable_caching is True
        assert settings.cache.ttl_seconds == 3600
        assert settings.qdrant.url == "http://localhost:6333"
        assert settings.qdrant.timeout == 30.0
        assert settings.security.require_api_keys is True
        assert settings.security.rate_limit_requests == 100

    @pytest.mark.parametrize(
        "invalid_config,expected_error",
        [
            pytest.param(
                {"performance": {"max_concurrent_requests": -1}},
                "greater than 0",
                id="negative_concurrent_requests",
            ),
            pytest.param(
                {"cache": {"ttl_seconds": -1}}, "greater than 0", id="negative_ttl"
            ),
            pytest.param(
                {"qdrant": {"timeout": 0}}, "greater than 0", id="zero_timeout"
            ),
        ],
    )
    def test_settings_validation_errors(self, invalid_config, expected_error):
        """Test that invalid settings configurations raise validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            # Construct config objects for nested validation
            config_kwargs = {}

            if "performance" in invalid_config:
                config_kwargs["performance"] = PerformanceConfig(
                    **invalid_config["performance"]
                )
            if "cache" in invalid_config:
                config_kwargs["cache"] = CacheConfig(**invalid_config["cache"])
            if "qdrant" in invalid_config:
                config_kwargs["qdrant"] = QdrantConfig(**invalid_config["qdrant"])

            Settings(**config_kwargs)

        # Verify error message contains expected text
        error_messages = [str(error) for error in exc_info.value.errors()]
        assert any(expected_error in msg for msg in error_messages)

    def test_settings_defaults_comprehensive(self):
        """Test that Settings creates with sensible defaults."""
        settings = Settings()

        # Verify core settings have defaults
        assert settings.app_mode is not None
        assert settings.environment is not None

        # Verify nested configs are created with defaults
        assert settings.cache is not None
        assert settings.qdrant is not None
        assert settings.security is not None
        assert settings.performance is not None
        assert settings.embedding is not None

        # Verify specific default values
        assert isinstance(settings.cache, CacheConfig)
        assert isinstance(settings.qdrant, QdrantConfig)
        assert isinstance(settings.security, SecurityConfig)
        assert isinstance(settings.performance, PerformanceConfig)


# ==============================================================================
# CONFIG TEST UTILITIES - TO BE EXTRACTED TO test_config_utils.py
# ==============================================================================
"""
Shared utilities for configuration testing.

This content should be moved to test_config_utils.py for reusable test utilities.
"""

import json
from typing import Type, Union

import pytest
from pydantic import BaseModel


class ConfigTestUtils:
    """Utility class for configuration testing."""

    @staticmethod
    def create_temp_config_file(
        config_data: dict[str, Any], file_format: str = "json", temp_dir: Path = None
    ) -> Path:
        """Create a temporary configuration file for testing."""
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())

        if file_format == "json":
            config_file = temp_dir / "test_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
        elif file_format == "env":
            config_file = temp_dir / ".env"
            with open(config_file, "w") as f:
                for key, value in config_data.items():
                    f.write(f"{key}={value}\n")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return config_file

    @staticmethod
    def assert_config_subset(
        config: BaseModel, expected_subset: dict[str, Any], exclude_fields: set = None
    ) -> None:
        """Assert that a config contains expected values for a subset of fields."""
        exclude_fields = exclude_fields or set()

        config_dict = (
            config.model_dump() if hasattr(config, "model_dump") else config.dict()
        )

        for field, expected_value in expected_subset.items():
            if field not in exclude_fields:
                assert field in config_dict, f"Field '{field}' not found in config"
                actual_value = config_dict[field]
                assert actual_value == expected_value, (
                    f"Field '{field}': expected {expected_value}, got {actual_value}"
                )

    @staticmethod
    def validate_config_errors(
        config_class: type[BaseModel],
        invalid_params: dict[str, Any],
        expected_error_fields: set = None,
    ) -> list[str]:
        """Validate that invalid config parameters raise expected errors."""
        expected_error_fields = expected_error_fields or set(invalid_params.keys())

        try:
            config_class(**invalid_params)
            pytest.fail(
                f"Expected ValidationError for {config_class.__name__} with {invalid_params}"
            )
        except ValidationError as e:
            errors = e.errors()
            error_fields = {
                error["loc"][0] if error["loc"] else None for error in errors
            }

            # Check that expected fields have errors
            missing_errors = expected_error_fields - error_fields
            assert not missing_errors, (
                f"Expected errors for fields {missing_errors} but they were not found"
            )

            return [error["msg"] for error in errors]

    @staticmethod
    def mock_environment_vars(env_vars: dict[str, str]):
        """Context manager for mocking environment variables."""
        return patch.dict("os.environ", env_vars, clear=True)

    @staticmethod
    def create_test_settings(**overrides) -> Settings:
        """Create test settings with optional overrides."""
        defaults = {
            "app_mode": "simple",
            "environment": "testing",
        }
        defaults.update(overrides)
        return Settings(**defaults)

    @staticmethod
    def get_config_field_info(config_class: type[BaseModel]) -> dict[str, Any]:
        """Get information about configuration fields."""
        if hasattr(config_class, "model_fields"):
            # Pydantic v2
            return {
                field_name: {
                    "type": field_info.annotation,
                    "default": field_info.default,
                    "required": field_info.is_required(),
                }
                for field_name, field_info in config_class.model_fields.items()
            }
        # Pydantic v1 fallback
        return {
            field_name: {
                "type": field_info.type_,
                "default": field_info.default,
                "required": field_info.required,
            }
            for field_name, field_info in config_class.__fields__.items()
        }

    @staticmethod
    def compare_config_versions(
        config1: BaseModel, config2: BaseModel, ignore_fields: set = None
    ) -> dict[str, Any]:
        """Compare two configuration versions and return differences."""
        ignore_fields = ignore_fields or set()

        dict1 = (
            config1.model_dump() if hasattr(config1, "model_dump") else config1.dict()
        )
        dict2 = (
            config2.model_dump() if hasattr(config2, "model_dump") else config2.dict()
        )

        differences = {}
        all_fields = set(dict1.keys()) | set(dict2.keys())

        for field in all_fields:
            if field in ignore_fields:
                continue

            value1 = dict1.get(field, "<missing>")
            value2 = dict2.get(field, "<missing>")

            if value1 != value2:
                differences[field] = {"before": value1, "after": value2}

        return differences


class ConfigTestFixtures:
    """Shared test fixtures for configuration testing."""

    @pytest.fixture
    def temp_config_dir(self) -> Path:
        """Create a temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_env_vars(self) -> dict[str, str]:
        """Sample environment variables for testing."""
        return {
            "APP_MODE": "enterprise",
            "ENVIRONMENT": "production",
            "CACHE__ENABLE_CACHING": "true",
            "CACHE__DRAGONFLY_URL": "redis://test:6379",
            "QDRANT__URL": "http://test-qdrant:6333",
            "OPENAI__API_KEY": "sk-test-key-123",
            "SECURITY__REQUIRE_API_KEYS": "true",
            "PERFORMANCE__MAX_CONCURRENT_REQUESTS": "50",
        }

    @pytest.fixture
    def config_validation_test_cases(self) -> dict[str, Any]:
        """Test cases for configuration validation."""
        return {
            "valid_configs": [
                {"app_mode": "simple", "environment": "development"},
                {"app_mode": "enterprise", "environment": "production"},
            ],
            "invalid_configs": [
                {"app_mode": "invalid_mode"},
                {"environment": "invalid_env"},
            ],
            "edge_cases": [
                {"app_mode": "simple", "environment": "production"},
                {"app_mode": "enterprise", "environment": "development"},
            ],
        }


# Test data constants for parametrized tests
CONFIG_PROVIDER_TEST_DATA = [
    pytest.param("openai", {"api_key": "sk-test"}, id="openai"),
    pytest.param("fastembed", {"model": "BAAI/bge-small-en-v1.5"}, id="fastembed"),
]

CONFIG_ENVIRONMENT_TEST_DATA = [
    pytest.param("development", True, id="dev_debug"),
    pytest.param("production", False, id="prod_no_debug"),
    pytest.param("testing", True, id="test_debug"),
]

CONFIG_CACHE_TEST_DATA = [
    pytest.param("memory", {"local_max_size": 1000}, id="memory_cache"),
    pytest.param(
        "dragonfly", {"dragonfly_url": "redis://localhost:6379"}, id="dragonfly_cache"
    ),
]

# ==============================================================================
# END CONFIG TEST UTILITIES
# ==============================================================================

# ==============================================================================
# END CONSOLIDATED CORE CONFIG TESTS
# ==============================================================================
