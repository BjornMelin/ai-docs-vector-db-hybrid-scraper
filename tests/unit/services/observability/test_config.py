"""Tests for observability configuration."""

from unittest.mock import patch

import pytest

from src.config.core import get_config, reset_config
from src.services.observability.config import (
    ObservabilityConfig,
    get_observability_config,
    get_resource_attributes,
)


class TestObservabilityConfig:
    """Test observability configuration functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservabilityConfig()

        assert config.enabled is False
        assert config.service_name == "ai-docs-vector-db"
        assert config.service_version == "1.0.0"
        assert config.service_namespace == "ai-docs"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.otlp_insecure is True
        assert config.trace_sample_rate == 1.0
        assert config.track_ai_operations is True
        assert config.track_costs is True
        assert config.instrument_fastapi is True
        assert config.instrument_httpx is True
        assert config.instrument_redis is True
        assert config.instrument_sqlalchemy is True
        assert config.console_exporter is False

    def test_enabled_configuration(self):
        """Test enabled observability configuration."""
        config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            otlp_endpoint="http://test.example.com:4317",
            trace_sample_rate=0.5,
        )

        assert config.enabled is True
        assert config.service_name == "test-service"
        assert config.otlp_endpoint == "http://test.example.com:4317"
        assert config.trace_sample_rate == 0.5

    def test_validation(self):
        """Test configuration validation."""
        # Test valid trace sample rate
        config = ObservabilityConfig(trace_sample_rate=0.5)
        assert config.trace_sample_rate == 0.5

        # Test invalid trace sample rate (should be clamped or validated)
        with pytest.raises(ValueError):
            ObservabilityConfig(trace_sample_rate=1.5)

        with pytest.raises(ValueError):
            ObservabilityConfig(trace_sample_rate=-0.1)

    def test_get_observability_config_integration(self):
        """Test getting observability config from main config."""
        try:
            # Reset config to ensure clean state
            reset_config()

            # Get config through the main config system
            config = get_observability_config()

            # Should return default ObservabilityConfig
            assert isinstance(config, ObservabilityConfig)
            assert config.enabled is False
            # Service name should be converted from app name
            assert config.service_name == "ai-documentation-vector-db"

        finally:
            reset_config()

    def test_get_resource_attributes(self):
        """Test resource attribute generation."""
        config = ObservabilityConfig(
            service_name="test-service",
            service_version="2.0.0",
            service_namespace="test-namespace",
        )

        attributes = get_resource_attributes(config)

        assert attributes["service.name"] == "test-service"
        assert attributes["service.version"] == "2.0.0"
        assert attributes["service.namespace"] == "test-namespace"
        assert attributes["deployment.environment"] == "development"
        assert attributes["application.type"] == "ai-documentation-system"

    def test_get_resource_attributes_development(self):
        """Test resource attributes in development environment."""
        config = ObservabilityConfig(deployment_environment="development")

        attributes = get_resource_attributes(config)

        assert attributes["deployment.environment"] == "development"

    def test_batch_configuration_defaults(self):
        """Test batch processing configuration defaults."""
        config = ObservabilityConfig()

        # These should have reasonable defaults for batch processing
        assert hasattr(config, "batch_schedule_delay")
        assert hasattr(config, "batch_max_queue_size")
        assert hasattr(config, "batch_max_export_batch_size")
        assert hasattr(config, "batch_export_timeout")

        # Verify default values
        assert config.batch_schedule_delay == 5000
        assert config.batch_max_queue_size == 2048
        assert config.batch_max_export_batch_size == 512
        assert config.batch_export_timeout == 30000


class TestObservabilityConfigIntegration:
    """Test observability configuration integration with main config."""

    def test_config_integration(self):
        """Test that observability config is properly integrated."""
        try:
            reset_config()

            # Get main config
            main_config = get_config()

            # Should have observability config
            assert hasattr(main_config, "observability")
            # The main config has a built-in observability config from core.py
            # The type check is tricky since there are two ObservabilityConfig classes
            assert hasattr(main_config.observability, "enabled")
            assert hasattr(main_config.observability, "service_name")

            # Test that get_observability_config returns a config with service name derived from main config
            obs_config = get_observability_config()
            assert (
                obs_config.service_name == "ai-documentation-vector-db"
            )  # Converted from app name

        finally:
            reset_config()

    def test_environment_variable_override(self):
        """Test configuration override via environment variables."""
        # The current implementation doesn't support direct env var override
        # It integrates with the main config system instead
        with patch.dict(
            "os.environ",
            {
                "AI_DOCS_MONITORING__ENABLE_METRICS": "true",
                "AI_DOCS_APP_NAME": "Test Service From Env",
                "AI_DOCS_VERSION": "2.0.0",
            },
        ):
            try:
                reset_config()
                get_observability_config.cache_clear()

                config = get_observability_config()

                assert config.enabled is True  # From monitoring.enable_metrics
                assert (
                    config.service_name == "test-service-from-env"
                )  # Converted from app name
                assert config.service_version == "2.0.0"

            finally:
                reset_config()
                get_observability_config.cache_clear()

    def test_nested_configuration_access(self):
        """Test accessing nested configuration through main config."""
        try:
            reset_config()

            main_config = get_config()

            # Should be able to access observability config through main config
            assert main_config.observability.enabled is False
            assert main_config.observability.service_name == "ai-docs-vector-db"

            # Should get a config derived from main config values
            obs_config = get_observability_config()
            assert (
                obs_config.enabled is False
            )  # Should match monitoring.enable_metrics (false by default)
            # Service name should be derived from app name conversion
            assert obs_config.service_name == "ai-documentation-vector-db"

        finally:
            reset_config()
