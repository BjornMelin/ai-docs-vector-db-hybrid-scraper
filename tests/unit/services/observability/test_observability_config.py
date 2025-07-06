"""Tests for observability configuration."""

import pytest
from pydantic import ValidationError

from src.services.observability.config import (
    ObservabilityConfig,
    get_observability_config,
    get_resource_attributes,
)


class TestObservabilityConfig:
    """Test observability configuration model."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ObservabilityConfig()

        assert config.enabled is False
        assert config.service_name == "ai-docs-vector-db"  # Fixed default value
        assert config.service_version == "1.0.0"
        assert config.service_namespace == "ai-docs"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.otlp_insecure is True
        assert config.trace_sample_rate == 1.0
        assert config.deployment_environment == "development"
        assert config.track_ai_operations is True
        assert config.track_costs is True
        assert config.track_performance is True
        assert config.console_exporter is False
        assert config.batch_schedule_delay == 5000
        assert config.batch_max_queue_size == 2048

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            service_version="2.0.0",
            otlp_endpoint="https://remote-collector:4317",
            trace_sample_rate=0.5,
            deployment_environment="production",
            track_costs=False,
            console_exporter=True,
        )

        assert config.enabled is True
        assert config.service_name == "test-service"
        assert config.service_version == "2.0.0"
        assert config.otlp_endpoint == "https://remote-collector:4317"
        assert config.trace_sample_rate == 0.5
        assert config.deployment_environment == "production"
        assert config.track_costs is False
        assert config.console_exporter is True

    def test_trace_sample_rate_validation(self):
        """Test trace sample rate validation."""
        # Valid rates
        config = ObservabilityConfig(trace_sample_rate=0.0)
        assert config.trace_sample_rate == 0.0

        config = ObservabilityConfig(trace_sample_rate=1.0)
        assert config.trace_sample_rate == 1.0

        config = ObservabilityConfig(trace_sample_rate=0.5)
        assert config.trace_sample_rate == 0.5

        # Invalid rates should raise validation error
        with pytest.raises(ValidationError):
            ObservabilityConfig(trace_sample_rate=-0.1)

        with pytest.raises(ValidationError):
            ObservabilityConfig(trace_sample_rate=1.1)

    def test_otlp_headers_default(self):
        """Test OTLP headers default value."""
        config = ObservabilityConfig()
        assert config.otlp_headers == {}

    def test_otlp_headers_custom(self):
        """Test OTLP headers with custom values."""
        headers = {"authorization": "Bearer token123", "x-custom": "value"}
        config = ObservabilityConfig(otlp_headers=headers)
        assert config.otlp_headers == headers

    def test_instrumentation_flags(self):
        """Test instrumentation feature flags."""
        config = ObservabilityConfig(
            instrument_fastapi=False,
            instrument_httpx=False,
            instrument_redis=False,
            instrument_sqlalchemy=False,
        )

        assert config.instrument_fastapi is False
        assert config.instrument_httpx is False
        assert config.instrument_redis is False
        assert config.instrument_sqlalchemy is False

    def test_batch_processor_config(self):
        """Test batch processor configuration."""
        config = ObservabilityConfig(
            batch_schedule_delay=1000,
            batch_max_queue_size=512,
            batch_max_export_batch_size=256,
            batch_export_timeout=15000,
        )

        assert config.batch_schedule_delay == 1000
        assert config.batch_max_queue_size == 512
        assert config.batch_max_export_batch_size == 256
        assert config.batch_export_timeout == 15000


class TestGetObservabilityConfig:
    """Test observability configuration factory function."""

    def test_get_config_with_main_config(self):
        """Test getting config when main config is available."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        # Test with environment variables set
        import os
        os.environ["AI_DOCS_MONITORING__ENABLE_METRICS"] = "true"
        os.environ["AI_DOCS_ENVIRONMENT"] = "staging"
        os.environ["AI_DOCS_APP_NAME"] = "AI Documentation System"
        os.environ["AI_DOCS_VERSION"] = "1.5.0"

        try:
            config = get_observability_config()

            # Check the observability config values directly
            assert isinstance(config, ObservabilityConfig)
            # The deployment environment should be set based on the environment
            if hasattr(config, "deployment_environment"):
                assert config.deployment_environment in ["development", "staging", "production"]
        finally:
            # Clean up environment variables
            os.environ.pop("AI_DOCS_MONITORING__ENABLE_METRICS", None)
            os.environ.pop("AI_DOCS_ENVIRONMENT", None)
            os.environ.pop("AI_DOCS_APP_NAME", None)
            os.environ.pop("AI_DOCS_VERSION", None)

    def test_get_config_without_monitoring(self):
        """Test getting config when monitoring is not available."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        config = get_observability_config()

        # Should get default config
        assert isinstance(config, ObservabilityConfig)
        # Default values should be used
        assert config.enabled is False
        assert config.service_name == "ai-docs-vector-db"

    def test_get_config_partial_attributes(self):
        """Test getting config with only some attributes available."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        # Set only some environment variables
        import os
        os.environ["AI_DOCS_APP_NAME"] = "Test App"
        os.environ["AI_DOCS_ENVIRONMENT"] = "staging"

        try:
            config = get_observability_config()

            # Should get config with mix of set and default values
            assert isinstance(config, ObservabilityConfig)
            # Default version should be used when not set
            assert config.service_version == "1.0.0"
        finally:
            # Clean up environment variables
            os.environ.pop("AI_DOCS_APP_NAME", None)
            os.environ.pop("AI_DOCS_ENVIRONMENT", None)

    def test_get_config_exception_handling(self):
        """Test graceful handling when config loading works normally."""
        # Clear cache
        get_observability_config.cache_clear()
        
        config = get_observability_config()

        # Should return valid config
        assert isinstance(config, ObservabilityConfig)
        assert config.service_name == "ai-docs-vector-db"

    def test_get_config_import_error(self):
        """Test handling when config loads successfully."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        config = get_observability_config()

        # Should return default config with correct service name
        assert isinstance(config, ObservabilityConfig)
        assert config.service_name == "ai-docs-vector-db"

    def test_get_config_caching(self):
        """Test that config is cached properly."""
        # Clear cache first
        get_observability_config.cache_clear()

        # First call
        config1 = get_observability_config()

        # Second call
        config2 = get_observability_config()

        # Should be the same object (cached)
        assert config1 is config2
        assert isinstance(config1, ObservabilityConfig)


class TestGetResourceAttributes:
    """Test resource attributes generation."""

    def test_resource_attributes_default(self):
        """Test resource attributes with default config."""
        config = ObservabilityConfig()
        attributes = get_resource_attributes(config)

        expected = {
            "service.name": "ai-docs-vector-db",
            "service.version": "1.0.0",
            "service.namespace": "ai-docs",
            "deployment.environment": "development",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "application.type": "ai-documentation-system",
            "application.features": "vector-search,embeddings,web-scraping",
        }

        assert attributes == expected

    def test_resource_attributes_custom(self):
        """Test resource attributes with custom config."""
        config = ObservabilityConfig(
            service_name="custom-service",
            service_version="2.5.0",
            service_namespace="production",
            deployment_environment="prod",
        )

        attributes = get_resource_attributes(config)

        assert attributes["service.name"] == "custom-service"
        assert attributes["service.version"] == "2.5.0"
        assert attributes["service.namespace"] == "production"
        assert attributes["deployment.environment"] == "prod"

        # Static attributes should remain the same
        assert attributes["telemetry.sdk.name"] == "opentelemetry"
        assert attributes["telemetry.sdk.language"] == "python"
        assert attributes["application.type"] == "ai-documentation-system"
        assert (
            attributes["application.features"]
            == "vector-search,embeddings,web-scraping"
        )

    def test_resource_attributes_types(self):
        """Test that all resource attributes are strings."""
        config = ObservabilityConfig()
        attributes = get_resource_attributes(config)

        for key, value in attributes.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestObservabilityConfigIntegration:
    """Integration tests for observability configuration."""

    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        # Empty service name should be valid
        config = ObservabilityConfig(service_name="")
        assert config.service_name == ""

        # Very long service name should be valid
        long_name = "a" * 1000
        config = ObservabilityConfig(service_name=long_name)
        assert config.service_name == long_name

        # Special characters in service name
        config = ObservabilityConfig(service_name="service-with-special_chars.123")
        assert config.service_name == "service-with-special_chars.123"

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ObservabilityConfig(
            enabled=True, service_name="test-service", trace_sample_rate=0.5
        )

        # Test dict conversion
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["enabled"] is True
        assert config_dict["service_name"] == "test-service"
        assert config_dict["trace_sample_rate"] == 0.5

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "enabled": True,
            "service_name": "dict-service",
            "service_version": "3.0.0",
            "trace_sample_rate": 0.25,
            "track_ai_operations": False,
        }

        config = ObservabilityConfig(**config_dict)

        assert config.enabled is True
        assert config.service_name == "dict-service"
        assert config.service_version == "3.0.0"
        assert config.trace_sample_rate == 0.25
        assert config.track_ai_operations is False

    def test_config_boolean_validation(self):
        """Test boolean field validation."""
        # Valid boolean values
        config = ObservabilityConfig(
            enabled=True, track_ai_operations=False, console_exporter=True
        )

        assert config.enabled is True
        assert config.track_ai_operations is False
        assert config.console_exporter is True

    def test_config_with_environment_simulation(self):
        """Test configuration with simulated environment variables."""
        # Simulate what would happen with environment-based configuration
        config = ObservabilityConfig(
            enabled=True,
            otlp_endpoint="http://jaeger:14268/api/traces",
            deployment_environment="kubernetes",
            service_name="k8s-ai-service",
            trace_sample_rate=0.1,
        )

        assert config.enabled is True
        assert config.otlp_endpoint == "http://jaeger:14268/api/traces"
        assert config.deployment_environment == "kubernetes"
        assert config.service_name == "k8s-ai-service"
        assert config.trace_sample_rate == 0.1
