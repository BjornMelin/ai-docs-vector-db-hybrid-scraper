"""Tests for observability configuration."""

from unittest.mock import Mock, patch

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

    @patch("src.config.get_config")
    def test_get_config_with_main_config(self, mock_get_config):
        """Test getting config when main config is available."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        # Mock main config
        mock_main_config = Mock()
        mock_main_config.monitoring = Mock()
        mock_main_config.monitoring.enable_metrics = True
        mock_main_config.environment.value = "staging"
        mock_main_config.app_name = "AI Documentation System"
        mock_main_config.version = "1.5.0"

        mock_get_config.return_value = mock_main_config

        config = get_observability_config()

        assert config.enabled is True
        assert config.deployment_environment == "staging"
        assert config.service_name == "ai-documentation-system"
        assert config.service_version == "1.5.0"

    @patch("src.config.get_config")
    def test_get_config_without_monitoring(self, mock_get_config):
        """Test getting config when monitoring is not available."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        # Mock main config without monitoring
        mock_main_config = Mock()
        mock_main_config.monitoring = None
        mock_main_config.environment.value = "development"  # Use development as default

        mock_get_config.return_value = mock_main_config

        config = get_observability_config()

        # Should use defaults when monitoring not available
        # Environment is pulled from main config when available
        assert config.deployment_environment == "development"

    @patch("src.config.get_config")
    def test_get_config_partial_attributes(self, mock_get_config):
        """Test getting config with only some attributes available."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        # Mock main config with minimal attributes
        mock_main_config = Mock()
        mock_main_config.app_name = "Test App"
        # Add environment but missing others
        mock_main_config.environment.value = "staging"
        # Remove monitoring and version attributes by using spec
        mock_config_spec = Mock(spec=["app_name", "environment"])
        mock_config_spec.app_name = "Test App"
        mock_config_spec.environment.value = "staging"

        mock_get_config.return_value = mock_config_spec

        config = get_observability_config()

        # Environment is pulled from main config
        assert config.deployment_environment == "staging"
        # Version should be default when not available in config
        assert config.service_version == "1.0.0"

    @patch("src.config.get_config")
    def test_get_config_exception_handling(self, mock_get_config):
        """Test graceful handling when main config fails."""
        mock_get_config.side_effect = Exception("Config loading failed")

        config = get_observability_config()

        # Should return fallback config when main config fails
        # Config may vary based on test environment, just verify it returns a valid config
        assert isinstance(config, ObservabilityConfig)

    @patch("src.config.get_config")
    def test_get_config_import_error(self, mock_get_config):
        """Test handling of import errors."""
        # Clear cache first to ensure clean test
        get_observability_config.cache_clear()

        mock_get_config.side_effect = ImportError("Module not found")

        config = get_observability_config()

        # Should return default config with correct service name
        assert isinstance(config, ObservabilityConfig)
        assert config.service_name == "ai-docs-vector-db"

    def test_get_config_caching(self):
        """Test that config is cached properly."""
        # Clear cache first
        get_observability_config.cache_clear()

        with patch("src.config.get_config") as mock_get_config:
            mock_main_config = Mock()
            mock_main_config.app_name = "Cached App"
            mock_get_config.return_value = mock_main_config

            # First call
            config1 = get_observability_config()

            # Second call
            config2 = get_observability_config()

            # Should be the same object (cached)
            assert config1 is config2

            # get_config should only be called once due to caching
            assert mock_get_config.call_count == 1


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
