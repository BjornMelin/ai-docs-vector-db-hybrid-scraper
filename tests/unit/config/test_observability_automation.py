"""Tests for Configuration Observability System.

Tests the observability configuration and monitoring capabilities.
"""

import pytest

from src.config.settings import (
    Environment,
    ObservabilityConfig,
    Settings,
)


class TestObservabilityConfig:
    """Test suite for observability configuration."""

    def test_default_observability_config(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()

        assert config.enabled is False  # Disabled by default
        assert config.service_name == "ai-docs-vector-db"
        assert config.service_version == "1.0.0"
        assert config.service_namespace == "ai-docs"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.trace_sample_rate == 1.0
        assert config.track_ai_operations is True
        assert config.track_costs is True

    def test_observability_instrumentation_settings(self):
        """Test observability instrumentation settings."""
        config = ObservabilityConfig()

        assert config.instrument_fastapi is True
        assert config.instrument_httpx is True
        assert config.instrument_redis is True
        assert config.instrument_sqlalchemy is True
        assert config.console_exporter is False

    def test_observability_in_settings(self):
        """Test observability configuration in main settings."""
        settings = Settings()

        assert hasattr(settings, "observability")
        assert isinstance(settings.observability, ObservabilityConfig)
        assert settings.observability.enabled is False

    def test_enabled_observability_config(self):
        """Test enabled observability configuration."""
        config = ObservabilityConfig(enabled=True)

        assert config.enabled is True
        assert config.track_ai_operations is True
        assert config.track_costs is True

    def test_custom_otlp_configuration(self):
        """Test custom OTLP configuration."""
        config = ObservabilityConfig(
            enabled=True,
            otlp_endpoint="http://custom-otel:4317",
            otlp_headers={"api-key": "test-key"},
            otlp_insecure=False,
        )

        assert config.otlp_endpoint == "http://custom-otel:4317"
        assert config.otlp_headers == {"api-key": "test-key"}
        assert config.otlp_insecure is False

    def test_trace_sampling_configuration(self):
        """Test trace sampling configuration."""
        # Full sampling
        config_full = ObservabilityConfig(trace_sample_rate=1.0)
        assert config_full.trace_sample_rate == 1.0

        # Partial sampling
        config_partial = ObservabilityConfig(trace_sample_rate=0.1)
        assert config_partial.trace_sample_rate == 0.1

        # No sampling
        config_none = ObservabilityConfig(trace_sample_rate=0.0)
        assert config_none.trace_sample_rate == 0.0

    def test_service_identification(self):
        """Test service identification configuration."""
        config = ObservabilityConfig(
            service_name="test-service",
            service_version="2.0.0",
            service_namespace="test-namespace",
        )

        assert config.service_name == "test-service"
        assert config.service_version == "2.0.0"
        assert config.service_namespace == "test-namespace"

    def test_ai_ml_tracking_configuration(self):
        """Test AI/ML specific tracking configuration."""
        config = ObservabilityConfig(track_ai_operations=True, track_costs=True)

        assert config.track_ai_operations is True
        assert config.track_costs is True

    def test_instrumentation_control(self):
        """Test instrumentation control configuration."""
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

    def test_debug_configuration(self):
        """Test debug configuration."""
        config = ObservabilityConfig(console_exporter=True)
        assert config.console_exporter is True


class TestObservabilityIntegration:
    """Test observability integration with main settings."""

    def test_observability_in_development(self):
        """Test observability configuration in development."""
        settings = Settings(environment=Environment.DEVELOPMENT)

        # Should be disabled by default in development
        assert settings.observability.enabled is False

    def test_observability_configuration_options(self):
        """Test various observability configuration options."""
        settings = Settings()

        # Update observability settings
        settings.observability.enabled = True
        settings.observability.service_name = "test-ai-docs"
        settings.observability.trace_sample_rate = 0.5

        assert settings.observability.enabled is True
        assert settings.observability.service_name == "test-ai-docs"
        assert settings.observability.trace_sample_rate == 0.5

    def test_observability_validation(self):
        """Test observability configuration validation."""
        # Valid sample rate
        config = ObservabilityConfig(trace_sample_rate=0.5)
        assert config.trace_sample_rate == 0.5

        # Test boundary values
        config_min = ObservabilityConfig(trace_sample_rate=0.0)
        assert config_min.trace_sample_rate == 0.0

        config_max = ObservabilityConfig(trace_sample_rate=1.0)
        assert config_max.trace_sample_rate == 1.0

    def test_observability_headers_configuration(self):
        """Test OTLP headers configuration."""
        headers = {"authorization": "Bearer token123", "x-api-key": "api-key-123"}

        config = ObservabilityConfig(otlp_headers=headers)
        assert config.otlp_headers == headers
        assert len(config.otlp_headers) == 2
