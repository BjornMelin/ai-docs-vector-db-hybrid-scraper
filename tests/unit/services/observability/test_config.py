"""Tests for the simplified observability configuration."""

# pylint: disable=duplicate-code

import os
from unittest.mock import patch

from src.services.observability.config import (
    ObservabilityConfig,
    clear_observability_cache,
    get_observability_config,
    get_resource_attributes,
)


class TestObservabilityConfig:
    """Behaviour of the dataclass itself."""

    def test_defaults(self) -> None:
        config = ObservabilityConfig()
        assert config.enabled is True
        assert config.service_name == "ai-docs-vector-db"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert tuple(config.instrumentations) == ("fastapi", "httpx")
        assert config.metrics_enabled is True

    def test_resource_attributes(self) -> None:
        config = ObservabilityConfig(
            service_name="example",
            service_version="2.1.0",
            environment="staging",
        )
        attrs = config.resource_attributes()
        assert attrs["service.name"] == "example"
        assert attrs["service.version"] == "2.1.0"
        assert attrs["deployment.environment"] == "staging"


class TestObservabilityConfigFromEnv:
    """Ensure environment overrides are respected."""

    def setup_method(self) -> None:
        clear_observability_cache()

    def teardown_method(self) -> None:
        clear_observability_cache()

    def test_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
                "OTEL_SERVICE_NAME": "env-service",
                "OTEL_SERVICE_VERSION": "3.5.1",
                "OTEL_ENVIRONMENT": "production",
                "OTEL_EXPORTER_OTLP_HEADERS": "authorization=Bearer token",
                "OTEL_EXPORTER_OTLP_INSECURE": "false",
                "AI_DOCS_OBSERVABILITY_METRICS_ENABLED": "false",
                "AI_DOCS_OBSERVABILITY_INSTRUMENTATIONS": "fastapi,logging",
            },
            clear=True,
        ):
            config = get_observability_config(force_refresh=True)

        assert config.otlp_endpoint == "http://collector:4317"
        assert config.service_name == "env-service"
        assert config.service_version == "3.5.1"
        assert config.environment == "production"
        assert config.metrics_enabled is False
        assert tuple(config.instrumentations) == ("fastapi", "logging")
        assert config.insecure_transport is False
        assert config.otlp_headers == {"authorization": "Bearer token"}

    def test_resource_attributes_helper(self) -> None:
        config = ObservabilityConfig(service_name="helper", service_version="1.2.3")
        attrs = get_resource_attributes(config)
        assert attrs["service.name"] == "helper"
        assert attrs["service.version"] == "1.2.3"
