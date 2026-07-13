"""Tests for the simplified observability configuration."""

from src import __version__
from src.config import Settings, get_settings, refresh_settings
from src.config.models import (
    Environment,
    ObservabilityConfig as SettingsObservabilityConfig,
)
from src.services.observability.config import (
    ObservabilityConfig,
    get_observability_config,
    get_resource_attributes,
)


class TestObservabilityConfig:
    """Behaviour of the dataclass itself."""

    def test_defaults(self) -> None:
        """Verify default configuration values."""
        config = ObservabilityConfig()
        assert config.enabled is False
        assert config.service_name == "ai-docs-vector-db"
        assert config.service_version == __version__
        assert config.otlp_endpoint == "http://localhost:4317"
        assert tuple(config.instrumentations) == ("fastapi", "httpx")
        assert config.ai_operation_metrics_enabled is True

    def test_resource_attributes(self) -> None:
        """Verify resource attributes are populated correctly."""
        config = ObservabilityConfig(
            service_name="example",
            service_version="2.1.0",
            environment="staging",
        )
        attrs = config.resource_attributes()
        assert attrs["service.name"] == "example"
        assert attrs["service.version"] == "2.1.0"
        assert attrs["deployment.environment"] == "staging"


class TestObservabilityConfigFromSettings:
    """Ensure canonical application settings drive runtime telemetry."""

    def test_settings_conversion(self) -> None:
        """Verify runtime values are derived from one Settings instance."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            observability=SettingsObservabilityConfig(
                enabled=True,
                service_name="configured-service",
                service_version="3.5.1",
                otlp_endpoint="http://collector:4317",
                otlp_headers={"authorization": "Bearer token"},
                otlp_insecure=False,
                track_ai_operations=False,
                track_costs=True,
                instrument_fastapi=True,
                instrument_httpx=False,
            ),
        )

        config = ObservabilityConfig.from_settings(settings)

        assert config.otlp_endpoint == "http://collector:4317"
        assert config.service_name == "configured-service"
        assert config.service_version == "3.5.1"
        assert config.environment == "production"
        assert config.ai_operation_metrics_enabled is True
        assert tuple(config.instrumentations) == ("fastapi", "logging")
        assert config.insecure_transport is False
        assert config.otlp_headers == {"authorization": "Bearer token"}

    def test_settings_refresh_is_visible_without_a_second_cache(self) -> None:
        """Telemetry config should follow the canonical Settings replacement."""
        original = get_settings()
        first = Settings(
            environment=Environment.TESTING,
            observability=SettingsObservabilityConfig(service_name="first"),
        )
        second = Settings(
            environment=Environment.TESTING,
            observability=SettingsObservabilityConfig(service_name="second"),
        )
        try:
            refresh_settings(settings=first)
            assert get_observability_config().service_name == "first"

            refresh_settings(settings=second)
            assert get_observability_config().service_name == "second"
        finally:
            refresh_settings(settings=original)

    def test_resource_attributes_helper(self) -> None:
        """Verify resource attributes helper function."""
        config = ObservabilityConfig(service_name="helper", service_version="1.2.3")
        attrs = get_resource_attributes(config)
        assert attrs["service.name"] == "helper"
        assert attrs["service.version"] == "1.2.3"
