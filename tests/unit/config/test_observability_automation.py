"""Tests for observability-related configuration models."""

from __future__ import annotations

from typing import cast

import pytest

from src.config import Config
from src.config.models import Environment, ObservabilityConfig


@pytest.fixture(name="default_config")
def fixture_default_config() -> ObservabilityConfig:
    """Return the default observability configuration."""
    return ObservabilityConfig()


class TestObservabilityConfig:
    """Validate the standalone observability configuration model."""

    def test_defaults(self, default_config: ObservabilityConfig) -> None:
        """Ensure the default configuration exposes expected baseline values."""
        expected_pairs = {
            "enabled": False,
            "service_name": "ai-docs-vector-db",
            "service_version": "1.0.0",
            "service_namespace": "ai-docs",
            "otlp_endpoint": "http://localhost:4317",
            "trace_sample_rate": 1.0,
            "track_ai_operations": True,
            "track_costs": True,
            "console_exporter": False,
        }
        for attribute, expected in expected_pairs.items():
            assert getattr(default_config, attribute) == expected

    @pytest.mark.parametrize(
        "attribute",
        (
            "instrument_fastapi",
            "instrument_httpx",
            "instrument_redis",
            "instrument_sqlalchemy",
        ),
    )
    def test_instrumentation_defaults(
        self, attribute: str, default_config: ObservabilityConfig
    ) -> None:
        """Default instrumentation flags are enabled for supported integrations."""
        assert getattr(default_config, attribute) is True

    def test_config_embeds_observability_model(self) -> None:
        """`Config` exposes an instance of `ObservabilityConfig`."""
        settings = Config()
        observability_config = cast(
            ObservabilityConfig, object.__getattribute__(settings, "observability")
        )
        assert isinstance(observability_config, ObservabilityConfig)
        assert observability_config.enabled is False

    def test_enable_and_customize(self, default_config: ObservabilityConfig) -> None:
        """Allow toggling runtime observability preferences."""
        updated = default_config.model_copy(
            update={
                "enabled": True,
                "service_name": "test-ai-docs",
                "trace_sample_rate": 0.5,
            }
        )
        assert updated.enabled is True
        assert updated.service_name == "test-ai-docs"
        assert updated.trace_sample_rate == 0.5

    @pytest.mark.parametrize("trace_sample_rate", (1.0, 0.1, 0.0))
    def test_trace_sampling_bounds(self, trace_sample_rate: float) -> None:
        """Support full, partial, and disabled tracing sample rates."""
        config = ObservabilityConfig(trace_sample_rate=trace_sample_rate)
        assert config.trace_sample_rate == trace_sample_rate

    def test_custom_otlp_configuration(self) -> None:
        """Persist custom OTLP endpoint, headers, and TLS preferences."""
        config = ObservabilityConfig(
            enabled=True,
            otlp_endpoint="http://custom-otel:4317",
            otlp_headers={"api-key": "test-key"},
            otlp_insecure=False,
        )
        assert config.otlp_endpoint == "http://custom-otel:4317"
        assert config.otlp_headers == {"api-key": "test-key"}
        assert config.otlp_insecure is False

    def test_custom_headers_shape(self) -> None:
        """Retain OTLP headers dictionaries without mutation."""
        headers = {"authorization": "Bearer token123", "x-api-key": "api-key-123"}
        config = ObservabilityConfig(otlp_headers=headers)
        assert config.otlp_headers == headers
        assert len(config.otlp_headers) == len(headers)

    def test_custom_instrumentation_disable(self) -> None:
        """Disable individual instrumentation hooks when requested."""
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


class TestConfigIntegration:
    """Verify observability configuration derived from application settings."""

    def test_development_defaults(self) -> None:
        """Development mode keeps observability disabled by default."""
        settings = Config(environment=Environment.DEVELOPMENT)
        observability_config = cast(
            ObservabilityConfig, object.__getattribute__(settings, "observability")
        )
        assert isinstance(observability_config, ObservabilityConfig)
        assert observability_config.enabled is False

    def test_runtime_updates(self) -> None:
        """Runtime modifications to nested observability config are retained."""
        settings = Config()
        observability_config = cast(
            ObservabilityConfig, object.__getattribute__(settings, "observability")
        )
        assert isinstance(observability_config, ObservabilityConfig)
        observability_config.enabled = True
        observability_config.service_name = "runtime-service"
        observability_config.trace_sample_rate = 0.5
        assert observability_config.enabled is True
        assert observability_config.service_name == "runtime-service"
        assert observability_config.trace_sample_rate == 0.5
