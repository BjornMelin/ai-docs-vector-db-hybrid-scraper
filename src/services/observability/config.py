"""Observability configuration utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src import __version__


DEFAULT_INSTRUMENTATIONS = ("fastapi", "httpx")


@runtime_checkable
class SettingsLike(Protocol):
    """Application settings required to derive the runtime telemetry config."""

    app_name: str
    version: str
    environment: Any
    observability: Any


@dataclass(slots=True)
class ObservabilityConfig:  # pylint: disable=too-many-instance-attributes
    """Runtime configuration used to bootstrap OpenTelemetry."""

    enabled: bool = False
    service_name: str = "ai-docs-vector-db"
    service_version: str = __version__
    environment: str = "development"
    otlp_endpoint: str = "http://localhost:4317"
    otlp_headers: Mapping[str, str] = field(default_factory=dict)
    insecure_transport: bool = True
    instrumentations: Iterable[str] = field(
        default_factory=lambda: DEFAULT_INSTRUMENTATIONS
    )
    metrics_enabled: bool = True
    console_exporter: bool = False
    log_correlation: bool = False

    def resource_attributes(self) -> dict[str, str]:
        """Return OpenTelemetry resource attributes for service identification."""
        return {
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.environment,
            "telemetry.sdk.language": "python",
        }

    @classmethod
    def from_settings(cls, settings: SettingsLike) -> ObservabilityConfig:
        """Derive runtime telemetry configuration from canonical settings."""
        observed = settings.observability
        instrumentations: list[str] = []
        if observed.instrument_fastapi:
            instrumentations.append("fastapi")
        if observed.instrument_httpx:
            instrumentations.append("httpx")
        if observed.track_ai_operations or observed.track_costs:
            instrumentations.append("logging")

        environment = getattr(settings.environment, "value", settings.environment)
        return cls(
            enabled=observed.enabled,
            service_name=observed.service_name or settings.app_name,
            service_version=observed.service_version or settings.version,
            environment=str(environment),
            otlp_endpoint=observed.otlp_endpoint,
            otlp_headers=dict(observed.otlp_headers),
            insecure_transport=observed.otlp_insecure,
            instrumentations=tuple(dict.fromkeys(instrumentations)),
            metrics_enabled=observed.track_ai_operations,
            console_exporter=observed.console_exporter,
            log_correlation=observed.track_ai_operations or observed.track_costs,
        )


def get_observability_config() -> ObservabilityConfig:
    """Derive observability configuration from the canonical Settings owner."""
    from src.config.loader import get_settings

    return ObservabilityConfig.from_settings(get_settings())


def get_resource_attributes(
    config: ObservabilityConfig | None = None,
) -> Mapping[str, str]:
    """Return resource attributes for telemetry exporters.

    Args:
        config: Optional configuration override.

    Returns:
        Mapping of OpenTelemetry resource attributes.
    """
    config = config or get_observability_config()
    return config.resource_attributes()
