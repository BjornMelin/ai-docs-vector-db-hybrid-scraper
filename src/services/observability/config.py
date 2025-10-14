"""Observability configuration utilities."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


DEFAULT_INSTRUMENTATIONS = ("fastapi", "httpx")


@dataclass(slots=True)
class ObservabilityConfig:  # pylint: disable=too-many-instance-attributes
    """Runtime configuration used to bootstrap OpenTelemetry."""

    enabled: bool = True
    service_name: str = "ai-docs-vector-db"
    service_version: str = "1.0.0"
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
    def from_env(
        cls, overrides: Mapping[str, Any] | None = None
    ) -> ObservabilityConfig:
        overrides = dict(overrides or {})

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            overrides.setdefault("otlp_endpoint", endpoint)

        service_name = os.getenv("OTEL_SERVICE_NAME")
        if service_name:
            overrides.setdefault("service_name", service_name)

        service_version = os.getenv("OTEL_SERVICE_VERSION")
        if service_version:
            overrides.setdefault("service_version", service_version)

        environment = os.getenv("OTEL_ENVIRONMENT", os.getenv("ENV", "development"))
        overrides.setdefault("environment", environment)

        enabled_env = os.getenv("AI_DOCS_OBSERVABILITY_ENABLED")
        if enabled_env is not None:
            overrides.setdefault("enabled", _coerce_bool(enabled_env))

        console_env = os.getenv("AI_DOCS_OBSERVABILITY_CONSOLE_EXPORTER")
        if console_env is not None:
            overrides.setdefault("console_exporter", _coerce_bool(console_env))

        metrics_env = os.getenv("AI_DOCS_OBSERVABILITY_METRICS_ENABLED")
        if metrics_env is not None:
            overrides.setdefault("metrics_enabled", _coerce_bool(metrics_env))

        headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
        if headers_env:
            overrides.setdefault("otlp_headers", _parse_headers(headers_env))

        insecure_env = os.getenv("OTEL_EXPORTER_OTLP_INSECURE")
        if insecure_env is not None:
            overrides.setdefault("insecure_transport", _coerce_bool(insecure_env))

        instrumentations_env = os.getenv("AI_DOCS_OBSERVABILITY_INSTRUMENTATIONS")
        if instrumentations_env:
            overrides.setdefault(
                "instrumentations",
                tuple(
                    item.strip()
                    for item in instrumentations_env.split(",")
                    if item.strip()
                ),
            )

        return cls(**overrides)


@lru_cache(maxsize=1)
def _load_config() -> ObservabilityConfig:
    return ObservabilityConfig.from_env()


def clear_observability_cache() -> None:
    _load_config.cache_clear()


def get_observability_config(*, force_refresh: bool = False) -> ObservabilityConfig:
    if force_refresh:
        _load_config.cache_clear()
    return _load_config()


def get_resource_attributes(
    config: ObservabilityConfig | None = None,
) -> Mapping[str, str]:
    config = config or get_observability_config()
    return config.resource_attributes()


def _coerce_bool(value: str | bool) -> bool:
    """Coerce environment variable string to boolean."""
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_headers(raw: str) -> MutableMapping[str, str]:
    """Parse comma-separated key=value pairs into header mapping."""
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        headers[key.strip()] = val.strip()
    return headers
