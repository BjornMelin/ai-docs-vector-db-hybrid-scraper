"""OpenTelemetry bootstrap helpers."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from .config import (
    DEFAULT_INSTRUMENTATIONS,
    ObservabilityConfig,
    get_observability_config,
)


@runtime_checkable
class SettingsLike(Protocol):
    """Structural protocol for application settings with observability data."""

    app_name: str | None
    version: str | None
    environment: Any
    observability: Any


LOGGER = logging.getLogger(__name__)


@dataclass
class _TelemetryState:
    """Track runtime telemetry providers for OpenTelemetry bootstrap."""

    tracer_provider: Any | None = None
    meter_provider: Any | None = None


_STATE = _TelemetryState()


def _has_explicit_instrumentation_preferences(observed: Any) -> bool:
    """Detect whether instrumentation toggles were explicitly provided."""
    preference_fields = {"instrument_fastapi", "instrument_httpx"}

    fields_set = getattr(observed, "model_fields_set", None)
    if fields_set is not None and preference_fields.intersection(fields_set):
        return True

    model_dump = getattr(observed, "model_dump", None)
    if callable(model_dump):
        result = model_dump(exclude_defaults=True, exclude_unset=True)
        if isinstance(result, Mapping):
            explicit_values = cast(Mapping[str, Any], result)
            if any(field in explicit_values for field in preference_fields):
                return True

    if isinstance(observed, Mapping) and any(
        field in observed for field in preference_fields
    ):
        return True

    sentinel = object()
    for field in preference_fields:
        if getattr(observed, field, sentinel) is not sentinel:
            return True

    return False


def _from_settings(settings: SettingsLike) -> ObservabilityConfig:
    """Create an :class:`ObservabilityConfig` from application settings."""
    observed = settings.observability
    instrumentations: list[str] = []
    if getattr(observed, "instrument_fastapi", False):
        instrumentations.append("fastapi")
    if getattr(observed, "instrument_httpx", False):
        instrumentations.append("httpx")
    if getattr(observed, "track_ai_operations", False) or getattr(
        observed, "track_costs", False
    ):
        instrumentations.append("logging")

    instrumentation_tuple = tuple(dict.fromkeys(instrumentations))
    if not instrumentation_tuple and not _has_explicit_instrumentation_preferences(
        observed
    ):
        instrumentation_tuple = DEFAULT_INSTRUMENTATIONS

    environment = getattr(settings, "environment", "development")
    environment_value = getattr(environment, "value", environment)

    return ObservabilityConfig(
        enabled=bool(observed.enabled),
        service_name=getattr(observed, "service_name", "")
        or getattr(settings, "app_name", "ai-docs-vector-db"),
        service_version=getattr(observed, "service_version", "")
        or getattr(settings, "version", "1.0.0"),
        environment=str(environment_value),
        otlp_endpoint=getattr(observed, "otlp_endpoint", "http://localhost:4317"),
        otlp_headers=dict(getattr(observed, "otlp_headers", {})),
        insecure_transport=bool(getattr(observed, "otlp_insecure", True)),
        instrumentations=instrumentation_tuple,
        metrics_enabled=bool(getattr(observed, "track_ai_operations", False)),
        console_exporter=bool(getattr(observed, "console_exporter", False)),
        log_correlation=bool(
            getattr(observed, "track_ai_operations", False)
            or getattr(observed, "track_costs", False)
        ),
    )


def _coerce_config(
    config: ObservabilityConfig | SettingsLike | None,
) -> ObservabilityConfig:
    """Normalize configuration inputs for observability initialization."""
    if config is None:
        return get_observability_config()
    if isinstance(config, ObservabilityConfig):
        return config
    if isinstance(config, SettingsLike):
        return _from_settings(config)
    msg = f"Unsupported observability configuration type: {type(config)!r}"
    raise TypeError(msg)


def _configure_instrumentations(instrumentations: Iterable[str]) -> None:
    """Enable optional OpenTelemetry instrumentations when available.

    Args:
        instrumentations: Collection of instrumentation module names to enable.
    """
    for name in instrumentations:
        try:
            if name == "fastapi":
                from opentelemetry.instrumentation.fastapi import (  # type: ignore[import-not-found]
                    FastAPIInstrumentor,  # pyright: ignore[reportMissingImports]
                )

                FastAPIInstrumentor().instrument()
            elif name == "httpx":
                from opentelemetry.instrumentation.httpx import (  # type: ignore[import-not-found]
                    HTTPXClientInstrumentor,  # pyright: ignore[reportMissingImports]
                )

                HTTPXClientInstrumentor().instrument()
            elif name == "requests":
                from opentelemetry.instrumentation.requests import (  # type: ignore[import-not-found]
                    RequestsInstrumentor,  # pyright: ignore[reportMissingImports]
                )

                RequestsInstrumentor().instrument()
            elif name == "logging":
                from opentelemetry.instrumentation.logging import (  # type: ignore[import-not-found]
                    LoggingInstrumentor,  # pyright: ignore[reportMissingImports]
                )

                LoggingInstrumentor().instrument(set_logging_format=True)
            else:
                LOGGER.debug("Unknown instrumentation '%s' requested", name)
        except ImportError:
            LOGGER.warning("Instrumentation '%s' not installed", name)


def initialize_observability(
    config: ObservabilityConfig | SettingsLike | None = None,
) -> bool:
    # pylint: disable=too-many-locals
    """Initialise OpenTelemetry providers based on configuration.

    Args:
        config: Optional observability configuration or full application settings.
            When omitted the value is loaded via ``get_observability_config``.

    Returns:
        bool: ``True`` when observability is enabled and providers are initialized.
    """
    runtime_config = _coerce_config(config)
    if not runtime_config.enabled:
        LOGGER.info("Observability disabled via configuration")
        return False

    if _STATE.tracer_provider is not None:
        return True  # already initialized

    try:
        from opentelemetry import (  # pyright: ignore[reportMissingImports]
            metrics,
            trace,
        )
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # type: ignore[import-not-found]
            OTLPMetricExporter,  # pyright: ignore[reportMissingImports]
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,  # pyright: ignore[reportMissingImports]
        )
        from opentelemetry.sdk.metrics import (  # pyright: ignore[reportMissingImports]
            MeterProvider,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.metrics.export import (  # type: ignore[import-not-found]
            PeriodicExportingMetricReader,  # pyright: ignore[reportMissingImports]
        )
        from opentelemetry.sdk.resources import (  # pyright: ignore[reportMissingImports]
            Resource,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.trace import (  # pyright: ignore[reportMissingImports]
            TracerProvider,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - validation guard
        LOGGER.warning("OpenTelemetry SDK not installed: %s", exc)
        return False

    resource = Resource.create(runtime_config.resource_attributes())

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(
        endpoint=runtime_config.otlp_endpoint,
        headers=dict(runtime_config.otlp_headers),
        insecure=runtime_config.insecure_transport,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    if runtime_config.console_exporter:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(tracer_provider)
    _STATE.tracer_provider = tracer_provider

    if runtime_config.metrics_enabled:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=runtime_config.otlp_endpoint,
                headers=dict(runtime_config.otlp_headers),
                insecure=runtime_config.insecure_transport,
            )
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)
        _STATE.meter_provider = meter_provider
    else:
        _STATE.meter_provider = None

    _configure_instrumentations(runtime_config.instrumentations)
    LOGGER.info(
        "Observability initialized - service=%s endpoint=%s",
        runtime_config.service_name,
        runtime_config.otlp_endpoint,
    )
    return True


def shutdown_observability() -> None:
    """Flush exporters and reset providers."""
    if _STATE.tracer_provider is not None:
        try:
            _STATE.tracer_provider.shutdown()
        finally:
            _STATE.tracer_provider = None
    if _STATE.meter_provider is not None:
        try:
            _STATE.meter_provider.shutdown()
        finally:
            _STATE.meter_provider = None


def is_observability_enabled() -> bool:
    """Check if observability is currently enabled and initialized.

    Returns:
        bool: ``True`` when a tracer provider has been configured.
    """
    return _STATE.tracer_provider is not None
