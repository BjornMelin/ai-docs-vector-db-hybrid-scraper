"""OpenTelemetry bootstrap helpers."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast

from .config import (
    ObservabilityConfig,
    SettingsLike,
    get_observability_config,
)


logger = logging.getLogger(__name__)


# Instrumentation registry intentionally uses string-based imports so optional
# extras remain truly optional while passing static analysis.
_INSTRUMENTATION_SPECS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "fastapi": (
        "opentelemetry.instrumentation.fastapi",
        "FastAPIInstrumentor",
        {},
    ),
    "httpx": (
        "opentelemetry.instrumentation.httpx",
        "HTTPXClientInstrumentor",
        {},
    ),
    "requests": (
        "opentelemetry.instrumentation.requests",
        "RequestsInstrumentor",
        {},
    ),
    "logging": (
        "opentelemetry.instrumentation.logging",
        "LoggingInstrumentor",
        {"set_logging_format": True},
    ),
}

if TYPE_CHECKING:

    class _SupportsInstrument(Protocol):
        """Protocol describing minimal instrumentation interface."""

        def instrument(self, **kwargs: Any) -> None:
            """Apply instrumentation to target with given configuration."""

else:

    class _SupportsInstrument:  # pragma: no cover - runtime placeholder
        """Runtime placeholder matching `_SupportsInstrument` protocol."""

        def instrument(self, **_: Any) -> None:
            """Raise NotImplementedError for runtime instrumentation."""
            raise NotImplementedError


@dataclass
class _TelemetryState:
    """Track runtime telemetry providers for OpenTelemetry bootstrap."""

    tracer_provider: Any | None = None
    meter_provider: Any | None = None


_STATE = _TelemetryState()


def _coerce_config(
    config: ObservabilityConfig | SettingsLike | None,
) -> ObservabilityConfig:
    """Normalize supported inputs for observability initialization."""
    if config is None:
        return get_observability_config()
    if isinstance(config, ObservabilityConfig):
        return config
    if isinstance(config, SettingsLike):
        return ObservabilityConfig.from_settings(config)
    msg = f"Unsupported observability configuration type: {type(config)!r}"
    raise TypeError(msg)


def _configure_instrumentations(instrumentations: Iterable[str]) -> None:
    """Enable optional OpenTelemetry instrumentations when available.

    Args:
        instrumentations: Collection of instrumentation module names to enable.
    """
    for name in instrumentations:
        spec = _INSTRUMENTATION_SPECS.get(name)
        if spec is None:
            logger.debug("Unknown instrumentation '%s' requested", name)
            continue

        try:
            module_path, attr_name, kwargs = spec
            instrumentor_cls = getattr(import_module(module_path), attr_name)
            instrumentor = cast("_SupportsInstrument", instrumentor_cls())
            instrumentor.instrument(**kwargs)
        except ImportError:
            logger.warning("Instrumentation '%s' not installed", name)
        except Exception as exc:
            logger.exception(
                "Failed to configure instrumentation '%s'", name, exc_info=exc
            )


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
        logger.info("Observability disabled via configuration")
        return False

    if _STATE.tracer_provider is not None:
        return True  # already initialized

    tracer_provider: Any | None = None
    meter_provider: Any | None = None
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create(runtime_config.resource_attributes())

        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(
            endpoint=runtime_config.otlp_endpoint,
            headers=dict(runtime_config.otlp_headers),
            insecure=runtime_config.insecure_transport,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        if runtime_config.console_exporter:
            tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        trace.set_tracer_provider(tracer_provider)
        _STATE.tracer_provider = tracer_provider

        if runtime_config.ai_operation_metrics_enabled:
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
        logger.info(
            "Observability initialized - service=%s endpoint=%s",
            runtime_config.service_name,
            runtime_config.otlp_endpoint,
        )
        return True
    except ImportError as exc:  # pragma: no cover - validation guard
        logger.warning("OpenTelemetry SDK not installed: %s", exc)
        return False
    except Exception as exc:
        logger.exception("Failed to initialize observability", exc_info=exc)
        if tracer_provider is not None:
            with suppress(Exception):
                tracer_provider.shutdown()
        if meter_provider is not None:
            with suppress(Exception):
                meter_provider.shutdown()
        _STATE.tracer_provider = None
        _STATE.meter_provider = None
        return False


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
