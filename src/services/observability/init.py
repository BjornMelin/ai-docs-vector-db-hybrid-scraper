"""OpenTelemetry bootstrap helpers."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .config import ObservabilityConfig, get_observability_config


LOGGER = logging.getLogger(__name__)


@dataclass
class _TelemetryState:
    """Track runtime telemetry providers for OpenTelemetry bootstrap."""

    tracer_provider: Any | None = None
    meter_provider: Any | None = None


_STATE = _TelemetryState()


def _configure_instrumentations(instrumentations: Iterable[str]) -> None:
    """Enable optional OpenTelemetry instrumentations when available.

    Args:
        instrumentations: Collection of instrumentation module names to enable.
    """

    for name in instrumentations:
        try:
            if name == "fastapi":
                from opentelemetry.instrumentation.fastapi import (  # type: ignore[import-not-found]
                    FastAPIInstrumentor,
                )

                FastAPIInstrumentor().instrument()
            elif name == "httpx":
                from opentelemetry.instrumentation.httpx import (  # type: ignore[import-not-found]
                    HTTPXClientInstrumentor,
                )

                HTTPXClientInstrumentor().instrument()
            elif name == "requests":
                from opentelemetry.instrumentation.requests import (  # type: ignore[import-not-found]
                    RequestsInstrumentor,
                )

                RequestsInstrumentor().instrument()
            elif name == "logging":
                from opentelemetry.instrumentation.logging import (  # type: ignore[import-not-found]
                    LoggingInstrumentor,
                )

                LoggingInstrumentor().instrument(set_logging_format=True)
            else:
                LOGGER.debug("Unknown instrumentation '%s' requested", name)
        except ImportError:
            LOGGER.warning("Instrumentation '%s' not installed", name)


def initialize_observability(config: ObservabilityConfig | None = None) -> bool:
    # pylint: disable=too-many-locals
    """Initialise OpenTelemetry providers based on configuration.

    Args:
        config: Optional pre-parsed observability configuration. When omitted the
            value is loaded via ``get_observability_config``.

    Returns:
        bool: ``True`` when observability is enabled and providers are initialised.
    """

    config = config or get_observability_config()
    if not config.enabled:
        LOGGER.info("Observability disabled via configuration")
        return False

    if _STATE.tracer_provider is not None:
        return True  # already initialised

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # type: ignore[import-not-found]
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import (
            MeterProvider,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.metrics.export import (  # type: ignore[import-not-found]
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import (
            Resource,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.trace import (
            TracerProvider,  # type: ignore[import-not-found]
        )
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError as exc:  # pragma: no cover - validation guard
        LOGGER.warning("OpenTelemetry SDK not installed: %s", exc)
        return False

    resource = Resource.create(config.resource_attributes())

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(
        endpoint=config.otlp_endpoint,
        headers=dict(config.otlp_headers),
        insecure=config.insecure_transport,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    if config.console_exporter:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(tracer_provider)
    _STATE.tracer_provider = tracer_provider

    if config.metrics_enabled:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=config.otlp_endpoint,
                headers=dict(config.otlp_headers),
                insecure=config.insecure_transport,
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

    _configure_instrumentations(config.instrumentations)
    LOGGER.info(
        "Observability initialised - service=%s endpoint=%s",
        config.service_name,
        config.otlp_endpoint,
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
