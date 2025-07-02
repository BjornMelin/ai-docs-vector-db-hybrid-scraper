"""Helper functions for telemetry initialization."""

import logging
from typing import TYPE_CHECKING


# Import for resource attributes - safe to import as it's configuration only
try:
    from .config import get_resource_attributes
except ImportError:
    get_resource_attributes = None

if TYPE_CHECKING:
    from .config import ObservabilityConfig

logger = logging.getLogger(__name__)

# Import attempt for optional dependencies
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Set to None for checks
    OTLPMetricExporter = None
    ConsoleSpanExporter = None
    FastAPIInstrumentor = None
    HTTPXClientInstrumentor = None
    RedisInstrumentor = None
    SQLAlchemyInstrumentor = None


def _validate_telemetry_components() -> bool:
    """Validate required telemetry components are available.

    Returns:
        True if components are available, False otherwise
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.error("OpenTelemetry not available")
        return False

    if metrics is None or trace is None or OTLPMetricExporter is None:
        logger.error("Required OpenTelemetry components not available")
        return False

    return True


def _create_resource(config: "ObservabilityConfig") -> object | None:
    """Create OpenTelemetry resource.

    Args:
        config: Observability configuration

    Returns:
        Resource object or None if creation fails
    """
    try:
        if get_resource_attributes is None:
            logger.error("Resource attributes function not available")
            return None

        return Resource.create(get_resource_attributes(config))
    except (ImportError, ValueError, TypeError):
        logger.exception("Failed to create resource")
        return None


def _initialize_tracing(config: "ObservabilityConfig", resource: object) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        config: Observability configuration
        resource: OpenTelemetry resource

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        tracer_provider = TracerProvider(resource=resource)

        # Configure OTLP span exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            headers=config.otlp_headers,
            insecure=config.otlp_insecure,
        )

        # Configure batch span processor
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            schedule_delay_millis=config.batch_schedule_delay,
            max_queue_size=config.batch_max_queue_size,
            max_export_batch_size=config.batch_max_export_batch_size,
            export_timeout_millis=config.batch_export_timeout,
        )

        tracer_provider.add_span_processor(span_processor)

        # Add console exporter for development
        if config.console_exporter and ConsoleSpanExporter is not None:
            console_processor = BatchSpanProcessor(ConsoleSpanExporter())
            tracer_provider.add_span_processor(console_processor)

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
    except (ImportError, ValueError, TypeError, ConnectionError):
        logger.exception("Failed to initialize tracing")
        return False
    else:
        return True


def _initialize_metrics(config: "ObservabilityConfig", resource: object) -> bool:
    """Initialize OpenTelemetry metrics.

    Args:
        config: Observability configuration
        resource: OpenTelemetry resource

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        # Initialize metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=config.otlp_endpoint,
                headers=config.otlp_headers,
                insecure=config.otlp_insecure,
            ),
            export_interval_millis=30000,  # 30 seconds
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )

        # Set global meter provider
        metrics.set_meter_provider(meter_provider)
    except (ImportError, ValueError, TypeError, ConnectionError):
        logger.exception("Failed to initialize metrics")
        return False
    else:
        return True


def _setup_fastapi_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup FastAPI instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_fastapi:
        return

    if FastAPIInstrumentor is None:
        logger.warning("FastAPI instrumentation not available")
        return

    try:
        FastAPIInstrumentor().instrument()
        logger.info("FastAPI auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError):
        logger.warning("Failed to enable FastAPI instrumentation")


def _setup_httpx_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup HTTPX instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_httpx:
        return

    if HTTPXClientInstrumentor is None:
        logger.warning("HTTPX instrumentation not available")
        return

    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX client auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError):
        logger.warning("Failed to enable HTTPX instrumentation")


def _setup_redis_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup Redis instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_redis:
        return

    if RedisInstrumentor is None:
        logger.warning("Redis instrumentation not available")
        return

    try:
        RedisInstrumentor().instrument()
        logger.info("Redis auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError):
        logger.warning("Failed to enable Redis instrumentation")


def _setup_sqlalchemy_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup SQLAlchemy instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_sqlalchemy:
        return

    if SQLAlchemyInstrumentor is None:
        logger.warning("SQLAlchemy instrumentation not available")
        return

    try:
        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError):
        logger.warning("Failed to enable SQLAlchemy instrumentation")
