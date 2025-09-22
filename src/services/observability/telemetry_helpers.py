"""Helper functions for telemetry initialization."""

import importlib
import logging
import sys
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
metrics = None
trace = None
OTLPMetricExporter = None
OTLPSpanExporter = None
FastAPIInstrumentor = None
HTTPXClientInstrumentor = None
RedisInstrumentor = None
SQLAlchemyInstrumentor = None
MeterProvider = None
PeriodicExportingMetricReader = None
Resource = None
TracerProvider = None
BatchSpanProcessor = None
ConsoleSpanExporter = None

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as _OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as _OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import (
        FastAPIInstrumentor as _FastAPIInstrumentor,
    )
    from opentelemetry.instrumentation.httpx import (
        HTTPXClientInstrumentor as _HTTPXClientInstrumentor,
    )
    from opentelemetry.instrumentation.redis import (
        RedisInstrumentor as _RedisInstrumentor,
    )
    from opentelemetry.instrumentation.sqlalchemy import (
        SQLAlchemyInstrumentor as _SQLAlchemyInstrumentor,
    )
    from opentelemetry.sdk.metrics import MeterProvider as _MeterProvider
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader as _PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource as _Resource
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as _BatchSpanProcessor,
        ConsoleSpanExporter as _ConsoleSpanExporter,
    )

    metrics = _otel_metrics
    trace = _otel_trace
    OTLPMetricExporter = _OTLPMetricExporter
    OTLPSpanExporter = _OTLPSpanExporter
    FastAPIInstrumentor = _FastAPIInstrumentor
    HTTPXClientInstrumentor = _HTTPXClientInstrumentor
    RedisInstrumentor = _RedisInstrumentor
    SQLAlchemyInstrumentor = _SQLAlchemyInstrumentor
    MeterProvider = _MeterProvider
    PeriodicExportingMetricReader = _PeriodicExportingMetricReader
    Resource = _Resource
    TracerProvider = _TracerProvider
    BatchSpanProcessor = _BatchSpanProcessor
    ConsoleSpanExporter = _ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


def _validate_telemetry_components() -> bool:
    """Validate required telemetry components are available."""

    if not OPENTELEMETRY_AVAILABLE:
        try:
            def _resolve(module_name: str, attr: str | None = None):
                module = sys.modules.get(module_name)
                if module is None:
                    module = importlib.import_module(module_name)
                return getattr(module, attr) if attr else module

            globals()["metrics"] = _resolve("opentelemetry.metrics")
            globals()["trace"] = _resolve("opentelemetry.trace")
            globals()["OTLPMetricExporter"] = _resolve(
                "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
                "OTLPMetricExporter",
            )
            globals()["OTLPSpanExporter"] = _resolve(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
                "OTLPSpanExporter",
            )
            globals()["MeterProvider"] = _resolve(
                "opentelemetry.sdk.metrics", "MeterProvider"
            )
            globals()["PeriodicExportingMetricReader"] = _resolve(
                "opentelemetry.sdk.metrics.export",
                "PeriodicExportingMetricReader",
            )
            globals()["Resource"] = _resolve("opentelemetry.sdk.resources", "Resource")
            globals()["TracerProvider"] = _resolve(
                "opentelemetry.sdk.trace", "TracerProvider"
            )

            trace_export_module = _resolve("opentelemetry.sdk.trace.export")
            globals()["BatchSpanProcessor"] = getattr(
                trace_export_module, "BatchSpanProcessor", None
            )
            globals()["ConsoleSpanExporter"] = getattr(
                trace_export_module, "ConsoleSpanExporter", None
            )
        except ImportError:
            logger.error("OpenTelemetry not available")
            return False
        except AttributeError:
            logger.error("Required OpenTelemetry components not available")
            return False
        except Exception as exc:  # pragma: no cover - safety net
            logger.error("Unexpected telemetry validation failure: %s", exc)
            raise

    if globals().get("metrics") is None:
        class _MetricsStub:
            def __init__(self) -> None:
                self._provider = None

            def set_meter_provider(self, provider):
                self._provider = provider

            def get_meter_provider(self):
                return self._provider

        globals()["metrics"] = _MetricsStub()

    if globals().get("trace") is None:
        class _TraceStub:
            def __init__(self) -> None:
                self._provider = None

            def set_tracer_provider(self, provider):
                self._provider = provider

            def get_tracer_provider(self):
                return self._provider

        globals()["trace"] = _TraceStub()

    required_components = [
        globals().get("OTLPMetricExporter"),
        globals().get("OTLPSpanExporter"),
        globals().get("MeterProvider"),
        globals().get("PeriodicExportingMetricReader"),
        globals().get("Resource"),
        globals().get("TracerProvider"),
        globals().get("BatchSpanProcessor"),
    ]
    if any(component is None for component in required_components):
        logger.error("Required OpenTelemetry components not available")
        return False

    return True


def _load_instrumentor(module_name: str, attr_name: str):
    """Load instrumentation class dynamically if not already imported."""

    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None

    return getattr(module, attr_name, None)


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


def _initialize_tracing(
    config: "ObservabilityConfig", resource: object
) -> object | None:
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
        return None
    else:
        return tracer_provider


def _initialize_metrics(config: "ObservabilityConfig", resource: object) -> object | None:
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
        return None
    else:
        return meter_provider


def _setup_fastapi_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup FastAPI instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_fastapi:
        return

    instrumentor = FastAPIInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.fastapi", "FastAPIInstrumentor"
        )

    if instrumentor is None:
        logger.warning("FastAPI instrumentation not available")
        return

    try:
        instrumentor().instrument()
        logger.info("FastAPI auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError, Exception):
        logger.warning("Failed to enable FastAPI instrumentation")


def _setup_httpx_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup HTTPX instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_httpx:
        return

    instrumentor = HTTPXClientInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"
        )

    if instrumentor is None:
        logger.warning("HTTPX instrumentation not available")
        return

    try:
        instrumentor().instrument()
        logger.info("HTTPX client auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError, Exception):
        logger.warning("Failed to enable HTTPX instrumentation")


def _setup_redis_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup Redis instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_redis:
        return

    instrumentor = RedisInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.redis", "RedisInstrumentor"
        )

    if instrumentor is None:
        logger.warning("Redis instrumentation not available")
        return

    try:
        instrumentor().instrument()
        logger.info("Redis auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError, Exception):
        logger.warning("Failed to enable Redis instrumentation")


def _setup_sqlalchemy_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup SQLAlchemy instrumentation.

    Args:
        config: Observability configuration
    """
    if not config.instrument_sqlalchemy:
        return

    instrumentor = SQLAlchemyInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.sqlalchemy",
            "SQLAlchemyInstrumentor",
        )

    if instrumentor is None:
        logger.warning("SQLAlchemy instrumentation not available")
        return

    try:
        instrumentor().instrument()
        logger.info("SQLAlchemy auto-instrumentation enabled")
    except (ConnectionError, OSError, PermissionError, Exception):
        logger.warning("Failed to enable SQLAlchemy instrumentation")
