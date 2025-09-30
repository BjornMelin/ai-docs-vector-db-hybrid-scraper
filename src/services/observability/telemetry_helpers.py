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
    metrics = importlib.import_module("opentelemetry.metrics")
    trace = importlib.import_module("opentelemetry.trace")
    OTLPMetricExporter = getattr(
        importlib.import_module(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter"
        ),
        "OTLPMetricExporter",
        None,
    )
    OTLPSpanExporter = getattr(
        importlib.import_module(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        ),
        "OTLPSpanExporter",
        None,
    )
    FastAPIInstrumentor = getattr(
        importlib.import_module("opentelemetry.instrumentation.fastapi"),
        "FastAPIInstrumentor",
        None,
    )
    HTTPXClientInstrumentor = getattr(
        importlib.import_module("opentelemetry.instrumentation.httpx"),
        "HTTPXClientInstrumentor",
        None,
    )
    RedisInstrumentor = getattr(
        importlib.import_module("opentelemetry.instrumentation.redis"),
        "RedisInstrumentor",
        None,
    )
    SQLAlchemyInstrumentor = getattr(
        importlib.import_module("opentelemetry.instrumentation.sqlalchemy"),
        "SQLAlchemyInstrumentor",
        None,
    )
    MeterProvider = getattr(
        importlib.import_module("opentelemetry.sdk.metrics"),
        "MeterProvider",
        None,
    )
    PeriodicExportingMetricReader = getattr(
        importlib.import_module("opentelemetry.sdk.metrics.export"),
        "PeriodicExportingMetricReader",
        None,
    )
    Resource = getattr(
        importlib.import_module("opentelemetry.sdk.resources"),
        "Resource",
        None,
    )
    TracerProvider = getattr(
        importlib.import_module("opentelemetry.sdk.trace"),
        "TracerProvider",
        None,
    )
    _trace_export = importlib.import_module("opentelemetry.sdk.trace.export")
    BatchSpanProcessor = getattr(_trace_export, "BatchSpanProcessor", None)
    ConsoleSpanExporter = getattr(_trace_export, "ConsoleSpanExporter", None)

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
        except ImportError as exc:
            logger.debug("OpenTelemetry not available: %s", exc)
            return False
        except AttributeError as exc:
            logger.debug("Required OpenTelemetry components not available: %s", exc)
            return False
        except Exception:  # pragma: no cover - safety net
            logger.exception("Unexpected telemetry validation failure")
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
        if get_resource_attributes is None or Resource is None:
            logger.error("Resource attributes function not available")
            return None

        return Resource.create(get_resource_attributes(config))  # type: ignore[attr-defined]
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
        tracer_provider = TracerProvider(resource=resource)  # type: ignore[operator]

        # Configure OTLP span exporter
        otlp_exporter = OTLPSpanExporter(  # type: ignore[operator]
            endpoint=config.otlp_endpoint,
            headers=config.otlp_headers,
            insecure=config.otlp_insecure,
        )

        # Configure batch span processor
        span_processor = BatchSpanProcessor(  # type: ignore[operator]
            otlp_exporter,
            schedule_delay_millis=config.batch_schedule_delay,
            max_queue_size=config.batch_max_queue_size,
            max_export_batch_size=config.batch_max_export_batch_size,
            export_timeout_millis=config.batch_export_timeout,
        )

        tracer_provider.add_span_processor(span_processor)

        # Add console exporter for development
        if config.console_exporter and ConsoleSpanExporter is not None:
            console_processor = BatchSpanProcessor(ConsoleSpanExporter())  # type: ignore[operator]
            tracer_provider.add_span_processor(console_processor)

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)  # type: ignore[attr-defined]
    except (ImportError, ValueError, TypeError, ConnectionError):
        logger.exception("Failed to initialize tracing")
        return None
    else:
        return tracer_provider


def _initialize_metrics(
    config: "ObservabilityConfig", resource: object
) -> object | None:
    """Initialize OpenTelemetry metrics.

    Args:
        config: Observability configuration
        resource: OpenTelemetry resource

    Returns:
        True if initialization successful, False otherwise

    """
    try:
        # Initialize metrics
        metric_reader = PeriodicExportingMetricReader(  # type: ignore[operator]
            OTLPMetricExporter(  # type: ignore[operator]
                endpoint=config.otlp_endpoint,
                headers=config.otlp_headers,
                insecure=config.otlp_insecure,
            ),
        )

        meter_provider = MeterProvider(  # type: ignore[operator]
            resource=resource,
            metric_readers=[metric_reader],
        )

        # Set global meter provider
        metrics.set_meter_provider(meter_provider)  # type: ignore[attr-defined]
    except (ImportError, ValueError, TypeError, ConnectionError):
        logger.exception("Failed to initialize metrics")
        return None
    else:
        return meter_provider


def _setup_fastapi_instrumentation(config: "ObservabilityConfig") -> bool:
    """Setup FastAPI instrumentation.

    Args:
        config: Observability configuration

    """
    if not config.instrument_fastapi:
        return False

    instrumentor = FastAPIInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.fastapi", "FastAPIInstrumentor"
        )
    else:
        try:
            importlib.import_module("opentelemetry.instrumentation.fastapi")
        except Exception as exc:  # noqa: BLE001 - instrumentation availability check
            logger.warning("Failed to enable FastAPI instrumentation: %s", exc)
            return False

    if instrumentor is None:
        logger.warning("FastAPI instrumentation not available")
        return False

    try:
        instrumentor().instrument()
    except (ConnectionError, OSError, PermissionError) as exc:
        logger.warning("Failed to enable FastAPI instrumentation: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - instrumentation should be fire-and-forget
        logger.warning("Failed to enable FastAPI instrumentation: %s", exc)
        return False
    else:
        logger.info("FastAPI auto-instrumentation enabled")
        return True


def _setup_httpx_instrumentation(config: "ObservabilityConfig") -> bool:
    """Setup HTTPX instrumentation.

    Args:
        config: Observability configuration

    """
    if not config.instrument_httpx:
        return False

    instrumentor = HTTPXClientInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"
        )
    else:
        try:
            importlib.import_module("opentelemetry.instrumentation.httpx")
        except Exception as exc:  # noqa: BLE001 - instrumentation availability check
            logger.warning("Failed to enable HTTPX instrumentation: %s", exc)
            return False

    if instrumentor is None:
        logger.warning("HTTPX instrumentation not available")
        return False

    try:
        instrumentor().instrument()
    except (ConnectionError, OSError, PermissionError) as exc:
        logger.warning("Failed to enable HTTPX instrumentation: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - instrumentation should be fire-and-forget
        logger.warning("Failed to enable HTTPX instrumentation: %s", exc)
        return False
    else:
        logger.info("HTTPX client auto-instrumentation enabled")
        return True


def _setup_redis_instrumentation(config: "ObservabilityConfig") -> bool:
    """Setup Redis instrumentation.

    Args:
        config: Observability configuration

    """
    if not config.instrument_redis:
        return False

    instrumentor = RedisInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.redis", "RedisInstrumentor"
        )
    else:
        try:
            importlib.import_module("opentelemetry.instrumentation.redis")
        except Exception as exc:  # noqa: BLE001 - instrumentation availability check
            logger.warning("Failed to enable Redis instrumentation: %s", exc)
            return False

    if instrumentor is None:
        logger.warning("Redis instrumentation not available")
        return False

    try:
        instrumentor().instrument()
    except (ConnectionError, OSError, PermissionError) as exc:
        logger.warning("Failed to enable Redis instrumentation: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - instrumentation should be fire-and-forget
        logger.warning("Failed to enable Redis instrumentation: %s", exc)
        return False
    else:
        logger.info("Redis auto-instrumentation enabled")
        return True


def _setup_sqlalchemy_instrumentation(config: "ObservabilityConfig") -> bool:
    """Setup SQLAlchemy instrumentation.

    Args:
        config: Observability configuration

    """
    if not config.instrument_sqlalchemy:
        return False

    instrumentor = SQLAlchemyInstrumentor
    if instrumentor is None:
        instrumentor = _load_instrumentor(
            "opentelemetry.instrumentation.sqlalchemy",
            "SQLAlchemyInstrumentor",
        )
    else:
        try:
            importlib.import_module("opentelemetry.instrumentation.sqlalchemy")
        except Exception as exc:  # noqa: BLE001 - instrumentation availability check
            logger.warning("Failed to enable SQLAlchemy instrumentation: %s", exc)
            return False

    if instrumentor is None:
        logger.warning("SQLAlchemy instrumentation not available")
        return False

    try:
        instrumentor().instrument()
    except (ConnectionError, OSError, PermissionError) as exc:
        logger.warning("Failed to enable SQLAlchemy instrumentation: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - instrumentation should be fire-and-forget
        logger.warning("Failed to enable SQLAlchemy instrumentation: %s", exc)
        return False
    else:
        logger.info("SQLAlchemy auto-instrumentation enabled")
        return True
