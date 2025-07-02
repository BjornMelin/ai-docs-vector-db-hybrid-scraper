"""OpenTelemetry initialization and setup for the AI documentation system.

Provides clean initialization patterns that integrate with the existing
service architecture while following OpenTelemetry best practices.
"""

import logging
from typing import TYPE_CHECKING, Any


# Optional OpenTelemetry imports - handled at runtime
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Mock classes for when OpenTelemetry is not available
    class OTLPSpanExporter:
        def __init__(self, *args, **kwargs):
            pass

    class MeterProvider:
        def __init__(self, *args, **kwargs):
            pass

    class PeriodicExportingMetricReader:
        def __init__(self, *args, **kwargs):
            pass

    class Resource:
        @staticmethod
        def create(*args, **kwargs):
            return None

    class TracerProvider:
        def __init__(self, *args, **kwargs):
            pass

    class BatchSpanProcessor:
        def __init__(self, *args, **kwargs):
            pass

    class MockMetrics:
        def set_meter_provider(self, *args, **kwargs):
            pass

    class MockTrace:
        def set_tracer_provider(self, *args, **kwargs):
            pass

    metrics = MockMetrics()
    trace = MockTrace()
    OPENTELEMETRY_AVAILABLE = False

# Additional OpenTelemetry imports - also optional
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
except ImportError:

    class OTLPMetricExporter:
        def __init__(self, *args, **kwargs):
            pass

    class ConsoleSpanExporter:
        def __init__(self, *args, **kwargs):
            pass

    class FastAPIInstrumentor:
        @staticmethod
        def instrument_app(*args, **kwargs):
            pass


# Additional instrumentation imports - also optional
try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
except ImportError:

    class HTTPXClientInstrumentor:
        @staticmethod
        def instrument(*args, **kwargs):
            pass

    class RedisInstrumentor:
        @staticmethod
        def instrument(*args, **kwargs):
            pass

    class SQLAlchemyInstrumentor:
        @staticmethod
        def instrument(*args, **kwargs):
            pass


if TYPE_CHECKING:
    from .config import ObservabilityConfig

# Optional imports for OpenTelemetry config - handled at runtime
try:
    from .config import get_observability_config, get_resource_attributes
except ImportError:
    get_observability_config = None
    get_resource_attributes = None


logger = logging.getLogger(__name__)

# Global references for cleanup
_tracer_provider: Any = None
_meter_provider: Any = None


def initialize_observability(config: "ObservabilityConfig" = None) -> bool:
    """Initialize OpenTelemetry observability infrastructure.

    Sets up distributed tracing, metrics collection, and AI-specific monitoring
    that integrates with the existing function-based service patterns.

    Args:
        config: Optional observability configuration

    Returns:
        True if initialization succeeded, False otherwise

    """
    global _tracer_provider, _meter_provider

    if config is None:
        if get_observability_config is None:
            logger.error("Config system not available")
            return False
        config = get_observability_config()

    if not config.enabled:
        logger.info("OpenTelemetry observability disabled by configuration")
        return False

    try:
        # Check for required OpenTelemetry components
        if metrics is None or trace is None or OTLPMetricExporter is None:
            logger.error("Required OpenTelemetry components not available")
            return False

        logger.info("Initializing OpenTelemetry observability...")

        # Create resource with service metadata
        if get_resource_attributes is None:
            logger.error("Resource attributes function not available")
            return False
        resource = Resource.create(get_resource_attributes(config))

        # Initialize tracing
        _tracer_provider = TracerProvider(resource=resource)

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

        _tracer_provider.add_span_processor(span_processor)

        # Add console exporter for development
        if config.console_exporter:
            if ConsoleSpanExporter is None:
                logger.warning("ConsoleSpanExporter not available")
            else:
                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                _tracer_provider.add_span_processor(console_processor)

        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Initialize metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=config.otlp_endpoint,
                headers=config.otlp_headers,
                insecure=config.otlp_insecure,
            ),
            export_interval_millis=30000,  # 30 seconds
        )

        _meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )

        # Set global meter provider
        metrics.set_meter_provider(_meter_provider)

        # Configure auto-instrumentation
        _setup_auto_instrumentation(config)

        logger.info(
            f"OpenTelemetry initialized successfully - "
            f"Service: {config.service_name}, "
            f"Endpoint: {config.otlp_endpoint}, "
            "Sample Rate"
        )

    except ImportError:
        logger.warning("OpenTelemetry packages not available")
        return False
    except (OSError, AttributeError, ModuleNotFoundError):
        logger.exception("Failed to initialize OpenTelemetry")
        return False
    else:
        return True


def _setup_auto_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup automatic instrumentation based on configuration.

    Args:
        config: Observability configuration

    """
    try:
        # FastAPI instrumentation
        if config.instrument_fastapi:
            if FastAPIInstrumentor is None:
                logger.warning("FastAPI instrumentation not available")
            else:
                try:
                    FastAPIInstrumentor().instrument()
                    logger.info("FastAPI auto-instrumentation enabled")
                except (ConnectionError, OSError, PermissionError):
                    logger.warning("Failed to enable FastAPI instrumentation")

        # HTTP client instrumentation
        if config.instrument_httpx:
            if HTTPXClientInstrumentor is None:
                logger.warning("HTTPX instrumentation not available")
            else:
                try:
                    HTTPXClientInstrumentor().instrument()
                    logger.info("HTTPX client auto-instrumentation enabled")
                except (ConnectionError, OSError, PermissionError):
                    logger.warning("Failed to enable HTTPX instrumentation")

        # Redis instrumentation
        if config.instrument_redis:
            if RedisInstrumentor is None:
                logger.warning("Redis instrumentation not available")
            else:
                try:
                    RedisInstrumentor().instrument()
                    logger.info("Redis auto-instrumentation enabled")
                except (ConnectionError, OSError, PermissionError):
                    logger.warning("Failed to enable Redis instrumentation")

        # SQLAlchemy instrumentation
        if config.instrument_sqlalchemy:
            if SQLAlchemyInstrumentor is None:
                logger.warning("SQLAlchemy instrumentation not available")
            else:
                try:
                    SQLAlchemyInstrumentor().instrument()
                    logger.info("SQLAlchemy auto-instrumentation enabled")
                except (OSError, PermissionError):
                    logger.warning("Failed to enable SQLAlchemy instrumentation")

    except (OSError, PermissionError):
        logger.warning("Auto-instrumentation setup failed")


def shutdown_observability() -> None:
    """Shutdown OpenTelemetry providers and flush pending data.

    Should be called on application shutdown to ensure all telemetry
    data is properly exported.
    """
    global _tracer_provider, _meter_provider

    # Shutdown tracer provider
    if _tracer_provider:
        try:
            logger.info("Shutting down OpenTelemetry tracer provider...")
            _tracer_provider.shutdown()
        except (OSError, PermissionError):
            logger.exception("Error during tracer provider shutdown")
        finally:
            _tracer_provider = None

    # Shutdown meter provider
    if _meter_provider:
        try:
            logger.info("Shutting down OpenTelemetry meter provider...")
            _meter_provider.shutdown()
        except (OSError, PermissionError):
            logger.exception("Error during meter provider shutdown")
        finally:
            _meter_provider = None

    logger.info("OpenTelemetry shutdown completed")


def is_observability_enabled() -> bool:
    """Check if observability is currently enabled and initialized.

    Returns:
        True if observability is enabled and initialized

    """
    return _tracer_provider is not None
