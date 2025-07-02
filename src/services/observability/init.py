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
    from .mock_telemetry import create_mock_telemetry

    (
        OTLPSpanExporter,
        MeterProvider,
        PeriodicExportingMetricReader,
        Resource,
        TracerProvider,
        BatchSpanProcessor,
        metrics,
        trace,
    ) = create_mock_telemetry()

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

# Telemetry helper imports
try:
    from .telemetry_helpers import (
        _create_resource,
        _initialize_metrics,
        _initialize_tracing,
        _setup_fastapi_instrumentation,
        _setup_httpx_instrumentation,
        _setup_redis_instrumentation,
        _setup_sqlalchemy_instrumentation,
        _validate_telemetry_components,
    )
except ImportError:
    # Mock functions for when telemetry helpers aren't available
    _create_resource = None
    _initialize_metrics = None
    _initialize_tracing = None
    _setup_fastapi_instrumentation = None
    _setup_httpx_instrumentation = None
    _setup_redis_instrumentation = None
    _setup_sqlalchemy_instrumentation = None
    _validate_telemetry_components = None


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
        if not _validate_telemetry_components or not _validate_telemetry_components():
            return False

        logger.info("Initializing OpenTelemetry observability...")

        resource = _create_resource(config) if _create_resource else None
        if resource is None:
            return False

        if not _initialize_tracing or not _initialize_tracing(config, resource):
            return False

        # Store the tracer provider reference for shutdown
        if trace and hasattr(trace, "get_tracer_provider"):
            _tracer_provider = trace.get_tracer_provider()

        if not _initialize_metrics or not _initialize_metrics(config, resource):
            return False

        # Store the meter provider reference for shutdown
        if metrics and hasattr(metrics, "get_meter_provider"):
            _meter_provider = metrics.get_meter_provider()

        _setup_auto_instrumentation(config)

        logger.info(
            f"OpenTelemetry initialized successfully - "
            f"Service: {config.service_name}, "
            f"Endpoint: {config.otlp_endpoint}"
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
        if _setup_fastapi_instrumentation:
            _setup_fastapi_instrumentation(config)
        if _setup_httpx_instrumentation:
            _setup_httpx_instrumentation(config)
        if _setup_redis_instrumentation:
            _setup_redis_instrumentation(config)
        if _setup_sqlalchemy_instrumentation:
            _setup_sqlalchemy_instrumentation(config)

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
