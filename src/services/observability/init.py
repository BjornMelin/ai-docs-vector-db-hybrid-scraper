import typing
"""OpenTelemetry initialization and setup for the AI documentation system.

Provides clean initialization patterns that integrate with the existing
service architecture while following OpenTelemetry best practices.
"""

import logging
from typing import Any

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
        from .config import get_observability_config

        config = get_observability_config()

    if not config.enabled:
        logger.info("OpenTelemetry observability disabled by configuration")
        return False

    try:
        # Import OpenTelemetry components
        from opentelemetry import metrics
        from opentelemetry import trace
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
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from .config import get_resource_attributes

        logger.info("Initializing OpenTelemetry observability...")

        # Create resource with service metadata
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
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

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
            f"Sample Rate: {config.trace_sample_rate}"
        )

        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not available: {e}")
        return False
    except Exception as e:
        logger.exception(f"Failed to initialize OpenTelemetry: {e}")
        return False


def _setup_auto_instrumentation(config: "ObservabilityConfig") -> None:
    """Setup automatic instrumentation based on configuration.

    Args:
        config: Observability configuration
    """
    try:
        # FastAPI instrumentation
        if config.instrument_fastapi:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

                FastAPIInstrumentor().instrument()
                logger.info("FastAPI auto-instrumentation enabled")
            except ImportError:
                logger.warning("FastAPI instrumentation not available")

        # HTTP client instrumentation
        if config.instrument_httpx:
            try:
                from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

                HTTPXClientInstrumentor().instrument()
                logger.info("HTTPX client auto-instrumentation enabled")
            except ImportError:
                logger.warning("HTTPX instrumentation not available")

        # Redis instrumentation
        if config.instrument_redis:
            try:
                from opentelemetry.instrumentation.redis import RedisInstrumentor

                RedisInstrumentor().instrument()
                logger.info("Redis auto-instrumentation enabled")
            except ImportError:
                logger.warning("Redis instrumentation not available")

        # SQLAlchemy instrumentation
        if config.instrument_sqlalchemy:
            try:
                from opentelemetry.instrumentation.sqlalchemy import (
                    SQLAlchemyInstrumentor,
                )

                SQLAlchemyInstrumentor().instrument()
                logger.info("SQLAlchemy auto-instrumentation enabled")
            except ImportError:
                logger.warning("SQLAlchemy instrumentation not available")

    except Exception as e:
        logger.warning(f"Auto-instrumentation setup failed: {e}")


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
        except Exception as e:
            logger.exception(f"Error during tracer provider shutdown: {e}")
        finally:
            _tracer_provider = None

    # Shutdown meter provider
    if _meter_provider:
        try:
            logger.info("Shutting down OpenTelemetry meter provider...")
            _meter_provider.shutdown()
        except Exception as e:
            logger.exception(f"Error during meter provider shutdown: {e}")
        finally:
            _meter_provider = None

    logger.info("OpenTelemetry shutdown completed")


def is_observability_enabled() -> bool:
    """Check if observability is currently enabled and initialized.

    Returns:
        True if observability is enabled and initialized
    """
    return _tracer_provider is not None
