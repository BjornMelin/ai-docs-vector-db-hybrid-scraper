"""OpenTelemetry initialization and setup for the AI documentation system."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import telemetry_helpers as _telemetry_helpers


try:
    from . import config as observability_config
except ImportError:  # pragma: no cover - optional dependency wiring
    observability_config = None

if TYPE_CHECKING:  # pragma: no cover - typing aids only
    from .config import ObservabilityConfig


logger = logging.getLogger(__name__)


@dataclass
class _ObservabilityState:
    """Mutable module-local state for telemetry providers."""

    tracer_provider: Any | None = None
    meter_provider: Any | None = None


_STATE = _ObservabilityState()


_validate_telemetry_components: Callable[[], bool] | None = getattr(
    _telemetry_helpers, "_validate_telemetry_components", None
)
_create_resource: Callable[[ObservabilityConfig], Any] | None = getattr(
    _telemetry_helpers, "_create_resource", None
)
_initialize_tracing: Callable[[ObservabilityConfig, Any], Any] | None = getattr(
    _telemetry_helpers, "_initialize_tracing", None
)
_initialize_metrics: Callable[[ObservabilityConfig, Any], Any] | None = getattr(
    _telemetry_helpers, "_initialize_metrics", None
)
_setup_fastapi_instrumentation: Callable[[ObservabilityConfig], bool] | None = getattr(
    _telemetry_helpers, "_setup_fastapi_instrumentation", None
)
_setup_httpx_instrumentation: Callable[[ObservabilityConfig], bool] | None = getattr(
    _telemetry_helpers, "_setup_httpx_instrumentation", None
)
_setup_redis_instrumentation: Callable[[ObservabilityConfig], bool] | None = getattr(
    _telemetry_helpers, "_setup_redis_instrumentation", None
)
_setup_sqlalchemy_instrumentation: Callable[[ObservabilityConfig], bool] | None = (
    getattr(_telemetry_helpers, "_setup_sqlalchemy_instrumentation", None)
)


class ObservabilityInitError(RuntimeError):
    """Raised when OpenTelemetry initialization cannot proceed."""


def _resolve_config(
    provided_config: ObservabilityConfig | None,
) -> ObservabilityConfig:
    """Resolve configuration or raise if unavailable."""

    if provided_config is not None:
        return provided_config

    if observability_config is None or not hasattr(
        observability_config, "get_observability_config"
    ):
        msg = "Config system not available"
        logger.error(msg)
        raise ObservabilityInitError(msg)

    return observability_config.get_observability_config()


def _perform_initialization(config: ObservabilityConfig) -> None:
    """Execute initialization steps for observability."""

    if _validate_telemetry_components is None:
        raise ObservabilityInitError("Telemetry validation helper unavailable")

    if not _validate_telemetry_components():
        raise ObservabilityInitError("OpenTelemetry components unavailable")

    if _create_resource is None:
        raise ObservabilityInitError("Resource creation helper unavailable")

    resource = _create_resource(config)
    if resource is None:
        raise ObservabilityInitError("Failed to create OpenTelemetry resource")

    if _initialize_tracing is None:
        raise ObservabilityInitError("Tracing initialization helper unavailable")

    tracer_provider = _initialize_tracing(config, resource)
    if tracer_provider is None:
        raise ObservabilityInitError("Failed to initialize tracing provider")

    trace_api = getattr(_telemetry_helpers, "trace", None)
    provider = None
    if trace_api and hasattr(trace_api, "get_tracer_provider"):
        provider = trace_api.get_tracer_provider()

    _STATE.tracer_provider = provider or tracer_provider

    if _initialize_metrics is None:
        raise ObservabilityInitError("Metrics initialization helper unavailable")

    meter_provider = _initialize_metrics(config, resource)
    if meter_provider is None:
        raise ObservabilityInitError("Failed to initialize meter provider")

    metrics_api = getattr(_telemetry_helpers, "metrics", None)
    provider = None
    if metrics_api and hasattr(metrics_api, "get_meter_provider"):
        provider = metrics_api.get_meter_provider()

    _STATE.meter_provider = provider or meter_provider

    _setup_auto_instrumentation(config)


def initialize_observability(config: ObservabilityConfig | None = None) -> bool:
    """Initialize OpenTelemetry observability infrastructure.

    Sets up distributed tracing, metrics collection, and AI-specific monitoring
    that integrates with the existing function-based service patterns.

    Args:
        config: Optional observability configuration

    Returns:
        True if initialization succeeded, False otherwise

    """
    try:
        resolved_config = _resolve_config(config)
    except ObservabilityInitError:
        return False

    if not resolved_config.enabled:
        logger.info("OpenTelemetry observability disabled by configuration")
        return False

    try:
        _perform_initialization(resolved_config)
    except ObservabilityInitError as exc:
        logger.error("Failed to initialize OpenTelemetry: %s", exc)
        return False
    except ImportError:
        logger.warning("OpenTelemetry packages not available")
        return False
    except (AttributeError, OSError, RuntimeError, ValueError):
        logger.exception("Failed to initialize OpenTelemetry")
        return False

    logger.info(
        "OpenTelemetry initialized successfully - Service: %s, Endpoint: %s",
        resolved_config.service_name,
        resolved_config.otlp_endpoint,
    )
    return True


def _setup_auto_instrumentation(config: ObservabilityConfig) -> None:
    """Setup automatic instrumentation based on configuration.

    Args:
        config: Observability configuration

    """
    try:
        fastapi_setup = _setup_fastapi_instrumentation
        if fastapi_setup and not fastapi_setup(config):
            logger.warning("Auto-instrumentation setup failed: fastapi")

        httpx_setup = _setup_httpx_instrumentation
        if httpx_setup and not httpx_setup(config):
            logger.warning("Auto-instrumentation setup failed: httpx")

        redis_setup = _setup_redis_instrumentation
        if redis_setup and not redis_setup(config):
            logger.warning("Auto-instrumentation setup failed: redis")

        sqlalchemy_setup = _setup_sqlalchemy_instrumentation
        if sqlalchemy_setup and not sqlalchemy_setup(config):
            logger.warning("Auto-instrumentation setup failed: sqlalchemy")

    except (
        ConnectionError,
        ImportError,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as exc:
        logger.warning("Auto-instrumentation setup failed: %s", exc)


def shutdown_observability() -> None:
    """Shutdown OpenTelemetry providers and flush pending data.

    Should be called on application shutdown to ensure all telemetry
    data is properly exported.
    """
    tracer_provider = _STATE.tracer_provider

    if tracer_provider:
        tracer_shutdown_error: str | None = None
        try:
            logger.info("Shutting down OpenTelemetry tracer provider...")
            tracer_provider.shutdown()
        except (OSError, PermissionError) as exc:
            logger.exception("Error during tracer provider shutdown")
            tracer_shutdown_error = str(exc)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - safety net
            logger.exception("Unexpected tracer shutdown failure")
            tracer_shutdown_error = str(exc)
        finally:
            _STATE.tracer_provider = None
            if tracer_shutdown_error:
                logger.error(
                    "Error during tracer provider shutdown: %s", tracer_shutdown_error
                )

    meter_provider = _STATE.meter_provider
    if meter_provider:
        meter_shutdown_error: str | None = None
        try:
            logger.info("Shutting down OpenTelemetry meter provider...")
            meter_provider.shutdown()
        except (OSError, PermissionError) as exc:
            logger.exception("Error during meter provider shutdown")
            meter_shutdown_error = str(exc)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - safety net
            logger.exception("Unexpected meter shutdown failure")
            meter_shutdown_error = str(exc)
        finally:
            _STATE.meter_provider = None
            if meter_shutdown_error:
                logger.error(
                    "Error during meter provider shutdown: %s", meter_shutdown_error
                )

    logger.info("OpenTelemetry shutdown completed")


def is_observability_enabled() -> bool:
    """Check if observability is currently enabled and initialized.

    Returns:
        True if observability is enabled and initialized

    """
    return _STATE.tracer_provider is not None
