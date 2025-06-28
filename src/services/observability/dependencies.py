"""FastAPI dependencies for OpenTelemetry observability integration.

Provides dependency injection functions that integrate OpenTelemetry
observability with the existing function-based service architecture.
"""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from .config import ObservabilityConfig, get_observability_config
from .init import initialize_observability, is_observability_enabled
from .tracking import (
    _NoOpMeter,
    _NoOpTracer,
    get_meter,
    get_tracer,
    record_ai_operation,
    track_cost,
)


logger = logging.getLogger(__name__)

# Configuration Dependencies
ObservabilityConfigDep = Annotated[
    ObservabilityConfig, Depends(get_observability_config)
]


@lru_cache
def get_observability_service() -> dict[str, any]:
    """Get observability service instance with singleton pattern.

    Initializes observability if not already done and returns service
    components for dependency injection.

    Returns:
        Dictionary with observability service components

    """
    try:
        config = get_observability_config()

        # Initialize if not already done
        if not is_observability_enabled() and config.enabled:
            initialize_observability(config)

        return {
            "config": config,
            "tracer": get_tracer("ai-docs-service"),
            "meter": get_meter("ai-docs-service"),
            "enabled": is_observability_enabled(),
        }

    except Exception:
        logger.warning("Failed to initialize observability service")
        return {
            "config": ObservabilityConfig(),
            "tracer": None,
            "meter": None,
            "enabled": False,
        }


ObservabilityServiceDep = Annotated[dict, Depends(get_observability_service)]


def get_ai_tracer(
    observability_service: ObservabilityServiceDep,
) -> any:
    """Get tracer for AI operations.

    Args:
        observability_service: Observability service dependency

    Returns:
        OpenTelemetry tracer or NoOp tracer

    """
    if observability_service["enabled"] and observability_service["tracer"]:
        return get_tracer("ai-operations")

    return _NoOpTracer()


AITracerDep = Annotated[any, Depends(get_ai_tracer)]


def get_service_meter(
    observability_service: ObservabilityServiceDep,
) -> any:
    """Get meter for service metrics.

    Args:
        observability_service: Observability service dependency

    Returns:
        OpenTelemetry meter or NoOp meter

    """
    if observability_service["enabled"] and observability_service["meter"]:
        return get_meter("service-metrics")

    return _NoOpMeter()


ServiceMeterDep = Annotated[any, Depends(get_service_meter)]


def create_span_context(
    operation_name: str,
    tracer: AITracerDep,
) -> any:
    """Create a span context for an operation.

    Args:
        operation_name: Name of the operation
        tracer: AI tracer dependency

    Returns:
        Span context manager

    """
    return tracer.start_as_current_span(operation_name)


# Utility functions for common observability patterns
async def record_ai_operation_metrics(
    operation_type: str,
    provider: str,
    success: bool,
    duration: float,
    _meter: ServiceMeterDep,
    **kwargs,
) -> None:
    """Record metrics for AI operations using dependency injection.

    Args:
        operation_type: Type of AI operation
        provider: AI service provider
        success: Whether operation succeeded
        duration: Operation duration in seconds
        meter: Service meter dependency
        **kwargs: Additional attributes

    """
    try:
        record_ai_operation(
            operation_type=operation_type,
            provider=provider,
            success=success,
            duration=duration,
            **kwargs,
        )

    except Exception:
        logger.debug("Failed to record AI operation metrics")


async def track_ai_cost_metrics(
    operation_type: str,
    provider: str,
    cost_usd: float,
    _meter: ServiceMeterDep,
    **kwargs,
) -> None:
    """Track AI operation costs using dependency injection.

    Args:
        operation_type: Type of AI operation
        provider: AI service provider
        cost_usd: Cost in USD
        meter: Service meter dependency
        **kwargs: Additional attributes

    """
    try:
        track_cost(
            operation_type=operation_type,
            provider=provider,
            cost_usd=cost_usd,
            **kwargs,
        )

    except Exception:
        logger.debug("Failed to track AI cost metrics")


# Health check for observability
async def get_observability_health(
    observability_service: ObservabilityServiceDep,
) -> dict[str, any]:
    """Get observability health status.

    Args:
        observability_service: Observability service dependency

    Returns:
        Health status dictionary

    """
    try:
        config = observability_service["config"]
        enabled = observability_service["enabled"]

        health = {
            "enabled": enabled,
            "service_name": config.service_name,
            "otlp_endpoint": config.otlp_endpoint if enabled else None,
            "instrumentation": {
                "fastapi": config.instrument_fastapi if enabled else False,
                "httpx": config.instrument_httpx if enabled else False,
                "redis": config.instrument_redis if enabled else False,
                "sqlalchemy": config.instrument_sqlalchemy if enabled else False,
            },
            "ai_tracking": {
                "operations": config.track_ai_operations if enabled else False,
                "costs": config.track_costs if enabled else False,
            },
            "status": "healthy" if enabled else "disabled",
        }

        return health

    except Exception as e:
        logger.exception("Failed to get observability health")
        return {
            "enabled": False,
            "status": "error",
            "error": str(e),
        }