"""Function-based monitoring with dependency injection.

Simplified monitoring functions that replace complex monitoring service classes.
Provides health checks, metrics collection, and system status with dependency injection.
"""

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import Depends, HTTPException

from src.config import Config

from .circuit_breaker import CircuitBreakerConfig, circuit_breaker
from .dependencies import get_config


logger = logging.getLogger(__name__)


# Simple in-memory metrics storage
_metrics: dict[str, Any] = {
    "counters": {},
    "gauges": {},
    "timers": {},
    "health_checks": {},
}
_metrics_lock = asyncio.Lock()


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def increment_counter(
    metric_name: str,
    value: int = 1,
    tags: dict[str, str] | None = None,
) -> None:
    """Increment a counter metric.

    Pure function replacement for complex metrics services.

    Args:
        metric_name: Name of the counter metric
        value: Value to increment by (default: 1)
        tags: Optional tags for the metric
    """
    async with _metrics_lock:
        key = f"{metric_name}:{','.join(f'{k}={v}' for k, v in (tags or {}).items())}"
        _metrics["counters"][key] = _metrics["counters"].get(key, 0) + value
        logger.debug(f"Incremented counter {key} by {value}")


async def set_gauge(
    metric_name: str,
    value: float,
    tags: dict[str, str] | None = None,
) -> None:
    """Set a gauge metric value.

    Args:
        metric_name: Name of the gauge metric
        value: Value to set
        tags: Optional tags for the metric
    """
    async with _metrics_lock:
        key = f"{metric_name}:{','.join(f'{k}={v}' for k, v in (tags or {}).items())}"
        _metrics["gauges"][key] = {
            "value": value,
            "timestamp": time.time(),
        }
        logger.debug(f"Set gauge {key} to {value}")


async def record_timer(
    metric_name: str,
    duration_ms: float,
    tags: dict[str, str] | None = None,
) -> None:
    """Record a timer metric.

    Args:
        metric_name: Name of the timer metric
        duration_ms: Duration in milliseconds
        tags: Optional tags for the metric
    """
    async with _metrics_lock:
        key = f"{metric_name}:{','.join(f'{k}={v}' for k, v in (tags or {}).items())}"
        if key not in _metrics["timers"]:
            _metrics["timers"][key] = []

        _metrics["timers"][key].append(
            {
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
        )

        # Keep only last 1000 measurements
        if len(_metrics["timers"][key]) > 1000:
            _metrics["timers"][key] = _metrics["timers"][key][-1000:]

        logger.debug(f"Recorded timer {key}: {duration_ms}ms")


async def get_metrics_summary() -> dict[str, Any]:
    """Get summary of all metrics.

    Returns:
        Dictionary containing metrics summary
    """
    async with _metrics_lock:
        summary = {
            "counters": dict(_metrics["counters"]),
            "gauges": {},
            "timers": {},
            "timestamp": time.time(),
        }

        # Process gauges
        for key, data in _metrics["gauges"].items():
            summary["gauges"][key] = data["value"]

        # Process timers with statistics
        for key, measurements in _metrics["timers"].items():
            if measurements:
                durations = [m["duration_ms"] for m in measurements]
                summary["timers"][key] = {
                    "count": len(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                }

        return summary


async def check_service_health(
    service_name: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Check health of a specific service.

    Pure function replacement for health check services.

    Args:
        service_name: Name of service to check
        config: Injected configuration

    Returns:
        Health check result
    """
    try:
        health_result = {
            "service": service_name,
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {},
        }

        # Basic service checks based on service name
        if service_name == "vector_db":
            # Check Qdrant connection
            try:
                # This would be replaced with actual Qdrant health check
                health_result["checks"]["qdrant_connection"] = "healthy"
                health_result["checks"]["collection_count"] = "unknown"
            except Exception as e:
                health_result["status"] = "unhealthy"
                health_result["checks"]["qdrant_connection"] = f"error: {e}"

        elif service_name == "cache":
            # Check cache connection
            try:
                # This would be replaced with actual cache health check
                health_result["checks"]["dragonfly_connection"] = "healthy"
                health_result["checks"]["memory_usage"] = "unknown"
            except Exception as e:
                health_result["status"] = "unhealthy"
                health_result["checks"]["dragonfly_connection"] = f"error: {e}"

        elif service_name == "embeddings":
            # Check embedding providers
            health_result["checks"]["openai_available"] = "unknown"
            health_result["checks"]["fastembed_available"] = "unknown"

        else:
            health_result["checks"]["generic"] = "healthy"

        # Store health check result
        async with _metrics_lock:
            _metrics["health_checks"][service_name] = health_result

        logger.debug(f"Health check for {service_name}: {health_result['status']}")
        return health_result

    except Exception as e:
        logger.exception(f"Health check failed for {service_name}: {e}")
        error_result = {
            "service": service_name,
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
        }

        async with _metrics_lock:
            _metrics["health_checks"][service_name] = error_result

        return error_result


async def get_system_status(
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Get overall system health status.

    Returns:
        System status summary
    """
    try:
        # Run health checks for key services
        services = ["vector_db", "cache", "embeddings", "crawling"]
        health_checks = {}

        for service in services:
            health_checks[service] = await check_service_health(service, config)

        # Determine overall status
        overall_status = "healthy"
        unhealthy_services = [
            name
            for name, check in health_checks.items()
            if check["status"] != "healthy"
        ]

        if unhealthy_services:
            overall_status = "degraded" if len(unhealthy_services) <= 1 else "unhealthy"

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "services": health_checks,
            "unhealthy_services": unhealthy_services,
            "metrics_summary": await get_metrics_summary(),
        }

    except Exception as e:
        logger.exception(f"System status check failed: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e),
        }


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, metric_name: str, tags: dict[str, str] | None = None):
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            await record_timer(self.metric_name, duration_ms, self.tags)


def timed(metric_name: str, tags: dict[str, str] | None = None):
    """Decorator for timing function execution.

    Usage:
        @timed("embedding_generation")
        async def generate_embeddings(...):
            # Function execution time automatically recorded
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with TimerContext(metric_name, tags):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


async def log_api_call(
    provider: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    request_size: int | None = None,
    response_size: int | None = None,
) -> None:
    """Log API call metrics.

    Simplified function for tracking API usage across providers.

    Args:
        provider: API provider name
        endpoint: API endpoint
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        request_size: Optional request size in bytes
        response_size: Optional response size in bytes
    """
    tags = {
        "provider": provider,
        "endpoint": endpoint,
        "status_code": str(status_code),
    }

    # Log basic metrics
    await increment_counter("api_calls_total", tags=tags)
    await record_timer("api_call_duration", duration_ms, tags=tags)

    if request_size:
        await record_timer("api_request_size", request_size, tags=tags)

    if response_size:
        await record_timer("api_response_size", response_size, tags=tags)

    # Log errors separately
    if status_code >= 400:
        await increment_counter("api_errors_total", tags=tags)

    logger.info(
        f"API call: {provider}/{endpoint} -> {status_code} ({duration_ms:.1f}ms)"
    )


async def get_performance_report() -> dict[str, Any]:
    """Generate performance report from collected metrics.

    Returns:
        Performance analysis report
    """
    try:
        metrics = await get_metrics_summary()

        report = {
            "timestamp": time.time(),
            "summary": {
                "total_api_calls": 0,
                "total_errors": 0,
                "average_response_time": 0,
            },
            "by_provider": {},
            "slowest_endpoints": [],
            "error_rates": {},
        }

        # Analyze API call metrics
        api_calls = [k for k in metrics["counters"] if "api_calls_total" in k]
        api_errors = [k for k in metrics["counters"] if "api_errors_total" in k]

        report["summary"]["total_api_calls"] = sum(
            metrics["counters"][k] for k in api_calls
        )
        report["summary"]["total_errors"] = sum(
            metrics["counters"][k] for k in api_errors
        )

        # Calculate average response time
        duration_timers = [
            k for k in metrics["timers"] if "api_call_duration" in k
        ]
        if duration_timers:
            total_duration = sum(
                metrics["timers"][k]["avg_ms"] * metrics["timers"][k]["count"]
                for k in duration_timers
            )
            total_calls = sum(metrics["timers"][k]["count"] for k in duration_timers)
            if total_calls > 0:
                report["summary"]["average_response_time"] = (
                    total_duration / total_calls
                )

        logger.debug("Generated performance report")
        return report

    except Exception as e:
        logger.exception(f"Performance report generation failed: {e}")
        return {
            "timestamp": time.time(),
            "error": str(e),
        }
