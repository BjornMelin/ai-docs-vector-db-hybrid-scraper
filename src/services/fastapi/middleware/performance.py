"""Performance monitoring middleware for production metrics and observability.

This middleware provides comprehensive performance monitoring including response times,
memory usage tracking, and slow request identification for production environments.
"""

import gc
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

import psutil
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.config import PerformanceConfig


logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Individual request performance metrics."""

    endpoint: str
    method: str
    status_code: int
    response_time: float
    memory_before: float | None = None
    memory_after: float | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EndpointStats:
    """Aggregated statistics for an endpoint."""

    total_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    error_count: int = 0
    slow_request_count: int = 0
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return (
            self.total_response_time / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return (
            (self.error_count / self.total_requests * 100)
            if self.total_requests > 0
            else 0.0
        )

    @property
    def slow_request_rate(self) -> float:
        """Calculate slow request rate percentage."""
        return (
            (self.slow_request_count / self.total_requests * 100)
            if self.total_requests > 0
            else 0.0
        )


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware with metrics collection.

    Features:
    - Response time tracking with percentiles
    - Memory usage monitoring (optional)
    - Slow request identification and logging
    - Per-endpoint statistics aggregation
    - Thread-safe metrics collection
    """

    def __init__(self, app: Callable, config: PerformanceConfig):
        """Initialize performance middleware.

        Args:
            app: ASGI application
            config: Performance configuration
        """
        super().__init__(app)
        self.config = config

        # Thread-safe metrics storage
        self._stats_lock = threading.Lock()
        self._endpoint_stats: dict[str, EndpointStats] = defaultdict(EndpointStats)
        self._recent_metrics: deque = deque(maxlen=1000)

        # Memory monitoring setup
        self._process = psutil.Process() if config.track_memory_usage else None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        if not self.config.enabled:
            return await call_next(request)

        endpoint = self._get_endpoint_key(request)

        # Start monitoring
        start_time = time.perf_counter()
        memory_before = (
            self._get_memory_usage() if self.config.track_memory_usage else None
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate metrics
            end_time = time.perf_counter()
            response_time = end_time - start_time
            memory_after = (
                self._get_memory_usage() if self.config.track_memory_usage else None
            )

            # Create metrics record
            metrics = RequestMetrics(
                endpoint=endpoint,
                method=request.method,
                status_code=response.status_code,
                response_time=response_time,
                memory_before=memory_before,
                memory_after=memory_after,
            )

            # Record metrics
            self._record_metrics(metrics)

            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.4f}"
            if memory_after and memory_before:
                memory_delta = memory_after - memory_before
                response.headers["X-Memory-Delta"] = f"{memory_delta:.2f}"

            return response

        except Exception:
            # Record failed request metrics
            end_time = time.perf_counter()
            response_time = end_time - start_time

            metrics = RequestMetrics(
                endpoint=endpoint,
                method=request.method,
                status_code=500,  # Assume 500 for exceptions
                response_time=response_time,
                memory_before=memory_before,
            )

            self._record_metrics(metrics)

            # Re-raise the exception
            raise

    def _get_endpoint_key(self, request: Request) -> str:
        """Generate endpoint key for metrics grouping.

        Args:
            request: HTTP request

        Returns:
            Endpoint identifier string
        """
        return f"{request.method}:{request.url.path}"

    def _get_memory_usage(self) -> float | None:
        """Get current memory usage in MB.

        Returns:
            Memory usage in MB or None if not available
        """
        try:
            if self._process:
                memory_info = self._process.memory_info()
                return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            logger.warning(f"Failed to get memory usage: {e}")
        return None

    def _record_metrics(self, metrics: RequestMetrics) -> None:
        """Record metrics in thread-safe manner.

        Args:
            metrics: Request metrics to record
        """
        with self._stats_lock:
            # Update endpoint statistics
            stats = self._endpoint_stats[metrics.endpoint]
            stats.total_requests += 1
            stats.total_response_time += metrics.response_time
            stats.min_response_time = min(
                stats.min_response_time, metrics.response_time
            )
            stats.max_response_time = max(
                stats.max_response_time, metrics.response_time
            )
            stats.recent_requests.append(metrics)

            # Count errors (4xx and 5xx)
            if metrics.status_code >= 400:
                stats.error_count += 1

            # Count slow requests
            if metrics.response_time > self.config.slow_request_threshold:
                stats.slow_request_count += 1

                # Log slow requests
                logger.warning(
                    "Slow request detected",
                    extra={
                        "endpoint": metrics.endpoint,
                        "method": metrics.method,
                        "response_time": metrics.response_time,
                        "threshold": self.config.slow_request_threshold,
                        "status_code": metrics.status_code,
                    },
                )

            # Store in recent metrics
            self._recent_metrics.append(metrics)

    def get_metrics_summary(self) -> dict:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary containing all performance metrics
        """
        with self._stats_lock:
            summary = {
                "total_requests": sum(
                    stats.total_requests for stats in self._endpoint_stats.values()
                ),
                "total_errors": sum(
                    stats.error_count for stats in self._endpoint_stats.values()
                ),
                "total_slow_requests": sum(
                    stats.slow_request_count for stats in self._endpoint_stats.values()
                ),
                "endpoints": {},
                "system": {},
            }

            # Per-endpoint metrics
            for endpoint, stats in self._endpoint_stats.items():
                summary["endpoints"][endpoint] = {
                    "total_requests": stats.total_requests,
                    "avg_response_time": stats.avg_response_time,
                    "min_response_time": stats.min_response_time
                    if stats.min_response_time != float("inf")
                    else 0,
                    "max_response_time": stats.max_response_time,
                    "error_rate": stats.error_rate,
                    "slow_request_rate": stats.slow_request_rate,
                }

            # System metrics
            if self.config.track_memory_usage and self._process:
                try:
                    memory_info = self._process.memory_info()
                    cpu_percent = self._process.cpu_percent()

                    summary["system"] = {
                        "memory_usage_mb": memory_info.rss / 1024 / 1024,
                        "memory_usage_percent": self._process.memory_percent(),
                        "cpu_percent": cpu_percent,
                    }
                except Exception:
                    logger.warning(f"Failed to get system metrics: {e}")

            return summary

    def get_endpoint_stats(self, endpoint: str) -> dict | None:
        """Get statistics for a specific endpoint.

        Args:
            endpoint: Endpoint identifier

        Returns:
            Endpoint statistics or None if not found
        """
        with self._stats_lock:
            if endpoint in self._endpoint_stats:
                stats = self._endpoint_stats[endpoint]
                return {
                    "total_requests": stats.total_requests,
                    "avg_response_time": stats.avg_response_time,
                    "min_response_time": stats.min_response_time
                    if stats.min_response_time != float("inf")
                    else 0,
                    "max_response_time": stats.max_response_time,
                    "error_count": stats.error_count,
                    "error_rate": stats.error_rate,
                    "slow_request_count": stats.slow_request_count,
                    "slow_request_rate": stats.slow_request_rate,
                }
        return None

    def get_recent_metrics(self, limit: int = 100) -> list[dict]:
        """Get recent request metrics.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of recent request metrics
        """
        with self._stats_lock:
            recent = list(self._recent_metrics)[-limit:]
            return [
                {
                    "endpoint": m.endpoint,
                    "method": m.method,
                    "status_code": m.status_code,
                    "response_time": m.response_time,
                    "memory_before": m.memory_before,
                    "memory_after": m.memory_after,
                    "timestamp": m.timestamp,
                }
                for m in recent
            ]

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._stats_lock:
            self._endpoint_stats.clear()
            self._recent_metrics.clear()

        # Force garbage collection if memory tracking is enabled
        if self.config.track_memory_usage:
            gc.collect()

    def get_health_status(self) -> dict:
        """Get overall health status based on metrics.

        Returns:
            Health status with recommendations
        """
        summary = self.get_metrics_summary()

        health = {"status": "healthy", "warnings": [], "metrics": summary}

        # Check error rates
        if summary["total_requests"] > 0:
            overall_error_rate = (
                summary["total_errors"] / summary["total_requests"]
            ) * 100
            if overall_error_rate > 10:
                health["status"] = "unhealthy"
                health["warnings"].append(f"High error rate: {overall_error_rate:.1f}%")
            elif overall_error_rate > 5:
                health["status"] = "degraded"
                health["warnings"].append(
                    f"Elevated error rate: {overall_error_rate:.1f}%"
                )

        # Check slow request rates
        if summary["total_requests"] > 0:
            slow_request_rate = (
                summary["total_slow_requests"] / summary["total_requests"]
            ) * 100
            if slow_request_rate > 20:
                health["status"] = "degraded"
                health["warnings"].append(
                    f"High slow request rate: {slow_request_rate:.1f}%"
                )

        # Check memory usage
        if "system" in summary and "memory_usage_percent" in summary["system"]:
            memory_percent = summary["system"]["memory_usage_percent"]
            if memory_percent > 90:
                health["status"] = "unhealthy"
                health["warnings"].append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                health["status"] = "degraded"
                health["warnings"].append(
                    f"Elevated memory usage: {memory_percent:.1f}%"
                )

        return health


# Export middleware class
__all__ = ["EndpointStats", "PerformanceMiddleware", "RequestMetrics"]
