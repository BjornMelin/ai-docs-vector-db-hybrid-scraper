"""FastAPI production middleware package.

This package provides a comprehensive set of production-ready middleware
components for the FastMCP server environment.
"""

from .compression import CompressionMiddleware
from .correlation import get_correlation_id
from .performance import EndpointStats, PerformanceMiddleware, RequestMetrics
from .security import CSRFProtectionMiddleware, SecurityMiddleware
from .timeout import BulkheadMiddleware, CircuitState, TimeoutMiddleware
from .tracing import TracingMiddleware


__all__ = [
    "BulkheadMiddleware",
    "CSRFProtectionMiddleware",
    "CircuitState",
    # Compression
    "CompressionMiddleware",
    "EndpointStats",
    # Performance monitoring
    "PerformanceMiddleware",
    "RequestMetrics",
    # Security
    "SecurityMiddleware",
    # Timeout and resilience
    "TimeoutMiddleware",
    # Tracing
    "TracingMiddleware",
    "get_correlation_id",
]
