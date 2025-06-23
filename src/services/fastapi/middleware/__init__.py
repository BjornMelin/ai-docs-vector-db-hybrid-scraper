import typing

"""FastAPI production middleware package.

This package provides a comprehensive set of production-ready middleware
components for the FastMCP server environment.
"""

from .compression import CompressionMiddleware
from .correlation import get_correlation_id
from .performance import EndpointStats
from .performance import PerformanceMiddleware
from .performance import RequestMetrics
from .security import CSRFProtectionMiddleware
from .security import SecurityMiddleware
from .timeout import BulkheadMiddleware
from .timeout import CircuitState
from .timeout import TimeoutMiddleware
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
