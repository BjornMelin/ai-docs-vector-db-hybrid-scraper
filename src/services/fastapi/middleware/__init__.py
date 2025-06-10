"""FastAPI production middleware package.

This package provides a comprehensive set of production-ready middleware
components for the FastMCP server environment.
"""

from .compression import CompressionMiddleware
from .performance import EndpointStats
from .performance import PerformanceMiddleware
from .performance import RequestMetrics
from .security import CSRFProtectionMiddleware
from .security import SecurityMiddleware
from .timeout import BulkheadMiddleware
from .timeout import CircuitState
from .timeout import TimeoutMiddleware
from .tracing import TracingMiddleware
from .tracing import get_correlation_id

__all__ = [
    # Compression
    "CompressionMiddleware",
    # Performance monitoring
    "PerformanceMiddleware",
    "RequestMetrics",
    "EndpointStats",
    # Security
    "SecurityMiddleware",
    "CSRFProtectionMiddleware",
    # Timeout and resilience
    "TimeoutMiddleware",
    "BulkheadMiddleware",
    "CircuitState",
    # Tracing
    "TracingMiddleware",
    "get_correlation_id",
]
