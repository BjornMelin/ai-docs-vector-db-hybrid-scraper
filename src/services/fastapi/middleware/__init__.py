"""
Production middleware package, consolidated and library-first.
"""

from .compression import BrotliCompressionMiddleware, CompressionMiddleware
from .correlation import generate_correlation_id, get_correlation_id, set_correlation_id
from .manager import MiddlewareSpec, apply_defaults
from .performance import PerformanceMiddleware, setup_prometheus
from .security import SecurityMiddleware, enable_global_rate_limit
from .timeout import BulkheadMiddleware, CircuitState, TimeoutConfig, TimeoutMiddleware
from .tracing import TracingMiddleware


__all__ = [
    "apply_defaults",
    "MiddlewareSpec",
    "CompressionMiddleware",
    "BrotliCompressionMiddleware",
    "SecurityMiddleware",
    "enable_global_rate_limit",
    "TimeoutMiddleware",
    "BulkheadMiddleware",
    "TimeoutConfig",
    "CircuitState",
    "PerformanceMiddleware",
    "setup_prometheus",
    "TracingMiddleware",
    "get_correlation_id",
    "set_correlation_id",
    "generate_correlation_id",
]
