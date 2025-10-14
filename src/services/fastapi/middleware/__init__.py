"""Production middleware package, consolidated and library-first."""

from .compression import BrotliCompressionMiddleware, CompressionMiddleware
from .correlation import generate_correlation_id, get_correlation_id, set_correlation_id
from .manager import MiddlewareSpec, apply_defaults
from .performance import PerformanceMiddleware, setup_prometheus
from .security import SecurityMiddleware, enable_global_rate_limit
from .timeout import BulkheadMiddleware, CircuitState, TimeoutConfig, TimeoutMiddleware
from .tracing import TracingMiddleware


__all__ = [
    "BrotliCompressionMiddleware",
    "BulkheadMiddleware",
    "CircuitState",
    "CompressionMiddleware",
    "MiddlewareSpec",
    "PerformanceMiddleware",
    "SecurityMiddleware",
    "TimeoutConfig",
    "TimeoutMiddleware",
    "TracingMiddleware",
    "apply_defaults",
    "enable_global_rate_limit",
    "generate_correlation_id",
    "get_correlation_id",
    "set_correlation_id",
    "setup_prometheus",
]
