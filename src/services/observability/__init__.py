"""OpenTelemetry observability foundation for AI documentation vector DB.

This module provides foundational OpenTelemetry integration patterns that
complement the existing function-based service architecture. Provides core
distributed tracing, AI/ML metrics collection, and cost tracking capabilities.
"""

from .config import ObservabilityConfig
from .config import get_observability_config
from .init import initialize_observability
from .init import shutdown_observability
from .middleware import FastAPIObservabilityMiddleware
from .tracking import get_tracer
from .tracking import instrument_function
from .tracking import record_ai_operation
from .tracking import track_cost

__all__ = [
    "FastAPIObservabilityMiddleware",
    "ObservabilityConfig",
    "get_observability_config",
    "get_tracer",
    "initialize_observability",
    "instrument_function",
    "record_ai_operation",
    "shutdown_observability",
    "track_cost",
]
