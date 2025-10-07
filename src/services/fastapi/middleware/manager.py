"""
Minimal middleware manager that wires library-first components.

Order:
1) Trusted host / CORS (configure in app factory if needed)
2) Correlation ID (asgi-correlation-id)
3) Compression (GZip / optional Brotli)
4) Security headers
5) Timeout + circuit breaker
6) Rate limit (slowapi)
7) Performance header
8) OTel + Prometheus instrumentation via helpers

This keeps the public surface tiny and library-focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asgi_correlation_id import CorrelationIdMiddleware  # type: ignore
from starlette.applications import Starlette

from .compression import BrotliCompressionMiddleware, CompressionMiddleware
from .correlation import get_correlation_id
from .performance import PerformanceMiddleware, setup_prometheus
from .security import SecurityMiddleware, enable_global_rate_limit
from .timeout import TimeoutConfig, TimeoutMiddleware


@dataclass(slots=True)
class MiddlewareSpec:
    """Simple spec that mirrors Starlette's add_middleware signature."""

    cls: type
    kwargs: dict[str, Any]


def apply_defaults(app: Starlette) -> None:
    """Apply a sensible production stack with minimal knobs."""
    # Trusted hosts: users should set allowed list at app construction time.
    # Example:
    # app.add_middleware(TrustedHostMiddleware, allowed_hosts=["example.com", "*.example.com"])  # noqa: E501

    app.add_middleware(CorrelationIdMiddleware, header_name="X-Request-ID")
    app.add_middleware(CompressionMiddleware, minimum_size=500)
    # If brotli is installed, stack it first; gzip will serve as fallback.
    app.add_middleware(BrotliCompressionMiddleware, quality=4)

    app.add_middleware(SecurityMiddleware)
    app.add_middleware(TimeoutMiddleware, config=TimeoutConfig())

    # Rate limiting (Redis optional)
    enable_global_rate_limit(app)

    # Headers + /metrics
    app.add_middleware(PerformanceMiddleware)
    setup_prometheus(app)


__all__ = [
    "MiddlewareSpec",
    "apply_defaults",
    "get_correlation_id",
    "PerformanceMiddleware",
    "TimeoutMiddleware",
    "SecurityMiddleware",
    "CompressionMiddleware",
    "BrotliCompressionMiddleware",
]
