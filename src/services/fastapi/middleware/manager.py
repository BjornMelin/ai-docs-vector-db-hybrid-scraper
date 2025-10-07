"""
Middleware manager.

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

from collections.abc import Iterable
from dataclasses import dataclass, field
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
    """Simple spec mirroring Starlette's add_middleware signature."""

    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)


_CLASS_REGISTRY: dict[str, MiddlewareSpec] = {
    "correlation": MiddlewareSpec(
        CorrelationIdMiddleware, {"header_name": "X-Request-ID"}
    ),
    "compression": MiddlewareSpec(CompressionMiddleware, {"minimum_size": 500}),
    "brotli": MiddlewareSpec(BrotliCompressionMiddleware, {"quality": 4}),
    "security": MiddlewareSpec(SecurityMiddleware),
    "timeout": MiddlewareSpec(TimeoutMiddleware, {"config": TimeoutConfig()}),
    "performance": MiddlewareSpec(PerformanceMiddleware),
}

_FUNCTION_REGISTRY: dict[str, Any] = {
    "rate_limiting": enable_global_rate_limit,
    "prometheus": setup_prometheus,
}

_DEFAULT_SEQUENCE: tuple[str, ...] = (
    "correlation",
    "compression",
    "brotli",
    "security",
    "timeout",
    "rate_limiting",
    "performance",
    "prometheus",
)


def apply_named_stack(app: Starlette, middleware_names: Iterable[str]) -> list[str]:
    """Apply middleware/components by registry name.

    Args:
        app: Target Starlette/FastAPI application.
        middleware_names: Iterable of registry keys. The special name
            ``"defaults"`` expands to the curated production sequence.

    Returns:
        List of successfully applied middleware identifiers.
    """

    applied: list[str] = []
    for name in middleware_names:
        if name == "defaults":
            applied.extend(apply_named_stack(app, _DEFAULT_SEQUENCE))
            continue

        spec = _CLASS_REGISTRY.get(name)
        if spec:
            app.add_middleware(spec.cls, **spec.kwargs)
            applied.append(name)
            continue

        func = _FUNCTION_REGISTRY.get(name)
        if func:
            func(app)
            applied.append(name)
    return applied


def apply_defaults(app: Starlette) -> None:
    """Apply a sensible production stack with minimal knobs."""

    apply_named_stack(app, _DEFAULT_SEQUENCE)


__all__ = [
    "MiddlewareSpec",
    "apply_defaults",
    "apply_named_stack",
    "get_correlation_id",
    "PerformanceMiddleware",
    "TimeoutMiddleware",
    "SecurityMiddleware",
    "CompressionMiddleware",
    "BrotliCompressionMiddleware",
]
