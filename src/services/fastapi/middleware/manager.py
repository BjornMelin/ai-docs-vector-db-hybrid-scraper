"""Middleware manager.

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

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


try:  # Optional dependency
    from asgi_correlation_id import CorrelationIdMiddleware  # type: ignore
except ImportError:  # pragma: no cover - optional middleware
    CorrelationIdMiddleware = None  # type: ignore[assignment]
from starlette.applications import Starlette

from src.services.circuit_breaker.provider import get_circuit_breaker_manager

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


_CLASS_REGISTRY: dict[str, MiddlewareSpec] = {}

if CorrelationIdMiddleware is not None:
    _CLASS_REGISTRY["correlation"] = MiddlewareSpec(
        CorrelationIdMiddleware, {"header_name": "X-Request-ID"}
    )

_CLASS_REGISTRY["compression"] = MiddlewareSpec(
    CompressionMiddleware, {"minimum_size": 500}
)

if BrotliCompressionMiddleware is not CompressionMiddleware:
    _CLASS_REGISTRY["brotli"] = MiddlewareSpec(
        BrotliCompressionMiddleware, {"quality": 4}
    )

_CLASS_REGISTRY["security"] = MiddlewareSpec(SecurityMiddleware)
_CLASS_REGISTRY["timeout"] = MiddlewareSpec(
    TimeoutMiddleware,
    {
        "config": TimeoutConfig(),
        "manager_resolver": get_circuit_breaker_manager,
    },
)
_CLASS_REGISTRY["performance"] = MiddlewareSpec(PerformanceMiddleware)

_FUNCTION_REGISTRY: dict[str, Callable[..., Any]] = {
    "rate_limiting": enable_global_rate_limit,
    "prometheus": setup_prometheus,
}

@contextmanager
def override_registry(
    *,
    classes: dict[str, MiddlewareSpec] | None = None,
    functions: dict[str, Callable[..., Any]] | None = None,
) -> Iterator[None]:
    """Temporarily override registry entries and restore them afterwards.

    Args:
        classes: Mapping of middleware identifiers to replacement specs.
        functions: Mapping of helper identifiers to replacement callables.
    """
    class_overrides = classes or {}
    function_overrides = functions or {}
    class_snapshot: dict[str, MiddlewareSpec | None] = {
        name: _CLASS_REGISTRY.get(name) for name in class_overrides
    }
    function_snapshot: dict[str, Callable[..., Any] | None] = {
        name: _FUNCTION_REGISTRY.get(name) for name in function_overrides
    }
    try:
        _CLASS_REGISTRY.update(class_overrides)
        _FUNCTION_REGISTRY.update(function_overrides)
        yield
    finally:
        for name, previous in class_snapshot.items():
            if previous is None:
                _CLASS_REGISTRY.pop(name, None)
            else:
                _CLASS_REGISTRY[name] = previous
        for name, previous in function_snapshot.items():
            if previous is None:
                _FUNCTION_REGISTRY.pop(name, None)
            else:
                _FUNCTION_REGISTRY[name] = previous


def is_registered(name: str) -> bool:
    """Return ``True`` when a middleware or helper is registered under ``name``."""
    return name in _CLASS_REGISTRY or name in _FUNCTION_REGISTRY


def _default_stack_names() -> tuple[str, ...]:
    """Return the canonical middleware ordering for the defaults alias."""
    names: list[str] = []
    if "correlation" in _CLASS_REGISTRY:
        names.append("correlation")

    compression_key = "brotli" if "brotli" in _CLASS_REGISTRY else "compression"
    names.append(compression_key)

    names.extend(
        name
        for name in (
            "security",
            "timeout",
            "rate_limiting",
            "performance",
            "prometheus",
        )
        if name in _CLASS_REGISTRY or name in _FUNCTION_REGISTRY
    )
    return tuple(names)


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
            applied.extend(apply_named_stack(app, _default_stack_names()))
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
    apply_named_stack(app, _default_stack_names())


__all__ = [
    "CompressionMiddleware",
    "MiddlewareSpec",
    "PerformanceMiddleware",
    "SecurityMiddleware",
    "TimeoutMiddleware",
    "apply_defaults",
    "apply_named_stack",
    "get_correlation_id",
    "is_registered",
    "override_registry",
]

if "brotli" in _CLASS_REGISTRY:
    __all__.append("BrotliCompressionMiddleware")
