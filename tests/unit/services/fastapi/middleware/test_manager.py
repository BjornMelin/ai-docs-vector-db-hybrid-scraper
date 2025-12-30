"""Tests for middleware manager helpers."""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette

from src.services.fastapi.middleware import manager


def _middleware_class_names(app: Starlette) -> list[str]:
    """Extract class names from Starlette middleware."""
    names: list[str] = []
    for middleware in app.user_middleware:
        cls_name = getattr(
            middleware.cls,
            "__name__",
            middleware.cls.__class__.__name__,
        )
        names.append(cls_name)
    return names


def test_apply_defaults_installs_expected_stack() -> None:
    """Test that apply_defaults installs the expected middleware stack."""
    called = False

    def _fake_setup(app: Starlette, *, include_default: bool = True):
        nonlocal called
        called = True
        return "instrumented"

    app = Starlette()

    with manager.override_registry(functions={"prometheus": _fake_setup}):
        manager.apply_defaults(app)

    assert called
    names = _middleware_class_names(app)
    # Starlette stores middleware in reverse order of addition.
    compression_candidates = {
        manager.CompressionMiddleware.__name__,
        manager.BrotliCompressionMiddleware.__name__,
    }
    assert "PerformanceMiddleware" in names
    assert "SlowAPIMiddleware" in names
    assert "TimeoutMiddleware" in names
    assert "SecurityMiddleware" in names
    assert any(name in compression_candidates for name in names)

    if manager.is_registered("correlation"):
        assert "CorrelationIdMiddleware" in names


def test_apply_defaults_sets_limiter_state() -> None:
    """Test that apply_defaults sets the limiter state."""
    limiter_instance = object()

    def _fake_enable(app: Starlette, **_: Any):
        app.state.limiter = limiter_instance
        return limiter_instance

    app = Starlette()
    with manager.override_registry(
        functions={
            "rate_limiting": _fake_enable,
            "prometheus": lambda app, **_: None,
        }
    ):
        manager.apply_defaults(app)

    assert app.state.limiter is limiter_instance


def test_apply_named_stack_returns_applied_names() -> None:
    """Test that apply_named_stack returns applied middleware names."""
    app = Starlette()
    with manager.override_registry(
        functions={
            "rate_limiting": lambda app, **_: None,
            "prometheus": lambda app, **_: None,
        }
    ):
        applied = manager.apply_named_stack(app, ["security", "performance", "unknown"])

    assert applied == ["security", "performance"]
