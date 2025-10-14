"""Tests for middleware manager helpers."""

from __future__ import annotations

from typing import Any

import pytest
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


def test_apply_defaults_installs_expected_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that apply_defaults installs the expected middleware stack."""
    called = False

    def _fake_setup(app: Starlette, *, include_default: bool = True):
        nonlocal called
        called = True
        return "instrumented"

    monkeypatch.setattr(manager, "setup_prometheus", _fake_setup)
    monkeypatch.setitem(manager._FUNCTION_REGISTRY, "prometheus", _fake_setup)  # type: ignore[attr-defined]
    app = Starlette()

    manager.apply_defaults(app)

    assert called is True
    names = _middleware_class_names(app)
    # Starlette stores middleware in reverse order of addition.
    expected = [
        "PerformanceMiddleware",
        "SlowAPIMiddleware",
        "TimeoutMiddleware",
        "SecurityMiddleware",
    ]

    compression_cls_name = manager.CompressionMiddleware.__name__
    if "brotli" in manager._CLASS_REGISTRY:  # type: ignore[attr-defined]
        compression_cls_name = manager.BrotliCompressionMiddleware.__name__
    expected.append(compression_cls_name)

    if "correlation" in manager._CLASS_REGISTRY:  # type: ignore[attr-defined]
        expected.append("CorrelationIdMiddleware")

    assert names == expected


def test_apply_defaults_sets_limiter_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that apply_defaults sets the limiter state."""
    limiter_instance = object()

    def _fake_enable(app: Starlette, **_: Any):
        app.state.limiter = limiter_instance
        return limiter_instance

    monkeypatch.setattr(manager, "enable_global_rate_limit", _fake_enable)
    monkeypatch.setitem(manager._FUNCTION_REGISTRY, "rate_limiting", _fake_enable)  # type: ignore[attr-defined]
    monkeypatch.setattr(manager, "setup_prometheus", lambda app, **_: None)
    monkeypatch.setitem(manager._FUNCTION_REGISTRY, "prometheus", lambda app, **_: None)  # type: ignore[attr-defined]

    app = Starlette()
    manager.apply_defaults(app)

    assert app.state.limiter is limiter_instance


def test_apply_named_stack_returns_applied_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that apply_named_stack returns applied middleware names."""
    monkeypatch.setattr(manager, "enable_global_rate_limit", lambda app, **_: None)
    monkeypatch.setattr(manager, "setup_prometheus", lambda app, **_: None)
    monkeypatch.setitem(
        manager._FUNCTION_REGISTRY, "rate_limiting", lambda app, **_: None
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(manager._FUNCTION_REGISTRY, "prometheus", lambda app, **_: None)  # type: ignore[attr-defined]

    app = Starlette()
    applied = manager.apply_named_stack(app, ["security", "performance", "unknown"])

    assert applied == ["security", "performance"]
