"""Tests for the FastAPI middleware manager."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware

from src.services.fastapi.middleware.manager import MiddlewareManager


@pytest.fixture(name="base_config")
def fixture_base_config():
    """Return a minimal configuration namespace for the manager."""

    security = SimpleNamespace(enable_rate_limiting=True)
    performance = SimpleNamespace(
        request_timeout=12.5,
        max_retries=2,
        retry_base_delay=0.5,
    )
    return SimpleNamespace(security=security, performance=performance)


def test_get_middleware_stack_orders_security_timeout_performance(base_config):
    """Security, timeout, performance middleware should appear in that order."""

    manager = MiddlewareManager(config=base_config)
    specs = manager.get_middleware_stack()

    assert len(specs) == 3
    class_names = [spec.cls.__name__ for spec in specs]
    assert class_names == [
        "SecurityMiddleware",
        "TimeoutMiddleware",
        "PerformanceMiddleware",
    ]


def test_apply_middleware_respects_requested_names(base_config):
    """apply_middleware should register only the requested middleware in order."""

    manager = MiddlewareManager(config=base_config)
    app = Starlette()

    manager.apply_middleware(app, ["timeout", "performance"])

    # Starlette stores middleware definitions in user_middleware in insertion order.
    middlewares = cast(list[Middleware], app.user_middleware)
    names = [cast(type[Any], middleware.cls).__name__ for middleware in middlewares]
    # Starlette inserts each middleware at the front of the list, so the most recently
    # added entry appears first. Reverse the expected order to match this behavior.
    assert names == ["PerformanceMiddleware", "TimeoutMiddleware"]


def test_apply_middleware_ignores_unknown_names(base_config):
    """Unknown middleware identifiers should be ignored without failure."""

    manager = MiddlewareManager(config=base_config)
    app = Starlette()

    manager.apply_middleware(app, ["unknown", "timeout"])

    middlewares = cast(list[Middleware], app.user_middleware)
    names = [cast(type[Any], middleware.cls).__name__ for middleware in middlewares]
    assert names == ["TimeoutMiddleware"]
