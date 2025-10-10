"""Unit tests for FastAPI application factory profiles."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import app_factory
from src.api.app_profiles import AppProfile
from src.architecture.modes import ApplicationMode
from src.services.fastapi.dependencies import get_health_checker


@pytest.fixture(autouse=True)
def _patch_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub container-managed lifespan to avoid touching real services."""

    @asynccontextmanager
    async def _noop_lifespan(_: FastAPI):
        yield

    async def _noop_initialize(_: Any) -> None:
        return None

    monkeypatch.setattr(app_factory, "container_lifespan", _noop_lifespan)
    monkeypatch.setattr(app_factory, "_initialize_mode_services", _noop_initialize)


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Create an app with a stubbed health checker dependency."""

    app_instance = app_factory.create_app(AppProfile.SIMPLE)

    class _StubHealthChecker:
        def __init__(self) -> None:
            self.checked = False

        async def check_all(self) -> None:
            self.checked = True

        def get_health_summary(self) -> dict[str, Any]:
            return {
                "overall_status": "healthy",
                "healthy_count": 2,
                "total_count": 2,
                "checks": {
                    "vector_db": {"status": "ok"},
                    "cache": {"status": "ok"},
                },
            }

    stub_checker = _StubHealthChecker()

    async def _get_stub_checker() -> _StubHealthChecker:
        return stub_checker

    app_instance.dependency_overrides[get_health_checker] = _get_stub_checker
    return app_instance


def test_create_app_sets_profile_and_mode(app: FastAPI) -> None:
    """The created app should expose profile and mode metadata on state."""

    assert app.state.profile == AppProfile.SIMPLE
    assert app.state.mode == AppProfile.SIMPLE.to_mode()


def test_root_endpoint_returns_mode_banner(app: FastAPI) -> None:
    """Root endpoint returns descriptive payload with mode details."""

    with TestClient(app) as client:
        response = client.get("/")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "running"
    assert payload["mode"] == ApplicationMode.SIMPLE.value
    assert payload["features"]["max_concurrent_crawls"] == 5


def test_health_endpoint_uses_stubbed_checker(app: FastAPI) -> None:
    """Health endpoint should honour the injected health checker summary."""

    with TestClient(app) as client:
        response = client.get("/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["healthy_count"] == 2
    assert payload["services"]["vector_db"]["status"] == "ok"


def test_openapi_exposes_versioned_routes(app: FastAPI) -> None:
    """The OpenAPI schema should include canonical v1 routers."""

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    schema = response.json()
    assert "/api/v1/search" in schema["paths"]
    assert "/api/v1/documents" in schema["paths"]
