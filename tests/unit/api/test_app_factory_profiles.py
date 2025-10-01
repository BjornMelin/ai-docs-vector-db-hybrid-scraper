"""Tests for profile-driven FastAPI app composition."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.app_factory as app_factory
from src.api.app_profiles import AppProfile
from src.architecture.service_factory import reset_service_factory


warnings.filterwarnings(
    "ignore", message="on_event is deprecated", category=DeprecationWarning
)


@pytest.fixture(autouse=True)
def _reset_global_factory() -> Generator[None, None, None]:
    """Reset the global service factory after each test to avoid leakage."""

    yield
    reset_service_factory()


@pytest.fixture(autouse=True)
def _patch_service_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent heavy service initialization during unit tests."""

    async def async_noop(_: Any) -> None:
        return None

    monkeypatch.setattr(app_factory, "_register_mode_services", lambda _: None)
    monkeypatch.setattr(app_factory, "_initialize_critical_services", async_noop)


def test_create_app_simple_profile_exposes_service_status() -> None:
    """The health endpoint should expose status entries for enabled services."""

    app = app_factory.create_app(AppProfile.SIMPLE)

    class StubFactory:
        def __init__(self) -> None:
            self.available = ["embedding_service", "search_service"]

        def get_available_services(self) -> list[str]:
            return self.available

        def get_service_status(self, name: str) -> dict[str, Any]:
            return {"name": name, "healthy": True}

        async def cleanup_all_services(self) -> None:  # pragma: no cover - unused
            return None

    stub_factory = StubFactory()
    app.state.service_factory = stub_factory
    app.state.mode_config.enabled_services = stub_factory.available

    client = TestClient(app)
    response = client.get("/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["available_services"] == stub_factory.available
    assert payload["service_status"] == [
        {"name": "embedding_service", "healthy": True},
        {"name": "search_service", "healthy": True},
    ]


def test_create_app_enterprise_requires_routers() -> None:
    """Enterprise profile fails fast when modules are missing."""

    with pytest.raises(RuntimeError, match="Enterprise profile requires"):
        app_factory.create_app(AppProfile.ENTERPRISE)


def test_fail_closed_service_raises_on_access() -> None:
    """Fail-closed services raise clear RuntimeError when invoked."""

    placeholder_cls = app_factory._build_fail_closed_service("analytics_service")  # type: ignore[attr-defined]
    placeholder = placeholder_cls()

    with pytest.raises(RuntimeError, match="analytics_service"):
        placeholder.query_metrics()  # type: ignore[attr-defined]
