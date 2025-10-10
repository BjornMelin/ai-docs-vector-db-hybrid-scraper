"""Tests for profile-driven FastAPI app composition."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import app_factory
from src.api.app_profiles import AppProfile


@pytest.fixture(autouse=True)
def _patch_service_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent heavy service initialization during unit tests."""

    async def async_noop(_: Any) -> None:
        return None

    monkeypatch.setattr(app_factory, "_register_mode_services", lambda _: None)
    monkeypatch.setattr(app_factory, "_initialize_critical_services", async_noop)
    monkeypatch.setattr(app_factory, "_apply_middleware_stack", lambda *_: None)


def test_create_app_simple_profile_exposes_service_status() -> None:
    """The health endpoint should expose status entries for enabled services."""

    app = app_factory.create_app(AppProfile.SIMPLE)

    class StubFactory:
        def __init__(self) -> None:
            self.available = ["embedding_service", "vector_db_service"]

        def get_available_services(self) -> list[str]:
            return self.available

        def get_service_status(self, name: str) -> dict[str, Any]:
            return {"name": name, "healthy": True}

        async def cleanup_all_services(self) -> None:  # pragma: no cover - unused
            return None

    stub_factory = StubFactory()
    app.state.service_factory = stub_factory
    app.state.mode_config.enabled_services = stub_factory.available

    with TestClient(app) as client:
        response = client.get("/health")
        payload = response.json()

        assert response.status_code == 200
        assert payload["available_services"] == stub_factory.available
        assert payload["service_status"] == [
            {"name": "embedding_service", "healthy": True},
            {"name": "vector_db_service", "healthy": True},
        ]


def test_create_app_enterprise_requires_routers() -> None:
    """Enterprise profile fails fast when modules are missing."""

    with pytest.raises(RuntimeError, match="Enterprise profile requires"):
        app_factory.create_app(AppProfile.ENTERPRISE)
