"""Unit tests for the configuration management API router."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.api.routers import config as config_router
from src.config import SecurityConfig, get_settings, refresh_settings


@pytest.fixture(autouse=True)
def _reset_settings(config_factory) -> Generator[None, None, None]:
    """Ensure each test starts with a pristine settings cache."""
    refresh_settings()
    yield
    refresh_settings()


@pytest.fixture()
def client(config_factory) -> TestClient:
    """Provide a FastAPI test client with the configuration router registered."""
    refresh_settings(
        settings=config_factory(
            mode=get_settings().mode,
            environment=get_settings().environment,
        )
    )
    app = FastAPI()
    app.include_router(config_router.router)
    return TestClient(app)


def test_read_settings_returns_snapshot(client: TestClient) -> None:
    """GET /config should return a sanitized settings snapshot."""
    response = client.get("/config")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["app_name"] == get_settings().app_name
    assert payload["version"] == get_settings().version
    assert payload["mode"] == get_settings().mode
    assert payload["environment"] == get_settings().environment.value
    assert payload["feature_flags"] == get_settings().get_feature_flags()


def test_read_status_reports_unified_configuration(client: TestClient) -> None:
    """GET /config/status mirrors the active settings state."""
    response = client.get("/config/status")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["mode"] == get_settings().mode
    assert payload["feature_flags"] == get_settings().get_feature_flags()


def test_refresh_endpoint_applies_overrides(client: TestClient) -> None:
    """POST /config/refresh should rebuild settings with overrides applied."""
    response = client.post(
        "/config/refresh",
        json={"overrides": {"app_name": "Refreshed App"}},
    )
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["snapshot"]["app_name"] == "Refreshed App"
    assert get_settings().app_name == "Refreshed App"


def test_api_key_required_blocks_requests(
    config_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API key protected endpoints should reject requests without credentials."""
    secure_settings = config_factory(
        security=SecurityConfig(
            api_key_required=True, api_keys=["super-secret"], api_key_header="X-API-Key"
        )
    )
    refresh_settings(settings=secure_settings)
    app = FastAPI()
    app.include_router(config_router.router)
    client = TestClient(app)

    response = client.get("/config")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

    response = client.get("/config", headers={"X-API-Key": "super-secret"})
    assert response.status_code == status.HTTP_200_OK


def test_read_settings_handles_invalid_snapshot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Configuration snapshot endpoint sanitizes malformed settings."""
    app = FastAPI()
    app.include_router(config_router.router)

    from dataclasses import dataclass

    @dataclass
    class BrokenObservability:
        enabled: bool = True

    @dataclass
    class BrokenSecurity:
        api_key_required: bool = False

    class BrokenSettings:
        app_name = "broken"
        version = "0.0.0"
        mode = "invalid"
        environment = None
        debug = True
        observability = BrokenObservability()
        security = BrokenSecurity()

        @staticmethod
        def get_feature_flags() -> dict[str, bool]:
            raise RuntimeError("unexpected feature failure")

    def _get_broken_settings() -> BrokenSettings:
        return BrokenSettings()

    monkeypatch.setattr(config_router, "get_settings", _get_broken_settings)

    with TestClient(app) as client, caplog.at_level("ERROR"):
        response = client.get("/config")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    payload = response.json()
    assert payload == {"detail": "Settings snapshot unavailable"}
    assert "Failed to serialize settings snapshot" in caplog.text
    assert "unexpected feature failure" not in str(payload)
