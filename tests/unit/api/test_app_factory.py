"""Tests for the unified FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from src.api import app_factory
from src.config.loader import load_settings, refresh_settings
from src.config.models import Environment


@pytest.fixture(autouse=True)
def _patch_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent heavy service initialization during unit tests."""

    async def async_noop(_: Any) -> None:
        return None

    @asynccontextmanager
    async def dummy_lifespan(_: Any) -> AsyncIterator[None]:
        yield

    monkeypatch.setattr(app_factory, "_initialize_services", async_noop)
    monkeypatch.setattr(app_factory, "container_lifespan", dummy_lifespan)


def test_create_app_registers_canonical_routes() -> None:
    """The generated application exposes the canonical routers."""

    app = app_factory.create_app()

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/search" in schema["paths"]
    assert "/api/v1/documents" in schema["paths"]


def test_create_app_route_registration_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Router import failures should not prevent app creation."""

    original_import = app_factory.import_module

    def broken_import(module_path: str) -> Any:
        if module_path.endswith(".search"):
            raise ImportError("Simulated import failure")
        return original_import(module_path)

    monkeypatch.setattr(app_factory, "import_module", broken_import)

    with caplog.at_level("ERROR"):
        app = app_factory.create_app()

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/search" not in schema["paths"]
    assert "/api/v1/documents" in schema["paths"]
    assert any("Simulated import failure" in message for message in caplog.messages)


def test_root_endpoint_reports_features() -> None:
    """The root endpoint surfaces configured feature flags."""

    app = app_factory.create_app()

    with TestClient(app) as client:
        response = client.get("/")

    payload = response.json()
    assert response.status_code == 200
    assert "features" in payload
    assert isinstance(payload["features"], dict)


def test_root_endpoint_error_handling_missing_settings(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Root endpoint should surface sanitized errors when settings fail."""

    app = app_factory.create_app()
    settings = app.state.settings

    def broken_feature_flags(_: Any) -> dict[str, bool]:
        raise RuntimeError("Settings missing or malformed")

    monkeypatch.setattr(
        type(settings), "get_feature_flags", broken_feature_flags, raising=True
    )

    with TestClient(app) as client, caplog.at_level("ERROR"):
        response = client.get("/")

    assert response.status_code == 500
    payload = response.json()
    assert payload == {"detail": "Configuration is unavailable"}
    assert "Failed to build root endpoint payload" in caplog.text
    assert "Settings missing or malformed" not in payload["detail"]


def test_features_endpoint_matches_root_payload() -> None:
    """The features endpoint mirrors the root feature summary."""

    app = app_factory.create_app()

    with TestClient(app) as client:
        root_payload = client.get("/").json()
        features_payload = client.get("/features").json()

    assert features_payload == root_payload["features"]


def _get_cors_middleware(app: FastAPI) -> dict[str, Any]:
    """Return the configuration dictionary for the CORS middleware."""

    middleware = next(
        entry for entry in app.user_middleware if entry.cls is CORSMiddleware
    )
    return middleware.options


def test_cors_disables_credentials_for_wildcard_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wildcard origins should disable credentials to satisfy Starlette constraints."""

    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    refresh_settings(
        settings=load_settings(environment=Environment.DEVELOPMENT)
    )

    try:
        app = app_factory.create_app()
        cors_options = _get_cors_middleware(app)

        assert cors_options["allow_origins"] == ["*"]
        assert cors_options["allow_credentials"] is False
    finally:
        refresh_settings(settings=load_settings())


def test_cors_retains_credentials_for_explicit_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit origins may continue to use credentials."""

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://example.test")
    refresh_settings(settings=load_settings(environment=Environment.PRODUCTION))

    try:
        app = app_factory.create_app()
        cors_options = _get_cors_middleware(app)

        assert cors_options["allow_origins"] == ["https://example.test"]
        assert cors_options["allow_credentials"] is True
    finally:
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        refresh_settings(settings=load_settings())
