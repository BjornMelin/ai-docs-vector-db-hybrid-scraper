"""Tests for the unified FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import app_factory


@pytest.fixture(autouse=True)
def _patch_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent heavy service initialization during unit tests."""

    async def async_noop() -> None:
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


def test_root_endpoint_reports_features() -> None:
    """The root endpoint surfaces configured feature flags."""

    app = app_factory.create_app()

    with TestClient(app) as client:
        response = client.get("/")

    payload = response.json()
    assert response.status_code == 200
    assert "features" in payload
    assert isinstance(payload["features"], dict)


def test_features_endpoint_matches_root_payload() -> None:
    """The features endpoint mirrors the root feature summary."""

    app = app_factory.create_app()

    with TestClient(app) as client:
        root_payload = client.get("/").json()
        features_payload = client.get("/features").json()

    assert features_payload == root_payload["features"]
