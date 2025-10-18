"""Shared fixtures for FastAPI middleware unit tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.fastapi.middleware import manager as middleware_manager


# Snapshot the baseline registries so every test starts from a clean slate.
_BASE_CLASS_REGISTRY = middleware_manager._CLASS_REGISTRY.copy()
_BASE_FUNCTION_REGISTRY = middleware_manager._FUNCTION_REGISTRY.copy()


@pytest.fixture(autouse=True)
def restore_manager_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset middleware registries between tests to keep state deterministic."""
    monkeypatch.setattr(
        middleware_manager,
        "_CLASS_REGISTRY",
        _BASE_CLASS_REGISTRY.copy(),
        raising=False,
    )
    monkeypatch.setattr(
        middleware_manager,
        "_FUNCTION_REGISTRY",
        _BASE_FUNCTION_REGISTRY.copy(),
        raising=False,
    )


@pytest.fixture(name="fastapi_app")
def fastapi_app_fixture() -> FastAPI:
    """Provide a fresh FastAPI application for tests that need one."""
    return FastAPI()


@pytest.fixture(name="test_client")
def test_client_fixture(fastapi_app: FastAPI) -> Generator[TestClient, None, None]:
    """Return a TestClient bound to the fastapi_app fixture."""
    with TestClient(fastapi_app) as client:
        yield client
