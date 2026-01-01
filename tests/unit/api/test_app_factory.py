"""Tests for the unified FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from src.api import app_factory
from src.config.loader import Settings, load_settings, refresh_settings
from src.config.models import Environment


_ORIGINAL_INITIALIZE_SERVICES = app_factory._initialize_services


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
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/v1/search" in paths
    assert "/api/v1/documents" in paths


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
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/v1/search" not in paths
    assert "/api/v1/documents" in paths
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
    options = getattr(middleware, "kwargs", {})
    return cast(dict[str, Any], options)


def test_cors_disables_credentials_for_wildcard_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wildcard origins should disable credentials to satisfy Starlette constraints."""
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    refresh_settings(settings=load_settings(environment=Environment.DEVELOPMENT))

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


def test_cache_initialization_enabled_recognizes_flags() -> None:
    """The cache toggle helper should respect enablement flags."""
    enabled = cast(
        Settings, SimpleNamespace(cache=SimpleNamespace(enable_caching=True))
    )
    disabled = cast(
        Settings, SimpleNamespace(cache=SimpleNamespace(enable_caching=False))
    )
    missing = cast(Settings, SimpleNamespace())

    assert app_factory._cache_initialization_enabled(enabled) is True
    assert app_factory._cache_initialization_enabled(disabled) is False
    assert app_factory._cache_initialization_enabled(missing) is False


@pytest.mark.asyncio()
async def test_ensure_database_ready_handles_cache_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database readiness should not touch cache when disabled."""
    settings = cast(
        Settings, SimpleNamespace(cache=SimpleNamespace(enable_caching=False))
    )
    vector_service = AsyncMock()
    cache_manager = AsyncMock()
    ping = AsyncMock()

    monkeypatch.setattr(app_factory, "resolve_vector_store_service", vector_service)
    monkeypatch.setattr(app_factory, "resolve_cache_manager", cache_manager)
    monkeypatch.setattr(app_factory, "_ping_dragonfly", ping)

    await app_factory._ensure_database_ready(settings)

    assert vector_service.await_count == 1
    assert cache_manager.await_count == 0
    assert ping.await_count == 0


@pytest.mark.asyncio()
async def test_ensure_database_ready_warms_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache warm-up should invoke cache manager and dragonfly ping."""
    settings = cast(
        Settings, SimpleNamespace(cache=SimpleNamespace(enable_caching=True))
    )
    vector_service = AsyncMock()
    cache_manager = AsyncMock()
    ping = AsyncMock()

    monkeypatch.setattr(app_factory, "resolve_vector_store_service", vector_service)
    monkeypatch.setattr(app_factory, "resolve_cache_manager", cache_manager)
    monkeypatch.setattr(app_factory, "_ping_dragonfly", ping)

    await app_factory._ensure_database_ready(settings)

    assert vector_service.await_count == 1
    assert cache_manager.await_count == 1
    assert ping.await_count == 1


@pytest.mark.asyncio()
async def test_initialize_services_invokes_all_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All critical services should be initialized when caching is enabled."""
    calls: list[str] = []

    def recorder(name: str):
        async def _inner(*_args: Any, **_kwargs: Any) -> None:
            calls.append(name)

        return _inner

    monkeypatch.setattr(app_factory, "resolve_embedding_manager", recorder("embed"))
    monkeypatch.setattr(app_factory, "resolve_vector_store_service", recorder("vector"))
    monkeypatch.setattr(app_factory, "_init_qdrant_client", recorder("qdrant"))
    monkeypatch.setattr(app_factory, "_ensure_database_ready", recorder("database"))
    monkeypatch.setattr(app_factory, "resolve_cache_manager", recorder("cache"))
    monkeypatch.setattr(app_factory, "_ping_dragonfly", recorder("dragonfly"))
    monkeypatch.setattr(
        app_factory,
        "resolve_content_intelligence_service",
        recorder("content"),
    )

    settings = cast(
        Settings, SimpleNamespace(cache=SimpleNamespace(enable_caching=True))
    )
    await _ORIGINAL_INITIALIZE_SERVICES(settings)

    assert set(calls) == {
        "embed",
        "vector",
        "qdrant",
        "database",
        "cache",
        "dragonfly",
        "content",
    }


@pytest.mark.asyncio()
async def test_initialize_services_skips_cache_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache initializers should be skipped when caching is disabled."""
    calls: list[str] = []

    def recorder(name: str):
        async def _inner(*_args: Any, **_kwargs: Any) -> None:
            calls.append(name)

        return _inner

    monkeypatch.setattr(app_factory, "resolve_embedding_manager", recorder("embed"))
    monkeypatch.setattr(app_factory, "resolve_vector_store_service", recorder("vector"))
    monkeypatch.setattr(app_factory, "_init_qdrant_client", recorder("qdrant"))
    monkeypatch.setattr(app_factory, "_ensure_database_ready", recorder("database"))
    monkeypatch.setattr(app_factory, "resolve_cache_manager", recorder("cache"))
    monkeypatch.setattr(app_factory, "_ping_dragonfly", recorder("dragonfly"))
    monkeypatch.setattr(
        app_factory,
        "resolve_content_intelligence_service",
        recorder("content"),
    )

    settings = cast(
        Settings, SimpleNamespace(cache=SimpleNamespace(enable_caching=False))
    )
    await _ORIGINAL_INITIALIZE_SERVICES(settings)

    assert set(calls) == {"embed", "vector", "qdrant", "database", "content"}


@pytest.mark.asyncio()
async def test_ping_dragonfly_handles_async_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dragonfly ping should await async ping operations."""
    called: dict[str, bool] = {"ping": False}

    class DummyClient:
        async def ping(self) -> None:
            called["ping"] = True

    class DummyContainer:
        def __init__(self) -> None:
            self.client = DummyClient()

        def dragonfly_client(self) -> DummyClient:
            return self.client

    monkeypatch.setattr(app_factory, "get_container", lambda: DummyContainer())

    await app_factory._ping_dragonfly()

    assert called["ping"] is True


@pytest.mark.asyncio()
async def test_ping_dragonfly_no_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ping helper should exit quietly when container is missing."""
    monkeypatch.setattr(app_factory, "get_container", lambda: None)
    await app_factory._ping_dragonfly()


@pytest.mark.asyncio()
async def test_init_qdrant_client_handles_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant readiness should await coroutine responses."""
    called: dict[str, bool] = {"collections": False}

    class DummyClient:
        async def get_collections(self) -> None:
            called["collections"] = True

    class DummyContainer:
        def __init__(self) -> None:
            self.client = DummyClient()

        def qdrant_client(self) -> DummyClient:
            return self.client

    monkeypatch.setattr(app_factory, "get_container", lambda: DummyContainer())

    await app_factory._init_qdrant_client()

    assert called["collections"] is True


@pytest.mark.asyncio()
async def test_init_qdrant_client_handles_missing_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant ping should ignore missing containers."""
    monkeypatch.setattr(app_factory, "get_container", lambda: None)
    await app_factory._init_qdrant_client()


@pytest.mark.asyncio()
async def test_build_app_lifespan_invokes_initializers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan context manager should run initialization hooks."""
    events: list[str] = []

    @asynccontextmanager
    async def fake_lifespan(_: Any) -> AsyncIterator[None]:
        events.append("enter")
        yield
        events.append("exit")

    async def fake_initialize(settings: Any) -> None:
        events.append(f"init:{settings.flag}")

    monkeypatch.setattr(app_factory, "container_lifespan", fake_lifespan)
    monkeypatch.setattr(app_factory, "_initialize_services", fake_initialize)

    lifespan = app_factory._build_app_lifespan(
        cast(Settings, SimpleNamespace(flag="ok"))
    )
    app = FastAPI()

    async with lifespan(app):
        events.append("inside")

    assert events == ["enter", "init:ok", "inside", "exit"]


def test_get_app_container_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_app_container` should return validated container instances."""

    class DummyContainer:
        pass

    app = FastAPI()
    container = DummyContainer()
    monkeypatch.setattr(app_factory, "ApplicationContainer", DummyContainer)
    app.state.container = container

    assert app_factory.get_app_container(app) is container


def test_get_app_container_missing_state() -> None:
    """Missing containers should raise informative errors."""
    app = FastAPI()
    with pytest.raises(RuntimeError):
        app_factory.get_app_container(app)


def test_get_app_container_rejects_wrong_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-container state should raise a TypeError."""

    class DummyContainer:
        pass

    app = FastAPI()
    monkeypatch.setattr(app_factory, "ApplicationContainer", DummyContainer)
    app.state.container = object()

    with pytest.raises(TypeError):
        app_factory.get_app_container(app)
