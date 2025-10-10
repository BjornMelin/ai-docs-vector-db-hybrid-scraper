"""Tests for application mode initialization helpers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from src.api import app_factory
from src.architecture.modes import ApplicationMode, get_mode_config


@pytest.mark.asyncio
async def test_initialize_mode_services_simple_mode(monkeypatch, mocker) -> None:
    """Simple mode should eagerly initialize core vector dependencies."""

    mode_config = get_mode_config(ApplicationMode.SIMPLE)
    embedding = mocker.AsyncMock()
    vector_service = mocker.AsyncMock()
    qdrant = mocker.AsyncMock()
    cache = mocker.AsyncMock()

    monkeypatch.setattr(app_factory, "core_get_embedding_manager", embedding)
    monkeypatch.setattr(app_factory, "core_get_vector_store_service", vector_service)
    monkeypatch.setattr(app_factory, "_init_qdrant_client", qdrant)
    monkeypatch.setattr(app_factory, "core_get_cache_manager", cache)

    await app_factory._initialize_mode_services(mode_config)

    embedding.assert_awaited()
    vector_service.assert_awaited()
    qdrant.assert_awaited()
    cache.assert_awaited()


@pytest.mark.asyncio
async def test_initialize_mode_services_enterprise_mode(monkeypatch, mocker) -> None:
    """Enterprise mode should initialize advanced services."""

    mode_config = get_mode_config(ApplicationMode.ENTERPRISE)
    embedding = mocker.AsyncMock()
    vector_service = mocker.AsyncMock()
    qdrant = mocker.AsyncMock()
    cache = mocker.AsyncMock()
    db_ready = mocker.AsyncMock()
    content_service = mocker.AsyncMock()

    monkeypatch.setattr(app_factory, "core_get_embedding_manager", embedding)
    monkeypatch.setattr(app_factory, "core_get_vector_store_service", vector_service)
    monkeypatch.setattr(app_factory, "_init_qdrant_client", qdrant)
    monkeypatch.setattr(app_factory, "core_get_cache_manager", cache)
    monkeypatch.setattr(app_factory, "_ensure_database_ready", db_ready)
    monkeypatch.setattr(
        app_factory, "core_get_content_intelligence_service", content_service
    )

    await app_factory._initialize_mode_services(mode_config)

    embedding.assert_awaited()
    vector_service.assert_awaited()
    qdrant.assert_awaited()
    cache.assert_awaited()
    db_ready.assert_awaited()
    content_service.assert_awaited()


def test_get_app_container_success() -> None:
    """The application helper should return the DI container when present."""

    app = FastAPI()
    dummy_container = app_factory.ApplicationContainer()
    app.state.container = dummy_container

    result = app_factory.get_app_container(app)
    assert result is dummy_container


def test_get_app_container_missing() -> None:
    """Accessing the helper without wiring a container should raise an error."""

    app = FastAPI()
    with pytest.raises(RuntimeError):
        app_factory.get_app_container(app)
