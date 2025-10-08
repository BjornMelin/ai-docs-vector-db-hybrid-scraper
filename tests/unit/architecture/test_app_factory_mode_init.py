"""Tests for application mode initialization helpers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from src.api import app_factory
from src.architecture.modes import ApplicationMode, get_mode_config


@pytest.mark.asyncio
async def test_initialize_mode_services_simple_mode(mocker) -> None:
    """Simple mode should eagerly initialize core vector dependencies."""

    mode_config = get_mode_config(ApplicationMode.SIMPLE)
    client_manager = mocker.Mock()
    client_manager.get_embedding_manager = mocker.AsyncMock()
    client_manager.get_vector_store_service = mocker.AsyncMock()
    client_manager.get_qdrant_client = mocker.AsyncMock()
    client_manager.get_cache_manager = mocker.AsyncMock()

    await app_factory._initialize_mode_services(client_manager, mode_config)

    client_manager.get_embedding_manager.assert_awaited_once()
    assert client_manager.get_vector_store_service.await_count >= 1
    client_manager.get_qdrant_client.assert_awaited_once()
    client_manager.get_cache_manager.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_mode_services_enterprise_mode(mocker) -> None:
    """Enterprise mode should initialize advanced services."""

    mode_config = get_mode_config(ApplicationMode.ENTERPRISE)
    client_manager = mocker.Mock()
    client_manager.get_embedding_manager = mocker.AsyncMock()
    client_manager.get_vector_store_service = mocker.AsyncMock()
    client_manager.get_qdrant_client = mocker.AsyncMock()
    client_manager.get_cache_manager = mocker.AsyncMock()
    client_manager.get_database_manager = mocker.AsyncMock()
    client_manager.get_content_intelligence_service = mocker.AsyncMock()

    await app_factory._initialize_mode_services(client_manager, mode_config)

    client_manager.get_embedding_manager.assert_awaited()
    client_manager.get_vector_store_service.assert_awaited()
    client_manager.get_qdrant_client.assert_awaited()
    client_manager.get_cache_manager.assert_awaited()
    client_manager.get_database_manager.assert_awaited()
    client_manager.get_content_intelligence_service.assert_awaited()


def test_get_app_client_manager_success() -> None:
    """The application helper should return the ClientManager when present."""

    app = FastAPI()
    dummy_manager = object()
    app.state.client_manager = dummy_manager

    result = app_factory.get_app_client_manager(app)
    assert result is dummy_manager


def test_get_app_client_manager_missing() -> None:
    """Accessing the helper without wiring a manager should raise an error."""

    app = FastAPI()
    with pytest.raises(RuntimeError):
        app_factory.get_app_client_manager(app)
