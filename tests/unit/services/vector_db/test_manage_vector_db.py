"""Deterministic coverage for the vector database management CLI."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.manage_vector_db import VectorDBManager, cli


@pytest.fixture()
def client_manager_stub() -> AsyncMock:
    """Provide a client manager with the minimal async surface used by the CLI."""
    manager = AsyncMock()
    manager.is_initialized = False
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.config = MagicMock()
    manager.config.qdrant.url = "http://localhost:6333"
    manager.get_qdrant_service = AsyncMock()
    manager.get_embedding_manager = AsyncMock()
    return manager


@pytest.fixture()
def manager(client_manager_stub: AsyncMock) -> VectorDBManager:
    """Instantiate the manager under test with the shared stub."""
    return VectorDBManager(client_manager_stub)


@pytest.mark.asyncio
async def test_initialize_invokes_client_manager(manager: VectorDBManager) -> None:
    """The manager delegates initialization to the shared client manager."""
    await manager.initialize()

    assert manager._initialized is True
    cast(AsyncMock, manager.client_manager.initialize).assert_awaited_once()


@pytest.mark.asyncio
async def test_list_collections_uses_qdrant_service(manager: VectorDBManager) -> None:
    """Listing collections proxies the request to the Qdrant service."""
    qdrant_service = AsyncMock()
    qdrant_service.list_collections = AsyncMock(return_value=["docs"])
    cast(
        AsyncMock, manager.client_manager.get_qdrant_service
    ).return_value = qdrant_service

    collections = await manager.list_collections()

    assert collections == ["docs"]
    qdrant_service.list_collections.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_collection_forwards_parameters(manager: VectorDBManager) -> None:
    """Creating a collection forwards the request and returns success state."""
    qdrant_service = AsyncMock()
    qdrant_service.create_collection = AsyncMock()
    cast(
        AsyncMock, manager.client_manager.get_qdrant_service
    ).return_value = qdrant_service

    result = await manager.create_collection("analytics", vector_size=256)

    assert result is True
    qdrant_service.create_collection.assert_awaited_once_with(
        collection_name="analytics", vector_size=256, distance="Cosine"
    )


@pytest.mark.asyncio
async def test_delete_collection_returns_boolean(manager: VectorDBManager) -> None:
    """Deleting a collection forwards the request and reports success."""
    qdrant_service = AsyncMock()
    qdrant_service.delete_collection = AsyncMock()
    cast(
        AsyncMock, manager.client_manager.get_qdrant_service
    ).return_value = qdrant_service

    result = await manager.delete_collection("obsolete")

    assert result is True
    qdrant_service.delete_collection.assert_awaited_once_with("obsolete")


def test_cli_list_collections() -> None:
    """The CLI list command prints discovered collections."""
    runner = CliRunner()

    manager_stub = AsyncMock(spec=VectorDBManager)
    manager_stub.list_collections = AsyncMock(return_value=["docs", "papers"])

    with (
        patch("src.manage_vector_db.create_embeddings", AsyncMock()),
        patch("src.manage_vector_db.setup_logging"),
        patch(
            "src.manage_vector_db._create_manager_from_context",
            return_value=manager_stub,
        ),
    ):
        result = runner.invoke(cli, ["list-collections"], obj={})

    assert result.exception is None, result.output
    assert result.exit_code == 0
    assert "docs" in result.output
    manager_stub.list_collections.assert_awaited_once()
