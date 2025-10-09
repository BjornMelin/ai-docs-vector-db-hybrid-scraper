"""Unit tests for the vector database management CLI and manager."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from src.contracts.retrieval import SearchRecord
from src.manage_vector_db import (
    CollectionCreationError,
    CollectionDeletionError,
    CollectionInfo,
    CollectionSchema,
    DatabaseStats,
    VectorDBManager,
    cli,
)
from src.services.errors import QdrantServiceError


@pytest.fixture()
def vector_service_mock() -> AsyncMock:
    """Provide an AsyncMock representing the VectorStoreService."""

    service = AsyncMock()
    service.list_collections = AsyncMock(return_value=["docs"])
    service.ensure_collection = AsyncMock(return_value=None)
    service.drop_collection = AsyncMock(return_value=None)
    service.collection_stats = AsyncMock(
        return_value={
            "points_count": 7,
            "vectors": {"default": {"size": 3}},
        }
    )
    service.search_documents = AsyncMock(
        return_value=[
            SearchRecord(
                id="doc-1",
                score=0.9,
                content="",
                metadata={"url": "https://example"},
            )
        ]
    )
    return service


@pytest.fixture()
def manager_setup(
    monkeypatch: pytest.MonkeyPatch, vector_service_mock: AsyncMock
) -> SimpleNamespace:
    """Create a VectorDBManager wired to stubbed client manager helpers."""

    client_manager = SimpleNamespace(
        get_vector_store_service=AsyncMock(return_value=vector_service_mock)
    )
    ensure_manager = AsyncMock(return_value=client_manager)
    shutdown_manager = AsyncMock()

    monkeypatch.setattr(
        "src.manage_vector_db.ensure_client_manager",
        ensure_manager,
    )
    monkeypatch.setattr(
        "src.manage_vector_db.shutdown_client_manager",
        shutdown_manager,
    )

    manager = VectorDBManager()

    return SimpleNamespace(
        manager=manager,
        ensure_manager=ensure_manager,
        shutdown_manager=shutdown_manager,
        vector_service=vector_service_mock,
    )


@pytest.mark.asyncio
async def test_initialize_use_client_manager(manager_setup: SimpleNamespace) -> None:
    """initialize should resolve the vector service via the client manager."""

    manager = manager_setup.manager

    await manager.initialize()

    manager_setup.ensure_manager.assert_awaited()
    service = await manager.get_vector_store_service()
    assert service is manager_setup.vector_service


@pytest.mark.asyncio
async def test_list_collections_uses_vector_service(
    manager_setup: SimpleNamespace,
) -> None:
    """list_collections should call the vector service and return results."""

    manager = manager_setup.manager

    collections = await manager.list_collections()

    assert collections == ["docs"]
    manager_setup.vector_service.list_collections.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_collection_builds_schema(
    manager_setup: SimpleNamespace,
) -> None:
    """create_collection should construct the schema and invoke ensure_collection."""

    manager = manager_setup.manager

    result = await manager.create_collection("analytics", vector_size=256)

    assert result is True
    manager_setup.vector_service.ensure_collection.assert_awaited_once()
    schema: CollectionSchema = (
        manager_setup.vector_service.ensure_collection.call_args.args[0]
    )
    assert schema.name == "analytics"
    assert schema.vector_size == 256


@pytest.mark.asyncio
async def test_delete_collection_drops_via_service(
    manager_setup: SimpleNamespace,
) -> None:
    """delete_collection should call drop_collection and report success."""

    manager = manager_setup.manager

    result = await manager.delete_collection("obsolete")

    assert result is True
    manager_setup.vector_service.drop_collection.assert_awaited_once_with("obsolete")


@pytest.mark.asyncio
async def test_get_collection_info_maps_stats(
    manager_setup: SimpleNamespace,
) -> None:
    """get_collection_info should translate stats into CollectionInfo."""

    manager = manager_setup.manager

    info = await manager.get_collection_info("docs")

    assert isinstance(info, CollectionInfo)
    assert info.vector_count == 7
    assert info.vector_size == 3


@pytest.mark.asyncio
async def test_search_documents_returns_models(
    manager_setup: SimpleNamespace,
) -> None:
    """search_documents should return canonical SearchRecord objects."""

    manager = manager_setup.manager

    results = await manager.search_documents("docs", "query", limit=1)

    assert len(results) == 1
    assert isinstance(results[0], SearchRecord)
    manager_setup.vector_service.search_documents.assert_awaited_once_with(
        "docs",
        "query",
        limit=1,
    )


@pytest.mark.asyncio
async def test_clear_collection_recreates(
    manager_setup: SimpleNamespace,
) -> None:
    """clear_collection should drop and then ensure the collection."""

    manager = manager_setup.manager

    result = await manager.clear_collection("docs")

    assert result is True
    manager_setup.vector_service.drop_collection.assert_awaited()
    manager_setup.vector_service.ensure_collection.assert_awaited()


@pytest.mark.asyncio
async def test_cleanup_shuts_down_manager(manager_setup: SimpleNamespace) -> None:
    """cleanup should release the client-manager-managed services."""

    manager = manager_setup.manager

    await manager.initialize()
    await manager.cleanup()

    manager_setup.shutdown_manager.assert_awaited_once()


def _run_cli(command: list[str], manager_stub: AsyncMock) -> Any:
    """Invoke the CLI with the provided command arguments."""

    runner = CliRunner()
    with (
        patch("src.manage_vector_db.setup_logging"),
        patch(
            "src.manage_vector_db._create_manager_from_context",
            return_value=manager_stub,
        ),
    ):
        return runner.invoke(cli, command, obj={})


def test_cli_list_collections_outputs_results() -> None:
    """list-collections command should print discovered collections."""

    manager_stub = AsyncMock()
    manager_stub.list_collections = AsyncMock(return_value=["docs"])
    manager_stub.cleanup = AsyncMock()

    result = _run_cli(["list-collections"], manager_stub)

    assert result.exit_code == 0, result.output
    assert "docs" in result.output
    manager_stub.list_collections.assert_awaited_once()
    manager_stub.cleanup.assert_awaited_once()


def test_cli_create_collection_reports_success() -> None:
    """create command should call VectorDBManager.create_collection."""

    manager_stub = AsyncMock()
    manager_stub.create_collection = AsyncMock(return_value=True)
    manager_stub.cleanup = AsyncMock()

    result = _run_cli(["create", "docs"], manager_stub)

    assert result.exit_code == 0, result.output
    assert "Successfully created collection" in result.output
    manager_stub.create_collection.assert_awaited_once_with("docs", vector_size=1536)
    manager_stub.cleanup.assert_awaited_once()


def test_cli_search_prints_results() -> None:
    """search command should display formatted search results."""

    search_result = SearchRecord(
        id="doc-1",
        score=0.8,
        url="https://example",
        title="Example",
        content="Preview",
        metadata={},
    )
    manager_stub = AsyncMock()
    manager_stub.search_documents = AsyncMock(return_value=[search_result])
    manager_stub.cleanup = AsyncMock()

    result = _run_cli(["search", "docs", "query"], manager_stub)

    assert result.exit_code == 0, result.output
    assert "Example" in result.output
    manager_stub.search_documents.assert_awaited_once_with("docs", "query", limit=5)
    manager_stub.cleanup.assert_awaited_once()


def test_cli_stats_prints_table() -> None:
    """stats command should render aggregate metrics."""

    stats = DatabaseStats(
        total_collections=1,
        total_vectors=3,
        collections=[CollectionInfo(name="docs", vector_count=3, vector_size=3)],
    )
    manager_stub = AsyncMock()
    manager_stub.get_database_stats = AsyncMock(return_value=stats)
    manager_stub.cleanup = AsyncMock()

    result = _run_cli(["stats"], manager_stub)

    assert result.exit_code == 0, result.output
    assert "Total Collections" in result.output
    manager_stub.get_database_stats.assert_awaited_once()
    manager_stub.cleanup.assert_awaited_once()


def test_collection_errors_subclass_qdrant_service_error() -> None:
    """Ensure collection lifecycle errors derive from QdrantServiceError."""

    assert issubclass(CollectionCreationError, QdrantServiceError)
    assert issubclass(CollectionDeletionError, QdrantServiceError)
