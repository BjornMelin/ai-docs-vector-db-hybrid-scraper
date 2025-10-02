"""Unit tests for the vector database management CLI and manager."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from src.infrastructure.client_manager import ClientManager
from src.manage_vector_db import (
    CollectionInfo,
    CollectionSchema,
    DatabaseStats,
    SearchResult,
    VectorDBManager,
    cli,
)


class ClientManagerHarness:
    """Minimal async client manager harness for VectorDBManager tests."""

    def __init__(self, vector_service: AsyncMock) -> None:
        self._service = vector_service
        self.initialize = AsyncMock(side_effect=self._mark_initialized)
        self.cleanup = AsyncMock()
        self.get_vector_store_service = AsyncMock(return_value=vector_service)
        self.is_initialized = False

    async def _mark_initialized(self) -> None:
        self.is_initialized = True


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
            SimpleNamespace(
                id="doc-1",
                score=0.9,
                payload={"url": "https://example"},
            )
        ]
    )
    return service


@pytest.fixture()
def client_manager_harness(vector_service_mock: AsyncMock) -> ClientManagerHarness:
    """Construct the client manager harness bound to the vector service mock."""

    return ClientManagerHarness(vector_service_mock)


@pytest.fixture()
def manager(client_manager_harness: ClientManagerHarness) -> VectorDBManager:
    """Create the VectorDBManager under test."""

    return VectorDBManager(client_manager=cast(ClientManager, client_manager_harness))


@pytest.mark.asyncio
async def test_initialize_invokes_client_manager(
    manager: VectorDBManager,
    client_manager_harness: ClientManagerHarness,
) -> None:
    """initialize should delegate to the client manager once."""

    await manager.initialize()

    assert manager._initialized is True
    assert client_manager_harness.initialize.await_count == 1


@pytest.mark.asyncio
async def test_list_collections_uses_vector_service(
    manager: VectorDBManager,
    vector_service_mock: AsyncMock,
) -> None:
    """list_collections should call the vector service and return results."""

    collections = await manager.list_collections()

    assert collections == ["docs"]
    vector_service_mock.list_collections.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_collection_builds_schema(
    manager: VectorDBManager,
    vector_service_mock: AsyncMock,
) -> None:
    """create_collection should construct the schema and invoke ensure_collection."""

    result = await manager.create_collection("analytics", vector_size=256)

    assert result is True
    vector_service_mock.ensure_collection.assert_awaited_once()
    schema: CollectionSchema = vector_service_mock.ensure_collection.call_args.args[0]
    assert schema.name == "analytics"
    assert schema.vector_size == 256


@pytest.mark.asyncio
async def test_delete_collection_drops_via_service(
    manager: VectorDBManager,
    vector_service_mock: AsyncMock,
) -> None:
    """delete_collection should call drop_collection and report success."""

    result = await manager.delete_collection("obsolete")

    assert result is True
    vector_service_mock.drop_collection.assert_awaited_once_with("obsolete")


@pytest.mark.asyncio
async def test_get_collection_info_maps_stats(
    manager: VectorDBManager,
    vector_service_mock: AsyncMock,
) -> None:
    """get_collection_info should translate stats into CollectionInfo."""

    info = await manager.get_collection_info("docs")

    assert isinstance(info, CollectionInfo)
    assert info.vector_count == 7
    assert info.vector_size == 3


@pytest.mark.asyncio
async def test_search_documents_returns_models(
    manager: VectorDBManager,
    vector_service_mock: AsyncMock,
) -> None:
    """search_documents should wrap adapter matches into SearchResult objects."""

    results = await manager.search_documents("docs", "query", limit=1)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    vector_service_mock.search_documents.assert_awaited_once_with(
        "docs",
        "query",
        limit=1,
    )


@pytest.mark.asyncio
async def test_clear_collection_recreates(
    manager: VectorDBManager,
    vector_service_mock: AsyncMock,
) -> None:
    """clear_collection should drop and then ensure the collection."""

    result = await manager.clear_collection("docs")

    assert result is True
    vector_service_mock.drop_collection.assert_awaited()
    vector_service_mock.ensure_collection.assert_awaited()


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
    assert manager_stub.list_collections.await_count == 1
    assert manager_stub.cleanup.await_count == 1


def test_cli_create_collection_reports_success() -> None:
    """create command should call VectorDBManager.create_collection."""

    manager_stub = AsyncMock()
    manager_stub.create_collection = AsyncMock(return_value=True)
    manager_stub.cleanup = AsyncMock()

    result = _run_cli(["create", "docs"], manager_stub)

    assert result.exit_code == 0, result.output
    assert "Successfully created collection" in result.output
    manager_stub.create_collection.assert_awaited_once_with("docs", vector_size=1536)


def test_cli_search_prints_results() -> None:
    """search command should display formatted search results."""

    search_result = SearchResult(
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
