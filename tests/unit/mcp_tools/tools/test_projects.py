"""Tests for the simplified project management tools."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.config.models import SearchStrategy
from src.contracts.retrieval import SearchRecord
from src.mcp_tools.models.requests import ProjectRequest
from src.mcp_tools.models.responses import ProjectInfo
from src.mcp_tools.tools.projects import register_tools


@pytest.fixture
def mock_vector_service() -> Mock:
    """Create a vector service mock with asynchronous helpers."""

    service = Mock()
    service.is_initialized.return_value = True
    service.ensure_collection = AsyncMock()
    service.drop_collection = AsyncMock()
    service.collection_stats = AsyncMock(return_value={"points_count": 5})
    service.list_collections = AsyncMock(return_value=["collection-a"])
    service.search_documents = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_client_manager(mock_vector_service: Mock, monkeypatch) -> Mock:
    """Provide a client manager mock wired to the vector service mock."""

    storage = Mock()
    storage.save_project = AsyncMock()
    storage.list_projects = AsyncMock(return_value=[])
    storage.get_project = AsyncMock(return_value=None)
    storage.update_project = AsyncMock()
    storage.delete_project = AsyncMock()

    manager = Mock()
    manager.get_project_storage = AsyncMock(return_value=storage)
    manager.get_vector_store_service = AsyncMock(return_value=mock_vector_service)
    manager.project_storage = storage
    return manager


@pytest.fixture
def project_storage(mock_client_manager: Mock) -> Mock:
    """Return the project storage mock extracted from the manager."""

    return mock_client_manager.get_project_storage.return_value


@pytest.fixture
def register(mock_vector_service: Mock, project_storage: Mock) -> dict[str, Callable]:
    """Register project tools and expose them for tests."""

    mock_mcp = MagicMock()
    registered_tools: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    register_tools(
        mock_mcp,
        vector_service=mock_vector_service,
        project_storage=project_storage,
    )
    return registered_tools


@pytest.fixture
def mock_context() -> Mock:
    """Build an MCP context mock with logging hooks."""

    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_registers_project_tools(register: dict[str, Callable]) -> None:
    """Ensure tool registration exposes the project tool set."""

    assert {
        "create_project",
        "list_projects",
        "update_project",
        "delete_project",
        "search_project",
    } <= set(register)


@pytest.mark.asyncio
async def test_create_project(
    register: dict[str, Callable], project_storage: Mock, mock_context: Mock
) -> None:
    """Verify creating a project persists metadata and responds with info."""

    request = ProjectRequest(name="Docs", description="Demo", quality_tier="balanced")
    result = await register["create_project"](request, mock_context)

    assert isinstance(result, ProjectInfo)
    project_storage.save_project.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_projects_enriches_stats(
    register: dict[str, Callable],
    project_storage: Mock,
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    """Confirm listing projects enriches entries with collection stats."""

    project_storage.list_projects.return_value = [
        {
            "id": "proj",
            "name": "Existing",
            "collection": "project_proj",
        }
    ]

    projects = await register["list_projects"](mock_context)

    assert projects[0].stats == {"points_count": 5}
    mock_vector_service.collection_stats.assert_awaited_once_with("project_proj")


@pytest.mark.asyncio
async def test_update_project_applies_mutations(
    register: dict[str, Callable], project_storage: Mock, mock_context: Mock
) -> None:
    """Ensure project updates mutate stored values as expected."""

    project_storage.get_project.return_value = {
        "id": "proj",
        "name": "Old",
        "description": None,
    }

    updated = await register["update_project"]("proj", name="New", ctx=mock_context)

    assert updated.name == "New"
    project_storage.update_project.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_project_drops_collection(
    register: dict[str, Callable],
    project_storage: Mock,
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    """Check deleting a project drops related collections and records."""

    project_storage.get_project.return_value = {
        "id": "proj",
        "collection": "project_proj",
    }

    status = await register["delete_project"]("proj", ctx=mock_context)

    assert status.status == "deleted"
    mock_vector_service.drop_collection.assert_awaited_once_with("project_proj")
    project_storage.delete_project.assert_awaited_once_with("proj")


@pytest.mark.asyncio
async def test_search_project_uses_hybrid(
    register: dict[str, Callable],
    project_storage: Mock,
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    """Validate hybrid search delegates to the vector service."""

    project_storage.get_project.return_value = {
        "id": "proj",
        "collection": "project_proj",
    }
    mock_vector_service.search_documents.return_value = [
        SearchRecord(id="doc", score=0.8, content="body")
    ]

    results = await register["search_project"](
        "proj", query="hello", strategy=SearchStrategy.HYBRID, ctx=mock_context
    )

    mock_vector_service.search_documents.assert_awaited_once()
    assert results[0].content == "body"
