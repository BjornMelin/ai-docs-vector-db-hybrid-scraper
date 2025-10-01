"""Tests for the VectorStore-backed document management tools."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.mcp_tools.tools.document_management import register_tools


@pytest.fixture
def vector_service() -> Mock:
    service = Mock()
    service.is_initialized.return_value = True
    service.ensure_collection = AsyncMock()
    service.collection_stats = AsyncMock(return_value={"points_count": 3})
    service.list_documents = AsyncMock(return_value=([{"id": "1"}], None))
    service.list_collections = AsyncMock(return_value=["workspace_docs", "other"])
    service.delete = AsyncMock()
    service.clear_collection = AsyncMock()
    return service


@pytest.fixture
def client_manager(vector_service: Mock) -> Mock:
    manager = Mock()
    manager.get_vector_store_service = AsyncMock(return_value=vector_service)
    return manager


@pytest.fixture
def context() -> Mock:
    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


@pytest.fixture
def tools(client_manager: Mock) -> dict[str, Callable]:
    mock_mcp = MagicMock()
    registered: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        registered[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    register_tools(mock_mcp, client_manager)
    return registered


@pytest.mark.asyncio
async def test_create_workspace_provisions_collections(
    tools: dict[str, Callable], vector_service: Mock, context: Mock
) -> None:
    response = await tools["create_document_workspace"](
        "workspace", ["docs", "guides"], ctx=context
    )

    assert response["collections"] == ["workspace_docs", "workspace_guides"]
    assert vector_service.ensure_collection.await_count == 2
    context.debug.assert_awaited()
    context.info.assert_awaited()


@pytest.mark.asyncio
async def test_manage_lifecycle_analyze_returns_stats(
    tools: dict[str, Callable], vector_service: Mock, context: Mock
) -> None:
    vector_service.collection_stats.return_value = {"points_count": 1}
    vector_service.list_documents.return_value = ([{"id": "1"}, {"id": "2"}], None)

    response = await tools["manage_document_lifecycle"](
        "workspace_docs", "analyze", ctx=context
    )

    assert response["action"] == "analyze"
    assert response["stats"]["points_count"] == 1
    assert len(response["sample_documents"]) == 2
    context.info.assert_awaited()


@pytest.mark.asyncio
async def test_manage_lifecycle_cleanup_requires_filters(
    tools: dict[str, Callable],
) -> None:
    with pytest.raises(ValueError):
        await tools["manage_document_lifecycle"]("docs", "cleanup")


@pytest.mark.asyncio
async def test_manage_lifecycle_cleanup_deletes_with_filters(
    tools: dict[str, Callable], vector_service: Mock, context: Mock
) -> None:
    response = await tools["manage_document_lifecycle"](
        "docs", "cleanup", filters={"tag": "draft"}, ctx=context
    )

    vector_service.delete.assert_awaited_once_with("docs", filters={"tag": "draft"})
    assert response["status"] == "deleted"
    context.info.assert_awaited()


@pytest.mark.asyncio
async def test_manage_lifecycle_optimize_resets_collection(
    tools: dict[str, Callable], vector_service: Mock, context: Mock
) -> None:
    response = await tools["manage_document_lifecycle"]("docs", "optimize", ctx=context)

    vector_service.clear_collection.assert_awaited_once_with("docs")
    assert response["status"] == "recreated"


@pytest.mark.asyncio
async def test_list_documents_returns_payload(
    tools: dict[str, Callable], vector_service: Mock, context: Mock
) -> None:
    response = await tools["list_documents"]("docs", limit=10, ctx=context)

    vector_service.list_documents.assert_awaited_once_with(
        "docs", limit=10, offset=None
    )
    assert response == {"documents": [{"id": "1"}], "next_offset": None}


@pytest.mark.asyncio
async def test_get_workspace_analytics_filters_collections(
    tools: dict[str, Callable], vector_service: Mock, context: Mock
) -> None:
    vector_service.collection_stats.return_value = {"points_count": 3}

    response = await tools["get_workspace_analytics"]("workspace", ctx=context)

    assert "workspace" in response["workspaces"]
    collections = response["workspaces"]["workspace"]["collections"]
    assert collections[0]["name"] == "workspace_docs"
    vector_service.collection_stats.assert_awaited()
    context.info.assert_awaited()
