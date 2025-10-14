"""Tests for DocumentService dependency injection."""

from unittest.mock import MagicMock, patch

import pytest

from src.mcp_services.document_service import DocumentService


@pytest.fixture()
def dependency_set() -> dict[str, MagicMock]:
    """Return a mapping of mocked dependencies for the service."""
    return {
        "vector_service": MagicMock(),
        "cache_manager": MagicMock(),
        "crawl_manager": MagicMock(),
        "content_service": MagicMock(),
        "project_storage": MagicMock(),
    }


def test_document_service_registers_tools(
    dependency_set: dict[str, MagicMock],
) -> None:
    """Verify that tool modules are called with the injected dependencies."""
    dependencies = dependency_set
    with (
        patch("src.mcp_tools.tools.documents.register_tools") as documents_register,
        patch(
            "src.mcp_tools.tools.collection_management.register_tools"
        ) as collections_register,
        patch("src.mcp_tools.tools.projects.register_tools") as projects_register,
        patch("src.mcp_tools.tools.crawling.register_tools") as crawling_register,
        patch(
            "src.mcp_tools.tools.content_intelligence.register_tools"
        ) as content_register,
    ):
        service = DocumentService(
            vector_service=dependencies["vector_service"],
            cache_manager=dependencies["cache_manager"],
            crawl_manager=dependencies["crawl_manager"],
            content_intelligence_service=dependencies["content_service"],
            project_storage=dependencies["project_storage"],
        )

    documents_register.assert_called_once_with(
        service.mcp,
        vector_service=dependencies["vector_service"],
        cache_manager=dependencies["cache_manager"],
        crawl_manager=dependencies["crawl_manager"],
        content_intelligence_service=dependencies["content_service"],
    )
    collections_register.assert_called_once_with(
        service.mcp,
        vector_service=dependencies["vector_service"],
        cache_manager=dependencies["cache_manager"],
    )
    projects_register.assert_called_once_with(
        service.mcp,
        vector_service=dependencies["vector_service"],
        project_storage=dependencies["project_storage"],
    )
    crawling_register.assert_called_once_with(
        service.mcp,
        crawl_manager=dependencies["crawl_manager"],
    )
    content_register.assert_called_once_with(
        service.mcp,
        content_service=dependencies["content_service"],
    )


def test_get_mcp_server_returns_instance(dependency_set: dict[str, MagicMock]) -> None:
    """DocumentService exposes the FastMCP instance via get_mcp_server."""
    service = DocumentService(
        vector_service=dependency_set["vector_service"],
        cache_manager=dependency_set["cache_manager"],
        crawl_manager=dependency_set["crawl_manager"],
        content_intelligence_service=dependency_set["content_service"],
        project_storage=dependency_set["project_storage"],
    )

    assert service.get_mcp_server() is service.mcp


@pytest.mark.asyncio()
async def test_get_service_info(dependency_set: dict[str, MagicMock]) -> None:
    """Service metadata reflects capabilities and status."""
    service = DocumentService(
        vector_service=dependency_set["vector_service"],
        cache_manager=dependency_set["cache_manager"],
        crawl_manager=dependency_set["crawl_manager"],
        content_intelligence_service=dependency_set["content_service"],
        project_storage=dependency_set["project_storage"],
    )

    info = await service.get_service_info()

    assert info["service"] == "document"
    assert info["status"] == "active"
    assert "collection_management" in info["capabilities"]
