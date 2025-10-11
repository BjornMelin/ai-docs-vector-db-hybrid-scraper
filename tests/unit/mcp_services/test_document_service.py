"""Unit tests for :mod:`src.mcp_services.document_service`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mcp_services.document_service import DocumentService


@pytest.fixture()
def tool_patches(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch tool modules to observe registration calls."""

    patches: dict[str, MagicMock] = {}
    module_targets = {
        "documents": "src.mcp_tools.tools.documents.register_tools",
        "collections": "src.mcp_tools.tools.collection_management.register_tools",
        "projects": "src.mcp_tools.tools.projects.register_tools",
        "crawling": "src.mcp_tools.tools.crawling.register_tools",
        "content": "src.mcp_tools.tools.content_intelligence.register_tools",
    }
    for key, target in module_targets.items():
        mock = MagicMock()
        monkeypatch.setattr(target, mock)
        patches[key] = mock
    return patches


def test_init_registers_all_tool_modules(tool_patches: dict[str, MagicMock]) -> None:
    """Constructor wires every canonical MCP tool module."""

    vector = object()
    cache = object()
    crawl = object()
    content = object()
    storage = object()

    service = DocumentService(
        "docs",
        vector_service=vector,  # type: ignore[arg-type]
        cache_manager=cache,  # type: ignore[arg-type]
        crawl_manager=crawl,
        content_intelligence_service=content,
        project_storage=storage,  # type: ignore[arg-type]
    )

    assert service.mcp.name == "docs"

    tool_patches["documents"].assert_called_once_with(
        service.mcp,
        vector_service=vector,
        cache_manager=cache,
        crawl_manager=crawl,
        content_intelligence_service=content,
    )
    tool_patches["collections"].assert_called_once_with(
        service.mcp, vector_service=vector, cache_manager=cache
    )
    tool_patches["projects"].assert_called_once_with(
        service.mcp, vector_service=vector, project_storage=storage
    )
    tool_patches["crawling"].assert_called_once_with(service.mcp, crawl_manager=crawl)
    tool_patches["content"].assert_called_once_with(
        service.mcp, content_service=content
    )


def test_get_mcp_server_returns_configured_instance() -> None:
    """`get_mcp_server` should return the FastMCP instance created at init."""

    service = DocumentService("custom")

    assert service.get_mcp_server() is service.mcp
    assert service.get_mcp_server().name == "custom"


@pytest.mark.asyncio
async def test_get_service_info_exposes_capabilities() -> None:
    """`get_service_info` produces stable metadata used by clients."""

    service = DocumentService()

    info = await service.get_service_info()

    assert info["service"] == "document"
    assert info["version"] == "3.0"
    assert info["status"] == "active"
    assert "web_crawling" in info["capabilities"]
    assert "content_processing" in info["capabilities"]
