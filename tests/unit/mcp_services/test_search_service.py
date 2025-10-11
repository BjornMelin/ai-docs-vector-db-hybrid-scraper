"""Unit tests for :mod:`src.mcp_services.search_service`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mcp_services.search_service import SearchService


@pytest.fixture()
def tool_patches(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch retrieval and web search tool registration entrypoints."""

    retrieval_register = MagicMock()
    web_search_register = MagicMock()
    monkeypatch.setattr(
        "src.mcp_tools.tools.retrieval.register_tools", retrieval_register
    )
    monkeypatch.setattr(
        "src.mcp_tools.tools.web_search.register_tools", web_search_register
    )
    return {"retrieval": retrieval_register, "web_search": web_search_register}


def test_init_registers_retrieval_and_web_search(
    tool_patches: dict[str, MagicMock],
) -> None:
    """Constructor should register both retrieval and web search tool sets."""

    vector_service = object()

    service = SearchService(vector_service=vector_service)  # type: ignore[arg-type]

    assert service.mcp.name == "search-service"
    tool_patches["retrieval"].assert_called_once_with(
        service.mcp, vector_service=vector_service
    )
    tool_patches["web_search"].assert_called_once_with(service.mcp)


def test_get_mcp_server_returns_instance() -> None:
    """`get_mcp_server` exposes the configured FastMCP instance."""

    service = SearchService("custom-search")

    assert service.get_mcp_server() is service.mcp
    assert service.get_mcp_server().name == "custom-search"


@pytest.mark.asyncio
async def test_get_service_info_returns_capabilities() -> None:
    """`get_service_info` exposes a stable capability list."""

    service = SearchService()

    info = await service.get_service_info()

    assert info["service"] == "search"
    assert info["status"] == "active"
    assert "vector_search" in info["capabilities"]
    assert "web_search" in info["capabilities"]
