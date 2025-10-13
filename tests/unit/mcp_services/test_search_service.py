"""Unit tests for :mod:`src.mcp_services.search_service`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.mcp_services.search_service import SearchService


@pytest.fixture()
def vector_service() -> MagicMock:
    return MagicMock(name="VectorStoreService")


def test_registers_retrieval_and_web_tools(vector_service: MagicMock) -> None:
    with (
        patch("src.mcp_services.search_service.retrieval.register_tools") as retrieval,
        patch("src.mcp_services.search_service.web_search.register_tools") as web,
    ):
        service = SearchService(vector_service=vector_service)

    retrieval.assert_called_once_with(service.mcp, vector_service=vector_service)
    web.assert_called_once_with(service.mcp)


def test_get_mcp_server_exposes_fastmcp(vector_service: MagicMock) -> None:
    with (
        patch("src.mcp_services.search_service.retrieval.register_tools"),
        patch("src.mcp_services.search_service.web_search.register_tools"),
    ):
        service = SearchService(vector_service=vector_service)

    server = service.get_mcp_server()
    assert server is service.mcp
    assert server.name == "search-service"


@pytest.mark.asyncio()
async def test_get_service_info_reports_capabilities(vector_service: MagicMock) -> None:
    with (
        patch("src.mcp_services.search_service.retrieval.register_tools"),
        patch("src.mcp_services.search_service.web_search.register_tools"),
    ):
        service = SearchService(vector_service=vector_service)

    info = await service.get_service_info()

    assert info["service"] == "search"
    assert info["status"] == "active"
    assert "web_search" in info["capabilities"]
    assert "vector_search" in info["capabilities"]
