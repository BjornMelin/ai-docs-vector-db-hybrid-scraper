"""Tests for SearchService dependency wiring."""

from unittest.mock import MagicMock

from src.mcp_services.search_service import SearchService


def test_search_service_registers_tools(monkeypatch) -> None:
    """Ensure retrieval and web search tools are wired with the provided services."""

    vector_service = MagicMock()
    mock_retrieval = MagicMock()
    mock_web = MagicMock()

    monkeypatch.setattr(
        "src.mcp_tools.tools.retrieval", "register_tools", mock_retrieval
    )
    monkeypatch.setattr("src.mcp_tools.tools.web_search", "register_tools", mock_web)

    service = SearchService(vector_service=vector_service)

    mock_retrieval.assert_called_once_with(service.mcp, vector_service=vector_service)
    mock_web.assert_called_once_with(service.mcp)


def test_get_mcp_server_returns_instance() -> None:
    """SearchService exposes its FastMCP instance via get_mcp_server."""

    service = SearchService(vector_service=MagicMock())
    assert service.get_mcp_server() is service.mcp
