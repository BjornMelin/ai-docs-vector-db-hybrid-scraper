"""Tests for MCP tools web_search module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from src.mcp_tools.tools.web_search import register_tools


@pytest.fixture(autouse=True)
def mock_security_validator(monkeypatch):
    """Provide a permissive MLSecurityValidator stub."""

    validator = Mock()
    validator.validate_query_string.side_effect = lambda value: f"safe:{value}"

    monkeypatch.setattr(
        "src.mcp_tools.tools.web_search.MLSecurityValidator.from_unified_config",
        Mock(return_value=validator),
    )
    return validator


@pytest.fixture
def fake_tavily(monkeypatch):
    """Stub Tavily client capturing interactions."""

    calls: dict[str, object] = {}

    class _FakeTavilyClient:  # pylint: disable=too-few-public-methods
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key

        def search(self, **kwargs):  # noqa: ANN003 - mirror third-party API
            calls["search_kwargs"] = kwargs
            return {
                "results": [
                    {
                        "title": "Example Result",
                        "url": "https://example.com",
                        "score": 0.9,
                    }
                ],
                "answer": "Example summary",
                "images": ["https://example.com/image.png"],
                "response_time": 123,
            }

    monkeypatch.setattr(
        "src.mcp_tools.tools.web_search.TavilyClient",
        _FakeTavilyClient,
    )
    return calls


@pytest.mark.asyncio
async def test_web_search_success(
    fake_mcp, monkeypatch, fake_tavily, mock_security_validator
):
    """web_search returns formatted Tavily payload and logs via context."""

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()
    register_tools(fake_mcp)
    tool = fake_mcp.tools["web_search"]

    result = await tool(
        query="vector search",
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_images=True,
        include_domains=["example.com"],
        exclude_domains=["blocked.com"],
        ctx=ctx,
    )

    assert result["query"] == "safe:vector search"
    assert result["results"] and result["results"][0]["url"] == "https://example.com"
    assert result["answer"] == "Example summary"
    assert result["images"] == ["https://example.com/image.png"]
    assert fake_tavily["api_key"] == "test-key"

    kwargs = fake_tavily["search_kwargs"]
    assert kwargs["max_results"] == 3
    assert kwargs["search_depth"] == "advanced"
    assert kwargs["include_answer"] is True
    assert kwargs["include_images"] is True
    assert kwargs["include_domains"] == ["example.com"]
    assert kwargs["exclude_domains"] == ["blocked.com"]

    mock_security_validator.validate_query_string.assert_called_once_with(
        "vector search"
    )
    ctx.info.assert_awaited()
    ctx.error.assert_not_called()


@pytest.mark.asyncio
async def test_web_search_missing_api_key(fake_mcp, monkeypatch, fake_tavily):
    """web_search returns error details when API key is absent."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()

    register_tools(fake_mcp)
    tool = fake_mcp.tools["web_search"]

    result = await tool(query="secure query", ctx=ctx)

    assert result["results"] == []
    assert "error" in result
    ctx.error.assert_awaited()


@pytest.mark.asyncio
async def test_advanced_web_search_uses_advanced_depth(
    fake_mcp, monkeypatch, fake_tavily
):
    """advanced_web_search forces advanced depth and includes answer data."""

    monkeypatch.setenv("TAVILY_API_KEY", "another-key")
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.debug = AsyncMock()

    register_tools(fake_mcp)
    tool = fake_mcp.tools["advanced_web_search"]

    result = await tool(
        query="documentation", max_results=4, include_raw_content=True, ctx=ctx
    )

    assert result["query"] == "safe:documentation"
    assert result["results"]
    assert result["answer"] == "Example summary"
    assert fake_tavily["search_kwargs"]["search_depth"] == "advanced"
    assert fake_tavily["search_kwargs"]["include_raw_content"] is True
    ctx.info.assert_awaited()
