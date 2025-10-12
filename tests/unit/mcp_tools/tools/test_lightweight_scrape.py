"""Tests for the lightweight scraping MCP tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.tools import lightweight_scrape as lightweight_scrape_module
from src.services.errors import CrawlServiceError


ToolSetup = tuple[
    Callable[..., Awaitable[dict[str, Any]]],
    MagicMock,
]


class StubMCP:
    """Stub FastMCP server recording registered tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}

    def tool(self, *_, **__):  # pragma: no cover - decorator wiring
        def decorator(func: Callable[..., Awaitable[dict[str, Any]]]):
            self.tools[func.__name__] = func
            return func

        return decorator


@pytest.fixture
def functions() -> dict[str, Callable[..., Any]]:
    """Expose private helpers for isolated testing."""

    return {
        "validate_formats": lightweight_scrape_module._validate_formats,
        "analyze_url": lightweight_scrape_module._analyze_url_suitability,
        "convert_formats": lightweight_scrape_module._convert_content_formats,
        "build_response": lightweight_scrape_module._build_success_response,
        "handle_failure": lightweight_scrape_module._handle_scrape_failure,
    }


def test_register_tools_uses_mcp_decorator() -> None:
    """Registering the tool should call the MCP decorator exactly once."""

    mcp = StubMCP()
    crawl_manager = MagicMock()

    lightweight_scrape_module.register_tools(mcp, crawl_manager=crawl_manager)

    assert "lightweight_scrape" in mcp.tools


@pytest.fixture
def ctx() -> AsyncMock:
    """Async MCP context stub with logging hooks."""

    context = AsyncMock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


@pytest.fixture
def tool_setup() -> ToolSetup:
    """Register the tool and wire a mocked crawl manager dependency."""

    crawl_manager = MagicMock()
    crawl_manager.analyze_url = AsyncMock(
        return_value={"recommended_tier": "lightweight"}
    )
    crawl_manager.scrape_url = AsyncMock()

    mcp = StubMCP()
    lightweight_scrape_module.register_tools(mcp, crawl_manager=crawl_manager)

    tool = mcp.tools["lightweight_scrape"]
    return tool, crawl_manager


@pytest.mark.asyncio
async def test_successful_scrape_returns_expected_payload(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Successful scrape should propagate metadata and performance metrics."""

    tool, crawl_manager = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "# Heading\n\nBody",
        "metadata": {
            "title": "Sample Page",
            "raw_html": "<h1>Heading</h1><p>Body</p>",
            "description": "Test description",
        },
        "provider": "lightweight",
        "quality_score": 0.9,
        "url": "https://example.com/page",
    }

    response = await tool(
        url="https://example.com/page",
        formats=["markdown"],
        ctx=ctx,
    )

    crawl_manager.scrape_url.assert_awaited_once_with(
        url="https://example.com/page", preferred_provider="lightweight"
    )

    assert response["success"] is True
    assert response["content"]["markdown"] == "# Heading\n\nBody"
    assert response["metadata"]["title"] == "Sample Page"
    assert response["metadata"]["url"] == "https://example.com/page"
    assert response["performance"]["provider"] == "lightweight"
    assert response["performance"]["suitable_for_tier"] is True


@pytest.mark.asyncio
async def test_multiple_formats_are_transformed(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Requested formats should be populated using the scrape result."""

    tool, crawl_manager = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "# Title",
        "metadata": {"raw_html": "<h1>Title</h1>"},
    }

    response = await tool(
        url="https://example.com/page",
        formats=["markdown", "html", "text"],
        ctx=ctx,
    )

    assert response["content"]["markdown"] == "# Title"
    assert response["content"]["html"] == "<h1>Title</h1>"
    assert response["content"]["text"] == "Title"


@pytest.mark.asyncio
async def test_url_not_suitable_logs_warning(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """When analyze_url flags a heavier tier, a warning should be emitted."""

    tool, crawl_manager = tool_setup
    crawl_manager.analyze_url.return_value = {"recommended_tier": "standard"}
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "",
        "metadata": {},
    }

    await tool(url="https://example.com/page", formats=None, ctx=ctx)

    crawl_manager.analyze_url.assert_awaited_once()
    assert ctx.warning.await_count == 1


@pytest.mark.asyncio
async def test_scrape_failure_raises_error(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Failed scrapes should raise CrawlServiceError with guidance."""

    tool, crawl_manager = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": False,
        "error": "boom",
        "failed_tiers": ["lightweight"],
        "provider": "lightweight",
    }

    with pytest.raises(CrawlServiceError):
        await tool(url="https://example.com/page", formats=None, ctx=ctx)


def test_validate_formats_rejects_unknown(
    functions: dict[str, Callable[..., Any]],
) -> None:
    """Unknown formats should raise ValueError."""

    validator = functions["validate_formats"]
    with pytest.raises(ValueError):
        validator(["markdown", "pdf"])


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"raw_html": "<p>Hi</p>"}, {"markdown": "Hi", "html": "<p>Hi</p>"}),
        (None, {"markdown": "Hi"}),
    ],
)
def test_convert_content_formats_handles_metadata(
    functions: dict[str, Callable[..., Any]],
    metadata: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> None:
    """Conversion helper should handle optional metadata gracefully."""

    result = functions["build_response"](
        {"content": "Hi", "metadata": metadata or {}, "url": "foo"},
        "foo",
        ["markdown", "html"] if "html" in expected else ["markdown"],
        elapsed_ms=1.0,
        can_handle=True,
    )

    for key, value in expected.items():
        assert result["content"][key] == value
