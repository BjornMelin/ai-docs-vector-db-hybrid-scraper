"""Tests for the lightweight scraping MCP tool."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.errors import CrawlServiceError


ToolSetup = tuple[
    Callable[..., Awaitable[dict[str, Any]]],
    MagicMock,
    AsyncMock,
    MagicMock,
]


@pytest.fixture(scope="module")
def lightweight_scrape_module() -> ModuleType:
    """Load the lightweight_scrape module without importing the full tool package."""

    module_name = "tests.unit.mcp_tools.tools.lightweight_scrape_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = (
        Path(__file__).resolve().parents[4]
        / "src/mcp_tools/tools/lightweight_scrape.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to load lightweight_scrape module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_register_tools_uses_mcp_decorator(
    lightweight_scrape_module: ModuleType,
) -> None:
    """Registering the tool should call the MCP decorator exactly once."""

    register_tools = lightweight_scrape_module.register_tools

    registry: dict[str, Callable[..., Awaitable[dict[str, object]]]] = {}

    def capture(func: Callable[..., Awaitable[dict[str, object]]]):
        registry[func.__name__] = func
        return func

    mcp = MagicMock()
    mcp.tool.return_value = capture
    client_manager = MagicMock()

    register_tools(mcp, client_manager)

    mcp.tool.assert_called_once()
    assert "lightweight_scrape" in registry


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
def tool_setup(
    lightweight_scrape_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> ToolSetup:
    """Register the tool and wire a mocked crawl manager dependency."""

    register_tools = lightweight_scrape_module.register_tools

    crawl_manager = MagicMock()
    crawl_manager.analyze_url = AsyncMock(
        return_value={"recommended_tier": "lightweight"}
    )
    crawl_manager.scrape_url = AsyncMock()

    get_crawl_manager_stub = AsyncMock(return_value=crawl_manager)
    monkeypatch.setattr(
        lightweight_scrape_module,
        "get_crawl_manager",
        get_crawl_manager_stub,
    )

    registry: dict[str, Callable[..., Awaitable[dict[str, object]]]] = {}

    def capture(func: Callable[..., Awaitable[dict[str, object]]]):
        registry[func.__name__] = func
        return func

    mcp = MagicMock()
    mcp.tool.return_value = capture

    client_manager = MagicMock()
    register_tools(mcp, client_manager)

    tool = registry["lightweight_scrape"]
    return tool, crawl_manager, get_crawl_manager_stub, client_manager


@pytest.mark.asyncio
async def test_successful_scrape_returns_expected_payload(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Successful scrape should propagate metadata and performance metrics."""

    tool, crawl_manager, get_crawl_manager_stub, client_manager = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "# Heading\n\nBody",
        "metadata": {
            "title": "Sample Page",
            "raw_html": "<h1>Heading</h1><p>Body</p>",
            "description": "Test description",
        },
        "tier_used": "lightweight",
        "quality_score": 0.9,
        "url": "https://example.com/page",
    }

    response = cast(
        dict[str, Any],
        await tool(
            url="https://example.com/page",
            formats=["markdown"],
            ctx=ctx,
        ),
    )

    get_crawl_manager_stub.assert_awaited_once_with(client_manager)
    crawl_manager.scrape_url.assert_awaited_once_with(
        url="https://example.com/page", preferred_provider="lightweight"
    )

    assert response["success"] is True
    assert response["content"]["markdown"] == "# Heading\n\nBody"
    assert response["metadata"]["title"] == "Sample Page"
    assert response["metadata"]["url"] == "https://example.com/page"
    assert response["performance"]["tier"] == "lightweight"
    assert response["performance"]["suitable_for_tier"] is True
    assert response["performance"]["elapsed_ms"] >= 0

    assert any(
        call.args[0] == "Starting lightweight scrape of https://example.com/page"
        for call in ctx.info.await_args_list
    )
    assert any(
        "Successfully scraped https://example.com/page" in call.args[0]
        for call in ctx.info.await_args_list
    )


@pytest.mark.asyncio
async def test_multiple_formats_are_transformed(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Requested formats should be populated using the scrape result."""

    tool, crawl_manager, _, _ = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "# Title",
        "metadata": {"raw_html": "<h1>Title</h1>"},
        "tier_used": "lightweight",
    }

    response = cast(
        dict[str, Any],
        await tool(
            url="https://example.com",
            formats=["markdown", "html", "text"],
            ctx=ctx,
        ),
    )

    assert response["content"]["markdown"] == "# Title"
    assert response["content"]["html"] == "<h1>Title</h1>"
    assert response["content"]["text"].strip() == "Title"


@pytest.mark.asyncio
async def test_invalid_format_raises_value_error(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Supplying an unsupported format should raise ValueError before scraping."""

    tool, crawl_manager, get_crawl_manager_stub, _ = tool_setup

    with pytest.raises(ValueError, match=r"Invalid formats: {'xml'}"):
        await tool(url="https://example.com", formats=["markdown", "xml"], ctx=ctx)

    get_crawl_manager_stub.assert_not_awaited()
    crawl_manager.scrape_url.assert_not_called()


@pytest.mark.asyncio
async def test_non_lightweight_recommendation_triggers_warning(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """URL suitability analysis should warn when the lightweight tier is discouraged."""

    tool, crawl_manager, _, _ = tool_setup
    crawl_manager.analyze_url.return_value = {
        "recommended_tier": "browser",
        "reason": "Dynamic content",
    }
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "content",
        "metadata": {},
        "tier_used": "lightweight",
    }

    await tool(url="https://example.com/app", formats=None, ctx=ctx)

    assert ctx.warning.await_count == 1
    warning_args = ctx.warning.await_args_list[0].args[0]
    assert "may not be optimal for lightweight scraping" in warning_args


@pytest.mark.asyncio
async def test_scrape_failure_raises_crawl_service_error(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Failed scrapes should raise CrawlServiceError and emit context logs."""

    tool, crawl_manager, _, _ = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": False,
        "error": "Extraction failed",
        "failed_tiers": ["lightweight"],
    }

    with pytest.raises(CrawlServiceError, match="Lightweight scraping failed:"):
        await tool(url="https://example.com/failure", formats=None, ctx=ctx)

    ctx.error.assert_awaited()
    ctx.info.assert_awaited_with(
        "Lightweight tier failed. This content requires browser-based scraping. "
        "Consider using standard search or crawl tools."
    )


@pytest.mark.asyncio
async def test_analyze_url_exception_defaults_to_success_path(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Analyzer failures should default to allowing the scrape attempt."""

    tool, crawl_manager, _, _ = tool_setup
    crawl_manager.analyze_url.side_effect = TimeoutError("Analyzer unavailable")
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "ok",
        "metadata": {},
        "tier_used": "lightweight",
    }

    response = cast(
        dict[str, Any], await tool(url="https://example.com", formats=None, ctx=ctx)
    )

    assert response["performance"]["suitable_for_tier"] is True
    assert ctx.warning.await_count == 0


@pytest.mark.asyncio
async def test_default_format_is_markdown(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """When no formats are provided the tool should return markdown content."""

    tool, crawl_manager, _, _ = tool_setup
    crawl_manager.scrape_url.return_value = {
        "success": True,
        "content": "# Default",
        "metadata": {},
        "tier_used": "lightweight",
    }

    response = cast(
        dict[str, Any], await tool(url="https://example.com", formats=None, ctx=ctx)
    )

    assert response["content"] == {"markdown": "# Default"}


def test_validate_formats_defaults_to_markdown(
    lightweight_scrape_module: ModuleType,
) -> None:
    """_validate_formats should use markdown when no formats are provided."""

    validate_formats = lightweight_scrape_module._validate_formats

    assert validate_formats(None) == ["markdown"]


def test_convert_content_formats_honors_metadata_raw_html(
    lightweight_scrape_module: ModuleType,
) -> None:
    """_convert_content_formats should prefer raw_html for HTML output."""

    convert = lightweight_scrape_module._convert_content_formats

    result = {
        "metadata": {"raw_html": "<article>body</article>"},
    }
    converted = convert("# body", ["html"], result)

    assert converted == {"html": "<article>body</article>"}


def test_convert_content_formats_strips_markdown_for_text(
    lightweight_scrape_module: ModuleType,
) -> None:
    """Plain-text conversion should remove simple markdown markers."""

    convert = lightweight_scrape_module._convert_content_formats

    converted = convert("# Title *bold*", ["text"], {"metadata": {}})

    assert converted == {"text": " Title bold"}
