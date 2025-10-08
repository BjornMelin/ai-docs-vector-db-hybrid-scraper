"""Tests for the lightweight scraping MCP tool."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Awaitable, Callable, Mapping
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

    if "src.infrastructure.client_manager" not in sys.modules:
        client_manager_stub = ModuleType("src.infrastructure.client_manager")

        class _StubClientManager:  # pragma: no cover - simple stand-in
            """Minimal stub to avoid importing full infrastructure graph."""

        cast(Any, client_manager_stub).ClientManager = _StubClientManager
        sys.modules["src.infrastructure.client_manager"] = client_manager_stub

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to load lightweight_scrape module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def functions(lightweight_scrape_module: ModuleType) -> dict[str, Any]:
    """Expose private helpers for isolated testing."""

    return {
        "validate_formats": lightweight_scrape_module._validate_formats,
        "analyze_url": lightweight_scrape_module._analyze_url_suitability,
        "convert_formats": lightweight_scrape_module._convert_content_formats,
        "build_response": lightweight_scrape_module._build_success_response,
        "handle_failure": lightweight_scrape_module._handle_scrape_failure,
    }


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
) -> ToolSetup:
    """Register the tool and wire a mocked crawl manager dependency."""

    register_tools = lightweight_scrape_module.register_tools

    crawl_manager = MagicMock()
    crawl_manager.analyze_url = AsyncMock(
        return_value={"recommended_tier": "lightweight"}
    )
    crawl_manager.scrape_url = AsyncMock()

    registry: dict[str, Callable[..., Awaitable[dict[str, object]]]] = {}

    def capture(func: Callable[..., Awaitable[dict[str, object]]]):
        registry[func.__name__] = func
        return func

    mcp = MagicMock()
    mcp.tool.return_value = capture

    client_manager = MagicMock()
    client_manager.get_crawl_manager = AsyncMock(return_value=crawl_manager)
    register_tools(mcp, client_manager)

    tool = registry["lightweight_scrape"]
    return tool, crawl_manager, client_manager.get_crawl_manager


@pytest.mark.asyncio
async def test_successful_scrape_returns_expected_payload(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """Successful scrape should propagate metadata and performance metrics."""

    tool, crawl_manager, crawl_getter = tool_setup
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

    crawl_getter.assert_awaited_once()
    crawl_manager.scrape_url.assert_awaited_once_with(
        url="https://example.com/page", tier="lightweight"
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

    tool, crawl_manager, _ = tool_setup
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

    tool, crawl_manager, crawl_getter = tool_setup

    with pytest.raises(ValueError, match=r"Invalid formats: {'xml'}"):
        await tool(url="https://example.com", formats=["markdown", "xml"], ctx=ctx)

    crawl_getter.assert_not_awaited()
    crawl_manager.scrape_url.assert_not_called()


@pytest.mark.asyncio
async def test_non_lightweight_recommendation_triggers_warning(
    tool_setup: ToolSetup,
    ctx: AsyncMock,
) -> None:
    """URL suitability analysis should warn when the lightweight tier is discouraged."""

    tool, crawl_manager, _ = tool_setup
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

    tool, crawl_manager, _ = tool_setup
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

    tool, crawl_manager, _ = tool_setup
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

    tool, crawl_manager, _ = tool_setup
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


def test_validate_formats_defaults_to_markdown(functions: Mapping[str, Any]) -> None:
    """_validate_formats should use markdown when no formats are provided."""

    validate_formats = functions["validate_formats"]

    assert validate_formats(None) == ["markdown"]


def test_validate_formats_rejects_invalid_entries(functions: Mapping[str, Any]) -> None:
    """Unsupported formats should raise ValueError with helpful messaging."""

    validate_formats = functions["validate_formats"]

    with pytest.raises(ValueError, match=r"Invalid formats: {'.*'}"):
        validate_formats(["markdown", "xml"])


@pytest.mark.asyncio
async def test_analyze_url_suitability_allows_missing_capability(
    functions: Mapping[str, Any],
) -> None:
    """Absent crawl manager capabilities should default to allowing the scrape."""

    analyze_url = functions["analyze_url"]

    result = await analyze_url(None, "https://example.com", None)

    assert result is True


@pytest.mark.asyncio
async def test_analyze_url_suitability_warns_on_other_tier(
    functions: Mapping[str, Any], ctx: AsyncMock
) -> None:
    """Warnings should be emitted when the recommended tier is not lightweight."""

    analyze_url = functions["analyze_url"]

    crawl_manager = MagicMock()
    crawl_manager.analyze_url = AsyncMock(
        return_value={"recommended_tier": "browser", "reason": "dynamic"}
    )

    can_handle = await analyze_url(crawl_manager, "https://example.com/app", ctx)

    assert can_handle is False
    ctx.warning.assert_awaited()


@pytest.mark.asyncio
async def test_analyze_url_suitability_recovers_from_exception(
    functions: Mapping[str, Any],
) -> None:
    """Exceptions raised during analysis should default to allowing lightweight tier."""

    analyze_url = functions["analyze_url"]

    crawl_manager = MagicMock()
    crawl_manager.analyze_url = AsyncMock(side_effect=TimeoutError("boom"))

    can_handle = await analyze_url(crawl_manager, "https://example.com/app", None)

    assert can_handle is True


def test_convert_content_formats_honors_metadata_raw_html(
    functions: Mapping[str, Any],
) -> None:
    """_convert_content_formats should prefer raw_html for HTML output."""

    convert = functions["convert_formats"]

    result = {
        "metadata": {"raw_html": "<article>body</article>"},
    }
    converted = convert("# body", ["html"], result)

    assert converted == {"html": "<article>body</article>"}


def test_convert_content_formats_strips_markdown_for_text(
    functions: Mapping[str, Any],
) -> None:
    """Plain-text conversion should remove simple markdown markers."""

    convert = functions["convert_formats"]

    converted = convert("# Title *bold*", ["text"], {"metadata": {}})

    assert converted == {"text": " Title bold"}


def test_build_success_response_merges_metadata(functions: Mapping[str, Any]) -> None:
    """_build_success_response should combine metadata and performance details."""

    build_response = functions["build_response"]

    scrape_result: Mapping[str, Any] = {
        "content": "# content",
        "title": "Example",
        "url": "https://example.com/source",
        "tier_used": "lightweight",
        "quality_score": 0.42,
        "metadata": {"raw_html": "<h1>Example</h1>", "author": "Docs"},
    }

    response = build_response(
        scrape_result,
        "https://example.com",
        ["html"],
        12.5,
        True,
    )

    assert response["success"] is True
    assert response["metadata"]["title"] == "Example"
    assert response["content"]["html"] == "<h1>Example</h1>"
    assert response["performance"]["tier"] == "lightweight"
    assert response["performance"]["suitable_for_tier"] is True


@pytest.mark.asyncio
async def test_handle_scrape_failure_raises_error_with_context(
    functions: Mapping[str, Any], ctx: AsyncMock
) -> None:
    """_handle_scrape_failure raises CrawlServiceError and emits logs."""

    handle_failure = functions["handle_failure"]

    failure_payload: Mapping[str, Any] = {
        "error": "Timeout",
        "failed_tiers": ["lightweight", "browser"],
    }

    with pytest.raises(CrawlServiceError, match="Lightweight scraping failed: Timeout"):
        await handle_failure(failure_payload, "https://example.com", ctx)

    ctx.error.assert_awaited()
    ctx.info.assert_awaited()


@pytest.mark.asyncio
async def test_handle_scrape_failure_handles_non_iterable_failed_tiers(
    functions: Mapping[str, Any],
) -> None:
    """Failed tier values that are not iterables should still be coerced into lists."""

    handle_failure = functions["handle_failure"]

    failure_payload: Mapping[str, Any] = {
        "error": "Boom",
        "failed_tiers": "lightweight",
    }

    with pytest.raises(CrawlServiceError, match="Boom"):
        await handle_failure(failure_payload, "https://example.com", None)
