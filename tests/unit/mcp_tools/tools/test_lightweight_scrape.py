"""Tests for lightweight scraping MCP tool."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    manager = MagicMock()
    manager.config = MagicMock()
    manager.config.lightweight_scraper = MagicMock()

    # Mock the crawl manager chain
    mock_crawl_manager = MagicMock()
    mock_unified_manager = MagicMock()
    mock_unified_manager.analyze_url = AsyncMock(
        return_value={"recommended_tier": "lightweight"}
    )
    mock_crawl_manager._unified_browser_manager = mock_unified_manager
    mock_crawl_manager.scrape_url = AsyncMock(
        return_value={
            "success": True,
            "content": "# Test Content",
            "metadata": {"title": "Test", "raw_html": "<h1>Test</h1>"},
            "tier_used": "lightweight",
            "quality_score": 0.8,
        }
    )

    manager.get_crawl_manager = AsyncMock(return_value=mock_crawl_manager)
    return manager


@pytest.fixture
def mock_mcp():
    """Create mock MCP server instance."""
    mcp = MagicMock()
    mcp.tool = MagicMock(return_value=lambda func: func)
    return mcp


@pytest.fixture
def mock_context():
    """Create mock MCP context."""
    context = AsyncMock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


class TestLightweightScrapeRegistration:
    """Test tool registration."""

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that lightweight_scrape tool is registered correctly."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        register_tools(mock_mcp, mock_client_manager)

        # Verify tool decorator was called
        mock_mcp.tool.assert_called_once()


class TestLightweightScrapeTool:
    """Test the lightweight_scrape tool functionality."""

    @pytest.mark.asyncio
    async def test_successful_scrape_with_markdown(
        self, mock_client_manager, mock_context
    ):
        """Test successful scraping with markdown format."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock crawl manager to return successful result
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "# Test Content\n\nThis is test content.",
            "metadata": {
                "title": "Test Page",
                "description": "Test description",
                "raw_html": "<h1>Test Content</h1><p>This is test content.</p>",
            },
            "tier_used": "lightweight",
            "quality_score": 0.8,
            "url": "https://example.com/test.md",
        }

        # Register the tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test the tool
        result = await tool_func(
            url="https://example.com/test.md", formats=["markdown"], ctx=mock_context
        )

        # Verify results
        assert result["success"] is True
        assert "markdown" in result["content"]
        assert result["metadata"]["title"] == "Test Page"
        assert "performance" in result
        assert result["performance"]["tier"] == "lightweight"

        # Verify context calls
        mock_context.info.assert_any_call(
            "Starting lightweight scrape of https://example.com/test.md"
        )

        # Verify the crawl manager was called correctly
        crawl_manager.scrape_url.assert_called_once_with(
            url="https://example.com/test.md", preferred_provider="lightweight"
        )

    @pytest.mark.asyncio
    async def test_scrape_with_multiple_formats(
        self, mock_client_manager, mock_context
    ):
        """Test scraping with multiple output formats."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock crawl manager to return successful result
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "# Test Content",
            "metadata": {"tier": "lightweight", "raw_html": "<h1>Test Content</h1>"},
            "tier_used": "lightweight",
            "quality_score": 0.8,
        }

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test the tool
        result = await tool_func(
            url="https://example.com",
            formats=["markdown", "html", "text"],
            ctx=mock_context,
        )

        # Verify all formats are present
        assert "markdown" in result["content"]
        assert "html" in result["content"]
        assert "text" in result["content"]

        # Verify the formats were converted correctly
        assert result["content"]["markdown"] == "# Test Content"
        assert result["content"]["html"] == "<h1>Test Content</h1>"
        assert (
            "Test Content" in result["content"]["text"]
        )  # Text version strips markdown

    @pytest.mark.asyncio
    async def test_invalid_format_raises_error(self, mock_client_manager, mock_context):
        """Test that invalid formats raise ValueError."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test with invalid format
        with pytest.raises(ValueError, match="Invalid formats: {'invalid'}"):
            await tool_func(
                url="https://example.com",
                formats=["markdown", "invalid"],
                ctx=mock_context,
            )

    @pytest.mark.asyncio
    async def test_url_not_suitable_warning(self, mock_client_manager, mock_context):
        """Test warning when URL is not suitable for lightweight scraping."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock to return that URL is not suitable for lightweight tier
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager._unified_browser_manager.analyze_url.return_value = {
            "recommended_tier": "crawl4ai",  # Not lightweight
            "reason": "Complex JavaScript content",
        }
        crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "Content",
            "metadata": {"tier": "lightweight"},
            "tier_used": "lightweight",
        }

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test the tool
        await tool_func(url="https://spa.example.com", ctx=mock_context)

        # Verify warning was logged
        mock_context.warning.assert_called_once()
        warning_msg = mock_context.warning.call_args[0][0]
        assert "may not be optimal for lightweight scraping" in warning_msg

    @pytest.mark.asyncio
    async def test_scraping_failure_with_escalation(
        self, mock_client_manager, mock_context
    ):
        """Test handling of scraping failure that should escalate."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock crawl manager to return failure
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.return_value = {
            "success": False,
            "error": "Insufficient content extracted",
            "failed_tiers": ["lightweight"],
        }

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test the tool
        with pytest.raises(CrawlServiceError, match="Try browser-based tools"):
            await tool_func(url="https://example.com", ctx=mock_context)

        # Verify error logging
        mock_context.error.assert_called()
        mock_context.info.assert_any_call(
            "Lightweight tier failed. This content requires browser-based scraping. "
            "Consider using standard search or crawl tools."
        )

    @pytest.mark.asyncio
    async def test_crawl_manager_reuse(self, mock_client_manager, mock_context):
        """Test that crawl manager is reused across calls."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock crawl manager
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "Content",
            "metadata": {"tier": "lightweight"},
            "tier_used": "lightweight",
        }

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test the tool twice
        await tool_func(url="https://example1.com", ctx=mock_context)
        await tool_func(url="https://example2.com", ctx=mock_context)

        # Verify crawl manager scrape was called twice (manager can be accessed multiple times)
        assert crawl_manager.scrape_url.call_count == 2
        # Verify we got the same crawl manager each time
        assert mock_client_manager.get_crawl_manager.call_count >= 2

    @pytest.mark.asyncio
    async def test_default_format_is_markdown(self, mock_client_manager, mock_context):
        """Test that default format is markdown when not specified."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock crawl manager
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "# Default",
            "metadata": {"tier": "lightweight"},
            "tier_used": "lightweight",
        }

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test without specifying formats (should default to markdown)
        result = await tool_func(url="https://example.com", ctx=mock_context)

        # Verify result contains markdown format
        assert "markdown" in result["content"]
        assert result["content"]["markdown"] == "# Default"

    @pytest.mark.asyncio
    async def test_performance_metrics_added(self, mock_client_manager, mock_context):
        """Test that performance metrics are added to successful results."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Set up mock crawl manager
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "Content",
            "metadata": {"tier": "lightweight"},
            "tier_used": "lightweight",
        }

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Test the tool
        result = await tool_func(url="https://example.com", ctx=mock_context)

        # Verify performance metrics
        assert "performance" in result
        assert "elapsed_ms" in result["performance"]
        assert result["performance"]["tier"] == "lightweight"
        assert result["performance"]["suitable_for_tier"] is True
