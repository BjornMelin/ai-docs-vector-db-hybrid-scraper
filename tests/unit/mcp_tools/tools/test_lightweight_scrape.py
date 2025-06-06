"""Tests for lightweight scraping MCP tool."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    manager = MagicMock()
    manager.config = MagicMock()
    manager.config.lightweight_scraper = MagicMock()
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

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=True)
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "# Test Content\n\nThis is test content."},
                "metadata": {
                    "title": "Test Page",
                    "description": "Test description",
                    "tier": "lightweight",
                },
            }
        )

        # Register the tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Set up the scraper on client manager
        mock_client_manager._lightweight_scraper = mock_scraper

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
        mock_scraper.can_handle.assert_called_once_with("https://example.com/test.md")
        mock_scraper.scrape_url.assert_called_once_with(
            "https://example.com/test.md", formats=["markdown"]
        )

    @pytest.mark.asyncio
    async def test_scrape_with_multiple_formats(
        self, mock_client_manager, mock_context
    ):
        """Test scraping with multiple output formats."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=True)
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {
                    "markdown": "# Test",
                    "html": "<h1>Test</h1>",
                    "text": "Test",
                },
                "metadata": {"tier": "lightweight"},
            }
        )

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Set up the scraper on client manager
        mock_client_manager._lightweight_scraper = mock_scraper

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

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=False)  # URL not suitable
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "Content"},
                "metadata": {"tier": "lightweight"},
            }
        )

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Set up the scraper on client manager
        mock_client_manager._lightweight_scraper = mock_scraper

        # Test the tool
        await tool_func(url="https://spa.example.com", ctx=mock_context)

        # Verify warning was logged
        mock_context.warning.assert_called_once()
        warning_msg = mock_context.warning.call_args[0][0]
        assert "not suitable for lightweight scraping" in warning_msg

    @pytest.mark.asyncio
    async def test_scraping_failure_with_escalation(
        self, mock_client_manager, mock_context
    ):
        """Test handling of scraping failure that should escalate."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=True)
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": False,
                "error": "Insufficient content extracted",
                "should_escalate": True,
            }
        )

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Set up the scraper on client manager
        mock_client_manager._lightweight_scraper = mock_scraper

        # Test the tool
        with pytest.raises(CrawlServiceError, match="Try browser-based tools"):
            await tool_func(url="https://example.com", ctx=mock_context)

        # Verify error logging
        mock_context.error.assert_called()
        mock_context.info.assert_any_call(
            "This content requires browser-based scraping. "
            "Consider using standard search or crawl tools."
        )

    @pytest.mark.asyncio
    async def test_scraper_initialization_once(self, mock_client_manager, mock_context):
        """Test that scraper is initialized only once and reused."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=True)
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "Content"},
                "metadata": {"tier": "lightweight"},
            }
        )
        mock_scraper.initialize = AsyncMock()

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
        with patch(
            "src.mcp_tools.tools.lightweight_scrape.LightweightScraper"
        ) as mock_class:
            mock_class.return_value = mock_scraper

            # Delete _lightweight_scraper if it exists (MagicMock creates it dynamically)
            if hasattr(mock_client_manager, "_lightweight_scraper"):
                delattr(mock_client_manager, "_lightweight_scraper")

            # First call - scraper should be created
            await tool_func(url="https://example1.com", ctx=mock_context)

            # Second call - scraper should be reused (now it exists on client_manager)
            await tool_func(url="https://example2.com", ctx=mock_context)

        # Verify scraper was created only once
        mock_class.assert_called_once()
        mock_scraper.initialize.assert_called_once()

        # Verify both URLs were scraped
        assert mock_scraper.scrape_url.call_count == 2

    @pytest.mark.asyncio
    async def test_default_format_is_markdown(self, mock_client_manager, mock_context):
        """Test that default format is markdown when not specified."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=True)
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "# Default"},
                "metadata": {"tier": "lightweight"},
            }
        )

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Set up the scraper on client manager
        mock_client_manager._lightweight_scraper = mock_scraper

        # Test without specifying formats
        await tool_func(url="https://example.com", ctx=mock_context)

        # Verify markdown format was used
        mock_scraper.scrape_url.assert_called_once_with(
            "https://example.com", formats=["markdown"]
        )

    @pytest.mark.asyncio
    async def test_performance_metrics_added(self, mock_client_manager, mock_context):
        """Test that performance metrics are added to successful results."""
        from src.mcp_tools.tools.lightweight_scrape import register_tools

        # Create mock scraper
        mock_scraper = AsyncMock()
        mock_scraper.can_handle = AsyncMock(return_value=True)
        mock_scraper.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "Content"},
                "metadata": {"tier": "lightweight"},
            }
        )

        # Register and get tool
        mock_mcp = MagicMock()
        tool_func = None

        def capture_tool(func):
            nonlocal tool_func
            tool_func = func
            return func

        mock_mcp.tool = MagicMock(return_value=capture_tool)
        register_tools(mock_mcp, mock_client_manager)

        # Set up the scraper on client manager
        mock_client_manager._lightweight_scraper = mock_scraper

        # Test the tool
        result = await tool_func(url="https://example.com", ctx=mock_context)

        # Verify performance metrics
        assert "performance" in result
        assert "elapsed_ms" in result["performance"]
        assert result["performance"]["tier"] == "lightweight"
        assert result["performance"]["suitable_for_tier"] is True
