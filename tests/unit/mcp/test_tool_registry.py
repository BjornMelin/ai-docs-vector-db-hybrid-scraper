"""Tests for the MCP tool registry system."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from src.infrastructure.client_manager import ClientManager
from src.mcp.tool_registry import register_all_tools


@pytest.mark.asyncio
async def test_register_all_tools():
    """Test that all tool modules are registered."""
    # Create mocks
    mcp = Mock(spec=FastMCP)
    mcp.tool = Mock(return_value=lambda f: f)  # Decorator mock

    client_manager = Mock(spec=ClientManager)

    # Mock all tool modules
    with patch("src.mcp.tools") as mock_tools:
        # Create mock modules
        for module_name in [
            "search",
            "documents",
            "embeddings",
            "collections",
            "projects",
            "advanced_search",
            "payload_indexing",
            "deployment",
            "analytics",
            "cache",
            "utilities",
        ]:
            module = Mock()
            module.register_tools = Mock()
            setattr(mock_tools, module_name, module)

        # Call register_all_tools
        await register_all_tools(mcp, client_manager)

        # Verify all modules were registered
        assert mock_tools.search.register_tools.called
        assert mock_tools.documents.register_tools.called
        assert mock_tools.embeddings.register_tools.called
        assert mock_tools.collections.register_tools.called
        assert mock_tools.projects.register_tools.called
        assert mock_tools.advanced_search.register_tools.called
        assert mock_tools.payload_indexing.register_tools.called
        assert mock_tools.deployment.register_tools.called
        assert mock_tools.analytics.register_tools.called
        assert mock_tools.cache.register_tools.called
        assert mock_tools.utilities.register_tools.called

        # Verify they were called with correct arguments
        for module_name in [
            "search",
            "documents",
            "embeddings",
            "collections",
            "projects",
            "advanced_search",
            "payload_indexing",
            "deployment",
            "analytics",
            "cache",
            "utilities",
        ]:
            module = getattr(mock_tools, module_name)
            module.register_tools.assert_called_once_with(mcp, client_manager)


@pytest.mark.asyncio
async def test_tool_registration_logs():
    """Test that tool registration logs appropriate messages."""
    mcp = Mock(spec=FastMCP)
    mcp.tool = Mock(return_value=lambda f: f)

    client_manager = Mock(spec=ClientManager)

    with patch("src.mcp.tools") as mock_tools:
        with patch("src.mcp.tool_registry.logger") as mock_logger:
            # Create mock modules
            for module_name in [
                "search",
                "documents",
                "embeddings",
                "collections",
                "projects",
                "advanced_search",
                "payload_indexing",
                "deployment",
                "analytics",
                "cache",
                "utilities",
            ]:
                module = Mock()
                module.register_tools = Mock()
                setattr(mock_tools, module_name, module)

            await register_all_tools(mcp, client_manager)

            # Verify logging
            assert mock_logger.info.call_count >= 4  # At least 4 info logs

            # Check for specific log messages
            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Registering core tools" in msg for msg in log_calls)
            assert any("Registering management tools" in msg for msg in log_calls)
            assert any("Registering advanced tools" in msg for msg in log_calls)
            assert any("Registering utility tools" in msg for msg in log_calls)
            assert any(
                "Successfully registered 11 tool modules" in msg for msg in log_calls
            )
