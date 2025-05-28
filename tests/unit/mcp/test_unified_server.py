"""Tests for the unified MCP server."""

import os
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP


@pytest.mark.asyncio
async def test_server_initialization():
    """Test that the server initializes correctly."""
    with patch("unified_mcp_server.configure_logging"):
        with patch("unified_mcp_server.ClientManager") as mock_cm:
            with patch("unified_mcp_server.register_all_tools") as mock_register:
                # Import after patching
                from unified_mcp_server import lifespan
                from unified_mcp_server import mcp

                # Verify server is created
                assert isinstance(mcp, FastMCP)
                assert mcp.name == "ai-docs-vector-db-unified"
                assert mcp.instructions is not None

                # Mock client manager
                mock_client_manager = AsyncMock()
                mock_client_manager.initialize = AsyncMock()
                mock_client_manager.cleanup = AsyncMock()
                mock_cm.from_unified_config.return_value = mock_client_manager

                # Test lifespan
                async with lifespan() as _:
                    # Verify initialization
                    mock_cm.from_unified_config.assert_called_once()
                    mock_client_manager.initialize.assert_called_once()
                    mock_register.assert_called_once_with(mcp, mock_client_manager)

                # Verify cleanup
                mock_client_manager.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_server_lifespan_error_handling():
    """Test that the server handles initialization errors gracefully."""
    with patch("unified_mcp_server.configure_logging"):
        with patch("unified_mcp_server.ClientManager") as mock_cm:
            with patch("unified_mcp_server.register_all_tools"):
                from unified_mcp_server import lifespan

                # Mock client manager that fails
                mock_client_manager = AsyncMock()
                mock_client_manager.initialize = AsyncMock(
                    side_effect=Exception("Initialization failed")
                )
                mock_client_manager.cleanup = AsyncMock()
                mock_cm.from_unified_config.return_value = mock_client_manager

                # Test that error is propagated
                with pytest.raises(Exception, match="Initialization failed"):
                    async with lifespan():
                        pass

                # Verify cleanup is still called
                mock_client_manager.cleanup.assert_called_once()


def test_server_run_stdio():
    """Test that the server runs with stdio transport by default."""
    with patch("unified_mcp_server.configure_logging"):
        with patch.dict(os.environ, {}, clear=True):
            from unified_mcp_server import mcp

            with patch.object(mcp, "run") as mock_run:
                # Execute the main block
                exec(
                    compile(
                        open("src/unified_mcp_server.py")
                        .read()
                        .split('if __name__ == "__main__":')[1],
                        "unified_mcp_server.py",
                        "exec",
                    )
                )

                # Verify stdio transport is used
                mock_run.assert_called_once_with(transport="stdio")


def test_server_run_http():
    """Test that the server runs with HTTP transport when configured."""
    with patch("unified_mcp_server.configure_logging"):
        with patch.dict(
            os.environ,
            {
                "FASTMCP_TRANSPORT": "streamable-http",
                "FASTMCP_HOST": "0.0.0.0",
                "FASTMCP_PORT": "9000",
            },
        ):
            from unified_mcp_server import mcp

            with patch.object(mcp, "run") as mock_run:
                # Execute the main block
                exec(
                    compile(
                        open("src/unified_mcp_server.py")
                        .read()
                        .split('if __name__ == "__main__":')[1],
                        "unified_mcp_server.py",
                        "exec",
                    )
                )

                # Verify HTTP transport is used with correct settings
                mock_run.assert_called_once_with(
                    transport="streamable-http", host="0.0.0.0", port=9000
                )
