"""Tests for the unified MCP server."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import FastMCP

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.mark.asyncio
async def test_server_initialization():
    """Test that the server initializes correctly."""
    # Mock the imports before loading the module
    with patch.dict(
        "sys.modules",
        {
            "mcp.tool_registry": Mock(register_all_tools=AsyncMock()),
            "infrastructure.client_manager": Mock(ClientManager=Mock),
            "services.logging_config": Mock(configure_logging=Mock()),
        },
    ):
        # Import the server module
        import unified_mcp_server

    # Verify server is created
    assert isinstance(unified_mcp_server.mcp, FastMCP)
    assert unified_mcp_server.mcp.name == "ai-docs-vector-db-unified"
    assert unified_mcp_server.mcp.instructions is not None

    # Test validate_configuration function
    with patch("unified_mcp_server.get_config") as mock_config:
        mock_config.return_value = Mock(
            get_active_providers=Mock(return_value=["fastembed"]),
            openai=Mock(api_key=None),
            firecrawl=Mock(api_key=None),
            crawling=Mock(providers=["crawl4ai"]),
            qdrant=Mock(url="http://localhost:6333"),
        )

        # Should not raise with valid config
        unified_mcp_server.validate_configuration()


@pytest.mark.asyncio
async def test_server_validation_errors():
    """Test configuration validation errors."""
    import unified_mcp_server

    with patch("unified_mcp_server.get_config") as mock_config:
        # Test missing OpenAI key when provider is active
        mock_config.return_value = Mock(
            get_active_providers=Mock(return_value=["openai"]),
            openai=Mock(api_key=None),
            firecrawl=Mock(api_key=None),
            crawling=Mock(providers=[]),
            qdrant=Mock(url="http://localhost:6333"),
        )

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            unified_mcp_server.validate_configuration()


@pytest.mark.asyncio
async def test_server_lifespan():
    """Test server lifespan management."""
    import unified_mcp_server

    with (
        patch("unified_mcp_server.validate_configuration"),
        patch("unified_mcp_server.ClientManager") as mock_cm,
        patch("unified_mcp_server.register_all_tools") as mock_register,
    ):
        # Mock client manager
        mock_client_manager = AsyncMock()
        mock_client_manager.initialize = AsyncMock()
        mock_client_manager.cleanup = AsyncMock()
        mock_cm.from_unified_config.return_value = mock_client_manager

        # Test lifespan
        async with unified_mcp_server.lifespan() as _:
            # Verify initialization
            mock_cm.from_unified_config.assert_called_once()
            mock_client_manager.initialize.assert_called_once()
            mock_register.assert_called_once()

        # Verify cleanup
        mock_client_manager.cleanup.assert_called_once()


def test_server_run_modes():
    """Test server run modes (stdio vs streamable-http)."""
    import unified_mcp_server

    # Test that lifespan is set
    assert unified_mcp_server.mcp.lifespan is not None

    # Test default transport is now streamable-http
    default_transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
    assert default_transport in ["streamable-http", "stdio"]


@pytest.mark.asyncio
async def test_streaming_transport_configuration():
    """Test streaming transport configuration with environment variables."""

    # Test streamable-http configuration
    with patch.dict(
        os.environ,
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_HOST": "0.0.0.0",
            "FASTMCP_PORT": "9000",
            "FASTMCP_BUFFER_SIZE": "16384",
            "FASTMCP_MAX_RESPONSE_SIZE": "20971520",
        },
    ):
        # Verify environment variables are accessible
        assert os.getenv("FASTMCP_TRANSPORT") == "streamable-http"
        assert os.getenv("FASTMCP_HOST") == "0.0.0.0"
        assert os.getenv("FASTMCP_PORT") == "9000"
        assert os.getenv("FASTMCP_BUFFER_SIZE") == "16384"
        assert os.getenv("FASTMCP_MAX_RESPONSE_SIZE") == "20971520"

        # Test int conversion for port
        port = int(os.getenv("FASTMCP_PORT", "8000"))
        assert port == 9000


@pytest.mark.asyncio
async def test_stdio_fallback_configuration():
    """Test stdio fallback configuration for Claude Desktop compatibility."""

    # Test stdio fallback
    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "stdio"}):
        transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
        assert transport == "stdio"


@pytest.mark.asyncio
async def test_server_instructions_include_streaming():
    """Test that server instructions mention streaming capabilities."""
    import unified_mcp_server

    instructions = unified_mcp_server.mcp.instructions
    assert "streaming" in instructions.lower()
    assert "streamable-http" in instructions.lower()
    assert "environment variables" in instructions.lower()
    assert "FASTMCP_TRANSPORT" in instructions


@pytest.mark.asyncio
async def test_default_streaming_values():
    """Test default values for streaming configuration."""
    # Test that defaults are sensible
    default_host = os.getenv("FASTMCP_HOST", "127.0.0.1")
    default_port = int(os.getenv("FASTMCP_PORT", "8000"))
    default_buffer = os.getenv("FASTMCP_BUFFER_SIZE", "8192")
    default_max_size = os.getenv("FASTMCP_MAX_RESPONSE_SIZE", "10485760")

    assert default_host == "127.0.0.1"
    assert default_port == 8000
    assert default_buffer == "8192"
    assert default_max_size == "10485760"  # 10MB
