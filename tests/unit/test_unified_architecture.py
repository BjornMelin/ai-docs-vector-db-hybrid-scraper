"""Tests for the new unified MCP server architecture."""

import os
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.infrastructure.client_manager import ClientManager
from src.infrastructure.client_manager import ClientManagerConfig


class TestClientManager:
    """Test the ClientManager functionality."""

    def test_singleton_pattern(self):
        """Test that ClientManager follows singleton pattern."""
        # Reset singleton for clean test
        ClientManager._instance = None

        config1 = ClientManagerConfig(qdrant_url="http://localhost:6333")
        cm1 = ClientManager(config1)

        config2 = ClientManagerConfig(qdrant_url="http://localhost:6334")
        cm2 = ClientManager(config2)

        # Should be the same instance
        assert cm1 is cm2

        # Config should not change (first config wins)
        assert cm1.config.qdrant_url == "http://localhost:6333"

        # Cleanup
        ClientManager._instance = None

    def test_from_unified_config(self):
        """Test factory method creates ClientManager correctly."""
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://test:6333",
                "OPENAI_API_KEY": "test-key",
                "REDIS_URL": "redis://test:6379",
            },
        ):
            cm = ClientManager.from_unified_config()

            assert cm.config.qdrant_url == "http://test:6333"
            assert cm.config.openai_api_key == "test-key"
            assert cm.config.redis_url == "redis://test:6379"

    @pytest.mark.asyncio
    async def test_initialize_and_cleanup(self):
        """Test initialization and cleanup lifecycle."""
        cm = ClientManager(ClientManagerConfig())

        # Test initialization
        assert not cm._initialized
        await cm.initialize()
        assert cm._initialized

        # Test cleanup
        await cm.cleanup()
        assert not cm._initialized

    @pytest.mark.asyncio
    async def test_managed_client_context(self):
        """Test managed client context manager."""
        cm = ClientManager(ClientManagerConfig())

        # Mock get_qdrant_client
        mock_client = AsyncMock()
        cm.get_qdrant_client = AsyncMock(return_value=mock_client)

        # Use managed client
        async with cm.managed_client("qdrant") as client:
            assert client is mock_client

        cm.get_qdrant_client.assert_called_once()


class TestServiceLayerIntegration:
    """Test service layer integration with new architecture."""

    @pytest.mark.asyncio
    async def test_qdrant_service_with_client_manager(self):
        """Test QdrantService works with ClientManager."""
        from src.services.core.qdrant_service import QdrantService

        # Create mock client manager
        cm = Mock(spec=ClientManager)

        # Create service
        service = QdrantService(cm)

        # Verify it accepts ClientManager
        assert service is not None

    @pytest.mark.asyncio
    async def test_embedding_manager_with_client_manager(self):
        """Test EmbeddingManager works with ClientManager."""
        from src.services.embeddings.manager import EmbeddingManager

        # Create mock client manager
        cm = Mock(spec=ClientManager)

        # Create service
        service = EmbeddingManager(cm)

        # Verify it accepts ClientManager
        assert service is not None

    @pytest.mark.asyncio
    async def test_cache_manager_with_client_manager(self):
        """Test CacheManager works with ClientManager."""
        from src.services.cache.manager import CacheManager

        # Create mock client manager
        cm = Mock(spec=ClientManager)

        # Create service
        service = CacheManager(cm)

        # Verify it accepts ClientManager
        assert service is not None


class TestModularToolRegistration:
    """Test the modular tool registration system."""

    @pytest.mark.asyncio
    async def test_register_all_tools_imports(self):
        """Test that register_all_tools imports all tool modules."""
        from fastmcp import FastMCP
        from src.mcp.tool_registry import register_all_tools

        # Create mocks
        mcp = Mock(spec=FastMCP)
        cm = Mock(spec=ClientManager)

        # Mock all tool modules
        with patch("src.mcp.tool_registry.tools") as mock_tools:
            # Create mock modules with register_tools functions
            tool_modules = [
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
            ]

            for module_name in tool_modules:
                module = Mock()
                module.register_tools = Mock()
                setattr(mock_tools, module_name, module)

            # Call register_all_tools
            await register_all_tools(mcp, cm)

            # Verify all modules were called
            for module_name in tool_modules:
                module = getattr(mock_tools, module_name)
                module.register_tools.assert_called_once_with(mcp, cm)


class TestUnifiedServerLifecycle:
    """Test the unified server lifecycle management."""

    @pytest.mark.asyncio
    async def test_lifespan_initialization(self):
        """Test server lifespan initialization."""
        with patch("src.unified_mcp_server.ClientManager") as MockCM:
            with patch("src.unified_mcp_server.register_all_tools") as mock_register:
                # Import after patching
                from src.unified_mcp_server import lifespan

                # Setup mocks
                mock_cm_instance = AsyncMock()
                mock_cm_instance.initialize = AsyncMock()
                mock_cm_instance.cleanup = AsyncMock()
                MockCM.from_unified_config.return_value = mock_cm_instance

                # Run lifespan
                async with lifespan():
                    # Verify initialization
                    MockCM.from_unified_config.assert_called_once()
                    mock_cm_instance.initialize.assert_called_once()
                    mock_register.assert_called_once()

                # Verify cleanup
                mock_cm_instance.cleanup.assert_called_once()

    def test_transport_configuration(self):
        """Test transport configuration from environment."""
        # Test stdio (default)
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.unified_mcp_server.mcp") as mock_mcp:
                # Execute main block
                exec(
                    """
if __name__ == "__main__":
    transport = os.getenv("FASTMCP_TRANSPORT", "stdio")
    
    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("FASTMCP_HOST", "127.0.0.1"),
            port=int(os.getenv("FASTMCP_PORT", "8000"))
        )
    else:
        mcp.run(transport="stdio")
""",
                    {"os": os, "mcp": mock_mcp},
                )

                mock_mcp.run.assert_called_once_with(transport="stdio")

        # Test HTTP transport
        with patch.dict(
            os.environ,
            {
                "FASTMCP_TRANSPORT": "streamable-http",
                "FASTMCP_HOST": "0.0.0.0",
                "FASTMCP_PORT": "9000",
            },
        ):
            with patch("src.unified_mcp_server.mcp") as mock_mcp:
                # Execute main block
                exec(
                    """
if __name__ == "__main__":
    transport = os.getenv("FASTMCP_TRANSPORT", "stdio")
    
    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("FASTMCP_HOST", "127.0.0.1"),
            port=int(os.getenv("FASTMCP_PORT", "8000"))
        )
    else:
        mcp.run(transport="stdio")
""",
                    {"os": os, "mcp": mock_mcp},
                )

                mock_mcp.run.assert_called_once_with(
                    transport="streamable-http", host="0.0.0.0", port=9000
                )
