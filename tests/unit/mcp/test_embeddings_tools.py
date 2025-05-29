"""Comprehensive tests for embeddings tools module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import Context
from src.infrastructure.client_manager import ClientManager
from src.mcp.models.requests import EmbeddingRequest
from src.mcp.tools import embeddings


class TestEmbeddingsTools:
    """Test embeddings tool functions."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager."""
        cm = Mock(spec=ClientManager)

        # Add unified_config for service initialization
        unified_config = Mock()
        unified_config.cache = Mock(enable_caching=True)
        unified_config.openai = Mock(api_key="test-key")
        unified_config.embedding = Mock(
            provider="openai", model="text-embedding-3-small"
        )
        cm.unified_config = unified_config

        return cm

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        ctx = Mock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance that captures registered tools."""
        mcp = Mock()
        mcp._tools = {}

        def tool_decorator(func=None, **kwargs):
            def wrapper(f):
                mcp._tools[f.__name__] = f
                return f

            return wrapper if func is None else wrapper(func)

        mcp.tool = tool_decorator
        return mcp

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that embedding tools are registered correctly."""
        # Register tools
        embeddings.register_tools(mock_mcp, mock_client_manager)

        # Check that tools were registered
        assert "generate_embeddings" in mock_mcp._tools
        assert "list_embedding_providers" in mock_mcp._tools

    @pytest.mark.asyncio
    async def test_generate_embeddings_basic(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test basic generate_embeddings functionality."""
        with patch("src.mcp.tools.embeddings.EmbeddingManager") as MockEmbeddingManager:
            # Configure mock
            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                return_value=Mock(
                    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    sparse_embeddings=None,
                    dimensions=3,
                    model="text-embedding-3-small",
                    total_tokens=20,
                )
            )
            mock_embedding_mgr.get_current_provider_info = Mock(
                return_value={
                    "name": "openai",
                    "model": "text-embedding-3-small",
                    "dimensions": 1536,
                    "max_tokens": 8191,
                }
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            # Register tools
            embeddings.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            embed_func = mock_mcp._tools["generate_embeddings"]

            # Create request
            request = EmbeddingRequest(
                texts=["test text 1", "test text 2"], model="text-embedding-3-small"
            )

            # Call the function (no ctx parameter)
            result = await embed_func(request)

            # Verify results
            assert result["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            assert result["count"] == 2
            assert result["dimensions"] == 3
            assert result["model"] == "text-embedding-3-small"
            assert result["provider"] == "openai"
            assert result["total_tokens"] == 20

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sparse(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test generate_embeddings with sparse embeddings."""
        with patch("src.mcp.tools.embeddings.EmbeddingManager") as MockEmbeddingManager:
            # Configure mock
            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                return_value=Mock(
                    embeddings=[[0.1, 0.2, 0.3]],
                    sparse_embeddings=[[0.0, 0.5, 0.0, 0.8]],
                    dimensions=3,
                    model="text-embedding-3-small",
                    total_tokens=10,
                )
            )
            mock_embedding_mgr.get_current_provider_info = Mock(
                return_value={"name": "openai", "model": "text-embedding-3-small"}
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            # Register tools
            embeddings.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            embed_func = mock_mcp._tools["generate_embeddings"]

            # Create request with sparse embeddings
            request = EmbeddingRequest(
                texts=["test with sparse"],
                model="text-embedding-3-small",
                generate_sparse=True,
            )

            # Call the function (no ctx parameter)
            result = await embed_func(request)

            # Verify results include sparse embeddings
            assert result["sparse_embeddings"] == [[0.0, 0.5, 0.0, 0.8]]
            assert result["sparse_embeddings"] is not None

    @pytest.mark.asyncio
    async def test_list_embedding_providers(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test list_embedding_providers functionality."""
        with patch("src.mcp.tools.embeddings.EmbeddingManager") as MockEmbeddingManager:
            # Configure mock
            mock_embedding_mgr = Mock()
            mock_embedding_mgr._openai_available = Mock(return_value=True)
            MockEmbeddingManager.return_value = mock_embedding_mgr

            # Register tools
            embeddings.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            list_func = mock_mcp._tools["list_embedding_providers"]

            # Call the function (no ctx parameter)
            result = await list_func()

            # Verify results - it returns a list of providers
            assert isinstance(result, list)
            assert len(result) == 2  # OpenAI and FastEmbed

            # Find providers by name
            openai_provider = next(p for p in result if p["name"] == "openai")
            fastembed_provider = next(p for p in result if p["name"] == "fastembed")

            # Verify OpenAI provider
            assert openai_provider["status"] == "available"
            assert len(openai_provider["models"]) == 3
            assert any(
                m["name"] == "text-embedding-3-small" for m in openai_provider["models"]
            )

            # Verify FastEmbed provider
            assert fastembed_provider["status"] == "available"
            assert len(fastembed_provider["models"]) == 3
            assert any(
                m["name"] == "BAAI/bge-small-en-v1.5"
                for m in fastembed_provider["models"]
            )

    @pytest.mark.asyncio
    async def test_generate_embeddings_error_handling(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test generate_embeddings error handling."""
        with patch("src.mcp.tools.embeddings.EmbeddingManager") as MockEmbeddingManager:
            # Configure mock to raise error
            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                side_effect=Exception("Embedding generation failed")
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            # Register tools
            embeddings.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            embed_func = mock_mcp._tools["generate_embeddings"]

            # Create request
            request = EmbeddingRequest(
                texts=["error test"], model="text-embedding-3-small"
            )

            # Call should raise error (no ctx parameter)
            with pytest.raises(Exception, match="Embedding generation failed"):
                await embed_func(request)

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_texts(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test generate_embeddings with empty texts."""
        with patch("src.mcp.tools.embeddings.EmbeddingManager") as MockEmbeddingManager:
            # Configure mock
            mock_embedding_mgr = AsyncMock()
            mock_embedding_mgr.generate_embeddings = AsyncMock(
                return_value=Mock(
                    embeddings=[],
                    sparse_embeddings=None,
                    dimensions=0,
                    model="text-embedding-3-small",
                    total_tokens=0,
                )
            )
            mock_embedding_mgr.get_current_provider_info = Mock(
                return_value={"name": "openai", "model": "text-embedding-3-small"}
            )
            MockEmbeddingManager.return_value = mock_embedding_mgr

            # Register tools
            embeddings.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            embed_func = mock_mcp._tools["generate_embeddings"]

            # Create request with empty texts
            request = EmbeddingRequest(texts=[], model="text-embedding-3-small")

            # Call the function (no ctx parameter)
            result = await embed_func(request)

            # Verify results
            assert result["embeddings"] == []
            assert result["count"] == 0
            assert result["total_tokens"] == 0
