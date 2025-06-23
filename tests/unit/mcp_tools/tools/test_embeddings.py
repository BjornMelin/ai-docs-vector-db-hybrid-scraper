"""Comprehensive test suite for MCP embeddings tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from src.mcp_tools.models.requests import EmbeddingRequest
from src.mcp_tools.models.responses import EmbeddingGenerationResponse


class TestEmbeddingsTools:
    """Test suite for embeddings MCP tools."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create a mock client manager with embedding service."""
        mock_manager = MagicMock()

        # Mock embedding manager
        mock_embedding = AsyncMock()

        # Create a smart mock that returns different results based on inputs
        async def mock_generate_embeddings(
            texts, model=None, batch_size=32, generate_sparse=False
        ):
            mock_result = MagicMock()
            # Generate embeddings based on number of input texts
            mock_result.embeddings = [
                [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1, 0.4 + i * 0.1]
                for i in range(len(texts))
            ]
            mock_result.sparse_embeddings = (
                [
                    [0.8 - i * 0.1, 0.0, 0.6 - i * 0.1, 0.0, 0.4 - i * 0.1]
                    for i in range(len(texts))
                ]
                if generate_sparse
                else None
            )
            mock_result.model = model if model else "BAAI/bge-small-en-v1.5"
            mock_result.total_tokens = len(texts) * 10  # 10 tokens per text
            return mock_result

        mock_embedding.generate_embeddings.side_effect = mock_generate_embeddings

        # Make get_current_provider_info synchronous (not async)
        def mock_provider_info():
            return {
                "name": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
            }

        mock_embedding.get_current_provider_info = mock_provider_info
        mock_manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)

        return mock_manager

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.debug = AsyncMock()
        mock_ctx.error = AsyncMock()
        mock_ctx.warning = AsyncMock()
        return mock_ctx

    @pytest.mark.asyncio
    async def test_generate_embeddings_basic(self, mock_client_manager, mock_context):
        """Test basic embedding generation."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        request = EmbeddingRequest(
            texts=["hello world", "machine learning"],
            model=None,
            batch_size=32,
            generate_sparse=False,
        )

        result = await generate_embeddings(request, mock_context)

        assert isinstance(result, EmbeddingGenerationResponse)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 4  # 4 dimensions
        assert len(result.embeddings[1]) == 4
        assert result.model == "BAAI/bge-small-en-v1.5"
        assert result.sparse_embeddings is None

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sparse(
        self, mock_client_manager, mock_context
    ):
        """Test embedding generation with sparse embeddings."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        request = EmbeddingRequest(
            texts=["document analysis", "vector search"],
            model="custom-model",
            batch_size=16,
            generate_sparse=True,
        )

        result = await generate_embeddings(request, mock_context)

        assert isinstance(result, EmbeddingGenerationResponse)
        assert len(result.embeddings) == 2
        assert result.sparse_embeddings is not None
        assert len(result.sparse_embeddings) == 2
        assert len(result.sparse_embeddings[0]) == 5
        assert len(result.sparse_embeddings[1]) == 5

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_list_embedding_providers(self, mock_client_manager, mock_context):
        """Test getting available embedding providers."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        list_embedding_providers = registered_tools["list_embedding_providers"]

        result = await list_embedding_providers(ctx=mock_context)

        assert isinstance(result, list)
        # The implementation returns a list of provider info

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_embeddings_error_handling(self, mock_client_manager, mock_context):
        """Test embeddings error handling."""
        from src.mcp_tools.tools.embeddings import register_tools

        # Make embedding manager raise an exception
        mock_embedding = AsyncMock()
        mock_embedding.generate_embeddings.side_effect = Exception(
            "Embedding service unavailable"
        )
        mock_client_manager.get_embedding_manager = AsyncMock(
            return_value=mock_embedding
        )

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        request = EmbeddingRequest(
            texts=["test text"], model=None, batch_size=32, generate_sparse=False
        )

        # Should raise the exception after logging
        with pytest.raises(Exception, match="Embedding service unavailable"):
            await generate_embeddings(request, mock_context)

        # Error should be logged
        mock_context.error.assert_called()

    @pytest.mark.asyncio
    async def test_empty_texts_list(self, mock_client_manager, mock_context):
        """Test handling empty texts list."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        request = EmbeddingRequest(
            texts=[], model=None, batch_size=32, generate_sparse=False
        )

        result = await generate_embeddings(request, mock_context)

        assert isinstance(result, EmbeddingGenerationResponse)
        assert result.embeddings == []  # Empty list for empty input
        assert result.total_tokens == 0  # No tokens for empty input

        # No specific warning required - just handle gracefully

    def test_embedding_request_validation(self):
        """Test embedding request model validation."""
        # Test valid request
        request = EmbeddingRequest(
            texts=["test text", "another text"],
            model="custom-model",
            batch_size=16,
            generate_sparse=True,
        )
        assert len(request.texts) == 2
        assert request.model == "custom-model"
        assert request.batch_size == 16
        assert request.generate_sparse is True

        # Test defaults
        default_request = EmbeddingRequest(texts=["test"])
        assert default_request.model is None
        assert default_request.batch_size == 32
        assert default_request.generate_sparse is False

    def test_embedding_response_validation(self):
        """Test embedding response model validation."""
        response = EmbeddingGenerationResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="test-model",
            sparse_embeddings=[[0.8, 0.6, 0.0]],
        )

        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 3
        assert response.model == "test-model"
        assert response.sparse_embeddings is not None
        assert len(response.sparse_embeddings) == 1

    @pytest.mark.asyncio
    async def test_context_logging_integration(self, mock_client_manager, mock_context):
        """Test that context logging is properly integrated."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]
        list_embedding_providers = registered_tools["list_embedding_providers"]

        # Test embeddings logging
        request = EmbeddingRequest(texts=["test"])
        await generate_embeddings(request, mock_context)
        assert mock_context.info.call_count >= 1

        # Reset and test providers logging
        mock_context.reset_mock()
        await list_embedding_providers(ctx=mock_context)
        assert mock_context.info.call_count >= 1

    def test_tool_registration(self, mock_client_manager):
        """Test that embedding tools are properly registered."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        register_tools(mock_mcp, mock_client_manager)

        # Should have registered 2 tools
        assert mock_mcp.tool.call_count == 2

    @pytest.mark.asyncio
    async def test_embedding_manager_interactions(
        self, mock_client_manager, mock_context
    ):
        """Test proper interaction with embedding manager."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]
        list_embedding_providers = registered_tools["list_embedding_providers"]

        # Test embedding manager is retrieved
        request = EmbeddingRequest(texts=["test"])
        await generate_embeddings(request, mock_context)
        mock_client_manager.get_embedding_manager.assert_called()

        # Test providers call
        await list_embedding_providers(ctx=mock_context)
        # Note: The list_embedding_providers doesn't call get_current_provider_info

    @pytest.mark.asyncio
    async def test_batch_size_handling(self, mock_client_manager, mock_context):
        """Test different batch sizes."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        # Test different batch sizes
        batch_sizes = [1, 16, 32, 64]

        for batch_size in batch_sizes:
            request = EmbeddingRequest(texts=["test text"], batch_size=batch_size)
            result = await generate_embeddings(request, mock_context)
            assert isinstance(result, EmbeddingGenerationResponse)
            assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_custom_model_specification(self, mock_client_manager, mock_context):
        """Test embedding generation with custom model."""
        from src.mcp_tools.tools.embeddings import register_tools

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, mock_client_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        request = EmbeddingRequest(
            texts=["test text"],
            model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            generate_sparse=False,
        )

        result = await generate_embeddings(request, mock_context)

        assert isinstance(result, EmbeddingGenerationResponse)
        assert len(result.embeddings) == 1

        # Verify embedding manager was called with custom model details
        embedding_manager = await mock_client_manager.get_embedding_manager()
        embedding_manager.generate_embeddings.assert_called()
