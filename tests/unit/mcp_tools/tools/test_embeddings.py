"""Comprehensive test suite for MCP embeddings tools."""

# pylint: disable=duplicate-code

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.models.requests import EmbeddingRequest
from src.mcp_tools.models.responses import EmbeddingGenerationResponse
from src.mcp_tools.tools.embeddings import register_tools


class TestEmbeddingsTools:
    """Test suite for embeddings MCP tools."""

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager service."""
        mock_embedding = AsyncMock()

        mock_embedding.estimate_cost = MagicMock(
            return_value={
                "fastembed": {
                    "estimated_tokens": 20,
                    "cost_per_token": 0.0,
                    "total_cost": 0.0,
                }
            }
        )
        mock_embedding.get_provider_info = MagicMock(
            return_value={
                "fastembed": {
                    "model": "BAAI/bge-small-en-v1.5",
                    "dimensions": 384,
                    "max_tokens": 512,
                    "cost_per_token": 0.0,
                }
            }
        )

        # Create a smart mock that returns different results based on inputs
        class _EmbeddingResult(dict):
            """Dictionary-like container mirroring embedding service response."""

            def __init__(
                self,
                embeddings: list[list[float]],
                *,
                sparse: list[list[float]] | None,
                metadata: dict[str, float | str | None],
            ) -> None:
                payload = {
                    "embeddings": embeddings,
                    "sparse_embeddings": sparse,
                    **metadata,
                }
                super().__init__(payload)

            # Exclude token counters from items so downstream **kwargs stay unique
            def items(self):  # pylint: disable=invalid-name
                data = dict(self)
                data.pop("tokens", None)
                return data.items()

        async def mock_generate_embeddings(
            texts,
            options=None,
            **_kwargs,
        ):
            embeddings = [
                [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1, 0.4 + i * 0.1]
                for i in range(len(texts))
            ]
            provider_name = getattr(options, "provider_name", None)
            provider_name = (
                provider_name
                if provider_name is not None
                else _kwargs.get("provider_name")
            )
            generate_sparse = getattr(options, "generate_sparse", None)
            if generate_sparse is None:
                generate_sparse = _kwargs.get("generate_sparse", False)
            sparse = (
                [
                    [0.8 - i * 0.1, 0.0, 0.6 - i * 0.1, 0.0, 0.4 - i * 0.1]
                    for i in range(len(texts))
                ]
                if generate_sparse
                else None
            )
            return _EmbeddingResult(
                embeddings,
                sparse=sparse,
                metadata={
                    "model": provider_name or "BAAI/bge-small-en-v1.5",
                    "provider": provider_name or "fastembed",
                    "tokens": len(texts) * 10,
                },
            )

        mock_embedding.generate_embeddings.side_effect = mock_generate_embeddings

        # Make get_current_provider_info synchronous (not async)
        def mock_provider_info():
            return {
                "name": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
            }

        mock_embedding.get_current_provider_info = mock_provider_info

        return mock_embedding

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
    async def test_generate_embeddings_basic(
        self, mock_embedding_manager, mock_context
    ):
        """Test basic embedding generation."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

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
        assert result.cost_estimate == 0.0
        assert result.total_tokens == 20

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sparse(
        self, mock_embedding_manager, mock_context
    ):
        """Test embedding generation with sparse embeddings."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

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
        assert result.cost_estimate == 0.0

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_list_embedding_providers(
        self, mock_embedding_manager, mock_context, monkeypatch
    ):
        """Test getting available embedding providers."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        # Reset provider cache
        monkeypatch.setattr(
            "src.mcp_tools.tools.embeddings._provider_cache",
            None,
            raising=False,
        )
        monkeypatch.setattr(
            "src.mcp_tools.tools.embeddings._provider_cache_expiry",
            None,
            raising=False,
        )
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

        list_embedding_providers = registered_tools["list_embedding_providers"]

        result = await list_embedding_providers(ctx=mock_context)

        assert isinstance(result, list)
        assert result[0].name == "fastembed"
        assert result[0].models[0]["name"] == "BAAI/bge-small-en-v1.5"

        # Verify context logging
        mock_context.info.assert_called()

    @pytest.mark.asyncio
    async def test_embeddings_error_handling(
        self, mock_embedding_manager, mock_context
    ):
        """Test embeddings error handling."""
        mock_embedding_manager.generate_embeddings.side_effect = Exception(
            "Embedding service unavailable"
        )

        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

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
    async def test_empty_texts_list(self, mock_embedding_manager, mock_context):
        """Test handling empty texts list."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

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
    async def test_context_logging_integration(
        self, mock_embedding_manager, mock_context
    ):
        """Test that context logging is properly integrated."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

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

    def test_tool_registration(self, mock_embedding_manager):
        """Test that embedding tools are properly registered."""
        mock_mcp = MagicMock()
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

        # Should have registered 2 tools
        assert mock_mcp.tool.call_count == 2

    @pytest.mark.asyncio
    async def test_embedding_manager_interactions(
        self, mock_embedding_manager, mock_context
    ):
        """Test proper interaction with embedding manager."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

        generate_embeddings = registered_tools["generate_embeddings"]
        list_embedding_providers = registered_tools["list_embedding_providers"]

        # Test embedding manager is retrieved
        request = EmbeddingRequest(texts=["test"])
        await generate_embeddings(request, mock_context)
        mock_embedding_manager.generate_embeddings.assert_called()

        # Test providers call
        await list_embedding_providers(ctx=mock_context)
        # Note: The list_embedding_providers doesn't call get_current_provider_info

    @pytest.mark.asyncio
    async def test_batch_size_handling(self, mock_embedding_manager, mock_context):
        """Test different batch sizes."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

        generate_embeddings = registered_tools["generate_embeddings"]

        # Test different batch sizes
        batch_sizes = [1, 16, 32, 64]

        for batch_size in batch_sizes:
            request = EmbeddingRequest(texts=["test text"], batch_size=batch_size)
            result = await generate_embeddings(request, mock_context)
            assert isinstance(result, EmbeddingGenerationResponse)
            assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_custom_model_specification(
        self, mock_embedding_manager, mock_context
    ):
        """Test embedding generation with custom model."""
        mock_mcp = MagicMock()
        registered_tools = {}

        def capture_tool(func):
            registered_tools[func.__name__] = func
            return func

        mock_mcp.tool.return_value = capture_tool
        register_tools(mock_mcp, embedding_manager=mock_embedding_manager)

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
        mock_embedding_manager.generate_embeddings.assert_called()
        _call_args, call_kwargs = mock_embedding_manager.generate_embeddings.call_args
        assert call_kwargs.get("texts") == ["test text"]
        assert (
            call_kwargs.get("provider_name") == "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert call_kwargs.get("generate_sparse") is False
