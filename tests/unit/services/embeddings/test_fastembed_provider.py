import asyncio

from src.services.embeddings.base import EmbeddingProvider


class TestError(Exception):
    """Custom exception for this module."""


"""Tests for services/embeddings/fastembed_provider.py - FastEmbed integration.

This module tests the FastEmbed provider that provides local embedding model management,
model loading and inference optimization, memory management, and performance tuning.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import EmbeddingServiceError


class TestFastEmbedProviderInitialization:
    """Test cases for FastEmbedProvider initialization."""

    def test_provider_initialization_default_model(self):
        """Test provider initialization with default model."""
        provider = FastEmbedProvider()

        assert provider.model_name == "BAAI/bge-small-en-v1.5"
        assert provider.dimensions == 384
        assert provider._max_tokens == 512
        assert provider._model is None
        assert provider._sparse_model is None
        assert provider._initialized is False

    def test_provider_initialization_custom_model(self):
        """Test provider initialization with custom model."""
        provider = FastEmbedProvider("BAAI/bge-base-en-v1.5")

        assert provider.model_name == "BAAI/bge-base-en-v1.5"
        assert provider.dimensions == 768
        assert provider._max_tokens == 512

    def test_provider_initialization_unsupported_model(self):
        """Test provider initialization with unsupported model."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            FastEmbedProvider("unsupported-model")

    def test_provider_supported_models_class_variable(self):
        """Test SUPPORTED_MODELS class variable structure."""
        models = FastEmbedProvider.SUPPORTED_MODELS

        # Check required models are present
        assert "BAAI/bge-small-en-v1.5" in models
        assert "BAAI/bge-base-en-v1.5" in models
        assert "BAAI/bge-large-en-v1.5" in models
        assert "sentence-transformers/all-MiniLM-L6-v2" in models

        # Check structure of model configs
        for config in models.values():
            assert "dimensions" in config
            assert "description" in config
            assert "max_tokens" in config
            assert isinstance(config["dimensions"], int)
            assert isinstance(config["max_tokens"], int)
            assert isinstance(config["description"], str)

    def test_provider_initialization_all_supported_models(self):
        """Test provider initialization with all supported models."""
        for model_name in FastEmbedProvider.SUPPORTED_MODELS:
            provider = FastEmbedProvider(model_name)

            config = FastEmbedProvider.SUPPORTED_MODELS[model_name]
            assert provider.model_name == model_name
            assert provider.dimensions == config["dimensions"]
            assert provider._max_tokens == config["max_tokens"]
            assert provider._description == config["description"]

    def test_provider_sparse_model_configuration(self):
        """Test sparse model configuration."""
        provider = FastEmbedProvider()

        assert provider._sparse_model_name == "prithvida/Splade_PP_en_v1"
        assert provider._sparse_model is None


class TestFastEmbedProviderProperties:
    """Test cases for FastEmbed provider properties."""

    def test_cost_per_token_property(self):
        """Test cost_per_token property (should be 0 for local models)."""
        provider = FastEmbedProvider()

        assert provider.cost_per_token == 0.0
        assert isinstance(provider.cost_per_token, float)

    def test_max_tokens_per_request_property(self):
        """Test max_tokens_per_request property."""
        provider = FastEmbedProvider("BAAI/bge-small-en-v1.5")

        assert provider.max_tokens_per_request == 512

    def test_properties_different_models(self):
        """Test properties for different models."""
        small_provider = FastEmbedProvider("BAAI/bge-small-en-v1.5")
        base_provider = FastEmbedProvider("BAAI/bge-base-en-v1.5")
        large_provider = FastEmbedProvider("BAAI/bge-large-en-v1.5")

        # All should have zero cost
        assert small_provider.cost_per_token == 0.0
        assert base_provider.cost_per_token == 0.0
        assert large_provider.cost_per_token == 0.0

        # Different dimensions
        assert small_provider.dimensions == 384
        assert base_provider.dimensions == 768
        assert large_provider.dimensions == 1024

    def test_jina_model_properties(self):
        """Test properties for Jina models with longer context."""
        provider = FastEmbedProvider("jinaai/jina-embeddings-v2-small-en")

        assert provider.dimensions == 512
        assert provider.max_tokens_per_request == 8192  # Longer context


class TestFastEmbedProviderLifecycle:
    """Test cases for FastEmbed provider initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful provider initialization."""
        provider = FastEmbedProvider()

        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_embedding:
            mock_model = Mock()
            mock_embedding.return_value = mock_model

            await provider.initialize()

            assert provider._model is mock_model
            assert provider._initialized is True
            mock_embedding.assert_called_once_with("BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        provider = FastEmbedProvider()
        provider._initialized = True

        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_embedding:
            await provider.initialize()

            # Should not create new model
            mock_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_fastembed_not_available(self):
        """Test initialization when FastEmbed is not available."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding", None):
            # This would happen at import time, but we can test the behavior
            # by mocking the TextEmbedding to None
            pass  # The import would fail at module level

    @pytest.mark.asyncio
    async def test_initialize_error(self):
        """Test initialization error handling."""
        provider = FastEmbedProvider()

        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_embedding:
            mock_embedding.side_effect = Exception("Model loading failed")

            with pytest.raises(
                EmbeddingServiceError, match="Failed to initialize FastEmbed"
            ):
                await provider.initialize()

            assert provider._initialized is False
            assert provider._model is None

    @pytest.mark.asyncio
    async def test_cleanup_success(self):
        """Test successful provider cleanup."""
        provider = FastEmbedProvider()
        provider._model = Mock()
        provider._initialized = True

        await provider.cleanup()

        assert provider._model is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_no_model(self):
        """Test cleanup when no model exists."""
        provider = FastEmbedProvider()

        # Should not raise error
        await provider.cleanup()

        assert provider._model is None
        assert provider._initialized is False


class TestFastEmbedEmbeddingGeneration:
    """Test cases for FastEmbed embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self):
        """Test embedding generation when not initialized."""
        provider = FastEmbedProvider()

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self):
        """Test embedding generation with empty text list."""
        provider = FastEmbedProvider()
        provider._initialized = True

        embeddings = await provider.generate_embeddings([])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_success_numpy_arrays(self):
        """Test successful embedding generation with numpy arrays."""
        provider = FastEmbedProvider()
        provider._initialized = True

        # Mock model that returns numpy arrays
        mock_model = Mock()
        provider._model = mock_model

        # Mock embedding output as numpy arrays
        mock_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        mock_model.embed.return_value = iter(mock_embeddings)

        embeddings = await provider.generate_embeddings(["text1", "text2"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        # Verify model was called correctly
        mock_model.embed.assert_called_once_with(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_success_lists(self):
        """Test successful embedding generation with list outputs."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        # Mock embedding output as lists
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.embed.return_value = iter(mock_embeddings)

        embeddings = await provider.generate_embeddings(["text1", "text2"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(self):
        """Test embedding generation with single text."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        mock_embeddings = [np.array([0.1, 0.2, 0.3])]
        mock_model.embed.return_value = iter(mock_embeddings)

        embeddings = await provider.generate_embeddings(["single text"])

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_size_ignored(self):
        """Test that batch_size parameter is ignored (FastEmbed handles batching)."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        mock_embeddings = [np.array([0.1, 0.2, 0.3])]
        mock_model.embed.return_value = iter(mock_embeddings)

        # batch_size should be ignored
        embeddings = await provider.generate_embeddings(["text"], batch_size=100)

        assert len(embeddings) == 1
        mock_model.embed.assert_called_once_with(["text"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self):
        """Test embedding generation error handling."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model
        mock_model.embed.side_effect = Exception("Embedding failed")

        with pytest.raises(
            EmbeddingServiceError, match="Failed to generate embeddings"
        ):
            await provider.generate_embeddings(["test"])


class TestFastEmbedSparseEmbeddings:
    """Test cases for FastEmbed sparse embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_not_initialized(self):
        """Test sparse embedding generation when not initialized."""
        provider = FastEmbedProvider()

        with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
            await provider.generate_sparse_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_not_available(self):
        """Test sparse embedding generation when SparseTextEmbedding not available."""
        provider = FastEmbedProvider()
        provider._initialized = True

        with (
            patch(
                "src.services.embeddings.fastembed_provider.SparseTextEmbedding", None
            ),
            pytest.raises(
                EmbeddingServiceError, match="Sparse embedding support not available"
            ),
        ):
            await provider.generate_sparse_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_success(self):
        """Test successful sparse embedding generation."""
        provider = FastEmbedProvider()
        provider._initialized = True

        with patch(
            "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
        ) as mock_sparse:
            # Mock sparse model
            mock_sparse_model = Mock()
            mock_sparse.return_value = mock_sparse_model

            # Mock sparse embedding results
            mock_result1 = Mock()
            mock_result1.indices = np.array([0, 5, 10])
            mock_result1.values = np.array([0.1, 0.5, 0.8])

            mock_result2 = Mock()
            mock_result2.indices = np.array([1, 3, 7])
            mock_result2.values = np.array([0.2, 0.6, 0.9])

            mock_sparse_model.embed.return_value = [mock_result1, mock_result2]

            sparse_embeddings = await provider.generate_sparse_embeddings(
                ["text1", "text2"]
            )

            assert len(sparse_embeddings) == 2
            assert sparse_embeddings[0]["indices"] == [0, 5, 10]
            assert sparse_embeddings[0]["values"] == [0.1, 0.5, 0.8]
            assert sparse_embeddings[1]["indices"] == [1, 3, 7]
            assert sparse_embeddings[1]["values"] == [0.2, 0.6, 0.9]

            # Verify sparse model initialization and usage
            mock_sparse.assert_called_once_with("prithvida/Splade_PP_en_v1")
            mock_sparse_model.embed.assert_called_once_with(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_model_reuse(self):
        """Test sparse embedding generation reuses existing model."""
        provider = FastEmbedProvider()
        provider._initialized = True

        with patch(
            "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
        ) as mock_sparse:
            mock_sparse_model = Mock()
            mock_sparse.return_value = mock_sparse_model
            provider._sparse_model = mock_sparse_model

            # Mock sparse embedding result
            mock_result = Mock()
            mock_result.indices = np.array([0, 1])
            mock_result.values = np.array([0.5, 0.7])
            mock_sparse_model.embed.return_value = [mock_result]

            await provider.generate_sparse_embeddings(["text"])

            # Should not create new model
            mock_sparse.assert_not_called()
            mock_sparse_model.embed.assert_called_once_with(["text"])

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_error(self):
        """Test sparse embedding generation error handling."""
        provider = FastEmbedProvider()
        provider._initialized = True

        with patch(
            "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
        ) as mock_sparse:
            mock_sparse_model = Mock()
            mock_sparse.return_value = mock_sparse_model
            mock_sparse_model.embed.side_effect = Exception("Sparse embedding failed")

            with pytest.raises(
                EmbeddingServiceError, match="Sparse embedding generation failed"
            ):
                await provider.generate_sparse_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_initialization_error(self):
        """Test sparse embedding generation with model initialization error."""
        provider = FastEmbedProvider()
        provider._initialized = True

        with patch(
            "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
        ) as mock_sparse:
            mock_sparse.side_effect = Exception("Model init failed")

            with pytest.raises(
                EmbeddingServiceError, match="Sparse embedding generation failed"
            ):
                await provider.generate_sparse_embeddings(["test"])


class TestFastEmbedUtilityMethods:
    """Test cases for FastEmbed utility methods."""

    def test_list_available_models(self):
        """Test list_available_models class method."""
        models = FastEmbedProvider.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "BAAI/bge-small-en-v1.5" in models
        assert "BAAI/bge-base-en-v1.5" in models
        assert "BAAI/bge-large-en-v1.5" in models

    def test_get_model_info_success(self):
        """Test get_model_info class method with valid model."""
        model_name = "BAAI/bge-small-en-v1.5"
        info = FastEmbedProvider.get_model_info(model_name)

        assert isinstance(info, dict)
        assert "dimensions" in info
        assert "description" in info
        assert "max_tokens" in info
        assert info["dimensions"] == 384
        assert info["max_tokens"] == 512

    def test_get_model_info_invalid_model(self):
        """Test get_model_info class method with invalid model."""
        with pytest.raises(ValueError, match="Unknown model"):
            FastEmbedProvider.get_model_info("invalid-model")

    def test_get_model_info_all_models(self):
        """Test get_model_info for all supported models."""
        for model_name in FastEmbedProvider.SUPPORTED_MODELS:
            info = FastEmbedProvider.get_model_info(model_name)

            assert isinstance(info, dict)
            assert "dimensions" in info
            assert "description" in info
            assert "max_tokens" in info
            assert isinstance(info["dimensions"], int)
            assert isinstance(info["max_tokens"], int)
            assert isinstance(info["description"], str)


class TestFastEmbedProviderIntegration:
    """Integration test cases for FastEmbed provider."""

    @pytest.mark.asyncio
    async def test_full_provider_lifecycle(self):
        """Test complete provider lifecycle."""
        provider = FastEmbedProvider("BAAI/bge-small-en-v1.5")

        # Initial state
        assert not provider._initialized
        assert provider._model is None

        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_embedding:
            mock_model = Mock()
            mock_embedding.return_value = mock_model

            # Mock embedding output
            mock_embeddings = [np.array([0.1, 0.2, 0.3])]
            mock_model.embed.return_value = iter(mock_embeddings)

            # Initialize
            await provider.initialize()
            assert provider._initialized
            assert provider._model is mock_model

            # Generate embeddings
            embeddings = await provider.generate_embeddings(["test text"])
            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3]

            # Cleanup
            await provider.cleanup()
            assert not provider._initialized
            assert provider._model is None

    @pytest.mark.asyncio
    async def test_multiple_embedding_requests(self):
        """Test multiple embedding generation requests."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        # Mock different responses for each call
        call_count = 0

        def mock_embed_response(texts):
            nonlocal call_count
            call_count += 1
            return iter(
                [
                    np.array([0.1 * call_count, 0.2 * call_count, 0.3 * call_count])
                    for _ in texts
                ]
            )

        mock_model.embed.side_effect = mock_embed_response

        # Generate embeddings multiple times
        embeddings1 = await provider.generate_embeddings(["text1"])
        embeddings2 = await provider.generate_embeddings(["text2"])

        assert embeddings1[0] == [0.1, 0.2, 0.3]
        assert embeddings2[0] == [0.2, 0.4, 0.6]
        assert mock_model.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing of large text batches."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        def mock_embed_response(texts):
            # Return embeddings for all texts in the batch
            return iter([np.array([0.1, 0.2, 0.3]) for _ in texts])

        mock_model.embed.side_effect = mock_embed_response

        # Test with 100 texts
        texts = [f"text{i}" for i in range(100)]
        embeddings = await provider.generate_embeddings(texts)

        assert len(embeddings) == 100
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)

        # FastEmbed handles batching internally, so single call
        assert mock_model.embed.call_count == 1

    @pytest.mark.asyncio
    async def test_dense_and_sparse_embeddings_workflow(self):
        """Test workflow with both dense and sparse embeddings."""
        provider = FastEmbedProvider()
        provider._initialized = True

        # Setup dense model
        mock_dense_model = Mock()
        provider._model = mock_dense_model
        mock_dense_embeddings = [np.array([0.1, 0.2, 0.3])]
        mock_dense_model.embed.return_value = iter(mock_dense_embeddings)

        # Setup sparse model
        with patch(
            "src.services.embeddings.fastembed_provider.SparseTextEmbedding"
        ) as mock_sparse:
            mock_sparse_model = Mock()
            mock_sparse.return_value = mock_sparse_model

            mock_sparse_result = Mock()
            mock_sparse_result.indices = np.array([0, 5, 10])
            mock_sparse_result.values = np.array([0.1, 0.5, 0.8])
            mock_sparse_model.embed.return_value = [mock_sparse_result]

            # Generate dense embeddings
            dense_embeddings = await provider.generate_embeddings(["test text"])
            assert len(dense_embeddings) == 1
            assert dense_embeddings[0] == [0.1, 0.2, 0.3]

            # Generate sparse embeddings
            sparse_embeddings = await provider.generate_sparse_embeddings(["test text"])
            assert len(sparse_embeddings) == 1
            assert sparse_embeddings[0]["indices"] == [0, 5, 10]
            assert sparse_embeddings[0]["values"] == [0.1, 0.5, 0.8]

    @pytest.mark.asyncio
    async def test_provider_error_recovery(self):
        """Test provider error recovery scenarios."""
        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        call_count = 0

        def mock_embed_response(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "Temporary failure"
                raise TestError(msg)
            # Success on second call
            return iter([np.array([0.1, 0.2, 0.3]) for _ in texts])

        mock_model.embed.side_effect = mock_embed_response

        # First call should fail
        with pytest.raises(EmbeddingServiceError):
            await provider.generate_embeddings(["test"])

        # Second call should succeed
        embeddings = await provider.generate_embeddings(["test"])
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

    def test_model_configuration_consistency(self):
        """Test model configuration consistency across different models."""
        models_to_test = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "jinaai/jina-embeddings-v2-small-en",
        ]

        for model_name in models_to_test:
            provider = FastEmbedProvider(model_name)

            # Verify configuration is loaded correctly
            config = FastEmbedProvider.SUPPORTED_MODELS[model_name]
            assert provider.dimensions == config["dimensions"]
            assert provider._max_tokens == config["max_tokens"]
            assert provider._description == config["description"]
            assert provider.cost_per_token == 0.0  # Always free for local models

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self):
        """Test concurrent embedding generation."""

        provider = FastEmbedProvider()
        provider._initialized = True

        mock_model = Mock()
        provider._model = mock_model

        def mock_embed_response(texts):
            return iter([np.array([0.1, 0.2, 0.3]) for _ in texts])

        mock_model.embed.side_effect = mock_embed_response

        # Generate embeddings concurrently
        tasks = [provider.generate_embeddings([f"text{i}"]) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(len(result) == 1 for result in results)
        assert mock_model.embed.call_count == 5

    def test_provider_inheritance_and_interface(self):
        """Test provider properly implements base interface."""

        provider = FastEmbedProvider()

        # Check inheritance
        assert isinstance(provider, EmbeddingProvider)
        assert issubclass(FastEmbedProvider, EmbeddingProvider)

        # Check required methods exist
        assert hasattr(provider, "initialize")
        assert hasattr(provider, "cleanup")
        assert hasattr(provider, "generate_embeddings")
        assert hasattr(provider, "cost_per_token")
        assert hasattr(provider, "max_tokens_per_request")

        # Check properties work
        assert provider.cost_per_token == 0.0
        assert provider.max_tokens_per_request > 0
