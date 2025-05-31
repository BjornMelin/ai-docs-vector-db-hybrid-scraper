"""Additional tests for FastEmbed provider coverage."""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.config.models import UnifiedConfig
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import EmbeddingServiceError


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=UnifiedConfig)
    config.fastembed = MagicMock()
    config.fastembed.model_name = "BAAI/bge-small-en-v1.5"
    config.fastembed.max_length = 512
    config.fastembed.batch_size = 32
    config.fastembed.cache_dir = "/tmp/fastembed"
    return config


@pytest.fixture
def provider(config):
    """Create FastEmbed provider."""
    return FastEmbedProvider(config)


class TestFastEmbedProviderCoverage:
    """Additional tests for FastEmbed provider."""

    @pytest.mark.asyncio
    async def test_cleanup_no_client(self, provider):
        """Test cleanup when no client exists."""
        provider._client = None
        await provider.cleanup()  # Should not raise
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, provider):
        """Test embedding empty list of texts."""
        with patch("fastembed.TextEmbedding") as mock_embedding:
            provider._client = mock_embedding

            result = await provider.embed_texts([])

            assert result == []
            mock_embedding.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_texts_single_batch(self, provider):
        """Test embedding texts that fit in a single batch."""
        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_client = MagicMock()
            mock_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
            mock_client.embed.return_value = mock_embeddings
            provider._client = mock_client
            provider._initialized = True

            texts = ["text1", "text2"]
            result = await provider.embed_texts(texts)

            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            mock_client.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_multiple_batches(self, provider):
        """Test embedding texts that require multiple batches."""
        provider.batch_size = 2  # Small batch size to force multiple batches

        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_client = MagicMock()
            # Return different embeddings for each batch
            mock_client.embed.side_effect = [
                [np.array([0.1, 0.2]), np.array([0.3, 0.4])],  # First batch
                [np.array([0.5, 0.6]), np.array([0.7, 0.8])],  # Second batch
                [np.array([0.9, 1.0])],  # Third batch
            ]
            provider._client = mock_client
            provider._initialized = True

            texts = ["text1", "text2", "text3", "text4", "text5"]
            result = await provider.embed_texts(texts)

            assert len(result) == 5
            assert mock_client.embed.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_texts_error_handling(self, provider):
        """Test error handling during embedding."""
        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_client = MagicMock()
            mock_client.embed.side_effect = Exception("Embedding failed")
            provider._client = mock_client
            provider._initialized = True

            with pytest.raises(
                EmbeddingServiceError, match="FastEmbed embedding failed"
            ):
                await provider.embed_texts(["test"])

    @pytest.mark.asyncio
    async def test_validate_texts_empty_text(self, provider):
        """Test validation rejects empty texts."""
        provider._validate_texts(["valid", "also valid"])  # Should pass

        with pytest.raises(ValueError, match="Empty text"):
            provider._validate_texts(["valid", "", "also valid"])

    def test_get_model_info(self, provider):
        """Test getting model information."""
        info = provider.get_model_info()

        assert info["provider"] == "fastembed"
        assert info["model"] == "BAAI/bge-small-en-v1.5"
        assert info["embedding_dim"] == 384
        assert info["max_tokens"] == 512
        assert "context_window" in info
        assert "supports_batch" in info

    def test_get_model_info_custom_model(self, config):
        """Test model info for different models."""
        # Test with base-en model
        config.fastembed.model_name = "BAAI/bge-base-en"
        provider = FastEmbedProvider(config)
        info = provider.get_model_info()
        assert info["embedding_dim"] == 768

        # Test with large-en model
        config.fastembed.model_name = "BAAI/bge-large-en"
        provider = FastEmbedProvider(config)
        info = provider.get_model_info()
        assert info["embedding_dim"] == 1024

    @pytest.mark.asyncio
    async def test_estimate_tokens(self, provider):
        """Test token estimation."""
        # Short text
        tokens = await provider.estimate_tokens("Hello world")
        assert 1 <= tokens <= 5

        # Longer text
        long_text = "This is a much longer text " * 20
        tokens = await provider.estimate_tokens(long_text)
        assert tokens > 20

    @pytest.mark.asyncio
    async def test_embed_query_vs_documents(self, provider):
        """Test that embed_query and embed_documents produce same output."""
        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_client = MagicMock()
            mock_client.embed.return_value = [np.array([0.1, 0.2, 0.3])]
            provider._client = mock_client
            provider._initialized = True

            query = "test query"

            # Test embed_query
            query_result = await provider.embed_query(query)

            # Test embed_documents with same text
            doc_result = await provider.embed_documents([query])

            assert query_result == doc_result[0]

    @pytest.mark.asyncio
    async def test_supports_async(self, provider):
        """Test async support check."""
        assert await provider.supports_async() is True

    def test_calculate_dimensions_unknown_model(self):
        """Test dimension calculation for unknown model."""
        provider = FastEmbedProvider(MagicMock())
        provider.model_name = "unknown/model"

        # Should return None for unknown models
        assert provider._calculate_dimensions() is None

    @pytest.mark.asyncio
    async def test_concurrent_embed_calls(self, provider):
        """Test concurrent embedding calls."""
        import asyncio

        with patch("fastembed.TextEmbedding") as mock_embedding:
            mock_client = MagicMock()
            mock_client.embed.return_value = [np.array([0.1, 0.2, 0.3])]
            provider._client = mock_client
            provider._initialized = True

            # Make concurrent calls
            tasks = [provider.embed_texts([f"text{i}"]) for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(len(r) == 1 for r in results)
