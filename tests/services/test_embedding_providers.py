"""Tests for embedding providers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.config.models import UnifiedConfig
from src.config.enums import EmbeddingProvider as EmbeddingProviderEnum
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        return OpenAIEmbeddingProvider(
            api_key="sk-test-key",
            model_name="text-embedding-3-small",
            dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, openai_provider):
        """Test provider initialization."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            await openai_provider.initialize()

            assert openai_provider._initialized
            mock_client.assert_called_once_with(api_key="sk-test-key")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, openai_provider):
        """Test embedding generation."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock embedding response
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_instance.embeddings.create.return_value = mock_response

            await openai_provider.initialize()

            texts = ["test text"]
            embeddings = await openai_provider.generate_embeddings(texts)

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1536
            assert isinstance(embeddings[0], list)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, openai_provider):
        """Test batch embedding generation."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock embedding responses
            mock_embeddings = []
            for i in range(5):
                mock_embedding = MagicMock()
                mock_embedding.embedding = [0.1 + i * 0.1] * 1536
                mock_embeddings.append(mock_embedding)

            mock_response = MagicMock()
            mock_response.data = mock_embeddings
            mock_instance.embeddings.create.return_value = mock_response

            await openai_provider.initialize()

            texts = ["text1", "text2", "text3", "text4", "text5"]
            embeddings = await openai_provider.generate_embeddings(texts)

            assert len(embeddings) == 5
            assert all(len(emb) == 1536 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, openai_provider):
        """Test rate limit error handling."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock rate limit error
            from openai import RateLimitError

            mock_instance.embeddings.create.side_effect = RateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )

            await openai_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await openai_provider.generate_embeddings(["test"])

            assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cleanup(self, openai_provider):
        """Test provider cleanup."""
        with patch("src.services.embeddings.openai_provider.AsyncOpenAI"):
            await openai_provider.initialize()
            assert openai_provider._initialized

            await openai_provider.cleanup()

            assert not openai_provider._initialized
            assert openai_provider._client is None


class TestFastEmbedProvider:
    """Test FastEmbed provider."""

    @pytest.fixture
    def fastembed_provider(self):
        """Create FastEmbed provider instance."""
        return FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_initialize(self, fastembed_provider):
        """Test provider initialization."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding") as mock_model:
            await fastembed_provider.initialize()

            assert fastembed_provider._initialized
            mock_model.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fastembed_provider):
        """Test embedding generation."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock embedding output
            mock_embeddings = np.array([[0.1] * 384, [0.2] * 384])
            mock_instance.embed.return_value = mock_embeddings

            await fastembed_provider.initialize()

            texts = ["text1", "text2"]
            embeddings = await fastembed_provider.generate_embeddings(texts)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            assert isinstance(embeddings[0], list)

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty(self, fastembed_provider):
        """Test handling of empty text list."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding"):
            await fastembed_provider.initialize()

            embeddings = await fastembed_provider.generate_embeddings([])

            assert embeddings == []

    @pytest.mark.asyncio
    async def test_error_handling(self, fastembed_provider):
        """Test error handling during embedding generation."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock error during embedding
            mock_instance.embed.side_effect = Exception("Model error")

            await fastembed_provider.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await fastembed_provider.generate_embeddings(["test"])

            assert "Failed to generate embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(self, fastembed_provider):
        """Test provider cleanup."""
        with patch("src.services.embeddings.fastembed_provider.TextEmbedding"):
            await fastembed_provider.initialize()
            assert fastembed_provider._initialized

            await fastembed_provider.cleanup()

            assert not fastembed_provider._initialized
            assert fastembed_provider._model is None


class TestEmbeddingManager:
    """Test embedding manager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig(
            openai__api_key="sk-test-key",
            embedding_provider=EmbeddingProviderEnum.OPENAI,
        )

    @pytest.fixture
    def embedding_manager(self, config):
        """Create embedding manager instance."""
        return EmbeddingManager(config)

    @pytest.mark.asyncio
    async def test_initialize(self, embedding_manager):
        """Test manager initialization."""
        with (
            patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai,
            patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()

            assert embedding_manager._initialized
            assert len(embedding_manager.providers) == 2
            assert "openai" in embedding_manager.providers
            assert "fastembed" in embedding_manager.providers

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_preferred_provider(self, embedding_manager):
        """Test embedding generation with preferred provider."""
        with (
            patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai,
            patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock OpenAI embeddings
            mock_openai_instance.generate_embeddings.return_value = [
                [0.1] * 1536,
                [0.2] * 1536,
            ]

            await embedding_manager.initialize()

            texts = ["text1", "text2"]
            embeddings = await embedding_manager.generate_embeddings(texts)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536
            mock_openai_instance.generate_embeddings.assert_called_once_with(texts)
            mock_fastembed_instance.generate_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_fallback(self, embedding_manager):
        """Test embedding generation with fallback to secondary provider."""
        with (
            patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai,
            patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock OpenAI failure
            mock_openai_instance.generate_embeddings.side_effect = EmbeddingServiceError(
                "OpenAI failed"
            )

            # Mock FastEmbed success
            mock_fastembed_instance.generate_embeddings.return_value = [
                [0.1] * 384,
                [0.2] * 384,
            ]

            await embedding_manager.initialize()

            texts = ["text1", "text2"]
            embeddings = await embedding_manager.generate_embeddings(texts)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            mock_openai_instance.generate_embeddings.assert_called_once()
            mock_fastembed_instance.generate_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_smart_provider_selection(self, embedding_manager):
        """Test smart provider selection based on quality tier."""
        with (
            patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai,
            patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock embeddings
            mock_fastembed_instance.generate_embeddings.return_value = [[0.1] * 384]

            await embedding_manager.initialize()

            # Force low quality tier to use FastEmbed
            texts = ["short text"]
            embeddings = await embedding_manager.generate_embeddings(
                texts, quality_tier=QualityTier.LOW
            )

            assert len(embeddings) == 1
            # Should use FastEmbed for low quality tier
            mock_fastembed_instance.generate_embeddings.assert_called()

    @pytest.mark.asyncio
    async def test_batch_processing(self, embedding_manager):
        """Test batch processing of large text lists."""
        with patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            # Mock batch responses
            batch_size = 100
            total_texts = 250
            mock_openai_instance.generate_embeddings.side_effect = [
                [[0.1] * 1536] * batch_size,  # First batch (100)
                [[0.2] * 1536] * batch_size,  # Second batch (100)
                [[0.3] * 1536] * 50,  # Third batch (50)
            ]

            await embedding_manager.initialize()

            texts = ["text"] * total_texts
            embeddings = await embedding_manager.generate_embeddings(texts)

            assert len(embeddings) == total_texts
            assert mock_openai_instance.generate_embeddings.call_count == 3

    @pytest.mark.asyncio
    async def test_cost_tracking(self, embedding_manager):
        """Test cost tracking for embeddings."""
        with patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536] * 10

            await embedding_manager.initialize()

            # Generate embeddings
            texts = ["test"] * 10
            await embedding_manager.generate_embeddings(texts)

            # Check cost tracking
            costs = embedding_manager.get_cost_summary()
            assert "openai" in costs
            assert costs["openai"]["requests"] > 0
            assert costs["openai"]["tokens"] > 0

    @pytest.mark.asyncio
    async def test_reranking(self, embedding_manager):
        """Test document reranking functionality."""
        with patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai:
            mock_openai_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance

            await embedding_manager.initialize()

            # Mock reranker if available
            if embedding_manager._reranker is not None:
                with patch.object(embedding_manager._reranker, "compute_score") as mock_score:
                    mock_score.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]

                    query = "test query"
                    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
                    
                    reranked = await embedding_manager.rerank_documents(
                        query, documents, top_k=3
                    )

                    assert len(reranked) == 3
                    # Should be ordered by score
                    assert reranked[0]["document"] == "doc1"
                    assert reranked[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_cleanup(self, embedding_manager):
        """Test manager cleanup."""
        with (
            patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai,
            patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()
            assert embedding_manager._initialized

            await embedding_manager.cleanup()

            assert not embedding_manager._initialized
            assert len(embedding_manager.providers) == 0
            mock_openai_instance.cleanup.assert_called_once()
            mock_fastembed_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, embedding_manager):
        """Test behavior when all providers fail."""
        with (
            patch("src.services.embeddings.manager.OpenAIEmbeddingProvider") as mock_openai,
            patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed,
        ):
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Both providers fail
            mock_openai_instance.generate_embeddings.side_effect = EmbeddingServiceError(
                "OpenAI failed"
            )
            mock_fastembed_instance.generate_embeddings.side_effect = EmbeddingServiceError(
                "FastEmbed failed"
            )

            await embedding_manager.initialize()

            with pytest.raises(EmbeddingServiceError) as exc_info:
                await embedding_manager.generate_embeddings(["test"])

            assert "All embedding providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialization_without_openai_key(self):
        """Test initialization without OpenAI API key."""
        config = UnifiedConfig()  # No API keys
        embedding_manager = EmbeddingManager(config)

        with patch("src.services.embeddings.manager.FastEmbedProvider") as mock_fastembed:
            mock_fastembed_instance = AsyncMock()
            mock_fastembed.return_value = mock_fastembed_instance

            await embedding_manager.initialize()

            assert embedding_manager._initialized
            assert len(embedding_manager.providers) == 1
            assert "fastembed" in embedding_manager.providers
            assert "openai" not in embedding_manager.providers