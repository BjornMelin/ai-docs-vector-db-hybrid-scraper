"""Tests for embedding providers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from src.services.config import APIConfig
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
            api_key="test-key",
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
            mock_client.assert_called_once_with(api_key="test-key")

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
            mock_embedding.embedding = [0.1, 0.2, 0.3]
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            mock_instance.embeddings.create.return_value = mock_response

            await openai_provider.initialize()

            embeddings = await openai_provider.generate_embeddings(
                ["test text"],
                batch_size=1,
            )

            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, openai_provider):
        """Test batch embedding generation."""
        with patch(
            "src.services.embeddings.openai_provider.AsyncOpenAI"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock embedding responses for batch processing
            # First batch: 2 embeddings
            mock_response1 = MagicMock()
            mock_response1.data = [
                MagicMock(embedding=[0.0, 0.0, 0.0]),
                MagicMock(embedding=[0.1, 0.2, 0.3]),
            ]

            # Second batch: 1 embedding
            mock_response2 = MagicMock()
            mock_response2.data = [
                MagicMock(embedding=[0.2, 0.4, 0.6]),
            ]

            # Configure mock to return different responses for each call
            mock_instance.embeddings.create.side_effect = [
                mock_response1,
                mock_response2,
            ]

            await openai_provider.initialize()

            texts = ["text1", "text2", "text3"]
            embeddings = await openai_provider.generate_embeddings(
                texts,
                batch_size=2,
            )

            assert len(embeddings) == 3
            assert embeddings[0] == [0.0, 0.0, 0.0]
            assert embeddings[1] == [0.1, 0.2, 0.3]
            assert embeddings[2] == [0.2, 0.4, 0.6]
            assert mock_instance.embeddings.create.call_count == 2  # 2 batches

    def test_cost_per_token(self, openai_provider):
        """Test cost calculation."""
        cost = openai_provider.cost_per_token
        assert cost == 0.02 / 1_000_000  # $0.02 per 1M tokens

    def test_unsupported_model(self):
        """Test unsupported model error."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            OpenAIEmbeddingProvider(
                api_key="test",
                model_name="invalid-model",
            )


class TestFastEmbedProvider:
    """Test FastEmbed provider."""

    @pytest.fixture
    def fastembed_provider(self):
        """Create FastEmbed provider instance."""
        return FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_initialize(self, fastembed_provider):
        """Test provider initialization."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            await fastembed_provider.initialize()

            assert fastembed_provider._initialized
            mock_model.assert_called_once_with("BAAI/bge-small-en-v1.5")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, fastembed_provider):
        """Test embedding generation."""
        with patch(
            "src.services.embeddings.fastembed_provider.TextEmbedding"
        ) as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock embedding result
            mock_instance.embed.return_value = [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, 0.6]),
            ]

            await fastembed_provider.initialize()

            embeddings = await fastembed_provider.generate_embeddings(
                ["text1", "text2"]
            )

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_cost_per_token(self, fastembed_provider):
        """Test cost calculation (should be 0 for local)."""
        assert fastembed_provider.cost_per_token == 0.0

    def test_unsupported_model(self):
        """Test unsupported model error."""
        with pytest.raises(EmbeddingServiceError, match="Unsupported model"):
            FastEmbedProvider(model_name="invalid-model")

    def test_list_available_models(self):
        """Test listing available models."""
        models = FastEmbedProvider.list_available_models()
        assert "BAAI/bge-small-en-v1.5" in models
        assert len(models) > 5


class TestEmbeddingManager:
    """Test embedding manager."""

    @pytest.fixture
    def api_config(self):
        """Create test API config."""
        return APIConfig(
            openai_api_key="test-key",
            enable_local_embeddings=True,
        )

    @pytest.fixture
    def embedding_manager(self, api_config):
        """Create embedding manager instance."""
        return EmbeddingManager(api_config)

    @pytest.mark.asyncio
    async def test_initialize(self, embedding_manager):
        """Test manager initialization."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
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
    async def test_generate_embeddings_with_quality_tier(self, embedding_manager):
        """Test embedding generation with quality tier."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock embeddings
            mock_openai_instance.generate_embeddings.return_value = [[0.1, 0.2]]
            mock_fastembed_instance.generate_embeddings.return_value = [[0.3, 0.4]]

            await embedding_manager.initialize()

            # Test FAST tier (should use fastembed)
            embeddings = await embedding_manager.generate_embeddings(
                ["test"],
                quality_tier=QualityTier.FAST,
            )
            mock_fastembed_instance.generate_embeddings.assert_called_once()

            # Test BEST tier (should use openai)
            await embedding_manager.generate_embeddings(
                ["test"],
                quality_tier=QualityTier.BEST,
            )
            mock_openai_instance.generate_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_estimate_cost(self, embedding_manager):
        """Test cost estimation."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            # Mock cost properties
            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await embedding_manager.initialize()

            costs = embedding_manager.estimate_cost(["test" * 100])

            assert "openai" in costs
            assert "fastembed" in costs
            assert costs["openai"]["total_cost"] > 0
            assert costs["fastembed"]["total_cost"] == 0

    @pytest.mark.asyncio
    async def test_get_optimal_provider(self, embedding_manager):
        """Test optimal provider selection."""
        with (
            patch(
                "src.services.embeddings.manager.OpenAIEmbeddingProvider"
            ) as mock_openai,
            patch(
                "src.services.embeddings.manager.FastEmbedProvider"
            ) as mock_fastembed,
        ):
            # Setup providers
            mock_openai_instance = AsyncMock()
            mock_fastembed_instance = AsyncMock()
            mock_openai.return_value = mock_openai_instance
            mock_fastembed.return_value = mock_fastembed_instance

            mock_openai_instance.cost_per_token = 0.00002
            mock_fastembed_instance.cost_per_token = 0.0

            await embedding_manager.initialize()

            # Small text should prefer local
            provider = await embedding_manager.get_optimal_provider(
                text_length=1000,
                quality_required=False,
            )
            assert provider == "fastembed"

            # Quality required should prefer OpenAI
            provider = await embedding_manager.get_optimal_provider(
                text_length=1000,
                quality_required=True,
            )
            assert provider == "openai"

    @pytest.mark.asyncio
    async def test_no_providers_available(self):
        """Test error when no providers available."""
        config = APIConfig(
            openai_api_key=None,
            enable_local_embeddings=False,
        )
        manager = EmbeddingManager(config)

        with pytest.raises(
            EmbeddingServiceError, match="No embedding providers available"
        ):
            await manager.initialize()
