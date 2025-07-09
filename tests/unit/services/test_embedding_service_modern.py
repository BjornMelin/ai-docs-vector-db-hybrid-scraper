"""Modern unit tests for embedding service (7/2025 best practices).

Tests follow AAA pattern, test behavior not implementation,
and use respx for HTTP mocking.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
from httpx import Response
from hypothesis import given, settings

from src.models.api_contracts import Document
from src.services.embeddings.manager import EmbeddingManager
from tests.fixtures.modern_test_fixtures import document_strategy


class TestEmbeddingServiceModern:
    """Test embedding service with modern patterns."""

    @pytest.mark.asyncio
    async def test_embed_single_document(self, mock_openai_client: AsyncMock) -> None:
        """Test embedding a single document returns correct vector."""
        # Arrange
        document = Document(
            content="Test content for embedding",
            metadata={"test": True},
            source="test.txt",
        )
        expected_embedding = [0.1] * 1536
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=expected_embedding)]
        )

        # Act
        service = EmbeddingManager(openai_client=mock_openai_client)
        result = await service.embed_document(document)

        # Assert
        assert result == expected_embedding
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_documents(self, mock_openai_client: AsyncMock) -> None:
        """Test batch embedding returns vectors for all documents."""
        # Arrange
        documents = [
            Document(content=f"Doc {i}", metadata={}, source=f"doc{i}.txt")
            for i in range(5)
        ]
        expected_embeddings = [[0.1 + i * 0.01] * 1536 for i in range(5)]
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=emb) for emb in expected_embeddings]
        )

        # Act
        service = EmbeddingManager(openai_client=mock_openai_client)
        results = await service.embed_batch(documents)

        # Assert
        assert len(results) == len(documents)
        assert results == expected_embeddings

    @pytest.mark.asyncio
    @given(document=document_strategy())
    @settings(max_examples=10)
    async def test_embed_various_documents(
        self, document: Document, mock_openai_client: AsyncMock
    ) -> None:
        """Property test: embedding any valid document returns vector."""
        # Arrange
        expected_embedding = [0.2] * 1536
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=expected_embedding)]
        )

        # Act
        service = EmbeddingManager(openai_client=mock_openai_client)
        result = await service.embed_document(document)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_retry_on_failure(self) -> None:
        """Test HTTP retry logic with respx mocking."""
        # Arrange
        respx.post("https://api.openai.com/v1/embeddings").mock(
            side_effect=[
                Response(502),  # First call fails
                Response(
                    200,
                    json={  # Second call succeeds
                        "data": [{"embedding": [0.3] * 1536}]
                    },
                ),
            ]
        )

        # Act
        service = EmbeddingManager()
        result = await service.embed_with_retry("Test content")

        # Assert
        assert result == [0.3] * 1536
        assert respx.calls.call_count == 2

    @pytest.mark.asyncio
    async def test_caching_behavior(
        self, mock_openai_client: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test caching prevents redundant API calls."""
        # Arrange
        document = Document(content="Cached content", metadata={}, source="cache.txt")
        cached_embedding = [0.4] * 1536
        mock_redis_client.get.return_value = cached_embedding

        # Act
        service = EmbeddingManager(
            openai_client=mock_openai_client, cache_client=mock_redis_client
        )
        result = await service.embed_document(document)

        # Assert
        assert result == cached_embedding
        mock_openai_client.embeddings.create.assert_not_called()
        mock_redis_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_openai_client: AsyncMock) -> None:
        """Test proper error handling and logging."""
        # Arrange
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")

        # Act & Assert
        service = EmbeddingManager(openai_client=mock_openai_client)
        with pytest.raises(Exception, match="API Error"):
            await service.embed_document(
                Document(content="Error test", metadata={}, source="error.txt")
            )
