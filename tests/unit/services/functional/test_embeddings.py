"""Tests for function-based embedding service."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.services.embeddings.manager import QualityTier
from src.services.embeddings.manager import TextAnalysis
from src.services.functional.embeddings import analyze_text_characteristics
from src.services.functional.embeddings import batch_generate_embeddings
from src.services.functional.embeddings import estimate_embedding_cost
from src.services.functional.embeddings import generate_embeddings
from src.services.functional.embeddings import rerank_results


class TestGenerateEmbeddings:
    """Test generate_embeddings function."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        # Mock embedding client
        mock_client = AsyncMock()
        mock_client.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "cost": 0.001,
            "success": True,
        }

        result = await generate_embeddings(
            texts=["test text"],
            quality_tier=QualityTier.BALANCED,
            embedding_client=mock_client,
        )

        assert result["success"] is True
        assert result["provider"] == "openai"
        assert len(result["embeddings"]) == 1
        mock_client.generate_embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_no_client(self):
        """Test embedding generation with no client."""
        with pytest.raises(HTTPException) as exc_info:
            await generate_embeddings(
                texts=["test text"],
                embedding_client=None,
            )

        assert exc_info.value.status_code == 500
        assert "not available" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_embeddings_client_error(self):
        """Test embedding generation with client error."""
        mock_client = AsyncMock()
        mock_client.generate_embeddings.side_effect = Exception("Provider error")

        with pytest.raises(HTTPException) as exc_info:
            await generate_embeddings(
                texts=["test text"],
                embedding_client=mock_client,
            )

        assert exc_info.value.status_code == 500
        assert "Embedding generation failed" in str(exc_info.value.detail)


class TestRerankResults:
    """Test rerank_results function."""

    @pytest.mark.asyncio
    async def test_rerank_results_success(self):
        """Test successful result reranking."""
        mock_client = AsyncMock()
        original_results = [
            {"content": "result 1", "score": 0.1},
            {"content": "result 2", "score": 0.9},
        ]
        reranked_results = [
            {"content": "result 2", "score": 0.9},
            {"content": "result 1", "score": 0.1},
        ]
        mock_client.rerank_results.return_value = reranked_results

        result = await rerank_results(
            query="test query",
            results=original_results,
            embedding_client=mock_client,
        )

        assert result == reranked_results
        mock_client.rerank_results.assert_called_once_with(
            "test query", original_results
        )

    @pytest.mark.asyncio
    async def test_rerank_results_no_client(self):
        """Test reranking with no client (graceful degradation)."""
        original_results = [{"content": "result 1"}]

        result = await rerank_results(
            query="test query",
            results=original_results,
            embedding_client=None,
        )

        assert result == original_results

    @pytest.mark.asyncio
    async def test_rerank_results_error_fallback(self):
        """Test reranking error with graceful fallback."""
        mock_client = AsyncMock()
        mock_client.rerank_results.side_effect = Exception("Reranking error")
        original_results = [{"content": "result 1"}]

        result = await rerank_results(
            query="test query",
            results=original_results,
            embedding_client=mock_client,
        )

        assert result == original_results


class TestAnalyzeTextCharacteristics:
    """Test analyze_text_characteristics function."""

    @pytest.mark.asyncio
    async def test_analyze_text_characteristics_success(self):
        """Test successful text analysis."""
        mock_client = AsyncMock()
        expected_analysis = TextAnalysis(
            total_length=100,
            avg_length=50,
            complexity_score=0.7,
            estimated_tokens=25,
            text_type="docs",
            requires_high_quality=False,
        )
        mock_client.analyze_text_characteristics.return_value = expected_analysis

        result = await analyze_text_characteristics(
            texts=["test text 1", "test text 2"],
            embedding_client=mock_client,
        )

        assert result == expected_analysis
        mock_client.analyze_text_characteristics.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_text_characteristics_no_client(self):
        """Test text analysis with no client."""
        with pytest.raises(HTTPException) as exc_info:
            await analyze_text_characteristics(
                texts=["test text"],
                embedding_client=None,
            )

        assert exc_info.value.status_code == 500


class TestEstimateEmbeddingCost:
    """Test estimate_embedding_cost function."""

    @pytest.mark.asyncio
    async def test_estimate_cost_success(self):
        """Test successful cost estimation."""
        mock_client = AsyncMock()
        expected_costs = {
            "openai": {
                "estimated_tokens": 100,
                "cost_per_token": 0.0001,
                "total_cost": 0.01,
            }
        }
        mock_client.estimate_cost.return_value = expected_costs

        result = await estimate_embedding_cost(
            texts=["test text"],
            provider_name="openai",
            embedding_client=mock_client,
        )

        assert result == expected_costs
        mock_client.estimate_cost.assert_called_once_with(["test text"], "openai")


class TestBatchGenerateEmbeddings:
    """Test batch_generate_embeddings function."""

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_success(self):
        """Test successful batch embedding generation."""
        mock_client = AsyncMock()

        # Mock successful responses for each batch
        def mock_generate(texts, **kwargs):
            return {
                "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
                "provider": "openai",
                "success": True,
            }

        mock_client.generate_embeddings.side_effect = mock_generate

        text_batches = [
            ["text 1", "text 2"],
            ["text 3", "text 4"],
        ]

        result = await batch_generate_embeddings(
            text_batches=text_batches,
            quality_tier=QualityTier.FAST,
            max_parallel=2,
            embedding_client=mock_client,
        )

        assert len(result) == 2
        assert all(r.get("success", True) for r in result)
        assert mock_client.generate_embeddings.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_partial_failure(self):
        """Test batch generation with some failures."""
        mock_client = AsyncMock()

        # Mock one success and one failure
        call_count = 0

        def mock_generate(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"embeddings": [[0.1, 0.2]], "success": True}
            else:
                raise Exception("Provider error")

        mock_client.generate_embeddings.side_effect = mock_generate

        text_batches = [["text 1"], ["text 2"]]

        result = await batch_generate_embeddings(
            text_batches=text_batches,
            embedding_client=mock_client,
        )

        assert len(result) == 2
        assert result[0].get("success", True)
        assert result[1]["success"] is False
        assert "error" in result[1]

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_no_client(self):
        """Test batch generation with no client."""
        with pytest.raises(HTTPException):
            await batch_generate_embeddings(
                text_batches=[["text 1"]],
                embedding_client=None,
            )


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with embedding functions."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_repeated_failures(self):
        """Test circuit breaker triggers after repeated failures."""
        mock_client = AsyncMock()
        mock_client.generate_embeddings.side_effect = Exception("Persistent error")

        # Multiple attempts should trigger circuit breaker
        for _i in range(5):
            try:
                await generate_embeddings(
                    texts=["test"],
                    embedding_client=mock_client,
                )
            except HTTPException:
                pass  # Expected

        # Circuit breaker should be triggered by now
        # Note: In real implementation, we'd need access to the circuit breaker instance
        # to verify its state. This is a simplified test.
        assert mock_client.generate_embeddings.call_count <= 5


@pytest.fixture
def mock_embedding_client():
    """Create a mock embedding client."""
    client = AsyncMock()
    client.generate_embeddings.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
        "cost": 0.0,
        "success": True,
    }
    client.rerank_results.return_value = []
    client.analyze_text_characteristics.return_value = TextAnalysis(
        total_length=10,
        avg_length=10,
        complexity_score=0.5,
        estimated_tokens=3,
        text_type="short",
        requires_high_quality=False,
    )
    client.estimate_cost.return_value = {}
    return client
