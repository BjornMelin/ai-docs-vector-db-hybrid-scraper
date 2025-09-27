"""Unit tests for the functional embedding facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.services.embeddings.manager import QualityTier, TextAnalysis
from src.services.functional.embeddings import (
    analyze_text_characteristics,
    batch_generate_embeddings,
    estimate_embedding_cost,
    generate_embeddings,
    get_provider_info,
    get_smart_recommendation,
    get_usage_report,
    rerank_results,
)


@pytest.fixture
def embedding_client_stub() -> AsyncMock:
    """Create a fully populated embedding client double."""
    client = AsyncMock()
    client.generate_embeddings.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
        "success": True,
    }
    client.rerank_results.return_value = [
        {"content": "high", "score": 0.9},
        {"content": "low", "score": 0.1},
    ]
    client.analyze_text_characteristics = MagicMock(
        return_value=TextAnalysis(
            total_length=100,
            avg_length=50,
            complexity_score=0.7,
            estimated_tokens=25,
            text_type="docs",
            requires_high_quality=False,
        )
    )
    client.estimate_cost = MagicMock(
        return_value={
            "fastembed": {
                "estimated_tokens": 10,
                "cost_per_token": 0.0,
                "_total_cost": 0.0,
            }
        }
    )
    client.get_provider_info = MagicMock(
        return_value={
            "fastembed": {"model": "BAAI/bge-small-en-v1.5", "max_dimension": 768}
        }
    )
    client.get_usage_report = MagicMock(
        return_value={
            "summary": {"total_requests": 2, "total_cost": 0.0},
            "by_provider": {"fastembed": {"requests": 2, "cost": 0.0}},
            "by_tier": {"fast": {"requests": 2, "cost": 0.0}},
            "budget": {"daily_limit": None, "daily_usage": 0.0},
        }
    )
    client.get_smart_provider_recommendation = MagicMock(
        return_value={
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
            "estimated_cost": 0.0,
            "reasoning": "Optimal for medium documents",
        }
    )
    return client


class TestGenerateEmbeddings:
    """Tests for ``generate_embeddings``."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return payload from the underlying client when execution succeeds."""
        result = await generate_embeddings(
            texts=["example"],
            quality_tier=QualityTier.BALANCED,
            embedding_client=embedding_client_stub,
        )

        assert result["embeddings"]
        embedding_client_stub.generate_embeddings.assert_awaited_once_with(
            texts=["example"],
            quality_tier=QualityTier.BALANCED,
            provider_name=None,
            max_cost=None,
            speed_priority=False,
            auto_select=True,
            generate_sparse=False,
        )

    @pytest.mark.asyncio
    async def test_generate_embeddings_missing_client(self) -> None:
        """Raise ``HTTPException`` when the dependency is not injected."""
        with pytest.raises(HTTPException, match="Embedding client not available"):
            await generate_embeddings(texts=["example"], embedding_client=None)

    @pytest.mark.asyncio
    async def test_generate_embeddings_client_error(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Wrap provider failures in ``HTTPException`` for consistent handling."""
        embedding_client_stub.generate_embeddings.side_effect = RuntimeError("boom")

        with pytest.raises(HTTPException, match="Embedding generation failed"):
            await generate_embeddings(
                texts=["example"], embedding_client=embedding_client_stub
            )


class TestRerankResults:
    """Tests for ``rerank_results``."""

    @pytest.mark.asyncio
    async def test_rerank_results_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return the reranked list when the client succeeds."""
        results = [{"content": "low", "score": 0.1}]

        response = await rerank_results(
            query="example",
            results=results,
            embedding_client=embedding_client_stub,
        )

        assert response[0]["score"] == pytest.approx(0.9)
        embedding_client_stub.rerank_results.assert_awaited_once_with(
            "example", results
        )

    @pytest.mark.asyncio
    async def test_rerank_results_no_client(self) -> None:
        """Return the input results when no client is available."""
        results = [{"content": "only"}]

        response = await rerank_results(
            query="example", results=results, embedding_client=None
        )

        assert response == results

    @pytest.mark.asyncio
    async def test_rerank_results_expected_error(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Gracefully fall back when the client raises a handled exception."""
        embedding_client_stub.rerank_results.side_effect = AttributeError(
            "missing impl"
        )
        results = [{"content": "fallback"}]

        response = await rerank_results(
            query="example",
            results=results,
            embedding_client=embedding_client_stub,
        )

        assert response == results


class TestAnalyzeTextCharacteristics:
    """Tests for ``analyze_text_characteristics``."""

    @pytest.mark.asyncio
    async def test_analyze_text_characteristics_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return the synchronous analysis from the client."""
        texts = ["chunk one", "chunk two"]

        analysis = await analyze_text_characteristics(texts, embedding_client_stub)

        assert analysis.text_type == "docs"
        embedding_client_stub.analyze_text_characteristics.assert_called_once_with(
            texts
        )

    @pytest.mark.asyncio
    async def test_analyze_text_characteristics_missing_client(self) -> None:
        """Raise when the dependency injection fails."""
        with pytest.raises(HTTPException, match="Embedding client not available"):
            await analyze_text_characteristics(["text"], None)

    @pytest.mark.asyncio
    async def test_analyze_text_characteristics_failure(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Wrap arbitrary exceptions raised by the client."""
        embedding_client_stub.analyze_text_characteristics.side_effect = RuntimeError(
            "boom"
        )

        with pytest.raises(HTTPException, match="Text analysis failed"):
            await analyze_text_characteristics(["text"], embedding_client_stub)


class TestEstimateEmbeddingCost:
    """Tests for ``estimate_embedding_cost``."""

    @pytest.mark.asyncio
    async def test_estimate_embedding_cost_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return the cost dictionary from the client."""
        costs = await estimate_embedding_cost(
            ["text"], "fastembed", embedding_client_stub
        )

        assert "fastembed" in costs
        embedding_client_stub.estimate_cost.assert_called_once_with(
            ["text"], "fastembed"
        )

    @pytest.mark.asyncio
    async def test_estimate_embedding_cost_missing_client(self) -> None:
        """Raise when the client dependency is absent."""
        with pytest.raises(HTTPException, match="Embedding client not available"):
            await estimate_embedding_cost(["text"], "fastembed", None)

    @pytest.mark.asyncio
    async def test_estimate_embedding_cost_failure(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Propagate errors as ``HTTPException`` for consistency."""
        embedding_client_stub.estimate_cost.side_effect = RuntimeError("boom")

        with pytest.raises(HTTPException, match="Cost estimation failed"):
            await estimate_embedding_cost(["text"], None, embedding_client_stub)


class TestProviderInfo:
    """Tests for ``get_provider_info``."""

    @pytest.mark.asyncio
    async def test_get_provider_info_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return provider metadata when the client responds."""
        info = await get_provider_info(embedding_client_stub)

        assert "fastembed" in info
        embedding_client_stub.get_provider_info.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_get_provider_info_no_client(self) -> None:
        """Return an empty dictionary when no client is available."""
        assert await get_provider_info(None) == {}

    @pytest.mark.asyncio
    async def test_get_provider_info_failure(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Convert client exceptions into ``HTTPException``."""
        embedding_client_stub.get_provider_info.side_effect = RuntimeError("boom")

        with pytest.raises(HTTPException, match="Provider info retrieval failed"):
            await get_provider_info(embedding_client_stub)


class TestSmartRecommendation:
    """Tests for ``get_smart_recommendation``."""

    @pytest.mark.asyncio
    async def test_get_smart_recommendation_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return the recommendation payload from the client."""
        recommendation = await get_smart_recommendation(
            ["doc"],
            quality_tier=QualityTier.BALANCED,
            max_cost=0.5,
            embedding_client=embedding_client_stub,
        )

        assert recommendation["provider"] == "fastembed"
        embedding_client_stub.get_smart_provider_recommendation.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_smart_recommendation_missing_client(self) -> None:
        """Raise when the embedding client is not injected."""
        with pytest.raises(HTTPException, match="Embedding client not available"):
            await get_smart_recommendation(["doc"], embedding_client=None)

    @pytest.mark.asyncio
    async def test_get_smart_recommendation_failure(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Translate arbitrary client errors into ``HTTPException``."""
        embedding_client_stub.get_smart_provider_recommendation.side_effect = (
            RuntimeError("boom")
        )

        with pytest.raises(HTTPException, match="Smart recommendation failed"):
            await get_smart_recommendation(
                ["doc"], embedding_client=embedding_client_stub
            )


class TestUsageReport:
    """Tests for ``get_usage_report``."""

    @pytest.mark.asyncio
    async def test_get_usage_report_with_client(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Return the report from the embedding client."""
        report = await get_usage_report(embedding_client_stub)

        assert report["summary"]["total_requests"] == 2
        embedding_client_stub.get_usage_report.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_get_usage_report_without_client(self) -> None:
        """Provide the default empty report when no client is available."""
        report = await get_usage_report(None)

        assert report["summary"]["total_requests"] == 0
        assert report["budget"]["daily_usage"] == 0.0

    @pytest.mark.asyncio
    async def test_get_usage_report_failure(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Wrap errors raised by the client in ``HTTPException``."""
        embedding_client_stub.get_usage_report.side_effect = RuntimeError("boom")

        with pytest.raises(HTTPException, match="Usage report retrieval failed"):
            await get_usage_report(embedding_client_stub)


class TestBatchGenerateEmbeddings:
    """Tests for ``batch_generate_embeddings``."""

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_success(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Process all batches concurrently and return individual results."""
        text_batches = [["doc-1"], ["doc-2"]]

        results = await batch_generate_embeddings(
            text_batches=text_batches,
            quality_tier=QualityTier.FAST,
            max_parallel=2,
            embedding_client=embedding_client_stub,
        )

        assert len(results) == 2
        assert all(result.get("success", True) for result in results)
        assert embedding_client_stub.generate_embeddings.await_count == 2

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_partial_failure(
        self, embedding_client_stub: AsyncMock
    ) -> None:
        """Mark failed batches without interrupting successes."""
        embedding_client_stub.generate_embeddings.side_effect = [
            {
                "embeddings": [[0.4, 0.5, 0.6]],
                "provider": "fastembed",
                "success": True,
            },
            HTTPException(status_code=500, detail="downstream failure"),
        ]

        text_batches = [["ok"], ["fail"]]

        results = await batch_generate_embeddings(
            text_batches=text_batches,
            embedding_client=embedding_client_stub,
        )

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "downstream failure" in results[1]["error"]

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings_without_client(self) -> None:
        """Return a failure result when no embedding client is provided."""
        results = await batch_generate_embeddings(
            text_batches=[["text"]], embedding_client=None
        )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Embedding client not available" in results[0]["error"]
