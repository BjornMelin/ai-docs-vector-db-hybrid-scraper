"""Tests for the result clustering service implementation."""

import pytest

from src.contracts.retrieval import SearchRecord
from src.services.query_processing.clustering import (
    ClusterGroup,
    ResultClusteringRequest,
    ResultClusteringResponse,
    ResultClusteringService,
)


class TestSearchRecord:
    """Test SearchRecord model."""

    def test_default_values(self):
        """Test default search record values."""
        result = SearchRecord(
            id="test_1", title="Test Title", content="Test content", score=0.85
        )

        assert result.id == "test_1"
        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.metadata is None

    def test_score_validation(self):
        """Test score validation."""
        # Valid scores
        result1 = SearchRecord(id="1", title="T", content="C", score=0.0)
        assert result1.score == 0.0

        result2 = SearchRecord(id="2", title="T", content="C", score=1.0)
        assert result2.score == 1.0

        # Invalid scores
        with pytest.raises(ValueError):
            SearchRecord(id="3", title="T", content="C", score=-0.1)


class TestClusterGroup:
    """Test ClusterGroup model."""

    def test_default_values(self):
        """Test default cluster group values."""
        results = [
            SearchRecord(id="1", title="T1", content="C1", score=0.8),
            SearchRecord(id="2", title="T2", content="C2", score=0.9),
        ]

        cluster = ClusterGroup(
            cluster_id=1,
            results=results,
        )

        assert cluster.cluster_id == 1
        assert cluster.results == results


class TestResultClusteringRequest:
    """Test ResultClusteringRequest model."""

    def test_default_values(self):
        """Test default request values."""
        results = [SearchRecord(id="1", title="T", content="C", score=0.8)]
        request = ResultClusteringRequest(results=results, max_clusters=3)

        assert request.results == results
        assert request.max_clusters == 3


class TestResultClusteringResponse:
    """Test ResultClusteringResponse model."""

    def test_default_values(self):
        """Test default response values."""
        clusters = [
            ClusterGroup(
                cluster_id=1,
                results=[SearchRecord(id="1", title="T", content="C", score=0.8)],
            )
        ]
        response = ResultClusteringResponse(clusters=clusters)

        assert response.clusters == clusters


class TestResultClusteringService:
    """Test ResultClusteringService."""

    @pytest.mark.asyncio
    async def test_cluster_results(self):
        """Test clustering results."""
        service = ResultClusteringService()
        await service.initialize()

        results = [
            SearchRecord(id="1", title="T1", content="C1", score=0.8),
            SearchRecord(id="2", title="T2", content="C2", score=0.9),
            SearchRecord(id="3", title="T3", content="C3", score=0.7),
        ]
        request = ResultClusteringRequest(results=results, max_clusters=2)

        response = await service.cluster_results(request)

        assert isinstance(response, ResultClusteringResponse)
        assert len(response.clusters) > 0
        for cluster in response.clusters:
            assert isinstance(cluster, ClusterGroup)
            assert len(cluster.results) > 0

    @pytest.mark.asyncio
    async def test_cluster_results_empty(self):
        """Test clustering with empty results."""
        service = ResultClusteringService()
        await service.initialize()

        request = ResultClusteringRequest(results=[], max_clusters=2)
        response = await service.cluster_results(request)

        assert isinstance(response, ResultClusteringResponse)
        assert response.clusters == []
