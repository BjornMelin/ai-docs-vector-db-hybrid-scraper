"""Score-based result clustering using scikit-learn.

This module provides functionality to cluster search results based on their
relevance scores using MiniBatchKMeans algorithm from scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from numpy import array
from pydantic import BaseModel, Field
from sklearn.cluster import MiniBatchKMeans

from src.contracts.retrieval import SearchRecord


class ClusterGroup(BaseModel):
    """Represents a group of clustered search results.

    Attributes:
        cluster_id: Unique identifier for the cluster.
        results: List of search results belonging to this cluster.
    """

    cluster_id: int
    results: list[SearchRecord]


class ResultClusteringRequest(BaseModel):
    """Request model for result clustering operation.

    Attributes:
        results: List of search results to cluster.
        max_clusters: Maximum number of clusters to create (1-10).
    """

    results: list[SearchRecord]
    max_clusters: int = Field(3, ge=1, le=10)


class ResultClusteringResponse(BaseModel):
    """Response model containing clustered search results.

    Attributes:
        clusters: List of cluster groups containing search results.
    """

    clusters: list[ClusterGroup]


def _cluster_scores(scores: Sequence[float], max_clusters: int) -> list[int]:
    """Cluster numeric scores using MiniBatchKMeans algorithm.

    Args:
        scores: Sequence of numeric scores to cluster.
        max_clusters: Maximum number of clusters to create.

    Returns:
        List of cluster labels for each score.
    """
    # Accept any sequence of numeric scores; convert to floats
    flattened = [float(score) for score in scores]
    values = array([[value] for value in flattened], dtype=float)
    unique_scores = len(set(flattened))
    k = max(1, min(max_clusters, unique_scores))
    model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
    return model.fit_predict(values).tolist()


@dataclass(slots=True)
class ResultClusteringService:
    """Service for clustering search results based on relevance scores.

    This service uses the MiniBatchKMeans algorithm to group search results
    into clusters based on their similarity scores, helping to organize
    and present results in a more structured manner.
    """

    async def initialize(self) -> None:  # pragma: no cover
        """Initialize the clustering service.

        This method is a no-op but maintained for interface consistency.
        """
        return

    async def cluster_results(
        self, request: ResultClusteringRequest
    ) -> ResultClusteringResponse:
        """Cluster search results based on their relevance scores.

        Args:
            request: Request containing search results and clustering parameters.

        Returns:
            Response containing clustered search results organized by relevance.
        """
        if not request.results:
            return ResultClusteringResponse(clusters=[])
        labels = _cluster_scores(
            [result.score for result in request.results], request.max_clusters
        )
        groups: dict[int, list[SearchRecord]] = {}
        for label, result in zip(labels, request.results, strict=False):
            groups.setdefault(int(label), []).append(result)
        clusters = [
            ClusterGroup(cluster_id=cluster_id, results=items)
            for cluster_id, items in sorted(groups.items())
        ]
        return ResultClusteringResponse(clusters=clusters)


__all__ = [
    "ResultClusteringService",
    "ResultClusteringRequest",
    "ResultClusteringResponse",
]
