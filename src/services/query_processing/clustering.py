# pylint: disable=too-many-lines

"""Result clustering service for grouping search results.

The module exposes clustering algorithms with caching, metrics, and optional
dependencies. Missing third-party libraries are detected at runtime so the
service can continue operating with reduced functionality.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from enum import Enum
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.contracts.retrieval import SearchRecord

from .utils import (
    STOP_WORDS,
    CacheManager,
    PerformanceTracker,
    build_cache_key,
    distance_for_metric,
    merge_performance_metadata,
    performance_snapshot,
    sklearn_metric_for,
)


# Optional clustering dependencies
try:  # pragma: no cover - import availability tested via unit suite
    import sklearn.cluster as sk_cluster  # type: ignore[import-not-found]
    import sklearn.metrics as sk_metrics  # type: ignore[import-not-found]
    import sklearn.mixture as sk_mixture  # type: ignore[import-not-found]
    import sklearn.neighbors as sk_neighbors  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - covered by availability checks
    sk_cluster = None  # type: ignore[assignment]
    sk_metrics = None  # type: ignore[assignment]
    sk_mixture = None  # type: ignore[assignment]
    sk_neighbors = None  # type: ignore[assignment]

try:  # pragma: no cover - import availability tested via unit suite
    import hdbscan  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - covered by availability checks
    hdbscan = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def _optional_attr(module: Any | None, name: str) -> Any | None:
    """Safely fetch an attribute from an optional module."""

    if module is None:
        return None
    return getattr(module, name, None)


DEFAULT_MAX_CLUSTERS = 8
EMBEDDING_STD_DIM_SAMPLE = 5
ADDITIONAL_STOP_WORDS = {
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "was",
    "one",
    "our",
    "day",
    "get",
    "has",
    "his",
    "how",
    "man",
    "from",
    "they",
    "know",
    "want",
    "been",
    "some",
    "time",
    "when",
    "come",
    "here",
    "just",
    "like",
    "many",
    "over",
    "take",
    "them",
    "well",
    "were",
    "her",
    "had",
    "him",
    "good",
    "much",
    "make",
    "very",
    "long",
    "such",
    "than",
}
TEXT_STOP_WORDS = STOP_WORDS.union(ADDITIONAL_STOP_WORDS)


class ClusteringMethod(str, Enum):
    """Clustering algorithms available for result grouping."""

    HDBSCAN = "hdbscan"
    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    AUTO = "auto"


class ClusteringScope(str, Enum):
    """Scope of clustering operations."""

    STRICT = "strict"
    MODERATE = "moderate"
    INCLUSIVE = "inclusive"
    ADAPTIVE = "adaptive"


class SimilarityMetric(str, Enum):
    """Similarity metrics for clustering."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    HAMMING = "hamming"


class SearchResult(SearchRecord):
    """Individual search result for clustering."""

    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result snippet or body")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding for clustering"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("embedding")
    @classmethod
    def validate_embedding_size(cls, value: list[float] | None) -> list[float] | None:
        """Ensure embeddings are non-empty when provided."""

        if value is not None and not value:
            msg = "Embedding cannot be empty"
            raise ValueError(msg)
        return value


class ClusterGroup(BaseModel):
    """A group of clustered search results."""

    cluster_id: int = Field(..., description="Cluster identifier")
    label: str | None = Field(None, description="Human-readable cluster label")
    results: list[SearchResult] = Field(..., description="Results in this cluster")
    centroid: list[float] | None = Field(
        None, description="Cluster centroid coordinates"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Cluster confidence score"
    )
    size: int = Field(..., ge=0, description="Number of results in cluster")
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average relevance score")
    coherence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Intra-cluster coherence"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Representative keywords"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Cluster metadata"
    )

    model_config = ConfigDict(extra="allow")


class OutlierResult(BaseModel):
    """Results that did not fit into any cluster."""

    result: SearchResult = Field(..., description="Outlier result")
    distance_to_nearest_cluster: float = Field(
        ..., ge=0.0, description="Distance to nearest cluster centroid"
    )
    outlier_score: float = Field(
        ..., ge=0.0, le=1.0, description="Outlier confidence score"
    )

    model_config = ConfigDict(extra="allow")


class ResultClusteringRequest(BaseModel):
    """Request for result clustering operations."""

    results: list[SearchResult] = Field(..., description="Results to cluster")
    query: str | None = Field(None, description="Original search query")

    method: ClusteringMethod = Field(
        default=ClusteringMethod.HDBSCAN,
        description="Preferred clustering method",
    )
    scope: ClusteringScope = Field(
        default=ClusteringScope.MODERATE, description="Clustering scope"
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Similarity metric"
    )

    min_cluster_size: int = Field(3, ge=2, le=50, description="Minimum cluster size")
    max_clusters: int | None = Field(
        default=None, ge=2, le=50, description="Maximum number of clusters"
    )
    num_clusters: int | None = Field(
        default=None, ge=2, le=50, description="Presentation cluster target"
    )
    min_samples: int | None = Field(
        default=None, ge=1, description="Minimum samples for density-based clustering"
    )
    eps: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Epsilon for DBSCAN/HDBSCAN"
    )

    min_cluster_confidence: float = Field(
        0.6, ge=0.0, le=1.0, description="Minimum cluster confidence threshold"
    )
    outlier_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Outlier detection threshold"
    )

    use_hierarchical: bool = Field(
        False, description="Use hierarchical clustering features"
    )
    generate_labels: bool = Field(True, description="Generate cluster labels")
    extract_keywords: bool = Field(
        True, description="Extract representative keywords for clusters"
    )

    max_processing_time_ms: float = Field(
        5000.0, ge=100.0, description="Maximum allowed processing time"
    )
    enable_caching: bool = Field(True, description="Enable clustering result caching")

    model_config = ConfigDict(extra="allow")

    @field_validator("results")
    @classmethod
    def validate_results_count(cls, value: list[SearchResult]) -> list[SearchResult]:
        """Ensure we have enough results for clustering."""

        if len(value) < 3:
            msg = "Need at least 3 results for clustering"
            raise ValueError(msg)
        return value


class ResultClusteringResult(BaseModel):
    """Result of clustering operations."""

    clusters: list[ClusterGroup] = Field(default_factory=list, description="Clusters")
    outliers: list[OutlierResult] = Field(default_factory=list, description="Outliers")
    method_used: ClusteringMethod = Field(..., description="Algorithm used")
    total_results: int = Field(..., ge=0, description="Total input results")
    clustered_results: int = Field(..., ge=0, description="Results in clusters")
    outlier_count: int = Field(..., ge=0, description="Number of outliers")
    cluster_count: int = Field(..., ge=0, description="Number of clusters")

    silhouette_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Silhouette coefficient"
    )
    calinski_harabasz_score: float | None = Field(
        default=None, ge=0.0, description="Calinski-Harabasz index"
    )
    davies_bouldin_score: float | None = Field(
        default=None, ge=0.0, description="Davies-Bouldin index"
    )

    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")
    cache_hit: bool = Field(
        False, description="Whether the result was returned from cache"
    )
    clustering_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific metadata"
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")


def _raise_value_error(message: str) -> NoReturn:
    """Helper for explicit ValueError raising (aids static analysis)."""

    raise ValueError(message)


class ResultClusteringService:  # pylint: disable=too-many-instance-attributes
    """Result clustering service supporting multiple algorithms."""

    def __init__(
        self,
        enable_hdbscan: bool = True,
        enable_advanced_metrics: bool = True,
        cache_size: int = 500,
    ) -> None:
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        self.enable_hdbscan = enable_hdbscan
        self.enable_advanced_metrics = enable_advanced_metrics
        self._cache = CacheManager(cache_size)
        self._cache_size = cache_size
        self.available_algorithms = self._check_algorithm_availability()
        self._performance = PerformanceTracker()

    async def cluster_results(  # pylint: disable=too-many-locals
        self, request: ResultClusteringRequest
    ) -> ResultClusteringResult:
        """Cluster search results using the specified algorithm."""

        start_time = time.time()

        try:
            if request.enable_caching:
                cached = self._get_cached_result(request)
                if cached:
                    cached.cache_hit = True
                    return cached

            if not self._validate_clustering_request(request):
                _raise_value_error("Invalid clustering request")

            embeddings = self._extract_embeddings(request.results)
            if embeddings is None:
                _raise_value_error("No valid embeddings found in results")

            method = self._select_clustering_method(request, embeddings)
            cluster_labels, metadata = await self._apply_clustering(
                embeddings, method, request
            )

            clusters = self._build_cluster_groups(
                request.results, cluster_labels, embeddings, request
            )
            outliers = self._identify_outliers(
                request.results, cluster_labels, embeddings, request
            )
            metrics = self._calculate_quality_metrics(
                embeddings, cluster_labels, request
            )

            processing_time_ms = float((time.time() - start_time) * 1000.0)

            result = ResultClusteringResult(
                clusters=clusters,
                outliers=outliers,
                method_used=method,
                total_results=len(request.results),
                clustered_results=sum(len(cluster.results) for cluster in clusters),
                outlier_count=len(outliers),
                cluster_count=len(clusters),
                silhouette_score=metrics.get("silhouette_score"),
                calinski_harabasz_score=metrics.get("calinski_harabasz_score"),
                davies_bouldin_score=metrics.get("davies_bouldin_score"),
                processing_time_ms=processing_time_ms,
                cache_hit=False,
                clustering_metadata={
                    **metadata,
                    "embedding_dimensions": int(embeddings.shape[1]),
                    "algorithm_parameters": self._get_algorithm_parameters(
                        method, request
                    ),
                },
            )

            if request.enable_caching:
                self._cache_result(request, result)

            self._update_performance_stats(method, processing_time_ms)

            self._logger.info(
                "Clustered %d results into %d clusters with %d outliers "
                "using %s in %.1f ms",
                len(request.results),
                len(result.clusters),
                len(result.outliers),
                method.value,
                processing_time_ms,
            )

            return result

        except Exception as exc:  # pragma: no cover - behaviour asserted by tests
            processing_time_ms = float((time.time() - start_time) * 1000.0)
            self._logger.exception("Result clustering failed")

            outliers = [
                OutlierResult(
                    result=search_result,
                    distance_to_nearest_cluster=1.0,
                    outlier_score=1.0,
                )
                for search_result in request.results
            ]

            fallback_method = (
                request.method
                if request.method != ClusteringMethod.AUTO
                else ClusteringMethod.KMEANS
            )

            error_message = str(exc) or "Result clustering failed"

            return ResultClusteringResult(
                clusters=[],
                outliers=outliers,
                method_used=fallback_method,
                total_results=len(request.results),
                clustered_results=0,
                outlier_count=len(outliers),
                cluster_count=0,
                silhouette_score=None,
                calinski_harabasz_score=None,
                davies_bouldin_score=None,
                processing_time_ms=processing_time_ms,
                cache_hit=False,
                clustering_metadata={"error": error_message},
            )

    def _check_algorithm_availability(self) -> dict[str, bool]:
        """Check which clustering algorithms are available."""

        availability = {
            "sklearn": _optional_attr(sk_cluster, "DBSCAN") is not None
            and _optional_attr(sk_cluster, "KMeans") is not None,
            "hdbscan": hdbscan is not None and self.enable_hdbscan,
            "numpy": True,
        }
        if not availability["hdbscan"]:
            self.enable_hdbscan = False
        return availability

    def _validate_clustering_request(self, request: ResultClusteringRequest) -> bool:
        """Validate clustering request parameters."""

        if len(request.results) < request.min_cluster_size:
            return False

        if request.method == ClusteringMethod.HDBSCAN and not self.enable_hdbscan:
            return False

        embeddings_with_data = sum(
            1 for result in request.results if result.embedding is not None
        )
        return embeddings_with_data >= request.min_cluster_size

    def _extract_embeddings(self, results: list[SearchResult]) -> FloatArray | None:
        """Extract embeddings from results."""

        vectors = [
            result.embedding
            for result in results
            if result.embedding is not None and len(result.embedding) > 0
        ]

        if len(vectors) < 3:
            return None

        embedding_array = np.array(vectors, dtype=float)
        if embedding_array.ndim != 2:
            return None

        return embedding_array

    def _select_clustering_method(
        self, request: ResultClusteringRequest, embeddings: FloatArray
    ) -> ClusteringMethod:
        """Select the optimal clustering method."""

        if request.method != ClusteringMethod.AUTO:
            return request.method

        sample_count = int(embeddings.shape[0])
        if self.enable_hdbscan and sample_count <= 50:
            return ClusteringMethod.HDBSCAN
        if sample_count < 100:
            return ClusteringMethod.DBSCAN
        if request.max_clusters is not None or request.num_clusters is not None:
            return ClusteringMethod.KMEANS
        return ClusteringMethod.AGGLOMERATIVE

    async def _apply_clustering(
        self,
        embeddings: FloatArray,
        method: ClusteringMethod,
        request: ResultClusteringRequest,
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply the selected clustering algorithm."""

        metadata: dict[str, Any] = {"method": method.value}

        if method == ClusteringMethod.HDBSCAN:
            return self._apply_hdbscan(embeddings, request, metadata)
        if method == ClusteringMethod.DBSCAN:
            return self._apply_dbscan(embeddings, request, metadata)
        if method == ClusteringMethod.KMEANS:
            return self._apply_kmeans(embeddings, request, metadata)
        if method == ClusteringMethod.AGGLOMERATIVE:
            return self._apply_agglomerative(embeddings, request, metadata)
        if method == ClusteringMethod.SPECTRAL:
            return self._apply_spectral(embeddings, request, metadata)
        if method == ClusteringMethod.GAUSSIAN_MIXTURE:
            return self._apply_gaussian_mixture(embeddings, request, metadata)

        _raise_value_error(f"Unsupported clustering method: {method}")

    def _apply_hdbscan(
        self,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply HDBSCAN clustering."""

        if hdbscan is None:
            _raise_value_error("HDBSCAN requires the 'hdbscan' package")

        min_cluster_size = max(request.min_cluster_size, 3)
        min_samples = request.min_samples or max(2, min_cluster_size - 1)

        metric = sklearn_metric_for(request.similarity_metric)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=request.eps,
        )

        labels = clusterer.fit_predict(embeddings)

        n_clusters, n_noise = self._cluster_noise_stats(labels)
        metadata.update(
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "cluster_persistence": getattr(clusterer, "cluster_persistence_", None),
            }
        )

        return labels, metadata

    def _apply_dbscan(
        self,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply DBSCAN clustering."""

        dbscan_cls = _optional_attr(sk_cluster, "DBSCAN")
        if dbscan_cls is None:
            _raise_value_error("DBSCAN requires scikit-learn")

        eps = request.eps or self._estimate_eps(
            embeddings, request.min_cluster_size, request.similarity_metric
        )
        min_samples = request.min_samples or request.min_cluster_size

        metric = sklearn_metric_for(request.similarity_metric)

        clusterer = dbscan_cls(eps=eps, min_samples=min_samples, metric=metric)
        labels = clusterer.fit_predict(embeddings)

        n_clusters, n_noise = self._cluster_noise_stats(labels)

        metadata.update(
            {
                "eps": float(eps),
                "min_samples": int(min_samples),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
            }
        )

        return labels, metadata

    def _apply_kmeans(
        self,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply K-means clustering."""

        kmeans_cls = _optional_attr(sk_cluster, "KMeans")
        if kmeans_cls is None:
            _raise_value_error("KMeans requires scikit-learn")

        n_clusters = self._resolve_cluster_count(embeddings, request)

        clusterer = kmeans_cls(
            n_clusters=int(n_clusters), random_state=42, n_init="auto"
        )
        labels = clusterer.fit_predict(embeddings)

        inertia_value = (
            float(clusterer.inertia_) if clusterer.inertia_ is not None else 0.0
        )

        metadata.update(
            {
                "n_clusters": int(n_clusters),
                "inertia": inertia_value,
                "n_iter": int(clusterer.n_iter_),
            }
        )

        return labels, metadata

    def _apply_agglomerative(
        self,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply agglomerative clustering."""

        agglomerative_cls = _optional_attr(sk_cluster, "AgglomerativeClustering")
        if agglomerative_cls is None:
            _raise_value_error("AgglomerativeClustering requires scikit-learn")

        n_clusters = self._resolve_cluster_count(embeddings, request)

        metric = request.similarity_metric.value
        linkage = "ward" if metric == "euclidean" else "average"
        if metric not in {"euclidean", "manhattan", "cosine"}:
            metric = "euclidean"

        clusterer = agglomerative_cls(
            n_clusters=int(n_clusters),
            metric=metric,
            linkage=linkage,
        )
        labels = clusterer.fit_predict(embeddings)

        metadata.update(
            {"n_clusters": int(n_clusters), "linkage": linkage, "metric": metric}
        )
        return labels, metadata

    def _apply_spectral(
        self,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply spectral clustering."""

        spectral_cls = _optional_attr(sk_cluster, "SpectralClustering")
        if spectral_cls is None:
            _raise_value_error("SpectralClustering requires scikit-learn")

        n_clusters = self._resolve_cluster_count(embeddings, request)

        affinity = (
            "cosine" if request.similarity_metric == SimilarityMetric.COSINE else "rbf"
        )

        clusterer = spectral_cls(
            n_clusters=int(n_clusters),
            affinity=affinity,
            random_state=42,
        )
        labels = clusterer.fit_predict(embeddings)

        metadata.update({"n_clusters": int(n_clusters), "affinity": affinity})
        return labels, metadata

    def _apply_gaussian_mixture(
        self,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[IntArray, dict[str, Any]]:
        """Apply Gaussian mixture model clustering."""

        gaussian_mixture_cls = _optional_attr(sk_mixture, "GaussianMixture")
        if gaussian_mixture_cls is None:
            _raise_value_error("GaussianMixture requires scikit-learn")

        n_components = self._resolve_cluster_count(embeddings, request)

        clusterer = gaussian_mixture_cls(
            n_components=int(n_components), covariance_type="full", random_state=42
        )
        labels = clusterer.fit_predict(embeddings)

        lower_bound = (
            float(clusterer.lower_bound_)
            if getattr(clusterer, "lower_bound_", None) is not None
            else 0.0
        )

        metadata.update(
            {
                "n_components": int(n_components),
                "bic": float(clusterer.bic(embeddings)),
                "aic": float(clusterer.aic(embeddings)),
                "lower_bound": lower_bound,
            }
        )
        return labels, metadata

    def _estimate_eps(
        self,
        embeddings: FloatArray,
        min_cluster_size: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> float:
        """Estimate eps parameter for DBSCAN using k-distance."""

        neighbors_cls = _optional_attr(sk_neighbors, "NearestNeighbors")
        if neighbors_cls is None:
            _raise_value_error("NearestNeighbors requires scikit-learn")

        neighbors = neighbors_cls(
            n_neighbors=min_cluster_size,
            metric=sklearn_metric_for(similarity_metric),
        )
        neighbors.fit(embeddings)
        distances, _ = neighbors.kneighbors(embeddings)
        k_distances = np.sort(distances[:, -1])
        eps = float(np.percentile(k_distances, 75))
        return float(min(eps, 0.5))

    def _build_cluster_groups(  # pylint: disable=too-many-locals
        self,
        results: list[SearchResult],
        cluster_labels: IntArray,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
    ) -> list[ClusterGroup]:
        """Build cluster groups from clustering results."""

        clusters: list[ClusterGroup] = []
        unique_labels = {int(label) for label in cluster_labels if label != -1}

        for cluster_id in sorted(unique_labels):
            mask = cluster_labels == cluster_id
            cluster_results = [
                results[i] for i, selected in enumerate(mask) if selected
            ]
            if len(cluster_results) < request.min_cluster_size:
                continue

            cluster_embeddings = embeddings[mask]
            summary = self._summarize_cluster(
                cluster_embeddings, cluster_results, embeddings, request
            )

            label = None
            if request.generate_labels:
                label = self._generate_cluster_label(cluster_results, request.query)

            keywords: list[str] = []
            if request.extract_keywords:
                keywords = self._extract_cluster_keywords(cluster_results)

            clusters.append(
                ClusterGroup(
                    cluster_id=int(cluster_id),
                    label=label,
                    results=cluster_results,
                    centroid=summary["centroid"],
                    confidence=summary["confidence"],
                    size=len(cluster_results),
                    avg_score=summary["avg_score"],
                    coherence_score=summary["coherence"],
                    keywords=keywords,
                    metadata={
                        "embedding_std": summary["embedding_std"],
                        "score_std": summary["score_std"],
                    },
                )
            )

        clusters.sort(key=lambda cluster: cluster.size, reverse=True)
        return clusters

    def _identify_outliers(
        self,
        results: list[SearchResult],
        cluster_labels: IntArray,
        embeddings: FloatArray,
        request: ResultClusteringRequest,
    ) -> list[OutlierResult]:
        """Identify outlier results that do not belong to any cluster."""

        outliers: list[OutlierResult] = []
        noise_indices = np.where(cluster_labels == -1)[0]
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label != -1]

        if not cluster_indices:
            return [
                OutlierResult(
                    result=results[int(index)],
                    distance_to_nearest_cluster=1.0,
                    outlier_score=1.0,
                )
                for index in noise_indices
            ]

        centroids = {}
        for label in {int(lbl) for lbl in cluster_labels if lbl != -1}:
            mask = cluster_labels == label
            centroids[label] = embeddings[mask].mean(axis=0)

        for index in noise_indices:
            embedding = embeddings[int(index)]
            min_distance = float("inf")
            for centroid in centroids.values():
                distance = distance_for_metric(
                    embedding, centroid, request.similarity_metric
                )
                min_distance = min(min_distance, distance)
            if min_distance == float("inf"):
                min_distance = 1.0

            outliers.append(
                OutlierResult(
                    result=results[int(index)],
                    distance_to_nearest_cluster=float(min_distance),
                    outlier_score=float(min(1.0, min_distance / 2.0)),
                )
            )

        return outliers

    def _calculate_coherence(self, cluster_embeddings: FloatArray) -> float:
        """Calculate intra-cluster coherence score."""

        if cluster_embeddings.shape[0] < 2:
            return 1.0

        norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = cluster_embeddings / norms
        similarity_matrix = normalised @ normalised.T
        upper = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
        if upper.size == 0:
            return 1.0
        coherence = float(np.mean(upper))
        return float(np.clip(coherence, 0.0, 1.0))

    def _calculate_cluster_confidence(
        self,
        cluster_embeddings: FloatArray,
        _all_embeddings: FloatArray,
        request: ResultClusteringRequest,
    ) -> float:
        """Calculate confidence score for a cluster."""

        if cluster_embeddings.shape[0] < 2:
            return 0.0

        coherence = self._calculate_coherence(cluster_embeddings)
        size_factor = min(
            1.0, cluster_embeddings.shape[0] / (request.min_cluster_size * 2)
        )
        confidence = (coherence * 0.7) + (size_factor * 0.3)
        return float(np.clip(confidence, 0.0, 1.0))

    def _calculate_quality_metrics(
        self,
        embeddings: FloatArray,
        cluster_labels: IntArray,
        request: ResultClusteringRequest,
    ) -> dict[str, float]:
        """Calculate clustering quality metrics."""

        if not self.enable_advanced_metrics:
            return {}

        silhouette_fn = _optional_attr(sk_metrics, "silhouette_score")
        calinski_fn = _optional_attr(sk_metrics, "calinski_harabasz_score")
        davies_fn = _optional_attr(sk_metrics, "davies_bouldin_score")

        if silhouette_fn is None:
            return {}

        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) < 2:
            return {}

        valid_labels = cluster_labels[valid_mask]
        if len({int(label) for label in valid_labels}) < 2:
            return {}

        valid_embeddings = embeddings[valid_mask]
        metrics: dict[str, float] = {}
        metric_name = sklearn_metric_for(request.similarity_metric)

        try:
            metrics["silhouette_score"] = float(
                silhouette_fn(valid_embeddings, valid_labels, metric=metric_name)
            )
            if calinski_fn is not None:
                metrics["calinski_harabasz_score"] = float(
                    calinski_fn(valid_embeddings, valid_labels)
                )
            if davies_fn is not None:
                metrics["davies_bouldin_score"] = float(
                    davies_fn(valid_embeddings, valid_labels)
                )
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            self._logger.warning("Failed to calculate quality metrics: %s", exc)

        return metrics

    def _resolve_cluster_count(
        self, embeddings: FloatArray, request: ResultClusteringRequest
    ) -> int:
        """Determine the target number of clusters for bounded algorithms."""

        explicit = request.max_clusters or request.num_clusters
        if explicit is not None:
            return int(explicit)
        fallback = max(2, embeddings.shape[0] // request.min_cluster_size)
        return int(min(DEFAULT_MAX_CLUSTERS, fallback))

    @staticmethod
    def _cluster_noise_stats(labels: IntArray) -> tuple[int, int]:
        """Return cluster and noise counts for label arrays."""

        n_noise = int(np.sum(labels == -1))
        n_clusters = int(len({int(lbl) for lbl in labels if lbl != -1}))
        return n_clusters, n_noise

    def _summarize_cluster(
        self,
        cluster_embeddings: FloatArray,
        cluster_results: list[SearchResult],
        all_embeddings: FloatArray,
        request: ResultClusteringRequest,
    ) -> dict[str, Any]:
        """Compute aggregate metrics for a cluster."""

        centroid_vector = cluster_embeddings.mean(axis=0)
        scores = np.array([res.score for res in cluster_results], dtype=float)
        coherence = self._calculate_coherence(cluster_embeddings)
        confidence = self._calculate_cluster_confidence(
            cluster_embeddings, all_embeddings, request
        )
        embedding_std = cluster_embeddings.std(axis=0)

        return {
            "centroid": [float(value) for value in centroid_vector],
            "avg_score": float(scores.mean()) if scores.size else 0.0,
            "score_std": float(scores.std()) if scores.size else 0.0,
            "coherence": float(coherence),
            "confidence": float(confidence),
            "embedding_std": [
                float(value) for value in embedding_std[:EMBEDDING_STD_DIM_SAMPLE]
            ],
        }

    def _generate_cluster_label(
        self, results: list[SearchResult], query: str | None = None
    ) -> str:
        """Generate human-readable label for a cluster."""

        titles = " ".join(result.title for result in results[:5])
        words = re.findall(r"\b\w{3,}\b", titles.lower())
        filtered = [word for word in words if word not in TEXT_STOP_WORDS]
        if query:
            filtered.extend(re.findall(r"\b\w{3,}\b", query.lower()))

        if not filtered:
            return "Cluster"

        counter = Counter(filtered)
        most_common = counter.most_common(2)
        label = " ".join(word.title() for word, _ in most_common)
        return label or "Cluster"

    def _extract_cluster_keywords(self, results: list[SearchResult]) -> list[str]:
        """Extract representative keywords for a cluster."""

        text = " ".join(result.content.lower() for result in results)
        words = re.findall(r"\b\w{3,}\b", text)
        filtered = [word for word in words if word not in TEXT_STOP_WORDS]
        counter = Counter(filtered)
        return [word for word, count in counter.most_common(8) if count >= 2]

    def _get_algorithm_parameters(
        self, method: ClusteringMethod, request: ResultClusteringRequest
    ) -> dict[str, Any]:
        """Get algorithm-specific parameters used."""

        params: dict[str, Any] = {
            "method": method.value,
            "min_cluster_size": request.min_cluster_size,
            "similarity_metric": request.similarity_metric.value,
            "scope": request.scope.value,
        }
        if request.max_clusters is not None:
            params["max_clusters"] = request.max_clusters
        if request.num_clusters is not None:
            params["num_clusters"] = request.num_clusters
        if request.min_samples is not None:
            params["min_samples"] = request.min_samples
        if request.eps is not None:
            params["eps"] = request.eps
        return params

    def _generate_cache_key(self, request: ResultClusteringRequest) -> str:
        """Generate cache key for request."""

        result_ids = sorted(result.id for result in request.results)
        parts = [
            ",".join(result_ids),
            request.method.value,
            str(request.min_cluster_size),
            str(request.max_clusters or ""),
            str(request.num_clusters or ""),
            request.similarity_metric.value,
        ]
        return build_cache_key(*parts)

    def _get_cached_result(
        self, request: ResultClusteringRequest
    ) -> ResultClusteringResult | None:
        """Get cached clustering result if available."""

        cache_key = self._generate_cache_key(request)
        return self._cache.get(cache_key)

    def _cache_result(
        self, request: ResultClusteringRequest, result: ResultClusteringResult
    ) -> None:
        """Cache clustering result."""

        cache_key = self._generate_cache_key(request)
        self._cache.set(cache_key, result)

    def _update_performance_stats(
        self, method: ClusteringMethod, processing_time: float
    ) -> None:
        """Update performance statistics."""

        self._performance.record(processing_time, label=method.value)

    def get_performance_stats(self) -> dict[str, Any]:
        """Return performance statistics including cache state."""

        snapshot = performance_snapshot(self._performance)
        formatted_stats = {
            "_total_clusterings": snapshot["total_operations"],
            "avg_processing_time": snapshot["avg_processing_time"],
            "method_usage": snapshot["counters"],
            "available_algorithms": dict(self.available_algorithms),
        }
        return merge_performance_metadata(
            performance_stats=formatted_stats,
            cache_tracker=self._cache.tracker,
            cache_size=len(self._cache),
        )

    def clear_cache(self) -> None:
        """Clear clustering cache."""

        self._cache.clear()

    @property
    def clustering_cache(self) -> dict[str, ResultClusteringResult]:
        """Return a snapshot of cached clustering results."""

        return self._cache.snapshot()

    @property
    def cache_size(self) -> int:
        """Return the configured cache size."""

        return self._cache_size

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache hit and miss counts."""

        return {
            "hits": self._cache.tracker.hits,
            "misses": self._cache.tracker.misses,
        }

    @property
    def performance_stats(self) -> dict[str, Any]:
        """Expose performance statistics via property access."""

        return self.get_performance_stats()
