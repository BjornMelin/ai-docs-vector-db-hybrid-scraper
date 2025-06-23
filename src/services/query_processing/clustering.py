
"""Result clustering service for semantic grouping of search results.

This module provides advanced result clustering capabilities using various algorithms
including HDBSCAN, DBSCAN, K-means, and agglomerative clustering to semantically
group search results for better organization and presentation.
"""

import logging
import warnings
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ClusteringMethod(str, Enum):
    """Clustering algorithms available for result grouping."""

    HDBSCAN = "hdbscan"  # Hierarchical DBSCAN (recommended)
    DBSCAN = "dbscan"  # Density-based clustering
    KMEANS = "kmeans"  # K-means clustering
    AGGLOMERATIVE = "agglomerative"  # Hierarchical agglomerative
    SPECTRAL = "spectral"  # Spectral clustering
    GAUSSIAN_MIXTURE = "gaussian_mixture"  # Gaussian mixture models
    AUTO = "auto"  # Automatic method selection


class ClusteringScope(str, Enum):
    """Scope of clustering operations."""

    STRICT = "strict"  # High confidence clusters only
    MODERATE = "moderate"  # Balanced clustering
    INCLUSIVE = "inclusive"  # Include more results in clusters
    ADAPTIVE = "adaptive"  # Adapt based on data characteristics


class SimilarityMetric(str, Enum):
    """Similarity metrics for clustering."""

    COSINE = "cosine"  # Cosine similarity (default for embeddings)
    EUCLIDEAN = "euclidean"  # Euclidean distance
    MANHATTAN = "manhattan"  # Manhattan distance
    JACCARD = "jaccard"  # Jaccard similarity
    HAMMING = "hamming"  # Hamming distance


class SearchResult(BaseModel):
    """Individual search result for clustering."""

    id: str = Field(..., description="Unique result identifier")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content or snippet")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    embedding: list[float] | None = Field(
        None, description="Vector embedding for clustering"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding_size(cls, v):
        """Validate embedding dimensions."""
        if v is not None and len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        return v


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


class OutlierResult(BaseModel):
    """Results that don't fit into any cluster."""

    result: SearchResult = Field(..., description="Outlier result")
    distance_to_nearest_cluster: float = Field(
        ..., ge=0.0, description="Distance to nearest cluster"
    )
    outlier_score: float = Field(
        ..., ge=0.0, le=1.0, description="Outlier confidence score"
    )


class ResultClusteringRequest(BaseModel):
    """Request for result clustering operations."""

    # Core clustering data
    results: list[SearchResult] = Field(..., description="Results to cluster")
    query: str | None = Field(None, description="Original search query")

    # Clustering configuration
    method: ClusteringMethod = Field(
        ClusteringMethod.HDBSCAN, description="Clustering algorithm"
    )
    scope: ClusteringScope = Field(
        ClusteringScope.MODERATE, description="Clustering scope"
    )
    similarity_metric: SimilarityMetric = Field(
        SimilarityMetric.COSINE, description="Similarity metric"
    )

    # Algorithm parameters
    min_cluster_size: int = Field(3, ge=2, le=20, description="Minimum cluster size")
    max_clusters: int | None = Field(
        None, ge=2, le=50, description="Maximum number of clusters"
    )
    min_samples: int | None = Field(
        None, ge=1, description="Minimum samples for HDBSCAN/DBSCAN"
    )
    eps: float | None = Field(None, ge=0.0, le=2.0, description="Epsilon for DBSCAN")

    # Quality controls
    min_cluster_confidence: float = Field(
        0.6, ge=0.0, le=1.0, description="Minimum cluster confidence"
    )
    outlier_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Outlier detection threshold"
    )

    # Processing options
    use_hierarchical: bool = Field(
        False, description="Use hierarchical clustering features"
    )
    generate_labels: bool = Field(
        True, description="Generate human-readable cluster labels"
    )
    extract_keywords: bool = Field(True, description="Extract representative keywords")

    # Performance settings
    max_processing_time_ms: float = Field(
        5000.0, ge=100.0, description="Maximum processing time"
    )
    enable_caching: bool = Field(True, description="Enable clustering result caching")

    @field_validator("results")
    @classmethod
    def validate_results_count(cls, v):
        """Validate minimum results for clustering."""
        if len(v) < 3:
            raise ValueError("Need at least 3 results for clustering")
        return v


class ResultClusteringResult(BaseModel):
    """Result of clustering operations."""

    clusters: list[ClusterGroup] = Field(..., description="Identified clusters")
    outliers: list[OutlierResult] = Field(
        default_factory=list, description="Outlier results"
    )

    # Clustering metadata
    method_used: ClusteringMethod = Field(..., description="Algorithm used")
    total_results: int = Field(..., ge=0, description="Total input results")
    clustered_results: int = Field(..., ge=0, description="Results in clusters")
    outlier_count: int = Field(..., ge=0, description="Number of outliers")
    cluster_count: int = Field(..., ge=0, description="Number of clusters")

    # Quality metrics
    silhouette_score: float | None = Field(
        None, ge=-1.0, le=1.0, description="Silhouette coefficient"
    )
    calinski_harabasz_score: float | None = Field(
        None, ge=0.0, description="Calinski-Harabasz index"
    )
    davies_bouldin_score: float | None = Field(
        None, ge=0.0, description="Davies-Bouldin index"
    )

    # Performance metrics
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")
    cache_hit: bool = Field(False, description="Whether result was cached")

    # Analysis metadata
    clustering_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific metadata"
    )


class ResultClusteringService:
    """Advanced result clustering service with multiple algorithms."""

    def __init__(
        self,
        enable_hdbscan: bool = True,
        enable_advanced_metrics: bool = True,
        cache_size: int = 500,
    ):
        """Initialize result clustering service.

        Args:
            enable_hdbscan: Enable HDBSCAN algorithm (requires hdbscan package)
            enable_advanced_metrics: Enable advanced clustering metrics
            cache_size: Size of clustering cache
        """
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Configuration
        self.enable_hdbscan = enable_hdbscan
        self.enable_advanced_metrics = enable_advanced_metrics

        # Caching
        self.clustering_cache = {}
        self.cache_size = cache_size
        self.cache_stats = {"hits": 0, "misses": 0}

        # Algorithm availability
        self.available_algorithms = self._check_algorithm_availability()

        # Performance tracking
        self.performance_stats = {
            "total_clusterings": 0,
            "avg_processing_time": 0.0,
            "method_usage": {},
        }

    async def cluster_results(
        self, request: ResultClusteringRequest
    ) -> ResultClusteringResult:
        """Cluster search results using the specified algorithm.

        Args:
            request: Clustering request with results and parameters

        Returns:
            ResultClusteringResult with clustered groups and metadata
        """
        import time

        start_time = time.time()

        try:
            # Check cache first
            if request.enable_caching:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    cached_result.cache_hit = True
                    return cached_result

            # Validate request
            if not self._validate_clustering_request(request):
                raise ValueError("Invalid clustering request")

            # Extract embeddings
            embeddings = self._extract_embeddings(request.results)
            if embeddings is None:
                raise ValueError("No valid embeddings found in results")

            # Select and apply clustering method
            method = self._select_clustering_method(request, embeddings)
            cluster_labels, cluster_metadata = await self._apply_clustering(
                embeddings, method, request
            )

            # Build cluster groups
            clusters = self._build_cluster_groups(
                request.results, cluster_labels, embeddings, request
            )

            # Identify outliers
            outliers = self._identify_outliers(
                request.results, cluster_labels, embeddings, request
            )

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                embeddings, cluster_labels, request
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Build result
            result = ResultClusteringResult(
                clusters=clusters,
                outliers=outliers,
                method_used=method,
                total_results=len(request.results),
                clustered_results=sum(len(c.results) for c in clusters),
                outlier_count=len(outliers),
                cluster_count=len(clusters),
                silhouette_score=quality_metrics.get("silhouette_score"),
                calinski_harabasz_score=quality_metrics.get("calinski_harabasz_score"),
                davies_bouldin_score=quality_metrics.get("davies_bouldin_score"),
                processing_time_ms=processing_time_ms,
                cache_hit=False,
                clustering_metadata={
                    **cluster_metadata,
                    "embedding_dimensions": embeddings.shape[1],
                    "algorithm_parameters": self._get_algorithm_parameters(
                        method, request
                    ),
                },
            )

            # Cache result
            if request.enable_caching:
                self._cache_result(request, result)

            # Update performance stats
            self._update_performance_stats(method, processing_time_ms)

            self._logger.info(
                f"Clustered {len(request.results)} results into {len(clusters)} clusters "
                f"with {len(outliers)} outliers using {method.value} in {processing_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._logger.error(f"Result clustering failed: {e}", exc_info=True)

            # Return fallback result
            return ResultClusteringResult(
                clusters=[],
                outliers=[],
                method_used=request.method,
                total_results=len(request.results),
                clustered_results=0,
                outlier_count=len(request.results),
                cluster_count=0,
                processing_time_ms=processing_time_ms,
                clustering_metadata={"error": str(e)},
            )

    def _check_algorithm_availability(self) -> dict[str, bool]:
        """Check which clustering algorithms are available."""
        algorithms = {}

        try:
            import sklearn.cluster  # noqa: F401

            algorithms["sklearn"] = True
        except ImportError:
            algorithms["sklearn"] = False

        try:
            import hdbscan  # noqa: F401

            algorithms["hdbscan"] = True
        except ImportError:
            algorithms["hdbscan"] = False
            self.enable_hdbscan = False

        try:
            import numpy  # noqa: F401

            algorithms["numpy"] = True
        except ImportError:
            algorithms["numpy"] = False

        return algorithms

    def _validate_clustering_request(self, request: ResultClusteringRequest) -> bool:
        """Validate clustering request parameters."""
        # Check minimum results
        if len(request.results) < request.min_cluster_size:
            return False

        # Check algorithm availability
        if request.method == ClusteringMethod.HDBSCAN and not self.enable_hdbscan:
            return False

        # Check embeddings
        valid_embeddings = sum(1 for r in request.results if r.embedding is not None)
        return not valid_embeddings < request.min_cluster_size

    def _extract_embeddings(self, results: list[SearchResult]) -> np.ndarray | None:
        """Extract and validate embeddings from results."""
        embeddings = []

        for result in results:
            if result.embedding is not None and len(result.embedding) > 0:
                embeddings.append(result.embedding)

        if len(embeddings) < 3:
            return None

        # Convert to numpy array and validate dimensions
        embedding_array = np.array(embeddings)

        # Check for consistent dimensions
        if embedding_array.ndim != 2:
            return None

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embedding_array = embedding_array / norms

        return embedding_array

    def _select_clustering_method(
        self, request: ResultClusteringRequest, embeddings: np.ndarray
    ) -> ClusteringMethod:
        """Select the optimal clustering method."""
        if request.method != ClusteringMethod.AUTO:
            return request.method

        # Auto-select based on data characteristics
        n_samples = embeddings.shape[0]

        if n_samples < 50 and self.enable_hdbscan:
            return ClusteringMethod.HDBSCAN
        elif n_samples < 100:
            return ClusteringMethod.DBSCAN
        elif request.max_clusters is not None:
            return ClusteringMethod.KMEANS
        else:
            return ClusteringMethod.AGGLOMERATIVE

    async def _apply_clustering(
        self,
        embeddings: np.ndarray,
        method: ClusteringMethod,
        request: ResultClusteringRequest,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply the selected clustering algorithm."""
        metadata = {"method": method.value}

        if method == ClusteringMethod.HDBSCAN:
            return self._apply_hdbscan(embeddings, request, metadata)
        elif method == ClusteringMethod.DBSCAN:
            return self._apply_dbscan(embeddings, request, metadata)
        elif method == ClusteringMethod.KMEANS:
            return self._apply_kmeans(embeddings, request, metadata)
        elif method == ClusteringMethod.AGGLOMERATIVE:
            return self._apply_agglomerative(embeddings, request, metadata)
        elif method == ClusteringMethod.SPECTRAL:
            return self._apply_spectral(embeddings, request, metadata)
        elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
            return self._apply_gaussian_mixture(embeddings, request, metadata)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

    def _apply_hdbscan(
        self,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply HDBSCAN clustering."""
        try:
            import hdbscan
        except ImportError as err:
            raise ImportError("HDBSCAN requires the 'hdbscan' package") from err

        # Configure parameters
        min_cluster_size = max(request.min_cluster_size, 3)
        min_samples = request.min_samples or max(2, min_cluster_size - 1)

        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="cosine"
            if request.similarity_metric == SimilarityMetric.COSINE
            else "euclidean",
            cluster_selection_epsilon=request.eps,
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        metadata.update(
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": len(set(cluster_labels))
                - (1 if -1 in cluster_labels else 0),
                "n_noise": list(cluster_labels).count(-1),
                "cluster_persistence": getattr(clusterer, "cluster_persistence_", None),
            }
        )

        return cluster_labels, metadata

    def _apply_dbscan(
        self,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply DBSCAN clustering."""
        from sklearn.cluster import DBSCAN

        # Auto-determine eps if not provided
        eps = request.eps
        if eps is None:
            eps = self._estimate_eps(embeddings, request.min_cluster_size)

        min_samples = request.min_samples or request.min_cluster_size

        # Apply DBSCAN
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine"
            if request.similarity_metric == SimilarityMetric.COSINE
            else "euclidean",
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        metadata.update(
            {
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": len(set(cluster_labels))
                - (1 if -1 in cluster_labels else 0),
                "n_noise": list(cluster_labels).count(-1),
            }
        )

        return cluster_labels, metadata

    def _apply_kmeans(
        self,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply K-means clustering."""
        from sklearn.cluster import KMeans

        # Determine number of clusters
        n_clusters = request.max_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(embeddings) // request.min_cluster_size))

        # Apply K-means
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        cluster_labels = clusterer.fit_predict(embeddings)

        metadata.update(
            {
                "n_clusters": n_clusters,
                "inertia": clusterer.inertia_,
                "n_iter": clusterer.n_iter_,
            }
        )

        return cluster_labels, metadata

    def _apply_agglomerative(
        self,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering

        # Determine number of clusters
        n_clusters = request.max_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(embeddings) // request.min_cluster_size))

        # Apply agglomerative clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", metric="euclidean"
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        metadata.update({"n_clusters": n_clusters, "linkage": "ward"})

        return cluster_labels, metadata

    def _apply_spectral(
        self,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply spectral clustering."""
        from sklearn.cluster import SpectralClustering

        # Determine number of clusters
        n_clusters = request.max_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(embeddings) // request.min_cluster_size))

        # Apply spectral clustering
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity="cosine"
            if request.similarity_metric == SimilarityMetric.COSINE
            else "rbf",
            random_state=42,
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        metadata.update({"n_clusters": n_clusters, "affinity": clusterer.affinity})

        return cluster_labels, metadata

    def _apply_gaussian_mixture(
        self,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply Gaussian mixture model clustering."""
        from sklearn.mixture import GaussianMixture

        # Determine number of components
        n_components = request.max_clusters
        if n_components is None:
            n_components = min(8, max(2, len(embeddings) // request.min_cluster_size))

        # Apply Gaussian mixture
        clusterer = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=42
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        metadata.update(
            {
                "n_components": n_components,
                "bic": clusterer.bic(embeddings),
                "aic": clusterer.aic(embeddings),
                "lower_bound": clusterer.lower_bound_,
            }
        )

        return cluster_labels, metadata

    def _estimate_eps(self, embeddings: np.ndarray, min_cluster_size: int) -> float:
        """Estimate eps parameter for DBSCAN using k-distance."""
        from sklearn.neighbors import NearestNeighbors

        # Use k = min_cluster_size for k-distance
        k = min_cluster_size

        neighbors = NearestNeighbors(n_neighbors=k, metric="cosine")
        neighbors.fit(embeddings)
        distances, _ = neighbors.kneighbors(embeddings)

        # Sort k-distances
        k_distances = np.sort(distances[:, -1])

        # Use knee point heuristic
        # Simple implementation: use 75th percentile
        eps = np.percentile(k_distances, 75)

        return min(eps, 0.5)  # Cap at reasonable maximum

    def _build_cluster_groups(
        self,
        results: list[SearchResult],
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
    ) -> list[ClusterGroup]:
        """Build cluster groups from clustering results."""
        clusters = []
        unique_labels = set(cluster_labels)

        # Remove noise label (-1) if present
        unique_labels.discard(-1)

        for cluster_id in unique_labels:
            # Get results in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_results = [
                results[i] for i in range(len(results)) if cluster_mask[i]
            ]
            cluster_embeddings = embeddings[cluster_mask]

            # Skip small clusters if required
            if len(cluster_results) < request.min_cluster_size:
                continue

            # Calculate cluster metrics
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            avg_score = np.mean([r.score for r in cluster_results])
            coherence_score = self._calculate_coherence(cluster_embeddings)
            confidence = self._calculate_cluster_confidence(
                cluster_embeddings, embeddings, request
            )

            # Generate cluster label and keywords
            label = None
            keywords = []
            if request.generate_labels:
                label = self._generate_cluster_label(cluster_results, request.query)
            if request.extract_keywords:
                keywords = self._extract_cluster_keywords(cluster_results)

            cluster = ClusterGroup(
                cluster_id=int(cluster_id),
                label=label,
                results=cluster_results,
                centroid=centroid,
                confidence=confidence,
                size=len(cluster_results),
                avg_score=avg_score,
                coherence_score=coherence_score,
                keywords=keywords,
                metadata={
                    "embedding_std": np.std(cluster_embeddings, axis=0).tolist()[
                        :5
                    ],  # First 5 dims
                    "score_std": np.std([r.score for r in cluster_results]),
                },
            )

            clusters.append(cluster)

        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)

        return clusters

    def _identify_outliers(
        self,
        results: list[SearchResult],
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        request: ResultClusteringRequest,
    ) -> list[OutlierResult]:
        """Identify outlier results that don't fit into clusters."""
        outliers = []

        # Find results labeled as noise (-1)
        noise_mask = cluster_labels == -1
        noise_indices = np.where(noise_mask)[0]

        for idx in noise_indices:
            result = results[idx]
            embedding = embeddings[idx]

            # Calculate distance to nearest cluster
            min_distance = float("inf")
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)

            for cluster_id in unique_labels:
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]

                if len(cluster_embeddings) > 0:
                    # Calculate distance to cluster centroid
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distance = np.linalg.norm(embedding - centroid)
                    min_distance = min(min_distance, distance)

            if min_distance == float("inf"):
                min_distance = 1.0

            # Calculate outlier score
            outlier_score = min(1.0, min_distance / 2.0)  # Normalize

            outlier = OutlierResult(
                result=result,
                distance_to_nearest_cluster=min_distance,
                outlier_score=outlier_score,
            )

            outliers.append(outlier)

        return outliers

    def _calculate_coherence(self, cluster_embeddings: np.ndarray) -> float:
        """Calculate intra-cluster coherence score."""
        if len(cluster_embeddings) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(cluster_embeddings)):
            for j in range(i + 1, len(cluster_embeddings)):
                similarity = np.dot(cluster_embeddings[i], cluster_embeddings[j])
                similarities.append(similarity)

        # Ensure coherence score is bounded between 0 and 1
        coherence = float(np.mean(similarities)) if similarities else 0.0
        return max(0.0, min(1.0, coherence))

    def _calculate_cluster_confidence(
        self,
        cluster_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        request: ResultClusteringRequest,
    ) -> float:
        """Calculate confidence score for a cluster."""
        if len(cluster_embeddings) < 2:
            return 0.0

        # Base confidence on cluster cohesion
        coherence = self._calculate_coherence(cluster_embeddings)

        # Adjust based on cluster size
        size_factor = min(1.0, len(cluster_embeddings) / (request.min_cluster_size * 2))

        # Combine factors
        confidence = coherence * 0.7 + size_factor * 0.3

        return max(0.0, min(1.0, confidence))

    def _calculate_quality_metrics(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        request: ResultClusteringRequest,
    ) -> dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}

        if not self.enable_advanced_metrics:
            return metrics

        try:
            from sklearn.metrics import calinski_harabasz_score
            from sklearn.metrics import davies_bouldin_score
            from sklearn.metrics import silhouette_score

            # Filter out noise points for metrics calculation
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 1 and len(set(cluster_labels[valid_mask])) > 1:
                valid_embeddings = embeddings[valid_mask]
                valid_labels = cluster_labels[valid_mask]

                # Silhouette score
                metrics["silhouette_score"] = silhouette_score(
                    valid_embeddings, valid_labels, metric="cosine"
                )

                # Calinski-Harabasz index
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                    valid_embeddings, valid_labels
                )

                # Davies-Bouldin index
                metrics["davies_bouldin_score"] = davies_bouldin_score(
                    valid_embeddings, valid_labels
                )

        except Exception as e:
            self._logger.warning(f"Failed to calculate quality metrics: {e}")

        return metrics

    def _generate_cluster_label(
        self, results: list[SearchResult], query: str | None = None
    ) -> str:
        """Generate human-readable label for a cluster."""
        # Simple implementation - extract common terms
        all_titles = " ".join(
            r.title.lower() for r in results[:5]
        )  # Use first 5 for efficiency

        # Extract most common meaningful words
        import re

        words = re.findall(r"\b\w{3,}\b", all_titles)

        # Remove common stop words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "had",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
        }

        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]

        if filtered_words:
            # Count word frequencies
            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Get most common words
            common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_words = [word for word, count in common_words[:3] if count > 1]

            if top_words:
                return " & ".join(top_words).title()

        # Fallback to generic labels
        return f"Cluster {hash(str([r.id for r in results[:3]])) % 1000}"

    def _extract_cluster_keywords(self, results: list[SearchResult]) -> list[str]:
        """Extract representative keywords for a cluster."""
        # Combine titles and content snippets
        text_content = ""
        for result in results[:10]:  # Limit for efficiency
            text_content += f"{result.title} {result.content} "

        # Extract keywords using simple frequency analysis
        import re

        words = re.findall(r"\b\w{4,}\b", text_content.lower())

        # Remove common words
        stop_words = {
            "that",
            "with",
            "have",
            "this",
            "will",
            "your",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
            "come",
            "here",
            "just",
            "like",
            "long",
            "make",
            "many",
            "over",
            "such",
            "take",
            "than",
            "them",
            "well",
            "were",
        }

        filtered_words = [w for w in words if w not in stop_words]

        # Count frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Get top keywords
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:8] if count >= 2]

    def _get_algorithm_parameters(
        self, method: ClusteringMethod, request: ResultClusteringRequest
    ) -> dict[str, Any]:
        """Get algorithm-specific parameters used."""
        params = {
            "method": method.value,
            "min_cluster_size": request.min_cluster_size,
            "similarity_metric": request.similarity_metric.value,
            "scope": request.scope.value,
        }

        if request.max_clusters:
            params["max_clusters"] = request.max_clusters
        if request.min_samples:
            params["min_samples"] = request.min_samples
        if request.eps:
            params["eps"] = request.eps

        return params

    def _get_cached_result(
        self, request: ResultClusteringRequest
    ) -> ResultClusteringResult | None:
        """Get cached clustering result."""
        cache_key = self._generate_cache_key(request)

        if cache_key in self.clustering_cache:
            self.cache_stats["hits"] += 1
            return self.clustering_cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    def _cache_result(
        self, request: ResultClusteringRequest, result: ResultClusteringResult
    ) -> None:
        """Cache clustering result."""
        if len(self.clustering_cache) >= self.cache_size:
            # Simple LRU eviction
            oldest_key = next(iter(self.clustering_cache))
            del self.clustering_cache[oldest_key]

        cache_key = self._generate_cache_key(request)
        self.clustering_cache[cache_key] = result

    def _generate_cache_key(self, request: ResultClusteringRequest) -> str:
        """Generate cache key for request."""
        # Simple cache key based on result IDs and parameters
        result_ids = sorted([r.id for r in request.results])
        key_components = [
            str(hash(tuple(result_ids))),
            request.method.value,
            str(request.min_cluster_size),
            str(request.max_clusters),
            request.similarity_metric.value,
        ]
        return "|".join(key_components)

    def _update_performance_stats(
        self, method: ClusteringMethod, processing_time: float
    ) -> None:
        """Update performance statistics."""
        self.performance_stats["total_clusterings"] += 1

        # Update average processing time
        total = self.performance_stats["total_clusterings"]
        current_avg = self.performance_stats["avg_processing_time"]
        self.performance_stats["avg_processing_time"] = (
            current_avg * (total - 1) + processing_time
        ) / total

        # Update method usage
        method_key = method.value
        if method_key not in self.performance_stats["method_usage"]:
            self.performance_stats["method_usage"][method_key] = 0
        self.performance_stats["method_usage"][method_key] += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            "cache_stats": self.cache_stats,
            "cache_size": len(self.clustering_cache),
            "available_algorithms": self.available_algorithms,
        }

    def clear_cache(self) -> None:
        """Clear clustering cache."""
        self.clustering_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
