"""Vector Embeddings Visualization for Semantic Similarity Spaces.

This module provides interactive visualization capabilities for vector embeddings,
semantic similarity analysis, and embedding quality assessment. Portfolio feature
showcasing deep ML understanding and data visualization expertise.
"""

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.services.base import BaseService


# Initialize numpy random generator
rng = np.random.default_rng()


logger = logging.getLogger(__name__)


class VectorPoint(BaseModel):
    """Represents a vector point in visualization space."""

    id: str = Field(..., description="Unique identifier for the point")
    x: float = Field(..., description="X coordinate in visualization space")
    y: float = Field(..., description="Y coordinate in visualization space")
    z: float | None = Field(None, description="Z coordinate for 3D visualization")
    text: str = Field(..., description="Original text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    cluster_id: int | None = Field(None, description="Cluster assignment")
    similarity_score: float | None = Field(
        None, description="Similarity score to query"
    )


class ClusterInfo(BaseModel):
    """Information about a vector cluster."""

    cluster_id: int = Field(..., description="Cluster identifier")
    centroid: tuple[float, float] = Field(
        ..., description="Cluster centroid coordinates"
    )
    size: int = Field(..., description="Number of points in cluster")
    label: str = Field(..., description="Human-readable cluster label")
    coherence_score: float = Field(..., description="Cluster coherence score")
    sample_texts: list[str] = Field(..., description="Sample texts from cluster")


class SimilarityRelation(BaseModel):
    """Represents a similarity relationship between vectors."""

    source_id: str = Field(..., description="Source vector ID")
    target_id: str = Field(..., description="Target vector ID")
    similarity_score: float = Field(..., description="Cosine similarity score")
    relation_type: str = Field(..., description="Type of relationship")


class EmbeddingQualityMetrics(BaseModel):
    """Quality metrics for embedding analysis."""

    dimensionality: int = Field(..., description="Embedding dimension")
    variance_explained: float = Field(
        ..., description="Variance explained by top components"
    )
    cluster_separation: float = Field(..., description="Average cluster separation")
    embedding_density: float = Field(..., description="Embedding space density")
    coherence_score: float = Field(..., description="Overall semantic coherence")
    quality_grade: str = Field(..., description="Overall quality grade A-F")


class VectorVisualizationEngine(BaseService):
    """Interactive vector embeddings visualization engine.

    This portfolio feature demonstrates:
    - Advanced machine learning visualization techniques
    - Dimensionality reduction (PCA, t-SNE) implementation
    - Semantic similarity analysis and clustering
    - Interactive data visualization preparation
    - Deep understanding of vector databases and embeddings
    - ML model quality assessment and interpretation
    """

    def __init__(self):
        """Initialize the vector visualization engine."""
        super().__init__()
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Visualization cache
        self.visualization_cache: dict[str, dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour cache

        # Analysis parameters
        self.max_points = 1000  # Maximum points to visualize
        self.min_clusters = 3
        self.max_clusters = 10
        self.similarity_threshold = 0.7

    async def initialize(self) -> None:
        """Initialize the visualization engine."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self._logger.info("VectorVisualizationEngine initialized successfully")

        except (AttributeError, ImportError, OSError):
            self._logger.exception("Failed to initialize VectorVisualizationEngine")
            raise

    async def create_embedding_visualization(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        method: str = "tsne",
        dimensions: int = 2,
    ) -> dict[str, Any]:
        """Create interactive visualization of embeddings.

        Args:
            embeddings: List of embedding vectors
            texts: Corresponding text content
            metadata: Optional metadata for each embedding
            query_embedding: Optional query embedding for similarity analysis
            query_text: Optional query text
            method: Dimensionality reduction method ('pca', 'tsne')
            dimensions: Output dimensions (2 or 3)

        Returns:
            Visualization data including points, clusters, and metrics

        """
        try:
            if not embeddings or not texts:
                return {"error": "Embeddings and texts are required"}

            if len(embeddings) != len(texts):
                return {"error": "Embeddings and texts must have same length"}

            # Convert to numpy array
            embeddings_array = np.array(embeddings)

            # Limit number of points for performance
            if len(embeddings) > self.max_points:
                indices = rng.choice(len(embeddings), self.max_points, replace=False)
                embeddings_array = embeddings_array[indices]
                texts = [texts[i] for i in indices]
                if metadata:
                    metadata = [metadata[i] for i in indices]

            # Dimensionality reduction
            if method.lower() == "pca":
                reducer = PCA(n_components=dimensions, random_state=42)
            elif method.lower() == "tsne":
                # Use PCA first if high dimensional
                if embeddings_array.shape[1] > 50:
                    # Limit n_components to min(50, n_samples-1, n_features)
                    n_components = min(
                        50, embeddings_array.shape[0] - 1, embeddings_array.shape[1]
                    )
                    if n_components > 0:
                        pca_pre = PCA(n_components=n_components, random_state=42)
                        embeddings_array = pca_pre.fit_transform(embeddings_array)

                reducer = TSNE(
                    n_components=dimensions,
                    random_state=42,
                    perplexity=min(30, len(embeddings) - 1),
                )
            else:
                return {"error": f"Unsupported method: {method}"}

            # Apply dimensionality reduction
            reduced_embeddings = reducer.fit_transform(embeddings_array)

            # Perform clustering
            clusters_info = await self._perform_clustering(
                embeddings_array, reduced_embeddings, texts
            )

            # Calculate similarities to query if provided
            similarities = None
            if query_embedding:
                query_array = np.array([query_embedding])
                similarities = cosine_similarity(
                    embeddings_array, query_array
                ).flatten()

            # Create vector points
            points = []
            for i, (coords, text) in enumerate(
                zip(reduced_embeddings, texts, strict=False)
            ):
                point_metadata = metadata[i] if metadata else {}

                point = VectorPoint(
                    id=f"point_{i}",
                    x=float(coords[0]),
                    y=float(coords[1]),
                    z=float(coords[2]) if dimensions == 3 else None,
                    text=text,
                    metadata=point_metadata,
                    cluster_id=clusters_info["assignments"][i]
                    if clusters_info
                    else None,
                    similarity_score=float(similarities[i])
                    if similarities is not None
                    else None,
                )
                points.append(point)

            # Find similarity relationships
            relationships = await self._find_similarity_relationships(
                embeddings_array, points
            )

            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                embeddings_array, reduced_embeddings, clusters_info
            )

            # Generate insights
            insights = await self._generate_visualization_insights(
                points, clusters_info, quality_metrics, similarities
            )

            return {
                "points": [point.model_dump() for point in points],
                "clusters": clusters_info["clusters"] if clusters_info else [],
                "relationships": [rel.model_dump() for rel in relationships],
                "quality_metrics": quality_metrics.model_dump(),
                "insights": insights,
                "method": method,
                "dimensions": dimensions,
                "query_point": {
                    "text": query_text,
                    "similarities_range": {
                        "min": float(np.min(similarities))
                        if similarities is not None
                        else None,
                        "max": float(np.max(similarities))
                        if similarities is not None
                        else None,
                        "mean": float(np.mean(similarities))
                        if similarities is not None
                        else None,
                    },
                }
                if query_embedding
                else None,
                "metadata": {
                    "total_points": len(points),
                    "original_dimensions": len(embeddings[0]),
                    "reduction_method": method,
                    "variance_preserved": getattr(
                        reducer, "explained_variance_ratio_", []
                    ).sum()
                    if hasattr(reducer, "explained_variance_ratio_")
                    else None,
                },
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to create embedding visualization")
            return {"error": "Failed to create visualization"}

    async def analyze_embedding_space(
        self, embeddings: list[list[float]], texts: list[str], sample_size: int = 500
    ) -> dict[str, Any]:
        """Analyze the overall structure and quality of embedding space.

        Args:
            embeddings: List of embedding vectors
            texts: Corresponding text content
            sample_size: Number of embeddings to sample for analysis

        Returns:
            Comprehensive analysis of embedding space

        """
        try:
            if not embeddings or not texts:
                return {"error": "Embeddings and texts are required"}

            # Sample if too many embeddings
            if len(embeddings) > sample_size:
                indices = rng.choice(len(embeddings), sample_size, replace=False)
                embeddings = [embeddings[i] for i in indices]
                texts = [texts[i] for i in indices]

            embeddings_array = np.array(embeddings)

            # Calculate distance metrics
            distances = await self._calculate_distance_metrics(embeddings_array)

            # Analyze dimensionality
            dimensionality_analysis = await self._analyze_dimensionality(
                embeddings_array
            )

            # Semantic coherence analysis
            coherence_analysis = await self._analyze_semantic_coherence(
                embeddings_array, texts
            )

            # Outlier detection
            outliers = await self._detect_outliers(embeddings_array, texts)

            # Density analysis
            density_analysis = await self._analyze_embedding_density(embeddings_array)

            return {
                "space_overview": {
                    "num_embeddings": len(embeddings),
                    "embedding_dimension": len(embeddings[0]),
                    "space_volume": float(np.prod(np.ptp(embeddings_array, axis=0))),
                    "avg_norm": float(
                        np.mean(np.linalg.norm(embeddings_array, axis=1))
                    ),
                },
                "distance_metrics": distances,
                "dimensionality": dimensionality_analysis,
                "coherence": coherence_analysis,
                "outliers": outliers,
                "density": density_analysis,
                "recommendations": await self._generate_space_recommendations(
                    embeddings_array, dimensionality_analysis, coherence_analysis
                ),
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to analyze embedding space")
            return {"error": "Failed to analyze embedding space"}

    async def compare_query_embeddings(
        self,
        query_embeddings: list[list[float]],
        query_texts: list[str],
        reference_embeddings: list[list[float]],
        reference_texts: list[str],
    ) -> dict[str, Any]:
        """Compare query embeddings against reference embeddings.

        Args:
            query_embeddings: Query embedding vectors
            query_texts: Query texts
            reference_embeddings: Reference embedding vectors
            reference_texts: Reference texts

        Returns:
            Comparison analysis with similarity patterns

        """
        try:
            query_array = np.array(query_embeddings)
            reference_array = np.array(reference_embeddings)

            # Calculate all pairwise similarities
            similarities = cosine_similarity(query_array, reference_array)

            # Find best matches for each query
            comparisons = []
            for i, (query_text, query_similarities) in enumerate(
                zip(query_texts, similarities, strict=False)
            ):
                # Get top matches
                top_indices = np.argsort(query_similarities)[-5:][::-1]
                top_matches = [
                    {
                        "text": reference_texts[idx],
                        "similarity": float(query_similarities[idx]),
                        "rank": rank + 1,
                    }
                    for rank, idx in enumerate(top_indices)
                ]

                # Calculate query quality metrics
                max_sim = float(np.max(query_similarities))
                avg_sim = float(np.mean(query_similarities))
                std_sim = float(np.std(query_similarities))

                comparisons.append(
                    {
                        "query": query_text,
                        "query_id": f"query_{i}",
                        "top_matches": top_matches,
                        "similarity_stats": {
                            "max_similarity": max_sim,
                            "avg_similarity": avg_sim,
                            "std_similarity": std_sim,
                            "selectivity": max_sim
                            - avg_sim,  # How selective the query is
                        },
                    }
                )

            # Overall comparison metrics
            all_similarities = similarities.flatten()
            overall_stats = {
                "total_comparisons": similarities.size,
                "avg_similarity": float(np.mean(all_similarities)),
                "max_similarity": float(np.max(all_similarities)),
                "min_similarity": float(np.min(all_similarities)),
                "similarity_distribution": {
                    "high_similarity": float(
                        np.sum(all_similarities > 0.8) / len(all_similarities)
                    ),
                    "medium_similarity": float(
                        np.sum((all_similarities > 0.5) & (all_similarities <= 0.8))
                        / len(all_similarities)
                    ),
                    "low_similarity": float(
                        np.sum(all_similarities <= 0.5) / len(all_similarities)
                    ),
                },
            }

            # Query quality assessment
            query_qualities = []
            for comp in comparisons:
                stats = comp["similarity_stats"]

                # Simple quality score based on selectivity and max similarity
                quality_score = (stats["max_similarity"] * 0.7) + (
                    stats["selectivity"] * 0.3
                )

                if quality_score > 0.8:
                    quality = "excellent"
                elif quality_score > 0.6:
                    quality = "good"
                elif quality_score > 0.4:
                    quality = "fair"
                else:
                    quality = "poor"

                query_qualities.append(
                    {
                        "query": comp["query"],
                        "quality_score": quality_score,
                        "quality_grade": quality,
                        "selectivity": stats["selectivity"],
                    }
                )

            return {
                "comparisons": comparisons,
                "overall_stats": overall_stats,
                "query_qualities": query_qualities,
                "recommendations": await self._generate_query_recommendations(
                    query_qualities, overall_stats
                ),
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to compare query embeddings")
            return {"error": "Failed to compare embeddings"}

    async def _perform_clustering(
        self,
        embeddings_array: np.ndarray,
        reduced_embeddings: np.ndarray,
        texts: list[str],
    ) -> dict[str, Any] | None:
        """Perform clustering on embeddings."""
        try:
            if len(embeddings_array) < self.min_clusters:
                return None

            # Determine optimal number of clusters using elbow method
            max_k = min(self.max_clusters, len(embeddings_array) // 2)
            if max_k < self.min_clusters:
                return None

            inertias = []
            k_range = range(self.min_clusters, max_k + 1)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(embeddings_array)
                inertias.append(kmeans.inertia_)

            # Simple elbow detection
            optimal_k = self._find_elbow_point(list(k_range), inertias)

            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(embeddings_array)

            # Create cluster info
            clusters = []
            for cluster_id in range(optimal_k):
                cluster_indices = np.where(cluster_assignments == cluster_id)[0]
                cluster_texts = [texts[i] for i in cluster_indices]

                # Calculate centroid in reduced space
                cluster_reduced_points = reduced_embeddings[cluster_indices]
                centroid = np.mean(cluster_reduced_points, axis=0)

                # Calculate coherence (average intra-cluster similarity)
                cluster_embeddings = embeddings_array[cluster_indices]
                if len(cluster_embeddings) > 1:
                    similarities = cosine_similarity(cluster_embeddings)
                    coherence = float(
                        np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                    )
                else:
                    coherence = 1.0

                # Generate cluster label (most common words)
                cluster_label = self._generate_cluster_label(cluster_texts)

                cluster = ClusterInfo(
                    cluster_id=cluster_id,
                    centroid=(float(centroid[0]), float(centroid[1])),
                    size=len(cluster_indices),
                    label=cluster_label,
                    coherence_score=coherence,
                    sample_texts=cluster_texts[:3],  # First 3 texts as samples
                )
                clusters.append(cluster)

            return {
                "clusters": [cluster.model_dump() for cluster in clusters],
                "assignments": cluster_assignments.tolist(),
                "optimal_k": optimal_k,
            }

        except (ImportError, OSError, PermissionError):
            self._logger.exception("Failed to perform clustering")
            return None

    def _find_elbow_point(self, k_values: list[int], inertias: list[float]) -> int:
        """Find the elbow point in the inertia curve."""
        if len(k_values) < 3:
            return k_values[0]

        # Calculate differences
        diffs = []
        for i in range(1, len(inertias)):
            diff = inertias[i - 1] - inertias[i]
            diffs.append(diff)

        # Find the point where the difference starts decreasing significantly
        max_diff_idx = np.argmax(diffs)
        return k_values[max_diff_idx + 1]  # +1 because diffs is one element shorter

    def _generate_cluster_label(self, texts: list[str]) -> str:
        """Generate a descriptive label for a cluster based on common words."""
        if not texts:
            return "Empty Cluster"

        # Simple word frequency approach
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return f"Cluster ({len(texts)} items)"

        # Get top 2 most common words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        label_words = [word for word, count in top_words]

        return " + ".join(label_words).title()

    async def _find_similarity_relationships(
        self, embeddings_array: np.ndarray, points: list[VectorPoint]
    ) -> list[SimilarityRelation]:
        """Find high-similarity relationships between points."""
        try:
            relationships = []

            # Calculate similarity matrix (sample for performance)
            if len(embeddings_array) > 100:
                # Sample pairs to avoid O(n²) complexity
                num_samples = 200
                indices = rng.choice(
                    len(embeddings_array),
                    min(num_samples, len(embeddings_array)),
                    replace=False,
                )
                sample_embeddings = embeddings_array[indices]
                sample_points = [points[i] for i in indices]
            else:
                sample_embeddings = embeddings_array
                sample_points = points

            similarities = cosine_similarity(sample_embeddings)

            # Find high similarity pairs
            for i in range(len(sample_embeddings)):
                for j in range(i + 1, len(sample_embeddings)):
                    similarity = similarities[i, j]

                    if similarity > self.similarity_threshold:
                        relation = SimilarityRelation(
                            source_id=sample_points[i].id,
                            target_id=sample_points[j].id,
                            similarity_score=float(similarity),
                            relation_type="high_similarity",
                        )
                        relationships.append(relation)

            # Sort by similarity score
            relationships.sort(key=lambda x: x.similarity_score, reverse=True)

            # Return top 50 relationships
            return relationships[:50]

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to find similarity relationships")
            return []

    async def _calculate_quality_metrics(
        self,
        embeddings_array: np.ndarray,
        _reduced_embeddings: np.ndarray,
        clusters_info: dict[str, Any] | None,
    ) -> EmbeddingQualityMetrics:
        """Calculate embedding quality metrics."""
        try:
            # Dimensionality metrics
            pca = PCA(
                n_components=min(
                    50, embeddings_array.shape[1], embeddings_array.shape[0]
                )
            )
            pca.fit(embeddings_array)
            variance_explained = float(
                np.sum(pca.explained_variance_ratio_[:10])
            )  # Top 10 components

            # Cluster separation (if clusters available)
            cluster_separation = 0.5  # Default
            if clusters_info and len(clusters_info["clusters"]) > 1:
                cluster_centers = [
                    cluster["centroid"] for cluster in clusters_info["clusters"]
                ]

                if len(cluster_centers) > 1:
                    center_distances = []
                    for i in range(len(cluster_centers)):
                        for j in range(i + 1, len(cluster_centers)):
                            dist = np.linalg.norm(
                                np.array(cluster_centers[i])
                                - np.array(cluster_centers[j])
                            )
                            center_distances.append(dist)

                    cluster_separation = float(np.mean(center_distances))

            # Embedding density (average pairwise distance)
            if len(embeddings_array) <= 100:
                pairwise_distances = []
                for i in range(len(embeddings_array)):
                    for j in range(i + 1, len(embeddings_array)):
                        dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
                        pairwise_distances.append(dist)

                embedding_density = (
                    float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
                )
            else:
                # Sample for large datasets
                sample_size = 50
                indices = rng.choice(len(embeddings_array), sample_size, replace=False)
                sample_embeddings = embeddings_array[indices]

                pairwise_distances = []
                for i in range(len(sample_embeddings)):
                    for j in range(i + 1, len(sample_embeddings)):
                        dist = np.linalg.norm(
                            sample_embeddings[i] - sample_embeddings[j]
                        )
                        pairwise_distances.append(dist)

                embedding_density = (
                    float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
                )

            # Overall coherence score (normalized combination of metrics)
            coherence_components = [
                variance_explained,
                min(cluster_separation / 10.0, 1.0),  # Normalize cluster separation
                max(0.0, 1.0 - embedding_density / 10.0),  # Inverse of density
            ]
            coherence_score = float(np.mean(coherence_components))

            # Quality grade
            if coherence_score >= 0.8:
                quality_grade = "A"
            elif coherence_score >= 0.7:
                quality_grade = "B"
            elif coherence_score >= 0.6:
                quality_grade = "C"
            elif coherence_score >= 0.5:
                quality_grade = "D"
            else:
                quality_grade = "F"

            return EmbeddingQualityMetrics(
                dimensionality=embeddings_array.shape[1],
                variance_explained=variance_explained,
                cluster_separation=cluster_separation,
                embedding_density=embedding_density,
                coherence_score=coherence_score,
                quality_grade=quality_grade,
            )

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to calculate quality metrics")
            return EmbeddingQualityMetrics(
                dimensionality=embeddings_array.shape[1],
                variance_explained=0.0,
                cluster_separation=0.0,
                embedding_density=0.0,
                coherence_score=0.0,
                quality_grade="F",
            )

    async def _generate_visualization_insights(
        self,
        points: list[VectorPoint],
        clusters_info: dict[str, Any] | None,
        quality_metrics: EmbeddingQualityMetrics,
        similarities: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        """Generate insights about the visualization."""
        try:
            insights = []

            # Quality insights
            if quality_metrics.quality_grade in ["A", "B"]:
                insights.append(
                    {
                        "type": "quality",
                        "title": "High Quality Embeddings",
                        "description": f"Embeddings show {quality_metrics.quality_grade}-grade quality with {quality_metrics.coherence_score:.2f} coherence score",
                        "positive": True,
                    }
                )
            elif quality_metrics.quality_grade in ["D", "F"]:
                insights.append(
                    {
                        "type": "quality",
                        "title": "Embedding Quality Concerns",
                        "description": f"Embeddings show {quality_metrics.quality_grade}-grade quality, consider retraining or preprocessing",
                        "positive": False,
                    }
                )

            # Clustering insights
            if clusters_info:
                num_clusters = len(clusters_info["clusters"])
                avg_cluster_size = len(points) / num_clusters

                if avg_cluster_size > 50:
                    insights.append(
                        {
                            "type": "clustering",
                            "title": "Large Clusters Detected",
                            "description": f"Average cluster size is {avg_cluster_size:.0f}, consider increasing cluster granularity",
                            "positive": False,
                        }
                    )
                elif avg_cluster_size < 5:
                    insights.append(
                        {
                            "type": "clustering",
                            "title": "Small Clusters Detected",
                            "description": f"Average cluster size is {avg_cluster_size:.1f}, embeddings might be too diverse",
                            "positive": False,
                        }
                    )
                else:
                    insights.append(
                        {
                            "type": "clustering",
                            "title": "Well-Balanced Clusters",
                            "description": f"Found {num_clusters} well-balanced clusters with good separation",
                            "positive": True,
                        }
                    )

            # Similarity insights
            if similarities is not None:
                high_sim_count = np.sum(similarities > 0.8)
                total_count = len(similarities)

                if high_sim_count / total_count > 0.3:
                    insights.append(
                        {
                            "type": "similarity",
                            "title": "High Query Relevance",
                            "description": f"{high_sim_count}/{total_count} embeddings show high similarity to query",
                            "positive": True,
                        }
                    )
                elif high_sim_count / total_count < 0.1:
                    insights.append(
                        {
                            "type": "similarity",
                            "title": "Low Query Relevance",
                            "description": f"Only {high_sim_count}/{total_count} embeddings are highly relevant to query",
                            "positive": False,
                        }
                    )

            # Dimensionality insights
            if quality_metrics.variance_explained > 0.8:
                insights.append(
                    {
                        "type": "dimensionality",
                        "title": "Efficient Dimensionality",
                        "description": f"Top components explain {quality_metrics.variance_explained:.1%} of variance",
                        "positive": True,
                    }
                )
            elif quality_metrics.variance_explained < 0.5:
                insights.append(
                    {
                        "type": "dimensionality",
                        "title": "High Dimensionality Detected",
                        "description": f"Low variance explanation ({quality_metrics.variance_explained:.1%}) suggests high intrinsic dimensionality",
                        "positive": False,
                    }
                )

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to generate visualization insights")
            return []

        return insights

    async def _calculate_distance_metrics(
        self, embeddings_array: np.ndarray
    ) -> dict[str, float]:
        """Calculate distance-based metrics for embedding space."""
        try:
            # Sample for large datasets
            if len(embeddings_array) > 200:
                indices = rng.choice(len(embeddings_array), 200, replace=False)
                sample_embeddings = embeddings_array[indices]
            else:
                sample_embeddings = embeddings_array

            # Calculate pairwise distances
            euclidean_distances = pdist(sample_embeddings, metric="euclidean")
            cosine_distances = pdist(sample_embeddings, metric="cosine")

            return {
                "avg_euclidean_distance": float(np.mean(euclidean_distances)),
                "std_euclidean_distance": float(np.std(euclidean_distances)),
                "avg_cosine_distance": float(np.mean(cosine_distances)),
                "std_cosine_distance": float(np.std(cosine_distances)),
                "max_euclidean_distance": float(np.max(euclidean_distances)),
                "min_euclidean_distance": float(np.min(euclidean_distances)),
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to calculate distance metrics")
            return {}

    async def _analyze_dimensionality(
        self, embeddings_array: np.ndarray
    ) -> dict[str, Any]:
        """Analyze the intrinsic dimensionality of embeddings."""
        try:
            # PCA analysis
            n_components = min(50, embeddings_array.shape[1], embeddings_array.shape[0])
            pca = PCA(n_components=n_components)
            pca.fit(embeddings_array)

            # Find effective dimensionality (95% variance)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = int(np.argmax(cumsum_variance >= 0.95) + 1)

            return {
                "original_dimensions": embeddings_array.shape[1],
                "effective_dimensions": effective_dim,
                "dimension_efficiency": effective_dim / embeddings_array.shape[1],
                "variance_explained_by_top_10": float(
                    np.sum(pca.explained_variance_ratio_[:10])
                ),
                "singular_values_ratio": float(
                    pca.singular_values_[0] / pca.singular_values_[-1]
                )
                if len(pca.singular_values_) > 1
                else 1.0,
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to analyze dimensionality")
            return {}

    async def _analyze_semantic_coherence(
        self, embeddings_array: np.ndarray, texts: list[str]
    ) -> dict[str, Any]:
        """Analyze semantic coherence of embeddings."""
        try:
            # Sample for performance
            if len(embeddings_array) > 100:
                indices = rng.choice(len(embeddings_array), 100, replace=False)
                sample_embeddings = embeddings_array[indices]
                sample_texts = [texts[i] for i in indices]
            else:
                sample_embeddings = embeddings_array
                sample_texts = texts

            # Calculate semantic coherence based on text similarity vs embedding similarity
            coherence_scores = []

            for i in range(min(20, len(sample_texts))):  # Sample pairs
                for j in range(i + 1, min(i + 11, len(sample_texts))):  # Check next 10
                    # Simple text similarity (word overlap)
                    words_i = set(sample_texts[i].lower().split())
                    words_j = set(sample_texts[j].lower().split())
                    text_similarity = (
                        len(words_i & words_j) / len(words_i | words_j)
                        if words_i | words_j
                        else 0
                    )

                    # Embedding similarity
                    emb_similarity = cosine_similarity(
                        sample_embeddings[i : i + 1], sample_embeddings[j : j + 1]
                    )[0, 0]

                    # Coherence is correlation between text and embedding similarity
                    coherence_scores.append((text_similarity, emb_similarity))

            if coherence_scores:
                text_sims, emb_sims = zip(*coherence_scores, strict=False)
                correlation = (
                    np.corrcoef(text_sims, emb_sims)[0, 1]
                    if len(coherence_scores) > 1
                    else 0
                )
            else:
                correlation = 0

            return {
                "semantic_correlation": float(correlation)
                if not np.isnan(correlation)
                else 0.0,
                "avg_embedding_similarity": float(np.mean(emb_sims))
                if coherence_scores
                else 0.0,
                "coherence_samples": len(coherence_scores),
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to analyze semantic coherence")
            return {}

    async def _detect_outliers(
        self, embeddings_array: np.ndarray, texts: list[str]
    ) -> dict[str, Any]:
        """Detect outlier embeddings."""
        try:
            # Calculate distances from centroid
            centroid = np.mean(embeddings_array, axis=0)
            distances = np.linalg.norm(embeddings_array - centroid, axis=1)

            # Find outliers (beyond 2 standard deviations)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            outlier_threshold = mean_dist + 2 * std_dist

            outlier_indices = np.where(distances > outlier_threshold)[0]

            outliers_info = [
                {
                    "text": texts[idx],
                    "distance_from_center": float(distances[idx]),
                    "z_score": float((distances[idx] - mean_dist) / std_dist),
                }
                for idx in outlier_indices[:10]  # Top 10 outliers
            ]

            return {
                "num_outliers": len(outlier_indices),
                "outlier_percentage": len(outlier_indices) / len(embeddings_array),
                "outlier_threshold": float(outlier_threshold),
                "sample_outliers": outliers_info,
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to detect outliers")
            return {}

    async def _analyze_embedding_density(
        self, embeddings_array: np.ndarray
    ) -> dict[str, Any]:
        """Analyze the density distribution of embeddings."""
        try:
            # Calculate local density using k-nearest neighbors
            k = min(10, len(embeddings_array) - 1)
            if k <= 0:
                return {}

            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings_array)
            distances, _ = nbrs.kneighbors(embeddings_array)

            # Average distance to k-th nearest neighbor (excluding self)
            kth_distances = distances[:, -1]  # Distance to k-th neighbor

            return {
                "avg_local_density": float(np.mean(1 / (kth_distances + 1e-8))),
                "density_variance": float(np.var(1 / (kth_distances + 1e-8))),
                "avg_knn_distance": float(np.mean(kth_distances)),
                "density_distribution": {
                    "p25": float(np.percentile(kth_distances, 25)),
                    "p50": float(np.percentile(kth_distances, 50)),
                    "p75": float(np.percentile(kth_distances, 75)),
                    "p95": float(np.percentile(kth_distances, 95)),
                },
            }

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to analyze embedding density")
            return {}

    async def _generate_space_recommendations(
        self,
        _embeddings_array: np.ndarray,
        dimensionality_analysis: dict[str, Any],
        coherence_analysis: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Generate recommendations for improving embedding space."""
        try:
            recommendations = []

            # Dimensionality recommendations
            if dimensionality_analysis.get("dimension_efficiency", 0) < 0.3:
                recommendations.append(
                    {
                        "type": "dimensionality",
                        "recommendation": "Consider dimensionality reduction or feature selection",
                        "reason": f"Only {dimensionality_analysis.get('dimension_efficiency', 0):.1%} of dimensions are effective",
                    }
                )

            # Coherence recommendations
            if coherence_analysis.get("semantic_correlation", 0) < 0.3:
                recommendations.append(
                    {
                        "type": "coherence",
                        "recommendation": "Improve text preprocessing or embedding model training",
                        "reason": f"Low semantic correlation ({coherence_analysis.get('semantic_correlation', 0):.2f})",
                    }
                )

            # Quality recommendations
            avg_similarity = coherence_analysis.get("avg_embedding_similarity", 0)
            if avg_similarity > 0.9:
                recommendations.append(
                    {
                        "type": "diversity",
                        "recommendation": "Increase data diversity or adjust embedding parameters",
                        "reason": f"Very high average similarity ({avg_similarity:.2f}) suggests lack of diversity",
                    }
                )
            elif avg_similarity < 0.1:
                recommendations.append(
                    {
                        "type": "clustering",
                        "recommendation": "Check for data quality issues or embedding model problems",
                        "reason": f"Very low average similarity ({avg_similarity:.2f}) suggests poor semantic capture",
                    }
                )

        except (AttributeError, OSError, PermissionError):
            self._logger.exception("Failed to generate space recommendations")
            return []

        return recommendations

    async def _generate_query_recommendations(
        self, query_qualities: list[dict[str, Any]], overall_stats: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Generate recommendations for query optimization."""
        try:
            recommendations = []

            # Overall similarity recommendations
            high_sim_rate = overall_stats.get("similarity_distribution", {}).get(
                "high_similarity", 0
            )

            if high_sim_rate < 0.1:
                recommendations.append(
                    {
                        "type": "relevance",
                        "recommendation": "Consider expanding the reference corpus or improving query preprocessing",
                        "reason": f"Only {high_sim_rate:.1%} of comparisons show high similarity",
                    }
                )
            elif high_sim_rate > 0.5:
                recommendations.append(
                    {
                        "type": "specificity",
                        "recommendation": "Consider making queries more specific or expanding reference diversity",
                        "reason": f"High similarity rate ({high_sim_rate:.1%}) may indicate overly broad matching",
                    }
                )

            # Query quality recommendations
            poor_quality_queries = [
                q for q in query_qualities if q["quality_grade"] == "poor"
            ]
            if len(poor_quality_queries) > len(query_qualities) * 0.3:
                recommendations.append(
                    {
                        "type": "query_quality",
                        "recommendation": "Improve query formulation or add context to low-quality queries",
                        "reason": f"{len(poor_quality_queries)} out of {len(query_qualities)} queries show poor quality",
                    }
                )

            # Selectivity recommendations
            low_selectivity_queries = [
                q for q in query_qualities if q["selectivity"] < 0.2
            ]
            if len(low_selectivity_queries) > len(query_qualities) * 0.5:
                recommendations.append(
                    {
                        "type": "selectivity",
                        "recommendation": "Make queries more specific to improve result selectivity",
                        "reason": f"{len(low_selectivity_queries)} queries show low selectivity",
                    }
                )

        except (AttributeError, ConnectionError, OSError):
            self._logger.exception("Failed to generate query recommendations")
            return []

        return recommendations

    async def cleanup(self) -> None:
        """Cleanup visualization engine resources."""
        if self._initialized:
            self.visualization_cache.clear()
            self._initialized = False
            self._logger.info("VectorVisualizationEngine cleaned up")
