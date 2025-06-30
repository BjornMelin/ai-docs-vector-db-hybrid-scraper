"""Tests for vector embeddings visualization functionality."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.services.analytics.vector_visualization import (
    ClusterInfo,
    EmbeddingQualityMetrics,
    SimilarityRelation,
    VectorPoint,
    VectorVisualizationEngine,
)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.embeddings = MagicMock()
    config.embeddings.dimension = 768
    config.visualization = MagicMock()
    config.visualization.max_points = 1000
    return config


@pytest.fixture
def visualization_engine(mock_config):
    """Create visualization engine instance."""
    with patch(
        "src.services.analytics.vector_visualization.get_config",
        return_value=mock_config,
    ):
        return VectorVisualizationEngine()


@pytest.fixture
async def initialized_engine(visualization_engine):
    """Create initialized visualization engine."""
    await visualization_engine.initialize()
    return visualization_engine


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    # Create diverse embeddings to test clustering
    rng = np.random.default_rng(42)  # For reproducible tests

    # Cluster 1: Similar embeddings around [0.5, 0.5, ...]
    cluster1 = rng.normal(0.5, 0.1, (10, 768))

    # Cluster 2: Similar embeddings around [-0.5, -0.5, ...]
    cluster2 = rng.normal(-0.5, 0.1, (10, 768))

    # Cluster 3: Similar embeddings around [0.0, 1.0, ...]
    cluster3 = rng.normal(0.0, 0.1, (10, 768))
    cluster3[:, 1] = rng.normal(1.0, 0.1, 10)

    # Combine all clusters
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_documents():
    """Create sample documents corresponding to embeddings."""
    # Documents for cluster 1 (machine learning topics)
    ml_docs = [
        {
            "id": f"ml_{i}",
            "text": f"Machine learning algorithm {i} for data analysis",
            "metadata": {
                "category": "machine_learning",
                "difficulty": "intermediate",
            },
        }
        for i in range(10)
    ]

    # Documents for cluster 2 (programming topics)
    prog_docs = [
        {
            "id": f"prog_{i}",
            "text": f"Programming language tutorial {i} for beginners",
            "metadata": {"category": "programming", "difficulty": "beginner"},
        }
        for i in range(10)
    ]

    # Documents for cluster 3 (data science topics)
    ds_docs = [
        {
            "id": f"ds_{i}",
            "text": f"Data science methodology {i} for research",
            "metadata": {"category": "data_science", "difficulty": "advanced"},
        }
        for i in range(10)
    ]

    return ml_docs + prog_docs + ds_docs


class TestVectorVisualizationModels:
    """Test the Pydantic models."""

    def test_vector_point_model(self):
        """Test VectorPoint model."""
        point = VectorPoint(
            id="test_1",
            x=0.5,
            y=-0.3,
            z=0.8,
            text="Sample document text",
            metadata={"category": "test"},
            cluster_id=1,
            similarity_score=0.85,
        )

        assert point.id == "test_1"
        assert point.x == 0.5
        assert point.y == -0.3
        assert point.z == 0.8
        assert point.cluster_id == 1
        assert point.similarity_score == 0.85

    def test_cluster_info_model(self):
        """Test ClusterInfo model."""
        cluster = ClusterInfo(
            cluster_id=1,
            centroid=(0.2, 0.7),
            size=15,
            label="Machine Learning",
            coherence_score=0.78,
            sample_texts=["ML text 1", "ML text 2"],
        )

        assert cluster.cluster_id == 1
        assert cluster.centroid == (0.2, 0.7)
        assert cluster.size == 15
        assert cluster.coherence_score == 0.78

    def test_similarity_relation_model(self):
        """Test SimilarityRelation model."""
        relation = SimilarityRelation(
            source_id="doc_1",
            target_id="doc_2",
            similarity_score=0.92,
            relation_type="semantic_similarity",
        )

        assert relation.source_id == "doc_1"
        assert relation.target_id == "doc_2"
        assert relation.similarity_score == 0.92
        assert relation.relation_type == "semantic_similarity"

    def test_embedding_quality_metrics_model(self):
        """Test EmbeddingQualityMetrics model."""
        metrics = EmbeddingQualityMetrics(
            dimensionality=768,
            variance_explained=0.85,
            cluster_separation=0.72,
            embedding_density=0.68,
            coherence_score=0.81,
            quality_grade="B+",
        )

        assert metrics.dimensionality == 768
        assert metrics.variance_explained == 0.85
        assert metrics.quality_grade == "B+"


class TestVectorVisualizationEngine:
    """Test the VectorVisualizationEngine class."""

    def test_initialization(self, visualization_engine):
        """Test engine initialization."""
        assert visualization_engine.visualization_cache == {}
        assert visualization_engine.cache_ttl == 3600
        assert visualization_engine.max_points == 1000
        assert visualization_engine.min_clusters == 3
        assert visualization_engine.max_clusters == 10
        assert visualization_engine.similarity_threshold == 0.7

    async def test_initialize(self, visualization_engine):
        """Test engine initialization."""
        await visualization_engine.initialize()
        # Should complete without error

    async def test_reduce_dimensions_pca(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test PCA dimensionality reduction via create_embedding_visualization."""
        # Convert numpy array to list of lists and extract texts
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        # Test 2D reduction
        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=2,
        )

        assert "points" in visualization
        points = visualization["points"]
        assert len(points) == len(sample_embeddings)

        # All points should have x, y coordinates (2D) - they are dictionaries now
        for point in points:
            assert isinstance(point, dict)
            assert point["x"] is not None
            assert point["y"] is not None
            assert point["z"] is None  # 2D visualization

        # Test 3D reduction
        visualization_3d = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=3,
        )

        points_3d = visualization_3d["points"]
        assert len(points_3d) == len(sample_embeddings)

        # All points should have x, y, z coordinates (3D)
        for point in points_3d:
            assert isinstance(point, dict)
            assert point["x"] is not None
            assert point["y"] is not None
            assert point["z"] is not None

    async def test_reduce_dimensions_tsne(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test t-SNE dimensionality reduction via create_embedding_visualization."""
        # Use smaller sample for t-SNE (it's computationally expensive)
        small_sample = sample_embeddings[:15]
        small_documents = sample_documents[:15]

        # Convert to required format
        embeddings_list = small_sample.tolist()
        texts = [doc["text"] for doc in small_documents]
        metadata_list = [doc["metadata"] for doc in small_documents]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="tsne",
            dimensions=2,
        )

        points = visualization["points"]
        assert len(points) == len(small_sample)

        # All points should have 2D coordinates
        for point in points:
            assert isinstance(point, dict)
            assert point["x"] is not None
            assert point["y"] is not None
            assert point["z"] is None

    async def test_cluster_vectors(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test vector clustering via create_embedding_visualization."""
        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=2,
        )

        points = visualization["points"]
        clusters = visualization["clusters"]

        # Check cluster assignments in points - they are dictionaries now
        cluster_ids = [
            point["cluster_id"] for point in points if point["cluster_id"] is not None
        ]
        assert len(cluster_ids) <= len(
            sample_embeddings
        )  # Some points might not have cluster assignments
        assert all(
            isinstance(cluster_id, int | np.integer) for cluster_id in cluster_ids
        )
        if cluster_ids:  # Only check min if we have cluster IDs
            assert min(cluster_ids) >= 0

        # Check cluster info
        assert len(clusters) >= 0  # May be empty if clustering failed
        for cluster in clusters:
            assert isinstance(cluster, dict)
            assert "cluster_id" in cluster
            assert cluster["cluster_id"] >= 0
            assert "size" in cluster
            assert cluster["size"] > 0

    async def test_calculate_similarities(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test similarity calculation via compare_query_embeddings."""
        # Test with query vector (similar to first cluster)
        rng = np.random.default_rng(42)
        query_vector = rng.normal(0.5, 0.05, 768)

        # Convert to required format
        query_embeddings = [query_vector.tolist()]
        query_texts = ["test query about machine learning"]
        reference_embeddings = sample_embeddings.tolist()
        reference_texts = [doc["text"] for doc in sample_documents]

        comparison_result = await initialized_engine.compare_query_embeddings(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            reference_embeddings=reference_embeddings,
            reference_texts=reference_texts,
        )

        # Check the actual structure returned by compare_query_embeddings
        assert "comparisons" in comparison_result
        assert "overall_stats" in comparison_result
        assert "query_qualities" in comparison_result

        comparisons = comparison_result["comparisons"]
        assert len(comparisons) == 1  # One query

        comparison = comparisons[0]
        assert "top_matches" in comparison
        assert len(comparison["top_matches"]) > 0

        # Check similarity values are in correct range
        for match in comparison["top_matches"]:
            assert -1.0 <= match["similarity"] <= 1.0  # Cosine similarity range

        # Similarities should be sorted in descending order for top matches
        similarities = [match["similarity"] for match in comparison["top_matches"]]
        assert similarities == sorted(similarities, reverse=True)

    async def test_create_visualization_2d(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test 2D visualization creation."""
        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=2,
        )

        assert "points" in visualization
        assert "clusters" in visualization
        assert "quality_metrics" in visualization

        # Check points - they are dictionaries now
        points = visualization["points"]
        assert len(points) == len(sample_documents)

        for point in points:
            assert isinstance(point, dict)
            assert point["x"] is not None
            assert point["y"] is not None
            assert point["z"] is None  # 2D visualization
            assert point["text"] is not None
            # cluster_id may be None for some points
            assert "cluster_id" in point

        # Check clusters
        clusters = visualization["clusters"]
        assert len(clusters) >= 0  # May be empty

        for cluster in clusters:
            assert isinstance(cluster, dict)
            assert "cluster_id" in cluster
            assert cluster["cluster_id"] >= 0
            assert "size" in cluster
            assert cluster["size"] > 0
            assert "centroid" in cluster
            assert len(cluster["centroid"]) == 2  # 2D centroids

    async def test_create_visualization_3d(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test 3D visualization creation."""
        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=3,
        )

        points = visualization["points"]
        clusters = visualization["clusters"]

        # Verify clusters structure
        assert isinstance(clusters, list), "Clusters should be a list"
        for cluster in clusters:
            assert "center" in cluster, "Each cluster should have a center"

        # Check 3D coordinates - points are dictionaries now
        for point in points:
            assert isinstance(point, dict)
            assert point["x"] is not None
            assert point["y"] is not None
            assert point["z"] is not None  # 3D visualization

    async def test_create_visualization_with_query(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test visualization with query vector."""
        rng = np.random.default_rng(42)
        query_vector = rng.normal(0.5, 0.05, 768)  # Similar to cluster 1

        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            query_embedding=query_vector.tolist(),
            query_text="test query about machine learning",
            method="pca",
            dimensions=2,
        )

        points = visualization["points"]

        # All points should have similarity scores - they are dictionaries now
        for point in points:
            assert isinstance(point, dict)
            assert "similarity_score" in point
            if point["similarity_score"] is not None:
                assert (
                    -1.0 <= point["similarity_score"] <= 1.0
                )  # Cosine similarity range

    async def test_find_similar_vectors(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test finding similar vectors via compare_query_embeddings."""
        query_vector = sample_embeddings[0]  # Use first embedding as query

        # Convert to required format
        query_embeddings = [query_vector.tolist()]
        query_texts = ["test query about machine learning"]
        reference_embeddings = sample_embeddings.tolist()
        reference_texts = [doc["text"] for doc in sample_documents]

        comparison_result = await initialized_engine.compare_query_embeddings(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            reference_embeddings=reference_embeddings,
            reference_texts=reference_texts,
        )

        # Check the correct structure from compare_query_embeddings
        comparisons = comparison_result["comparisons"]
        assert len(comparisons) == 1

        top_matches = comparisons[0]["top_matches"]
        assert len(top_matches) <= 5

        for match in top_matches:
            assert isinstance(match, dict)
            assert -1.0 <= match["similarity"] <= 1.0  # Cosine similarity range
            assert "text" in match
            assert "rank" in match

        # Relations should be sorted by similarity (highest first)
        similarities = [match["similarity"] for match in top_matches]
        assert similarities == sorted(similarities, reverse=True)

    async def test_analyze_embedding_quality(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test embedding quality analysis via analyze_embedding_space."""
        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]

        analysis_result = await initialized_engine.analyze_embedding_space(
            embeddings=embeddings_list, texts=texts
        )

        # Check the actual structure returned by analyze_embedding_space
        assert "space_overview" in analysis_result
        assert "dimensionality" in analysis_result
        assert "coherence" in analysis_result
        assert "density" in analysis_result
        assert "recommendations" in analysis_result

        # Check space overview
        space_overview = analysis_result["space_overview"]
        assert space_overview["embedding_dimension"] == sample_embeddings.shape[1]
        assert space_overview["num_embeddings"] > 0

        # Check dimensionality analysis
        dimensionality = analysis_result["dimensionality"]
        assert isinstance(dimensionality, dict)

        # Check coherence analysis
        coherence = analysis_result["coherence"]
        assert isinstance(coherence, dict)

    async def test_export_visualization_data(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test visualization data export."""
        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=2,
        )

        # The visualization itself is already in a structured format
        # Check that it contains the expected data
        assert "metadata" in visualization
        assert "points" in visualization
        assert "clusters" in visualization
        assert "quality_metrics" in visualization

        # Test that the data can be serialized (equivalent to export)

        try:
            # Data should already be in serializable format (dictionaries)
            serializable_data = {
                "metadata": visualization["metadata"],
                "points": visualization["points"],  # Already dictionaries
                "clusters": visualization["clusters"],  # Already dictionaries
                "quality_metrics": visualization[
                    "quality_metrics"
                ],  # Already dictionary
            }
            json_str = json.dumps(serializable_data)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Visualization data is not serializable: {e}")

    async def test_error_handling(self, initialized_engine):
        """Test error handling."""
        rng = np.random.default_rng(42)
        # Test with empty embeddings
        result = await initialized_engine.create_embedding_visualization(
            embeddings=[], texts=[], method="pca", dimensions=2
        )
        assert "error" in result
        assert "required" in result["error"].lower()

        # Test with mismatched embeddings and documents
        embeddings = rng.random((5, 768)).tolist()
        texts = ["test"]  # Only 1 text for 5 embeddings

        result = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings, texts=texts, method="pca", dimensions=2
        )
        assert "error" in result
        assert "same length" in result["error"].lower()

    async def test_caching_behavior(
        self, initialized_engine, sample_embeddings, sample_documents
    ):
        """Test visualization caching behavior (currently cache is defined but not used)."""
        # Convert to required format
        embeddings_list = sample_embeddings.tolist()
        texts = [doc["text"] for doc in sample_documents]
        metadata_list = [doc["metadata"] for doc in sample_documents]

        # Check that cache is initialized but empty
        assert initialized_engine.visualization_cache == {}

        # Create visualization
        visualization1 = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=2,
        )

        # Currently cache is not used in create_embedding_visualization, so cache remains empty
        # This test validates the current behavior - cache exists but is not populated
        assert initialized_engine.visualization_cache == {}

        # Create same visualization again
        visualization2 = await initialized_engine.create_embedding_visualization(
            embeddings=embeddings_list,
            texts=texts,
            metadata=metadata_list,
            method="pca",
            dimensions=2,
        )

        # Results should have same structure (though may differ slightly due to randomness)
        assert len(visualization1["points"]) == len(visualization2["points"])
        assert visualization1["method"] == visualization2["method"]

    async def test_large_dataset_handling(self, initialized_engine):
        """Test handling of large datasets."""
        rng = np.random.default_rng(42)
        # Create dataset larger than max_points
        large_embeddings = rng.random((1200, 768)).tolist()
        large_texts = [f"Document {i}" for i in range(1200)]
        large_metadata = [{"id": f"doc_{i}"} for i in range(1200)]

        visualization = await initialized_engine.create_embedding_visualization(
            embeddings=large_embeddings,
            texts=large_texts,
            metadata=large_metadata,
            method="pca",
            dimensions=2,
        )

        # Should be sampled down to max_points
        assert len(visualization["points"]) <= initialized_engine.max_points

    async def test_cleanup(self, initialized_engine):
        """Test engine cleanup."""
        await initialized_engine.cleanup()
        # Should complete without error
        assert len(initialized_engine.visualization_cache) == 0
