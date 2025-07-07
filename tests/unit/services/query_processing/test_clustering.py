"""Tests for the result clustering service implementation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.services.query_processing.clustering import (
    ClusterGroup,
    ClusteringMethod,
    ClusteringScope,
    OutlierResult,
    ResultClusteringRequest,
    ResultClusteringResult,
    ResultClusteringService,
    SearchResult,
    SimilarityMetric,
)


class TestClusteringMethod:
    """Test ClusteringMethod enum."""

    def test_all_values(self):
        """Test all clustering method values."""
        assert ClusteringMethod.HDBSCAN == "hdbscan"
        assert ClusteringMethod.DBSCAN == "dbscan"
        assert ClusteringMethod.KMEANS == "kmeans"
        assert ClusteringMethod.AGGLOMERATIVE == "agglomerative"
        assert ClusteringMethod.SPECTRAL == "spectral"
        assert ClusteringMethod.GAUSSIAN_MIXTURE == "gaussian_mixture"
        assert ClusteringMethod.AUTO == "auto"

    def test_enum_iteration(self):
        """Test enum iteration."""
        methods = list(ClusteringMethod)
        assert len(methods) == 7
        assert ClusteringMethod.HDBSCAN in methods
        assert ClusteringMethod.AUTO in methods


class TestClusteringScope:
    """Test ClusteringScope enum."""

    def test_all_values(self):
        """Test all clustering scope values."""
        assert ClusteringScope.STRICT == "strict"
        assert ClusteringScope.MODERATE == "moderate"
        assert ClusteringScope.INCLUSIVE == "inclusive"
        assert ClusteringScope.ADAPTIVE == "adaptive"

    def test_enum_iteration(self):
        """Test enum iteration."""
        scopes = list(ClusteringScope)
        assert len(scopes) == 4
        assert ClusteringScope.MODERATE in scopes


class TestSimilarityMetric:
    """Test SimilarityMetric enum."""

    def test_all_values(self):
        """Test all similarity metric values."""
        assert SimilarityMetric.COSINE == "cosine"
        assert SimilarityMetric.EUCLIDEAN == "euclidean"
        assert SimilarityMetric.MANHATTAN == "manhattan"
        assert SimilarityMetric.JACCARD == "jaccard"
        assert SimilarityMetric.HAMMING == "hamming"

    def test_enum_iteration(self):
        """Test enum iteration."""
        metrics = list(SimilarityMetric)
        assert len(metrics) == 5
        assert SimilarityMetric.COSINE in metrics


class TestSearchResult:
    """Test SearchResult model."""

    def test_default_values(self):
        """Test default search result values."""
        result = SearchResult(
            id="test_1", title="Test Title", content="Test content", score=0.85
        )

        assert result.id == "test_1"
        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.embedding is None
        assert result.metadata == {}

    def test_with_embedding(self):
        """Test search result with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"source": "web", "type": "article"}

        result = SearchResult(
            id="test_2",
            title="Test with Embedding",
            content="Content with embedding",
            score=0.92,
            embedding=embedding,
            metadata=metadata,
        )

        assert result.embedding == embedding
        assert result.metadata == metadata

    def test_score_validation(self):
        """Test score validation."""
        # Valid scores
        result1 = SearchResult(id="1", title="T", content="C", score=0.0)
        assert result1.score == 0.0

        result2 = SearchResult(id="2", title="T", content="C", score=1.0)
        assert result2.score == 1.0

        # Invalid scores
        with pytest.raises(ValueError):
            SearchResult(id="3", title="T", content="C", score=-0.1)

        with pytest.raises(ValueError):
            SearchResult(id="4", title="T", content="C", score=1.1)

    def test_embedding_validation(self):
        """Test embedding validation."""
        # Valid embedding
        result = SearchResult(
            id="1", title="T", content="C", score=0.5, embedding=[0.1, 0.2, 0.3]
        )
        assert result.embedding == [0.1, 0.2, 0.3]

        # Empty embedding should raise error
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            SearchResult(id="2", title="T", content="C", score=0.5, embedding=[])


class TestClusterGroup:
    """Test ClusterGroup model."""

    def test_default_values(self):
        """Test default cluster group values."""
        results = [
            SearchResult(id="1", title="T1", content="C1", score=0.8),
            SearchResult(id="2", title="T2", content="C2", score=0.9),
        ]

        cluster = ClusterGroup(
            cluster_id=1,
            results=results,
            confidence=0.85,
            size=2,
            avg_score=0.85,
            coherence_score=0.75,
        )

        assert cluster.cluster_id == 1
        assert cluster.label is None
        assert cluster.results == results
        assert cluster.centroid is None
        assert cluster.confidence == 0.85
        assert cluster.size == 2
        assert cluster.avg_score == 0.85
        assert cluster.coherence_score == 0.75
        assert cluster.keywords == []
        assert cluster.metadata == {}

    def test_with_all_fields(self):
        """Test cluster group with all fields."""
        results = [SearchResult(id="1", title="T", content="C", score=0.8)]
        centroid = [0.1, 0.2, 0.3]
        keywords = ["test", "cluster"]
        metadata = {"algorithm": "hdbscan"}

        cluster = ClusterGroup(
            cluster_id=5,
            label="Test Cluster",
            results=results,
            centroid=centroid,
            confidence=0.92,
            size=1,
            avg_score=0.8,
            coherence_score=0.88,
            keywords=keywords,
            metadata=metadata,
        )

        assert cluster.label == "Test Cluster"
        assert cluster.centroid == centroid
        assert cluster.keywords == keywords
        assert cluster.metadata == metadata

    def test_validation(self):
        """Test cluster group validation."""
        results = [SearchResult(id="1", title="T", content="C", score=0.8)]

        # Valid values
        cluster = ClusterGroup(
            cluster_id=1,
            results=results,
            confidence=0.0,
            size=0,
            avg_score=1.0,
            coherence_score=0.5,
        )
        assert cluster.confidence == 0.0
        assert cluster.size == 0

        # Invalid confidence
        with pytest.raises(ValueError):
            ClusterGroup(
                cluster_id=1,
                results=results,
                confidence=-0.1,
                size=1,
                avg_score=0.5,
                coherence_score=0.5,
            )

        # Invalid size
        with pytest.raises(ValueError):
            ClusterGroup(
                cluster_id=1,
                results=results,
                confidence=0.5,
                size=-1,
                avg_score=0.5,
                coherence_score=0.5,
            )


class TestOutlierResult:
    """Test OutlierResult model."""

    def test_creation(self):
        """Test outlier result creation."""
        result = SearchResult(id="1", title="T", content="C", score=0.8)

        outlier = OutlierResult(
            result=result, distance_to_nearest_cluster=1.5, outlier_score=0.85
        )

        assert outlier.result == result
        assert outlier.distance_to_nearest_cluster == 1.5
        assert outlier.outlier_score == 0.85

    def test_validation(self):
        """Test outlier result validation."""
        result = SearchResult(id="1", title="T", content="C", score=0.8)

        # Valid values
        outlier = OutlierResult(
            result=result, distance_to_nearest_cluster=0.0, outlier_score=0.0
        )
        assert outlier.distance_to_nearest_cluster == 0.0
        assert outlier.outlier_score == 0.0

        # Invalid distance
        with pytest.raises(ValueError):
            OutlierResult(
                result=result, distance_to_nearest_cluster=-0.1, outlier_score=0.5
            )

        # Invalid score
        with pytest.raises(ValueError):
            OutlierResult(
                result=result, distance_to_nearest_cluster=1.0, outlier_score=1.1
            )


class TestResultClusteringRequest:
    """Test ResultClusteringRequest model."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return [
            SearchResult(
                id=f"result_{i}",
                title=f"Title {i}",
                content=f"Content {i}",
                score=0.5 + (i * 0.1),
                embedding=[i * 0.1, i * 0.2, i * 0.3],
            )
            for i in range(5)
        ]

    def test_default_values(self, sample_results):
        """Test default request values."""
        request = ResultClusteringRequest(results=sample_results)

        assert request.results == sample_results
        assert request.query is None
        assert request.method == ClusteringMethod.HDBSCAN
        assert request.scope == ClusteringScope.MODERATE
        assert request.similarity_metric == SimilarityMetric.COSINE
        assert request.min_cluster_size == 3
        assert request.max_clusters is None
        assert request.min_samples is None
        assert request.eps is None
        assert request.min_cluster_confidence == 0.6
        assert request.outlier_threshold == 0.3
        assert request.use_hierarchical is False
        assert request.generate_labels is True
        assert request.extract_keywords is True
        assert request.max_processing_time_ms == 5000.0
        assert request.enable_caching is True

    def test_custom_values(self, sample_results):
        """Test request with custom values."""
        request = ResultClusteringRequest(
            results=sample_results,
            query="test query",
            method=ClusteringMethod.KMEANS,
            scope=ClusteringScope.STRICT,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
            min_cluster_size=5,
            max_clusters=10,
            min_samples=3,
            eps=0.5,
            min_cluster_confidence=0.8,
            outlier_threshold=0.2,
            use_hierarchical=True,
            generate_labels=False,
            extract_keywords=False,
            max_processing_time_ms=3000.0,
            enable_caching=False,
        )

        assert request.query == "test query"
        assert request.method == ClusteringMethod.KMEANS
        assert request.scope == ClusteringScope.STRICT
        assert request.similarity_metric == SimilarityMetric.EUCLIDEAN
        assert request.min_cluster_size == 5
        assert request.max_clusters == 10
        assert request.min_samples == 3
        assert request.eps == 0.5
        assert request.min_cluster_confidence == 0.8
        assert request.outlier_threshold == 0.2
        assert request.use_hierarchical is True
        assert request.generate_labels is False
        assert request.extract_keywords is False
        assert request.max_processing_time_ms == 3000.0
        assert request.enable_caching is False

    def test_validation_minimum_results(self):
        """Test validation of minimum results."""
        # Too few results
        results = [
            SearchResult(id="1", title="T", content="C", score=0.8),
            SearchResult(id="2", title="T", content="C", score=0.9),
        ]

        with pytest.raises(ValueError, match="Need at least 3 results for clustering"):
            ResultClusteringRequest(results=results)

    def test_parameter_validation(self, sample_results):
        """Test parameter validation."""
        # Invalid min_cluster_size
        with pytest.raises(ValueError):
            ResultClusteringRequest(
                results=sample_results,
                min_cluster_size=1,  # Too small
            )

        # Invalid max_clusters
        with pytest.raises(ValueError):
            ResultClusteringRequest(
                results=sample_results,
                max_clusters=1,  # Too small
            )

        with pytest.raises(ValueError):
            ResultClusteringRequest(
                results=sample_results,
                max_clusters=100,  # Too large
            )

        # Invalid confidence range
        with pytest.raises(ValueError):
            ResultClusteringRequest(results=sample_results, min_cluster_confidence=-0.1)

        # Invalid eps range
        with pytest.raises(ValueError):
            ResultClusteringRequest(
                results=sample_results,
                eps=3.0,  # Too large
            )


class TestResultClusteringResult:
    """Test ResultClusteringResult model."""

    @pytest.fixture
    def sample_cluster(self):
        """Create sample cluster."""
        results = [SearchResult(id="1", title="T", content="C", score=0.8)]
        return ClusterGroup(
            cluster_id=1,
            results=results,
            confidence=0.8,
            size=1,
            avg_score=0.8,
            coherence_score=0.7,
        )

    @pytest.fixture
    def sample_outlier(self):
        """Create sample outlier."""
        result = SearchResult(id="2", title="T", content="C", score=0.6)
        return OutlierResult(
            result=result, distance_to_nearest_cluster=1.5, outlier_score=0.9
        )

    def test_default_values(self, sample_cluster):
        """Test default result values."""
        result = ResultClusteringResult(
            clusters=[sample_cluster],
            method_used=ClusteringMethod.HDBSCAN,
            _total_results=5,
            clustered_results=1,
            outlier_count=4,
            cluster_count=1,
            processing_time_ms=150.0,
        )

        assert result.clusters == [sample_cluster]
        assert result.outliers == []
        assert result.method_used == ClusteringMethod.HDBSCAN
        assert result._total_results == 5
        assert result.clustered_results == 1
        assert result.outlier_count == 4
        assert result.cluster_count == 1
        assert result.silhouette_score is None
        assert result.calinski_harabasz_score is None
        assert result.davies_bouldin_score is None
        assert result.processing_time_ms == 150.0
        assert result.cache_hit is False
        assert result.clustering_metadata == {}

    def test_with_all_fields(self, sample_cluster, sample_outlier):
        """Test result with all fields."""
        metadata = {"algorithm": "hdbscan", "parameters": {"min_size": 3}}

        result = ResultClusteringResult(
            clusters=[sample_cluster],
            outliers=[sample_outlier],
            method_used=ClusteringMethod.HDBSCAN,
            _total_results=5,
            clustered_results=1,
            outlier_count=1,
            cluster_count=1,
            silhouette_score=0.75,
            calinski_harabasz_score=120.5,
            davies_bouldin_score=0.8,
            processing_time_ms=250.0,
            cache_hit=True,
            clustering_metadata=metadata,
        )

        assert result.outliers == [sample_outlier]
        assert result.silhouette_score == 0.75
        assert result.calinski_harabasz_score == 120.5
        assert result.davies_bouldin_score == 0.8
        assert result.cache_hit is True
        assert result.clustering_metadata == metadata

    def test_validation(self, sample_cluster):
        """Test result validation."""
        # Valid silhouette score range
        result = ResultClusteringResult(
            clusters=[sample_cluster],
            method_used=ClusteringMethod.HDBSCAN,
            _total_results=1,
            clustered_results=1,
            outlier_count=0,
            cluster_count=1,
            processing_time_ms=100.0,
            silhouette_score=-1.0,
        )
        assert result.silhouette_score == -1.0

        # Invalid silhouette score
        with pytest.raises(ValueError):
            ResultClusteringResult(
                clusters=[sample_cluster],
                method_used=ClusteringMethod.HDBSCAN,
                _total_results=1,
                clustered_results=1,
                outlier_count=0,
                cluster_count=1,
                processing_time_ms=100.0,
                silhouette_score=-1.1,
            )


class TestResultClusteringService:
    """Test ResultClusteringService implementation."""

    @pytest.fixture
    def clustering_service(self):
        """Create clustering service instance."""
        return ResultClusteringService(
            enable_hdbscan=False,  # Disable to avoid import issues in tests
            enable_advanced_metrics=False,  # Disable to avoid sklearn metrics
            cache_size=100,
        )

    @pytest.fixture
    def sample_results_with_embeddings(self):
        """Create sample results with embeddings."""
        rng = np.random.default_rng(42)  # For reproducible tests
        return [
            SearchResult(
                id=f"result_{i}",
                title=f"Document {i}",
                content=f"Content about topic {i % 3}",
                score=0.5 + (i * 0.05),
                embedding=rng.random(10).tolist(),
            )
            for i in range(10)
        ]

    @pytest.fixture
    def clustering_request(self, sample_results_with_embeddings):
        """Create clustering request."""
        return ResultClusteringRequest(
            results=sample_results_with_embeddings,
            method=ClusteringMethod.KMEANS,  # Use K-means to avoid HDBSCAN
            min_cluster_size=2,
            max_clusters=3,
        )

    def test_initialization(self):
        """Test service initialization."""
        service = ResultClusteringService(
            enable_hdbscan=True, enable_advanced_metrics=True, cache_size=500
        )

        # HDBSCAN might not be available in test environment
        assert isinstance(service.enable_hdbscan, bool)
        assert service.enable_advanced_metrics is True
        assert service.cache_size == 500
        assert service.clustering_cache == {}
        assert service.cache_stats == {"hits": 0, "misses": 0}
        assert isinstance(service.available_algorithms, dict)
        assert isinstance(service.performance_stats, dict)

    def test_check_algorithm_availability(self, clustering_service):
        """Test algorithm availability checking."""
        algorithms = clustering_service._check_algorithm_availability()

        assert isinstance(algorithms, dict)
        assert "sklearn" in algorithms
        assert "numpy" in algorithms
        # HDBSCAN might not be available in test environment
        assert "hdbscan" in algorithms

    def test_validate_clustering_request_valid(
        self, clustering_service, clustering_request
    ):
        """Test validating valid clustering request."""
        is_valid = clustering_service._validate_clustering_request(clustering_request)
        assert is_valid is True

    def test_validate_clustering_request_invalid(
        self, clustering_service, sample_results_with_embeddings
    ):
        """Test validating invalid clustering requests."""
        # Create a valid request first, then test validation logic directly
        valid_request = ResultClusteringRequest(
            results=sample_results_with_embeddings,
            min_cluster_size=15,  # This makes it invalid (more than available results)
        )
        is_valid = clustering_service._validate_clustering_request(valid_request)
        assert is_valid is False

        # HDBSCAN not available
        hdbscan_request = ResultClusteringRequest(
            results=sample_results_with_embeddings, method=ClusteringMethod.HDBSCAN
        )
        is_valid = clustering_service._validate_clustering_request(hdbscan_request)
        assert is_valid is False  # Since we disabled HDBSCAN

    def test_extract_embeddings_valid(
        self, clustering_service, sample_results_with_embeddings
    ):
        """Test extracting valid embeddings."""
        embeddings = clustering_service._extract_embeddings(
            sample_results_with_embeddings
        )

        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_results_with_embeddings)
        assert embeddings.shape[1] == 10  # Embedding dimension

    def test_extract_embeddings_invalid(self, clustering_service):
        """Test extracting embeddings from invalid results."""
        # Results without embeddings
        results_no_embeddings = [
            SearchResult(id="1", title="T", content="C", score=0.8),
            SearchResult(id="2", title="T", content="C", score=0.9),
            SearchResult(id="3", title="T", content="C", score=0.7),
        ]

        embeddings = clustering_service._extract_embeddings(results_no_embeddings)
        assert embeddings is None

        # Test with results that have None embeddings explicitly
        results_none_embeddings = [
            SearchResult(id="1", title="T", content="C", score=0.8, embedding=None),
            SearchResult(id="2", title="T", content="C", score=0.9, embedding=None),
        ]

        embeddings = clustering_service._extract_embeddings(results_none_embeddings)
        assert embeddings is None

    def test_select_clustering_method_auto(self, clustering_service):
        """Test automatic clustering method selection."""
        # Small dataset -> should select DBSCAN (since HDBSCAN disabled)
        rng = np.random.default_rng(42)
        small_embeddings = rng.random((10, 5))
        request = ResultClusteringRequest(
            results=[
                SearchResult(id=str(i), title="T", content="C", score=0.5)
                for i in range(10)
            ],
            method=ClusteringMethod.AUTO,
            min_cluster_size=2,
        )

        method = clustering_service._select_clustering_method(request, small_embeddings)
        assert method == ClusteringMethod.DBSCAN

        # Medium dataset -> should select DBSCAN
        rng = np.random.default_rng(42)
        medium_embeddings = rng.random((80, 5))
        method = clustering_service._select_clustering_method(
            request, medium_embeddings
        )
        assert method == ClusteringMethod.DBSCAN

        # Large dataset with max_clusters -> should select K-means
        large_embeddings = rng.random((150, 5))
        request.max_clusters = 5
        method = clustering_service._select_clustering_method(request, large_embeddings)
        assert method == ClusteringMethod.KMEANS

    def test_select_clustering_method_explicit(self, clustering_service):
        """Test explicit clustering method selection."""
        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 5))
        request = ResultClusteringRequest(
            results=[
                SearchResult(id=str(i), title="T", content="C", score=0.5)
                for i in range(10)
            ],
            method=ClusteringMethod.KMEANS,
        )

        method = clustering_service._select_clustering_method(request, embeddings)
        assert method == ClusteringMethod.KMEANS

    @patch("sklearn.cluster.KMeans")
    def test_apply_kmeans(self, mock_kmeans_class, clustering_service):
        """Test applying K-means clustering."""
        # Mock K-means clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, 2])
        mock_clusterer.inertia_ = 50.0
        mock_clusterer.n_iter_ = 5
        mock_kmeans_class.return_value = mock_clusterer

        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 3))
        request = ResultClusteringRequest(
            results=[
                SearchResult(id=str(i), title="T", content="C", score=0.5)
                for i in range(5)
            ],
            max_clusters=3,
        )
        metadata = {"method": "kmeans"}

        labels, result_metadata = clustering_service._apply_kmeans(
            embeddings, request, metadata
        )

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 5
        assert "n_clusters" in result_metadata
        assert "inertia" in result_metadata
        assert "n_iter" in result_metadata
        mock_kmeans_class.assert_called_once()

    @patch("sklearn.cluster.DBSCAN")
    def test_apply_dbscan(self, mock_dbscan_class, clustering_service):
        """Test applying DBSCAN clustering."""
        # Mock DBSCAN clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
        mock_dbscan_class.return_value = mock_clusterer

        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 3))
        request = ResultClusteringRequest(
            results=[
                SearchResult(id=str(i), title="T", content="C", score=0.5)
                for i in range(5)
            ],
            eps=0.5,
            min_samples=2,
        )
        metadata = {"method": "dbscan"}

        labels, result_metadata = clustering_service._apply_dbscan(
            embeddings, request, metadata
        )

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 5
        assert "eps" in result_metadata
        assert "min_samples" in result_metadata
        assert "n_clusters" in result_metadata
        assert "n_noise" in result_metadata

    @patch("sklearn.cluster.AgglomerativeClustering")
    def test_apply_agglomerative(self, mock_agglomerative_class, clustering_service):
        """Test applying agglomerative clustering."""
        # Mock agglomerative clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, 2])
        mock_agglomerative_class.return_value = mock_clusterer

        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 3))
        request = ResultClusteringRequest(
            results=[
                SearchResult(id=str(i), title="T", content="C", score=0.5)
                for i in range(5)
            ],
            max_clusters=3,
        )
        metadata = {"method": "agglomerative"}

        labels, result_metadata = clustering_service._apply_agglomerative(
            embeddings, request, metadata
        )

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 5
        assert "n_clusters" in result_metadata
        assert "linkage" in result_metadata

    def test_estimate_eps(self, clustering_service):
        """Test eps estimation for DBSCAN."""
        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 5))
        min_cluster_size = 3

        eps = clustering_service._estimate_eps(embeddings, min_cluster_size)

        assert isinstance(eps, float)
        assert eps > 0.0
        assert eps <= 0.5  # Should be capped

    def test_build_cluster_groups(
        self, clustering_service, sample_results_with_embeddings
    ):
        """Test building cluster groups."""
        # Use controlled embeddings to ensure coherence stays within bounds
        rng = np.random.default_rng(42)  # Set seed for reproducible test
        embeddings = (
            rng.random((10, 5)) * 0.5
        )  # Smaller values to keep coherence reasonable
        cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1])
        request = ResultClusteringRequest(
            results=sample_results_with_embeddings,
            min_cluster_size=2,
            generate_labels=True,
            extract_keywords=True,
        )

        clusters = clustering_service._build_cluster_groups(
            sample_results_with_embeddings, cluster_labels, embeddings, request
        )

        assert isinstance(clusters, list)
        assert len(clusters) == 3  # Three clusters (excluding noise -1)

        for cluster in clusters:
            assert isinstance(cluster, ClusterGroup)
            assert cluster.size >= request.min_cluster_size
            assert len(cluster.results) == cluster.size
            assert cluster.centroid is not None
            assert 0.0 <= cluster.confidence <= 1.0
            assert 0.0 <= cluster.coherence_score <= 1.0

    def test_identify_outliers(
        self, clustering_service, sample_results_with_embeddings
    ):
        """Test identifying outliers."""
        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 5))
        cluster_labels = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, 0])
        request = ResultClusteringRequest(results=sample_results_with_embeddings)

        outliers = clustering_service._identify_outliers(
            sample_results_with_embeddings, cluster_labels, embeddings, request
        )

        assert isinstance(outliers, list)
        # Should have 3 outliers (indices 6, 7, 8 with label -1)
        assert len(outliers) == 3

        for outlier in outliers:
            assert isinstance(outlier, OutlierResult)
            assert outlier.distance_to_nearest_cluster >= 0.0
            assert 0.0 <= outlier.outlier_score <= 1.0

    def test_calculate_coherence(self, clustering_service):
        """Test calculating cluster coherence."""
        # Perfect coherence (identical embeddings)
        identical_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        coherence = clustering_service._calculate_coherence(identical_embeddings)
        assert coherence == pytest.approx(1.0, abs=0.01)

        # Single embedding
        single_embedding = np.array([[1.0, 0.0]])
        coherence = clustering_service._calculate_coherence(single_embedding)
        assert coherence == 1.0

        # Random embeddings
        rng = np.random.default_rng(42)
        random_embeddings = rng.random((5, 3))
        coherence = clustering_service._calculate_coherence(random_embeddings)
        assert 0.0 <= coherence <= 1.0

    def test_calculate_cluster_confidence(self, clustering_service):
        """Test calculating cluster confidence."""
        rng = np.random.default_rng(42)
        cluster_embeddings = rng.random((5, 3))
        all_embeddings = rng.random((20, 3))
        request = ResultClusteringRequest(
            results=[
                SearchResult(id=str(i), title="T", content="C", score=0.5)
                for i in range(20)
            ],
            min_cluster_size=3,
        )

        confidence = clustering_service._calculate_cluster_confidence(
            cluster_embeddings, all_embeddings, request
        )

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_generate_cluster_label(self, clustering_service):
        """Test generating cluster labels."""
        results = [
            SearchResult(
                id="1",
                title="Python Programming Tutorial",
                content="Learn Python",
                score=0.8,
            ),
            SearchResult(
                id="2",
                title="Advanced Python Concepts",
                content="Python advanced",
                score=0.9,
            ),
            SearchResult(
                id="3",
                title="Python Data Science",
                content="Data science with Python",
                score=0.7,
            ),
        ]

        label = clustering_service._generate_cluster_label(results, "python tutorial")

        assert isinstance(label, str)
        assert len(label) > 0
        # Should contain Python since it's common in titles
        assert "Python" in label or "python" in label.lower()

    def test_extract_cluster_keywords(self, clustering_service):
        """Test extracting cluster keywords."""
        results = [
            SearchResult(
                id="1",
                title="Machine Learning Tutorial",
                content="Learn machine learning algorithms",
                score=0.8,
            ),
            SearchResult(
                id="2",
                title="Deep Learning Guide",
                content="Neural networks and deep learning",
                score=0.9,
            ),
            SearchResult(
                id="3",
                title="AI and Machine Learning",
                content="Artificial intelligence concepts",
                score=0.7,
            ),
        ]

        keywords = clustering_service._extract_cluster_keywords(results)

        assert isinstance(keywords, list)
        assert len(keywords) >= 0
        # Should contain relevant keywords
        keyword_text = " ".join(keywords).lower()
        assert any(word in keyword_text for word in ["learning", "machine"])

    def test_get_algorithm_parameters(self, clustering_service):
        """Test getting algorithm parameters."""
        # Create valid request with sufficient results
        results = [
            SearchResult(
                id=f"result_{i}", title=f"Title {i}", content=f"Content {i}", score=0.5
            )
            for i in range(5)
        ]

        request = ResultClusteringRequest(
            results=results,
            method=ClusteringMethod.KMEANS,
            min_cluster_size=3,
            max_clusters=5,
            similarity_metric=SimilarityMetric.COSINE,
        )

        params = clustering_service._get_algorithm_parameters(
            ClusteringMethod.KMEANS, request
        )

        assert isinstance(params, dict)
        assert params["method"] == "kmeans"
        assert params["min_cluster_size"] == 3
        assert params["max_clusters"] == 5
        assert params["similarity_metric"] == "cosine"

    def test_cache_operations(self, clustering_service, clustering_request):
        """Test caching operations."""
        # Generate cache key
        cache_key = clustering_service._generate_cache_key(clustering_request)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

        # Initially no cached result
        cached_result = clustering_service._get_cached_result(clustering_request)
        assert cached_result is None
        assert clustering_service.cache_stats["misses"] == 1

        # Cache a result
        result = ResultClusteringResult(
            clusters=[],
            method_used=ClusteringMethod.KMEANS,
            _total_results=10,
            clustered_results=0,
            outlier_count=10,
            cluster_count=0,
            processing_time_ms=100.0,
        )

        clustering_service._cache_result(clustering_request, result)

        # Should retrieve cached result
        cached_result = clustering_service._get_cached_result(clustering_request)
        assert cached_result is not None
        assert cached_result.method_used == ClusteringMethod.KMEANS
        assert clustering_service.cache_stats["hits"] == 1

    def test_performance_stats_update(self, clustering_service):
        """Test performance statistics update."""
        initial_stats = clustering_service.get_performance_stats()
        assert initial_stats["_total_clusterings"] == 0

        # Update stats
        clustering_service._update_performance_stats(ClusteringMethod.KMEANS, 150.0)

        updated_stats = clustering_service.get_performance_stats()
        assert updated_stats["_total_clusterings"] == 1
        assert updated_stats["avg_processing_time"] == 150.0
        assert updated_stats["method_usage"]["kmeans"] == 1

    def test_clear_cache(self, clustering_service, clustering_request):
        """Test clearing cache."""
        # Add something to cache
        result = ResultClusteringResult(
            clusters=[],
            method_used=ClusteringMethod.KMEANS,
            _total_results=5,
            clustered_results=0,
            outlier_count=5,
            cluster_count=0,
            processing_time_ms=100.0,
        )
        clustering_service._cache_result(clustering_request, result)

        assert len(clustering_service.clustering_cache) == 1

        # Clear cache
        clustering_service.clear_cache()

        assert len(clustering_service.clustering_cache) == 0
        assert clustering_service.cache_stats == {"hits": 0, "misses": 0}

    @pytest.mark.asyncio
    @patch("sklearn.cluster.KMeans")
    @pytest.mark.asyncio
    async def test_cluster_results_success(
        self, mock_kmeans_class, clustering_service, clustering_request
    ):
        """Test successful clustering."""
        # Mock K-means
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array(
            [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
        )
        mock_clusterer.inertia_ = 50.0
        mock_clusterer.n_iter_ = 5
        mock_kmeans_class.return_value = mock_clusterer

        result = await clustering_service.cluster_results(clustering_request)

        assert isinstance(result, ResultClusteringResult)
        assert result.method_used == ClusteringMethod.KMEANS
        assert result._total_results == 10
        assert result.processing_time_ms > 0
        assert result.cache_hit is False

    @pytest.mark.asyncio
    async def test_cluster_results_cached(self, clustering_service, clustering_request):
        """Test clustering with cached result."""
        # Pre-cache a result
        cached_result = ResultClusteringResult(
            clusters=[],
            method_used=ClusteringMethod.KMEANS,
            _total_results=10,
            clustered_results=0,
            outlier_count=10,
            cluster_count=0,
            processing_time_ms=100.0,
            cache_hit=False,
        )
        clustering_service._cache_result(clustering_request, cached_result)

        result = await clustering_service.cluster_results(clustering_request)

        assert result.cache_hit is True
        assert result.method_used == ClusteringMethod.KMEANS

    @pytest.mark.asyncio
    async def test_cluster_results_error_handling(self, clustering_service):
        """Test error handling in clustering."""
        # Invalid request (no embeddings)
        invalid_request = ResultClusteringRequest(
            results=[
                SearchResult(id="1", title="T", content="C", score=0.5),
                SearchResult(id="2", title="T", content="C", score=0.6),
                SearchResult(id="3", title="T", content="C", score=0.7),
            ]
        )

        result = await clustering_service.cluster_results(invalid_request)

        # Should return fallback result
        assert isinstance(result, ResultClusteringResult)
        assert result.cluster_count == 0
        assert result.outlier_count == 3
        assert "error" in result.clustering_metadata

    @pytest.mark.asyncio
    async def test_cluster_results_validation_failure(self, clustering_service):
        """Test clustering with validation failure."""
        # Create a request that will fail validation but not Pydantic validation
        # Use sufficient results but set min_cluster_size too high
        results = [
            SearchResult(
                id=f"result_{i}",
                title=f"Title {i}",
                content=f"Content {i}",
                score=0.5,
                embedding=[0.1, 0.2],
            )
            for i in range(5)
        ]

        request = ResultClusteringRequest(
            results=results,
            min_cluster_size=10,  # This will cause validation to fail
        )

        result = await clustering_service.cluster_results(request)

        # Should return fallback result
        assert result.cluster_count == 0
        assert result.outlier_count == 5

    @pytest.mark.asyncio
    async def test_edge_case_single_cluster(self, clustering_service):
        """Test edge case with results forming single cluster."""
        # Create results with very similar embeddings
        similar_embedding = [0.5, 0.5, 0.5]
        results = [
            SearchResult(
                id=f"result_{i}",
                title=f"Similar Document {i}",
                content="Very similar content",
                score=0.8,
                embedding=[
                    x + (i * 0.01) for x in similar_embedding
                ],  # Slightly different
            )
            for i in range(5)
        ]

        request = ResultClusteringRequest(
            results=results,
            method=ClusteringMethod.KMEANS,
            max_clusters=2,  # Minimum allowed is 2
            min_cluster_size=3,
        )

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0])
            mock_clusterer.inertia_ = 0.1
            mock_clusterer.n_iter_ = 3
            mock_kmeans.return_value = mock_clusterer

            result = await clustering_service.cluster_results(request)

        assert result.cluster_count == 1
        assert result.clustered_results == 5
        assert result.outlier_count == 0

    @pytest.mark.asyncio
    async def test_edge_case_all_outliers(self, clustering_service):
        """Test edge case where all results are outliers."""
        # Create results with very different embeddings
        results = [
            SearchResult(
                id=f"result_{i}",
                title=f"Unique Document {i}",
                content=f"Completely different content {i}",
                score=0.5,
                embedding=[
                    i * 2.0 if j == i else 0.0 for j in range(5)
                ],  # Orthogonal embeddings
            )
            for i in range(5)
        ]

        request = ResultClusteringRequest(
            results=results,
            method=ClusteringMethod.DBSCAN,
            min_cluster_size=3,
            eps=0.1,  # Very small eps to prevent clustering
        )

        with patch("sklearn.cluster.DBSCAN") as mock_dbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array(
                [-1, -1, -1, -1, -1]
            )  # All outliers
            mock_dbscan.return_value = mock_clusterer

            result = await clustering_service.cluster_results(request)

        assert result.cluster_count == 0
        assert result.clustered_results == 0
        assert result.outlier_count == 5

    @pytest.mark.asyncio
    async def test_different_similarity_metrics(
        self, clustering_service, sample_results_with_embeddings
    ):
        """Test clustering with different similarity metrics."""
        metrics_to_test = [
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.MANHATTAN,
        ]

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array(
                [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
            )
            mock_clusterer.inertia_ = 50.0
            mock_clusterer.n_iter_ = 5
            mock_kmeans.return_value = mock_clusterer

            for metric in metrics_to_test:
                request = ResultClusteringRequest(
                    results=sample_results_with_embeddings,
                    method=ClusteringMethod.KMEANS,
                    similarity_metric=metric,
                    max_clusters=3,
                )

                result = await clustering_service.cluster_results(request)
                assert isinstance(result, ResultClusteringResult)
                assert result.method_used == ClusteringMethod.KMEANS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
