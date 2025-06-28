"""Unit tests for vector search models."""

import pytest
from pydantic import ValidationError
from src.config.enums import FusionAlgorithm
from src.config.enums import SearchAccuracy
from src.config.enums import VectorType
from src.models.vector_search import AdaptiveSearchParams
from src.models.vector_search import CollectionStats
from src.models.vector_search import FilteredSearchRequest
from src.models.vector_search import FusionConfig
from src.models.vector_search import HybridSearchRequest
from src.models.vector_search import HyDESearchRequest
from src.models.vector_search import IndexingRequest
from src.models.vector_search import MultiStageSearchRequest
from src.models.vector_search import OptimizationRequest
from src.models.vector_search import PrefetchConfig
from src.models.vector_search import RetrievalMetrics
from src.models.vector_search import SearchParams
from src.models.vector_search import SearchResponse
from src.models.vector_search import SearchResult
from src.models.vector_search import SearchStage
from src.models.vector_search import VectorSearchConfig


class TestSearchStage:
    """Test SearchStage model."""

    def test_required_fields(self):
        """Test required fields."""
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=10,
        )
        assert stage.query_vector == [0.1, 0.2, 0.3]
        assert stage.vector_name == "dense"
        assert stage.vector_type == VectorType.DENSE
        assert stage.limit == 10

    def test_optional_fields(self):
        """Test optional fields."""
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=10,
        )
        assert stage.filter is None
        assert stage.search_params is None

    def test_with_filter_and_params(self):
        """Test stage with filter and search params."""
        filter_dict = {"status": "active", "category": "tech"}
        search_params = {"hnsw_ef": 100}

        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=20,
            filter=filter_dict,
            search_params=search_params,
        )
        assert stage.filter == filter_dict
        assert stage.search_params == search_params


class TestPrefetchConfig:
    """Test PrefetchConfig model."""

    def test_default_values(self):
        """Test default field values."""
        config = PrefetchConfig()
        assert config.sparse_multiplier == 5.0
        assert config.hyde_multiplier == 3.0
        assert config.dense_multiplier == 2.0
        assert config.max_sparse_limit == 500
        assert config.max_dense_limit == 200
        assert config.max_hyde_limit == 150

    def test_calculate_prefetch_limit_sparse(self):
        """Test prefetch limit calculation for sparse vectors."""
        config = PrefetchConfig()

        # Within limit
        assert config.calculate_prefetch_limit(VectorType.SPARSE, 50) == 250  # 50 * 5.0

        # Exceeds max limit
        assert (
            config.calculate_prefetch_limit(VectorType.SPARSE, 200) == 500
        )  # capped at max

    def test_calculate_prefetch_limit_hyde(self):
        """Test prefetch limit calculation for HyDE vectors."""
        config = PrefetchConfig()

        # Within limit
        assert config.calculate_prefetch_limit(VectorType.HYDE, 30) == 90  # 30 * 3.0

        # Exceeds max limit
        assert (
            config.calculate_prefetch_limit(VectorType.HYDE, 100) == 150
        )  # capped at max

    def test_calculate_prefetch_limit_dense(self):
        """Test prefetch limit calculation for dense vectors."""
        config = PrefetchConfig()

        # Within limit
        assert config.calculate_prefetch_limit(VectorType.DENSE, 50) == 100  # 50 * 2.0

        # Exceeds max limit
        assert (
            config.calculate_prefetch_limit(VectorType.DENSE, 150) == 200
        )  # capped at max

    def test_custom_config(self):
        """Test custom configuration."""
        config = PrefetchConfig(
            sparse_multiplier=4.0,
            hyde_multiplier=2.5,
            dense_multiplier=1.5,
            max_sparse_limit=400,
            max_dense_limit=150,
            max_hyde_limit=100,
        )
        assert config.sparse_multiplier == 4.0
        assert config.max_sparse_limit == 400


class TestSearchParams:
    """Test SearchParams model."""

    def test_default_values(self):
        """Test default field values."""
        params = SearchParams()
        assert params.accuracy_level == SearchAccuracy.BALANCED
        assert params.hnsw_ef is None
        assert params.exact is False

    def test_from_accuracy_level_fast(self):
        """Test creating params from FAST accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.FAST)
        assert params.accuracy_level == SearchAccuracy.FAST
        assert params.hnsw_ef == 50
        assert params.exact is False

    def test_from_accuracy_level_balanced(self):
        """Test creating params from BALANCED accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.BALANCED)
        assert params.accuracy_level == SearchAccuracy.BALANCED
        assert params.hnsw_ef == 100
        assert params.exact is False

    def test_from_accuracy_level_accurate(self):
        """Test creating params from ACCURATE accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.ACCURATE)
        assert params.accuracy_level == SearchAccuracy.ACCURATE
        assert params.hnsw_ef == 200
        assert params.exact is False

    def test_from_accuracy_level_exact(self):
        """Test creating params from EXACT accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.EXACT)
        assert params.accuracy_level == SearchAccuracy.EXACT
        assert params.hnsw_ef is None  # Not used for exact search
        assert params.exact is True

    def test_custom_params(self):
        """Test custom search params."""
        params = SearchParams(
            accuracy_level=SearchAccuracy.FAST,
            hnsw_ef=75,
            exact=False,
        )
        assert params.hnsw_ef == 75


class TestFusionConfig:
    """Test FusionConfig model."""

    def test_default_values(self):
        """Test default field values."""
        config = FusionConfig()
        assert config.algorithm == FusionAlgorithm.RRF
        assert config.auto_select is True

    def test_select_fusion_algorithm(self):
        """Test fusion algorithm selection based on query type."""
        assert FusionConfig.select_fusion_algorithm("hybrid") == FusionAlgorithm.RRF
        assert (
            FusionConfig.select_fusion_algorithm("multi_stage") == FusionAlgorithm.RRF
        )
        assert FusionConfig.select_fusion_algorithm("reranking") == FusionAlgorithm.DBSF
        assert FusionConfig.select_fusion_algorithm("hyde") == FusionAlgorithm.RRF
        assert (
            FusionConfig.select_fusion_algorithm("unknown") == FusionAlgorithm.RRF
        )  # default

    def test_custom_config(self):
        """Test custom fusion config."""
        config = FusionConfig(
            algorithm=FusionAlgorithm.DBSF,
            auto_select=False,
        )
        assert config.algorithm == FusionAlgorithm.DBSF
        assert config.auto_select is False


class TestMultiStageSearchRequest:
    """Test MultiStageSearchRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        stages = [
            SearchStage(
                query_vector=[0.1, 0.2],
                vector_name="dense",
                vector_type=VectorType.DENSE,
                limit=10,
            )
        ]
        request = MultiStageSearchRequest(
            collection_name="test_collection",
            stages=stages,
        )
        assert request.collection_name == "test_collection"
        assert len(request.stages) == 1

    def test_default_values(self):
        """Test default field values."""
        stages = [
            SearchStage(
                query_vector=[0.1, 0.2],
                vector_name="dense",
                vector_type=VectorType.DENSE,
                limit=10,
            )
        ]
        request = MultiStageSearchRequest(
            collection_name="test",
            stages=stages,
        )
        assert isinstance(request.fusion_config, FusionConfig)
        assert isinstance(request.search_params, SearchParams)
        assert request.limit == 10
        assert request.score_threshold == 0.0

    def test_multiple_stages(self):
        """Test request with multiple stages."""
        stages = [
            SearchStage(
                query_vector=[0.1, 0.2],
                vector_name="dense",
                vector_type=VectorType.DENSE,
                limit=20,
            ),
            SearchStage(
                query_vector=[0.3, 0.4],
                vector_name="sparse",
                vector_type=VectorType.SPARSE,
                limit=30,
            ),
        ]
        request = MultiStageSearchRequest(
            collection_name="test",
            stages=stages,
            limit=15,
            score_threshold=0.5,
        )
        assert len(request.stages) == 2
        assert request.limit == 15
        assert request.score_threshold == 0.5


class TestHyDESearchRequest:
    """Test HyDESearchRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = HyDESearchRequest(
            collection_name="test_collection",
            query="What is machine learning?",
        )
        assert request.collection_name == "test_collection"
        assert request.query == "What is machine learning?"

    def test_default_values(self):
        """Test default field values."""
        request = HyDESearchRequest(
            collection_name="test",
            query="test query",
        )
        assert request.num_hypothetical_docs == 5
        assert request.limit == 10
        assert isinstance(request.fusion_config, FusionConfig)
        assert isinstance(request.search_params, SearchParams)

    def test_custom_values(self):
        """Test custom field values."""
        request = HyDESearchRequest(
            collection_name="test",
            query="test query",
            num_hypothetical_docs=3,
            limit=20,
        )
        assert request.num_hypothetical_docs == 3
        assert request.limit == 20


class TestFilteredSearchRequest:
    """Test FilteredSearchRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = FilteredSearchRequest(
            collection_name="test",
            query_vector=[0.1, 0.2, 0.3],
            filters={"status": "active"},
        )
        assert request.collection_name == "test"
        assert request.query_vector == [0.1, 0.2, 0.3]
        assert request.filters == {"status": "active"}

    def test_default_values(self):
        """Test default field values."""
        request = FilteredSearchRequest(
            collection_name="test",
            query_vector=[0.1, 0.2],
            filters={},
        )
        assert request.limit == 10
        assert isinstance(request.search_params, SearchParams)
        assert request.score_threshold == 0.0

    def test_complex_filters(self):
        """Test request with complex filters."""
        filters = {
            "status": "active",
            "category": ["tech", "science"],
            "score": {"gte": 0.8},
        }
        request = FilteredSearchRequest(
            collection_name="test",
            query_vector=[0.1, 0.2, 0.3],
            filters=filters,
            limit=20,
            score_threshold=0.7,
        )
        assert request.filters == filters
        assert request.limit == 20
        assert request.score_threshold == 0.7


class TestHybridSearchRequest:
    """Test HybridSearchRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = HybridSearchRequest(
            collection_name="test",
            dense_vector=[0.1, 0.2, 0.3],
        )
        assert request.collection_name == "test"
        assert request.dense_vector == [0.1, 0.2, 0.3]

    def test_default_values(self):
        """Test default field values."""
        request = HybridSearchRequest(
            collection_name="test",
            dense_vector=[0.1, 0.2],
        )
        assert request.sparse_vector is None
        assert request.dense_weight == 0.7
        assert request.sparse_weight == 0.3
        assert request.limit == 10
        assert isinstance(request.search_params, SearchParams)
        assert request.score_threshold == 0.0

    def test_weight_constraints(self):
        """Test weight field constraints."""
        # Valid weights
        HybridSearchRequest(
            collection_name="test", dense_vector=[0.1], dense_weight=0.0
        )
        HybridSearchRequest(
            collection_name="test", dense_vector=[0.1], dense_weight=1.0
        )
        HybridSearchRequest(
            collection_name="test", dense_vector=[0.1], sparse_weight=0.0
        )
        HybridSearchRequest(
            collection_name="test", dense_vector=[0.1], sparse_weight=1.0
        )

        # Invalid weights
        with pytest.raises(ValidationError):
            HybridSearchRequest(
                collection_name="test", dense_vector=[0.1], dense_weight=-0.1
            )
        with pytest.raises(ValidationError):
            HybridSearchRequest(
                collection_name="test", dense_vector=[0.1], dense_weight=1.1
            )
        with pytest.raises(ValidationError):
            HybridSearchRequest(
                collection_name="test", dense_vector=[0.1], sparse_weight=-0.1
            )
        with pytest.raises(ValidationError):
            HybridSearchRequest(
                collection_name="test", dense_vector=[0.1], sparse_weight=1.1
            )

    def test_with_sparse_vector(self):
        """Test request with sparse vector."""
        sparse_vector = {"indices": [0, 5, 10], "values": [0.5, 0.8, 0.2]}
        request = HybridSearchRequest(
            collection_name="test",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=sparse_vector,
            dense_weight=0.6,
            sparse_weight=0.4,
        )
        assert request.sparse_vector == sparse_vector
        assert request.dense_weight == 0.6
        assert request.sparse_weight == 0.4


class TestSearchResult:
    """Test SearchResult model."""

    def test_required_fields(self):
        """Test required fields."""
        result = SearchResult(id="doc123", score=0.95)
        assert result.id == "doc123"
        assert result.score == 0.95

    def test_default_values(self):
        """Test default field values."""
        result = SearchResult(id="doc123", score=0.95)
        assert result.payload == {}
        assert result.vector is None

    def test_with_payload_and_vector(self):
        """Test result with payload and vector."""
        payload = {
            "title": "Test Document",
            "content": "This is a test",
            "metadata": {"author": "John Doe"},
        }
        vector = [0.1, 0.2, 0.3, 0.4]

        result = SearchResult(
            id="doc123",
            score=0.95,
            payload=payload,
            vector=vector,
        )
        assert result.payload == payload
        assert result.vector == vector


class TestSearchResponse:
    """Test SearchResponse model."""

    def test_default_values(self):
        """Test default field values."""
        response = SearchResponse()
        assert response.results == []
        assert response._total_count == 0
        assert response.query_time_ms == 0.0
        assert response.search_params == {}

    def test_with_results(self):
        """Test response with results."""
        results = [
            SearchResult(id="doc1", score=0.9),
            SearchResult(id="doc2", score=0.8),
        ]
        search_params = {"hnsw_ef": 100, "limit": 10}

        response = SearchResponse(
            results=results,
            _total_count=50,
            query_time_ms=15.5,
            search_params=search_params,
        )
        assert len(response.results) == 2
        assert response._total_count == 50
        assert response.query_time_ms == 15.5
        assert response.search_params == search_params


class TestRetrievalMetrics:
    """Test RetrievalMetrics model."""

    def test_default_values(self):
        """Test default field values."""
        metrics = RetrievalMetrics()
        assert metrics.query_vector_time_ms == 0.0
        assert metrics.search_time_ms == 0.0
        assert metrics._total_time_ms == 0.0
        assert metrics.results_count == 0
        assert metrics.filtered_count == 0
        assert metrics.cache_hit is False
        assert metrics.hnsw_ef_used is None

    def test_with_metrics(self):
        """Test metrics with values."""
        metrics = RetrievalMetrics(
            query_vector_time_ms=5.0,
            search_time_ms=10.0,
            _total_time_ms=15.5,
            results_count=20,
            filtered_count=15,
            cache_hit=True,
            hnsw_ef_used=100,
        )
        assert metrics.query_vector_time_ms == 5.0
        assert metrics.search_time_ms == 10.0
        assert metrics._total_time_ms == 15.5
        assert metrics.results_count == 20
        assert metrics.filtered_count == 15
        assert metrics.cache_hit is True
        assert metrics.hnsw_ef_used == 100


class TestAdaptiveSearchParams:
    """Test AdaptiveSearchParams model."""

    def test_default_values(self):
        """Test default field values."""
        params = AdaptiveSearchParams()
        assert params.time_budget_ms == 100
        assert params.min_results == 5
        assert params.max_ef == 200
        assert params.min_ef == 50
        assert params.ef_step == 25

    def test_constraints(self):
        """Test field constraints."""
        # Valid values
        AdaptiveSearchParams(time_budget_ms=1)
        AdaptiveSearchParams(min_results=1)
        AdaptiveSearchParams(max_ef=1)
        AdaptiveSearchParams(min_ef=1)
        AdaptiveSearchParams(ef_step=1)

        # Invalid values
        with pytest.raises(ValidationError):
            AdaptiveSearchParams(time_budget_ms=0)
        with pytest.raises(ValidationError):
            AdaptiveSearchParams(min_results=0)
        with pytest.raises(ValidationError):
            AdaptiveSearchParams(max_ef=0)
        with pytest.raises(ValidationError):
            AdaptiveSearchParams(min_ef=0)
        with pytest.raises(ValidationError):
            AdaptiveSearchParams(ef_step=0)

    def test_custom_values(self):
        """Test custom parameter values."""
        params = AdaptiveSearchParams(
            time_budget_ms=200,
            min_results=10,
            max_ef=300,
            min_ef=25,
            ef_step=50,
        )
        assert params.time_budget_ms == 200
        assert params.min_results == 10
        assert params.max_ef == 300
        assert params.min_ef == 25
        assert params.ef_step == 50


class TestIndexingRequest:
    """Test IndexingRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = IndexingRequest(
            collection_name="test",
            field_name="category",
        )
        assert request.collection_name == "test"
        assert request.field_name == "category"

    def test_default_values(self):
        """Test default field values."""
        request = IndexingRequest(
            collection_name="test",
            field_name="category",
        )
        assert request.field_type == "keyword"
        assert request.wait is True

    def test_custom_values(self):
        """Test custom field values."""
        request = IndexingRequest(
            collection_name="test",
            field_name="price",
            field_type="float",
            wait=False,
        )
        assert request.field_type == "float"
        assert request.wait is False


class TestCollectionStats:
    """Test CollectionStats model."""

    def test_required_fields(self):
        """Test required fields."""
        stats = CollectionStats(name="test_collection")
        assert stats.name == "test_collection"

    def test_default_values(self):
        """Test default field values."""
        stats = CollectionStats(name="test")
        assert stats.points_count == 0
        assert stats.vectors_count == 0
        assert stats.indexed_fields == []
        assert stats.status == "unknown"
        assert stats.config == {}

    def test_with_data(self):
        """Test stats with data."""
        config = {"vector_size": 384, "distance": "Cosine"}
        stats = CollectionStats(
            name="test",
            points_count=1000,
            vectors_count=1000,
            indexed_fields=["category", "status", "price"],
            status="green",
            config=config,
        )
        assert stats.points_count == 1000
        assert stats.vectors_count == 1000
        assert len(stats.indexed_fields) == 3
        assert stats.status == "green"
        assert stats.config == config


class TestOptimizationRequest:
    """Test OptimizationRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = OptimizationRequest(collection_name="test")
        assert request.collection_name == "test"

    def test_default_values(self):
        """Test default field values."""
        request = OptimizationRequest(collection_name="test")
        assert request.optimization_type == "auto"
        assert request.force is False

    def test_custom_values(self):
        """Test custom field values."""
        request = OptimizationRequest(
            collection_name="test",
            optimization_type="hnsw",
            force=True,
        )
        assert request.optimization_type == "hnsw"
        assert request.force is True


class TestVectorSearchConfig:
    """Test VectorSearchConfig model."""

    def test_default_values(self):
        """Test default field values."""
        config = VectorSearchConfig()
        assert isinstance(config.default_prefetch, PrefetchConfig)
        assert isinstance(config.default_search_params, SearchParams)
        assert isinstance(config.default_fusion, FusionConfig)
        assert config.enable_metrics is True
        assert config.enable_adaptive_search is True
        assert config.cache_search_results is True
        assert config.result_cache_ttl == 300

    def test_custom_config(self):
        """Test custom configuration."""
        prefetch = PrefetchConfig(sparse_multiplier=6.0)
        search_params = SearchParams(accuracy_level=SearchAccuracy.FAST)
        fusion = FusionConfig(algorithm=FusionAlgorithm.DBSF)

        config = VectorSearchConfig(
            default_prefetch=prefetch,
            default_search_params=search_params,
            default_fusion=fusion,
            enable_metrics=False,
            enable_adaptive_search=False,
            cache_search_results=False,
            result_cache_ttl=600,
        )
        assert config.default_prefetch.sparse_multiplier == 6.0
        assert config.default_search_params.accuracy_level == SearchAccuracy.FAST
        assert config.default_fusion.algorithm == FusionAlgorithm.DBSF
        assert config.enable_metrics is False
        assert config.enable_adaptive_search is False
        assert config.cache_search_results is False
        assert config.result_cache_ttl == 600
