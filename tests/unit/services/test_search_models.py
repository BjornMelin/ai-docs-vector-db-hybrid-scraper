"""Tests for search models."""

from src.config.enums import FusionAlgorithm
from src.config.enums import SearchAccuracy
from src.config.enums import VectorType
from src.services.utilities.search_models import FilteredSearchRequest
from src.services.utilities.search_models import FusionConfig
from src.services.utilities.search_models import HyDESearchRequest
from src.services.utilities.search_models import MultiStageSearchRequest
from src.services.utilities.search_models import PrefetchConfig
from src.services.utilities.search_models import SearchParams
from src.services.utilities.search_models import SearchStage


class TestSearchStage:
    """Test SearchStage model."""

    def test_search_stage_creation(self):
        """Test creating a search stage."""
        stage = SearchStage(
            query_vector=[0.1] * 384,
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=50,
            filter={"category": "docs"},
            search_params={"hnsw_ef": 100},
        )

        assert stage.vector_name == "dense"
        assert stage.vector_type == VectorType.DENSE
        assert stage.limit == 50
        assert stage.filter == {"category": "docs"}
        assert stage.search_params == {"hnsw_ef": 100}

    def test_search_stage_minimal(self):
        """Test creating search stage with minimal params."""
        stage = SearchStage(
            query_vector=[0.1] * 384,
            vector_name="sparse",
            vector_type=VectorType.SPARSE,
            limit=100,
        )

        assert stage.filter is None
        assert stage.search_params is None


class TestPrefetchConfig:
    """Test PrefetchConfig model."""

    def test_default_prefetch_config(self):
        """Test default prefetch configuration."""
        config = PrefetchConfig()

        assert config.sparse_multiplier == 5.0
        assert config.hyde_multiplier == 3.0
        assert config.dense_multiplier == 2.0
        assert config.max_sparse_limit == 500
        assert config.max_dense_limit == 200
        assert config.max_hyde_limit == 150

    def test_calculate_prefetch_limit_sparse(self):
        """Test calculating prefetch limit for sparse vectors."""
        config = PrefetchConfig()

        # Normal case
        limit = config.calculate_prefetch_limit(VectorType.SPARSE, 50)
        assert limit == 250  # 50 * 5.0

        # Exceeds max limit
        limit = config.calculate_prefetch_limit(VectorType.SPARSE, 200)
        assert limit == 500  # Capped at max_sparse_limit

    def test_calculate_prefetch_limit_hyde(self):
        """Test calculating prefetch limit for HyDE vectors."""
        config = PrefetchConfig()

        # Normal case
        limit = config.calculate_prefetch_limit(VectorType.HYDE, 30)
        assert limit == 90  # 30 * 3.0

        # Exceeds max limit
        limit = config.calculate_prefetch_limit(VectorType.HYDE, 100)
        assert limit == 150  # Capped at max_hyde_limit

    def test_calculate_prefetch_limit_dense(self):
        """Test calculating prefetch limit for dense vectors."""
        config = PrefetchConfig()

        # Normal case
        limit = config.calculate_prefetch_limit(VectorType.DENSE, 40)
        assert limit == 80  # 40 * 2.0

        # Exceeds max limit
        limit = config.calculate_prefetch_limit(VectorType.DENSE, 150)
        assert limit == 200  # Capped at max_dense_limit

    def test_custom_prefetch_config(self):
        """Test custom prefetch configuration."""
        config = PrefetchConfig(
            sparse_multiplier=6.0,
            hyde_multiplier=4.0,
            dense_multiplier=3.0,
            max_sparse_limit=600,
            max_dense_limit=300,
            max_hyde_limit=200,
        )

        assert config.calculate_prefetch_limit(VectorType.SPARSE, 50) == 300
        assert config.calculate_prefetch_limit(VectorType.HYDE, 40) == 160
        assert config.calculate_prefetch_limit(VectorType.DENSE, 60) == 180


class TestSearchParams:
    """Test SearchParams model."""

    def test_default_search_params(self):
        """Test default search parameters."""
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
        assert params.exact is True
        assert params.hnsw_ef is None


class TestFusionConfig:
    """Test FusionConfig model."""

    def test_default_fusion_config(self):
        """Test default fusion configuration."""
        config = FusionConfig()

        assert config.algorithm == FusionAlgorithm.RRF
        assert config.auto_select is True

    def test_select_fusion_algorithm_hybrid(self):
        """Test selecting fusion algorithm for hybrid query."""
        algo = FusionConfig.select_fusion_algorithm("hybrid")
        assert algo == FusionAlgorithm.RRF

    def test_select_fusion_algorithm_multi_stage(self):
        """Test selecting fusion algorithm for multi-stage query."""
        algo = FusionConfig.select_fusion_algorithm("multi_stage")
        assert algo == FusionAlgorithm.RRF

    def test_select_fusion_algorithm_reranking(self):
        """Test selecting fusion algorithm for reranking query."""
        algo = FusionConfig.select_fusion_algorithm("reranking")
        assert algo == FusionAlgorithm.DBSF

    def test_select_fusion_algorithm_hyde(self):
        """Test selecting fusion algorithm for HyDE query."""
        algo = FusionConfig.select_fusion_algorithm("hyde")
        assert algo == FusionAlgorithm.RRF

    def test_select_fusion_algorithm_unknown(self):
        """Test selecting fusion algorithm for unknown query type."""
        algo = FusionConfig.select_fusion_algorithm("unknown")
        assert algo == FusionAlgorithm.RRF  # Default


class TestMultiStageSearchRequest:
    """Test MultiStageSearchRequest model."""

    def test_multi_stage_search_request(self):
        """Test creating multi-stage search request."""
        stages = [
            SearchStage(
                query_vector=[0.1] * 384,
                vector_name="dense",
                vector_type=VectorType.DENSE,
                limit=50,
            ),
            SearchStage(
                query_vector=[0.2] * 384,
                vector_name="sparse",
                vector_type=VectorType.SPARSE,
                limit=100,
            ),
        ]

        request = MultiStageSearchRequest(
            collection_name="test_collection",
            stages=stages,
            limit=20,
            score_threshold=0.5,
        )

        assert request.collection_name == "test_collection"
        assert len(request.stages) == 2
        assert request.limit == 20
        assert request.score_threshold == 0.5
        assert isinstance(request.fusion_config, FusionConfig)
        assert isinstance(request.search_params, SearchParams)

    def test_multi_stage_search_request_custom_config(self):
        """Test creating request with custom configurations."""
        stage = SearchStage(
            query_vector=[0.1] * 384,
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=50,
        )

        fusion_config = FusionConfig(algorithm=FusionAlgorithm.DBSF, auto_select=False)

        search_params = SearchParams.from_accuracy_level(SearchAccuracy.ACCURATE)

        request = MultiStageSearchRequest(
            collection_name="test_collection",
            stages=[stage],
            fusion_config=fusion_config,
            search_params=search_params,
            limit=10,
        )

        assert request.fusion_config.algorithm == FusionAlgorithm.DBSF
        assert request.search_params.hnsw_ef == 200


class TestHyDESearchRequest:
    """Test HyDESearchRequest model."""

    def test_hyde_search_request_default(self):
        """Test creating HyDE search request with defaults."""
        request = HyDESearchRequest(
            collection_name="test_collection", query="What is machine learning?"
        )

        assert request.collection_name == "test_collection"
        assert request.query == "What is machine learning?"
        assert request.num_hypothetical_docs == 5
        assert request.limit == 10
        assert isinstance(request.fusion_config, FusionConfig)
        assert isinstance(request.search_params, SearchParams)

    def test_hyde_search_request_custom(self):
        """Test creating HyDE search request with custom params."""
        fusion_config = FusionConfig(algorithm=FusionAlgorithm.DBSF)
        search_params = SearchParams.from_accuracy_level(SearchAccuracy.FAST)

        request = HyDESearchRequest(
            collection_name="test_collection",
            query="How does neural network work?",
            num_hypothetical_docs=10,
            limit=20,
            fusion_config=fusion_config,
            search_params=search_params,
        )

        assert request.num_hypothetical_docs == 10
        assert request.limit == 20
        assert request.fusion_config.algorithm == FusionAlgorithm.DBSF
        assert request.search_params.hnsw_ef == 50


class TestFilteredSearchRequest:
    """Test FilteredSearchRequest model."""

    def test_filtered_search_request_basic(self):
        """Test creating filtered search request."""
        request = FilteredSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1] * 384,
            filters={"category": "docs", "language": "en"},
            limit=20,
        )

        assert request.collection_name == "test_collection"
        assert len(request.query_vector) == 384
        assert request.filters == {"category": "docs", "language": "en"}
        assert request.limit == 20
        assert request.score_threshold == 0.0
        assert isinstance(request.search_params, SearchParams)

    def test_filtered_search_request_with_threshold(self):
        """Test creating filtered search request with score threshold."""
        search_params = SearchParams.from_accuracy_level(SearchAccuracy.EXACT)

        request = FilteredSearchRequest(
            collection_name="test_collection",
            query_vector=[0.2] * 1536,
            filters={"doc_type": "api", "framework": "pytorch"},
            limit=50,
            search_params=search_params,
            score_threshold=0.7,
        )

        assert request.limit == 50
        assert request.score_threshold == 0.7
        assert request.search_params.exact is True
        assert request.filters["doc_type"] == "api"
        assert request.filters["framework"] == "pytorch"
