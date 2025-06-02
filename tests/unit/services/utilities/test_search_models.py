"""Tests for advanced search models."""

import pytest
from pydantic import ValidationError
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
    """Tests for SearchStage model."""

    def test_search_stage_creation_minimal(self):
        """Test SearchStage creation with minimal required fields."""
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
        assert stage.filter is None
        assert stage.search_params is None

    def test_search_stage_creation_complete(self):
        """Test SearchStage creation with all fields."""
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="sparse",
            vector_type=VectorType.SPARSE,
            limit=50,
            filter={"category": "api"},
            search_params={"hnsw_ef": 100},
        )

        assert stage.query_vector == [0.1, 0.2, 0.3]
        assert stage.vector_name == "sparse"
        assert stage.vector_type == VectorType.SPARSE
        assert stage.limit == 50
        assert stage.filter == {"category": "api"}
        assert stage.search_params == {"hnsw_ef": 100}

    def test_search_stage_validation_error_missing_required(self):
        """Test SearchStage validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            SearchStage(
                vector_name="dense",
                vector_type=VectorType.DENSE,
                limit=10,
                # Missing query_vector
            )

        assert "query_vector" in str(exc_info.value)

    def test_search_stage_different_vector_types(self):
        """Test SearchStage with different vector types."""
        # Dense vector
        dense_stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=10,
        )
        assert dense_stage.vector_type == VectorType.DENSE

        # Sparse vector
        sparse_stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="sparse",
            vector_type=VectorType.SPARSE,
            limit=10,
        )
        assert sparse_stage.vector_type == VectorType.SPARSE

        # HyDE vector
        hyde_stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="hyde",
            vector_type=VectorType.HYDE,
            limit=10,
        )
        assert hyde_stage.vector_type == VectorType.HYDE

    def test_search_stage_serialization(self):
        """Test SearchStage serialization and deserialization."""
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=20,
            filter={"type": "document"},
        )

        # Test serialization
        stage_dict = stage.model_dump()
        assert stage_dict["query_vector"] == [0.1, 0.2, 0.3]
        assert stage_dict["vector_name"] == "dense"
        assert stage_dict["vector_type"] == "dense"
        assert stage_dict["limit"] == 20
        assert stage_dict["filter"] == {"type": "document"}

        # Test deserialization
        restored_stage = SearchStage.model_validate(stage_dict)
        assert restored_stage.query_vector == [0.1, 0.2, 0.3]
        assert restored_stage.vector_name == "dense"
        assert restored_stage.vector_type == VectorType.DENSE
        assert restored_stage.limit == 20
        assert restored_stage.filter == {"type": "document"}


class TestPrefetchConfig:
    """Tests for PrefetchConfig model."""

    def test_prefetch_config_defaults(self):
        """Test PrefetchConfig with default values."""
        config = PrefetchConfig()

        assert config.sparse_multiplier == 5.0
        assert config.hyde_multiplier == 3.0
        assert config.dense_multiplier == 2.0
        assert config.max_sparse_limit == 500
        assert config.max_dense_limit == 200
        assert config.max_hyde_limit == 150

    def test_prefetch_config_custom_values(self):
        """Test PrefetchConfig with custom values."""
        config = PrefetchConfig(
            sparse_multiplier=4.0,
            hyde_multiplier=2.5,
            dense_multiplier=1.8,
            max_sparse_limit=400,
            max_dense_limit=180,
            max_hyde_limit=120,
        )

        assert config.sparse_multiplier == 4.0
        assert config.hyde_multiplier == 2.5
        assert config.dense_multiplier == 1.8
        assert config.max_sparse_limit == 400
        assert config.max_dense_limit == 180
        assert config.max_hyde_limit == 120

    def test_calculate_prefetch_limit_sparse(self):
        """Test prefetch limit calculation for sparse vectors."""
        config = PrefetchConfig()

        # Normal case
        limit = config.calculate_prefetch_limit(VectorType.SPARSE, 50)
        assert limit == 250  # 50 * 5.0

        # Capped case
        limit = config.calculate_prefetch_limit(VectorType.SPARSE, 200)
        assert limit == 500  # Capped at max_sparse_limit

    def test_calculate_prefetch_limit_hyde(self):
        """Test prefetch limit calculation for HyDE vectors."""
        config = PrefetchConfig()

        # Normal case
        limit = config.calculate_prefetch_limit(VectorType.HYDE, 30)
        assert limit == 90  # 30 * 3.0

        # Capped case
        limit = config.calculate_prefetch_limit(VectorType.HYDE, 60)
        assert limit == 150  # Capped at max_hyde_limit

    def test_calculate_prefetch_limit_dense(self):
        """Test prefetch limit calculation for dense vectors."""
        config = PrefetchConfig()

        # Normal case
        limit = config.calculate_prefetch_limit(VectorType.DENSE, 40)
        assert limit == 80  # 40 * 2.0

        # Capped case
        limit = config.calculate_prefetch_limit(VectorType.DENSE, 150)
        assert limit == 200  # Capped at max_dense_limit

    def test_calculate_prefetch_limit_edge_cases(self):
        """Test prefetch limit calculation edge cases."""
        config = PrefetchConfig()

        # Zero final limit
        limit = config.calculate_prefetch_limit(VectorType.DENSE, 0)
        assert limit == 0

        # Small final limit
        limit = config.calculate_prefetch_limit(VectorType.SPARSE, 1)
        assert limit == 5  # 1 * 5.0

    def test_calculate_prefetch_limit_custom_multipliers(self):
        """Test prefetch limit calculation with custom multipliers."""
        config = PrefetchConfig(
            sparse_multiplier=10.0,
            hyde_multiplier=5.0,
            dense_multiplier=3.0,
        )

        sparse_limit = config.calculate_prefetch_limit(VectorType.SPARSE, 20)
        assert sparse_limit == 200  # 20 * 10.0

        hyde_limit = config.calculate_prefetch_limit(VectorType.HYDE, 20)
        assert hyde_limit == 100  # 20 * 5.0

        dense_limit = config.calculate_prefetch_limit(VectorType.DENSE, 20)
        assert dense_limit == 60  # 20 * 3.0


class TestSearchParams:
    """Tests for SearchParams model."""

    def test_search_params_defaults(self):
        """Test SearchParams with default values."""
        params = SearchParams()

        assert params.accuracy_level == SearchAccuracy.BALANCED
        assert params.hnsw_ef is None
        assert params.exact is False

    def test_search_params_custom_values(self):
        """Test SearchParams with custom values."""
        params = SearchParams(
            accuracy_level=SearchAccuracy.ACCURATE,
            hnsw_ef=200,
            exact=True,
        )

        assert params.accuracy_level == SearchAccuracy.ACCURATE
        assert params.hnsw_ef == 200
        assert params.exact is True

    def test_search_params_from_accuracy_level_fast(self):
        """Test SearchParams creation from FAST accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.FAST)

        assert params.accuracy_level == SearchAccuracy.FAST
        assert params.hnsw_ef == 50
        assert params.exact is False

    def test_search_params_from_accuracy_level_balanced(self):
        """Test SearchParams creation from BALANCED accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.BALANCED)

        assert params.accuracy_level == SearchAccuracy.BALANCED
        assert params.hnsw_ef == 100
        assert params.exact is False

    def test_search_params_from_accuracy_level_accurate(self):
        """Test SearchParams creation from ACCURATE accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.ACCURATE)

        assert params.accuracy_level == SearchAccuracy.ACCURATE
        assert params.hnsw_ef == 200
        assert params.exact is False

    def test_search_params_from_accuracy_level_exact(self):
        """Test SearchParams creation from EXACT accuracy level."""
        params = SearchParams.from_accuracy_level(SearchAccuracy.EXACT)

        assert params.accuracy_level == SearchAccuracy.EXACT
        assert params.hnsw_ef is None  # Not relevant for exact search
        assert params.exact is True

    def test_search_params_serialization(self):
        """Test SearchParams serialization and deserialization."""
        params = SearchParams(
            accuracy_level=SearchAccuracy.ACCURATE,
            hnsw_ef=150,
            exact=False,
        )

        # Test serialization
        params_dict = params.model_dump()
        assert params_dict["accuracy_level"] == "accurate"
        assert params_dict["hnsw_ef"] == 150
        assert params_dict["exact"] is False

        # Test deserialization
        restored_params = SearchParams.model_validate(params_dict)
        assert restored_params.accuracy_level == SearchAccuracy.ACCURATE
        assert restored_params.hnsw_ef == 150
        assert restored_params.exact is False


class TestFusionConfig:
    """Tests for FusionConfig model."""

    def test_fusion_config_defaults(self):
        """Test FusionConfig with default values."""
        config = FusionConfig()

        assert config.algorithm == FusionAlgorithm.RRF
        assert config.auto_select is True

    def test_fusion_config_custom_values(self):
        """Test FusionConfig with custom values."""
        config = FusionConfig(
            algorithm=FusionAlgorithm.DBSF,
            auto_select=False,
        )

        assert config.algorithm == FusionAlgorithm.DBSF
        assert config.auto_select is False

    def test_select_fusion_algorithm_hybrid(self):
        """Test fusion algorithm selection for hybrid queries."""
        algorithm = FusionConfig.select_fusion_algorithm("hybrid")
        assert algorithm == FusionAlgorithm.RRF

    def test_select_fusion_algorithm_multi_stage(self):
        """Test fusion algorithm selection for multi-stage queries."""
        algorithm = FusionConfig.select_fusion_algorithm("multi_stage")
        assert algorithm == FusionAlgorithm.RRF

    def test_select_fusion_algorithm_reranking(self):
        """Test fusion algorithm selection for reranking queries."""
        algorithm = FusionConfig.select_fusion_algorithm("reranking")
        assert algorithm == FusionAlgorithm.DBSF

    def test_select_fusion_algorithm_hyde(self):
        """Test fusion algorithm selection for HyDE queries."""
        algorithm = FusionConfig.select_fusion_algorithm("hyde")
        assert algorithm == FusionAlgorithm.RRF

    def test_select_fusion_algorithm_unknown(self):
        """Test fusion algorithm selection for unknown query type."""
        algorithm = FusionConfig.select_fusion_algorithm("unknown_type")
        assert algorithm == FusionAlgorithm.RRF  # Default

    def test_fusion_config_serialization(self):
        """Test FusionConfig serialization."""
        config = FusionConfig(
            algorithm=FusionAlgorithm.DBSF,
            auto_select=False,
        )

        config_dict = config.model_dump()
        assert config_dict["algorithm"] == "dbsf"
        assert config_dict["auto_select"] is False

        # Test deserialization
        restored_config = FusionConfig.model_validate(config_dict)
        assert restored_config.algorithm == FusionAlgorithm.DBSF
        assert restored_config.auto_select is False


class TestMultiStageSearchRequest:
    """Tests for MultiStageSearchRequest model."""

    def test_multi_stage_search_request_minimal(self):
        """Test MultiStageSearchRequest with minimal required fields."""
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=10,
        )

        request = MultiStageSearchRequest(
            collection_name="test_collection",
            stages=[stage],
        )

        assert request.collection_name == "test_collection"
        assert len(request.stages) == 1
        assert request.stages[0] == stage
        assert isinstance(request.fusion_config, FusionConfig)
        assert isinstance(request.search_params, SearchParams)
        assert request.limit == 10
        assert request.score_threshold == 0.0

    def test_multi_stage_search_request_complete(self):
        """Test MultiStageSearchRequest with all fields."""
        stage1 = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=50,
        )
        stage2 = SearchStage(
            query_vector=[0.4, 0.5, 0.6],
            vector_name="sparse",
            vector_type=VectorType.SPARSE,
            limit=100,
        )

        fusion_config = FusionConfig(algorithm=FusionAlgorithm.DBSF)
        search_params = SearchParams(accuracy_level=SearchAccuracy.ACCURATE)

        request = MultiStageSearchRequest(
            collection_name="test_collection",
            stages=[stage1, stage2],
            fusion_config=fusion_config,
            search_params=search_params,
            limit=20,
            score_threshold=0.5,
        )

        assert request.collection_name == "test_collection"
        assert len(request.stages) == 2
        assert request.fusion_config == fusion_config
        assert request.search_params == search_params
        assert request.limit == 20
        assert request.score_threshold == 0.5

    def test_multi_stage_search_request_validation_error(self):
        """Test MultiStageSearchRequest validation errors."""
        with pytest.raises(ValidationError):
            MultiStageSearchRequest(
                collection_name="test_collection",
                # Missing required stages field
            )

    def test_multi_stage_search_request_serialization(self):
        """Test MultiStageSearchRequest serialization."""
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=10,
        )

        request = MultiStageSearchRequest(
            collection_name="test_collection",
            stages=[stage],
            limit=15,
            score_threshold=0.3,
        )

        request_dict = request.model_dump()
        assert request_dict["collection_name"] == "test_collection"
        assert len(request_dict["stages"]) == 1
        assert request_dict["limit"] == 15
        assert request_dict["score_threshold"] == 0.3

        # Test deserialization
        restored_request = MultiStageSearchRequest.model_validate(request_dict)
        assert restored_request.collection_name == "test_collection"
        assert len(restored_request.stages) == 1
        assert restored_request.limit == 15
        assert restored_request.score_threshold == 0.3


class TestHyDESearchRequest:
    """Tests for HyDESearchRequest model."""

    def test_hyde_search_request_minimal(self):
        """Test HyDESearchRequest with minimal required fields."""
        request = HyDESearchRequest(
            collection_name="test_collection",
            query="What is machine learning?",
        )

        assert request.collection_name == "test_collection"
        assert request.query == "What is machine learning?"
        assert request.num_hypothetical_docs == 5  # Default
        assert request.limit == 10  # Default
        assert isinstance(request.fusion_config, FusionConfig)
        assert isinstance(request.search_params, SearchParams)

    def test_hyde_search_request_complete(self):
        """Test HyDESearchRequest with all fields."""
        fusion_config = FusionConfig(algorithm=FusionAlgorithm.RRF)
        search_params = SearchParams(accuracy_level=SearchAccuracy.FAST)

        request = HyDESearchRequest(
            collection_name="test_collection",
            query="Explain neural networks",
            num_hypothetical_docs=3,
            limit=20,
            fusion_config=fusion_config,
            search_params=search_params,
        )

        assert request.collection_name == "test_collection"
        assert request.query == "Explain neural networks"
        assert request.num_hypothetical_docs == 3
        assert request.limit == 20
        assert request.fusion_config == fusion_config
        assert request.search_params == search_params

    def test_hyde_search_request_validation_error(self):
        """Test HyDESearchRequest validation errors."""
        with pytest.raises(ValidationError):
            HyDESearchRequest(
                collection_name="test_collection",
                # Missing required query field
            )

    def test_hyde_search_request_serialization(self):
        """Test HyDESearchRequest serialization."""
        request = HyDESearchRequest(
            collection_name="test_collection",
            query="How does backpropagation work?",
            num_hypothetical_docs=7,
            limit=25,
        )

        request_dict = request.model_dump()
        assert request_dict["collection_name"] == "test_collection"
        assert request_dict["query"] == "How does backpropagation work?"
        assert request_dict["num_hypothetical_docs"] == 7
        assert request_dict["limit"] == 25

        # Test deserialization
        restored_request = HyDESearchRequest.model_validate(request_dict)
        assert restored_request.collection_name == "test_collection"
        assert restored_request.query == "How does backpropagation work?"
        assert restored_request.num_hypothetical_docs == 7
        assert restored_request.limit == 25


class TestFilteredSearchRequest:
    """Tests for FilteredSearchRequest model."""

    def test_filtered_search_request_minimal(self):
        """Test FilteredSearchRequest with minimal required fields."""
        request = FilteredSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            filters={"category": "api"},
        )

        assert request.collection_name == "test_collection"
        assert request.query_vector == [0.1, 0.2, 0.3]
        assert request.filters == {"category": "api"}
        assert request.limit == 10  # Default
        assert isinstance(request.search_params, SearchParams)
        assert request.score_threshold == 0.0  # Default

    def test_filtered_search_request_complete(self):
        """Test FilteredSearchRequest with all fields."""
        search_params = SearchParams(accuracy_level=SearchAccuracy.EXACT)

        request = FilteredSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3, 0.4],
            filters={"category": "api", "type": "function"},
            limit=50,
            search_params=search_params,
            score_threshold=0.7,
        )

        assert request.collection_name == "test_collection"
        assert request.query_vector == [0.1, 0.2, 0.3, 0.4]
        assert request.filters == {"category": "api", "type": "function"}
        assert request.limit == 50
        assert request.search_params == search_params
        assert request.score_threshold == 0.7

    def test_filtered_search_request_validation_error(self):
        """Test FilteredSearchRequest validation errors."""
        with pytest.raises(ValidationError):
            FilteredSearchRequest(
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3],
                # Missing required filters field
            )

    def test_filtered_search_request_complex_filters(self):
        """Test FilteredSearchRequest with complex filters."""
        complex_filters = {
            "must": [
                {"key": "category", "match": {"value": "api"}},
                {"key": "language", "match": {"value": "python"}},
            ],
            "should": [
                {"key": "difficulty", "match": {"value": "beginner"}},
            ],
        }

        request = FilteredSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            filters=complex_filters,
        )

        assert request.filters == complex_filters

    def test_filtered_search_request_serialization(self):
        """Test FilteredSearchRequest serialization."""
        request = FilteredSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            filters={"category": "tutorial"},
            limit=30,
            score_threshold=0.4,
        )

        request_dict = request.model_dump()
        assert request_dict["collection_name"] == "test_collection"
        assert request_dict["query_vector"] == [0.1, 0.2, 0.3]
        assert request_dict["filters"] == {"category": "tutorial"}
        assert request_dict["limit"] == 30
        assert request_dict["score_threshold"] == 0.4

        # Test deserialization
        restored_request = FilteredSearchRequest.model_validate(request_dict)
        assert restored_request.collection_name == "test_collection"
        assert restored_request.query_vector == [0.1, 0.2, 0.3]
        assert restored_request.filters == {"category": "tutorial"}
        assert restored_request.limit == 30
        assert restored_request.score_threshold == 0.4


class TestSearchModelsIntegration:
    """Integration tests for search models."""

    def test_complete_multi_stage_workflow(self):
        """Test complete multi-stage search workflow."""
        # Create stages
        dense_stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=50,
            filter={"type": "document"},
        )

        sparse_stage = SearchStage(
            query_vector=[1.0, 2.0, 3.0],
            vector_name="sparse",
            vector_type=VectorType.SPARSE,
            limit=100,
            filter={"category": "api"},
        )

        # Create fusion config
        fusion_config = FusionConfig(
            algorithm=FusionAlgorithm.RRF,
            auto_select=False,
        )

        # Create search params
        search_params = SearchParams.from_accuracy_level(SearchAccuracy.BALANCED)

        # Create request
        request = MultiStageSearchRequest(
            collection_name="documentation",
            stages=[dense_stage, sparse_stage],
            fusion_config=fusion_config,
            search_params=search_params,
            limit=25,
            score_threshold=0.6,
        )

        # Verify all components work together
        assert len(request.stages) == 2
        assert request.stages[0].vector_type == VectorType.DENSE
        assert request.stages[1].vector_type == VectorType.SPARSE
        assert request.fusion_config.algorithm == FusionAlgorithm.RRF
        assert request.search_params.hnsw_ef == 100
        assert request.limit == 25

    def test_search_models_serialization_round_trip(self):
        """Test serialization round trip for all search models."""
        # Test SearchStage
        stage = SearchStage(
            query_vector=[0.1, 0.2, 0.3],
            vector_name="dense",
            vector_type=VectorType.DENSE,
            limit=10,
        )
        stage_dict = stage.model_dump()
        restored_stage = SearchStage.model_validate(stage_dict)
        assert restored_stage.query_vector == stage.query_vector

        # Test PrefetchConfig
        prefetch_config = PrefetchConfig(sparse_multiplier=4.0)
        prefetch_dict = prefetch_config.model_dump()
        restored_prefetch = PrefetchConfig.model_validate(prefetch_dict)
        assert restored_prefetch.sparse_multiplier == 4.0

        # Test SearchParams
        search_params = SearchParams(accuracy_level=SearchAccuracy.FAST)
        params_dict = search_params.model_dump()
        restored_params = SearchParams.model_validate(params_dict)
        assert restored_params.accuracy_level == SearchAccuracy.FAST

        # Test FusionConfig
        fusion_config = FusionConfig(algorithm=FusionAlgorithm.DBSF)
        fusion_dict = fusion_config.model_dump()
        restored_fusion = FusionConfig.model_validate(fusion_dict)
        assert restored_fusion.algorithm == FusionAlgorithm.DBSF

    def test_prefetch_config_with_all_vector_types(self):
        """Test PrefetchConfig calculations with all vector types."""
        config = PrefetchConfig()
        final_limit = 40

        # Test all vector types
        dense_limit = config.calculate_prefetch_limit(VectorType.DENSE, final_limit)
        sparse_limit = config.calculate_prefetch_limit(VectorType.SPARSE, final_limit)
        hyde_limit = config.calculate_prefetch_limit(VectorType.HYDE, final_limit)

        # Verify relationships
        assert sparse_limit > hyde_limit > dense_limit
        assert dense_limit == 80  # 40 * 2.0
        assert hyde_limit == 120  # 40 * 3.0
        assert sparse_limit == 200  # 40 * 5.0

    def test_search_accuracy_to_params_mapping(self):
        """Test mapping from search accuracy to parameters."""
        # Test all accuracy levels
        fast_params = SearchParams.from_accuracy_level(SearchAccuracy.FAST)
        balanced_params = SearchParams.from_accuracy_level(SearchAccuracy.BALANCED)
        accurate_params = SearchParams.from_accuracy_level(SearchAccuracy.ACCURATE)
        exact_params = SearchParams.from_accuracy_level(SearchAccuracy.EXACT)

        # Verify ef values increase with accuracy (except exact)
        assert fast_params.hnsw_ef < balanced_params.hnsw_ef < accurate_params.hnsw_ef
        assert exact_params.exact is True
        assert not fast_params.exact
        assert not balanced_params.exact
        assert not accurate_params.exact
