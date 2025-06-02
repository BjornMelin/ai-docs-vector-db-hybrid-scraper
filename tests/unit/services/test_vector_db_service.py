"""Tests for QdrantService facade."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.service import QdrantService


class TestQdrantService:
    """Test cases for QdrantService facade."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=UnifiedConfig)
        config.qdrant = MagicMock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = "test-api-key"
        config.qdrant.timeout = 30
        config.qdrant.prefer_grpc = False
        return config

    @pytest.fixture
    def service(self, mock_config):
        """Create QdrantService instance."""
        return QdrantService(mock_config)

    @pytest.fixture
    def initialized_service(self, service):
        """Create initialized QdrantService with mocked modules."""
        # Mock all the internal modules
        service._client_manager = AsyncMock()
        service._client_manager.get_client.return_value = AsyncMock()
        service._collections = AsyncMock()
        service._search = AsyncMock()
        service._indexing = AsyncMock()
        service._documents = AsyncMock()
        service._initialized = True
        return service

    async def test_initialization_success(self, mock_config):
        """Test successful service initialization."""
        with (
            patch("src.services.vector_db.service.QdrantClient") as mock_client_class,
            patch(
                "src.services.vector_db.service.QdrantCollections"
            ) as mock_collections_class,
            patch("src.services.vector_db.service.QdrantSearch") as mock_search_class,
            patch(
                "src.services.vector_db.service.QdrantIndexing"
            ) as mock_indexing_class,
            patch(
                "src.services.vector_db.service.QdrantDocuments"
            ) as mock_documents_class,
        ):
            # Mock client manager
            mock_client_manager = AsyncMock()
            mock_client = AsyncMock()
            mock_client_manager.get_client.return_value = mock_client
            mock_client_class.return_value = mock_client_manager

            # Mock modules
            mock_collections = AsyncMock()
            mock_collections_class.return_value = mock_collections

            # Create service AFTER patching so it gets the mocked client
            service = QdrantService(mock_config)
            await service.initialize()

            assert service._initialized is True
            mock_client_manager.initialize.assert_called_once()
            mock_collections.initialize.assert_called_once()

    async def test_initialization_already_initialized(self, service):
        """Test initialization when already initialized."""
        service._initialized = True
        await service.initialize()
        # Should return early without creating new modules

    async def test_initialization_error(self, mock_config):
        """Test initialization error."""
        with patch("src.services.vector_db.service.QdrantClient") as mock_client_class:
            # Mock client manager that fails during initialize
            mock_client_manager = AsyncMock()
            mock_client_manager.initialize.side_effect = Exception(
                "Client initialization failed"
            )
            mock_client_class.return_value = mock_client_manager

            # Create service and try to initialize
            service = QdrantService(mock_config)

            with pytest.raises(
                QdrantServiceError, match="Failed to initialize QdrantService"
            ):
                await service.initialize()

            assert service._initialized is False

    async def test_cleanup_success(self, initialized_service):
        """Test successful cleanup."""
        # Store references before cleanup (cleanup sets them to None)
        collections_mock = initialized_service._collections
        client_manager_mock = initialized_service._client_manager

        await initialized_service.cleanup()

        collections_mock.cleanup.assert_called_once()
        client_manager_mock.cleanup.assert_called_once()
        assert initialized_service._initialized is False

    async def test_cleanup_partial_initialization(self, service):
        """Test cleanup with partial initialization."""
        collections_mock = AsyncMock()
        service._collections = collections_mock
        service._client_manager = None

        await service.cleanup()

        collections_mock.cleanup.assert_called_once()

    async def test_create_collection_delegation(self, initialized_service):
        """Test create_collection delegates to collections module."""
        initialized_service._collections.create_collection.return_value = True

        result = await initialized_service.create_collection(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
            collection_type="general",
        )

        assert result is True
        initialized_service._collections.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
            collection_type="general",
        )

    async def test_delete_collection_delegation(self, initialized_service):
        """Test delete_collection delegates to collections module."""
        initialized_service._collections.delete_collection.return_value = True

        result = await initialized_service.delete_collection("test_collection")

        assert result is True
        initialized_service._collections.delete_collection.assert_called_once_with(
            "test_collection"
        )

    async def test_list_collections_delegation(self, initialized_service):
        """Test list_collections delegates to collections module."""
        expected_collections = ["collection1", "collection2"]
        initialized_service._collections.list_collections.return_value = (
            expected_collections
        )

        result = await initialized_service.list_collections()

        assert result == expected_collections
        initialized_service._collections.list_collections.assert_called_once()

    async def test_list_collections_details_delegation(self, initialized_service):
        """Test list_collections_details delegates to collections module."""
        expected_details = [{"name": "collection1", "vector_count": 100}]
        initialized_service._collections.list_collections_details.return_value = (
            expected_details
        )

        result = await initialized_service.list_collections_details()

        assert result == expected_details
        initialized_service._collections.list_collections_details.assert_called_once()

    async def test_get_collection_info_delegation(self, initialized_service):
        """Test get_collection_info delegates to collections module."""
        expected_info = {"status": "green", "vectors_count": 100}
        initialized_service._collections.get_collection_info.return_value = (
            expected_info
        )

        result = await initialized_service.get_collection_info("test_collection")

        assert result == expected_info
        initialized_service._collections.get_collection_info.assert_called_once_with(
            "test_collection"
        )

    async def test_trigger_collection_optimization_delegation(
        self, initialized_service
    ):
        """Test trigger_collection_optimization delegates to collections module."""
        initialized_service._collections.trigger_collection_optimization.return_value = True

        result = await initialized_service.trigger_collection_optimization(
            "test_collection"
        )

        assert result is True
        initialized_service._collections.trigger_collection_optimization.assert_called_once_with(
            "test_collection"
        )

    async def test_hybrid_search_delegation(self, initialized_service):
        """Test hybrid_search delegates to search module."""
        expected_results = [{"id": "point1", "score": 0.9}]
        initialized_service._search.hybrid_search.return_value = expected_results

        query_vector = [0.1, 0.2, 0.3]
        sparse_vector = {1: 0.5, 2: 0.3}

        result = await initialized_service.hybrid_search(
            collection_name="test_collection",
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
            score_threshold=0.5,
            fusion_type="rrf",
            search_accuracy="balanced",
        )

        assert result == expected_results
        initialized_service._search.hybrid_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
            score_threshold=0.5,
            fusion_type="rrf",
            search_accuracy="balanced",
        )

    async def test_multi_stage_search_delegation(self, initialized_service):
        """Test multi_stage_search delegates to search module."""
        expected_results = [{"id": "point1", "score": 0.9}]
        initialized_service._search.multi_stage_search.return_value = expected_results

        stages = [{"query_vector": [0.1, 0.2], "vector_name": "dense", "limit": 50}]

        result = await initialized_service.multi_stage_search(
            collection_name="test_collection",
            stages=stages,
            limit=10,
            fusion_algorithm="rrf",
            search_accuracy="balanced",
        )

        assert result == expected_results
        initialized_service._search.multi_stage_search.assert_called_once_with(
            collection_name="test_collection",
            stages=stages,
            limit=10,
            fusion_algorithm="rrf",
            search_accuracy="balanced",
        )

    async def test_hyde_search_delegation(self, initialized_service):
        """Test hyde_search delegates to search module."""
        expected_results = [{"id": "point1", "score": 0.9}]
        initialized_service._search.hyde_search.return_value = expected_results

        result = await initialized_service.hyde_search(
            collection_name="test_collection",
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            hypothetical_embeddings=[[0.2, 0.3, 0.4]],
            limit=10,
            fusion_algorithm="rrf",
            search_accuracy="balanced",
        )

        assert result == expected_results
        initialized_service._search.hyde_search.assert_called_once()

    async def test_filtered_search_delegation(self, initialized_service):
        """Test filtered_search delegates to search module."""
        expected_results = [{"id": "point1", "score": 0.9}]
        initialized_service._search.filtered_search.return_value = expected_results

        query_vector = [0.1] * 1536
        filters = {"doc_type": "api"}

        result = await initialized_service.filtered_search(
            collection_name="test_collection",
            query_vector=query_vector,
            filters=filters,
            limit=10,
            search_accuracy="balanced",
        )

        assert result == expected_results
        initialized_service._search.filtered_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            filters=filters,
            limit=10,
            search_accuracy="balanced",
        )

    async def test_create_payload_indexes_delegation(self, initialized_service):
        """Test create_payload_indexes delegates to indexing module."""
        await initialized_service.create_payload_indexes("test_collection")

        initialized_service._indexing.create_payload_indexes.assert_called_once_with(
            "test_collection"
        )

    async def test_list_payload_indexes_delegation(self, initialized_service):
        """Test list_payload_indexes delegates to indexing module."""
        expected_indexes = ["field1", "field2"]
        initialized_service._indexing.list_payload_indexes.return_value = (
            expected_indexes
        )

        result = await initialized_service.list_payload_indexes("test_collection")

        assert result == expected_indexes
        initialized_service._indexing.list_payload_indexes.assert_called_once_with(
            "test_collection"
        )

    async def test_drop_payload_index_delegation(self, initialized_service):
        """Test drop_payload_index delegates to indexing module."""
        await initialized_service.drop_payload_index("test_collection", "test_field")

        initialized_service._indexing.drop_payload_index.assert_called_once_with(
            "test_collection", "test_field"
        )

    async def test_reindex_collection_delegation(self, initialized_service):
        """Test reindex_collection delegates to indexing module."""
        await initialized_service.reindex_collection("test_collection")

        initialized_service._indexing.reindex_collection.assert_called_once_with(
            "test_collection"
        )

    async def test_get_payload_index_stats_delegation(self, initialized_service):
        """Test get_payload_index_stats delegates to indexing module."""
        expected_stats = {
            "collection_name": "test_collection",
            "indexed_fields_count": 5,
        }
        initialized_service._indexing.get_payload_index_stats.return_value = (
            expected_stats
        )

        result = await initialized_service.get_payload_index_stats("test_collection")

        assert result == expected_stats
        initialized_service._indexing.get_payload_index_stats.assert_called_once_with(
            "test_collection"
        )

    async def test_validate_index_health_delegation(self, initialized_service):
        """Test validate_index_health delegates to indexing module."""
        expected_health = {"status": "healthy", "health_score": 95.0}
        initialized_service._indexing.validate_index_health.return_value = (
            expected_health
        )

        result = await initialized_service.validate_index_health("test_collection")

        assert result == expected_health
        initialized_service._indexing.validate_index_health.assert_called_once_with(
            "test_collection"
        )

    async def test_get_index_usage_stats_delegation(self, initialized_service):
        """Test get_index_usage_stats delegates to indexing module."""
        expected_stats = {
            "collection_name": "test_collection",
            "optimization_suggestions": [],
        }
        initialized_service._indexing.get_index_usage_stats.return_value = (
            expected_stats
        )

        result = await initialized_service.get_index_usage_stats("test_collection")

        assert result == expected_stats
        initialized_service._indexing.get_index_usage_stats.assert_called_once_with(
            "test_collection"
        )

    async def test_upsert_points_delegation(self, initialized_service):
        """Test upsert_points delegates to documents module."""
        initialized_service._documents.upsert_points.return_value = True

        points = [{"id": "point1", "vector": [0.1, 0.2], "payload": {"title": "Test"}}]

        result = await initialized_service.upsert_points(
            collection_name="test_collection", points=points, batch_size=100
        )

        assert result is True
        initialized_service._documents.upsert_points.assert_called_once_with(
            collection_name="test_collection", points=points, batch_size=100
        )

    async def test_get_points_delegation(self, initialized_service):
        """Test get_points delegates to documents module."""
        expected_points = [{"id": "point1", "payload": {"title": "Test"}}]
        initialized_service._documents.get_points.return_value = expected_points

        result = await initialized_service.get_points(
            collection_name="test_collection",
            point_ids=["point1", "point2"],
            with_payload=True,
            with_vectors=False,
        )

        assert result == expected_points
        initialized_service._documents.get_points.assert_called_once_with(
            collection_name="test_collection",
            point_ids=["point1", "point2"],
            with_payload=True,
            with_vectors=False,
        )

    async def test_delete_points_delegation(self, initialized_service):
        """Test delete_points delegates to documents module."""
        initialized_service._documents.delete_points.return_value = True

        result = await initialized_service.delete_points(
            collection_name="test_collection",
            point_ids=["point1", "point2"],
            filter_condition=None,
        )

        assert result is True
        initialized_service._documents.delete_points.assert_called_once_with(
            collection_name="test_collection",
            point_ids=["point1", "point2"],
            filter_condition=None,
        )

    async def test_update_point_payload_delegation(self, initialized_service):
        """Test update_point_payload delegates to documents module."""
        initialized_service._documents.update_point_payload.return_value = True

        payload = {"new_field": "new_value"}

        result = await initialized_service.update_point_payload(
            collection_name="test_collection",
            point_id="point1",
            payload=payload,
            replace=False,
        )

        assert result is True
        initialized_service._documents.update_point_payload.assert_called_once_with(
            collection_name="test_collection",
            point_id="point1",
            payload=payload,
            replace=False,
        )

    async def test_count_points_delegation(self, initialized_service):
        """Test count_points delegates to documents module."""
        initialized_service._documents.count_points.return_value = 1000

        result = await initialized_service.count_points(
            collection_name="test_collection",
            filter_condition={"doc_type": "api"},
            exact=True,
        )

        assert result == 1000
        initialized_service._documents.count_points.assert_called_once_with(
            collection_name="test_collection",
            filter_condition={"doc_type": "api"},
            exact=True,
        )

    async def test_scroll_points_delegation(self, initialized_service):
        """Test scroll_points delegates to documents module."""
        expected_result = {"points": [{"id": "point1"}], "next_offset": "offset123"}
        initialized_service._documents.scroll_points.return_value = expected_result

        result = await initialized_service.scroll_points(
            collection_name="test_collection",
            limit=100,
            offset="offset456",
            filter_condition={"doc_type": "api"},
            with_payload=True,
            with_vectors=False,
        )

        assert result == expected_result
        initialized_service._documents.scroll_points.assert_called_once_with(
            collection_name="test_collection",
            limit=100,
            offset="offset456",
            filter_condition={"doc_type": "api"},
            with_payload=True,
            with_vectors=False,
        )

    async def test_clear_collection_delegation(self, initialized_service):
        """Test clear_collection delegates to documents module."""
        initialized_service._documents.clear_collection.return_value = True

        result = await initialized_service.clear_collection("test_collection")

        assert result is True
        initialized_service._documents.clear_collection.assert_called_once_with(
            "test_collection"
        )

    async def test_create_collection_with_hnsw_optimization_delegation(
        self, initialized_service
    ):
        """Test create_collection_with_hnsw_optimization delegates to collections module."""
        initialized_service._collections.create_collection.return_value = True

        result = await initialized_service.create_collection_with_hnsw_optimization(
            collection_name="test_collection",
            vector_size=1536,
            collection_type="api_reference",
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
        )

        assert result is True
        initialized_service._collections.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
            collection_type="api_reference",
        )

    async def test_get_hnsw_configuration_info_delegation(self, initialized_service):
        """Test get_hnsw_configuration_info delegates to collections module."""
        expected_info = {"hnsw_parameters": {"m": 16, "ef_construct": 200}}
        # Replace async mock with regular mock for synchronous method
        from unittest.mock import MagicMock

        sync_collections_mock = MagicMock()
        sync_collections_mock.get_hnsw_configuration_info.return_value = expected_info
        initialized_service._collections = sync_collections_mock

        result = initialized_service.get_hnsw_configuration_info("api_reference")

        assert result == expected_info
        sync_collections_mock.get_hnsw_configuration_info.assert_called_once_with(
            "api_reference"
        )

    async def test_search_with_adaptive_ef_compatibility(self, initialized_service):
        """Test search_with_adaptive_ef compatibility method."""
        expected_search_results = [{"id": "point1", "score": 0.9}]
        initialized_service._search.filtered_search.return_value = (
            expected_search_results
        )

        query_vector = [0.1] * 1536

        result = await initialized_service.search_with_adaptive_ef(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=10,
            time_budget_ms=100,
            score_threshold=0.5,
        )

        assert result["results"] == expected_search_results
        assert result["adaptive_ef_used"] == 100
        assert result["time_budget_ms"] == 100
        assert "actual_time_ms" in result
        assert "filtered_count" in result

    async def test_search_with_adaptive_ef_score_filtering(self, initialized_service):
        """Test search_with_adaptive_ef with score threshold filtering."""
        # Mock results with different scores
        search_results = [
            {"id": "point1", "score": 0.9},
            {"id": "point2", "score": 0.4},  # Below threshold
            {"id": "point3", "score": 0.7},
        ]
        initialized_service._search.filtered_search.return_value = search_results

        query_vector = [0.1] * 1536

        result = await initialized_service.search_with_adaptive_ef(
            collection_name="test_collection",
            query_vector=query_vector,
            score_threshold=0.5,
        )

        # Should filter out point2 with score 0.4
        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "point1"
        assert result["results"][1]["id"] == "point3"

    async def test_optimize_collection_hnsw_parameters_compatibility(
        self, initialized_service
    ):
        """Test optimize_collection_hnsw_parameters compatibility method."""
        expected_config = {"hnsw_parameters": {"m": 16, "ef_construct": 200}}
        # Replace async mock with regular mock for synchronous method
        from unittest.mock import MagicMock

        sync_collections_mock = MagicMock()
        sync_collections_mock.get_hnsw_configuration_info.return_value = expected_config
        initialized_service._collections = sync_collections_mock

        test_queries = [[0.1, 0.2], [0.3, 0.4]]

        result = await initialized_service.optimize_collection_hnsw_parameters(
            collection_name="test_collection",
            collection_type="api_reference",
            test_queries=test_queries,
        )

        assert result["collection_name"] == "test_collection"
        assert result["collection_type"] == "api_reference"
        assert result["current_configuration"] == expected_config["hnsw_parameters"]
        assert result["test_queries_processed"] == 2

    async def test_optimize_collection_hnsw_parameters_no_queries(
        self, initialized_service
    ):
        """Test optimize_collection_hnsw_parameters with no test queries."""
        expected_config = {"hnsw_parameters": {"m": 16, "ef_construct": 200}}
        # Replace async mock with regular mock for synchronous method
        from unittest.mock import MagicMock

        sync_collections_mock = MagicMock()
        sync_collections_mock.get_hnsw_configuration_info.return_value = expected_config
        initialized_service._collections = sync_collections_mock

        result = await initialized_service.optimize_collection_hnsw_parameters(
            collection_name="test_collection", collection_type="general"
        )

        assert result["test_queries_processed"] == 0

    async def test_validate_initialized_not_initialized(self, service):
        """Test validation when service is not initialized."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.create_collection("test", 1536)

    async def test_validate_initialized_no_collections_module(self, service):
        """Test validation when collections module is not available."""
        service._initialized = True
        service._collections = None

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.create_collection("test", 1536)

    async def test_inheritance_from_base_service(self, service):
        """Test that QdrantService inherits from BaseService."""
        from src.services.base import BaseService

        assert isinstance(service, BaseService)

    async def test_config_assignment(self, service, mock_config):
        """Test config is properly assigned."""
        assert service.config is mock_config

    async def test_context_manager_usage(self, mock_config):
        """Test QdrantService can be used as context manager."""
        service = QdrantService(mock_config)

        with (
            patch.object(service, "initialize") as mock_init,
            patch.object(service, "cleanup") as mock_cleanup,
        ):
            async with service:
                pass

            mock_init.assert_called_once()
            mock_cleanup.assert_called_once()

    async def test_module_initialization_order(self, mock_config):
        """Test that modules are initialized in correct order."""
        with (
            patch("src.services.vector_db.service.QdrantClient") as mock_client_class,
            patch(
                "src.services.vector_db.service.QdrantCollections"
            ) as mock_collections_class,
            patch("src.services.vector_db.service.QdrantSearch") as mock_search_class,
            patch(
                "src.services.vector_db.service.QdrantIndexing"
            ) as mock_indexing_class,
            patch(
                "src.services.vector_db.service.QdrantDocuments"
            ) as mock_documents_class,
        ):
            # Mock client manager
            mock_client_manager = AsyncMock()
            mock_client = AsyncMock()
            # get_client() is synchronous, so use a regular property, not async
            mock_client_manager.get_client = MagicMock(return_value=mock_client)
            mock_client_class.return_value = mock_client_manager

            # Mock modules with async initialize methods
            mock_collections = AsyncMock()
            mock_search = AsyncMock()
            mock_indexing = AsyncMock()
            mock_documents = AsyncMock()

            mock_collections_class.return_value = mock_collections
            mock_search_class.return_value = mock_search
            mock_indexing_class.return_value = mock_indexing
            mock_documents_class.return_value = mock_documents

            # Create service AFTER patching so it gets the mocked client
            service = QdrantService(mock_config)
            await service.initialize()

            # Verify initialization order
            mock_client_manager.initialize.assert_called_once()
            mock_collections.initialize.assert_called_once()

            # All modules should be created with the shared client
            mock_collections_class.assert_called_once_with(service.config, mock_client)
            mock_search_class.assert_called_once_with(mock_client, service.config)
            mock_indexing_class.assert_called_once_with(mock_client, service.config)
            mock_documents_class.assert_called_once_with(mock_client, service.config)

    async def test_error_propagation(self, initialized_service):
        """Test that errors from modules are properly propagated."""
        # Test that errors from delegated methods are propagated
        initialized_service._collections.create_collection.side_effect = (
            QdrantServiceError("Collection error")
        )

        with pytest.raises(QdrantServiceError, match="Collection error"):
            await initialized_service.create_collection("test", 1536)

    async def test_all_delegation_methods_check_initialization(self, service):
        """Test that all delegation methods check initialization."""
        # List of methods that should check initialization
        delegation_methods = [
            ("create_collection", ("test", 1536)),
            ("delete_collection", ("test",)),
            ("list_collections", ()),
            ("list_collections_details", ()),
            ("get_collection_info", ("test",)),
            ("trigger_collection_optimization", ("test",)),
            ("hybrid_search", ("test", [0.1, 0.2])),
            (
                "multi_stage_search",
                (
                    "test",
                    [{"query_vector": [0.1], "vector_name": "dense", "limit": 10}],
                ),
            ),
            ("hyde_search", ("test", "query", [0.1], [[0.2]])),
            ("filtered_search", ("test", [0.1] * 1536, {})),
            ("create_payload_indexes", ("test",)),
            ("list_payload_indexes", ("test",)),
            ("drop_payload_index", ("test", "field")),
            ("reindex_collection", ("test",)),
            ("get_payload_index_stats", ("test",)),
            ("validate_index_health", ("test",)),
            ("get_index_usage_stats", ("test",)),
            ("upsert_points", ("test", [])),
            ("get_points", ("test", [])),
            ("delete_points", ("test",)),
            ("update_point_payload", ("test", "point1", {})),
            ("count_points", ("test",)),
            ("scroll_points", ("test",)),
            ("clear_collection", ("test",)),
        ]

        for method_name, args in delegation_methods:
            method = getattr(service, method_name)
            with pytest.raises(QdrantServiceError, match="Service not initialized"):
                await method(*args)
