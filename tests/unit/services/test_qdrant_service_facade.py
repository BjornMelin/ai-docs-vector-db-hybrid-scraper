"""Tests for QdrantService facade integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.service import QdrantService


class TestQdrantServiceFacade:
    """Test QdrantService facade integration with all modules."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    async def service(self, config):
        """Create QdrantService facade."""
        return QdrantService(config)

    async def test_service_initialization(self, service):
        """Test service initialization without actual connections."""
        assert service.config is not None
        assert service._client_manager is not None
        assert service._collections is None  # Not initialized yet
        assert service._search is None
        assert service._indexing is None
        assert service._documents is None

    async def test_service_initialization_full(self, service):
        """Test full service initialization with mocked dependencies."""
        # Mock the client manager and client
        mock_client_manager = AsyncMock()
        mock_client = AsyncMock()
        mock_client_manager.get_client.return_value = mock_client
        service._client_manager = mock_client_manager

        # Mock all module classes to return mock instances
        with patch('src.services.vector_db.collections.QdrantCollections') as mock_collections_class, \
             patch('src.services.vector_db.search.QdrantSearch') as mock_search_class, \
             patch('src.services.vector_db.indexing.QdrantIndexing') as mock_indexing_class, \
             patch('src.services.vector_db.documents.QdrantDocuments') as mock_documents_class:

            # Create mock instances that will be returned by the classes
            mock_collections = AsyncMock()
            mock_search = MagicMock()
            mock_indexing = MagicMock()
            mock_documents = MagicMock()

            mock_collections_class.return_value = mock_collections
            mock_search_class.return_value = mock_search
            mock_indexing_class.return_value = mock_indexing
            mock_documents_class.return_value = mock_documents

            await service.initialize()

            assert service._initialized is True
            assert service._collections is not None
            assert service._search is not None
            assert service._indexing is not None
            assert service._documents is not None

            # Verify modules were initialized with correct parameters
            mock_collections_class.assert_called_once_with(service.config, mock_client)
            mock_search_class.assert_called_once_with(mock_client, service.config)
            mock_indexing_class.assert_called_once_with(mock_client, service.config)
            mock_documents_class.assert_called_once_with(mock_client, service.config)

            # Verify collections module was initialized
            mock_collections.initialize.assert_called_once()

    async def test_service_cleanup(self, service):
        """Test service cleanup."""
        # Mock initialized modules
        mock_collections = AsyncMock()
        mock_client_manager = AsyncMock()
        
        service._collections = mock_collections
        service._client_manager = mock_client_manager
        service._search = MagicMock()
        service._indexing = MagicMock()
        service._documents = MagicMock()
        service._initialized = True

        await service.cleanup()

        # Verify cleanup was called
        mock_collections.cleanup.assert_called_once()
        mock_client_manager.cleanup.assert_called_once()

        # Verify modules were reset
        assert service._collections is None
        assert service._search is None
        assert service._indexing is None
        assert service._documents is None
        assert service._initialized is False

    async def test_collection_api_delegation(self, service):
        """Test that collection API methods delegate to QdrantCollections."""
        mock_collections = AsyncMock()
        service._collections = mock_collections
        service._initialized = True

        # Test create_collection delegation
        mock_collections.create_collection.return_value = True
        result = await service.create_collection("test_collection", 1536, "Cosine")
        assert result is True
        mock_collections.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name=None,
            enable_quantization=True,
            collection_type="general",
        )

        # Test list_collections delegation
        mock_collections.list_collections.return_value = ["collection1", "collection2"]
        result = await service.list_collections()
        assert result == ["collection1", "collection2"]
        mock_collections.list_collections.assert_called_once()

        # Test delete_collection delegation
        mock_collections.delete_collection.return_value = True
        result = await service.delete_collection("test_collection")
        assert result is True
        mock_collections.delete_collection.assert_called_once_with("test_collection")

    async def test_search_api_delegation(self, service):
        """Test that search API methods delegate to QdrantSearch."""
        mock_search = AsyncMock()
        service._search = mock_search
        service._initialized = True

        # Test hybrid_search delegation
        query_vector = [0.1] * 1536
        sparse_vector = {1: 0.5, 2: 0.3}
        
        mock_search.hybrid_search.return_value = [{"id": "test-1", "score": 0.95}]
        
        result = await service.hybrid_search(
            collection_name="test_collection",
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
        )
        
        assert result == [{"id": "test-1", "score": 0.95}]
        mock_search.hybrid_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
            score_threshold=0.0,
            fusion_type="rrf",
            search_accuracy="balanced",
        )

        # Test filtered_search delegation
        filters = {"doc_type": "api", "language": "python"}
        await service.filtered_search("test_collection", query_vector, filters)
        mock_search.filtered_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            filters=filters,
            limit=10,
            search_accuracy="balanced",
        )

    async def test_indexing_api_delegation(self, service):
        """Test that indexing API methods delegate to QdrantIndexing."""
        mock_indexing = AsyncMock()
        service._indexing = mock_indexing
        service._initialized = True

        # Test create_payload_indexes delegation
        await service.create_payload_indexes("test_collection")
        mock_indexing.create_payload_indexes.assert_called_once_with("test_collection")

        # Test list_payload_indexes delegation
        mock_indexing.list_payload_indexes.return_value = ["doc_type", "language"]
        result = await service.list_payload_indexes("test_collection")
        assert result == ["doc_type", "language"]
        mock_indexing.list_payload_indexes.assert_called_once_with("test_collection")

        # Test validate_index_health delegation
        health_report = {"status": "healthy", "health_score": 95.0}
        mock_indexing.validate_index_health.return_value = health_report
        result = await service.validate_index_health("test_collection")
        assert result == health_report
        mock_indexing.validate_index_health.assert_called_once_with("test_collection")

    async def test_documents_api_delegation(self, service):
        """Test that documents API methods delegate to QdrantDocuments."""
        mock_documents = AsyncMock()
        service._documents = mock_documents
        service._initialized = True

        # Test upsert_points delegation
        points = [{"id": "doc-1", "vector": [0.1] * 1536, "payload": {"title": "Test"}}]
        mock_documents.upsert_points.return_value = True
        
        result = await service.upsert_points("test_collection", points, 100)
        assert result is True
        mock_documents.upsert_points.assert_called_once_with(
            collection_name="test_collection",
            points=points,
            batch_size=100,
        )

        # Test get_points delegation
        point_ids = ["doc-1", "doc-2"]
        mock_documents.get_points.return_value = [{"id": "doc-1", "payload": {}}]
        
        result = await service.get_points("test_collection", point_ids)
        assert result == [{"id": "doc-1", "payload": {}}]
        mock_documents.get_points.assert_called_once_with(
            collection_name="test_collection",
            point_ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        # Test count_points delegation
        mock_documents.count_points.return_value = 1000
        result = await service.count_points("test_collection")
        assert result == 1000
        mock_documents.count_points.assert_called_once_with(
            collection_name="test_collection",
            filter_condition=None,
            exact=True,
        )

    async def test_hnsw_optimization_api_delegation(self, service):
        """Test that HNSW optimization API methods delegate correctly."""
        mock_collections = AsyncMock()
        service._collections = mock_collections
        service._initialized = True

        # Test create_collection_with_hnsw_optimization delegation
        await service.create_collection_with_hnsw_optimization(
            "test_collection", 1536, "api_reference"
        )
        mock_collections.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name=None,
            enable_quantization=True,
            collection_type="api_reference",
        )

        # Test get_hnsw_configuration_info delegation (non-async method)
        config_info = {"hnsw_parameters": {"m": 32, "ef_construct": 400}}
        mock_collections.get_hnsw_configuration_info.return_value = config_info
        
        result = service.get_hnsw_configuration_info("api_reference")
        assert result == config_info
        mock_collections.get_hnsw_configuration_info.assert_called_once_with("api_reference")

    async def test_compatibility_methods(self, service):
        """Test compatibility methods for legacy API."""
        mock_search = AsyncMock()
        mock_collections = AsyncMock()
        service._search = mock_search
        service._collections = mock_collections
        service._initialized = True

        # Test search_with_adaptive_ef (compatibility method)
        query_vector = [0.1] * 1536
        mock_search.filtered_search.return_value = [{"id": "test-1", "score": 0.95}]
        
        result = await service.search_with_adaptive_ef(
            "test_collection", query_vector, limit=10, score_threshold=0.8
        )
        
        assert "results" in result
        assert "adaptive_ef_used" in result
        assert "time_budget_ms" in result
        
        # Test optimize_collection_hnsw_parameters (compatibility method)
        config_info = {"hnsw_parameters": {"m": 32, "ef_construct": 400}}
        mock_collections.get_hnsw_configuration_info.return_value = config_info
        
        result = await service.optimize_collection_hnsw_parameters(
            "test_collection", "api_reference"
        )
        
        assert "collection_name" in result
        assert "optimization_results" in result
        assert result["collection_name"] == "test_collection"

    async def test_validation_not_initialized(self, service):
        """Test validation when service is not initialized."""
        # Service is not initialized
        assert service._initialized is False

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.create_collection("test", 1536)

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.hybrid_search("test", [0.1] * 1536)

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.upsert_points("test", [])

    async def test_initialization_error_handling(self, service):
        """Test error handling during initialization."""
        with patch('src.services.vector_db.client.QdrantClient') as mock_client_class:
            mock_client_manager = AsyncMock()
            mock_client_manager.initialize.side_effect = Exception("Connection failed")
            service._client_manager = mock_client_manager

            with pytest.raises(QdrantServiceError, match="Failed to initialize QdrantService"):
                await service.initialize()

            assert service._initialized is False

    async def test_double_initialization(self, service):
        """Test that double initialization is handled gracefully."""
        with patch('src.services.vector_db.client.QdrantClient'):
            mock_client_manager = AsyncMock()
            mock_client = AsyncMock()
            mock_client_manager.get_client.return_value = mock_client
            service._client_manager = mock_client_manager

            with patch('src.services.vector_db.collections.QdrantCollections') as mock_collections_class, \
                 patch('src.services.vector_db.search.QdrantSearch'), \
                 patch('src.services.vector_db.indexing.QdrantIndexing'), \
                 patch('src.services.vector_db.documents.QdrantDocuments'):

                mock_collections = AsyncMock()
                mock_collections_class.return_value = mock_collections

                # First initialization
                await service.initialize()
                assert service._initialized is True

                # Second initialization should return early
                await service.initialize()
                
                # Should only be called once
                mock_collections_class.assert_called_once()

    async def test_api_coverage(self, service):
        """Test that facade covers all expected API methods."""
        # Collection management methods
        assert hasattr(service, 'create_collection')
        assert hasattr(service, 'delete_collection')
        assert hasattr(service, 'list_collections')
        assert hasattr(service, 'get_collection_info')

        # Search methods
        assert hasattr(service, 'hybrid_search')
        assert hasattr(service, 'multi_stage_search')
        assert hasattr(service, 'hyde_search')
        assert hasattr(service, 'filtered_search')

        # Indexing methods
        assert hasattr(service, 'create_payload_indexes')
        assert hasattr(service, 'list_payload_indexes')
        assert hasattr(service, 'validate_index_health')

        # Document methods
        assert hasattr(service, 'upsert_points')
        assert hasattr(service, 'get_points')
        assert hasattr(service, 'delete_points')
        assert hasattr(service, 'count_points')

        # HNSW optimization methods
        assert hasattr(service, 'create_collection_with_hnsw_optimization')
        assert hasattr(service, 'get_hnsw_configuration_info')

        # Compatibility methods
        assert hasattr(service, 'search_with_adaptive_ef')
        assert hasattr(service, 'optimize_collection_hnsw_parameters')