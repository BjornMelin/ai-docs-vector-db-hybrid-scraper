"""Tests for blue-green deployment service."""

from unittest.mock import AsyncMock

import pytest
from src.config import UnifiedConfig
from src.services.deployment.blue_green import BlueGreenDeployment
from src.services.errors import ServiceError


class TestBlueGreenDeployment:
    """Test BlueGreenDeployment service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    def mock_qdrant_service(self):
        """Create mock QdrantService."""
        service = AsyncMock()
        service.get_collection_stats.return_value = {
            "vectors_count": 1000,
            "indexed_vectors_count": 1000
        }
        return service

    @pytest.fixture
    def mock_alias_manager(self):
        """Create mock QdrantAliasManager."""
        manager = AsyncMock()
        manager.get_collection_for_alias.return_value = "blue_collection_v1"
        manager.clone_collection_schema.return_value = True
        manager.copy_collection_data.return_value = True
        manager.switch_alias.return_value = True
        manager.safe_delete_collection.return_value = True
        return manager

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock EmbeddingManager."""
        manager = AsyncMock()
        manager.generate_embedding.return_value = {
            "embedding": [0.1] * 1536
        }
        return manager

    @pytest.fixture
    async def deployment_service(self, config, mock_qdrant_service, mock_alias_manager):
        """Create BlueGreenDeployment instance."""
        service = BlueGreenDeployment(
            config=config,
            qdrant_service=mock_qdrant_service,
            alias_manager=mock_alias_manager
        )
        await service.initialize()
        return service

    async def test_initialization(self, config, mock_qdrant_service, mock_alias_manager):
        """Test service initialization."""
        service = BlueGreenDeployment(
            config=config,
            qdrant_service=mock_qdrant_service,
            alias_manager=mock_alias_manager
        )

        assert not service._initialized
        await service.initialize()
        assert service._initialized

    async def test_cleanup(self, deployment_service):
        """Test service cleanup."""
        assert deployment_service._initialized
        await deployment_service.cleanup()
        assert not deployment_service._initialized

    async def test_deploy_new_version_success(self, deployment_service, mock_qdrant_service, mock_alias_manager, mock_embedding_manager):
        """Test successful deployment of new version."""
        deployment_service.embeddings = mock_embedding_manager

        alias_name = "production"
        data_source = "collection:source_collection"
        validation_queries = ["test query"]

        # Mock successful validation
        mock_qdrant_service.query.return_value = [
            {"score": 0.8, "payload": {"content": "result1"}},
            {"score": 0.7, "payload": {"content": "result2"}},
            {"score": 0.6, "payload": {"content": "result3"}},
            {"score": 0.5, "payload": {"content": "result4"}},
            {"score": 0.4, "payload": {"content": "result5"}}
        ]

        result = await deployment_service.deploy_new_version(
            alias_name=alias_name,
            data_source=data_source,
            validation_queries=validation_queries,
            health_check_duration=1,  # Very short duration for testing
            health_check_interval=0.1  # Very short interval for testing
        )

        # Verify
        assert result["success"] is True
        assert "new_collection" in result
        assert "old_collection" in result
        assert result["alias"] == alias_name
        mock_alias_manager.clone_collection_schema.assert_called_once()
        mock_alias_manager.copy_collection_data.assert_called_once()
        mock_alias_manager.switch_alias.assert_called_once()

    async def test_deploy_new_version_validation_failure(self, deployment_service, mock_qdrant_service, mock_alias_manager, mock_embedding_manager):
        """Test deployment with validation failure."""
        deployment_service.embeddings = mock_embedding_manager

        alias_name = "production"
        data_source = "collection:source_collection"
        validation_queries = ["test query"]

        # Mock validation failure (low scores)
        mock_qdrant_service.query.return_value = [
            {"score": 0.3, "payload": {"content": "result1"}},
            {"score": 0.2, "payload": {"content": "result2"}}
        ]

        with pytest.raises(ServiceError, match="Deployment failed"):
            await deployment_service.deploy_new_version(
                alias_name=alias_name,
                data_source=data_source,
                validation_queries=validation_queries,
                health_check_duration=1,  # Very short duration for testing
                health_check_interval=0.1  # Very short interval for testing
            )

    async def test_deploy_new_version_no_alias(self, deployment_service, mock_alias_manager):
        """Test deployment when alias doesn't exist."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        with pytest.raises(ServiceError, match="No collection found for alias"):
            await deployment_service.deploy_new_version(
                alias_name="nonexistent",
                data_source="collection:source",
                validation_queries=["test"],
                health_check_duration=1,  # Very short duration for testing
                health_check_interval=0.1  # Very short interval for testing
            )

    async def test_get_deployment_status_success(self, deployment_service, mock_alias_manager, mock_qdrant_service):
        """Test getting deployment status."""
        alias_name = "production"
        active_collection = "green_v2"

        mock_alias_manager.get_collection_for_alias.return_value = active_collection
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 1500,
            "indexed_vectors_count": 1500
        }

        status = await deployment_service.get_deployment_status(alias_name)

        assert status["alias"] == alias_name
        assert status["collection"] == active_collection
        assert status["status"] == "active"
        assert status["vectors_count"] == 1500
        assert status["indexed_vectors_count"] == 1500

    async def test_get_deployment_status_alias_not_found(self, deployment_service, mock_alias_manager):
        """Test getting status for non-existent alias."""
        alias_name = "nonexistent"

        mock_alias_manager.get_collection_for_alias.return_value = None

        status = await deployment_service.get_deployment_status(alias_name)

        assert status["alias"] == alias_name
        assert status["status"] == "not_found"
        assert status["collection"] is None

    async def test_get_deployment_status_error(self, deployment_service, mock_alias_manager, mock_qdrant_service):
        """Test getting status when service error occurs."""
        alias_name = "production"

        mock_alias_manager.get_collection_for_alias.return_value = "collection"
        mock_qdrant_service.get_collection_stats.side_effect = Exception("Stats error")

        status = await deployment_service.get_deployment_status(alias_name)

        assert status["alias"] == alias_name
        assert status["status"] == "error"
        assert "error" in status

    async def test_validate_collection_no_embedding_manager(self, deployment_service):
        """Test validation without embedding manager."""
        result = await deployment_service._validate_collection(
            collection_name="test_collection",
            validation_queries=["test query"]
        )

        # Should skip validation and return True
        assert result is True

    async def test_validate_collection_insufficient_results(self, deployment_service, mock_qdrant_service, mock_embedding_manager):
        """Test validation with insufficient search results."""
        deployment_service.embeddings = mock_embedding_manager

        # Mock insufficient results
        mock_qdrant_service.query.return_value = [
            {"score": 0.8, "payload": {"content": "result1"}},
            {"score": 0.7, "payload": {"content": "result2"}}
        ]  # Less than 5 results

        result = await deployment_service._validate_collection(
            collection_name="test_collection",
            validation_queries=["test query"]
        )

        assert result is False

    async def test_populate_collection_unknown_source(self, deployment_service):
        """Test populating collection with unknown data source type."""
        with pytest.raises(ServiceError, match="Unknown data source type"):
            await deployment_service._populate_collection(
                collection_name="test_collection",
                data_source="unknown:source"
            )

    async def test_populate_collection_backup_not_implemented(self, deployment_service):
        """Test populating collection from backup (not implemented)."""
        with pytest.raises(NotImplementedError, match="Backup restore not yet implemented"):
            await deployment_service._populate_collection(
                collection_name="test_collection",
                data_source="backup:some_backup"
            )

    async def test_populate_collection_crawl_not_implemented(self, deployment_service):
        """Test populating collection from crawl (not implemented)."""
        with pytest.raises(NotImplementedError, match="Fresh crawl population not yet implemented"):
            await deployment_service._populate_collection(
                collection_name="test_collection",
                data_source="crawl:some_crawl"
            )
