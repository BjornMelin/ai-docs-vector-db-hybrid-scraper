"""Tests for BlueGreenDeployment with task queue integration."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.deployment.blue_green import BlueGreenDeployment
from src.services.errors import ServiceError


class TestBlueGreenDeploymentWithTaskQueue:
    """Test BlueGreenDeployment with task queue integration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Mock(spec=UnifiedConfig)

    @pytest.fixture
    def qdrant_service(self):
        """Create mock Qdrant service."""
        mock_service = AsyncMock()
        mock_service.get_collection_stats = AsyncMock(
            return_value={
                "vectors_count": 1000,
                "indexed_vectors_count": 1000,
                "points_count": 1000,
            }
        )
        return mock_service

    @pytest.fixture
    def alias_manager(self):
        """Create mock alias manager with task queue."""
        mock_manager = AsyncMock()
        mock_manager._task_queue_manager = AsyncMock()
        return mock_manager

    @pytest.fixture
    def embedding_manager(self):
        """Create mock embedding manager."""
        mock_manager = AsyncMock()
        mock_manager.generate_embedding = AsyncMock(
            return_value={"embedding": [0.1] * 1536}
        )
        return mock_manager

    @pytest.fixture
    def blue_green(self, config, qdrant_service, alias_manager, embedding_manager):
        """Create BlueGreenDeployment instance."""
        return BlueGreenDeployment(
            config=config,
            qdrant_service=qdrant_service,
            alias_manager=alias_manager,
            embedding_manager=embedding_manager,
        )

    @pytest.mark.asyncio
    async def test_deploy_new_version_with_task_queue(
        self, blue_green, alias_manager, qdrant_service
    ):
        """Test deploy_new_version uses task queue for cleanup."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(
            return_value="blue_collection"
        )
        alias_manager.clone_collection_schema = AsyncMock(return_value=True)
        alias_manager.copy_collection_data = AsyncMock(return_value=1000)
        alias_manager.switch_alias = AsyncMock()
        alias_manager.safe_delete_collection = AsyncMock()

        # Mock validation to pass
        qdrant_service.query = AsyncMock(
            return_value=[
                {"score": 0.9, "id": "1"},
                {"score": 0.85, "id": "2"},
                {"score": 0.8, "id": "3"},
                {"score": 0.75, "id": "4"},
                {"score": 0.7, "id": "5"},
            ]
        )

        # Mock monitoring to pass quickly
        with patch("asyncio.sleep", return_value=None):
            # Execute
            result = await blue_green.deploy_new_version(
                alias_name="test_alias",
                data_source="collection:source_collection",
                validation_queries=["test query 1", "test query 2"],
                health_check_duration=1,  # Very short for test
                health_check_interval=1,
            )

        # Verify
        assert result["success"] is True
        assert result["old_collection"] == "blue_collection"
        assert result["alias"] == "test_alias"
        assert "new_collection" in result

        # Verify safe_delete_collection was called (which uses task queue)
        alias_manager.safe_delete_collection.assert_called_once_with("blue_collection")

    @pytest.mark.asyncio
    async def test_deploy_validation_failure_no_cleanup(
        self, blue_green, alias_manager, qdrant_service
    ):
        """Test deploy with validation failure doesn't schedule cleanup."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(
            return_value="blue_collection"
        )
        alias_manager.clone_collection_schema = AsyncMock(return_value=True)
        alias_manager.copy_collection_data = AsyncMock(return_value=1000)

        # Mock validation to fail
        qdrant_service.query = AsyncMock(
            return_value=[
                {"score": 0.5, "id": "1"},  # Low score
            ]
        )

        # Mock rollback
        blue_green._rollback = AsyncMock()

        # Execute
        with pytest.raises(ServiceError, match="Validation failed"):
            await blue_green.deploy_new_version(
                alias_name="test_alias",
                data_source="collection:source_collection",
                validation_queries=["test query"],
                rollback_on_failure=True,
            )

        # Verify no cleanup was scheduled
        alias_manager.safe_delete_collection.assert_not_called()

        # Verify rollback was called
        blue_green._rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_with_different_data_sources(self, blue_green, alias_manager):
        """Test deploy with different data source types."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(
            return_value="blue_collection"
        )
        alias_manager.clone_collection_schema = AsyncMock(return_value=True)

        # Test backup source (not implemented) - wrapped in ServiceError
        with pytest.raises(ServiceError, match="Deployment failed.*Backup restore"):
            await blue_green.deploy_new_version(
                alias_name="test_alias",
                data_source="backup:/path/to/backup",
                validation_queries=[],
            )

        # Test crawl source (not implemented) - wrapped in ServiceError  
        with pytest.raises(ServiceError, match="Deployment failed.*Fresh crawl"):
            await blue_green.deploy_new_version(
                alias_name="test_alias",
                data_source="crawl:config_name",
                validation_queries=[],
            )

        # Test unknown source
        with pytest.raises(ServiceError, match="Unknown data source"):
            await blue_green.deploy_new_version(
                alias_name="test_alias",
                data_source="unknown:source",
                validation_queries=[],
            )

    @pytest.mark.asyncio
    async def test_get_deployment_status(
        self, blue_green, alias_manager, qdrant_service
    ):
        """Test get_deployment_status."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(
            return_value="test_collection"
        )
        qdrant_service.get_collection_stats = AsyncMock(
            return_value={
                "vectors_count": 5000,
                "indexed_vectors_count": 4500,
            }
        )

        # Execute
        status = await blue_green.get_deployment_status("test_alias")

        # Verify
        assert status["alias"] == "test_alias"
        assert status["status"] == "active"
        assert status["collection"] == "test_collection"
        assert status["vectors_count"] == 5000
        assert status["indexed_vectors_count"] == 4500

    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(self, blue_green, alias_manager):
        """Test get_deployment_status for non-existent alias."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(return_value=None)

        # Execute
        status = await blue_green.get_deployment_status("nonexistent")

        # Verify
        assert status["alias"] == "nonexistent"
        assert status["status"] == "not_found"
        assert status["collection"] is None
