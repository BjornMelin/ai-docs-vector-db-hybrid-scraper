"""Tests for Blue-Green deployment pattern."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.deployment.blue_green import BlueGreenDeployment
from src.services.errors import ServiceError


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    config.performance = MagicMock()
    config.performance.timeout = 30
    return config


@pytest.fixture
def mock_qdrant_service():
    """Create mock Qdrant service."""
    service = AsyncMock()
    service.query = AsyncMock(
        return_value=[
            {"id": "doc1", "score": 0.95, "payload": {"content": "test"}},
            {"id": "doc2", "score": 0.85, "payload": {"content": "test2"}},
        ]
    )
    service.get_collection_stats = AsyncMock(
        return_value={
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
        }
    )
    service._client = MagicMock()
    service._client.delete_collection = AsyncMock()
    return service


@pytest.fixture
def mock_alias_manager():
    """Create mock alias manager."""
    manager = AsyncMock()
    manager.get_collection_for_alias = AsyncMock(return_value="current_collection")
    manager.clone_collection_schema = AsyncMock()
    manager.copy_collection_data = AsyncMock()
    manager.switch_alias = AsyncMock()
    manager.safe_delete_collection = AsyncMock()
    return manager


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()
    manager.generate_embedding = AsyncMock(
        return_value={
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model": "test-model",
        }
    )
    return manager


@pytest.fixture
def blue_green_deployment(
    mock_config, mock_qdrant_service, mock_alias_manager, mock_embedding_manager
):
    """Create BlueGreenDeployment instance."""
    deployment = BlueGreenDeployment(
        config=mock_config,
        qdrant_service=mock_qdrant_service,
        alias_manager=mock_alias_manager,
        embedding_manager=mock_embedding_manager,
    )
    return deployment


class TestBlueGreenDeployment:
    """Test BlueGreenDeployment class."""

    @pytest.mark.asyncio
    async def test_initialization(self, blue_green_deployment):
        """Test deployment service initialization."""
        await blue_green_deployment.initialize()
        assert blue_green_deployment._initialized is True

        await blue_green_deployment.cleanup()
        assert blue_green_deployment._initialized is False

    @pytest.mark.asyncio
    async def test_initialization_without_embedding_manager(
        self, mock_config, mock_qdrant_service, mock_alias_manager
    ):
        """Test initialization without embedding manager."""
        deployment = BlueGreenDeployment(
            config=mock_config,
            qdrant_service=mock_qdrant_service,
            alias_manager=mock_alias_manager,
            embedding_manager=None,
        )

        await deployment.initialize()
        assert deployment._initialized is True
        assert deployment.embeddings is None

    @pytest.mark.asyncio
    async def test_deploy_new_version_no_current_collection(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment when no current collection exists."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        with pytest.raises(
            ServiceError, match="No collection found for alias test_alias"
        ):
            await blue_green_deployment.deploy_new_version(
                alias_name="test_alias",
                data_source="collection:source_coll",
                validation_queries=["test query"],
            )

    @pytest.mark.asyncio
    async def test_deploy_new_version_success(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test successful deployment."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        # Mock datetime for consistent green collection name
        with patch("src.services.deployment.blue_green.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            result = await blue_green_deployment.deploy_new_version(
                alias_name="test_alias",
                data_source="collection:source_coll",
                validation_queries=["test query 1", "test query 2"],
                validation_threshold=0.8,
                health_check_interval=5,
                health_check_duration=60,
            )

        expected_green_collection = "test_alias_20240101_120000"

        # Verify workflow steps
        mock_alias_manager.clone_collection_schema.assert_called_once_with(
            source="blue_collection",
            target=expected_green_collection,
        )
        mock_alias_manager.copy_collection_data.assert_called_once_with(
            source="source_coll",
            target=expected_green_collection,
        )
        mock_alias_manager.switch_alias.assert_called_once_with(
            alias_name="test_alias",
            new_collection=expected_green_collection,
        )

        # Verify result
        assert result["success"] is True
        assert result["old_collection"] == "blue_collection"
        assert result["new_collection"] == expected_green_collection
        assert result["alias"] == "test_alias"
        assert result["deployed_at"] == "2024-01-01T12:00:00"

    @pytest.mark.asyncio
    async def test_deploy_new_version_validation_failure(
        self, blue_green_deployment, mock_alias_manager, mock_embedding_manager
    ):
        """Test deployment with validation failure."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        # Mock embedding generation to return low scores
        mock_embedding_manager.generate_embedding.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "model": "test-model",
        }

        # Mock Qdrant to return low-scoring results
        blue_green_deployment.qdrant.query.return_value = [
            {
                "id": "doc1",
                "score": 0.5,
                "payload": {"content": "test"},
            },  # Below threshold
        ]

        with patch("src.services.deployment.blue_green.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            with pytest.raises(
                ServiceError, match="Validation failed for new collection"
            ):
                await blue_green_deployment.deploy_new_version(
                    alias_name="test_alias",
                    data_source="collection:source_coll",
                    validation_queries=["test query"],
                    validation_threshold=0.7,
                )

    @pytest.mark.asyncio
    async def test_deploy_new_version_with_rollback(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment failure with rollback."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"
        mock_alias_manager.clone_collection_schema.side_effect = Exception(
            "Schema clone failed"
        )

        with patch("src.services.deployment.blue_green.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            with pytest.raises(ServiceError, match="Deployment failed"):
                await blue_green_deployment.deploy_new_version(
                    alias_name="test_alias",
                    data_source="collection:source_coll",
                    validation_queries=["test query"],
                    rollback_on_failure=True,
                )

        # Verify rollback was attempted
        mock_alias_manager.switch_alias.assert_called()

    @pytest.mark.asyncio
    async def test_deploy_new_version_without_rollback(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment failure without rollback."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"
        mock_alias_manager.clone_collection_schema.side_effect = Exception(
            "Schema clone failed"
        )

        with patch("src.services.deployment.blue_green.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            with pytest.raises(ServiceError, match="Deployment failed"):
                await blue_green_deployment.deploy_new_version(
                    alias_name="test_alias",
                    data_source="collection:source_coll",
                    validation_queries=["test query"],
                    rollback_on_failure=False,
                )

    @pytest.mark.asyncio
    async def test_populate_collection_from_collection(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test populating collection from another collection."""
        await blue_green_deployment._populate_collection(
            collection_name="target_collection",
            data_source="collection:source_collection",
        )

        mock_alias_manager.copy_collection_data.assert_called_once_with(
            source="source_collection",
            target="target_collection",
        )

    @pytest.mark.asyncio
    async def test_populate_collection_backup_not_implemented(
        self, blue_green_deployment
    ):
        """Test populating collection from backup (not implemented)."""
        with pytest.raises(
            NotImplementedError, match="Backup restore not yet implemented"
        ):
            await blue_green_deployment._populate_collection(
                collection_name="target_collection",
                data_source="backup:backup_file.tar.gz",
            )

    @pytest.mark.asyncio
    async def test_populate_collection_crawl_not_implemented(
        self, blue_green_deployment
    ):
        """Test populating collection from crawl (not implemented)."""
        with pytest.raises(
            NotImplementedError, match="Fresh crawl population not yet implemented"
        ):
            await blue_green_deployment._populate_collection(
                collection_name="target_collection",
                data_source="crawl:https://example.com",
            )

    @pytest.mark.asyncio
    async def test_populate_collection_unknown_source(self, blue_green_deployment):
        """Test populating collection with unknown data source."""
        with pytest.raises(ServiceError, match="Unknown data source type"):
            await blue_green_deployment._populate_collection(
                collection_name="target_collection",
                data_source="unknown:some_source",
            )

    @pytest.mark.asyncio
    async def test_validate_collection_without_embedding_manager(
        self, blue_green_deployment
    ):
        """Test validation without embedding manager."""
        blue_green_deployment.embeddings = None

        result = await blue_green_deployment._validate_collection(
            collection_name="test_collection",
            validation_queries=["query1", "query2"],
        )

        assert result is True  # Should pass without validation

    @pytest.mark.asyncio
    async def test_validate_collection_success(
        self, blue_green_deployment, mock_qdrant_service, mock_embedding_manager
    ):
        """Test successful collection validation."""
        mock_embedding_manager.generate_embedding.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model": "test-model",
        }

        mock_qdrant_service.query.return_value = [
            {"id": "doc1", "score": 0.85, "payload": {"content": "test"}},
            {"id": "doc2", "score": 0.80, "payload": {"content": "test2"}},
            {"id": "doc3", "score": 0.75, "payload": {"content": "test3"}},
            {"id": "doc4", "score": 0.70, "payload": {"content": "test4"}},
            {"id": "doc5", "score": 0.65, "payload": {"content": "test5"}},
        ]

        result = await blue_green_deployment._validate_collection(
            collection_name="test_collection",
            validation_queries=["query1", "query2"],
            threshold=0.7,
        )

        assert result is True
        assert mock_embedding_manager.generate_embedding.call_count == 2
        assert mock_qdrant_service.query.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_collection_insufficient_results(
        self, blue_green_deployment, mock_qdrant_service, mock_embedding_manager
    ):
        """Test validation failure due to insufficient results."""
        mock_qdrant_service.query.return_value = [
            {"id": "doc1", "score": 0.85, "payload": {"content": "test"}},
            {"id": "doc2", "score": 0.80, "payload": {"content": "test2"}},
        ]  # Only 2 results, need at least 5

        result = await blue_green_deployment._validate_collection(
            collection_name="test_collection",
            validation_queries=["query1"],
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_collection_low_score(
        self, blue_green_deployment, mock_qdrant_service, mock_embedding_manager
    ):
        """Test validation failure due to low scores."""
        mock_qdrant_service.query.return_value = [
            {
                "id": "doc1",
                "score": 0.5,
                "payload": {"content": "test"},
            },  # Below threshold
            {"id": "doc2", "score": 0.45, "payload": {"content": "test2"}},
            {"id": "doc3", "score": 0.40, "payload": {"content": "test3"}},
            {"id": "doc4", "score": 0.35, "payload": {"content": "test4"}},
            {"id": "doc5", "score": 0.30, "payload": {"content": "test5"}},
        ]

        result = await blue_green_deployment._validate_collection(
            collection_name="test_collection",
            validation_queries=["query1"],
            threshold=0.7,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_collection_embedding_error(
        self, blue_green_deployment, mock_embedding_manager
    ):
        """Test validation failure due to embedding error."""
        mock_embedding_manager.generate_embedding.side_effect = Exception(
            "Embedding failed"
        )

        result = await blue_green_deployment._validate_collection(
            collection_name="test_collection",
            validation_queries=["query1"],
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_collection_query_error(
        self, blue_green_deployment, mock_qdrant_service, mock_embedding_manager
    ):
        """Test validation failure due to query error."""
        mock_qdrant_service.query.side_effect = Exception("Query failed")

        result = await blue_green_deployment._validate_collection(
            collection_name="test_collection",
            validation_queries=["query1"],
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_monitor_after_switch_success(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test successful monitoring after switch."""
        mock_alias_manager.get_collection_for_alias.return_value = "new_collection"
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
        }

        # Mock the event loop to speed up the test
        with (
            patch("asyncio.get_event_loop") as mock_loop,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_loop.return_value.time.side_effect = [
                0,
                5,
                10,
                65,
            ]  # Simulate time progression
            mock_sleep.return_value = None

            await blue_green_deployment._monitor_after_switch(
                alias_name="test_alias",
                duration_seconds=60,
                check_interval=5,
            )

        # Verify monitoring calls were made
        assert mock_alias_manager.get_collection_for_alias.call_count >= 1
        assert mock_qdrant_service.get_collection_stats.call_count >= 1

    @pytest.mark.asyncio
    async def test_monitor_after_switch_alias_missing(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test monitoring with missing alias."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        with (
            patch("asyncio.get_event_loop") as mock_loop,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_loop.return_value.time.side_effect = [
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                65,
            ]
            mock_sleep.return_value = None

            with pytest.raises(ServiceError, match="Too many errors after switch"):
                await blue_green_deployment._monitor_after_switch(
                    alias_name="test_alias",
                    duration_seconds=60,
                    check_interval=5,
                )

    @pytest.mark.asyncio
    async def test_monitor_after_switch_empty_collection(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test monitoring with empty collection."""
        mock_alias_manager.get_collection_for_alias.return_value = "empty_collection"
        mock_qdrant_service.get_collection_stats.return_value = {"vectors_count": 0}

        with (
            patch("asyncio.get_event_loop") as mock_loop,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_loop.return_value.time.side_effect = [
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                65,
            ]
            mock_sleep.return_value = None

            with pytest.raises(ServiceError, match="Too many errors after switch"):
                await blue_green_deployment._monitor_after_switch(
                    alias_name="test_alias",
                    duration_seconds=60,
                    check_interval=5,
                )

    @pytest.mark.asyncio
    async def test_rollback_success(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test successful rollback."""
        await blue_green_deployment._rollback(
            alias_name="test_alias",
            old_collection="blue_collection",
            new_collection="failed_green_collection",
        )

        mock_alias_manager.switch_alias.assert_called_once_with(
            alias_name="test_alias",
            new_collection="blue_collection",
        )
        mock_qdrant_service._client.delete_collection.assert_called_once_with(
            "failed_green_collection"
        )

    @pytest.mark.asyncio
    async def test_rollback_failure(self, blue_green_deployment, mock_alias_manager):
        """Test rollback failure."""
        mock_alias_manager.switch_alias.side_effect = Exception("Rollback failed")

        # Should not raise exception, just log error
        await blue_green_deployment._rollback(
            alias_name="test_alias",
            old_collection="blue_collection",
            new_collection="failed_green_collection",
        )

    @pytest.mark.asyncio
    async def test_get_deployment_status_active(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test getting status for active deployment."""
        mock_alias_manager.get_collection_for_alias.return_value = "active_collection"
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 1500,
            "indexed_vectors_count": 1400,
        }

        status = await blue_green_deployment.get_deployment_status("test_alias")

        assert status["alias"] == "test_alias"
        assert status["status"] == "active"
        assert status["collection"] == "active_collection"
        assert status["vectors_count"] == 1500
        assert status["indexed_vectors_count"] == 1400

    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test getting status for non-existent alias."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        status = await blue_green_deployment.get_deployment_status("nonexistent_alias")

        assert status["alias"] == "nonexistent_alias"
        assert status["status"] == "not_found"
        assert status["collection"] is None

    @pytest.mark.asyncio
    async def test_get_deployment_status_error(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test getting status when error occurs."""
        mock_alias_manager.get_collection_for_alias.return_value = "error_collection"
        mock_qdrant_service.get_collection_stats.side_effect = Exception("Stats error")

        status = await blue_green_deployment.get_deployment_status("test_alias")

        assert status["alias"] == "test_alias"
        assert status["status"] == "error"
        assert status["collection"] == "error_collection"
        assert status["error"] == "Stats error"

    def test_service_inheritance(self, blue_green_deployment):
        """Test that BlueGreenDeployment properly inherits from BaseService."""
        from src.services.base import BaseService

        assert isinstance(blue_green_deployment, BaseService)
        assert hasattr(blue_green_deployment, "_initialized")

    @pytest.mark.asyncio
    async def test_deployment_with_minimal_parameters(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment with minimal parameters (using defaults)."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with (
            patch("src.services.deployment.blue_green.datetime") as mock_datetime,
            patch.object(
                blue_green_deployment, "_monitor_after_switch"
            ) as mock_monitor,
        ):
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            result = await blue_green_deployment.deploy_new_version(
                alias_name="minimal_alias",
                data_source="collection:source",
                validation_queries=["test"],
            )

        # Verify default parameters were used
        mock_monitor.assert_called_once_with(
            "minimal_alias",
            duration_seconds=300,  # Default health_check_duration
            check_interval=10,  # Default health_check_interval
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_deployment_operations(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test that concurrent deployment operations work correctly."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        # Create multiple deployment tasks
        tasks = []
        for i in range(3):
            with patch("src.services.deployment.blue_green.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = (
                    f"2024010{i}_120000"
                )
                mock_datetime.now.return_value.isoformat.return_value = (
                    f"2024-01-0{i}T12:00:00"
                )

                task = blue_green_deployment.deploy_new_version(
                    alias_name=f"concurrent_alias_{i}",
                    data_source="collection:source",
                    validation_queries=["test"],
                )
                tasks.append(task)

        # Wait for all deployments to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (or at least not crash)
        for result in results:
            assert not isinstance(result, Exception) or isinstance(result, ServiceError)

    @pytest.mark.asyncio
    async def test_deployment_with_empty_validation_queries(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment with empty validation queries list."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with patch("src.services.deployment.blue_green.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            result = await blue_green_deployment.deploy_new_version(
                alias_name="empty_validation_alias",
                data_source="collection:source",
                validation_queries=[],  # Empty list
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_deployment_cleanup_task_creation(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test that cleanup task is properly created without RUF006 warning."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with (
            patch("src.services.deployment.blue_green.datetime") as mock_datetime,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )

            await blue_green_deployment.deploy_new_version(
                alias_name="cleanup_test_alias",
                data_source="collection:source",
                validation_queries=["test"],
            )

        # Verify that asyncio.create_task was called for cleanup
        mock_create_task.assert_called()
        call_args = mock_create_task.call_args[0][0]

        # Verify it's a coroutine for safe_delete_collection
        assert hasattr(call_args, "__await__")  # It's a coroutine
