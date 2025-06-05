"""Comprehensive tests for enhanced BlueGreenDeployment with production monitoring."""

import asyncio
import tempfile
import time
from pathlib import Path
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
    config.data_dir = Path(tempfile.mkdtemp())
    return config


@pytest.fixture
def mock_qdrant_service():
    """Create mock Qdrant service."""
    service = AsyncMock()
    service.get_collection_stats = AsyncMock(
        return_value={
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
            "points_count": 1000,
        }
    )
    service.query = AsyncMock(
        return_value=[
            {"id": "doc1", "score": 0.95, "payload": {"content": "test"}},
            {"id": "doc2", "score": 0.85, "payload": {"content": "test2"}},
        ]
    )
    service._client = AsyncMock()
    service._client.delete_collection = AsyncMock()
    return service


@pytest.fixture
def mock_alias_manager():
    """Create mock alias manager."""
    manager = AsyncMock()
    manager.get_collection_for_alias = AsyncMock(return_value="blue_collection")
    manager.clone_collection_schema = AsyncMock()
    manager.switch_alias = AsyncMock()
    manager.copy_collection_data = AsyncMock()
    manager.safe_delete_collection = AsyncMock()
    return manager


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = AsyncMock()
    manager.generate_embedding = AsyncMock(
        return_value={"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "model": "test-model"}
    )
    return manager


@pytest.fixture
async def blue_green_deployment(
    mock_config, mock_qdrant_service, mock_alias_manager, mock_embedding_manager
):
    """Create BlueGreenDeployment instance."""
    deployment = BlueGreenDeployment(
        mock_config, mock_qdrant_service, mock_alias_manager, mock_embedding_manager
    )
    await deployment.initialize()
    yield deployment
    await deployment.cleanup()


class TestBlueGreenDeploymentInitialization:
    """Test BlueGreenDeployment initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialization_success(
        self, mock_config, mock_qdrant_service, mock_alias_manager
    ):
        """Test successful initialization."""
        deployment = BlueGreenDeployment(
            mock_config, mock_qdrant_service, mock_alias_manager
        )

        await deployment.initialize()
        assert deployment._initialized is True

        await deployment.cleanup()
        assert deployment._initialized is False

    def test_initialization_with_optional_embedding_manager(
        self,
        mock_config,
        mock_qdrant_service,
        mock_alias_manager,
        mock_embedding_manager,
    ):
        """Test initialization with embedding manager."""
        deployment = BlueGreenDeployment(
            mock_config, mock_qdrant_service, mock_alias_manager, mock_embedding_manager
        )

        assert deployment.embeddings == mock_embedding_manager


class TestBlueGreenDeploymentExecution:
    """Test blue-green deployment execution flow."""

    @pytest.mark.asyncio
    async def test_deploy_new_version_success(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test successful blue-green deployment."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        # Mock the collection population
        with (
            patch.object(
                blue_green_deployment, "_populate_collection"
            ) as mock_populate,
            patch.object(
                blue_green_deployment, "_validate_collection"
            ) as mock_validate,
            patch.object(
                blue_green_deployment, "_monitor_after_switch"
            ) as mock_monitor,
        ):
            mock_validate.return_value = True

            result = await blue_green_deployment.deploy_new_version(
                alias_name="test_alias",
                data_source="collection:source_collection",
                validation_queries=["test query"],
                rollback_on_failure=True,
                validation_threshold=0.7,
            )

        assert result["success"] is True
        assert result["old_collection"] == "blue_collection"
        assert result["alias"] == "test_alias"
        assert "new_collection" in result
        assert "deployed_at" in result

        # Verify the deployment flow
        mock_alias_manager.clone_collection_schema.assert_called_once()
        mock_populate.assert_called_once()
        mock_validate.assert_called_once()
        mock_alias_manager.switch_alias.assert_called_once()
        mock_monitor.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_new_version_no_existing_collection(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment when no existing collection is found."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        with pytest.raises(ServiceError, match="No collection found for alias"):
            await blue_green_deployment.deploy_new_version(
                alias_name="nonexistent_alias",
                data_source="collection:source",
                validation_queries=["test"],
            )

    @pytest.mark.asyncio
    async def test_deploy_new_version_validation_failure(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment with validation failure and rollback."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with (
            patch.object(blue_green_deployment, "_populate_collection"),
            patch.object(
                blue_green_deployment, "_validate_collection"
            ) as mock_validate,
            patch.object(blue_green_deployment, "_rollback") as mock_rollback,
        ):
            mock_validate.return_value = False

            with pytest.raises(ServiceError, match="Validation failed"):
                await blue_green_deployment.deploy_new_version(
                    alias_name="test_alias",
                    data_source="collection:source",
                    validation_queries=["test"],
                    rollback_on_failure=True,
                )

            mock_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_new_version_no_rollback_on_failure(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment failure without rollback."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with (
            patch.object(blue_green_deployment, "_populate_collection"),
            patch.object(
                blue_green_deployment, "_validate_collection"
            ) as mock_validate,
            patch.object(blue_green_deployment, "_rollback") as mock_rollback,
        ):
            mock_validate.return_value = False

            with pytest.raises(ServiceError):
                await blue_green_deployment.deploy_new_version(
                    alias_name="test_alias",
                    data_source="collection:source",
                    validation_queries=["test"],
                    rollback_on_failure=False,
                )

            mock_rollback.assert_not_called()


class TestDataSourceHandling:
    """Test different data source handling."""

    @pytest.mark.asyncio
    async def test_populate_collection_from_existing(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test populating collection from existing collection."""
        await blue_green_deployment._populate_collection(
            "new_collection", "collection:source_collection"
        )

        mock_alias_manager.copy_collection_data.assert_called_once_with(
            source="source_collection", target="new_collection"
        )

    @pytest.mark.asyncio
    async def test_populate_collection_backup_not_implemented(
        self, blue_green_deployment
    ):
        """Test that backup restore raises NotImplementedError with guidance."""
        with pytest.raises(
            NotImplementedError,
            match="Backup restore.*not yet implemented.*Integration needed",
        ):
            await blue_green_deployment._populate_collection(
                "new_collection", "backup:/path/to/backup"
            )

    @pytest.mark.asyncio
    async def test_populate_collection_crawl_not_implemented(
        self, blue_green_deployment
    ):
        """Test that fresh crawl raises NotImplementedError with guidance."""
        with pytest.raises(
            NotImplementedError,
            match="Fresh crawl.*not yet implemented.*Integration needed",
        ):
            await blue_green_deployment._populate_collection(
                "new_collection", "crawl:config_name"
            )

    @pytest.mark.asyncio
    async def test_populate_collection_unknown_source(self, blue_green_deployment):
        """Test handling of unknown data source type."""
        with pytest.raises(ServiceError, match="Unknown data source type"):
            await blue_green_deployment._populate_collection(
                "new_collection", "unknown:source"
            )


class TestCollectionValidation:
    """Test collection validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_collection_success(
        self, blue_green_deployment, mock_qdrant_service, mock_embedding_manager
    ):
        """Test successful collection validation."""
        mock_qdrant_service.query.return_value = [
            {"id": "doc1", "score": 0.8, "payload": {"content": "test"}},
            {"id": "doc2", "score": 0.75, "payload": {"content": "test2"}},
            {"id": "doc3", "score": 0.72, "payload": {"content": "test3"}},
            {"id": "doc4", "score": 0.71, "payload": {"content": "test4"}},
            {"id": "doc5", "score": 0.70, "payload": {"content": "test5"}},
        ]

        result = await blue_green_deployment._validate_collection(
            "test_collection", ["test query 1", "test query 2"], threshold=0.7
        )

        assert result is True
        assert mock_embedding_manager.generate_embedding.call_count == 2
        assert mock_qdrant_service.query.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_collection_insufficient_results(
        self, blue_green_deployment, mock_qdrant_service
    ):
        """Test validation failure due to insufficient results."""
        mock_qdrant_service.query.return_value = [
            {"id": "doc1", "score": 0.8, "payload": {"content": "test"}},
        ]  # Only 1 result, need 5

        result = await blue_green_deployment._validate_collection(
            "test_collection", ["test query"], threshold=0.7
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_collection_low_score(
        self, blue_green_deployment, mock_qdrant_service
    ):
        """Test validation failure due to low score."""
        mock_qdrant_service.query.return_value = [
            {
                "id": "doc1",
                "score": 0.5,
                "payload": {"content": "test"},
            },  # Below threshold
            {"id": "doc2", "score": 0.6, "payload": {"content": "test2"}},
            {"id": "doc3", "score": 0.55, "payload": {"content": "test3"}},
            {"id": "doc4", "score": 0.52, "payload": {"content": "test4"}},
            {"id": "doc5", "score": 0.58, "payload": {"content": "test5"}},
        ]

        result = await blue_green_deployment._validate_collection(
            "test_collection", ["test query"], threshold=0.7
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_collection_no_embedding_manager(
        self, mock_config, mock_qdrant_service, mock_alias_manager
    ):
        """Test validation skips when no embedding manager available."""
        deployment = BlueGreenDeployment(
            mock_config, mock_qdrant_service, mock_alias_manager, embedding_manager=None
        )

        result = await deployment._validate_collection(
            "test_collection", ["test query"]
        )

        assert result is True  # Should skip validation

    @pytest.mark.asyncio
    async def test_validate_collection_embedding_error(
        self, blue_green_deployment, mock_embedding_manager
    ):
        """Test validation handles embedding generation errors."""
        mock_embedding_manager.generate_embedding.side_effect = Exception(
            "Embedding failed"
        )

        result = await blue_green_deployment._validate_collection(
            "test_collection", ["test query"]
        )

        assert result is False


class TestEnhancedMonitoring:
    """Test enhanced monitoring functionality."""

    @pytest.mark.asyncio
    async def test_monitor_after_switch_comprehensive(
        self,
        blue_green_deployment,
        mock_alias_manager,
        mock_qdrant_service,
        mock_embedding_manager,
    ):
        """Test comprehensive monitoring after alias switch."""
        mock_alias_manager.get_collection_for_alias.return_value = "green_collection"
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
            "points_count": 1000,
        }

        # Mock search performance test
        mock_qdrant_service.query.return_value = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "test"}},
        ]

        # Use shorter duration for testing
        await blue_green_deployment._monitor_after_switch(
            "test_alias",
            duration_seconds=2,  # Short duration for test
            check_interval=1,
        )

        # Verify monitoring calls were made
        mock_alias_manager.get_collection_for_alias.assert_called()
        mock_qdrant_service.get_collection_stats.assert_called()
        mock_embedding_manager.generate_embedding.assert_called()
        mock_qdrant_service.query.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_after_switch_alias_not_found(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test monitoring handles alias not found errors."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        # Should complete without raising exception
        await blue_green_deployment._monitor_after_switch(
            "nonexistent_alias", duration_seconds=1, check_interval=1
        )

    @pytest.mark.asyncio
    async def test_monitor_after_switch_empty_collection(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test monitoring handles empty collection."""
        mock_alias_manager.get_collection_for_alias.return_value = "empty_collection"
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 0,  # Empty collection
            "indexed_vectors_count": 0,
            "points_count": 0,
        }

        # Should complete without raising exception
        await blue_green_deployment._monitor_after_switch(
            "test_alias", duration_seconds=1, check_interval=1
        )

    @pytest.mark.asyncio
    async def test_monitor_after_switch_high_search_latency(
        self,
        blue_green_deployment,
        mock_alias_manager,
        mock_qdrant_service,
        mock_embedding_manager,
    ):
        """Test monitoring detects high search latency."""
        mock_alias_manager.get_collection_for_alias.return_value = "slow_collection"

        # Mock slow search
        async def slow_query(*args, **kwargs):
            await asyncio.sleep(1.1)  # Simulate > 1 second latency
            return [{"id": "doc1", "score": 0.9}]

        mock_qdrant_service.query.side_effect = slow_query

        # Should complete but log warnings about high latency
        await blue_green_deployment._monitor_after_switch(
            "test_alias", duration_seconds=1, check_interval=1
        )

    @pytest.mark.asyncio
    async def test_monitor_after_switch_too_many_errors(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test monitoring stops after too many errors."""
        # Mock alias manager to consistently return None (error condition)
        mock_alias_manager.get_collection_for_alias.return_value = None

        with pytest.raises(ServiceError, match="Too many errors"):
            await blue_green_deployment._monitor_after_switch(
                "failing_alias",
                duration_seconds=2,  # Shorter duration for test
                check_interval=0.1,  # Fast checks to trigger error limit quickly
            )

    def test_log_monitoring_summary_with_data(self, blue_green_deployment):
        """Test monitoring summary logging with metrics data."""
        metrics = {
            "search_latencies": [
                {"latency_ms": 100, "timestamp": time.time()},
                {"latency_ms": 120, "timestamp": time.time()},
                {"latency_ms": 90, "timestamp": time.time()},
            ],
            "collection_stats": [
                {
                    "vectors_count": 1000,
                    "indexed_vectors_count": 1000,
                    "timestamp": time.time(),
                }
            ],
        }

        # Should not raise exception
        blue_green_deployment._log_monitoring_summary("test_alias", metrics, final=True)

    def test_log_monitoring_summary_empty_data(self, blue_green_deployment):
        """Test monitoring summary logging with empty data."""
        metrics = {"search_latencies": [], "collection_stats": []}

        # Should not raise exception
        blue_green_deployment._log_monitoring_summary("test_alias", metrics)


class TestRollbackFunctionality:
    """Test rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_success(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test successful rollback operation."""
        await blue_green_deployment._rollback(
            "test_alias", "blue_collection", "failed_green_collection"
        )

        # Verify alias was switched back
        mock_alias_manager.switch_alias.assert_called_once_with(
            alias_name="test_alias", new_collection="blue_collection"
        )

        # Verify failed collection was deleted
        mock_qdrant_service._client.delete_collection.assert_called_once_with(
            "failed_green_collection"
        )

    @pytest.mark.asyncio
    async def test_rollback_alias_switch_failure(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test rollback handles alias switch failure."""
        mock_alias_manager.switch_alias.side_effect = Exception("Switch failed")

        # Should not raise exception
        await blue_green_deployment._rollback(
            "test_alias", "blue_collection", "failed_green_collection"
        )

    @pytest.mark.asyncio
    async def test_rollback_collection_deletion_failure(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test rollback handles collection deletion failure."""
        mock_qdrant_service._client.delete_collection.side_effect = Exception(
            "Delete failed"
        )

        # Should not raise exception
        await blue_green_deployment._rollback(
            "test_alias", "blue_collection", "failed_green_collection"
        )


class TestDeploymentStatus:
    """Test deployment status functionality."""

    @pytest.mark.asyncio
    async def test_get_deployment_status_success(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test getting deployment status for active alias."""
        mock_alias_manager.get_collection_for_alias.return_value = "active_collection"
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
        }

        status = await blue_green_deployment.get_deployment_status("test_alias")

        assert status["alias"] == "test_alias"
        assert status["status"] == "active"
        assert status["collection"] == "active_collection"
        assert status["vectors_count"] == 1000
        assert status["indexed_vectors_count"] == 1000

    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test getting status for nonexistent alias."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        status = await blue_green_deployment.get_deployment_status("nonexistent_alias")

        assert status["alias"] == "nonexistent_alias"
        assert status["status"] == "not_found"
        assert status["collection"] is None

    @pytest.mark.asyncio
    async def test_get_deployment_status_error(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test getting status when stats collection fails."""
        mock_alias_manager.get_collection_for_alias.return_value = "error_collection"
        mock_qdrant_service.get_collection_stats.side_effect = Exception("Stats failed")

        status = await blue_green_deployment.get_deployment_status("test_alias")

        assert status["alias"] == "test_alias"
        assert status["status"] == "error"
        assert status["collection"] == "error_collection"
        assert "error" in status


class TestProductionMonitoringIntegration:
    """Test production monitoring integration points."""

    def test_monitoring_todo_comments_present(self):
        """Test that monitoring TODO comments are comprehensive."""
        import inspect

        from src.services.deployment.blue_green import BlueGreenDeployment

        source = inspect.getsource(BlueGreenDeployment._monitor_after_switch)

        # Verify key TODO comments for production integration
        assert "TODO: Integration with external monitoring systems" in source
        assert "Query APM systems" in source
        assert "Check application logs" in source
        assert "Monitor downstream service health" in source
        assert "Track business metrics" in source

    @pytest.mark.asyncio
    async def test_search_performance_monitoring(
        self,
        blue_green_deployment,
        mock_alias_manager,
        mock_qdrant_service,
        mock_embedding_manager,
    ):
        """Test search performance monitoring functionality."""
        mock_alias_manager.get_collection_for_alias.return_value = "test_collection"

        # Mock fast search
        mock_qdrant_service.query.return_value = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "test"}},
        ]

        await blue_green_deployment._monitor_after_switch(
            "test_alias", duration_seconds=1, check_interval=1
        )

        # Verify search performance test was executed
        mock_embedding_manager.generate_embedding.assert_called_with("test query")
        mock_qdrant_service.query.assert_called()

    @pytest.mark.asyncio
    async def test_collection_statistics_monitoring(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test collection statistics monitoring."""
        mock_alias_manager.get_collection_for_alias.return_value = (
            "monitored_collection"
        )
        mock_qdrant_service.get_collection_stats.return_value = {
            "vectors_count": 5000,
            "indexed_vectors_count": 4800,
            "points_count": 5000,
        }

        await blue_green_deployment._monitor_after_switch(
            "test_alias", duration_seconds=1, check_interval=1
        )

        # Verify collection stats were monitored
        mock_qdrant_service.get_collection_stats.assert_called_with(
            "monitored_collection"
        )


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_deploy_with_general_exception(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment handles general exceptions gracefully."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"
        mock_alias_manager.clone_collection_schema.side_effect = Exception(
            "Unexpected error"
        )

        with pytest.raises(ServiceError, match="Deployment failed"):
            await blue_green_deployment.deploy_new_version(
                alias_name="test_alias",
                data_source="collection:source",
                validation_queries=["test"],
            )

    @pytest.mark.asyncio
    async def test_validation_with_query_error(
        self, blue_green_deployment, mock_qdrant_service
    ):
        """Test validation handles query errors gracefully."""
        mock_qdrant_service.query.side_effect = Exception("Query failed")

        result = await blue_green_deployment._validate_collection(
            "test_collection", ["test query"]
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_monitoring_with_stats_error(
        self, blue_green_deployment, mock_alias_manager, mock_qdrant_service
    ):
        """Test monitoring handles stats collection errors."""
        mock_alias_manager.get_collection_for_alias.return_value = "test_collection"
        mock_qdrant_service.get_collection_stats.side_effect = Exception("Stats failed")

        # Should complete without raising exception
        await blue_green_deployment._monitor_after_switch(
            "test_alias", duration_seconds=1, check_interval=1
        )


class TestPerformance:
    """Test performance-related functionality."""

    @pytest.mark.asyncio
    async def test_concurrent_deployments_not_supported(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test that concurrent deployments to same alias are handled properly."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with (
            patch.object(blue_green_deployment, "_populate_collection"),
            patch.object(
                blue_green_deployment, "_validate_collection"
            ) as mock_validate,
            patch.object(blue_green_deployment, "_monitor_after_switch"),
        ):
            mock_validate.return_value = True

            # Start two deployments to same alias
            task1 = blue_green_deployment.deploy_new_version(
                "same_alias", "collection:source1", ["test1"]
            )
            task2 = blue_green_deployment.deploy_new_version(
                "same_alias", "collection:source2", ["test2"]
            )

            # Both should complete (blue-green doesn't prevent concurrent deployments)
            results = await asyncio.gather(task1, task2, return_exceptions=True)

            # At least one should succeed
            successes = [r for r in results if isinstance(r, dict) and r.get("success")]
            assert len(successes) >= 1

    @pytest.mark.asyncio
    async def test_deployment_with_large_validation_set(
        self, blue_green_deployment, mock_alias_manager
    ):
        """Test deployment with large validation query set."""
        mock_alias_manager.get_collection_for_alias.return_value = "blue_collection"

        with (
            patch.object(blue_green_deployment, "_populate_collection"),
            patch.object(
                blue_green_deployment, "_validate_collection"
            ) as mock_validate,
            patch.object(blue_green_deployment, "_monitor_after_switch"),
        ):
            mock_validate.return_value = True

            # Large validation set
            validation_queries = [f"test query {i}" for i in range(100)]

            result = await blue_green_deployment.deploy_new_version(
                "test_alias", "collection:source", validation_queries
            )

            assert result["success"] is True
            mock_validate.assert_called_once_with(
                mock_validate.call_args[0][0],  # collection name (dynamic)
                validation_queries,
                threshold=0.7,
            )
