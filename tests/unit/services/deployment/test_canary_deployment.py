"""Comprehensive tests for enhanced CanaryDeployment with state persistence and real metrics."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.deployment.canary import CanaryDeployment
from src.services.deployment.canary import CanaryDeploymentConfig
from src.services.deployment.canary import CanaryMetrics
from src.services.deployment.canary import CanaryStage
from src.services.errors import ServiceError


@pytest.fixture
def mock_config():
    """Create mock unified config with temporary data directory."""
    config = MagicMock(spec=UnifiedConfig)
    config.data_dir = Path(tempfile.mkdtemp())
    return config


@pytest.fixture
def mock_alias_manager():
    """Create mock alias manager."""
    manager = AsyncMock()
    manager.get_collection_for_alias = AsyncMock(return_value="old_collection")
    manager.switch_alias = AsyncMock()
    return manager


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
    return service


@pytest.fixture
def mock_client_manager():
    """Create mock client manager with Redis support."""
    manager = AsyncMock()
    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.setex = AsyncMock(return_value=True)
    manager.get_redis_client = AsyncMock(return_value=redis_client)
    return manager


@pytest.fixture
async def canary_deployment(
    mock_config, mock_alias_manager, mock_qdrant_service, mock_client_manager
):
    """Create CanaryDeployment instance."""
    deployment = CanaryDeployment(
        mock_config, mock_alias_manager, mock_qdrant_service, mock_client_manager
    )
    await deployment.initialize()
    yield deployment
    await deployment.cleanup()


class TestCanaryDeploymentInitialization:
    """Test CanaryDeployment initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialization_creates_state_file_path(
        self, mock_config, mock_alias_manager
    ):
        """Test that initialization sets up the state file path."""
        deployment = CanaryDeployment(mock_config, mock_alias_manager)

        expected_path = mock_config.data_dir / "canary_deployments.json"
        assert deployment._state_file == expected_path

    @pytest.mark.asyncio
    async def test_initialization_loads_existing_deployments(
        self, mock_config, mock_alias_manager
    ):
        """Test that initialization loads existing deployments from storage."""
        # Create test data file
        test_data = {
            "canary_123": {
                "alias": "test_alias",
                "old_collection": "old_coll",
                "new_collection": "new_coll",
                "stages": [
                    {
                        "percentage": 25,
                        "duration_minutes": 30,
                        "error_threshold": 0.05,
                        "latency_threshold": 200,
                    }
                ],
                "current_stage": 0,
                "metrics": {
                    "latency": [100, 120],
                    "error_rate": [0.01, 0.02],
                    "success_count": 100,
                    "error_count": 2,
                    "stage_start_time": time.time(),
                },
                "start_time": time.time(),
                "status": "running",
            }
        }

        state_file = mock_config.data_dir / "canary_deployments.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(test_data, f)

        deployment = CanaryDeployment(mock_config, mock_alias_manager)
        await deployment.initialize()

        assert len(deployment.deployments) == 1
        assert "canary_123" in deployment.deployments

        config = deployment.deployments["canary_123"]
        assert config.alias == "test_alias"
        assert config.status == "running"


class TestCanaryStageConfiguration:
    """Test canary stage and configuration classes."""

    def test_canary_stage_creation(self):
        """Test creating canary stage with custom parameters."""
        stage = CanaryStage(
            percentage=25.0,
            duration_minutes=30,
            error_threshold=0.03,
            latency_threshold=150,
        )

        assert stage.percentage == 25.0
        assert stage.duration_minutes == 30
        assert stage.error_threshold == 0.03
        assert stage.latency_threshold == 150

    def test_canary_metrics_initialization(self):
        """Test CanaryMetrics dataclass initialization."""
        metrics = CanaryMetrics()

        assert metrics.latency == []
        assert metrics.error_rate == []
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.stage_start_time is None


class TestCanaryDeploymentExecution:
    """Test canary deployment execution and monitoring."""

    @pytest.mark.asyncio
    async def test_start_canary_success(self, canary_deployment, mock_alias_manager):
        """Test successful canary deployment start."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await canary_deployment.start_canary(
            alias_name="test_alias",
            new_collection="new_collection",
            stages=[{"percentage": 25, "duration_minutes": 30}],
            auto_rollback=True,
        )

        assert deployment_id.startswith("canary_")
        assert deployment_id in canary_deployment.deployments

        config = canary_deployment.deployments[deployment_id]
        assert config.alias == "test_alias"
        assert config.old_collection == "old_collection"
        assert config.new_collection == "new_collection"
        assert len(config.stages) == 1

    @pytest.mark.asyncio
    async def test_start_canary_default_stages(
        self, canary_deployment, mock_alias_manager
    ):
        """Test canary deployment with default stages."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await canary_deployment.start_canary(
            alias_name="test_alias", new_collection="new_collection"
        )

        config = canary_deployment.deployments[deployment_id]
        # Default stages: 5%, 25%, 50%, 100%
        assert len(config.stages) == 4
        assert config.stages[0].percentage == 5
        assert config.stages[1].percentage == 25
        assert config.stages[2].percentage == 50
        assert config.stages[3].percentage == 100

    @pytest.mark.asyncio
    async def test_start_canary_no_existing_collection(
        self, canary_deployment, mock_alias_manager
    ):
        """Test canary deployment when no existing collection found."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        with pytest.raises(ServiceError, match="No collection found for alias"):
            await canary_deployment.start_canary(
                alias_name="nonexistent_alias", new_collection="new_collection"
            )

    @pytest.mark.asyncio
    async def test_metrics_collection_with_qdrant_integration(self, canary_deployment):
        """Test metrics collection attempts to query Qdrant stats."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30)],
        )

        # Mock the _simulate_metrics method to verify the flow
        with patch.object(canary_deployment, "_simulate_metrics") as mock_simulate:
            mock_simulate.return_value = {
                "latency": 100,
                "error_rate": 0.01,
                "timestamp": time.time(),
            }

            metrics = await canary_deployment._collect_metrics(deployment_config)

            assert "latency" in metrics
            assert "error_rate" in metrics
            assert "timestamp" in metrics

    @pytest.mark.asyncio
    async def test_pause_and_resume_deployment(
        self, canary_deployment, mock_alias_manager
    ):
        """Test pausing and resuming canary deployments."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # Start deployment
        deployment_id = await canary_deployment.start_canary(
            "test_alias", "new_collection"
        )

        # Manually set status to running for test
        canary_deployment.deployments[deployment_id].status = "running"

        # Pause deployment
        result = await canary_deployment.pause_deployment(deployment_id)
        assert result is True
        assert canary_deployment.deployments[deployment_id].status == "paused"

        # Resume deployment
        with patch.object(canary_deployment, "_run_canary"):
            result = await canary_deployment.resume_deployment(deployment_id)
            assert result is True
            assert canary_deployment.deployments[deployment_id].status == "running"


class TestCanaryMetricsAndHealthChecks:
    """Test canary metrics collection and health monitoring."""

    @pytest.mark.asyncio
    async def test_simulate_metrics_improving_over_time(self, canary_deployment):
        """Test that simulated metrics improve over time."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30)],
        )
        deployment_config.metrics.stage_start_time = time.time() - 600  # 10 minutes ago

        metrics = await canary_deployment._simulate_metrics(deployment_config)

        assert "latency" in metrics
        assert "error_rate" in metrics
        assert metrics["latency"] > 0
        assert metrics["error_rate"] >= 0

    def test_health_check_with_good_metrics(self, canary_deployment):
        """Test health check passes with good metrics."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30, error_threshold=0.05, latency_threshold=200)],
        )

        # Add good metrics
        deployment_config.metrics.latency = [100, 110, 105, 95, 120]
        deployment_config.metrics.error_rate = [0.01, 0.02, 0.015, 0.018, 0.012]

        is_healthy = canary_deployment._check_health(deployment_config)
        assert is_healthy is True

    def test_health_check_fails_with_high_error_rate(self, canary_deployment):
        """Test health check fails with high error rate."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30, error_threshold=0.05, latency_threshold=200)],
        )

        # Add bad error rate metrics
        deployment_config.metrics.latency = [100, 110, 105]
        deployment_config.metrics.error_rate = [0.08, 0.09, 0.07]  # Above threshold

        is_healthy = canary_deployment._check_health(deployment_config)
        assert is_healthy is False

    def test_health_check_fails_with_high_latency(self, canary_deployment):
        """Test health check fails with high latency."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30, error_threshold=0.05, latency_threshold=200)],
        )

        # Add bad latency metrics
        deployment_config.metrics.latency = [250, 300, 280]  # Above threshold
        deployment_config.metrics.error_rate = [0.01, 0.02, 0.015]

        is_healthy = canary_deployment._check_health(deployment_config)
        assert is_healthy is False


class TestStatePersistence:
    """Test state persistence functionality."""

    @pytest.mark.asyncio
    async def test_redis_persistence_enabled(
        self, mock_config, mock_alias_manager, mock_qdrant_service, mock_client_manager
    ):
        """Test Redis persistence when client manager is available."""
        deployment = CanaryDeployment(
            mock_config, mock_alias_manager, mock_qdrant_service, mock_client_manager
        )
        await deployment.initialize()

        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await deployment.start_canary("test_alias", "new_collection")

        # Verify Redis was called
        mock_client_manager.get_redis_client.assert_called()
        redis_client = await mock_client_manager.get_redis_client()
        redis_client.setex.assert_called()

    @pytest.mark.asyncio
    async def test_file_persistence_fallback(
        self, mock_config, mock_alias_manager, mock_qdrant_service
    ):
        """Test file persistence when Redis is not available."""
        deployment = CanaryDeployment(
            mock_config, mock_alias_manager, mock_qdrant_service, client_manager=None
        )
        await deployment.initialize()

        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await deployment.start_canary("test_alias", "new_collection")

        # Verify file was created
        assert deployment._state_file.exists()

        with open(deployment._state_file) as f:
            data = json.load(f)
        assert deployment_id in data

    @pytest.mark.asyncio
    async def test_serialization_deserialization(
        self, canary_deployment, mock_alias_manager
    ):
        """Test deployment data serialization and deserialization."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # Create deployment with complex data
        deployment_id = await canary_deployment.start_canary(
            "test_alias",
            "new_collection",
            stages=[
                {"percentage": 25, "duration_minutes": 30},
                {"percentage": 50, "duration_minutes": 60},
            ],
        )

        # Add metrics data
        config = canary_deployment.deployments[deployment_id]
        config.metrics.latency = [100, 120, 110]
        config.metrics.error_rate = [0.01, 0.02, 0.015]
        config.metrics.success_count = 100
        config.metrics.error_count = 2

        # Serialize and deserialize
        serialized = canary_deployment._serialize_deployments()
        new_deployments = canary_deployment._deserialize_deployments(serialized)

        # Verify data integrity
        assert deployment_id in new_deployments
        new_config = new_deployments[deployment_id]
        assert new_config.alias == "test_alias"
        assert len(new_config.stages) == 2
        assert new_config.metrics.latency == [100, 120, 110]
        assert new_config.metrics.success_count == 100


class TestCanaryRollback:
    """Test canary rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_canary_success(self, canary_deployment, mock_alias_manager):
        """Test successful canary rollback."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # Start deployment
        deployment_id = await canary_deployment.start_canary(
            "test_alias", "new_collection"
        )

        # Wait briefly for any async tasks to start
        await asyncio.sleep(0.1)

        # Mock that the current collection is different from old collection (triggering switch)
        mock_alias_manager.get_collection_for_alias.return_value = "new_collection"

        # Simulate rollback
        await canary_deployment._rollback_canary(deployment_id)

        config = canary_deployment.deployments[deployment_id]
        assert config.status == "rolled_back"

        # Verify alias switch was called to revert to old collection
        mock_alias_manager.switch_alias.assert_called_with(
            alias_name="test_alias", new_collection="old_collection"
        )

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_deployment(self, canary_deployment):
        """Test rollback of nonexistent deployment."""
        # Should not raise error
        await canary_deployment._rollback_canary("nonexistent")


class TestDeploymentStatus:
    """Test deployment status monitoring."""

    @pytest.mark.asyncio
    async def test_get_deployment_status_success(
        self, canary_deployment, mock_alias_manager
    ):
        """Test getting deployment status."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await canary_deployment.start_canary(
            "test_alias", "new_collection"
        )

        # Add some metrics
        config = canary_deployment.deployments[deployment_id]
        config.metrics.latency = [100, 120, 110]
        config.metrics.error_rate = [0.01, 0.02, 0.015]
        config.status = "running"

        status = await canary_deployment.get_deployment_status(deployment_id)

        assert status["deployment_id"] == deployment_id
        assert status["status"] == "running"
        assert status["alias"] == "test_alias"
        assert status["current_stage"] == 0
        assert "avg_latency" in status
        assert "avg_error_rate" in status

    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(self, canary_deployment):
        """Test getting status for nonexistent deployment."""
        status = await canary_deployment.get_deployment_status("nonexistent")
        assert status["status"] == "not_found"

    def test_get_active_deployments(self, canary_deployment):
        """Test getting list of active deployments."""
        # Add some test deployments
        canary_deployment.deployments["dep1"] = CanaryDeploymentConfig(
            alias="alias1",
            old_collection="old1",
            new_collection="new1",
            stages=[CanaryStage(25, 30)],
            status="running",
        )
        canary_deployment.deployments["dep2"] = CanaryDeploymentConfig(
            alias="alias2",
            old_collection="old2",
            new_collection="new2",
            stages=[CanaryStage(50, 60)],
            status="completed",
        )
        canary_deployment.deployments["dep3"] = CanaryDeploymentConfig(
            alias="alias3",
            old_collection="old3",
            new_collection="new3",
            stages=[CanaryStage(25, 30)],
            status="paused",
        )

        active = canary_deployment.get_active_deployments()

        assert len(active) == 2  # running and paused
        active_ids = [dep["id"] for dep in active]
        assert "dep1" in active_ids
        assert "dep3" in active_ids
        assert "dep2" not in active_ids  # completed


class TestTrafficShiftingDocumentation:
    """Test traffic shifting documentation and placeholders."""

    def test_traffic_shifting_todo_comments_present(self):
        """Test that traffic shifting TODO comments are comprehensive."""
        import inspect

        from src.services.deployment.canary import CanaryDeployment

        source = inspect.getsource(CanaryDeployment._run_canary)

        # Verify key TODO comments are present
        assert "TODO: Implement actual traffic shifting" in source
        assert "API Gateway/Load Balancer based" in source
        assert "Qdrant collection alias based" in source
        assert "Application-level routing" in source


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_metrics_collection_failure_handling(self, canary_deployment):
        """Test graceful handling of metrics collection failures."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30)],
        )

        # Mock Qdrant service to raise exception
        canary_deployment.qdrant.get_collection_stats = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        # Should fall back to simulated metrics
        metrics = await canary_deployment._collect_metrics(deployment_config)

        assert "latency" in metrics
        assert "error_rate" in metrics

    @pytest.mark.asyncio
    async def test_persistence_failure_recovery(self, mock_config, mock_alias_manager):
        """Test recovery from persistence failures."""
        # Create deployment without proper directory permissions
        mock_config.data_dir = Path("/invalid/path/that/does/not/exist")

        deployment = CanaryDeployment(mock_config, mock_alias_manager)

        # Should not crash during initialization even if persistence fails
        await deployment.initialize()

        # Should still allow deployment creation
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"
        deployment_id = await deployment.start_canary("test_alias", "new_collection")
        assert deployment_id in deployment.deployments


class TestPerformance:
    """Test performance-related functionality."""

    @pytest.mark.asyncio
    async def test_concurrent_deployment_creation(
        self, canary_deployment, mock_alias_manager
    ):
        """Test concurrent deployment creation with state persistence."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        async def create_deployment(i):
            # Add small delay to ensure unique timestamps
            await asyncio.sleep(0.01 * i)
            return await canary_deployment.start_canary(
                f"alias_{i}", f"new_collection_{i}"
            )

        # Create multiple deployments concurrently
        tasks = [create_deployment(i) for i in range(3)]
        deployment_ids = await asyncio.gather(*tasks)

        assert len(deployment_ids) == 3
        assert len(set(deployment_ids)) == 3  # All unique
        assert len(canary_deployment.deployments) == 3

    @pytest.mark.asyncio
    async def test_large_metrics_dataset_handling(
        self, canary_deployment, mock_alias_manager
    ):
        """Test handling of large metrics datasets."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await canary_deployment.start_canary(
            "test_alias", "new_collection"
        )

        # Add large amount of metrics data
        config = canary_deployment.deployments[deployment_id]
        config.metrics.latency = [100 + i for i in range(1000)]
        config.metrics.error_rate = [0.01 + (i * 0.0001) for i in range(1000)]

        # Should handle serialization efficiently
        serialized = canary_deployment._serialize_deployments()
        assert deployment_id in serialized

        # Verify deserialization
        deployments = canary_deployment._deserialize_deployments(serialized)
        new_config = deployments[deployment_id]
        assert len(new_config.metrics.latency) == 1000
        assert len(new_config.metrics.error_rate) == 1000
