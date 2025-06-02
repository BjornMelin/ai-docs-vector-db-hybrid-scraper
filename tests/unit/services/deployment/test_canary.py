"""Tests for Canary deployment strategies."""

import time
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
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    config.performance = MagicMock()
    config.performance.timeout = 30
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
        }
    )
    return service


@pytest.fixture
def canary_deployment(mock_config, mock_alias_manager, mock_qdrant_service):
    """Create CanaryDeployment instance."""
    deployment = CanaryDeployment(
        config=mock_config,
        alias_manager=mock_alias_manager,
        qdrant_service=mock_qdrant_service,
    )
    return deployment


class TestCanaryStage:
    """Test CanaryStage dataclass."""

    def test_canary_stage_creation(self):
        """Test creating canary stage."""
        stage = CanaryStage(
            percentage=25.0,
            duration_minutes=60,
            error_threshold=0.02,
            latency_threshold=150.0,
        )

        assert stage.percentage == 25.0
        assert stage.duration_minutes == 60
        assert stage.error_threshold == 0.02
        assert stage.latency_threshold == 150.0

    def test_canary_stage_defaults(self):
        """Test default values for canary stage."""
        stage = CanaryStage(
            percentage=50.0,
            duration_minutes=30,
        )

        assert stage.percentage == 50.0
        assert stage.duration_minutes == 30
        assert stage.error_threshold == 0.05  # Default 5%
        assert stage.latency_threshold == 200  # Default 200ms


class TestCanaryMetrics:
    """Test CanaryMetrics dataclass."""

    def test_canary_metrics_creation(self):
        """Test creating canary metrics."""
        metrics = CanaryMetrics()

        assert metrics.latency == []
        assert metrics.error_rate == []
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.stage_start_time is None

    def test_canary_metrics_data_tracking(self):
        """Test tracking data in canary metrics."""
        metrics = CanaryMetrics()

        metrics.latency.extend([100.5, 95.2, 110.8])
        metrics.error_rate.extend([0.01, 0.02, 0.015])
        metrics.success_count = 150
        metrics.error_count = 3
        metrics.stage_start_time = time.time()

        assert len(metrics.latency) == 3
        assert len(metrics.error_rate) == 3
        assert metrics.success_count == 150
        assert metrics.error_count == 3
        assert metrics.stage_start_time is not None


class TestCanaryDeploymentConfig:
    """Test CanaryDeploymentConfig dataclass."""

    def test_canary_deployment_config_creation(self):
        """Test creating canary deployment config."""
        stages = [
            CanaryStage(percentage=10, duration_minutes=30),
            CanaryStage(percentage=50, duration_minutes=60),
            CanaryStage(percentage=100, duration_minutes=0),
        ]

        config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_coll",
            new_collection="new_coll",
            stages=stages,
        )

        assert config.alias == "test_alias"
        assert config.old_collection == "old_coll"
        assert config.new_collection == "new_coll"
        assert len(config.stages) == 3
        assert config.current_stage == 0
        assert isinstance(config.metrics, CanaryMetrics)
        assert config.status == "pending"
        assert isinstance(config.start_time, float)

    def test_canary_deployment_config_with_custom_values(self):
        """Test creating config with custom values."""
        custom_metrics = CanaryMetrics()
        custom_metrics.success_count = 100

        config = CanaryDeploymentConfig(
            alias="custom_alias",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(percentage=100, duration_minutes=0)],
            current_stage=1,
            metrics=custom_metrics,
            start_time=1234567890.0,
            status="running",
        )

        assert config.current_stage == 1
        assert config.metrics.success_count == 100
        assert config.start_time == 1234567890.0
        assert config.status == "running"


class TestCanaryDeployment:
    """Test CanaryDeployment class."""

    @pytest.mark.asyncio
    async def test_initialization(self, canary_deployment):
        """Test canary deployment initialization."""
        await canary_deployment.initialize()
        assert canary_deployment._initialized is True

        await canary_deployment.cleanup()
        assert canary_deployment._initialized is False

    @pytest.mark.asyncio
    async def test_initialization_without_qdrant_service(
        self, mock_config, mock_alias_manager
    ):
        """Test initialization without Qdrant service."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            qdrant_service=None,
        )

        await deployment.initialize()
        assert deployment._initialized is True
        assert deployment.qdrant is None

    @pytest.mark.asyncio
    async def test_start_canary_with_default_stages(
        self, canary_deployment, mock_alias_manager
    ):
        """Test starting canary with default stages."""
        mock_alias_manager.get_collection_for_alias.return_value = "existing_collection"

        with patch("time.time", return_value=1234567890):
            deployment_id = await canary_deployment.start_canary(
                alias_name="test_alias",
                new_collection="new_collection",
            )

        assert deployment_id == "canary_1234567890"
        assert deployment_id in canary_deployment.deployments

        deployment_config = canary_deployment.deployments[deployment_id]
        assert deployment_config.alias == "test_alias"
        assert deployment_config.old_collection == "existing_collection"
        assert deployment_config.new_collection == "new_collection"
        assert len(deployment_config.stages) == 4  # Default stages

        # Verify default stages
        assert deployment_config.stages[0].percentage == 5
        assert deployment_config.stages[1].percentage == 25
        assert deployment_config.stages[2].percentage == 50
        assert deployment_config.stages[3].percentage == 100

    @pytest.mark.asyncio
    async def test_start_canary_with_custom_stages(
        self, canary_deployment, mock_alias_manager
    ):
        """Test starting canary with custom stages."""
        mock_alias_manager.get_collection_for_alias.return_value = "existing_collection"

        custom_stages = [
            {"percentage": 10, "duration_minutes": 15},
            {
                "percentage": 100,
                "duration_minutes": 0,
                "error_threshold": 0.01,
                "latency_threshold": 100,
            },
        ]

        with patch("time.time", return_value=1234567890):
            deployment_id = await canary_deployment.start_canary(
                alias_name="custom_alias",
                new_collection="custom_new_collection",
                stages=custom_stages,
                auto_rollback=False,
            )

        deployment_config = canary_deployment.deployments[deployment_id]
        assert len(deployment_config.stages) == 2
        assert deployment_config.stages[0].percentage == 10
        assert deployment_config.stages[0].duration_minutes == 15
        assert deployment_config.stages[1].error_threshold == 0.01
        assert deployment_config.stages[1].latency_threshold == 100

    @pytest.mark.asyncio
    async def test_start_canary_no_existing_collection(
        self, canary_deployment, mock_alias_manager
    ):
        """Test starting canary when no existing collection found."""
        mock_alias_manager.get_collection_for_alias.return_value = None

        with pytest.raises(
            ServiceError, match="No collection found for alias test_alias"
        ):
            await canary_deployment.start_canary(
                alias_name="test_alias",
                new_collection="new_collection",
            )

    @pytest.mark.asyncio
    async def test_run_canary_complete_success(
        self, canary_deployment, mock_alias_manager
    ):
        """Test successful canary run through all stages."""
        # Create deployment config
        deployment_id = "test_deployment"
        stages = [
            CanaryStage(
                percentage=50, duration_minutes=0
            ),  # No duration for quick test
            CanaryStage(percentage=100, duration_minutes=0),
        ]
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=stages,
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock check_health to always return True
        with patch.object(canary_deployment, "_check_health", return_value=True):
            await canary_deployment._run_canary(deployment_id, auto_rollback=True)

        # Verify final state
        assert deployment_config.status == "completed"
        mock_alias_manager.switch_alias.assert_called_once_with(
            alias_name="test_alias",
            new_collection="new_collection",
        )

    @pytest.mark.asyncio
    async def test_run_canary_health_check_failure(
        self, canary_deployment, mock_alias_manager
    ):
        """Test canary run with health check failure."""
        deployment_id = "test_deployment"
        stages = [CanaryStage(percentage=50, duration_minutes=0)]
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=stages,
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock check_health to return False
        with (
            patch.object(canary_deployment, "_check_health", return_value=False),
            patch.object(canary_deployment, "_rollback_canary") as mock_rollback,
        ):
            await canary_deployment._run_canary(deployment_id, auto_rollback=True)

        # Verify rollback was called
        mock_rollback.assert_called_once_with(deployment_id)

    @pytest.mark.asyncio
    async def test_run_canary_health_check_failure_no_rollback(self, canary_deployment):
        """Test canary run with health check failure and no rollback."""
        deployment_id = "test_deployment"
        stages = [CanaryStage(percentage=50, duration_minutes=0)]
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=stages,
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock check_health to return False
        with patch.object(canary_deployment, "_check_health", return_value=False):
            await canary_deployment._run_canary(deployment_id, auto_rollback=False)

        # Verify status was set to failed
        assert deployment_config.status == "failed"

    @pytest.mark.asyncio
    async def test_run_canary_exception_with_rollback(self, canary_deployment):
        """Test canary run with exception and auto rollback."""
        deployment_id = "test_deployment"
        stages = [CanaryStage(percentage=50, duration_minutes=0)]
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=stages,
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock check_health to raise exception
        with (
            patch.object(
                canary_deployment,
                "_check_health",
                side_effect=Exception("Health check error"),
            ),
            patch.object(canary_deployment, "_rollback_canary") as mock_rollback,
        ):
            await canary_deployment._run_canary(deployment_id, auto_rollback=True)

        # Verify rollback was called and status set to failed
        assert deployment_config.status == "failed"
        mock_rollback.assert_called_once_with(deployment_id)

    @pytest.mark.asyncio
    async def test_run_canary_nonexistent_deployment(self, canary_deployment):
        """Test running canary for non-existent deployment."""
        # Should return gracefully without error
        await canary_deployment._run_canary("nonexistent_deployment")

    @pytest.mark.asyncio
    async def test_monitor_stage_success(self, canary_deployment):
        """Test successful stage monitoring."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock collect_metrics to return good metrics
        with (
            patch.object(
                canary_deployment,
                "_collect_metrics",
                return_value={"latency": 100, "error_rate": 0.01},
            ),
            patch("time.time", side_effect=[1000, 1060, 1120]),
            patch("asyncio.sleep"),
        ):
            await canary_deployment._monitor_stage(
                deployment_id=deployment_id,
                duration_seconds=60,
                error_threshold=0.05,
                latency_threshold=200,
            )

        # Verify metrics were collected (should be at least 1 measurement)
        assert len(deployment_config.metrics.latency) >= 1
        assert len(deployment_config.metrics.error_rate) >= 1
        if deployment_config.metrics.latency:
            assert deployment_config.metrics.latency[0] == 100
        if deployment_config.metrics.error_rate:
            assert deployment_config.metrics.error_rate[0] == 0.01

    @pytest.mark.asyncio
    async def test_monitor_stage_error_threshold_exceeded(self, canary_deployment):
        """Test monitoring with error threshold exceeded."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock collect_metrics to return high error rate
        with (
            patch.object(
                canary_deployment,
                "_collect_metrics",
                return_value={"latency": 100, "error_rate": 0.10},
            ),
            patch("time.time", side_effect=[1000, 1010]),
            patch("asyncio.sleep"),
            pytest.raises(ServiceError, match="Error rate threshold exceeded"),
        ):
            await canary_deployment._monitor_stage(
                deployment_id=deployment_id,
                duration_seconds=5,  # Short duration to ensure exception is raised quickly
                error_threshold=0.05,
                latency_threshold=200,
            )

    @pytest.mark.asyncio
    async def test_monitor_stage_latency_threshold_exceeded(self, canary_deployment):
        """Test monitoring with latency threshold exceeded."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock collect_metrics to return high latency
        with (
            patch.object(
                canary_deployment,
                "_collect_metrics",
                return_value={"latency": 300, "error_rate": 0.01},
            ),
            patch("time.time", side_effect=[1000, 1010]),
            patch("asyncio.sleep"),
            pytest.raises(ServiceError, match="Latency threshold exceeded"),
        ):
            await canary_deployment._monitor_stage(
                deployment_id=deployment_id,
                duration_seconds=5,  # Short duration to ensure exception is raised quickly
                error_threshold=0.05,
                latency_threshold=200,
            )

    @pytest.mark.asyncio
    async def test_monitor_stage_nonexistent_deployment(self, canary_deployment):
        """Test monitoring non-existent deployment."""
        # Should return gracefully without error
        await canary_deployment._monitor_stage(
            deployment_id="nonexistent",
            duration_seconds=60,
            error_threshold=0.05,
            latency_threshold=200,
        )

    @pytest.mark.asyncio
    async def test_collect_metrics_with_stage_start_time(self, canary_deployment):
        """Test collecting metrics with stage start time set."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        deployment_config.metrics.stage_start_time = time.time()

        with (
            patch("random.uniform", side_effect=[-10, 0.005]),
            patch("time.time", return_value=1234567890),
        ):
            metrics = await canary_deployment._collect_metrics(deployment_config)

        assert "latency" in metrics
        assert "error_rate" in metrics
        assert "timestamp" in metrics
        assert metrics["latency"] == 90  # 100 + (-10)
        assert metrics["error_rate"] == 0.025  # max(0, 0.02 + 0.005)
        assert metrics["timestamp"] == 1234567890

    @pytest.mark.asyncio
    async def test_collect_metrics_without_stage_start_time(self, canary_deployment):
        """Test collecting metrics without stage start time."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        # stage_start_time is None

        with patch("time.time", return_value=1234567890):
            metrics = await canary_deployment._collect_metrics(deployment_config)

        assert metrics["latency"] == 100.0
        assert metrics["error_rate"] == 0.01
        assert metrics["timestamp"] == 1234567890

    def test_check_health_no_metrics(self, canary_deployment):
        """Test health check with no metrics collected."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )

        result = canary_deployment._check_health(deployment_config)
        assert result is True  # Should pass when no metrics

    def test_check_health_insufficient_data(self, canary_deployment):
        """Test health check with insufficient metrics data."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        deployment_config.metrics.latency = [100.0]  # Only 1 sample
        deployment_config.metrics.error_rate = []  # No error rate data

        result = canary_deployment._check_health(deployment_config)
        assert result is True  # Should pass with insufficient data

    def test_check_health_healthy_metrics(self, canary_deployment):
        """Test health check with healthy metrics."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[
                CanaryStage(
                    percentage=50,
                    duration_minutes=1,
                    error_threshold=0.05,
                    latency_threshold=200,
                )
            ],
        )
        deployment_config.metrics.latency = [90, 95, 100, 85, 105]  # Avg: 95
        deployment_config.metrics.error_rate = [
            0.01,
            0.02,
            0.015,
            0.008,
            0.012,
        ]  # Avg: 0.013

        result = canary_deployment._check_health(deployment_config)
        assert result is True

    def test_check_health_unhealthy_error_rate(self, canary_deployment):
        """Test health check with unhealthy error rate."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[
                CanaryStage(
                    percentage=50,
                    duration_minutes=1,
                    error_threshold=0.05,
                    latency_threshold=200,
                )
            ],
        )
        deployment_config.metrics.latency = [90, 95, 100]  # Healthy latency
        deployment_config.metrics.error_rate = [
            0.06,
            0.07,
            0.08,
        ]  # Avg: 0.07 > 0.05 threshold

        result = canary_deployment._check_health(deployment_config)
        assert result is False

    def test_check_health_unhealthy_latency(self, canary_deployment):
        """Test health check with unhealthy latency."""
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[
                CanaryStage(
                    percentage=50,
                    duration_minutes=1,
                    error_threshold=0.05,
                    latency_threshold=200,
                )
            ],
        )
        deployment_config.metrics.latency = [
            250,
            300,
            280,
        ]  # Avg: 276.67 > 200 threshold
        deployment_config.metrics.error_rate = [0.01, 0.02, 0.015]  # Healthy error rate

        result = canary_deployment._check_health(deployment_config)
        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_canary_success(self, canary_deployment, mock_alias_manager):
        """Test successful canary rollback."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Current collection is different from old collection
        mock_alias_manager.get_collection_for_alias.return_value = "new_collection"

        await canary_deployment._rollback_canary(deployment_id)

        # Verify rollback actions
        mock_alias_manager.switch_alias.assert_called_once_with(
            alias_name="test_alias",
            new_collection="old_collection",
        )
        assert deployment_config.status == "rolled_back"

    @pytest.mark.asyncio
    async def test_rollback_canary_already_rolled_back(
        self, canary_deployment, mock_alias_manager
    ):
        """Test rollback when already pointing to old collection."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Current collection is same as old collection
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        await canary_deployment._rollback_canary(deployment_id)

        # Verify no switch was needed
        mock_alias_manager.switch_alias.assert_not_called()
        assert deployment_config.status == "rolled_back"

    @pytest.mark.asyncio
    async def test_rollback_canary_failure(self, canary_deployment, mock_alias_manager):
        """Test rollback failure."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=1)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        mock_alias_manager.get_collection_for_alias.side_effect = Exception(
            "Rollback failed"
        )

        await canary_deployment._rollback_canary(deployment_id)

        assert deployment_config.status == "rollback_failed"

    @pytest.mark.asyncio
    async def test_rollback_canary_nonexistent_deployment(self, canary_deployment):
        """Test rollback for non-existent deployment."""
        # Should return gracefully without error
        await canary_deployment._rollback_canary("nonexistent")

    @pytest.mark.asyncio
    async def test_get_deployment_status_found(self, canary_deployment):
        """Test getting deployment status for existing deployment."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[
                CanaryStage(percentage=25, duration_minutes=30),
                CanaryStage(percentage=100, duration_minutes=0),
            ],
            current_stage=0,
            start_time=time.time() - 1800,  # Started 30 minutes ago
            status="running",
        )
        deployment_config.metrics.latency = [95.0, 100.5, 92.3]
        deployment_config.metrics.error_rate = [0.01, 0.015, 0.008]

        canary_deployment.deployments[deployment_id] = deployment_config

        status = await canary_deployment.get_deployment_status(deployment_id)

        assert status["deployment_id"] == deployment_id
        assert status["status"] == "running"
        assert status["alias"] == "test_alias"
        assert status["old_collection"] == "old_collection"
        assert status["new_collection"] == "new_collection"
        assert status["current_stage"] == 0
        assert status["total_stages"] == 2
        assert status["current_percentage"] == 25
        assert 29 < status["duration_minutes"] < 31  # Approximately 30 minutes
        assert abs(status["avg_latency"] - 95.93) < 0.1  # (95.0 + 100.5 + 92.3) / 3
        assert status["last_latency"] == 92.3
        assert (
            abs(status["avg_error_rate"] - 0.011) < 0.001
        )  # (0.01 + 0.015 + 0.008) / 3
        assert status["last_error_rate"] == 0.008

    @pytest.mark.asyncio
    async def test_get_deployment_status_not_found(self, canary_deployment):
        """Test getting status for non-existent deployment."""
        status = await canary_deployment.get_deployment_status("nonexistent")
        assert status == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_get_deployment_status_no_metrics(self, canary_deployment):
        """Test getting status with no metrics collected."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        status = await canary_deployment.get_deployment_status(deployment_id)

        assert "avg_latency" not in status
        assert "last_latency" not in status
        assert "avg_error_rate" not in status
        assert "last_error_rate" not in status

    @pytest.mark.asyncio
    async def test_pause_deployment_success(self, canary_deployment):
        """Test successfully pausing deployment."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            status="running",
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        result = await canary_deployment.pause_deployment(deployment_id)

        assert result is True
        assert deployment_config.status == "paused"

    @pytest.mark.asyncio
    async def test_pause_deployment_not_found(self, canary_deployment):
        """Test pausing non-existent deployment."""
        result = await canary_deployment.pause_deployment("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_pause_deployment_not_running(self, canary_deployment):
        """Test pausing deployment that's not running."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            status="completed",
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        result = await canary_deployment.pause_deployment(deployment_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_deployment_success(self, canary_deployment):
        """Test successfully resuming deployment."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            status="paused",
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        with patch("asyncio.create_task") as mock_create_task:
            result = await canary_deployment.resume_deployment(deployment_id)

        assert result is True
        assert deployment_config.status == "running"
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_deployment_not_found(self, canary_deployment):
        """Test resuming non-existent deployment."""
        result = await canary_deployment.resume_deployment("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_deployment_not_paused(self, canary_deployment):
        """Test resuming deployment that's not paused."""
        deployment_id = "test_deployment"
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            status="running",
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        result = await canary_deployment.resume_deployment(deployment_id)
        assert result is False

    def test_get_active_deployments_empty(self, canary_deployment):
        """Test getting active deployments when none exist."""
        active = canary_deployment.get_active_deployments()
        assert active == []

    def test_get_active_deployments_with_data(self, canary_deployment):
        """Test getting active deployments with mixed states."""
        # Add running deployment
        running_config = CanaryDeploymentConfig(
            alias="running_alias",
            old_collection="old1",
            new_collection="new1",
            stages=[
                CanaryStage(percentage=25, duration_minutes=30),
                CanaryStage(percentage=100, duration_minutes=0),
            ],
            current_stage=0,
            status="running",
        )
        canary_deployment.deployments["running_1"] = running_config

        # Add paused deployment
        paused_config = CanaryDeploymentConfig(
            alias="paused_alias",
            old_collection="old2",
            new_collection="new2",
            stages=[CanaryStage(percentage=50, duration_minutes=60)],
            current_stage=0,
            status="paused",
        )
        canary_deployment.deployments["paused_1"] = paused_config

        # Add completed deployment (should not appear)
        completed_config = CanaryDeploymentConfig(
            alias="completed_alias",
            old_collection="old3",
            new_collection="new3",
            stages=[CanaryStage(percentage=100, duration_minutes=0)],
            status="completed",
        )
        canary_deployment.deployments["completed_1"] = completed_config

        active = canary_deployment.get_active_deployments()

        assert len(active) == 2

        # Find running deployment
        running_dep = next(d for d in active if d["id"] == "running_1")
        assert running_dep["alias"] == "running_alias"
        assert running_dep["status"] == "running"
        assert running_dep["current_stage"] == 0
        assert running_dep["current_percentage"] == 25

        # Find paused deployment
        paused_dep = next(d for d in active if d["id"] == "paused_1")
        assert paused_dep["alias"] == "paused_alias"
        assert paused_dep["status"] == "paused"
        assert paused_dep["current_percentage"] == 50

    def test_service_inheritance(self, canary_deployment):
        """Test that CanaryDeployment properly inherits from BaseService."""
        from src.services.base import BaseService

        assert isinstance(canary_deployment, BaseService)
        assert hasattr(canary_deployment, "_initialized")

    @pytest.mark.asyncio
    async def test_canary_task_creation_avoiding_ruf006(
        self, canary_deployment, mock_alias_manager
    ):
        """Test that canary task creation properly avoids RUF006 warning."""
        mock_alias_manager.get_collection_for_alias.return_value = "existing_collection"

        with (
            patch("asyncio.create_task") as mock_create_task,
            patch("time.time", return_value=1234567890),
        ):
            await canary_deployment.start_canary(
                alias_name="test_alias",
                new_collection="new_collection",
            )

        # Verify that asyncio.create_task was called
        mock_create_task.assert_called()
        call_args = mock_create_task.call_args[0][0]

        # Verify it's a coroutine for _run_canary
        assert hasattr(call_args, "__await__")  # It's a coroutine

    @pytest.mark.asyncio
    async def test_stage_with_zero_duration(self, canary_deployment):
        """Test handling stages with zero duration."""
        deployment_id = "test_deployment"
        stages = [CanaryStage(percentage=100, duration_minutes=0)]  # Zero duration
        deployment_config = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=stages,
        )
        canary_deployment.deployments[deployment_id] = deployment_config

        # Mock check_health to always return True
        with (
            patch.object(canary_deployment, "_check_health", return_value=True),
            patch.object(canary_deployment, "_monitor_stage") as mock_monitor,
        ):
            await canary_deployment._run_canary(deployment_id, auto_rollback=True)

        # Verify monitoring was not called for zero duration stage
        mock_monitor.assert_not_called()
        assert deployment_config.status == "completed"
