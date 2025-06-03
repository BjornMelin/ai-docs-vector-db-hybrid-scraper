"""Tests for CanaryDeployment with task queue integration."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.deployment.canary import CanaryDeployment
from src.services.deployment.canary import CanaryDeploymentConfig
from src.services.deployment.canary import CanaryStage
from src.services.errors import ServiceError


class TestCanaryDeploymentWithTaskQueue:
    """Test CanaryDeployment with task queue integration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=UnifiedConfig)
        config.data_dir = Mock()
        config.data_dir.__truediv__ = Mock(return_value="test_path")
        return config

    @pytest.fixture
    def alias_manager(self):
        """Create mock alias manager."""
        return AsyncMock()

    @pytest.fixture
    def qdrant_service(self):
        """Create mock Qdrant service."""
        return AsyncMock()

    @pytest.fixture
    def task_queue_manager(self):
        """Create mock task queue manager."""
        return AsyncMock()

    @pytest.fixture
    def canary(self, config, alias_manager, qdrant_service, task_queue_manager):
        """Create CanaryDeployment with task queue."""
        canary = CanaryDeployment(
            config=config,
            alias_manager=alias_manager,
            qdrant_service=qdrant_service,
            client_manager=None,
            task_queue_manager=task_queue_manager,
        )
        # Mock save/load methods
        canary._save_deployments = AsyncMock()
        canary._load_deployments = AsyncMock()
        return canary

    @pytest.mark.asyncio
    async def test_start_canary_with_queue(self, canary, alias_manager, task_queue_manager):
        """Test start_canary uses task queue."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(return_value="old_collection")
        task_queue_manager.enqueue = AsyncMock(return_value="job_canary_123")

        # Execute
        deployment_id = await canary.start_canary(
            alias_name="test_alias",
            new_collection="new_collection",
            stages=[
                {"percentage": 10, "duration_minutes": 5},
                {"percentage": 50, "duration_minutes": 10},
                {"percentage": 100, "duration_minutes": 0},
            ],
            auto_rollback=True
        )

        # Verify
        assert deployment_id.startswith("canary_")

        # Check deployment was saved
        assert deployment_id in canary.deployments
        deployment = canary.deployments[deployment_id]
        assert deployment.alias == "test_alias"
        assert deployment.old_collection == "old_collection"
        assert deployment.new_collection == "new_collection"
        assert deployment.status == "queued"
        assert len(deployment.stages) == 3

        # Verify task was queued
        task_queue_manager.enqueue.assert_called_once()
        call_args = task_queue_manager.enqueue.call_args
        assert call_args[0][0] == "run_canary_deployment"
        assert call_args[1]["deployment_id"] == deployment_id
        assert call_args[1]["auto_rollback"] is True

        # Verify deployment config was serialized
        deployment_config = call_args[1]["deployment_config"]
        assert isinstance(deployment_config, dict)
        assert deployment_config["alias"] == "test_alias"

    @pytest.mark.asyncio
    async def test_start_canary_queue_failure(self, canary, alias_manager, task_queue_manager):
        """Test start_canary handles queue failure."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(return_value="old_collection")
        task_queue_manager.enqueue = AsyncMock(return_value=None)  # Queue failure

        # Execute and expect error
        with pytest.raises(ServiceError, match="Failed to start canary deployment"):
            await canary.start_canary(
                alias_name="test_alias",
                new_collection="new_collection",
            )

    @pytest.mark.asyncio
    async def test_start_canary_no_queue_fallback(self, config, alias_manager, qdrant_service):
        """Test start_canary without task queue falls back to direct execution."""
        # Create canary without task queue
        canary = CanaryDeployment(
            config=config,
            alias_manager=alias_manager,
            qdrant_service=qdrant_service,
            task_queue_manager=None,  # No task queue
        )
        canary._save_deployments = AsyncMock()
        canary._load_deployments = AsyncMock()
        canary._run_canary = AsyncMock()

        alias_manager.get_collection_for_alias = AsyncMock(return_value="old_collection")

        # Execute
        with patch("asyncio.create_task") as mock_create_task:
            deployment_id = await canary.start_canary(
                alias_name="test_alias",
                new_collection="new_collection",
            )

        # Verify fallback to direct execution
        mock_create_task.assert_called_once()
        assert deployment_id.startswith("canary_")

    @pytest.mark.asyncio
    async def test_resume_deployment_with_queue(self, canary, task_queue_manager):
        """Test resume_deployment uses task queue."""
        # Setup - create a paused deployment
        deployment_id = "canary_test_123"
        deployment = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=10)],
            status="paused",
        )
        canary.deployments[deployment_id] = deployment
        task_queue_manager.enqueue = AsyncMock(return_value="job_resume_456")

        # Execute
        result = await canary.resume_deployment(deployment_id)

        # Verify
        assert result is True
        assert deployment.status == "running"

        # Verify task was queued
        task_queue_manager.enqueue.assert_called_once()
        call_args = task_queue_manager.enqueue.call_args
        assert call_args[0][0] == "run_canary_deployment"
        assert call_args[1]["deployment_id"] == deployment_id
        assert call_args[1]["auto_rollback"] is True

    @pytest.mark.asyncio
    async def test_resume_deployment_queue_failure(self, canary, task_queue_manager):
        """Test resume_deployment handles queue failure."""
        # Setup
        deployment_id = "canary_test_789"
        deployment = CanaryDeploymentConfig(
            alias="test_alias",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=10)],
            status="paused",
        )
        canary.deployments[deployment_id] = deployment
        task_queue_manager.enqueue = AsyncMock(return_value=None)  # Queue failure

        # Execute
        result = await canary.resume_deployment(deployment_id)

        # Verify
        assert result is False
        assert deployment.status == "paused"  # Status reverted

    @pytest.mark.asyncio
    async def test_get_active_deployments(self, canary):
        """Test get_active_deployments returns running and paused deployments."""
        # Setup deployments
        canary.deployments = {
            "canary_1": CanaryDeploymentConfig(
                alias="alias1",
                old_collection="old1",
                new_collection="new1",
                stages=[CanaryStage(percentage=50, duration_minutes=10)],
                status="running",
                current_stage=0,
            ),
            "canary_2": CanaryDeploymentConfig(
                alias="alias2",
                old_collection="old2",
                new_collection="new2",
                stages=[CanaryStage(percentage=100, duration_minutes=0)],
                status="completed",
                current_stage=0,
            ),
            "canary_3": CanaryDeploymentConfig(
                alias="alias3",
                old_collection="old3",
                new_collection="new3",
                stages=[CanaryStage(percentage=25, duration_minutes=5)],
                status="paused",
                current_stage=0,
            ),
        }

        # Execute
        active = canary.get_active_deployments()

        # Verify
        assert len(active) == 2
        assert any(d["id"] == "canary_1" for d in active)
        assert any(d["id"] == "canary_3" for d in active)
        assert not any(d["id"] == "canary_2" for d in active)
