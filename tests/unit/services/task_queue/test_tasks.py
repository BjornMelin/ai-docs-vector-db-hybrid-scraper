"""Tests for task queue task functions."""

import time
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.services.task_queue.tasks import TASK_REGISTRY
from src.services.task_queue.tasks import delete_collection
from src.services.task_queue.tasks import persist_cache
from src.services.task_queue.tasks import run_canary_deployment


class TestTaskFunctions:
    """Test task queue task functions."""

    @pytest.mark.asyncio
    async def test_delete_collection_success(self):
        """Test successful collection deletion."""
        ctx = {"job_id": "test_job"}
        collection_name = "test_collection"
        grace_period_minutes = 1  # Use short period for testing

        with (
            patch("src.services.task_queue.tasks.get_config") as mock_get_config,
            patch(
                "src.services.task_queue.tasks.ClientManager"
            ) as mock_client_manager_class,
            patch(
                "src.services.task_queue.tasks.QdrantAliasManager"
            ) as mock_alias_manager_class,
            patch("asyncio.sleep") as mock_sleep,
        ):  # Mock sleep to speed up test
            # Mock config
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            # Mock client manager
            mock_client_manager = AsyncMock()
            mock_client_manager_class.return_value = mock_client_manager

            # Mock qdrant client
            mock_qdrant_client = AsyncMock()
            mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client

            # Mock alias manager
            mock_alias_manager = AsyncMock()
            mock_alias_manager_class.return_value = mock_alias_manager
            mock_alias_manager.list_aliases.return_value = {}  # No aliases

            result = await delete_collection(ctx, collection_name, grace_period_minutes)

            # Verify sleep was called with correct duration
            mock_sleep.assert_called_once_with(grace_period_minutes * 60)

            # Verify collection was deleted
            mock_qdrant_client.delete_collection.assert_called_once_with(
                collection_name
            )

            # Verify result
            assert result["status"] == "success"
            assert result["collection"] == collection_name
            assert "duration" in result

    @pytest.mark.asyncio
    async def test_delete_collection_has_aliases(self):
        """Test collection deletion when aliases exist."""
        ctx = {"job_id": "test_job"}
        collection_name = "test_collection"

        with (
            patch("src.services.task_queue.tasks.get_config") as mock_get_config,
            patch(
                "src.services.task_queue.tasks.ClientManager"
            ) as mock_client_manager_class,
            patch(
                "src.services.task_queue.tasks.QdrantAliasManager"
            ) as mock_alias_manager_class,
            patch("asyncio.sleep"),
        ):
            # Mock setup
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            mock_client_manager = AsyncMock()
            mock_client_manager_class.return_value = mock_client_manager
            mock_qdrant_client = AsyncMock()
            mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
            mock_alias_manager = AsyncMock()
            mock_alias_manager_class.return_value = mock_alias_manager

            # Collection has aliases
            mock_alias_manager.list_aliases.return_value = {"alias1": collection_name}

            result = await delete_collection(ctx, collection_name, 1)

            # Verify collection was NOT deleted
            mock_qdrant_client.delete_collection.assert_not_called()

            # Verify result
            assert result["status"] == "skipped"
            assert result["reason"] == "collection_has_aliases"

    @pytest.mark.asyncio
    async def test_delete_collection_error(self):
        """Test collection deletion with error."""
        ctx = {"job_id": "test_job"}
        collection_name = "test_collection"

        with (
            patch("src.services.task_queue.tasks.get_config") as mock_get_config,
            patch(
                "src.services.task_queue.tasks.ClientManager"
            ) as mock_client_manager_class,
            patch("asyncio.sleep"),
        ):
            # Mock config
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            # Client manager raises exception
            mock_client_manager_class.side_effect = Exception("Connection failed")

            result = await delete_collection(ctx, collection_name, 1)

            # Verify error result
            assert result["status"] == "failed"
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_persist_cache_async_function(self):
        """Test cache persistence with async function."""
        ctx = {"job_id": "test_job"}
        key = "test_key"
        value = "test_value"
        persist_func_module = "test_module"
        persist_func_name = "async_persist"
        delay = 0.1  # Short delay for testing

        # Mock async persist function
        async def mock_async_persist(k, v):
            assert k == key
            assert v == value

        with (
            patch("asyncio.sleep") as mock_sleep,
            patch("importlib.import_module") as mock_import,
            patch("asyncio.iscoroutinefunction", return_value=True),
        ):
            # Mock module and function
            mock_module = Mock()
            mock_import.return_value = mock_module
            mock_module.async_persist = mock_async_persist

            result = await persist_cache(
                ctx, key, value, persist_func_module, persist_func_name, delay
            )

            # Verify sleep was called
            mock_sleep.assert_called_once_with(delay)

            # Verify result
            assert result["status"] == "success"
            assert result["key"] == key

    @pytest.mark.asyncio
    async def test_persist_cache_sync_function(self):
        """Test cache persistence with sync function."""
        ctx = {"job_id": "test_job"}
        key = "test_key"
        value = "test_value"
        persist_func_module = "test_module"
        persist_func_name = "sync_persist"

        # Mock sync persist function
        def mock_sync_persist(k, v):
            assert k == key
            assert v == value

        with (
            patch("asyncio.sleep"),
            patch("importlib.import_module") as mock_import,
            patch("asyncio.iscoroutinefunction", return_value=False),
        ):
            # Mock module and function
            mock_module = Mock()
            mock_import.return_value = mock_module
            mock_module.sync_persist = mock_sync_persist

            result = await persist_cache(
                ctx, key, value, persist_func_module, persist_func_name
            )

            # Verify result
            assert result["status"] == "success"
            assert result["key"] == key

    @pytest.mark.asyncio
    async def test_persist_cache_error(self):
        """Test cache persistence with error."""
        ctx = {"job_id": "test_job"}

        with (
            patch("asyncio.sleep"),
            patch(
                "importlib.import_module", side_effect=ImportError("Module not found")
            ),
        ):
            result = await persist_cache(ctx, "key", "value", "bad_module", "func")

            # Verify error result
            assert result["status"] == "failed"
            assert "Module not found" in result["error"]

    @pytest.mark.asyncio
    async def test_run_canary_deployment_success(self):
        """Test successful canary deployment."""
        ctx = {"job_id": "test_job"}
        deployment_id = "canary_123"
        deployment_config = {
            "alias": "test_alias",
            "old_collection": "old_coll",
            "new_collection": "new_coll",
            "stages": [{"percentage": 10, "duration_minutes": 1}],
            "current_stage": 0,
            "metrics": {
                "latency": [],
                "error_rate": [],
                "success_count": 0,
                "error_count": 0,
                "stage_start_time": None,
            },
            "start_time": time.time(),
            "status": "pending",
        }

        with (
            patch("src.services.task_queue.tasks.get_config") as mock_get_config,
            patch(
                "src.services.task_queue.tasks.ClientManager"
            ) as mock_client_manager_class,
            patch(
                "src.services.task_queue.tasks.QdrantAliasManager"
            ) as mock_alias_manager_class,
            patch(
                "src.services.deployment.canary.CanaryDeployment"
            ) as mock_canary_class,
            patch(
                "src.services.deployment.canary.CanaryDeploymentConfig"
            ) as mock_config_class,
            patch("src.services.deployment.canary.CanaryMetrics") as mock_metrics_class,
            patch("src.services.deployment.canary.CanaryStage") as mock_stage_class,
            patch(
                "src.services.vector_db.service.QdrantService"
            ) as mock_qdrant_service_class,
        ):
            # Mock config
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            # Mock client manager
            mock_client_manager = AsyncMock()
            mock_client_manager_class.return_value = mock_client_manager
            mock_qdrant_client = AsyncMock()
            mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client

            # Mock services
            mock_alias_manager = AsyncMock()
            mock_alias_manager_class.return_value = mock_alias_manager

            mock_qdrant_service = AsyncMock()
            mock_qdrant_service_class.return_value = mock_qdrant_service

            # Mock canary instance
            mock_canary = AsyncMock()
            mock_canary_class.return_value = mock_canary
            mock_canary.get_deployment_status.return_value = {"status": "completed"}
            mock_canary.deployments = {}

            # Mock config classes
            mock_stage_class.return_value = Mock()
            mock_metrics_class.return_value = Mock()
            mock_config_class.return_value = Mock()

            result = await run_canary_deployment(ctx, deployment_id, deployment_config)

            # Verify result
            assert result["status"] == "success"
            assert result["deployment_id"] == deployment_id

    @pytest.mark.asyncio
    async def test_run_canary_deployment_error(self):
        """Test canary deployment with error."""
        ctx = {"job_id": "test_job"}
        deployment_id = "canary_123"
        deployment_config = {}

        with patch(
            "src.services.task_queue.tasks.get_config",
            side_effect=Exception("Config error"),
        ):
            result = await run_canary_deployment(ctx, deployment_id, deployment_config)

            # Verify error result
            assert result["status"] == "failed"
            assert "Config error" in result["error"]


class TestTaskRegistry:
    """Test task registry."""

    def test_task_registry_contains_all_tasks(self):
        """Test that task registry contains all expected tasks."""
        expected_tasks = [
            "delete_collection",
            "persist_cache",
            "run_canary_deployment",
        ]

        for task_name in expected_tasks:
            assert task_name in TASK_REGISTRY

        # Verify all tasks are ARQ Function objects with coroutines
        for task_func in TASK_REGISTRY.values():
            # ARQ Function objects have a 'coroutine' attribute
            assert hasattr(task_func, "coroutine")
            assert callable(task_func.coroutine)
