"""Tests for task queue task functions."""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.services.task_queue.tasks import TASK_MAP
from src.services.task_queue.tasks import delete_collection
from src.services.task_queue.tasks import persist_cache

# NOTE: Many task functions were removed during V1 simplification
# as they were over-engineered for a system with 0 users.
# This test file is kept minimal to test remaining functionality.


class TestTaskFunctions:
    """Test task queue task functions."""

    @pytest.mark.asyncio
    async def test_delete_collection_success(self):
        """Test successful collection deletion."""
        # Mock dependencies
        ctx = {"redis": AsyncMock()}
        collection_name = "test_collection"
        grace_period_minutes = 60

        # Mock the client manager and its dependencies
        mock_qdrant_service = AsyncMock()
        mock_qdrant_service.delete_collection.return_value = True

        mock_client_manager = AsyncMock()
        mock_client_manager.get_qdrant_service.return_value = mock_qdrant_service

        # Mock the QdrantAliasManager
        mock_alias_manager = AsyncMock()
        mock_alias_manager.check_collection_deletion_safety.return_value = (True, [])
        mock_alias_manager.cleanup_after_collection_deletion.return_value = None

        with (
            patch(
                "src.services.task_queue.tasks.ClientManager.from_unified_config",
                return_value=mock_client_manager,
            ),
            patch(
                "src.services.task_queue.tasks.QdrantAliasManager",
                return_value=mock_alias_manager,
            ),
            patch("src.services.task_queue.tasks.logger") as mock_logger,
        ):
            # Execute the task
            result = await delete_collection(ctx, collection_name, grace_period_minutes)

            # Verify the result
            assert result["status"] == "success"
            assert result["collection_name"] == collection_name
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_delete_collection_error(self):
        """Test collection deletion error handling."""
        ctx = {"redis": AsyncMock()}
        collection_name = "test_collection"
        grace_period_minutes = 60

        # Mock client manager to raise an exception
        with patch(
            "src.services.task_queue.tasks.ClientManager.from_unified_config",
            side_effect=Exception("Database error"),
        ):
            result = await delete_collection(ctx, collection_name, grace_period_minutes)

            # Should return error result
            assert result["status"] == "error"
            assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_persist_cache_success(self):
        """Test successful cache persistence."""
        ctx = {"redis": AsyncMock()}
        cache_key = "test_cache_key"
        cache_data = {"data": "test_data"}

        # Mock the client manager and cache manager
        mock_cache_manager = AsyncMock()
        mock_cache_manager.persist_to_distributed.return_value = True

        mock_client_manager = AsyncMock()
        mock_client_manager.get_cache_manager.return_value = mock_cache_manager

        with (
            patch(
                "src.services.task_queue.tasks.ClientManager.from_unified_config",
                return_value=mock_client_manager,
            ),
            patch("src.services.task_queue.tasks.logger") as mock_logger,
        ):
            # Execute the task
            result = await persist_cache(ctx, cache_key, cache_data)

            # Verify the result
            assert result["status"] == "success"
            assert result["cache_key"] == cache_key
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_persist_cache_error(self):
        """Test cache persistence error handling."""
        ctx = {"redis": AsyncMock()}
        cache_key = "test_cache_key"
        cache_data = {"data": "test_data"}

        # Mock client manager to raise an exception
        with patch(
            "src.services.task_queue.tasks.ClientManager.from_unified_config",
            side_effect=Exception("Cache error"),
        ):
            result = await persist_cache(ctx, cache_key, cache_data)

            # Should return error result
            assert result["status"] == "error"
            assert "Cache error" in result["error"]


class TestTaskRegistry:
    """Test task registry."""

    def test_task_registry_contains_all_tasks(self):
        """Test that task registry contains all expected tasks."""
        expected_tasks = [
            "delete_collection",
            "persist_cache",
        ]

        for task_name in expected_tasks:
            assert task_name in TASK_MAP

        # Verify all tasks are ARQ Function objects with coroutines
        for task_func in TASK_MAP.values():
            # ARQ Function objects have a 'coroutine' attribute
            assert hasattr(task_func, "coroutine")
            assert callable(task_func.coroutine)
