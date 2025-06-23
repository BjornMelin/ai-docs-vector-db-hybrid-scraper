"""Tests for task queue task functions."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.task_queue.tasks import TASK_MAP, delete_collection, persist_cache


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

        # Mock the async sleep to avoid waiting
        with (
            patch("src.services.task_queue.tasks.asyncio.sleep"),
            patch("src.services.task_queue.tasks.get_config") as mock_get_config,
            patch(
                "src.services.task_queue.tasks.ClientManager"
            ) as mock_client_manager_class,
            patch(
                "src.services.task_queue.tasks.QdrantAliasManager"
            ) as mock_alias_manager_class,
            patch("src.services.task_queue.tasks.logger") as mock_logger,
        ):
            # Mock config
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            # Mock client manager
            mock_client_manager = AsyncMock()
            mock_qdrant_client = AsyncMock()
            mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
            mock_client_manager_class.return_value = mock_client_manager

            # Mock alias manager
            mock_alias_manager = AsyncMock()
            mock_alias_manager.list_aliases.return_value = {}  # No aliases
            mock_alias_manager_class.return_value = mock_alias_manager

            # Execute the task
            result = await delete_collection(ctx, collection_name, grace_period_minutes)

            # Verify the result
            assert result["status"] == "success"
            assert result["collection"] == collection_name
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_delete_collection_error(self):
        """Test collection deletion error handling."""
        ctx = {"redis": AsyncMock()}
        collection_name = "test_collection"
        grace_period_minutes = 60

        # Mock to raise an exception
        with (
            patch("src.services.task_queue.tasks.asyncio.sleep"),
            patch(
                "src.services.task_queue.tasks.get_config",
                side_effect=Exception("Database error"),
            ),
        ):
            result = await delete_collection(ctx, collection_name, grace_period_minutes)

            # Should return error result
            assert result["status"] == "failed"
            assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_persist_cache_success(self):
        """Test successful cache persistence."""
        ctx = {"redis": AsyncMock()}
        cache_key = "test_cache_key"
        cache_data = {"data": "test_data"}
        persist_func_module = "src.services.cache.patterns"
        persist_func_name = "persist_to_storage"

        # Mock the persist function module and function
        mock_persist_func = AsyncMock()

        with (
            patch("src.services.task_queue.tasks.asyncio.sleep"),  # Mock the delay
            patch("importlib.import_module") as mock_import,
        ):
            # Mock the module and function
            mock_module = Mock()
            mock_module.persist_to_storage = mock_persist_func
            mock_import.return_value = mock_module

            # Execute the task - need to pass delay parameter too
            result = await persist_cache(
                ctx,
                cache_key,
                cache_data,
                persist_func_module,
                persist_func_name,
                delay=0.1,
            )

            # Verify the result
            assert result["status"] == "success"
            assert result["key"] == cache_key
            assert "duration" in result
            mock_persist_func.assert_called_once_with(cache_key, cache_data)

    @pytest.mark.asyncio
    async def test_persist_cache_error(self):
        """Test cache persistence error handling."""
        ctx = {"redis": AsyncMock()}
        cache_key = "test_cache_key"
        cache_data = {"data": "test_data"}
        persist_func_module = "src.services.cache.patterns"
        persist_func_name = "persist_to_storage"

        # Mock import to raise an exception
        with patch(
            "importlib.import_module",
            side_effect=Exception("Cache error"),
        ):
            result = await persist_cache(
                ctx, cache_key, cache_data, persist_func_module, persist_func_name
            )

            # Should return error result
            assert result["status"] == "failed"
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
