"""Tests for QdrantAliasManager with task queue integration."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from src.config import UnifiedConfig
from src.services.core.qdrant_alias_manager import QdrantAliasManager


class TestQdrantAliasManagerWithTaskQueue:
    """Test QdrantAliasManager with task queue integration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Mock(spec=UnifiedConfig)

    @pytest.fixture
    def client(self):
        """Create mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    def task_queue_manager(self):
        """Create mock task queue manager."""
        return AsyncMock()

    @pytest.fixture
    def alias_manager(self, config, client, task_queue_manager):
        """Create QdrantAliasManager with task queue."""
        return QdrantAliasManager(config, client, task_queue_manager)

    @pytest.mark.asyncio
    async def test_safe_delete_collection_with_queue(self, alias_manager, task_queue_manager):
        """Test safe_delete_collection uses task queue."""
        # Setup
        alias_manager.list_aliases = AsyncMock(return_value={"alias1": "other_collection"})
        task_queue_manager.enqueue = AsyncMock(return_value="job_123")

        # Execute
        await alias_manager.safe_delete_collection("test_collection", grace_period_minutes=30)

        # Verify
        task_queue_manager.enqueue.assert_called_once_with(
            "delete_collection",
            collection_name="test_collection",
            grace_period_minutes=30,
            _delay=1800,  # 30 * 60
        )

    @pytest.mark.asyncio
    async def test_safe_delete_collection_no_queue(self, config, client):
        """Test safe_delete_collection without task queue."""
        # Create manager without task queue
        alias_manager = QdrantAliasManager(config, client, task_queue_manager=None)
        alias_manager.list_aliases = AsyncMock(return_value={"alias1": "other_collection"})

        # Execute - should not raise exception
        await alias_manager.safe_delete_collection("test_collection")

        # No assertions needed - just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_safe_delete_collection_with_aliases(self, alias_manager, task_queue_manager):
        """Test safe_delete_collection skips when collection has aliases."""
        # Setup - collection has an alias
        alias_manager.list_aliases = AsyncMock(return_value={"alias1": "test_collection"})

        # Execute
        await alias_manager.safe_delete_collection("test_collection")

        # Verify - should not enqueue
        task_queue_manager.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_safe_delete_collection_queue_failure(self, alias_manager, task_queue_manager):
        """Test safe_delete_collection handles queue failure."""
        # Setup
        alias_manager.list_aliases = AsyncMock(return_value={"alias1": "other_collection"})
        task_queue_manager.enqueue = AsyncMock(return_value=None)  # Failure

        # Execute - should not raise exception
        await alias_manager.safe_delete_collection("test_collection")

        # Verify enqueue was attempted
        task_queue_manager.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_alias_with_delete(self, alias_manager, task_queue_manager):
        """Test switch_alias with delete_old option uses task queue."""
        # Setup
        alias_manager.get_collection_for_alias = AsyncMock(return_value="old_collection")
        alias_manager.list_aliases = AsyncMock(return_value={})  # No aliases for safe_delete
        task_queue_manager.enqueue = AsyncMock(return_value="job_456")

        # Execute
        result = await alias_manager.switch_alias(
            alias_name="test_alias",
            new_collection="new_collection",
            delete_old=True
        )

        # Verify
        assert result == "old_collection"

        # Verify safe_delete_collection was called (which uses task queue)
        task_queue_manager.enqueue.assert_called_once_with(
            "delete_collection",
            collection_name="old_collection",
            grace_period_minutes=60,  # Default
            _delay=3600,  # 60 * 60
        )
