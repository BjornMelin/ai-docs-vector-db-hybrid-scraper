"""Tests for TaskQueueManager - simplified without backwards compatibility."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import Config
from src.services.task_queue.manager import TaskQueueManager


@pytest.fixture
def config():
    """Create test configuration."""
    config = Mock(spec=Config)
    config.task_queue = Mock()
    config.task_queue.redis_url = "redis://localhost:6379"
    config.task_queue.redis_password = None
    config.task_queue.redis_database = 1
    config.task_queue.queue_name = "test_queue"
    return config


@pytest.fixture
def manager(config):
    """Create TaskQueueManager instance."""
    return TaskQueueManager(config)


@pytest.fixture
def test_redis_password():
    """Test Redis password value."""
    return "override_pass"


class TestTaskQueueManager:
    """Test TaskQueueManager functionality."""

    def test_init(self, config):
        """Test manager initialization."""
        manager = TaskQueueManager(config)
        assert manager.config == config
        assert manager._redis_pool is None
        assert not manager._initialized

    def test_create_redis_settings_basic(self, manager):
        """Test Redis settings creation with basic URL."""
        settings = manager._create_redis_settings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.password is None
        assert settings.database == 1

    def test_create_redis_settings_with_auth(self, manager, test_redis_password):
        """Test Redis settings with authentication."""
        manager.config.task_queue.redis_url = "redis://user:pass@localhost:6380"
        manager.config.task_queue.redis_password = test_redis_password

        settings = manager._create_redis_settings()
        assert settings.host == "localhost"
        assert settings.port == 6380
        assert settings.password == test_redis_password

    def test_create_redis_settings_custom_port(self, manager):
        """Test Redis settings with custom port."""
        manager.config.task_queue.redis_url = "redis://localhost:9999"

        settings = manager._create_redis_settings()
        assert settings.host == "localhost"
        assert settings.port == 9999

    @pytest.mark.asyncio
    async def test_initialize_success(self, manager):
        """Test successful initialization."""
        mock_pool = AsyncMock()

        with patch(
            "src.services.task_queue.manager.create_pool", return_value=mock_pool
        ):
            await manager.initialize()

        assert manager._redis_pool == mock_pool
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self, manager):
        """Test initialization failure."""
        with (
            patch(
                "src.services.task_queue.manager.create_pool",
                side_effect=Exception("Connection failed"),
            ),
            pytest.raises(Exception, match="Connection failed"),
        ):
            await manager.initialize()

        assert manager._redis_pool is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """Test cleanup method."""
        mock_pool = AsyncMock()
        manager._redis_pool = mock_pool
        manager._initialized = True

        await manager.cleanup()

        mock_pool.close.assert_called_once()
        assert manager._redis_pool is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_enqueue_success(self, manager):
        """Test successful job enqueueing."""
        mock_pool = AsyncMock()
        mock_job = Mock()
        mock_job.job_id = "job_123"
        mock_pool.enqueue_job.return_value = mock_job
        manager._redis_pool = mock_pool

        job_id = await manager.enqueue(
            "test_task", arg1="value1", _delay=60, kwarg1="value2"
        )

        assert job_id == "job_123"
        mock_pool.enqueue_job.assert_called_once_with(
            "test_task",
            _queue_name="test_queue",
            _defer_by=60,
            arg1="value1",
            kwarg1="value2",
        )

    @pytest.mark.asyncio
    async def test_enqueue_not_initialized(self, manager):
        """Test enqueueing when not initialized."""
        job_id = await manager.enqueue("test_task")
        assert job_id is None

    @pytest.mark.asyncio
    async def test_enqueue_failure(self, manager):
        """Test enqueueing failure."""
        mock_pool = AsyncMock()
        mock_pool.enqueue_job.return_value = None
        manager._redis_pool = mock_pool

        job_id = await manager.enqueue("test_task")
        assert job_id is None

    @pytest.mark.asyncio
    async def test_get_job_status_success(self, manager):
        """Test successful job status retrieval."""
        mock_pool = AsyncMock()
        mock_job = Mock()
        mock_job.status = "complete"
        mock_job.function = "test_task"
        mock_job.args = ["arg1"]
        mock_job._kwargs = {"key": "value"}
        mock_job.enqueue_time = None
        mock_job.start_time = None
        mock_job.finish_time = None
        mock_job.result = "success"
        mock_job.error = None
        mock_pool.job.return_value = mock_job
        manager._redis_pool = mock_pool

        status = await manager.get_job_status("job_123")

        assert status["status"] == "complete"
        assert status["job_id"] == "job_123"
        assert status["result"] == "success"

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, manager):
        """Test job status when job not found."""
        mock_pool = AsyncMock()
        mock_pool.job.return_value = None
        manager._redis_pool = mock_pool

        status = await manager.get_job_status("nonexistent")

        assert status["status"] == "not_found"
        assert status["job_id"] == "nonexistent"

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, manager):
        """Test successful job cancellation."""
        mock_pool = AsyncMock()
        mock_job = AsyncMock()
        mock_job.status = "deferred"
        mock_pool.job.return_value = mock_job
        manager._redis_pool = mock_pool

        result = await manager.cancel_job("job_123")

        assert result is True
        mock_job.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_job_not_deferred(self, manager):
        """Test cancelling non-deferred job."""
        mock_pool = AsyncMock()
        mock_job = Mock()
        mock_job.status = "running"
        mock_pool.job.return_value = mock_job
        manager._redis_pool = mock_pool

        result = await manager.cancel_job("job_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, manager):
        """Test queue statistics retrieval."""
        mock_pool = AsyncMock()
        manager._redis_pool = mock_pool

        stats = await manager.get_queue_stats()

        # Current implementation returns basic stats
        assert "pending" in stats
        assert "running" in stats
        assert "complete" in stats
        assert "failed" in stats

    def test_get_redis_settings(self, manager):
        """Test Redis settings getter."""
        settings = manager.get_redis_settings()
        assert settings.host == "localhost"
        assert settings.port == 6379
