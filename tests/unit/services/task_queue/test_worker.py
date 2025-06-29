"""Tests for ARQ worker configuration."""

from unittest.mock import Mock, patch

from src.services.task_queue.worker import (
    WorkerSettings,
    cron_jobs,
    functions,
    job_timeout,
    max_jobs,
    on_job_end,
    on_job_start,
    on_shutdown,
    on_startup,
    redis_settings,
)


class TestWorkerSettings:
    """Test ARQ worker settings."""

    def test_worker_settings_attributes(self):
        """Test worker settings class attributes."""
        # Test task registry is properly loaded
        assert hasattr(WorkerSettings, "functions")
        assert isinstance(WorkerSettings.functions, list)
        assert len(WorkerSettings.functions) > 0

        # Test cron jobs
        assert hasattr(WorkerSettings, "cron_jobs")
        assert isinstance(WorkerSettings.cron_jobs, list)

        # Test worker configuration
        assert WorkerSettings.max_jobs == 10
        assert WorkerSettings.job_timeout == 3600
        assert WorkerSettings.max_tries == 3

        # Test health check settings
        assert WorkerSettings.health_check_interval == 60
        assert WorkerSettings.health_check_key == "arq:health-check"

        # Test queue settings
        assert WorkerSettings.queue_name == "default"

    def test_get_redis_settings_basic(self):
        """Test Redis settings generation with basic configuration."""
        mock_config = Mock()
        mock_config.task_queue = Mock()
        mock_config.task_queue.redis_url = "redis://localhost:6379"
        mock_config.task_queue.redis_password = None
        mock_config.task_queue.redis_database = 1

        with patch(
            "src.services.task_queue.worker.get_config", return_value=mock_config
        ):
            settings = WorkerSettings.get_redis_settings()

        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.password is None
        assert settings.database == 1

    def test_get_redis_settings_with_auth(self):
        """Test Redis settings with authentication."""
        mock_config = Mock()
        mock_config.task_queue = Mock()
        mock_config.task_queue.redis_url = "redis://user:testpass@localhost:6380"
        mock_config.task_queue.redis_password = "test_override_pass"
        mock_config.task_queue.redis_database = 2

        with patch(
            "src.services.task_queue.worker.get_config", return_value=mock_config
        ):
            settings = WorkerSettings.get_redis_settings()

        assert settings.host == "localhost"
        assert settings.port == 6380
        assert settings.password == "test_override_pass"
        assert settings.database == 2

    def test_get_redis_settings_url_parsing(self):
        """Test different Redis URL formats."""
        test_cases = [
            # (url, expected_host, expected_port)
            ("redis://localhost", "localhost", 6379),
            ("redis://localhost:9999", "localhost", 9999),
            ("redis://example.com:1234", "example.com", 1234),
            ("localhost:6379", "localhost", 6379),  # Without redis:// prefix
            ("localhost", "localhost", 6379),  # Just hostname
        ]

        for url, expected_host, expected_port in test_cases:
            mock_config = Mock()
            mock_config.task_queue = Mock()
            mock_config.task_queue.redis_url = url
            mock_config.task_queue.redis_password = None
            mock_config.task_queue.redis_database = 0

            with patch(
                "src.services.task_queue.worker.get_config", return_value=mock_config
            ):
                settings = WorkerSettings.get_redis_settings()

            assert settings.host == expected_host
            assert settings.port == expected_port

    def test_on_startup(self):
        """Test worker startup callback."""
        ctx = {"worker_id": "test_worker"}

        # Should not raise any exceptions
        WorkerSettings.on_startup(ctx)

    def test_on_shutdown(self):
        """Test worker shutdown callback."""
        ctx = {"worker_id": "test_worker"}

        # Should not raise any exceptions
        WorkerSettings.on_shutdown(ctx)

    def test_on_job_start(self):
        """Test job start callback."""
        ctx = {"job_id": "job_123", "job_try": {"function": "test_task"}}

        # Should not raise any exceptions
        WorkerSettings.on_job_start(ctx)

    def test_on_job_end(self):
        """Test job end callback."""
        ctx = {
            "job_id": "job_123",
            "job_try": {"function": "test_task"},
            "result": "success",
        }

        # Should not raise any exceptions
        WorkerSettings.on_job_end(ctx)

    def test_module_level_settings(self):
        """Test module-level settings for ARQ worker."""
        # These are imported and used by ARQ directly
        # Verify they exist and have expected types
        assert redis_settings is not None
        assert isinstance(max_jobs, int)
        assert isinstance(job_timeout, int)
        assert isinstance(functions, list)
        assert isinstance(cron_jobs, list)
        assert callable(on_startup)
        assert callable(on_shutdown)
        assert callable(on_job_start)
        assert callable(on_job_end)
