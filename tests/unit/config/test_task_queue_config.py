"""Tests for TaskQueueConfig model."""

import pytest
from pydantic import ValidationError
from src.config.models import TaskQueueConfig


class TestTaskQueueConfig:
    """Test TaskQueueConfig validation and defaults."""

    def test_default_values(self):
        """Test default TaskQueueConfig values."""
        config = TaskQueueConfig()

        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_password is None
        assert config.redis_database == 1
        assert config.max_jobs == 10
        assert config.job_timeout == 3600
        assert config.job_ttl == 86400
        assert config.max_tries == 3
        assert config.retry_delay == 60.0
        assert config.queue_name == "default"
        assert config.health_check_interval == 60
        assert config.worker_pool_size == 4

    def test_custom_values(self):
        """Test TaskQueueConfig with custom values."""
        config = TaskQueueConfig(
            redis_url="redis://custom:6380",
            redis_password="secret123",
            redis_database=5,
            max_jobs=20,
            job_timeout=7200,
            job_ttl=172800,
            max_tries=5,
            retry_delay=120.0,
            queue_name="high_priority",
            health_check_interval=30,
            worker_pool_size=8,
        )

        assert config.redis_url == "redis://custom:6380"
        assert config.redis_password == "secret123"
        assert config.redis_database == 5
        assert config.max_jobs == 20
        assert config.job_timeout == 7200
        assert config.job_ttl == 172800
        assert config.max_tries == 5
        assert config.retry_delay == 120.0
        assert config.queue_name == "high_priority"
        assert config.health_check_interval == 30
        assert config.worker_pool_size == 8

    def test_redis_database_validation(self):
        """Test redis_database validation."""
        # Valid range
        config = TaskQueueConfig(redis_database=0)
        assert config.redis_database == 0

        config = TaskQueueConfig(redis_database=15)
        assert config.redis_database == 15

        # Invalid - negative
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(redis_database=-1)
        assert "greater than or equal to 0" in str(exc_info.value)

        # Invalid - too high
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(redis_database=16)
        assert "less than or equal to 15" in str(exc_info.value)

    def test_positive_value_validation(self):
        """Test validation of positive numeric fields."""
        # max_jobs must be positive
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(max_jobs=0)
        assert "greater than 0" in str(exc_info.value)

        # job_timeout must be positive
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(job_timeout=-1)
        assert "greater than 0" in str(exc_info.value)

        # retry_delay must be positive
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(retry_delay=0.0)
        assert "greater than 0" in str(exc_info.value)

        # worker_pool_size must be positive
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(worker_pool_size=0)
        assert "greater than 0" in str(exc_info.value)

    def test_max_tries_validation(self):
        """Test max_tries validation."""
        # Valid range
        config = TaskQueueConfig(max_tries=1)
        assert config.max_tries == 1

        config = TaskQueueConfig(max_tries=10)
        assert config.max_tries == 10

        # Invalid - too low
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(max_tries=0)
        assert "greater than 0" in str(exc_info.value)

        # Invalid - too high
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(max_tries=11)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            TaskQueueConfig(unknown_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_from_dict(self):
        """Test creating TaskQueueConfig from dictionary."""
        data = {
            "redis_url": "redis://localhost:6379",
            "redis_database": 2,
            "max_jobs": 15,
            "queue_name": "background",
        }

        config = TaskQueueConfig(**data)

        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_database == 2
        assert config.max_jobs == 15
        assert config.queue_name == "background"
        # Other fields should have defaults
        assert config.job_timeout == 3600
        assert config.retry_delay == 60.0
