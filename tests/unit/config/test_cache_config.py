"""Test CacheConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import CacheConfig


class TestCacheConfig:
    """Test CacheConfig model validation and behavior."""

    def test_default_values(self):
        """Test CacheConfig with default values."""
        config = CacheConfig()

        # Check boolean flags
        assert config.enable_caching is True
        assert config.enable_local_cache is True
        assert config.enable_dragonfly_cache is True

        # Check URLs and connections
        assert config.dragonfly_url == "redis://localhost:6379"

        # Check TTL settings (in seconds)
        assert config.ttl_embeddings == 86400  # 24 hours
        assert config.ttl_crawl == 3600  # 1 hour
        assert config.ttl_queries == 7200  # 2 hours

        # Check local cache limits
        assert config.local_max_size == 1000
        assert config.local_max_memory_mb == 100.0

        # Check Redis settings
        assert config.redis_password is None
        assert config.redis_ssl is False
        assert config.redis_pool_size == 10

    def test_custom_values(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            enable_caching=False,
            enable_local_cache=False,
            enable_dragonfly_cache=True,
            dragonfly_url="redis://custom-host:6380",
            ttl_embeddings=43200,  # 12 hours
            ttl_crawl=1800,  # 30 minutes
            ttl_queries=3600,  # 1 hour
            local_max_size=500,
            local_max_memory_mb=50.0,
            redis_password="secret123",
            redis_ssl=True,
            redis_pool_size=20,
        )

        assert config.enable_caching is False
        assert config.enable_local_cache is False
        assert config.dragonfly_url == "redis://custom-host:6380"
        assert config.ttl_embeddings == 43200
        assert config.local_max_size == 500
        assert config.local_max_memory_mb == 50.0
        assert config.redis_password == "secret123"
        assert config.redis_ssl is True
        assert config.redis_pool_size == 20

    def test_ttl_non_negative_constraint(self):
        """Test that TTL values cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(ttl_embeddings=-1)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("ttl_embeddings",)
        assert "greater than or equal to 0" in str(errors[0]["msg"])

    def test_local_cache_size_positive_constraint(self):
        """Test that local cache size must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(local_max_size=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("local_max_size",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_local_cache_memory_positive_constraint(self):
        """Test that local cache memory must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(local_max_memory_mb=-10.0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("local_max_memory_mb",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_redis_pool_size_positive_constraint(self):
        """Test that Redis pool size must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(redis_pool_size=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("redis_pool_size",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(enable_caching=True, unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_field_types(self):
        """Test field type validation."""
        # Test boolean fields with invalid types
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(enable_caching={"value": True})  # Dict can't coerce to bool

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("enable_caching",)

        # Test integer fields with invalid types
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(ttl_embeddings="24hours")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("ttl_embeddings",)

        # Test float fields with invalid types
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(local_max_memory_mb="100MB")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("local_max_memory_mb",)

    def test_model_dump(self):
        """Test model serialization."""
        config = CacheConfig(enable_caching=True, redis_password="secret")

        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["enable_caching"] is True
        assert data["redis_password"] == "secret"
        assert "dragonfly_url" in data
        assert "ttl_embeddings" in data

    def test_model_dump_json(self):
        """Test model JSON serialization."""
        config = CacheConfig(local_max_memory_mb=150.5, redis_pool_size=15)

        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert '"local_max_memory_mb":150.5' in json_str
        assert '"redis_pool_size":15' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = CacheConfig(enable_caching=True, ttl_embeddings=86400)

        # Use model_copy instead of copy
        updated = original.model_copy(update={"ttl_embeddings": 43200})

        assert original.ttl_embeddings == 86400
        assert updated.ttl_embeddings == 43200
        assert updated.enable_caching is True  # Other fields preserved
