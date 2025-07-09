"""Redis client provider."""

import logging
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


try:
    import redis.asyncio as redis
except ImportError:
    # Create a placeholder if redis is not available
    class RedisModule:
        """Placeholder Redis module when redis is not available."""

        class Redis:
            """Placeholder Redis client class."""

    redis = RedisModule()


logger = logging.getLogger(__name__)


class RedisConfig(BaseModel):
    """Configuration for Redis client with validation."""

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "password": None,
                    "socket_timeout": 5.0,
                    "socket_keepalive": True,
                    "max_connections": 10,
                    "retry_on_timeout": True,
                }
            ]
        },
    )

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")
    socket_timeout: float = Field(
        default=5.0, ge=0.1, le=300.0, description="Socket timeout in seconds"
    )
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")
    max_connections: int = Field(
        default=10, ge=1, le=100, description="Maximum connections in pool"
    )
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    encoding: str = Field(default="utf-8", description="String encoding")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate Redis host format."""
        if not v or v.strip() == "":
            msg = "Redis host cannot be empty"
            raise ValueError(msg)
        # Basic validation for common invalid hostnames
        if " " in v:
            msg = "Redis host cannot contain spaces"
            raise ValueError(msg)
        return v.strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str | None) -> str | None:
        """Validate Redis password."""
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty string to None
        return v

    @computed_field
    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL."""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"

    @computed_field
    @property
    def is_secure(self) -> bool:
        """Check if Redis configuration uses authentication."""
        return self.password is not None and len(self.password) > 0

    @computed_field
    @property
    def estimated_memory_usage_mb(self) -> float:
        """Estimate memory usage based on connection pool size."""
        # Rough estimate: each connection ~1MB
        return self.max_connections * 1.0


class RedisMetrics(BaseModel):
    """Metrics for Redis client operations."""

    model_config = ConfigDict(validate_assignment=True)

    total_operations: int = Field(
        default=0, ge=0, description="Total operations performed"
    )
    successful_operations: int = Field(
        default=0, ge=0, description="Successful operations"
    )
    failed_operations: int = Field(default=0, ge=0, description="Failed operations")
    cache_hits: int = Field(default=0, ge=0, description="Cache hits")
    cache_misses: int = Field(default=0, ge=0, description="Cache misses")
    avg_response_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average response time"
    )
    last_health_check: float | None = Field(
        default=None, description="Last health check timestamp"
    )

    @model_validator(mode="after")
    def validate_operation_consistency(self) -> "RedisMetrics":
        """Validate that operation counts are consistent."""
        if self.successful_operations + self.failed_operations > self.total_operations:
            msg = (
                "Sum of successful and failed operations cannot exceed total operations"
            )
            raise ValueError(msg)
        if self.cache_hits + self.cache_misses > self.total_operations:
            # This is allowed since not all operations are cache operations
            pass
        return self

    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate operation success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations

    @computed_field
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return self.cache_hits / total_cache_ops

    @computed_field
    @property
    def performance_category(self) -> str:
        """Categorize Redis performance."""
        if self.success_rate >= 0.98 and self.cache_hit_rate >= 0.8:
            return "excellent"
        if self.success_rate >= 0.95 and self.cache_hit_rate >= 0.6:
            return "good"
        if self.success_rate >= 0.90:
            return "fair"
        return "poor"


class RedisClientProvider:
    """Provider for Redis client with health checks and circuit breaker."""

    def __init__(
        self,
        redis_client: redis.Redis,
        config: RedisConfig | None = None,
    ):
        self._client = redis_client
        self._config = config
        self._healthy = True
        self._metrics = RedisMetrics()

    @property
    def client(self) -> redis.Redis | None:
        """Get the Redis client if available and healthy."""
        if not self._healthy:
            return None
        return self._client

    async def health_check(self) -> bool:
        """Check Redis client health."""
        try:
            if not self._client:
                return False

            # Simple ping to check connectivity
            await self._client.ping()
        except (AttributeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning("Redis health check failed: %s", e)
            self._healthy = False
            return False
        else:
            self._healthy = True
            return True

    async def get(self, key: str) -> Any | None:
        """Get value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Redis client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        """Set value with optional expiration.

        Args:
            key: Cache key
            value: Value to cache
            ex: Expiration in seconds

        Returns:
            True if successful

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Redis client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.set(key, value, ex=ex)

    async def delete(self, *keys: str) -> int:
        """Delete keys.

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Redis client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Check if keys exist.

        Args:
            *keys: Keys to check

        Returns:
            Number of existing keys

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Redis client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.exists(*keys)
