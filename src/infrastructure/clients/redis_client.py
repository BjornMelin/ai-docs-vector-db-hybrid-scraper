"""Redis client provider."""

import logging
from typing import Any


try:
    import redis.asyncio as redis
except ImportError:
    # Create a placeholder if redis is not available
    class redis:
        class Redis:
            pass


logger = logging.getLogger(__name__)


class RedisClientProvider:
    """Provider for Redis client with health checks and circuit breaker."""

    def __init__(
        self,
        redis_client: redis.Redis,
    ):
        self._client = redis_client
        self._healthy = True

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
            self._healthy = True
            return True
        except Exception as e:
            logger.warning(
                f"Redis health check failed: {e}"
            )  # TODO: Convert f-string to logging format
            self._healthy = False
            return False

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
            raise RuntimeError("Redis client is not available or unhealthy")

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
            raise RuntimeError("Redis client is not available or unhealthy")

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
            raise RuntimeError("Redis client is not available or unhealthy")

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
            raise RuntimeError("Redis client is not available or unhealthy")

        return await self.client.exists(*keys)
