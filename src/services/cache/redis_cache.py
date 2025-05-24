"""Redis cache implementation with async support and connection pooling."""

import json
import logging
from typing import Any

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError

from .base import CacheInterface

logger = logging.getLogger(__name__)


class RedisCache(CacheInterface[Any]):
    """Redis-based distributed cache with connection pooling."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int | None = 3600,  # 1 hour
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        max_retries: int = 3,
        key_prefix: str = "",
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # bytes
    ):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            retry_on_timeout: Enable retry on timeout
            max_retries: Maximum retry attempts
            key_prefix: Prefix for all keys
            enable_compression: Enable value compression
            compression_threshold: Minimum size for compression
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold

        # Configure retry strategy
        retry_strategy = None
        if retry_on_timeout:
            retry_strategy = Retry(
                ExponentialBackoff(),
                max_retries,
                supported_errors=(ConnectionError, TimeoutError),
            )

        # Create connection pool
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            retry=retry_strategy,
            decode_responses=False,  # We'll handle encoding/decoding
        )

        self._client: redis.Redis | None = None

    @property
    async def client(self) -> redis.Redis:
        """Get Redis client (lazy initialization)."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
            # Test connection
            await self._client.ping()
        return self._client

    def _make_key(self, key: str) -> str:
        """Create full key with prefix."""
        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        # Convert to JSON string then encode
        json_str = json.dumps(value, separators=(",", ":"))
        data = json_str.encode("utf-8")

        # Compress if enabled and above threshold
        if self.enable_compression and len(data) > self.compression_threshold:
            import zlib

            # Add compression marker
            data = b"Z:" + zlib.compress(data, level=6)

        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not data:
            return None

        # Check for compression marker
        if self.enable_compression and data.startswith(b"Z:"):
            import zlib

            data = zlib.decompress(data[2:])

        # Decode and parse JSON
        json_str = data.decode("utf-8")
        return json.loads(json_str)

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            data = await client.get(full_key)

            if data is None:
                return None

            return self._deserialize(data)

        except RedisError as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in Redis with TTL."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            data = self._serialize(value)

            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            if ttl is not None:
                await client.setex(full_key, ttl, data)
            else:
                await client.set(full_key, data)

            return True

        except RedisError as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            result = await client.delete(full_key)
            return bool(result)

        except RedisError as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            result = await client.exists(full_key)
            return bool(result)

        except RedisError as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False

    async def clear(self) -> int:
        """Clear all cache entries with prefix."""
        try:
            client = await self.client

            if self.key_prefix:
                # Use SCAN to find all keys with prefix
                pattern = f"{self.key_prefix}*"
                count = 0

                async for key in client.scan_iter(match=pattern, count=100):
                    await client.delete(key)
                    count += 1

                return count
            else:
                # Flush entire database (use with caution!)
                await client.flushdb()
                return -1  # Unknown count

        except RedisError as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def size(self) -> int:
        """Get approximate cache size."""
        try:
            client = await self.client

            if self.key_prefix:
                # Count keys with prefix
                pattern = f"{self.key_prefix}*"
                count = 0

                async for _ in client.scan_iter(match=pattern, count=100):
                    count += 1

                return count
            else:
                # Get total database size
                info = await client.info("keyspace")
                # Parse db0 keys count
                db_info = info.get("db0", {})
                if isinstance(db_info, dict):
                    return db_info.get("keys", 0)
                return 0

        except RedisError as e:
            logger.error(f"Redis size error: {e}")
            return 0

    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
        await self.pool.aclose()

    # Batch operations using pipeline
    async def get_many(self, keys: list[str]) -> dict[str, Any | None]:
        """Get multiple values using pipeline."""
        try:
            client = await self.client
            results = {}

            async with client.pipeline(transaction=False) as pipe:
                # Queue all get operations
                for key in keys:
                    full_key = self._make_key(key)
                    pipe.get(full_key)

                # Execute pipeline
                values = await pipe.execute()

                # Process results
                for key, data in zip(keys, values, strict=False):
                    if data is not None:
                        results[key] = self._deserialize(data)
                    else:
                        results[key] = None

            return results

        except RedisError as e:
            logger.error(f"Redis get_many error: {e}")
            # Return None for all keys on error
            return dict.fromkeys(keys)

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> dict[str, bool]:
        """Set multiple values using pipeline."""
        try:
            client = await self.client
            results = {}

            if ttl is None:
                ttl = self.default_ttl

            async with client.pipeline(transaction=False) as pipe:
                # Queue all set operations
                for key, value in items.items():
                    full_key = self._make_key(key)
                    data = self._serialize(value)

                    if ttl is not None:
                        pipe.setex(full_key, ttl, data)
                    else:
                        pipe.set(full_key, data)

                # Execute pipeline
                responses = await pipe.execute()

                # All SET operations return True on success
                for key, response in zip(items.keys(), responses, strict=False):
                    results[key] = bool(response)

            return results

        except RedisError as e:
            logger.error(f"Redis set_many error: {e}")
            # Return False for all keys on error
            return dict.fromkeys(items, False)

    async def delete_many(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values using pipeline."""
        try:
            client = await self.client
            results = {}

            # Convert to full keys
            full_keys = [self._make_key(key) for key in keys]

            # Delete all at once
            await client.delete(*full_keys)

            # We don't know which specific keys were deleted
            # So we'll do individual checks
            for key in keys:
                full_key = self._make_key(key)
                exists = await client.exists(full_key)
                results[key] = not bool(exists)

            return results

        except RedisError as e:
            logger.error(f"Redis delete_many error: {e}")
            # Return False for all keys on error
            return dict.fromkeys(keys, False)

    # Additional Redis-specific methods
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key in seconds."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            ttl = await client.ttl(full_key)
            return max(0, ttl)  # -1 means no TTL, -2 means key doesn't exist

        except RedisError as e:
            logger.error(f"Redis ttl error for key {key}: {e}")
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        """Set new TTL for existing key."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            result = await client.expire(full_key, ttl)
            return bool(result)

        except RedisError as e:
            logger.error(f"Redis expire error for key {key}: {e}")
            return False
