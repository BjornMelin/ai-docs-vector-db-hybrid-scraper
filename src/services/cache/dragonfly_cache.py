import typing
"""DragonflyDB cache implementation with advanced performance optimizations."""

import json  # noqa: PLC0415
import logging  # noqa: PLC0415
from typing import Any

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError

from .base import CacheInterface

logger = logging.getLogger(__name__)

# Import monitoring registry for metrics integration
try:
    from ..monitoring.metrics import get_metrics_registry

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class DragonflyCache(CacheInterface[Any]):
    """High-performance cache using DragonflyDB with Redis compatibility."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int | None = 3600,  # 1 hour
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        socket_keepalive: bool = True,
        retry_on_timeout: bool = True,
        max_retries: int = 3,
        key_prefix: str = "",
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # bytes
    ):
        """Initialize DragonflyDB cache with optimizations.

        Args:
            redis_url: DragonflyDB connection URL (Redis-compatible)
            default_ttl: Default TTL in seconds
            max_connections: Maximum connections in pool (DragonflyDB scales better)
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            socket_keepalive: Enable socket keepalive
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

        # Configure retry strategy with exponential backoff
        retry_strategy = None
        if retry_on_timeout:
            retry_strategy = Retry(
                backoff=ExponentialBackoff(base=0.1, cap=1.0),
                retries=max_retries,
                supported_errors=(ConnectionError, TimeoutError),
            )

        # Create optimized connection pool for DragonflyDB
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=socket_keepalive,
            retry=retry_strategy,
            decode_responses=False,  # Handle encoding/decoding manually
        )

        self._client: redis.Redis | None = None

        # Initialize metrics registry if available
        self.metrics_registry = None
        if MONITORING_AVAILABLE:
            try:
                self.metrics_registry = get_metrics_registry()
                logger.debug("DragonflyDB cache monitoring enabled")
            except Exception as e:
                logger.debug(f"DragonflyDB cache monitoring disabled: {e}")

    @property
    async def client(self) -> redis.Redis:
        """Get DragonflyDB client with lazy initialization."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
            # Test connection and ensure DragonflyDB is responding
            await self._client.ping()
            logger.info("DragonflyDB cache connection established")
        return self._client

    def _make_key(self, key: str) -> str:
        """Create full key with prefix."""
        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage with optional compression."""
        # Convert to JSON string then encode
        json_str = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        data = json_str.encode("utf-8")

        # Compress if enabled and above threshold
        if self.enable_compression and len(data) > self.compression_threshold:
            import zlib

            # Use compression with DragonflyDB optimization
            # DragonflyDB handles zstd natively, but we'll use zlib for compatibility
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
        """Get value from DragonflyDB."""
        # Monitor cache operations with Prometheus if available
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_cache_performance(
                cache_type="dragonfly", operation="get"
            )

            async def _monitored_get():
                return await self._execute_get(key)

            return await decorator(_monitored_get)()
        else:
            return await self._execute_get(key)

    async def _execute_get(self, key: str) -> Any | None:
        """Execute the actual get operation."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            data = await client.get(full_key)

            if data is None:
                return None

            return self._deserialize(data)

        except RedisError as e:
            logger.error(f"DragonflyDB get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in DragonflyDB with TTL and conditional options."""
        # Monitor cache operations with Prometheus if available
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_cache_performance(
                cache_type="dragonfly", operation="set"
            )

            async def _monitored_set():
                return await self._execute_set(key, value, ttl, nx, xx)

            return await decorator(_monitored_set)()
        else:
            return await self._execute_set(key, value, ttl, nx, xx)

    async def _execute_set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Execute the actual set operation."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            data = self._serialize(value)

            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Set with conditional flags
            result = await client.set(
                full_key,
                data,
                ex=ttl,  # TTL in seconds
                nx=nx,  # Only set if not exists
                xx=xx,  # Only set if exists
            )

            return bool(result)

        except RedisError as e:
            logger.error(f"DragonflyDB set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from DragonflyDB."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            result = await client.delete(full_key)
            return bool(result)

        except RedisError as e:
            logger.error(f"DragonflyDB delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in DragonflyDB."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            result = await client.exists(full_key)
            return bool(result)

        except RedisError as e:
            logger.error(f"DragonflyDB exists error for key {key}: {e}")
            return False

    async def clear(self) -> int:
        """Clear cache entries with prefix."""
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
            logger.error(f"DragonflyDB clear error: {e}")
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
            logger.error(f"DragonflyDB size error: {e}")
            return 0

    async def close(self) -> None:
        """Close DragonflyDB connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
        await self.pool.aclose()

    # Batch operations using DragonflyDB's superior pipeline performance
    async def get_many(self, keys: list[str]) -> dict[str, Any | None]:
        """Get multiple values using optimized pipeline."""
        try:
            client = await self.client
            results = {}

            # DragonflyDB handles pipelines more efficiently than Redis
            async with client.pipeline(transaction=False) as pipe:
                # Queue all get operations
                for key in keys:
                    full_key = self._make_key(key)
                    pipe.get(full_key)

                # Execute pipeline - DragonflyDB's superior performance shines here
                values = await pipe.execute()

                # Process results
                for key, data in zip(keys, values, strict=False):
                    if data is not None:
                        results[key] = self._deserialize(data)
                    else:
                        results[key] = None

            return results

        except RedisError as e:
            logger.error(f"DragonflyDB get_many error: {e}")
            # Return None for all keys on error
            return dict.fromkeys(keys)

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> dict[str, bool]:
        """Set multiple values using optimized pipeline."""
        try:
            client = await self.client
            results = {}

            if ttl is None:
                ttl = self.default_ttl

            # Use DragonflyDB's enhanced pipeline performance
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
            logger.error(f"DragonflyDB set_many error: {e}")
            # Return False for all keys on error
            return dict.fromkeys(items, False)

    async def delete_many(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values efficiently."""
        try:
            client = await self.client
            results = {}

            # Convert to full keys
            full_keys = [self._make_key(key) for key in keys]

            # DragonflyDB handles bulk deletes very efficiently
            deleted_count = await client.delete(*full_keys)

            # For exact results, we'd need individual checks
            # For performance, assume uniform success/failure
            if deleted_count == len(keys):
                # All keys deleted
                results = dict.fromkeys(keys, True)
            elif deleted_count == 0:
                # No keys deleted
                results = dict.fromkeys(keys, False)
            else:
                # Mixed results - need individual checks for accuracy
                for key in keys:
                    full_key = self._make_key(key)
                    exists = await client.exists(full_key)
                    results[key] = not bool(exists)

            return results

        except RedisError as e:
            logger.error(f"DragonflyDB delete_many error: {e}")
            # Return False for all keys on error
            return dict.fromkeys(keys, False)

    # DragonflyDB-specific optimized methods
    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values efficiently using MGET."""
        try:
            client = await self.client
            full_keys = [self._make_key(key) for key in keys]

            # DragonflyDB's MGET performance is superior to Redis
            values = await client.mget(full_keys)

            results = []
            for value in values:
                if value is not None:
                    results.append(self._deserialize(value))
                else:
                    results.append(None)

            return results

        except RedisError as e:
            logger.error(f"DragonflyDB mget error: {e}")
            return [None] * len(keys)

    async def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values efficiently using MSET + EXPIRE."""
        try:
            client = await self.client

            # Serialize values
            serialized = {}
            for key, value in mapping.items():
                full_key = self._make_key(key)
                serialized[full_key] = self._serialize(value)

            # Use pipeline for atomic operation
            async with client.pipeline() as pipe:
                pipe.mset(serialized)

                # Set TTL if provided
                if ttl:
                    for full_key in serialized:
                        pipe.expire(full_key, ttl)

                await pipe.execute()

            return True

        except RedisError as e:
            logger.error(f"DragonflyDB mset error: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key in seconds."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            ttl = await client.ttl(full_key)
            return max(0, ttl)  # -1 means no TTL, -2 means key doesn't exist

        except RedisError as e:
            logger.error(f"DragonflyDB ttl error for key {key}: {e}")
            return 0

    async def expire(self, key: str, ttl: int) -> bool:
        """Set new TTL for existing key."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            result = await client.expire(full_key, ttl)
            return bool(result)

        except RedisError as e:
            logger.error(f"DragonflyDB expire error for key {key}: {e}")
            return False

    async def scan_keys(self, pattern: str, count: int = 100) -> list[str]:
        """Scan keys matching pattern (DragonflyDB optimized)."""
        try:
            client = await self.client
            full_pattern = f"{self.key_prefix}{pattern}" if self.key_prefix else pattern

            keys = []
            async for key in client.scan_iter(match=full_pattern, count=count):
                # Remove prefix from returned keys
                processed_key = key
                if self.key_prefix and key.startswith(self.key_prefix.encode()):
                    processed_key = key[len(self.key_prefix) :]
                keys.append(
                    processed_key.decode("utf-8")
                    if isinstance(processed_key, bytes)
                    else processed_key
                )

            return keys

        except RedisError as e:
            logger.error(f"DragonflyDB scan_keys error: {e}")
            return []

    async def get_memory_usage(self, key: str) -> int:
        """Get memory usage of a key (DragonflyDB feature)."""
        try:
            client = await self.client
            full_key = self._make_key(key)
            # DragonflyDB supports MEMORY USAGE command
            usage = await client.memory_usage(full_key)
            return usage or 0

        except (RedisError, AttributeError) as e:
            logger.debug(f"DragonflyDB memory_usage error for key {key}: {e}")
            return 0
