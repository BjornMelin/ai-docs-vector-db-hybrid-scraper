"""Dragonfly-backed cache implementation using the async redis client."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis
from redis.exceptions import RedisError

from src.services.cache.base import CacheInterface
from src.services.observability.tracing import log_extra_with_trace, trace_function


logger = logging.getLogger(__name__)


def _log_extra(event: str, **metadata: Any) -> dict[str, Any]:
    """Return structured logging extras enriched with trace identifiers."""

    metadata.setdefault("component", "cache.dragonfly")
    return log_extra_with_trace(event, **metadata)


class DragonflyCache(CacheInterface[Any]):
    """Async cache implementation backed by a Dragonfly deployment."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        *,
        default_ttl: int | None = 3600,
        key_prefix: str = "",
        max_connections: int = 50,
        client: redis.Redis | None = None,
    ) -> None:
        """Initialise the cache wrapper around the redis-py asyncio client.

        Args:
            redis_url: Connection string for the Dragonfly server.
            default_ttl: Default expiry in seconds applied to entries.
            key_prefix: Prefix appended to every key stored in Dragonfly.
            max_connections: Size of the connection pool when the client is
                instantiated by this cache wrapper.
            client: Optional preconfigured redis client instance. When
                provided the cache will not manage the client's lifecycle.
        """

        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.max_connections = max_connections
        self.compression_enabled = False

        if client is None:
            self._client = redis.Redis.from_url(
                redis_url,
                max_connections=max_connections,
                decode_responses=False,
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
            pool = getattr(client, "connection_pool", None)
            if pool is not None:
                self.max_connections = getattr(
                    pool, "max_connections", self.max_connections
                )

    @property
    def client(self) -> redis.Redis:
        """Return the underlying async redis client."""

        return self._client

    def _format_key(self, key: str) -> str:
        """Apply the configured prefix to ``key`` when present."""

        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def _serialize(self, value: Any) -> bytes:
        """Serialise ``value`` to bytes using JSON encoding."""

        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    def _deserialize(self, payload: bytes) -> Any:
        """Deserialize cached payload back into a Python object."""

        return json.loads(payload.decode("utf-8"))

    async def initialize(self) -> None:
        """Verify connectivity with the backing Dragonfly instance."""

        try:
            await self.client.ping()
        except RedisError:  # pragma: no cover - network dependant
            logger.exception(
                "Failed to initialise Dragonfly cache",
                extra=_log_extra("cache.dragonfly.initialize", url=self.redis_url),
            )
            raise

    @trace_function("cache.dragonfly.get")
    async def get(self, key: str) -> Any | None:
        """Return cached value for ``key`` or ``None`` when missing."""

        try:
            result = await self.client.get(self._format_key(key))
        except RedisError:
            logger.exception(
                "Dragonfly get failed for key %s",
                key,
                extra=_log_extra("cache.dragonfly.get", key=key),
            )
            return None

        if result is None:
            return None
        return self._deserialize(result)

    @trace_function("cache.dragonfly.set")
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store ``value`` under ``key`` with an optional TTL."""

        expiry = ttl if ttl is not None else self.default_ttl
        try:
            payload = self._serialize(value)
        except (TypeError, ValueError):
            logger.exception(
                "Failed serialising cache payload for %s",
                key,
                extra=_log_extra("cache.dragonfly.serialize", key=key),
            )
            return False

        try:
            return bool(
                await self.client.set(self._format_key(key), payload, ex=expiry)
            )
        except RedisError:
            logger.exception(
                "Dragonfly set failed for key %s",
                key,
                extra=_log_extra("cache.dragonfly.set", key=key, ttl=expiry),
            )
            return False

    async def delete(self, key: str) -> bool:
        """Delete ``key`` from the distributed cache."""

        try:
            return bool(await self.client.delete(self._format_key(key)))
        except RedisError:
            logger.exception(
                "Dragonfly delete failed for key %s",
                key,
                extra=_log_extra("cache.dragonfly.delete", key=key),
            )
            return False

    async def exists(self, key: str) -> bool:
        """Return whether ``key`` exists in Dragonfly."""

        try:
            return bool(await self.client.exists(self._format_key(key)))
        except RedisError:
            return False

    async def clear(self) -> int:
        """Remove all keys managed by this cache instance."""

        pattern = f"{self.key_prefix}*" if self.key_prefix else None
        return await self.clear_pattern(pattern or "*")

    async def clear_pattern(self, pattern: str) -> int:
        """Remove keys matching ``pattern`` within Dragonfly."""

        deleted = 0
        full_pattern = pattern
        if self.key_prefix and not pattern.startswith(self.key_prefix):
            full_pattern = f"{self.key_prefix}{pattern}"

        try:
            async for key in self.client.scan_iter(match=full_pattern, count=200):
                await self.client.delete(key)
                deleted += 1
        except RedisError:
            logger.exception(
                "Dragonfly clear_pattern failed for %s",
                pattern,
                extra=_log_extra("cache.dragonfly.clear_pattern", pattern=pattern),
            )
            return deleted

        return deleted

    async def scan_keys(self, pattern: str, count: int = 200) -> list[str]:
        """Return keys matching ``pattern`` without deleting them."""

        keys: list[str] = []
        full_pattern = pattern
        if self.key_prefix and not pattern.startswith(self.key_prefix):
            full_pattern = f"{self.key_prefix}{pattern}"

        try:
            async for key in self.client.scan_iter(match=full_pattern, count=count):
                decoded = key.decode("utf-8") if isinstance(key, bytes) else key
                if self.key_prefix and decoded.startswith(self.key_prefix):
                    decoded = decoded[len(self.key_prefix) :]
                keys.append(decoded)
        except RedisError:
            logger.exception(
                "Dragonfly scan_keys failed for %s",
                pattern,
                extra=_log_extra("cache.dragonfly.scan_keys", pattern=pattern),
            )
            return []

        return keys

    async def size(self) -> int:
        """Return an approximate count of cached entries."""

        try:
            if self.key_prefix:
                count = 0
                async for _ in self.client.scan_iter(
                    match=f"{self.key_prefix}*", count=200
                ):
                    count += 1
                return count
            return int(await self.client.dbsize())
        except RedisError:
            logger.exception(
                "Dragonfly size probe failed",
                extra=_log_extra("cache.dragonfly.size"),
            )
            return 0

    async def ttl(self, key: str) -> int:
        """Return remaining TTL for ``key`` in seconds."""

        try:
            ttl_value = await self.client.ttl(self._format_key(key))
        except RedisError:
            logger.exception(
                "Dragonfly ttl lookup failed for %s",
                key,
                extra=_log_extra("cache.dragonfly.ttl", key=key),
            )
            return 0
        return max(0, int(ttl_value))

    async def expire(self, key: str, ttl: int) -> bool:
        """Apply a TTL to an existing key."""

        try:
            return bool(await self.client.expire(self._format_key(key), ttl))
        except RedisError:
            logger.exception(
                "Dragonfly expire failed for %s",
                key,
                extra=_log_extra("cache.dragonfly.expire", key=key, ttl=ttl),
            )
            return False

    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Fetch multiple keys at once using MGET."""

        try:
            full_keys = [self._format_key(key) for key in keys]
            results = await self.client.mget(full_keys)
        except RedisError:
            logger.exception(
                "Dragonfly mget failed for %s keys",
                len(keys),
                extra=_log_extra("cache.dragonfly.mget", key_count=len(keys)),
            )
            return [None] * len(keys)

        payloads: list[Any | None] = []
        for value in results:
            if value is None:
                payloads.append(None)
            else:
                payloads.append(self._deserialize(value))
        return payloads

    async def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> bool:
        """Store multiple keys atomically, optionally applying a TTL."""

        if not mapping:
            return True

        serialised: dict[str, bytes] = {}
        try:
            for key, value in mapping.items():
                serialised[self._format_key(key)] = self._serialize(value)
        except (TypeError, ValueError):
            logger.exception(
                "Dragonfly mset serialisation failed",
                extra=_log_extra("cache.dragonfly.mset", key_count=len(mapping)),
            )
            return False

        try:
            if ttl is None:
                await self.client.mset(serialised)
            else:
                pipeline = self.client.pipeline(transaction=True)
                for key, payload in serialised.items():
                    pipeline.set(key, payload, ex=ttl)
                await pipeline.execute()
        except RedisError:
            logger.exception(
                "Dragonfly mset failed",
                extra=_log_extra("cache.dragonfly.mset", key_count=len(mapping)),
            )
            return False
        else:
            return True

    async def delete_many(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple keys in a single pipelined operation."""

        if not keys:
            return {}

        formatted_keys = [self._format_key(key) for key in keys]
        pipeline = self.client.pipeline(transaction=True)
        for key in formatted_keys:
            pipeline.delete(key)

        try:
            results = await pipeline.execute()
        except RedisError:
            logger.exception(
                "Dragonfly delete_many failed",
                extra=_log_extra("cache.dragonfly.delete_many", key_count=len(keys)),
            )
            return dict.fromkeys(keys, False)

        return {
            original: bool(deleted)
            for original, deleted in zip(keys, results, strict=False)
        }

    async def close(self) -> None:
        """Close the underlying client when this instance owns it."""

        if self._owns_client:
            await self.client.aclose()
