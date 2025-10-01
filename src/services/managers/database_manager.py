"""Database manager for vector store and cache operations."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import redis
from dependency_injector.wiring import Provide, inject

from src.config import CacheType
from src.infrastructure.container import ApplicationContainer
from src.services.errors import APIError
from src.services.vector_db.adapter_base import VectorRecord


if TYPE_CHECKING:
    from src.services.cache.manager import CacheManager
    from src.services.vector_db.service import VectorStoreService

# Imports to avoid circular dependencies
try:
    from src.config import get_config
    from src.services.cache.manager import CacheManager as _CacheManager
    from src.services.vector_db.service import VectorStoreService as _VectorStoreService
except ImportError:
    get_config = None
    _VectorStoreService = None
    _CacheManager = None

logger = logging.getLogger(__name__)


def _raise_required_services_not_available() -> None:
    """Raise ImportError for required services not available."""

    msg = "Required services not available"
    raise ImportError(msg)


class DatabaseManager:
    """Focused manager for database and cache operations.

    Handles vector database operations, Redis caching,
    and distributed cache management with single responsibility.
    """

    def __init__(self):
        """Initialize database manager."""
        self._redis_client: redis.Redis | None = None
        self._cache_manager: CacheManager | None = None
        self._vector_service: VectorStoreService | None = None
        self._initialized = False

    @inject
    async def initialize(
        self,
        qdrant_client: Any | None = Provide[ApplicationContainer.qdrant_client],
        redis_client: redis.Redis = Provide[ApplicationContainer.redis_client],
    ) -> None:
        """Initialize database clients using dependency injection.

        Args:
            qdrant_client: vector store service from DI container
            redis_client: Redis client from DI container
        """

        if self._initialized:
            return

        self._redis_client = redis_client

        # Initialize cache manager
        if _CacheManager is None:
            msg = "CacheManager not available"
            raise ImportError(msg)

        cache_root = Path("cache") / "database_manager"
        self._cache_manager = _CacheManager(
            enable_local_cache=True,
            enable_distributed_cache=True,
            enable_specialized_caches=True,
            local_cache_path=cache_root,
        )

        # Initialize vector store service
        if get_config is None or _VectorStoreService is None:
            _raise_required_services_not_available()
        else:
            try:
                config = get_config()
                self._vector_service = _VectorStoreService(config)
                await self._vector_service.initialize()
            except Exception as exc:
                logger.exception("Failed to initialize vector store service")
                msg = f"Failed to initialize vector store service: {exc}"
                raise APIError(msg) from exc

        self._initialized = True
        logger.info("DatabaseManager initialized with DI clients")

    async def cleanup(self) -> None:
        """Cleanup database resources."""

        if self._cache_manager:
            await self._cache_manager.close()
            self._cache_manager = None

        if self._vector_service:
            await self._vector_service.cleanup()
            self._vector_service = None

        self._redis_client = None
        self._initialized = False
        logger.info("DatabaseManager cleaned up")

    # Vector Store Operations
    async def get_collections(self) -> list[str]:
        """Get list of vector store collections."""

        if not self._vector_service:
            msg = "Vector store service not available"
            raise APIError(msg)

        try:
            return await self._vector_service.list_collections()
        except Exception as exc:
            logger.exception("Failed to get collections")
            msg = f"Failed to get collections: {exc}"
            raise APIError(msg) from exc

    async def store_embeddings(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> bool:
        """Store embeddings in a vector collection.

        Args:
            collection_name: Name of the collection
            points: List of points to store

        Returns:
            True if successful
        """

        if not self._vector_service:
            msg = "Vector store service not available"
            raise APIError(msg)

        try:
            records: list[VectorRecord] = []
            for point in points:
                vector = point.get("vector")
                if vector is None:
                    msg = "Each point must include a dense vector"
                    raise ValueError(msg)
                records.append(
                    VectorRecord(
                        id=str(point.get("id")),
                        vector=list(vector),
                        payload=point.get("payload"),
                        sparse_vector=point.get("sparse_vector"),
                    )
                )
            await self._vector_service.upsert_vectors(collection_name, records)
        except Exception as exc:
            logger.exception("Failed to store embeddings")
            msg = f"Failed to store embeddings: {exc}"
            raise APIError(msg) from exc

        return True

    async def search_similar(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in the vector store.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results
            filter_conditions: Optional filter conditions

        Returns:
            List of search results
        """

        if not self._vector_service:
            msg = "Vector store service not available"
            raise APIError(msg)

        try:
            matches = await self._vector_service.search_vector(
                collection_name,
                query_vector,
                limit=limit,
                filters=filter_conditions,
            )
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "payload": dict(match.payload or {}),
                }
                for match in matches
            ]
        except Exception as exc:
            logger.exception("Failed to search vectors")
            msg = f"Failed to search vectors: {exc}"
            raise APIError(msg) from exc

    # Cache Operations
    async def cache_get(
        self,
        key: str,
        cache_type: CacheType = CacheType.LOCAL,
        default: Any = None,
    ) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            cache_type: Type of cached data
            default: Default value if not found

        Returns:
            Cached value or default
        """

        if not self._cache_manager:
            return default

        try:
            return await self._cache_manager.get(key, cache_type, default)
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Cache get failed for %s: %s", key, e)
            return default

    async def cache_set(
        self,
        key: str,
        value: Any,
        cache_type: CacheType = CacheType.LOCAL,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cached data
            ttl: Time to live in seconds

        Returns:
            True if successful
        """

        if not self._cache_manager:
            return False

        try:
            return await self._cache_manager.set(key, value, cache_type, ttl)
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Cache set failed for %s: %s", key, e)
            return False

    async def cache_delete(
        self,
        key: str,
        cache_type: CacheType = CacheType.LOCAL,
    ) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key
            cache_type: Type of cached data

        Returns:
            True if successful
        """

        if not self._cache_manager:
            return False

        try:
            return await self._cache_manager.delete(key, cache_type)
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Cache delete failed for %s: %s", key, e)
            return False

    # Redis Operations
    async def redis_ping(self) -> bool:
        """Check Redis connectivity.

        Returns:
            True if Redis is responsive
        """

        if not self._redis_client:
            return False

        try:
            await self._redis_client.ping()
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Redis ping failed: %s", e)
            return False

        return True

    async def redis_set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Set value in Redis.

        Args:
            key: Redis key
            value: Value to set
            ex: Expiration time in seconds

        Returns:
            True if successful
        """

        if not self._redis_client:
            return False

        try:
            await self._redis_client.set(key, value, ex=ex)
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Redis set failed for %s: %s", key, e)
            return False

        return True

    async def redis_get(self, key: str) -> str | None:
        """Get value from Redis."""

        if not self._redis_client:
            return None

        try:
            return await self._redis_client.get(key)
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning("Redis get failed for %s: %s", key, e)
            return None

    # Status and Metrics
    async def get_status(self) -> dict[str, Any]:
        """Get database manager status.

        Returns:
            Status information for all components
        """

        status = {
            "initialized": self._initialized,
            "vector_store": {
                "available": self._vector_service is not None,
                "collections": [],
            },
            "redis": {
                "available": self._redis_client is not None,
                "connected": False,
            },
            "cache": {
                "available": self._cache_manager is not None,
            },
        }

        # Check vector store status
        if self._vector_service:
            try:
                collections = await self.get_collections()
                status["vector_store"]["collections"] = collections
            except (ConnectionError, OSError, PermissionError):
                status["vector_store"]["available"] = False

        # Check Redis status
        if self._redis_client:
            status["redis"]["connected"] = await self.redis_ping()

        # Get cache stats
        if self._cache_manager:
            try:
                cache_stats = await self._cache_manager.get_stats()
                status["cache"]["stats"] = cache_stats
            except (ConnectionError, OSError, PermissionError):
                status["cache"]["available"] = False

        return status

    async def get_cache_manager(self) -> Optional["CacheManager"]:
        """Get cache manager instance."""

        return self._cache_manager

    async def get_vector_service(self) -> Optional["VectorStoreService"]:
        """Get vector store service instance."""

        return self._vector_service
