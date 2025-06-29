"""Database manager for Qdrant and cache operations."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from dependency_injector.wiring import Provide, inject

from src.config.enums import CacheType
from src.infrastructure.container import ApplicationContainer
from src.services.errors import APIError


if TYPE_CHECKING:
    import redis
    from qdrant_client.async_qdrant_client import AsyncQdrantClient

    from src.services.cache.manager import CacheManager
    from src.services.vector_db.service import QdrantService

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Focused manager for database and cache operations.

    Handles Qdrant vector database operations, Redis caching,
    and distributed cache management with single responsibility.
    """

    def __init__(self):
        """Initialize database manager."""
        self._qdrant_client: AsyncQdrantClient | None = None
        self._redis_client: redis.Redis | None = None
        self._cache_manager: CacheManager | None = None
        self._qdrant_service: QdrantService | None = None
        self._initialized = False

    @inject
    async def initialize(
        self,
        qdrant_client: "AsyncQdrantClient" = Provide[
            ApplicationContainer.qdrant_client
        ],
        redis_client: "redis.Redis" = Provide[ApplicationContainer.redis_client],
    ) -> None:
        """Initialize database clients using dependency injection.

        Args:
            qdrant_client: Qdrant client from DI container
            redis_client: Redis client from DI container
        """
        if self._initialized:
            return

        self._qdrant_client = qdrant_client
        self._redis_client = redis_client

        # Initialize cache manager
        from src.services.cache.manager import CacheManager

        self._cache_manager = CacheManager(
            enable_local_cache=True,
            enable_distributed_cache=True,
            enable_specialized_caches=True,
        )

        # Initialize Qdrant service
        try:
            from src.config import get_config
            from src.services.vector_db.service import QdrantService

            config = get_config()
            self._qdrant_service = QdrantService(config)
            await self._qdrant_service.initialize()
        except ImportError:
            logger.warning("QdrantService not available")

        self._initialized = True
        logger.info("DatabaseManager initialized with DI clients")

    async def cleanup(self) -> None:
        """Cleanup database resources."""
        if self._cache_manager:
            await self._cache_manager.close()
            self._cache_manager = None

        if self._qdrant_service:
            await self._qdrant_service.cleanup()
            self._qdrant_service = None

        self._qdrant_client = None
        self._redis_client = None
        self._initialized = False
        logger.info("DatabaseManager cleaned up")

    # Qdrant Operations
    async def get_collections(self) -> list[str]:
        """Get list of Qdrant collections.

        Returns:
            List of collection names

        Raises:
            APIError: If Qdrant client not available
        """
        if not self._qdrant_client:
            raise APIError("Qdrant client not available")

        try:
            collections = await self._qdrant_client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(
                f"Failed to get collections: {e}"
            )  # TODO: Convert f-string to logging format
            raise APIError(f"Failed to get collections: {e}") from e

    async def store_embeddings(
        self, collection_name: str, points: list[dict[str, Any]]
    ) -> bool:
        """Store embeddings in Qdrant collection.

        Args:
            collection_name: Name of the collection
            points: List of points to store

        Returns:
            True if successful

        Raises:
            APIError: If storage fails
        """
        if not self._qdrant_service:
            raise APIError("Qdrant service not available")

        try:
            await self._qdrant_service.upsert_points(collection_name, points)
            return True
        except Exception as e:
            logger.error(
                f"Failed to store embeddings: {e}"
            )  # TODO: Convert f-string to logging format
            raise APIError(f"Failed to store embeddings: {e}") from e

    async def search_similar(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in Qdrant.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results
            filter_conditions: Optional filter conditions

        Returns:
            List of search results

        Raises:
            APIError: If search fails
        """
        if not self._qdrant_service:
            raise APIError("Qdrant service not available")

        try:
            results = await self._qdrant_service.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                filter_conditions=filter_conditions,
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to search vectors: {e}"
            )  # TODO: Convert f-string to logging format
            raise APIError(f"Failed to search vectors: {e}") from e

    # Cache Operations
    async def cache_get(
        self, key: str, cache_type: CacheType = CacheType.CRAWL, default: Any = None
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
        except Exception as e:
            logger.warning(
                f"Cache get failed for {key}: {e}"
            )  # TODO: Convert f-string to logging format
            return default

    async def cache_set(
        self,
        key: str,
        value: Any,
        cache_type: CacheType = CacheType.CRAWL,
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
        except Exception as e:
            logger.warning(
                f"Cache set failed for {key}: {e}"
            )  # TODO: Convert f-string to logging format
            return False

    async def cache_delete(
        self, key: str, cache_type: CacheType = CacheType.CRAWL
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
        except Exception as e:
            logger.warning(
                f"Cache delete failed for {key}: {e}"
            )  # TODO: Convert f-string to logging format
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
            return True
        except Exception as e:
            logger.warning(
                f"Redis ping failed: {e}"
            )  # TODO: Convert f-string to logging format
            return False

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
            return True
        except Exception as e:
            logger.warning(
                f"Redis set failed for {key}: {e}"
            )  # TODO: Convert f-string to logging format
            return False

    async def redis_get(self, key: str) -> str | None:
        """Get value from Redis.

        Args:
            key: Redis key

        Returns:
            Value or None if not found
        """
        if not self._redis_client:
            return None

        try:
            return await self._redis_client.get(key)
        except Exception as e:
            logger.warning(
                f"Redis get failed for {key}: {e}"
            )  # TODO: Convert f-string to logging format
            return None

    # Status and Metrics
    async def get_status(self) -> dict[str, Any]:
        """Get database manager status.

        Returns:
            Status information for all components
        """
        status = {
            "initialized": self._initialized,
            "qdrant": {
                "available": self._qdrant_client is not None,
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

        # Check Qdrant status
        if self._qdrant_client:
            try:
                collections = await self.get_collections()
                status["qdrant"]["collections"] = collections
            except Exception:
                status["qdrant"]["available"] = False

        # Check Redis status
        if self._redis_client:
            status["redis"]["connected"] = await self.redis_ping()

        # Get cache stats
        if self._cache_manager:
            try:
                cache_stats = await self._cache_manager.get_stats()
                status["cache"]["stats"] = cache_stats
            except Exception:
                status["cache"]["available"] = False

        return status

    async def get_cache_manager(self) -> Optional["CacheManager"]:
        """Get cache manager instance.

        Returns:
            CacheManager instance or None
        """
        return self._cache_manager

    async def get_qdrant_service(self) -> Optional["QdrantService"]:
        """Get Qdrant service instance.

        Returns:
            QdrantService instance or None
        """
        return self._qdrant_service
