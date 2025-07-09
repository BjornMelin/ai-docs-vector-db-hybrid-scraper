"""
Example implementation of production-ready connection pool manager.

This demonstrates KISS/DRY/YAGNI principles while maintaining enterprise standards.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import redis.asyncio as redis
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams


logger = logging.getLogger(__name__)


class ConnectionHealth(BaseModel):
    """Health status for a connection."""

    is_healthy: bool = Field(default=True)
    last_check: float = Field(default_factory=time.time)
    consecutive_failures: int = Field(default=0)
    latency_ms: float | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ConnectionPoolManager:
    """
    Production-ready connection pool manager with health checks.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker pattern
    - Connection validation
    - Performance monitoring
    """

    def __init__(
        self,
        qdrant_config: dict,
        redis_config: dict,
        max_retries: int = 3,
        health_check_interval: int = 30,
    ):
        """Initialize connection pool manager."""
        self.qdrant_config = qdrant_config
        self.redis_config = redis_config
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval

        # Connection instances
        self._qdrant_client: AsyncQdrantClient | None = None
        self._redis_pool: redis.ConnectionPool | None = None
        self._redis_client: redis.Redis | None = None

        # Health tracking
        self._qdrant_health = ConnectionHealth()
        self._redis_health = ConnectionHealth()

        # Background tasks
        self._health_check_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all connections with retry logic."""
        if self._initialized:
            return

        logger.info("Initializing connection pool manager")

        # Initialize Qdrant
        await self._initialize_qdrant()

        # Initialize Redis
        await self._initialize_redis()

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._initialized = True
        logger.info("Connection pool manager initialized successfully")

    async def _initialize_qdrant(self) -> None:
        """Initialize Qdrant client with retry."""
        for attempt in range(self.max_retries):
            try:
                self._qdrant_client = AsyncQdrantClient(**self.qdrant_config)

                # Validate connection
                await self._qdrant_client.get_collections()

                self._qdrant_health.is_healthy = True
                self._qdrant_health.consecutive_failures = 0
                logger.info("Qdrant connection established")
                return

            except Exception as e:
                self._qdrant_health.consecutive_failures += 1
                self._qdrant_health.error_message = str(e)

                if attempt < self.max_retries - 1:
                    delay = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Qdrant connection failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        f"Failed to connect to Qdrant after {self.max_retries} attempts"
                    )
                    self._qdrant_health.is_healthy = False
                    raise

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection pool with retry."""
        for attempt in range(self.max_retries):
            try:
                # Create connection pool
                self._redis_pool = redis.ConnectionPool(
                    **self.redis_config,
                    max_connections=50,
                    decode_responses=True,
                )

                # Create client from pool
                self._redis_client = redis.Redis(connection_pool=self._redis_pool)

                # Validate connection
                await self._redis_client.ping()

                self._redis_health.is_healthy = True
                self._redis_health.consecutive_failures = 0
                logger.info("Redis connection established")
                return

            except Exception as e:
                self._redis_health.consecutive_failures += 1
                self._redis_health.error_message = str(e)

                if attempt < self.max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        f"Redis connection failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        f"Failed to connect to Redis after {self.max_retries} attempts"
                    )
                    self._redis_health.is_healthy = False
                    raise

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check Qdrant health
                await self._check_qdrant_health()

                # Check Redis health
                await self._check_redis_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health check error: {e}")

    async def _check_qdrant_health(self) -> None:
        """Check Qdrant connection health."""
        if not self._qdrant_client:
            return

        start_time = time.time()
        try:
            # Simple health check query
            await self._qdrant_client.get_collections()

            latency_ms = (time.time() - start_time) * 1000
            self._qdrant_health.latency_ms = latency_ms
            self._qdrant_health.is_healthy = True
            self._qdrant_health.consecutive_failures = 0
            self._qdrant_health.last_check = time.time()

            # Log warning if latency is high
            if latency_ms > 100:
                logger.warning(f"Qdrant latency high: {latency_ms:.2f}ms")

        except Exception as e:
            self._qdrant_health.consecutive_failures += 1
            self._qdrant_health.error_message = str(e)
            self._qdrant_health.last_check = time.time()

            # Circuit breaker logic
            if self._qdrant_health.consecutive_failures >= 3:
                self._qdrant_health.is_healthy = False
                logger.exception(
                    f"Qdrant marked unhealthy after {self._qdrant_health.consecutive_failures} failures"
                )

    async def _check_redis_health(self) -> None:
        """Check Redis connection health."""
        if not self._redis_client:
            return

        start_time = time.time()
        try:
            # Simple health check
            await self._redis_client.ping()

            latency_ms = (time.time() - start_time) * 1000
            self._redis_health.latency_ms = latency_ms
            self._redis_health.is_healthy = True
            self._redis_health.consecutive_failures = 0
            self._redis_health.last_check = time.time()

            # Log warning if latency is high
            if latency_ms > 50:
                logger.warning(f"Redis latency high: {latency_ms:.2f}ms")

        except Exception as e:
            self._redis_health.consecutive_failures += 1
            self._redis_health.error_message = str(e)
            self._redis_health.last_check = time.time()

            # Circuit breaker logic
            if self._redis_health.consecutive_failures >= 3:
                self._redis_health.is_healthy = False
                logger.exception(
                    f"Redis marked unhealthy after {self._redis_health.consecutive_failures} failures"
                )

    @asynccontextmanager
    async def get_qdrant_client(self) -> AsyncIterator[AsyncQdrantClient]:
        """Get Qdrant client with health check."""
        if not self._qdrant_health.is_healthy:
            msg = "Qdrant connection is unhealthy"
            raise ConnectionError(msg)

        if not self._qdrant_client:
            msg = "Qdrant client not initialized"
            raise RuntimeError(msg)

        yield self._qdrant_client

    @asynccontextmanager
    async def get_redis_client(self) -> AsyncIterator[redis.Redis]:
        """Get Redis client with health check."""
        if not self._redis_health.is_healthy:
            msg = "Redis connection is unhealthy"
            raise ConnectionError(msg)

        if not self._redis_client:
            msg = "Redis client not initialized"
            raise RuntimeError(msg)

        yield self._redis_client

    async def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Create Qdrant collection if it doesn't exist."""
        async with self.get_qdrant_client() as client:
            try:
                # Check if collection exists
                collections = await client.get_collections()
                if collection_name in [c.name for c in collections.collections]:
                    return False

                # Create collection
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info(f"Created collection: {collection_name}")
                return True

            except Exception as e:
                logger.exception(f"Failed to create collection {collection_name}: {e}")
                raise

    def get_health_status(self) -> dict:
        """Get comprehensive health status."""
        return {
            "qdrant": {
                "healthy": self._qdrant_health.is_healthy,
                "latency_ms": self._qdrant_health.latency_ms,
                "consecutive_failures": self._qdrant_health.consecutive_failures,
                "last_check": self._qdrant_health.last_check,
                "error": self._qdrant_health.error_message,
            },
            "redis": {
                "healthy": self._redis_health.is_healthy,
                "latency_ms": self._redis_health.latency_ms,
                "consecutive_failures": self._redis_health.consecutive_failures,
                "last_check": self._redis_health.last_check,
                "error": self._redis_health.error_message,
            },
            "initialized": self._initialized,
        }

    async def cleanup(self) -> None:
        """Cleanup all connections and resources."""
        logger.info("Cleaning up connection pool manager")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close Redis connections
        if self._redis_client:
            await self._redis_client.close()

        if self._redis_pool:
            await self._redis_pool.disconnect()

        # Close Qdrant client
        if self._qdrant_client:
            await self._qdrant_client.close()

        self._initialized = False
        logger.info("Connection pool manager cleaned up")


# Example usage
async def main():
    """Example usage of connection pool manager."""
    # Configuration
    qdrant_config = {
        "host": "localhost",
        "port": 6333,
        "timeout": 30,
    }

    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "socket_timeout": 5,
        "socket_keepalive": True,
    }

    # Initialize manager
    manager = ConnectionPoolManager(qdrant_config, redis_config)

    try:
        await manager.initialize()

        # Create collection
        await manager.create_collection_if_not_exists("documents", vector_size=1536)

        # Use Qdrant client
        async with manager.get_qdrant_client() as client:
            collections = await client.get_collections()
            print(f"Collections: {[c.name for c in collections.collections]}")

        # Use Redis client
        async with manager.get_redis_client() as client:
            await client.set("test_key", "test_value", ex=60)
            value = await client.get("test_key")
            print(f"Redis value: {value}")

        # Check health
        health = manager.get_health_status()
        print(f"Health status: {health}")

    finally:
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
