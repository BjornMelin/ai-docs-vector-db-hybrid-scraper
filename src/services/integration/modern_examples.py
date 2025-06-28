"""Integration examples for modern library implementations.

This module provides examples of how to use the modernized circuit breaker,
caching, and rate limiting implementations in your application code.
"""

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, Request

from src.config import Config
from src.services.cache.modern import ModernCacheManager
from src.services.circuit_breaker.modern import ModernCircuitBreakerManager
from src.services.middleware.rate_limiting import setup_rate_limiting
from src.services.migration.library_migration import LibraryMigrationManager


logger = logging.getLogger(__name__)


class ModernServiceExample:
    """Example service using modern library implementations."""

    def __init__(
        self,
        config: Config,
        redis_url: str = "redis://localhost:6379",
    ):
        """Initialize example service with modern implementations."""
        self.config = config
        self.redis_url = redis_url
        self.circuit_breaker_manager = ModernCircuitBreakerManager(redis_url, config)
        self.cache_manager = ModernCacheManager(redis_url, config=config)

    @property
    def embedding_service(self):
        """Get embedding service with circuit breaker protection."""
        return EmbeddingServiceWithProtection(
            self.circuit_breaker_manager,
            self.cache_manager,
        )

    @property
    def search_service(self):
        """Get search service with caching."""
        return SearchServiceWithCaching(self.cache_manager)

    async def close(self) -> None:
        """Clean up service resources."""
        await self.circuit_breaker_manager.close()
        await self.cache_manager.close()


class EmbeddingServiceWithProtection:
    """Example embedding service using modern circuit breaker and caching."""

    def __init__(
        self,
        circuit_breaker_manager: ModernCircuitBreakerManager,
        cache_manager: ModernCacheManager,
    ):
        """Initialize embedding service with protection."""
        self.circuit_breaker_manager = circuit_breaker_manager
        self.cache_manager = cache_manager

    @cache_manager.cache_embeddings(ttl=86400)  # Cache for 24 hours
    async def generate_embedding(
        self, text: str, model: str = "default"
    ) -> List[float]:
        """Generate embedding with caching and circuit breaker protection.

        Args:
            text: Text to embed
            model: Model name to use

        Returns:
            Embedding vector
        """
        # Circuit breaker protection for external API calls
        return await self.circuit_breaker_manager.protected_call(
            service_name=f"embedding_{model}",
            func=self._generate_embedding_impl,
            text=text,
            model=model,
        )

    async def _generate_embedding_impl(self, text: str, model: str) -> List[float]:
        """Internal implementation of embedding generation."""
        # Simulate external API call
        await asyncio.sleep(0.1)

        # Simulate potential failure
        if len(text) > 1000:
            raise ValueError("Text too long for embedding")

        # Return mock embedding
        return [0.1] * 1536

    async def generate_batch_embeddings(
        self, texts: List[str], model: str = "default"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with protection."""
        embeddings = []

        for text in texts:
            try:
                embedding = await self.generate_embedding(text, model)
                embeddings.append(embedding)
            except Exception as e:
                logger.exception(f"Failed to generate embedding for text: {e}")
                # Use fallback embedding or skip
                embeddings.append([0.0] * 1536)

        return embeddings


class SearchServiceWithCaching:
    """Example search service using modern caching."""

    def __init__(self, cache_manager: ModernCacheManager):
        """Initialize search service with caching."""
        self.cache_manager = cache_manager

    @cache_manager.cache_search_results(ttl=3600)  # Cache for 1 hour
    async def search_documents(
        self,
        query: str,
        filters: Dict[str, Any] | None = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search documents with caching.

        Args:
            query: Search query
            filters: Search filters
            limit: Maximum results to return

        Returns:
            Search results
        """
        return await self._search_documents_impl(query, filters or {}, limit)

    async def _search_documents_impl(
        self,
        query: str,
        filters: Dict[str, Any],
        limit: int,
    ) -> Dict[str, Any]:
        """Internal implementation of document search."""
        # Simulate search operation
        await asyncio.sleep(0.2)

        return {
            "query": query,
            "filters": filters,
            "results": [
                {"id": i, "title": f"Document {i}", "score": 0.9 - (i * 0.1)}
                for i in range(min(limit, 5))
            ],
            "total": min(limit, 5),
        }

    async def invalidate_search_cache(self, pattern: str = "*") -> bool:
        """Invalidate search cache entries matching pattern."""
        try:
            count = await self.cache_manager.invalidate_pattern(pattern)
            logger.info(f"Invalidated {count} search cache entries")
            return True
        except Exception as e:
            logger.exception(f"Failed to invalidate search cache: {e}")
            return False


def create_fastapi_app_with_modern_features(config: Config) -> FastAPI:
    """Create FastAPI app with modern rate limiting and middleware.

    Args:
        config: Application configuration

    Returns:
        FastAPI app with modern features enabled
    """
    app = FastAPI(
        title="AI Docs API with Modern Libraries",
        description="Example API using modern circuit breaker, caching, and rate limiting",
    )

    # Determine Redis URL
    redis_url = getattr(config.cache, "dragonfly_url", "redis://localhost:6379")

    # Set up rate limiting
    rate_limiter = setup_rate_limiting(app, redis_url, config)

    # Initialize services
    modern_service = ModernServiceExample(config, redis_url)

    # Store in app state for access in endpoints
    app.state.modern_service = modern_service
    app.state.rate_limiter = rate_limiter

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Initializing modern services...")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up services on shutdown."""
        await modern_service.close()
        logger.info("Modern services cleaned up")

    # Example endpoints with modern features

    @app.get("/embeddings")
    @rate_limiter.limit("10/minute")
    async def generate_embedding_endpoint(
        request: Request,
        text: str,
        model: str = "default",
    ):
        """Generate embedding with rate limiting and caching."""
        service = request.app.state.modern_service
        try:
            embedding = await service.embedding_service.generate_embedding(text, model)
            return {"embedding": embedding, "model": model}
        except Exception as e:
            logger.exception(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/search")
    @rate_limiter.limit("50/minute")
    async def search_documents_endpoint(
        request: Request,
        query: str,
        limit: int = 10,
    ):
        """Search documents with rate limiting and caching."""
        service = request.app.state.modern_service
        try:
            results = await service.search_service.search_documents(query, limit=limit)
            return results
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health/modern")
    async def modern_health_check(request: Request):
        """Health check for modern services."""
        service = request.app.state.modern_service

        # Check circuit breaker status
        cb_status = await service.circuit_breaker_manager.get_all_statuses()

        # Check cache status
        cache_stats = await service.cache_manager.get_stats()

        # Check rate limiter status
        rate_limiter_stats = await service.rate_limiter.get_stats()

        return {
            "status": "healthy",
            "circuit_breakers": cb_status,
            "cache": cache_stats,
            "rate_limiter": rate_limiter_stats,
        }

    @app.post("/admin/cache/clear")
    @rate_limiter.limit("5/minute")
    async def clear_cache_endpoint(request: Request):
        """Clear cache (admin endpoint)."""
        service = request.app.state.modern_service
        try:
            success = await service.cache_manager.clear()
            return {"success": success, "message": "Cache cleared"}
        except Exception as e:
            logger.exception(f"Cache clear failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/circuit-breaker/{service_name}/reset")
    @rate_limiter.limit("5/minute")
    async def reset_circuit_breaker_endpoint(
        request: Request,
        service_name: str,
    ):
        """Reset circuit breaker (admin endpoint)."""
        service = request.app.state.modern_service
        try:
            success = await service.circuit_breaker_manager.reset_breaker(service_name)
            return {"success": success, "service": service_name}
        except Exception as e:
            logger.exception(f"Circuit breaker reset failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


async def example_usage_patterns():
    """Example usage patterns for modern implementations."""

    # Example 1: Direct usage of modern cache manager
    cache_manager = ModernCacheManager("redis://localhost:6379")

    # Declarative caching with decorator
    @cache_manager.cache_embeddings(ttl=3600)
    async def cached_function(text: str) -> List[float]:
        # Expensive operation
        await asyncio.sleep(1)
        return [0.1] * 1536

    # Example 2: Circuit breaker usage
    cb_manager = ModernCircuitBreakerManager("redis://localhost:6379")

    async def protected_external_call():
        """Example of protected external API call."""
        return await cb_manager.protected_call(
            service_name="external_api",
            func=simulate_external_api_call,
            param1="value1",
        )

    async def simulate_external_api_call(param1: str):
        """Simulate external API call that might fail."""
        await asyncio.sleep(0.1)
        return {"result": f"Success with {param1}"}

    # Example 3: Migration manager usage
    from src.services.migration.library_migration import (
        MigrationMode,
        create_migration_manager,
    )

    config = Config()  # Your config instance
    migration_manager = create_migration_manager(
        config=config,
        mode=MigrationMode.GRADUAL,
        redis_url="redis://localhost:6379",
    )

    await migration_manager.initialize()

    # Get services through migration manager
    modern_cache = await migration_manager.get_cache_manager()
    modern_cb = await migration_manager.get_circuit_breaker("my_service")

    # Check migration status
    status = await migration_manager.get_migration_status()
    print(f"Migration status: {status}")

    # Cleanup
    await cache_manager.close()
    await cb_manager.close()
    await migration_manager.cleanup()


if __name__ == "__main__":
    # Run example usage patterns
    asyncio.run(example_usage_patterns())
