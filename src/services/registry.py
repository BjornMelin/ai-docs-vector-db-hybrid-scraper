"""Service registry providing centralized lifecycle management."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import Settings, get_settings
from src.config.models import CacheType
from src.infrastructure.client_manager import ClientManager
from src.services.cache.manager import CacheManager
from src.services.circuit_breaker import CircuitBreakerManager
from src.services.content_intelligence.service import ContentIntelligenceService
from src.services.core.project_storage import ProjectStorage
from src.services.managers.crawling_manager import CrawlingManager
from src.services.managers.database_manager import DatabaseManager
from src.services.managers.embedding_manager import EmbeddingManager


if TYPE_CHECKING:
    from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceRegistry:  # pylint: disable=too-many-instance-attributes
    """Collects initialized infrastructure services for reuse."""

    config: Settings
    client_manager: ClientManager
    circuit_breaker_manager: CircuitBreakerManager
    cache_manager: CacheManager
    database_manager: DatabaseManager
    embedding_manager: EmbeddingManager
    vector_service: VectorStoreService
    crawl_manager: CrawlingManager
    content_intelligence: ContentIntelligenceService
    project_storage: ProjectStorage

    @classmethod
    async def build(cls) -> ServiceRegistry:
        """Create and initialize the service registry."""

        config = get_settings()

        client_manager = ClientManager.from_unified_config()
        await client_manager.initialize()

        circuit_breaker_manager = CircuitBreakerManager(
            redis_url=config.cache.redis_url,
            config=config,
        )

        try:
            vector_service = await client_manager.get_vector_store_service()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to initialize vector service") from exc

        cache_config = config.cache
        cache_manager = CacheManager(
            dragonfly_url=cache_config.redis_url,
            enable_local_cache=cache_config.enable_local_cache,
            enable_distributed_cache=cache_config.enable_redis_cache,
            local_max_size=cache_config.local_max_size,
            local_max_memory_mb=float(cache_config.local_max_memory_mb),
            distributed_ttl_seconds={
                CacheType.REDIS: max(
                    cache_config.ttl_embeddings,
                    cache_config.ttl_search_results,
                ),
                CacheType.EMBEDDINGS: cache_config.ttl_embeddings,
                CacheType.SEARCH: cache_config.ttl_search_results,
                CacheType.CRAWL: cache_config.ttl_crawl,
                CacheType.HYBRID: max(
                    cache_config.ttl_embeddings,
                    cache_config.ttl_search_results,
                ),
            },
            local_cache_path=config.cache_dir / "services",
            memory_pressure_threshold=cache_config.memory_pressure_threshold,
        )

        try:
            embedding_manager = EmbeddingManager()
            await embedding_manager.initialize(
                config=config,
                client_manager=client_manager,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to initialize embedding manager") from exc

        try:
            database_manager = DatabaseManager()
            await database_manager.initialize(
                qdrant_client=await client_manager.get_qdrant_client(),
                redis_client=await client_manager.get_redis_client(),
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to initialize database manager") from exc

        try:
            crawl_manager = CrawlingManager()
            await crawl_manager.initialize(config=config)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to initialize crawl manager") from exc

        try:
            content_intelligence = ContentIntelligenceService(
                config=config,
                embedding_manager=embedding_manager,
                cache_manager=cache_manager,
            )
            await content_intelligence.initialize()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Failed to initialize content intelligence service"
            ) from exc

        try:
            data_dir = getattr(config, "data_dir", None)
            if data_dir is None:
                raise RuntimeError("Configuration missing data_dir for project storage")
            project_storage = ProjectStorage(data_dir=Path(data_dir))
            await project_storage.initialize()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to initialize project storage") from exc

        logger.info("Service registry initialized")

        return cls(
            config=config,
            client_manager=client_manager,
            circuit_breaker_manager=circuit_breaker_manager,
            cache_manager=cache_manager,
            database_manager=database_manager,
            embedding_manager=embedding_manager,
            vector_service=vector_service,
            crawl_manager=crawl_manager,
            content_intelligence=content_intelligence,
            project_storage=project_storage,
        )

    async def shutdown(self) -> None:
        """Release all managed services."""

        await self._safe_cleanup(
            self.content_intelligence.cleanup, "content intelligence"
        )
        await self._safe_cleanup(self.crawl_manager.cleanup, "crawl manager")
        await self._safe_cleanup(self.database_manager.cleanup, "database manager")
        await self._safe_cleanup(self.embedding_manager.cleanup, "embedding manager")
        await self._safe_cleanup(self.vector_service.cleanup, "vector service")
        await self._safe_cleanup(self.project_storage.cleanup, "project storage")
        if hasattr(self.cache_manager, "close"):
            await self._safe_cleanup(self.cache_manager.close, "cache manager")
        await self._safe_cleanup(self.client_manager.cleanup, "client manager")
        await self._safe_cleanup(
            self.circuit_breaker_manager.close, "circuit breaker manager"
        )
        logger.info("Service registry shutdown complete")

    @staticmethod
    async def _safe_cleanup(cleanup_callable, component_name: str) -> None:
        """Invoke cleanup callable and handle errors without aborting shutdown."""

        try:
            result = cleanup_callable()
            if inspect.isawaitable(result):
                await result
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to cleanup %s", component_name)


_registry_lock = asyncio.Lock()
_registry: ServiceRegistry | None = None


async def ensure_service_registry(force: bool = False) -> ServiceRegistry:  # pylint: disable=global-statement
    """Build the service registry if needed and return it."""

    global _registry  # pylint: disable=global-statement
    if _registry is not None and not force:
        return _registry

    async with _registry_lock:
        if _registry is None or force:
            if _registry is not None and force:
                await _registry.shutdown()
            _registry = await ServiceRegistry.build()
    return _registry


def get_service_registry() -> ServiceRegistry:
    """Return the current registry or raise if not initialized."""

    if _registry is None:
        msg = "Service registry not initialized"
        raise RuntimeError(msg)
    return _registry


async def shutdown_service_registry() -> None:  # pylint: disable=global-statement
    """Shutdown and clear the current registry."""

    global _registry  # pylint: disable=global-statement
    if _registry is None:
        return

    await _registry.shutdown()
    _registry = None


__all__ = [
    "ServiceRegistry",
    "ensure_service_registry",
    "get_service_registry",
    "shutdown_service_registry",
]
