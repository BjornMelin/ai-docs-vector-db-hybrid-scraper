"""Unified Service Manager for MCP Server."""

import logging
from typing import Any

from ..config import get_config
from ..services.base import BaseService
from ..services.cache.manager import CacheManager
from ..services.cache.manager import CacheType
from ..services.crawling.manager import CrawlManager
from ..services.embeddings.manager import EmbeddingManager
from ..services.project_storage import ProjectStorage
from ..services.qdrant_service import QdrantService
from ..services.rate_limiter import RateLimitManager

logger = logging.getLogger(__name__)


class UnifiedServiceManager(BaseService):
    """Manages all services for the unified MCP server"""

    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.embedding_manager: EmbeddingManager | None = None
        self.crawl_manager: CrawlManager | None = None
        self.qdrant_service: QdrantService | None = None
        self.cache_manager: CacheManager | None = None
        self.project_storage: ProjectStorage | None = None
        self.rate_limiter: RateLimitManager | None = None
        self.projects: dict[str, dict[str, Any]] = {}  # Keep for backward compatibility
        self._initialized = False

    async def initialize(self):
        """Initialize all services"""
        if self._initialized:
            return

        try:
            # Initialize rate limiter
            self.rate_limiter = RateLimitManager(self.config)
            logger.info("Rate limiter initialized")

            # Initialize services with unified configuration
            self.embedding_manager = EmbeddingManager(
                self.config, rate_limiter=self.rate_limiter
            )
            self.crawl_manager = CrawlManager(
                self.config, rate_limiter=self.rate_limiter
            )
            self.qdrant_service = QdrantService(self.config)

            # CacheManager uses specific parameters from unified config
            self.cache_manager = CacheManager(
                redis_url=self.config.cache.redis_url,
                enable_local_cache=self.config.cache.enable_local_cache,
                enable_redis_cache=self.config.cache.enable_redis_cache,
                local_max_size=self.config.cache.local_max_size,
                local_max_memory_mb=self.config.cache.local_max_memory_mb,
                redis_ttl_seconds={
                    CacheType.EMBEDDINGS: self.config.cache.ttl_embeddings,
                    CacheType.CRAWL_RESULTS: self.config.cache.ttl_crawl,
                    CacheType.QUERY_RESULTS: self.config.cache.ttl_queries,
                },
            )

            # Initialize project storage with data_dir from config
            self.project_storage = ProjectStorage(data_dir=self.config.data_dir)

            # Initialize each service
            await self.embedding_manager.initialize()
            await self.crawl_manager.initialize()
            await self.qdrant_service.initialize()
            await self.cache_manager.initialize()
            await self.project_storage.initialize()

            # Load projects from storage
            self.projects = await self.project_storage.load_projects()

            self._initialized = True
            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    async def cleanup(self):
        """Cleanup all services"""
        if self.embedding_manager:
            await self.embedding_manager.cleanup()
        if self.crawl_manager:
            await self.crawl_manager.cleanup()
        if self.qdrant_service:
            await self.qdrant_service.cleanup()
        if self.cache_manager:
            await self.cache_manager.cleanup()
        if self.project_storage:
            await self.project_storage.cleanup()
        if self.rate_limiter:
            await self.rate_limiter.cleanup()

        self._initialized = False
        logger.info("All services cleaned up")
