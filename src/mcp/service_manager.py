"""Unified Service Manager for MCP Server."""

import logging
from typing import Any

from ..config import get_config
from ..infrastructure.client_manager import ClientManager
from ..services.base import BaseService
from ..services.cache.manager import CacheManager
from ..services.cache.manager import CacheType
from ..services.crawling.manager import CrawlManager
from ..services.deployment import ABTestingManager
from ..services.deployment import BlueGreenDeployment
from ..services.deployment import CanaryDeployment
from ..services.embeddings.manager import EmbeddingManager
from ..services.hyde.config import HyDEMetricsConfig
from ..services.hyde.config import HyDEPromptConfig
from ..services.hyde.engine import HyDEQueryEngine
from ..services.project_storage import ProjectStorage
from ..services.qdrant_alias_manager import QdrantAliasManager
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
        self.alias_manager: QdrantAliasManager | None = None
        self.blue_green: BlueGreenDeployment | None = None
        self.ab_testing: ABTestingManager | None = None
        self.canary: CanaryDeployment | None = None
        self.cache_manager: CacheManager | None = None
        self.project_storage: ProjectStorage | None = None
        self.rate_limiter: RateLimitManager | None = None
        self.hyde_engine: HyDEQueryEngine | None = None
        self.client_manager: ClientManager | None = None
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

            # Initialize client manager
            self.client_manager = ClientManager()
            await self.client_manager.initialize()

            # Initialize each service
            await self.embedding_manager.initialize()
            await self.crawl_manager.initialize()
            await self.qdrant_service.initialize()
            await self.cache_manager.initialize()
            await self.project_storage.initialize()

            # Initialize alias manager and deployment services
            self.alias_manager = QdrantAliasManager(
                config=self.config, client=self.qdrant_service._client
            )
            await self.alias_manager.initialize()

            self.blue_green = BlueGreenDeployment(
                config=self.config,
                qdrant_service=self.qdrant_service,
                alias_manager=self.alias_manager,
                embedding_manager=self.embedding_manager,
            )
            await self.blue_green.initialize()

            self.ab_testing = ABTestingManager(
                config=self.config, qdrant_service=self.qdrant_service
            )
            await self.ab_testing.initialize()

            self.canary = CanaryDeployment(
                config=self.config,
                alias_manager=self.alias_manager,
                qdrant_service=self.qdrant_service,
            )
            await self.canary.initialize()

            # Initialize HyDE engine with all dependencies
            if self.config.hyde.enable_hyde:
                # Convert config to component configs
                prompt_config = HyDEPromptConfig()
                metrics_config = HyDEMetricsConfig(
                    ab_testing_enabled=self.config.hyde.ab_testing_enabled,
                    control_group_percentage=self.config.hyde.control_group_percentage,
                )

                self.hyde_engine = HyDEQueryEngine(
                    config=self.config.hyde,
                    prompt_config=prompt_config,
                    metrics_config=metrics_config,
                    embedding_manager=self.embedding_manager,
                    qdrant_service=self.qdrant_service,
                    cache_manager=self.cache_manager.dragonfly_cache
                    if hasattr(self.cache_manager, "dragonfly_cache")
                    else self.cache_manager,
                    llm_client=self.client_manager.openai_client,
                )
                await self.hyde_engine.initialize()
                logger.info("HyDE engine initialized")

            # Load projects from storage
            self.projects = await self.project_storage.load_projects()

            self._initialized = True
            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    async def cleanup(self):
        """Cleanup all services"""
        if self.hyde_engine:
            await self.hyde_engine.cleanup()
        if self.canary:
            await self.canary.cleanup()
        if self.ab_testing:
            await self.ab_testing.cleanup()
        if self.blue_green:
            await self.blue_green.cleanup()
        if self.alias_manager:
            await self.alias_manager.cleanup()
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
        if self.client_manager:
            await self.client_manager.cleanup()

        self._initialized = False
        logger.info("All services cleaned up")
