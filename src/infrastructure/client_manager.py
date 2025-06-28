"""Simplified client coordination layer using focused service managers."""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional

from dependency_injector.wiring import Provide, inject

from src.infrastructure.clients import (
    FirecrawlClientProvider,
    HTTPClientProvider,
    OpenAIClientProvider,
    QdrantClientProvider,
    RedisClientProvider,
)
from src.infrastructure.container import ApplicationContainer, get_container
from src.services.errors import APIError


if TYPE_CHECKING:
    from src.services.managers import (
        CrawlingManagerService,
        DatabaseManager,
        EmbeddingManagerService,
        MonitoringManager,
    )

logger = logging.getLogger(__name__)


class ClientManager:
    """Simplified client coordination layer using focused service managers.

    Coordinates focused service managers instead of handling everything directly.
    Provides backward compatibility while delegating to specialized managers.
    """

    _instance: "ClientManager | None" = None
    _lock = asyncio.Lock()
    _init_lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton instance with thread safety."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize client manager."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Provider compatibility layer
        self._providers: dict[str, Any] = {}

        # Focused service managers
        self._database_manager: DatabaseManager | None = None
        self._embedding_manager: EmbeddingManagerService | None = None
        self._crawling_manager: CrawlingManagerService | None = None
        self._monitoring_manager: MonitoringManager | None = None
        self._parallel_processing_system: Any | None = None

        self._initialized = False

    @inject
    def initialize_providers(
        self,
        openai_provider: OpenAIClientProvider = Provide[
            ApplicationContainer.openai_provider
        ],
        qdrant_provider: QdrantClientProvider = Provide[
            ApplicationContainer.qdrant_provider
        ],
        redis_provider: RedisClientProvider = Provide[
            ApplicationContainer.redis_provider
        ],
        firecrawl_provider: FirecrawlClientProvider = Provide[
            ApplicationContainer.firecrawl_provider
        ],
        http_provider: HTTPClientProvider = Provide[ApplicationContainer.http_provider],
    ) -> None:
        """Initialize client providers using dependency injection."""
        self._providers = {
            "openai": openai_provider,
            "qdrant": qdrant_provider,
            "redis": redis_provider,
            "firecrawl": firecrawl_provider,
            "http": http_provider,
        }

    async def initialize(self) -> None:
        """Initialize client manager and focused service managers."""
        if self._initialized:
            return
        container = get_container()
        if container:
            container.wire(modules=[__name__])
            self.initialize_providers()
        await self._initialize_service_managers()
        self._initialized = True
        logger.info("ClientManager initialized with focused service managers")

    async def _initialize_service_managers(self) -> None:
        """Initialize all focused service managers."""
        try:
            # Import focused managers
            from src.services.managers import (
                CrawlingManagerService,
                DatabaseManager,
                EmbeddingManagerService,
                MonitoringManager,
            )

            # Initialize database manager
            self._database_manager = DatabaseManager()
            await self._database_manager.initialize()
            logger.info("DatabaseManager initialized")

            # Initialize embedding manager
            self._embedding_manager = EmbeddingManagerService()
            await self._embedding_manager.initialize()
            logger.info("EmbeddingManager service initialized")

            # Initialize crawling manager
            self._crawling_manager = CrawlingManagerService()
            await self._crawling_manager.initialize()
            logger.info("CrawlingManager service initialized")

            # Initialize monitoring manager
            self._monitoring_manager = MonitoringManager()
            await self._monitoring_manager.initialize()
            logger.info("MonitoringManager initialized")

            # Initialize parallel processing system
            await self._initialize_parallel_processing_system()

            # Register health checks with monitoring manager
            self._register_health_checks()

        except Exception as e:
            logger.error(f"Failed to initialize service managers: {e}")  # TODO: Convert f-string to logging format
            raise

    async def _initialize_parallel_processing_system(self) -> None:
        """Initialize the parallel processing system using dependency injection."""
        try:
            container = get_container()
            if container and self._embedding_manager:
                # Get the parallel processing system from the container
                self._parallel_processing_system = container.parallel_processing_system(
                    embedding_manager=self._embedding_manager
                )
                logger.info("Parallel processing system initialized")
            else:
                logger.warning(
                    "Cannot initialize parallel processing system: container or embedding manager not available"
                )
        except Exception as e:
            logger.error(f"Failed to initialize parallel processing system: {e}")  # TODO: Convert f-string to logging format
            # Continue without parallel processing
            self._parallel_processing_system = None

    def _register_health_checks(self) -> None:
        """Register health checks for all services with monitoring manager."""
        if not self._monitoring_manager:
            return
        for name, manager in [
            ("database", self._database_manager),
            ("embedding", self._embedding_manager),
            ("crawling", self._crawling_manager),
        ]:
            if manager:
                self._monitoring_manager.register_health_check(
                    name, lambda m=manager: self._check_manager_health(m)
                )

    async def _check_manager_health(self, manager) -> bool:
        """Check manager health."""
        if not manager:
            return False
        try:
            status = await manager.get_status()
            return status.get("initialized", False)
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup all service managers and resources."""

        # Cleanup service managers
        if self._monitoring_manager:
            await self._monitoring_manager.cleanup()
            self._monitoring_manager = None

        if self._crawling_manager:
            await self._crawling_manager.cleanup()
            self._crawling_manager = None

        if self._embedding_manager:
            await self._embedding_manager.cleanup()
            self._embedding_manager = None

        if self._database_manager:
            await self._database_manager.cleanup()
            self._database_manager = None

        self._providers.clear()
        self._initialized = False
        logger.info("ClientManager cleaned up")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._init_lock:
            cls._instance = None

    async def get_openai_client(self):
        provider = self._providers.get("openai")
        return provider.client if provider else None

    async def get_qdrant_client(self):
        provider = self._providers.get("qdrant")
        if not provider:
            raise APIError("Qdrant client provider not available")
        return provider.client

    async def get_redis_client(self):
        provider = self._providers.get("redis")
        if not provider:
            raise APIError("Redis client provider not available")
        return provider.client

    async def get_firecrawl_client(self):
        provider = self._providers.get("firecrawl")
        return provider.client if provider else None

    async def get_http_client(self):
        provider = self._providers.get("http")
        if not provider:
            raise APIError("HTTP client provider not available")
        return provider.client

    async def get_database_manager(self) -> Optional["DatabaseManager"]:
        return self._database_manager

    async def get_embedding_manager(self) -> Optional["EmbeddingManagerService"]:
        return self._embedding_manager

    async def get_crawling_manager(self) -> Optional["CrawlingManagerService"]:
        return self._crawling_manager

    async def get_monitoring_manager(self) -> Optional["MonitoringManager"]:
        return self._monitoring_manager

    # Service Manager Access Methods

    async def get_database_manager(self) -> Optional["DatabaseManager"]:
        return self._database_manager

    async def get_embedding_manager(self) -> Optional["EmbeddingManagerService"]:
        return self._embedding_manager

    async def get_crawling_manager(self) -> Optional["CrawlingManagerService"]:
        return self._crawling_manager

    async def get_monitoring_manager(self) -> Optional["MonitoringManager"]:
        return self._monitoring_manager

    async def get_cache_manager(self):
        if not self._database_manager:
            raise APIError("Database manager not available")
        return await self._database_manager.get_cache_manager()

    async def get_qdrant_service(self):
        if not self._database_manager:
            raise APIError("Database manager not available")
        return await self._database_manager.get_qdrant_service()

    async def get_crawl_manager(self):
        if not self._crawling_manager:
            raise APIError("Crawling manager not available")
        return self._crawling_manager.get_core_manager()

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        if not self._monitoring_manager:
            return {}
        return await self._monitoring_manager.get_health_status()

    async def get_overall_health(self) -> dict[str, Any]:
        if not self._monitoring_manager:
            return {"overall_healthy": False, "error": "Monitoring not available"}
        return await self._monitoring_manager.get_overall_health()

    async def get_service_status(self) -> dict[str, Any]:
        status = {
            "initialized": self._initialized,
            "database": None,
            "embedding": None,
            "crawling": None,
            "monitoring": None,
            "parallel_processing": None,
        }
        for name, manager in [
            ("database", self._database_manager),
            ("embedding", self._embedding_manager),
            ("crawling", self._crawling_manager),
            ("monitoring", self._monitoring_manager),
        ]:
            if manager:
                try:
                    status[name] = await manager.get_status()
                except Exception as e:
                    status[name] = {"error": str(e)}
        return status

    @asynccontextmanager
    async def managed_client(self, client_type: str):
        getters = {
            "qdrant": self.get_qdrant_client,
            "openai": self.get_openai_client,
            "firecrawl": self.get_firecrawl_client,
            "redis": self.get_redis_client,
            "http": self.get_http_client,
            "database": self.get_database_manager,
            "embedding": self.get_embedding_manager,
            "crawling": self.get_crawling_manager,
            "monitoring": self.get_monitoring_manager,
            "parallel_processing": self.get_parallel_processing_system,
        }
        if client_type not in getters:
            raise ValueError(f"Unknown client type: {client_type}")
        try:
            yield await getters[client_type]()
        except Exception:
            logger.exception(f"Error using {client_type} client")
            raise

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False