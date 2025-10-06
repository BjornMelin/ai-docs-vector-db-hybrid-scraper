"""Client coordination layer using function-based dependencies."""

import asyncio
import importlib
import logging
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from dependency_injector.wiring import Provide, inject

from src.config import Config, get_config
from src.infrastructure.clients import (
    FirecrawlClientProvider,
    HTTPClientProvider,
    OpenAIClientProvider,
    QdrantClientProvider,
    RedisClientProvider,
)
from src.infrastructure.container import ApplicationContainer, get_container
from src.services.browser.automation_router import AutomationRouter
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import APIError
from src.services.vector_db.service import VectorStoreService


# Import dependencies for health checks
if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    HealthStatusCallable = Callable[[], Awaitable[dict[str, dict[str, Any]]]]
    OverallHealthCallable = Callable[[], Awaitable[dict[str, Any]]]
else:
    HealthStatusCallable = Callable[[], Awaitable[dict[str, dict[str, Any]]]]
    OverallHealthCallable = Callable[[], Awaitable[dict[str, Any]]]

_dependencies_module = None
try:  # pragma: no cover - optional dependency
    _dependencies_module = importlib.import_module("src.services.dependencies")
except ImportError:
    _dependencies_module = None

if _dependencies_module is not None:
    _deps_get_health_status = getattr(_dependencies_module, "get_health_status", None)
    _deps_get_overall_health = getattr(_dependencies_module, "get_overall_health", None)
else:
    _deps_get_health_status = None
    _deps_get_overall_health = None

deps_get_health_status: HealthStatusCallable | None = (
    cast(HealthStatusCallable, _deps_get_health_status)
    if _deps_get_health_status is not None
    else None
)
deps_get_overall_health: OverallHealthCallable | None = (
    cast(OverallHealthCallable, _deps_get_overall_health)
    if _deps_get_overall_health is not None
    else None
)


logger = logging.getLogger(__name__)


class ClientManager:  # pylint: disable=too-many-public-methods
    """Client coordination layer using function-based dependencies."""

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
        self._parallel_processing_system: Any | None = None
        self._initialized = False
        self._vector_store_service: VectorStoreService | None = None
        self._config = get_config()
        self._automation_router: AutomationRouter | None = None

    @property
    def config(self) -> Config:
        """Return the lazily loaded application configuration."""

        return self._config

    @inject
    def initialize_providers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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

    @property
    def is_initialized(self) -> bool:
        """Check if the client manager is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize client manager with function-based dependencies."""
        if self._initialized:
            return
        container = get_container()
        if container:
            container.wire(modules=[__name__])
            self.initialize_providers()
        await self._initialize_parallel_processing_system()
        self._initialized = True
        logger.info("ClientManager initialized with function-based dependencies")

    async def _initialize_parallel_processing_system(self) -> None:
        """Initialize the parallel processing system using dependency injection."""
        try:
            container = get_container()
            if container:
                # Get the parallel processing system from the container
                # Note: embedding manager is now accessed via
                # function-based dependencies
                self._parallel_processing_system = (
                    container.parallel_processing_system()
                )
                logger.info("Parallel processing system initialized")
            else:
                logger.warning(
                    "Cannot initialize parallel processing system: "
                    "container not available"
                )
        except (ImportError, AttributeError, RuntimeError):
            logger.exception("Failed to initialize parallel processing system")
            # Continue without parallel processing
            self._parallel_processing_system = None

    async def cleanup(self) -> None:
        """Cleanup resources (function-based dependencies are stateless)."""
        if self._vector_store_service:
            await self._vector_store_service.cleanup()
            self._vector_store_service = None
        self._providers.clear()
        self._parallel_processing_system = None
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
            msg = "Qdrant client provider not available"
            raise APIError(msg)
        return provider.client

    async def get_vector_store_service(self) -> VectorStoreService:
        if self._vector_store_service:
            return self._vector_store_service

        model_name = getattr(self._config.fastembed, "model", "BAAI/bge-small-en-v1.5")
        provider = FastEmbedProvider(model_name=model_name)
        service = VectorStoreService(self._config, self, provider)
        await service.initialize()
        self._vector_store_service = service
        return service

    async def get_qdrant_service(self) -> VectorStoreService:
        """Backward-compatible alias for legacy callers."""

        logger.warning(
            "get_qdrant_service() is deprecated; use get_vector_store_service() instead"
        )
        return await self.get_vector_store_service()

    async def get_redis_client(self):
        provider = self._providers.get("redis")
        if not provider:
            msg = "Redis client provider not available"
            raise APIError(msg)
        return provider.client

    async def get_firecrawl_client(self):
        provider = self._providers.get("firecrawl")
        return provider.client if provider else None

    async def get_http_client(self):
        provider = self._providers.get("http")
        if not provider:
            msg = "HTTP client provider not available"
            raise APIError(msg)
        return provider.client

    # Function-based dependency access methods (backward compatibility)

    async def get_database_manager(self):
        """Backward compatibility: returns None since we use function-based deps."""
        logger.warning(
            "get_database_manager() deprecated - use function-based dependencies "
            "from src.services.dependencies"
        )

    async def get_embedding_manager(self):
        """Backward compatibility: returns None since we use function-based deps."""
        logger.warning(
            "get_embedding_manager() deprecated - use function-based dependencies "
            "from src.services.dependencies"
        )

    async def get_crawling_manager(self):
        """Backward compatibility: returns None since we use function-based deps."""
        logger.warning(
            "get_crawling_manager() deprecated - use function-based dependencies "
            "from src.services.dependencies"
        )

    async def get_monitoring_manager(self):
        """Backward compatibility: returns None since we use function-based deps."""
        logger.warning(
            "get_monitoring_manager() deprecated - use function-based dependencies "
            "from src.services.dependencies"
        )

    async def get_cache_manager(self):
        """Backward compatibility: use function-based dependencies."""
        logger.warning(
            "get_cache_manager() deprecated - use get_redis_client() or "
            "function-based cache dependencies"
        )
        return await self.get_redis_client()

    async def get_crawl_manager(self):
        """Backward compatibility: returns None since we use function-based deps."""
        logger.warning(
            "get_crawl_manager() deprecated - use function-based crawling dependencies"
        )

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status using function-based dependencies."""
        if deps_get_health_status:
            return await deps_get_health_status()
        logger.warning(
            "Health status monitoring not available - function-based dependency "
            "not found"
        )
        return {}

    async def get_overall_health(self) -> dict[str, Any]:
        """Get overall health using function-based dependencies."""
        if deps_get_overall_health:
            return await deps_get_overall_health()
        return {
            "overall_healthy": False,
            "error": "Health monitoring not available",
        }

    async def get_service_status(self) -> dict[str, Any]:
        """Get service status using function-based dependencies."""
        return {
            "initialized": self._initialized,
            "mode": "function_based_dependencies",
            "providers": list(self._providers.keys()),
            "parallel_processing": self._parallel_processing_system is not None,
            "note": "Using function-based dependencies instead of Manager classes",
        }

    async def get_parallel_processing_system(self):
        """Get parallel processing system instance."""
        return self._parallel_processing_system

    async def get_browser_automation_router(self):
        """Get browser automation router for intelligent scraping.

        Returns:
            AutomationRouter instance for intelligent browser automation
        """

        if self._automation_router is None:
            self._automation_router = AutomationRouter(self._config)
            await self._automation_router.initialize()
        return self._automation_router

    @asynccontextmanager
    async def managed_client(self, client_type: str):
        getters = {
            "qdrant": self.get_qdrant_client,
            "openai": self.get_openai_client,
            "firecrawl": self.get_firecrawl_client,
            "redis": self.get_redis_client,
            "http": self.get_http_client,
            "parallel_processing": self.get_parallel_processing_system,
        }
        if client_type not in getters:
            msg = (
                f"Unknown client type: {client_type}. Available: {list(getters.keys())}"
            )
            raise ValueError(msg)
        try:
            yield await getters[client_type]()
        except (ConnectionError, TimeoutError, APIError, ValueError, RuntimeError):
            logger.exception("Error using {client_type} client")
            raise

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    @classmethod
    def from_unified_config(cls) -> "ClientManager":
        """Create ClientManager instance from unified config.

        Used by function-based dependencies for singleton pattern.
        """
        return cls()
