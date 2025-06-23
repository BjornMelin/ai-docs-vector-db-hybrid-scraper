import typing
"""Centralized API client management with singleton pattern and health checks."""

import asyncio
import contextlib
import logging
import threading
import time
from contextlib import asynccontextmanager

# Import for type hints (avoid circular import by using TYPE_CHECKING)
from typing import TYPE_CHECKING
from typing import Any

import redis.asyncio as redis
from firecrawl import AsyncFirecrawlApp
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.config import Config
from src.infrastructure.shared import CircuitBreaker
from src.infrastructure.shared import ClientHealth
from src.infrastructure.shared import ClientState
from src.services.errors import APIError

if TYPE_CHECKING:
    pass

# Import for actual usage
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class ClientManager:
    """Centralized API client management with singleton pattern and health checks."""

    _instance: typing.Optional["ClientManager"] = None
    _lock = asyncio.Lock()
    _init_lock = threading.Lock()  # Thread-safe lock for singleton creation

    def __new__(cls, config: Config | None = None):
        """Ensure singleton instance with thread safety."""
        if cls._instance is None:
            with cls._init_lock:
                # Double-check pattern for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def from_unified_config(cls) -> "ClientManager":
        """Create ClientManager from unified configuration.

        This factory method loads configuration from environment variables
        and creates a properly configured ClientManager instance.
        """

        # Load the unified configuration
        from src.config import get_config

        unified_config = get_config()
        return cls(unified_config)

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance for testing purposes.

        This method is thread-safe and should only be used in tests.
        """
        with cls._init_lock:
            if (
                cls._instance is not None
                and hasattr(cls._instance, "_health_check_task")
                and cls._instance._health_check_task
                and not cls._instance._health_check_task.done()
            ):
                cls._instance._health_check_task.cancel()
            cls._instance = None

    def __init__(self, config: Config | None = None):
        """Initialize client manager.

        Args:
            config: Unified configuration instance
        """
        # Skip re-initialization if already initialized
        if hasattr(self, "_initialized") and self._initialized:
            if config and config != self.config:
                raise ValueError(
                    "ClientManager already initialized with different config. "
                    "Use cleanup() and reinitialize or create a new instance."
                )
            return

        if config is None:
            from src.config import get_config

            config = get_config()

        self.config = config
        self._clients: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._health: dict[str, ClientHealth] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._initialized = False
        self._health_check_task: asyncio.Task | None = None

        # Service instances (lazy-initialized)
        self._qdrant_service: Any = None
        self._embedding_manager: Any = None
        self._cache_manager: Any = None
        self._crawl_manager: Any = None
        self._hyde_engine: Any = None
        self._project_storage: Any = None
        # Enterprise deployment infrastructure components
        self._feature_flag_manager: Any = None
        self._ab_testing_manager: Any = None
        self._blue_green_deployment: Any = None
        self._canary_deployment: Any = None
        self._browser_automation_router: Any = None
        self._task_queue_manager: Any = None
        self._content_intelligence_service: Any = None
        self._database_manager: DatabaseManager | None = None
        self._advanced_search_orchestrator: Any = None
        self._rag_generator: Any = None
        self._service_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Initialize client manager and start health checks."""
        if self._initialized:
            return

        self._initialized = True
        # Start background health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("ClientManager initialized")

    async def cleanup(self) -> None:
        """Cleanup all clients and resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Clean up service instances
        service_names = [
            "_qdrant_service",
            "_embedding_manager",
            "_cache_manager",
            "_crawl_manager",
            "_hyde_engine",
            "_project_storage",
            "_rag_generator",
            "_feature_flag_manager",
            "_ab_testing_manager",
            "_blue_green_deployment",
            "_canary_deployment",
            "_content_intelligence_service",
            "_database_manager",
        ]
        for service_name in service_names:
            if hasattr(self, service_name):
                service = getattr(self, service_name)
                if service and hasattr(service, "cleanup"):
                    try:
                        await service.cleanup()
                        logger.info(f"Cleaned up {service_name}")
                    except Exception as e:
                        logger.exception(f"Error cleaning up {service_name}: {e}")
                setattr(self, service_name, None)

        # Close all clients
        for name, client in self._clients.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "aclose"):
                    await client.aclose()
                logger.info(f"Closed {name} client")
            except Exception as e:
                logger.exception(f"Error closing {name} client: {e}")

        self._clients.clear()
        self._health.clear()
        self._circuit_breakers.clear()

        # Reset service instances
        self._qdrant_service = None
        self._embedding_manager = None
        self._cache_manager = None
        self._crawl_manager = None
        self._hyde_engine = None
        # Removed deployment infrastructure cleanup
        self._browser_automation_router = None
        self._content_intelligence_service = None
        self._database_manager = None

        self._initialized = False
        logger.info("ClientManager cleaned up")

    async def get_qdrant_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client with health checks.

        Returns:
            AsyncQdrantClient instance

        Raises:
            APIError: If client is unhealthy or initialization fails
        """
        return await self._get_or_create_client(
            "qdrant",
            self._create_qdrant_client,
            self._check_qdrant_health,
        )

    async def get_openai_client(self) -> AsyncOpenAI | None:
        """Get or create OpenAI client.

        Returns:
            AsyncOpenAI instance or None if no API key

        Raises:
            APIError: If client is unhealthy
        """
        if not self.config.openai.api_key:
            return None

        return await self._get_or_create_client(
            "openai",
            self._create_openai_client,
            self._check_openai_health,
        )

    async def get_firecrawl_client(self) -> AsyncFirecrawlApp | None:
        """Get or create Firecrawl client.

        Returns:
            AsyncFirecrawlApp instance or None if no API key

        Raises:
            APIError: If client is unhealthy
        """
        if not self.config.firecrawl.api_key:
            return None

        return await self._get_or_create_client(
            "firecrawl",
            self._create_firecrawl_client,
            self._check_firecrawl_health,
        )

    async def get_redis_client(self) -> redis.Redis:
        """Get or create Redis client with connection pooling.

        Returns:
            Redis client instance

        Raises:
            APIError: If client is unhealthy
        """
        return await self._get_or_create_client(
            "redis",
            self._create_redis_client,
            self._check_redis_health,
        )

    # Service getter methods
    async def get_qdrant_service(self):
        """Get or create QdrantService instance."""
        if self._qdrant_service is None:
            if "qdrant_service" not in self._service_locks:
                self._service_locks["qdrant_service"] = asyncio.Lock()

            async with self._service_locks["qdrant_service"]:
                if self._qdrant_service is None:
                    from src.services.vector_db.service import QdrantService

                    self._qdrant_service = QdrantService(
                        self.config, client_manager=self
                    )
                    await self._qdrant_service.initialize()
                    logger.info("Initialized QdrantService")

        return self._qdrant_service

    async def get_embedding_manager(self):
        """Get or create EmbeddingManager instance."""
        if self._embedding_manager is None:
            if "embedding_manager" not in self._service_locks:
                self._service_locks["embedding_manager"] = asyncio.Lock()

            async with self._service_locks["embedding_manager"]:
                if self._embedding_manager is None:
                    from src.services.embeddings.manager import EmbeddingManager

                    # Pass ClientManager to EmbeddingManager
                    self._embedding_manager = EmbeddingManager(
                        config=self.config,
                        client_manager=self,
                    )
                    await self._embedding_manager.initialize()
                    logger.info("Initialized EmbeddingManager")

        return self._embedding_manager

    async def get_cache_manager(self):
        """Get or create CacheManager instance."""
        if self._cache_manager is None:
            if "cache_manager" not in self._service_locks:
                self._service_locks["cache_manager"] = asyncio.Lock()

            async with self._service_locks["cache_manager"]:
                if self._cache_manager is None:
                    from src.services.cache.manager import CacheManager

                    self._cache_manager = CacheManager(
                        dragonfly_url=self.config.cache.dragonfly_url,
                        enable_local_cache=self.config.cache.enable_local_cache,
                        enable_distributed_cache=self.config.cache.enable_dragonfly_cache,
                        local_max_size=self.config.cache.local_max_size,
                        local_max_memory_mb=self.config.cache.local_max_memory_mb,
                        distributed_ttl_seconds=self.config.cache.cache_ttl_seconds,
                    )
                    await self._cache_manager.initialize()
                    logger.info("Initialized CacheManager")

        return self._cache_manager

    async def get_crawl_manager(self):
        """Get or create CrawlManager instance."""
        if self._crawl_manager is None:
            if "crawl_manager" not in self._service_locks:
                self._service_locks["crawl_manager"] = asyncio.Lock()

            async with self._service_locks["crawl_manager"]:
                if self._crawl_manager is None:
                    from src.services.crawling.manager import CrawlManager

                    # CrawlManager expects rate_limiter but we'll pass None for now
                    self._crawl_manager = CrawlManager(
                        config=self.config,
                        rate_limiter=None,
                    )
                    await self._crawl_manager.initialize()
                    logger.info("Initialized CrawlManager")

        return self._crawl_manager

    async def get_hyde_engine(self):
        """Get or create HyDEEngine instance."""
        if self._hyde_engine is None:
            if "hyde_engine" not in self._service_locks:
                self._service_locks["hyde_engine"] = asyncio.Lock()

            async with self._service_locks["hyde_engine"]:
                if self._hyde_engine is None:
                    from src.services.hyde.config import HyDEConfig
                    from src.services.hyde.config import HyDEMetricsConfig
                    from src.services.hyde.config import HyDEPromptConfig
                    from src.services.hyde.engine import HyDEQueryEngine

                    # Get dependencies
                    embedding_manager = await self.get_embedding_manager()
                    qdrant_service = await self.get_qdrant_service()
                    cache_manager = await self.get_cache_manager()
                    openai_client = await self.get_openai_client()

                    # Create HyDE configurations from UnifiedConfig
                    hyde_config = HyDEConfig.from_unified_config(self.config.hyde)
                    prompt_config = HyDEPromptConfig()
                    metrics_config = HyDEMetricsConfig()

                    self._hyde_engine = HyDEQueryEngine(
                        config=hyde_config,
                        prompt_config=prompt_config,
                        metrics_config=metrics_config,
                        embedding_manager=embedding_manager,
                        qdrant_service=qdrant_service,
                        cache_manager=cache_manager,
                        llm_client=openai_client,
                    )
                    await self._hyde_engine.initialize()
                    logger.info("Initialized HyDEQueryEngine")

        return self._hyde_engine

    async def get_project_storage(self):
        """Get or create ProjectStorage instance."""
        if self._project_storage is None:
            if "project_storage" not in self._service_locks:
                self._service_locks["project_storage"] = asyncio.Lock()

            async with self._service_locks["project_storage"]:
                if self._project_storage is None:
                    from src.services.core.project_storage import ProjectStorage

                    self._project_storage = ProjectStorage(
                        data_dir=self.config.data_dir
                    )
                    await self._project_storage.initialize()
                    logger.info("Initialized ProjectStorage")

        return self._project_storage

    # Enterprise deployment infrastructure methods restored with feature flag control:
    # - get_feature_flag_manager() - Feature flag management with Flagsmith integration
    # - get_ab_testing_manager() - A/B testing with statistical analysis
    # - get_blue_green_deployment() - Zero-downtime blue-green deployments
    # - get_canary_deployment() - Progressive canary rollouts with automated monitoring
    # These provide enterprise-grade deployment capabilities while maintaining simplicity for personal use

    async def get_browser_automation_router(self):
        """Get or create BrowserAutomationRouter instance."""
        if self._browser_automation_router is None:
            if "browser_automation_router" not in self._service_locks:
                self._service_locks["browser_automation_router"] = asyncio.Lock()

            async with self._service_locks["browser_automation_router"]:
                if self._browser_automation_router is None:
                    from src.services.browser.browser_router import (
                        EnhancedAutomationRouter,
                    )

                    self._browser_automation_router = EnhancedAutomationRouter(
                        config=self.config,
                    )
                    await self._browser_automation_router.initialize()
                    logger.info(
                        "Initialized EnhancedAutomationRouter with performance tracking"
                    )

        return self._browser_automation_router

    async def get_task_queue_manager(self):
        """Get or create TaskQueueManager instance."""
        if self._task_queue_manager is None:
            if "task_queue_manager" not in self._service_locks:
                self._service_locks["task_queue_manager"] = asyncio.Lock()

            async with self._service_locks["task_queue_manager"]:
                if self._task_queue_manager is None:
                    from src.services.task_queue.manager import TaskQueueManager

                    self._task_queue_manager = TaskQueueManager(
                        config=self.config,
                    )
                    await self._task_queue_manager.initialize()
                    logger.info("Initialized TaskQueueManager")

        return self._task_queue_manager

    async def get_content_intelligence_service(self):
        """Get or create ContentIntelligenceService instance."""
        if self._content_intelligence_service is None:
            if "content_intelligence_service" not in self._service_locks:
                self._service_locks["content_intelligence_service"] = asyncio.Lock()

            async with self._service_locks["content_intelligence_service"]:
                if self._content_intelligence_service is None:
                    from src.services.content_intelligence.service import (
                        ContentIntelligenceService,
                    )

                    # Get dependencies
                    embedding_manager = await self.get_embedding_manager()
                    cache_manager = await self.get_cache_manager()

                    self._content_intelligence_service = ContentIntelligenceService(
                        config=self.config,
                        embedding_manager=embedding_manager,
                        cache_manager=cache_manager,
                    )
                    await self._content_intelligence_service.initialize()
                    logger.info("Initialized ContentIntelligenceService")

        return self._content_intelligence_service

    async def get_rag_generator(self):
        """Get or create RAGGenerator instance."""
        if self._rag_generator is None:
            if "rag_generator" not in self._service_locks:
                self._service_locks["rag_generator"] = asyncio.Lock()

            async with self._service_locks["rag_generator"]:
                if self._rag_generator is None:
                    from src.services.rag import RAGGenerator

                    self._rag_generator = RAGGenerator(
                        config=self.config.rag,
                        client_manager=self,
                    )
                    await self._rag_generator.initialize()
                    logger.info("Initialized RAGGenerator")

        return self._rag_generator

    async def get_database_manager(self) -> DatabaseManager:
        """Get or create enterprise DatabaseManager instance."""
        if self._database_manager is None:
            if "database_manager" not in self._service_locks:
                self._service_locks["database_manager"] = asyncio.Lock()

            async with self._service_locks["database_manager"]:
                if self._database_manager is None:
                    # Create enterprise monitoring components
                    from .database.monitoring import LoadMonitor
                    from .database.monitoring import QueryMonitor

                    load_monitor = LoadMonitor()
                    query_monitor = QueryMonitor()

                    # Create circuit breaker for enterprise resilience
                    circuit_breaker = CircuitBreaker(
                        failure_threshold=5,
                        recovery_timeout=60.0,
                        half_open_requests=1,
                    )

                    self._database_manager = DatabaseManager(
                        config=self.config,
                        load_monitor=load_monitor,
                        query_monitor=query_monitor,
                        circuit_breaker=circuit_breaker,
                    )
                    await self._database_manager.initialize()
                    logger.info(
                        "Initialized enterprise DatabaseManager with ML monitoring"
                    )

        return self._database_manager

    async def get_advanced_search_orchestrator(self):
        """Get or create AdvancedSearchOrchestrator instance."""
        if self._advanced_search_orchestrator is None:
            if "advanced_search_orchestrator" not in self._service_locks:
                self._service_locks["advanced_search_orchestrator"] = asyncio.Lock()

            async with self._service_locks["advanced_search_orchestrator"]:
                if self._advanced_search_orchestrator is None:
                    from src.services.query_processing import AdvancedSearchOrchestrator

                    self._advanced_search_orchestrator = AdvancedSearchOrchestrator(
                        enable_all_features=True, enable_performance_optimization=True
                    )
                    logger.info("Initialized AdvancedSearchOrchestrator")

        return self._advanced_search_orchestrator

    # Enterprise Deployment Services

    async def get_feature_flag_manager(self):
        """Get or create FeatureFlagManager instance."""
        if self._feature_flag_manager is None:
            if "feature_flag_manager" not in self._service_locks:
                self._service_locks["feature_flag_manager"] = asyncio.Lock()

            async with self._service_locks["feature_flag_manager"]:
                if self._feature_flag_manager is None:
                    from src.services.deployment.feature_flags import FeatureFlagConfig
                    from src.services.deployment.feature_flags import FeatureFlagManager

                    # Create feature flag config from deployment config
                    flag_config = FeatureFlagConfig(
                        enabled=self.config.deployment.enable_feature_flags,
                        api_key=self.config.deployment.flagsmith_api_key,
                        environment_key=self.config.deployment.flagsmith_environment_key,
                        api_url=self.config.deployment.flagsmith_api_url,
                    )

                    self._feature_flag_manager = FeatureFlagManager(flag_config)
                    await self._feature_flag_manager.initialize()
                    logger.info("Initialized FeatureFlagManager")

        return self._feature_flag_manager

    async def get_ab_testing_manager(self):
        """Get or create ABTestingManager instance."""
        if not self.config.deployment.enable_ab_testing:
            return None

        if self._ab_testing_manager is None:
            if "ab_testing_manager" not in self._service_locks:
                self._service_locks["ab_testing_manager"] = asyncio.Lock()

            async with self._service_locks["ab_testing_manager"]:
                if self._ab_testing_manager is None:
                    from src.services.deployment.ab_testing import ABTestingManager

                    # Get dependencies
                    qdrant_service = await self.get_qdrant_service()
                    cache_manager = await self.get_cache_manager()
                    feature_flag_manager = await self.get_feature_flag_manager()

                    self._ab_testing_manager = ABTestingManager(
                        qdrant_service=qdrant_service,
                        cache_manager=cache_manager,
                        feature_flag_manager=feature_flag_manager,
                    )
                    await self._ab_testing_manager.initialize()
                    logger.info("Initialized ABTestingManager")

        return self._ab_testing_manager

    async def get_blue_green_deployment(self):
        """Get or create BlueGreenDeployment instance."""
        if not self.config.deployment.enable_blue_green:
            return None

        if self._blue_green_deployment is None:
            if "blue_green_deployment" not in self._service_locks:
                self._service_locks["blue_green_deployment"] = asyncio.Lock()

            async with self._service_locks["blue_green_deployment"]:
                if self._blue_green_deployment is None:
                    from src.services.deployment.blue_green import BlueGreenDeployment

                    # Get dependencies
                    qdrant_service = await self.get_qdrant_service()
                    cache_manager = await self.get_cache_manager()
                    feature_flag_manager = await self.get_feature_flag_manager()

                    self._blue_green_deployment = BlueGreenDeployment(
                        qdrant_service=qdrant_service,
                        cache_manager=cache_manager,
                        feature_flag_manager=feature_flag_manager,
                    )
                    await self._blue_green_deployment.initialize()
                    logger.info("Initialized BlueGreenDeployment")

        return self._blue_green_deployment

    async def get_canary_deployment(self):
        """Get or create CanaryDeployment instance."""
        if not self.config.deployment.enable_canary:
            return None

        if self._canary_deployment is None:
            if "canary_deployment" not in self._service_locks:
                self._service_locks["canary_deployment"] = asyncio.Lock()

            async with self._service_locks["canary_deployment"]:
                if self._canary_deployment is None:
                    from src.services.deployment.canary import CanaryDeployment

                    # Get dependencies
                    qdrant_service = await self.get_qdrant_service()
                    cache_manager = await self.get_cache_manager()
                    feature_flag_manager = await self.get_feature_flag_manager()

                    self._canary_deployment = CanaryDeployment(
                        qdrant_service=qdrant_service,
                        cache_manager=cache_manager,
                        feature_flag_manager=feature_flag_manager,
                    )
                    await self._canary_deployment.initialize()
                    logger.info("Initialized CanaryDeployment")

        return self._canary_deployment

    async def _get_or_create_client(
        self,
        name: str,
        create_func,
        health_check_func,
    ) -> Any:
        """Get existing client or create new one with health checks.

        Args:
            name: Client name
            create_func: Function to create client
            health_check_func: Function to check client health

        Returns:
            Client instance

        Raises:
            APIError: If client is unhealthy or creation fails
        """
        # Check circuit breaker state
        if name in self._circuit_breakers:
            breaker = self._circuit_breakers[name]
            if breaker.state == ClientState.FAILED:
                raise APIError(f"{name} client circuit breaker is open")

        # Get or create client
        if name not in self._clients:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()

            async with self._locks[name]:
                # Double-check after acquiring lock
                if name not in self._clients:
                    try:
                        # Create circuit breaker if not exists
                        if name not in self._circuit_breakers:
                            # Use configuration values for circuit breaker settings
                            failure_threshold = getattr(
                                self.config.performance,
                                "circuit_breaker_failure_threshold",
                                5,
                            )
                            recovery_timeout = getattr(
                                self.config.performance,
                                "circuit_breaker_recovery_timeout",
                                60.0,
                            )
                            half_open_requests = getattr(
                                self.config.performance,
                                "circuit_breaker_half_open_requests",
                                1,
                            )

                            self._circuit_breakers[name] = CircuitBreaker(
                                failure_threshold=failure_threshold,
                                recovery_timeout=recovery_timeout,
                                half_open_requests=half_open_requests,
                            )

                        # Create client with circuit breaker
                        breaker = self._circuit_breakers[name]
                        client = await breaker.call(create_func)
                        self._clients[name] = client

                        # Initialize health status
                        self._health[name] = ClientHealth(
                            state=ClientState.HEALTHY,
                            last_check=time.time(),
                        )

                        logger.info(f"Created {name} client")

                    except Exception as e:
                        logger.exception(f"Failed to create {name} client: {e}")
                        raise APIError(f"Failed to create {name} client: {e}") from e

        # Check health status
        if name in self._health:
            health = self._health[name]
            max_failures = getattr(
                self.config.performance, "max_consecutive_failures", 3
            )
            if (
                health.state == ClientState.FAILED
                and health.consecutive_failures >= max_failures
            ):
                raise APIError(f"{name} client is unhealthy: {health.last_error}")

        return self._clients[name]

    async def _create_qdrant_client(self) -> AsyncQdrantClient:
        """Create Qdrant client instance."""
        client = AsyncQdrantClient(
            url=self.config.qdrant.url,
            api_key=self.config.qdrant.api_key,
            timeout=self.config.qdrant.timeout,
            prefer_grpc=self.config.qdrant.prefer_grpc,
        )

        # Validate connection
        await client.get_collections()
        return client

    async def _create_openai_client(self) -> AsyncOpenAI:
        """Create OpenAI client instance."""
        return AsyncOpenAI(
            api_key=self.config.openai.api_key,
            timeout=self.config.openai.max_tokens_per_minute
            / 1000,  # Convert to seconds
            max_retries=self.config.performance.max_retries,
        )

    async def _create_firecrawl_client(self) -> AsyncFirecrawlApp:
        """Create Firecrawl client instance."""
        return AsyncFirecrawlApp(
            api_key=self.config.firecrawl.api_key,
        )

    async def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with connection pooling."""
        return redis.from_url(
            self.config.cache.dragonfly_url,
            max_connections=self.config.cache.redis_pool_size,
            decode_responses=True,  # From cache config
        )

    async def _check_qdrant_health(self) -> bool:
        """Check Qdrant client health."""
        try:
            client = self._clients.get("qdrant")
            if not client:
                return False

            # Use circuit breaker for health check
            breaker = self._circuit_breakers.get("qdrant")
            if breaker:
                await breaker.call(client.get_collections)
            else:
                await client.get_collections()

            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def _check_openai_health(self) -> bool:
        """Check OpenAI client health."""
        try:
            client = self._clients.get("openai")
            if not client:
                return False

            # Simple API call to check connectivity
            breaker = self._circuit_breakers.get("openai")
            if breaker:
                await breaker.call(client.models.list)
            else:
                await client.models.list()

            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    async def _check_firecrawl_health(self) -> bool:
        """Check Firecrawl client health."""
        try:
            client = self._clients.get("firecrawl")
            # Firecrawl doesn't have a direct health endpoint
            # We'll assume it's healthy if client exists
            return bool(client)
        except Exception as e:
            logger.warning(f"Firecrawl health check failed: {e}")
            return False

    async def _check_redis_health(self) -> bool:
        """Check Redis client health."""
        try:
            client = self._clients.get("redis")
            if not client:
                return False

            # Simple ping to check connectivity
            breaker = self._circuit_breakers.get("redis")
            if breaker:
                await breaker.call(client.ping)
            else:
                await client.ping()

            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    async def _run_single_health_check(self, name: str, check_func) -> None:
        """Run a single health check and update client status."""
        try:
            # Run health check with timeout
            health_timeout = getattr(
                self.config.performance, "health_check_timeout", 5.0
            )
            is_healthy = await asyncio.wait_for(
                check_func(),
                timeout=health_timeout,
            )

            # Update health status
            if name not in self._health:
                self._health[name] = ClientHealth(
                    state=ClientState.HEALTHY,
                    last_check=time.time(),
                )

            health = self._health[name]
            previous_state = health.state
            health.last_check = time.time()

            if is_healthy:
                # Check if client was previously failed and should be recreated
                if previous_state == ClientState.FAILED:
                    await self._recreate_client_if_needed(name)
                    logger.info(f"{name} client recovered and recreated")

                health.state = ClientState.HEALTHY
                health.consecutive_failures = 0
                health.last_error = None
            else:
                health.consecutive_failures += 1
                max_failures = getattr(
                    self.config.performance, "max_consecutive_failures", 3
                )
                health.state = (
                    ClientState.DEGRADED
                    if health.consecutive_failures < max_failures
                    else ClientState.FAILED
                )
                health.last_error = "Health check returned false"

        except TimeoutError:
            logger.exception(f"{name} health check timed out")
            self._update_health_failure(name, "Health check timeout")
        except Exception as e:
            logger.exception(f"{name} health check error: {e}")
            self._update_health_failure(name, str(e))

    async def _recreate_client_if_needed(self, name: str) -> None:
        """Recreate a client if it was previously failed."""
        if name not in self._clients:
            return

        try:
            # Remove old client
            old_client = self._clients[name]
            if hasattr(old_client, "close"):
                await old_client.close()
            elif hasattr(old_client, "aclose"):
                await old_client.aclose()

            # Remove from clients dict to force recreation on next access
            del self._clients[name]

            # Reset circuit breaker
            if name in self._circuit_breakers:
                breaker = self._circuit_breakers[name]
                breaker._failure_count = 0
                breaker._state = ClientState.HEALTHY
                breaker._half_open_attempts = 0

            logger.info(f"Recreated {name} client after recovery")

        except Exception as e:
            logger.warning(f"Failed to recreate {name} client: {e}")

    async def _health_check_loop(self) -> None:
        """Background task to periodically check client health."""
        health_checks = {
            "qdrant": self._check_qdrant_health,
            "openai": self._check_openai_health,
            "firecrawl": self._check_firecrawl_health,
            "redis": self._check_redis_health,
        }

        while True:
            try:
                health_interval = getattr(
                    self.config.performance, "health_check_interval", 30.0
                )
                await asyncio.sleep(health_interval)

                # Run health checks in parallel for better performance
                active_clients = [
                    (name, check_func)
                    for name, check_func in health_checks.items()
                    if name in self._clients
                ]

                if not active_clients:
                    continue

                # Create tasks for parallel execution
                tasks = []
                for name, check_func in active_clients:
                    task = asyncio.create_task(
                        self._run_single_health_check(name, check_func),
                        name=f"health_check_{name}",
                    )
                    tasks.append(task)

                # Wait for all health checks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    def _update_health_failure(self, name: str, error: str) -> None:
        """Update health status for a failure."""
        if name not in self._health:
            self._health[name] = ClientHealth(
                state=ClientState.FAILED,
                last_check=time.time(),
                last_error=error,
                consecutive_failures=1,
            )
        else:
            health = self._health[name]
            health.last_check = time.time()
            health.last_error = error
            health.consecutive_failures += 1
            max_failures = getattr(
                self.config.performance, "max_consecutive_failures", 3
            )
            health.state = (
                ClientState.DEGRADED
                if health.consecutive_failures < max_failures
                else ClientState.FAILED
            )

    @asynccontextmanager
    async def managed_client(self, client_type: str):
        """Context manager for automatic client lifecycle management.

        Args:
            client_type: Type of client ("qdrant", "openai", "firecrawl", "redis", "database", "rag", "feature_flags", "ab_testing", "blue_green", "canary")

        Yields:
            Client instance

        Example:
            async with client_manager.managed_client("qdrant") as client:
                await client.get_collections()

            async with client_manager.managed_client("database") as db_manager:
                async with db_manager.get_session() as session:
                    result = await session.execute("SELECT 1")
        """
        client_getters = {
            "qdrant": self.get_qdrant_client,
            "openai": self.get_openai_client,
            "firecrawl": self.get_firecrawl_client,
            "redis": self.get_redis_client,
            "database": self.get_database_manager,
            "rag": self.get_rag_generator,
            # Deployment services
            "feature_flags": self.get_feature_flag_manager,
            "ab_testing": self.get_ab_testing_manager,
            "blue_green": self.get_blue_green_deployment,
            "canary": self.get_canary_deployment,
        }

        if client_type not in client_getters:
            raise ValueError(f"Unknown client type: {client_type}")

        try:
            client = await client_getters[client_type]()
            yield client
        except Exception as e:
            logger.exception(f"Error using {client_type} client: {e}")
            raise

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all clients.

        Returns:
            Dictionary mapping client names to health information
        """
        status = {}

        for name, health in self._health.items():
            status[name] = {
                "state": health.state.value,
                "last_check": health.last_check,
                "last_error": health.last_error,
                "consecutive_failures": health.consecutive_failures,
                "circuit_breaker_state": (
                    self._circuit_breakers[name].state.value
                    if name in self._circuit_breakers
                    else None
                ),
            }

        return status

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False
