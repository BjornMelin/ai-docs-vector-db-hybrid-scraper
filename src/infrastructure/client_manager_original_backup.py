"""Client coordination layer using dependency injection."""

import asyncio
import contextlib
import logging
import threading
import time
from contextlib import asynccontextmanager

from dependency_injector.wiring import Provide, inject

from src.infrastructure.clients import (
from typing import Any

    FirecrawlClientProvider,
    HTTPClientProvider,
    OpenAIClientProvider,
    QdrantClientProvider,
    RedisClientProvider,
)
from src.infrastructure.container import ApplicationContainer, get_container
from src.infrastructure.shared import ClientHealth, ClientState
from src.services.errors import APIError


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ClientManager:
    """Client coordination layer using dependency injection.

    This class provides a simplified interface to client providers
    and maintains backward compatibility while using DI internally.
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

        self._providers: dict[str, Any] = {}
        self._health: dict[str, ClientHealth] = {}
        self._initialized = False
        self._health_check_task: asyncio.Task | None = None

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
        """Initialize client providers using dependency injection.

        Args:
            openai_provider: OpenAI client provider
            qdrant_provider: Qdrant client provider
            redis_provider: Redis client provider
            firecrawl_provider: Firecrawl client provider
            http_provider: HTTP client provider
        """
        self._providers = {
            "openai": openai_provider,
            "qdrant": qdrant_provider,
            "redis": redis_provider,
            "firecrawl": firecrawl_provider,
            "http": http_provider,
        }

    async def initialize(self) -> None:
        """Initialize client manager and start health checks."""
        if self._initialized:
            return

        # Initialize providers using DI
        container = get_container()
        if container:
            container.wire(modules=[__name__])
            self.initialize_providers()

        self._initialized = True
        # Start background health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("ClientManager initialized with dependency injection")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._init_lock:
            if (
                cls._instance is not None
                and hasattr(cls._instance, "_health_check_task")
                and cls._instance._health_check_task
                and not cls._instance._health_check_task.done()
            ):
                cls._instance._health_check_task.cancel()
            cls._instance = None

    # Simplified client getters using DI providers
    async def get_openai_client(self):
        """Get OpenAI client through provider."""
        provider = self._providers.get("openai")
        if not provider:
            return None
        return provider.client

    async def get_qdrant_client(self):
        """Get Qdrant client through provider."""
        provider = self._providers.get("qdrant")
        if not provider:
            raise APIError("Qdrant client provider not available")
        return provider.client

    async def get_redis_client(self):
        """Get Redis client through provider."""
        provider = self._providers.get("redis")
        if not provider:
            raise APIError("Redis client provider not available")
        return provider.client

    async def get_firecrawl_client(self):
        """Get Firecrawl client through provider."""
        provider = self._providers.get("firecrawl")
        if not provider:
            return None
        return provider.client

    async def get_http_client(self):
        """Get HTTP client through provider."""
        provider = self._providers.get("http")
        if not provider:
            raise APIError("HTTP client provider not available")
        return provider.client

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        self._providers.clear()
        self._health.clear()
        self._initialized = False
        logger.info("ClientManager cleaned up")

    async def _health_check_loop(self) -> None:
        """Background task to periodically check client health."""
        while True:
            try:
                await asyncio.sleep(30.0)  # Health check interval

                # Run health checks for all providers
                tasks = []
                for name, provider in self._providers.items():
                    task = asyncio.create_task(
                        self._run_single_health_check(name, provider),
                        name=f"health_check_{name}",
                    )
                    tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception:
                logger.exception("Error in health check loop")
                await asyncio.sleep(10)

    async def _run_single_health_check(self, name: str, provider: Any) -> None:
        """Run a single health check and update status."""
        try:
            is_healthy = await provider.health_check()

            if name not in self._health:
                self._health[name] = ClientHealth(
                    state=ClientState.HEALTHY,
                    last_check=time.time(),
                )

            health = self._health[name]
            health.last_check = time.time()

            if is_healthy:
                health.state = ClientState.HEALTHY
                health.consecutive_failures = 0
                health.last_error = None
            else:
                health.consecutive_failures += 1
                health.state = (
                    ClientState.DEGRADED
                    if health.consecutive_failures < 3
                    else ClientState.FAILED
                )
                health.last_error = "Health check returned false"

        except Exception as e:
            logger.exception(f"{name} health check error")
            if name not in self._health:
                self._health[name] = ClientHealth(
                    state=ClientState.FAILED,
                    last_check=time.time(),
                    last_error=str(e),
                    consecutive_failures=1,
                )
            else:
                health = self._health[name]
                health.last_check = time.time()
                health.last_error = str(e)
                health.consecutive_failures += 1
                health.state = ClientState.FAILED

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all clients."""
        status = {}
        for name, health in self._health.items():
            status[name] = {
                "state": health.state.value,
                "last_check": health.last_check,
                "last_error": health.last_error,
                "consecutive_failures": health.consecutive_failures,
            }
        return status

    @asynccontextmanager
    async def managed_client(self, client_type: str):
        """Context manager for automatic client lifecycle management."""
        client_getters = {
            "qdrant": self.get_qdrant_client,
            "openai": self.get_openai_client,
            "firecrawl": self.get_firecrawl_client,
            "redis": self.get_redis_client,
            "http": self.get_http_client,
        }

        if client_type not in client_getters:
            msg = f"Unknown client type: {client_type}"
            raise ValueError(msg)

        try:
            client = await client_getters[client_type]()
            yield client
        except Exception:
            logger.exception(f"Error using {client_type} client")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False

        # Backward compatibility - remove all the massive service getter methods
        # Services should use dependency injection directly instead of going through ClientManager
        """Cleanup all clients and resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Clean up migration manager
        if self._migration_manager:
            try:
                await self._migration_manager.cleanup()
                logger.info("Cleaned up migration manager")
            except Exception:
                logger.exception("Error cleaning up migration manager")
            self._migration_manager = None

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
                    except Exception:
                        logger.exception(f"Error cleaning up {service_name}")
                setattr(self, service_name, None)

        # Close all clients
        for name, client in self._clients.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "aclose"):
                    await client.aclose()
                logger.info(f"Closed {name} client")
            except Exception:
                logger.exception(f"Error closing {name} client")

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
                    if QdrantService is None:
                        msg = "QdrantService not available"
                        raise ImportError(msg)

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
                    if EmbeddingManager is None:
                        msg = "EmbeddingManager not available"
                        raise ImportError(msg)

                    # Pass ClientManager to EmbeddingManager
                    self._embedding_manager = EmbeddingManager(
                        config=self.config,
                        client_manager=self,
                    )
                    await self._embedding_manager.initialize()
                    logger.info("Initialized EmbeddingManager")

        return self._embedding_manager

    async def get_cache_manager(self):
        """Get or create CacheManager instance with auto-detected Redis when available."""
        if self._cache_manager is None:
            if "cache_manager" not in self._service_locks:
                self._service_locks["cache_manager"] = asyncio.Lock()

            async with self._service_locks["cache_manager"]:
                if self._cache_manager is None:
                    if CacheManager is None:
                        msg = "CacheManager not available"
                        raise ImportError(msg)

                    # Check for auto-detected Redis service
                    auto_detected_redis = self._get_auto_detected_service("redis")

                    if auto_detected_redis:
                        logger.info(
                            f"Using auto-detected Redis for cache: {auto_detected_redis.connection_string}"
                        )
                        dragonfly_url = auto_detected_redis.connection_string
                        enable_distributed_cache = True
                    else:
                        dragonfly_url = self.config.cache.dragonfly_url
                        enable_distributed_cache = (
                            self.config.cache.enable_dragonfly_cache
                        )

                    self._cache_manager = CacheManager(
                        dragonfly_url=dragonfly_url,
                        enable_local_cache=self.config.cache.enable_local_cache,
                        enable_distributed_cache=enable_distributed_cache,
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
                    if CrawlManager is None:
                        msg = "CrawlManager not available"
                        raise ImportError(msg)

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
                    if (
                        HyDEConfig is None
                        or HyDEMetricsConfig is None
                        or HyDEPromptConfig is None
                    ):
                        msg = "HyDE configuration classes not available"
                        raise ImportError(msg)

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
                    if ProjectStorage is None:
                        msg = "ProjectStorage not available"
                        raise ImportError(msg)

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
                    if EnhancedAutomationRouter is None:
                        msg = "EnhancedAutomationRouter not available"
                        raise ImportError(msg)

                    self._browser_automation_router = EnhancedAutomationRouter(
                        config=self.config,
                    )
                    await self._browser_automation_router.initialize()
                    logger.info(
                        "Initialized EnhancedAutomationRouter with performance tracking"
                    )

        return self._browser_automation_router

    async def get_task_queue_manager(self):
        """Get or create TaskQueueManager instance with auto-detected Redis when available."""
        if self._task_queue_manager is None:
            if "task_queue_manager" not in self._service_locks:
                self._service_locks["task_queue_manager"] = asyncio.Lock()

            async with self._service_locks["task_queue_manager"]:
                if self._task_queue_manager is None:
                    if TaskQueueManager is None:
                        msg = "TaskQueueManager not available"
                        raise ImportError(msg)

                    # Check for auto-detected Redis service
                    auto_detected_redis = self._get_auto_detected_service("redis")

                    if auto_detected_redis:
                        logger.info(
                            f"Using auto-detected Redis for task queue: {auto_detected_redis.connection_string}"
                        )
                        # Create modified config with auto-detected Redis
                        config = deepcopy(self.config)
                        config.task_queue.redis_url = (
                            auto_detected_redis.connection_string
                        )
                    else:
                        config = self.config

                    self._task_queue_manager = TaskQueueManager(
                        config=config,
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
                    if ContentIntelligenceService is None:
                        msg = "ContentIntelligenceService not available"
                        raise ImportError(msg)

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
                    if RAGGenerator is None:
                        msg = "RAGGenerator not available"
                        raise ImportError(msg)

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
                    if LoadMonitor is None or QueryMonitor is None:
                        msg = "Database monitoring components not available"
                        raise ImportError(msg)

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

    async def get_search_orchestrator(self):
        """Get or create AdvancedSearchOrchestrator instance."""
        if self._search_orchestrator is None:
            if "search_orchestrator" not in self._service_locks:
                self._service_locks["search_orchestrator"] = asyncio.Lock()

            async with self._service_locks["search_orchestrator"]:
                if self._search_orchestrator is None:
                    if AdvancedSearchOrchestrator is None:
                        msg = "AdvancedSearchOrchestrator not available"
                        raise ImportError(msg)

                    self._search_orchestrator = AdvancedSearchOrchestrator(
                        enable_all_features=True, enable_performance_optimization=True
                    )
                    logger.info("Initialized AdvancedSearchOrchestrator")

        return self._search_orchestrator

    async def _initialize_migration_manager(self) -> None:
        """Initialize the migration manager for modern libraries."""
        try:
            # Determine Redis URL for modern implementations
            auto_detected_redis = self._get_auto_detected_service("redis")
            if auto_detected_redis:
                redis_url = auto_detected_redis.connection_string
            else:
                redis_url = getattr(
                    self.config.cache, "dragonfly_url", "redis://localhost:6379"
                )

            # Determine migration mode from config
            migration_mode = MigrationMode.GRADUAL
            if hasattr(self.config, "migration") and hasattr(
                self.config.migration, "mode"
            ):
                migration_mode = MigrationMode(self.config.migration.mode)

            # Create and initialize migration manager
            from src.services.migration.library_migration import (
                create_migration_manager,
            )

            self._migration_manager = create_migration_manager(
                config=self.config,
                mode=migration_mode,
                redis_url=redis_url,
            )
            await self._migration_manager.initialize()

            logger.info(
                f"Migration manager initialized with mode: {migration_mode.value}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize migration manager: {e}")
            # Continue without migration manager - will use legacy implementations

    async def get_migration_manager(self) -> LibraryMigrationManager | None:
        """Get the migration manager instance.

        Returns:
            LibraryMigrationManager instance or None if not available
        """
        return self._migration_manager

    async def get_modern_circuit_breaker(self, service_name: str = "default"):
        """Get modern circuit breaker through migration manager.

        Args:
            service_name: Name of the service

        Returns:
            Circuit breaker instance (modern or legacy based on migration state)
        """
        if self._migration_manager:
            return await self._migration_manager.get_circuit_breaker(service_name)
        else:
            # Fallback to legacy circuit breaker
            if service_name not in self._circuit_breakers:
                self._circuit_breakers[service_name] = CircuitBreaker()
            return self._circuit_breakers[service_name]

    async def get_modern_cache_manager(self):
        """Get modern cache manager through migration manager.

        Returns:
            Cache manager instance (modern or legacy based on migration state)
        """
        if self._migration_manager:
            return await self._migration_manager.get_cache_manager()
        else:
            # Fallback to legacy cache manager
            return await self.get_cache_manager()

    # Enterprise Deployment Services

    async def get_feature_flag_manager(self):
        """Get or create FeatureFlagManager instance."""
        if self._feature_flag_manager is None:
            if "feature_flag_manager" not in self._service_locks:
                self._service_locks["feature_flag_manager"] = asyncio.Lock()

            async with self._service_locks["feature_flag_manager"]:
                if self._feature_flag_manager is None:
                    if FeatureFlagConfig is None or FeatureFlagManager is None:
                        msg = "Feature flag components not available"
                        raise ImportError(msg)

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
                    if ABTestingManager is None:
                        msg = "ABTestingManager not available"
                        raise ImportError(msg)

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
                    if BlueGreenDeployment is None:
                        msg = "BlueGreenDeployment not available"
                        raise ImportError(msg)

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
                    if CanaryDeployment is None:
                        msg = "CanaryDeployment not available"
                        raise ImportError(msg)

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
        _health_check_func,
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
                msg = f"{name} client circuit breaker is open"
                raise APIError(msg)

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
                        logger.exception(f"Failed to create {name} client")
                        msg = f"Failed to create {name} client: {e}"
                        raise APIError(msg) from e

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
                msg = f"{name} client is unhealthy: {health.last_error}"
                raise APIError(msg)

        return self._clients[name]

    def _get_auto_detected_service(self, service_type: str) -> Any:
        """Get auto-detected service configuration if available.

        Args:
            service_type: Type of service (redis, qdrant, postgresql)

        Returns:
            Auto-detected service config or None

        """
        if not self._auto_detected_services:
            return None

        return next(
            (
                service
                for service in self._auto_detected_services.services
                if service.service_type == service_type and service.is_available
            ),
            None,
        )

    def _is_service_auto_detected(self, service_type: str) -> bool:
        """Check if a service was auto-detected and is available."""
        return self._get_auto_detected_service(service_type) is not None

    def _log_auto_detection_usage(
        self, service_type: str, auto_detected_service: Any
    ) -> None:
        """Log when using auto-detected service configuration."""
        logger.info(
            f"Using auto-detected {service_type} service: "
            f"{auto_detected_service.host}:{auto_detected_service.port} "
            f"(detection time: {auto_detected_service.detection_time_ms:.1f}ms)"
        )

    async def _create_qdrant_client(self) -> AsyncQdrantClient:
        """Create Qdrant client instance using auto-detected service when available."""
        # Check for auto-detected Qdrant service
        auto_detected = self._get_auto_detected_service("qdrant")

        if auto_detected:
            self._log_auto_detection_usage("qdrant", auto_detected)

            # Use auto-detected configuration with gRPC preference
            prefer_grpc = auto_detected.metadata.get("grpc_available", False)
            grpc_port = auto_detected.metadata.get("grpc_port", 6334)

            if prefer_grpc:
                client = AsyncQdrantClient(
                    host=auto_detected.host,
                    port=grpc_port,
                    prefer_grpc=True,
                    timeout=self.config.qdrant.timeout,
                    api_key=self.config.qdrant.api_key,
                )
            else:
                client = AsyncQdrantClient(
                    host=auto_detected.host,
                    port=auto_detected.port,
                    prefer_grpc=False,
                    timeout=self.config.qdrant.timeout,
                    api_key=self.config.qdrant.api_key,
                )
        else:
            # Fall back to manual configuration
            logger.info("Using manual Qdrant configuration (no auto-detection)")
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
        """Create Redis client with connection pooling using auto-detected service when available."""
        # Check for auto-detected Redis service
        auto_detected = self._get_auto_detected_service("redis")

        if auto_detected:
            self._log_auto_detection_usage("redis", auto_detected)

            # Use auto-detected Redis configuration
            redis_url = auto_detected.connection_string
            pool_config = auto_detected.pool_config

            return redis.from_url(
                redis_url,
                max_connections=pool_config.get("max_connections", 20),
                decode_responses=pool_config.get("decode_responses", True),
                retry_on_timeout=pool_config.get("retry_on_timeout", True),
                socket_keepalive=pool_config.get("socket_keepalive", True),
                socket_keepalive_options=pool_config.get(
                    "socket_keepalive_options", {}
                ),
                protocol=pool_config.get("protocol", 3),  # Redis 8.2 RESP3
            )
        # Fall back to manual configuration
        logger.info("Using manual Redis configuration (no auto-detection)")
        return redis.from_url(
            self.config.cache.dragonfly_url,
            max_connections=getattr(self.config.cache, "redis_pool_size", 20),
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
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False
        else:
            return True

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
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
        else:
            return True

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
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
        else:
            return True

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
            logger.exception(f"{name} health check error")
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
            except Exception:
                logger.exception("Error in health check loop")
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
            msg = f"Unknown client type: {client_type}"
            raise ValueError(msg)

        try:
            client = await client_getters[client_type]()
            yield client
        except Exception:
            logger.exception(f"Error using {client_type} client")
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