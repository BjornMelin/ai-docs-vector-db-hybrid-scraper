"""Dependency injection container for the application."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

import aiohttp
import redis.asyncio as redis
from dependency_injector import containers, providers
from dependency_injector.providers import Singleton  # pylint: disable=no-name-in-module
from dependency_injector.wiring import Provide
from firecrawl import AsyncFirecrawlApp
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient


# Import parallel processing components
try:
    from src.services.processing.parallel_integration import (
        OptimizationConfig,
        ParallelProcessingSystem,
    )
except ImportError:
    OptimizationConfig = None
    ParallelProcessingSystem = None


logger = logging.getLogger(__name__)


def _create_openai_client(config: Any) -> AsyncOpenAI:
    """Create OpenAI client with configuration.

    Args:
        config: Configuration object

    Returns:
        OpenAI client

    """
    try:
        api_key = getattr(getattr(config, "openai", None), "api_key", None) or ""
        max_retries = (
            getattr(getattr(config, "performance", None), "max_retries", None) or 3
        )
        return AsyncOpenAI(api_key=api_key, max_retries=max_retries)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to create OpenAI client with config: %s", e)
        return AsyncOpenAI(api_key="", max_retries=3)


def _create_qdrant_client(config: Any) -> AsyncQdrantClient:
    """Create Qdrant client with configuration.

    Args:
        config: Configuration object

    Returns:
        Qdrant client

    """
    try:
        qdrant_config = getattr(config, "qdrant", None)
        url = getattr(qdrant_config, "url", None) or "http://localhost:6333"
        api_key = getattr(qdrant_config, "api_key", None)
        timeout = float(getattr(qdrant_config, "timeout", None) or 30.0)
        prefer_grpc = getattr(qdrant_config, "prefer_grpc", None) or False
        return AsyncQdrantClient(
            url=url, api_key=api_key, timeout=timeout, prefer_grpc=prefer_grpc
        )
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to create Qdrant client with config: %s", e)
        return AsyncQdrantClient(url="http://localhost:6333")


def _create_redis_client(config: Any) -> redis.Redis:
    """Create Redis client with configuration.

    Args:
        config: Configuration object

    Returns:
        Redis client

    """
    try:
        cache_config = getattr(config, "cache", None)
        url = getattr(cache_config, "dragonfly_url", None) or "redis://localhost:6379"
        pool_size = getattr(cache_config, "redis_pool_size", None) or 20
        return redis.from_url(url, max_connections=pool_size, decode_responses=True)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to create Redis client with config: %s", e)
        return redis.from_url(
            "redis://localhost:6379", max_connections=20, decode_responses=True
        )


def _create_firecrawl_client(config: Any) -> AsyncFirecrawlApp:
    """Create Firecrawl client with configuration.

    Args:
        config: Configuration object

    Returns:
        Firecrawl client

    """
    try:
        firecrawl_config = getattr(config, "firecrawl", None)
        api_key = getattr(firecrawl_config, "api_key", None) or ""
        return AsyncFirecrawlApp(api_key=api_key)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning("Failed to create Firecrawl client with config: %s", e)
        return AsyncFirecrawlApp(api_key="")


async def _create_http_client() -> AsyncGenerator[Any]:
    """Create HTTP client with proper lifecycle management.

    Yields:
        HTTP client session

    """
    async with asyncio.timeout(30.0):
        timeout_config = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            yield session


def _create_parallel_processing_system(embedding_manager: Any) -> Any:
    """Create parallel processing system with optimizations.

    Args:
        embedding_manager: EmbeddingManager instance

    Returns:
        ParallelProcessingSystem instance

    """
    if OptimizationConfig and ParallelProcessingSystem:
        # Create optimization configuration
        config = OptimizationConfig(
            enable_parallel_processing=True,
            enable_caching=True,
            enable_optimized_algorithms=True,
            performance_monitoring=True,
            auto_optimization=True,
        )

        try:
            return ParallelProcessingSystem(embedding_manager, config)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Failed to create parallel processing system: %s", e)
            # Fall through to mock system

    # Return a minimal mock system if creation fails or components unavailable
    class MockParallelProcessingSystem:
        async def get_system_status(self):
            return {"system_health": {"status": "unavailable"}}

        async def cleanup(self):
            pass

    return MockParallelProcessingSystem()


class ApplicationContainer(containers.DeclarativeContainer):  # pylint: disable=c-extension-no-member
    """Application dependency injection container."""

    # Configuration
    config = providers.Configuration()  # pylint: disable=c-extension-no-member

    # Core client providers - using Factory for safe initialization
    openai_client = providers.Factory(  # pylint: disable=c-extension-no-member
        _create_openai_client,
        config=config,
    )

    qdrant_client = providers.Factory(  # pylint: disable=c-extension-no-member
        _create_qdrant_client,
        config=config,
    )

    redis_client = providers.Factory(  # pylint: disable=c-extension-no-member
        _create_redis_client,
        config=config,
    )

    firecrawl_client = providers.Factory(  # pylint: disable=c-extension-no-member
        _create_firecrawl_client,
        config=config,
    )

    # HTTP client with session management
    http_client = providers.Resource(  # pylint: disable=c-extension-no-member
        _create_http_client,
    )

    # Client provider layer
    openai_provider = Singleton(
        "src.infrastructure.clients.openai_client.OpenAIClientProvider",
        openai_client=openai_client,
    )

    qdrant_provider = Singleton(
        "src.infrastructure.clients.qdrant_client.QdrantClientProvider",
        qdrant_client=qdrant_client,
    )

    redis_provider = Singleton(
        "src.infrastructure.clients.redis_client.RedisClientProvider",
        redis_client=redis_client,
    )

    firecrawl_provider = Singleton(
        "src.infrastructure.clients.firecrawl_client.FirecrawlClientProvider",
        firecrawl_client=firecrawl_client,
    )

    http_provider = Singleton(
        "src.infrastructure.clients.http_client.HTTPClientProvider",
        http_client=http_client,
    )

    # Parallel processing system
    parallel_processing_system = providers.Factory(  # pylint: disable=c-extension-no-member
        _create_parallel_processing_system,
        embedding_manager=providers.DelegatedFactory(  # pylint: disable=c-extension-no-member
            "src.services.embeddings.manager.EmbeddingManager"
        ),
    )

    # Lifecycle management
    startup_tasks = providers.List()  # pylint: disable=c-extension-no-member
    shutdown_tasks = providers.List()  # pylint: disable=c-extension-no-member


class ContainerManager:
    """Manager for dependency injection container lifecycle."""

    def __init__(self):
        self.container: ApplicationContainer | None = None
        self._initialized = False

    async def initialize(self, config: Any) -> ApplicationContainer:
        """Initialize the container with configuration.

        Args:
            config: Application configuration

        Returns:
            Initialized container
        """

        if self._initialized:
            if self.container is None:
                raise RuntimeError("Container manager in inconsistent state")
            return self.container

        self.container = ApplicationContainer()
        self.container.config.from_dict(self._config_to_dict(config))

        # Initialize resource providers
        await self.container.init_resources()  # pyright: ignore[reportGeneralTypeIssues]

        self._initialized = True
        logger.info("Dependency injection container initialized")
        return self.container

    async def shutdown(self) -> None:
        """Shutdown the container and cleanup resources."""
        if self._initialized and self.container is not None:
            await self.container.shutdown_resources()  # pyright: ignore[reportGeneralTypeIssues]
            self.container = None
            self._initialized = False
            logger.info("Dependency injection container shutdown")

    def _config_to_dict(self, config: Any) -> dict:
        """Convert config object to dictionary for dependency-injector.

        Args:
            config: Configuration object

        Returns:
            Configuration dictionary

        """
        try:
            # Try to convert using model_dump if it's a Pydantic model
            if hasattr(config, "model_dump"):
                return config.model_dump()
            # Try to convert using dict() if it's a dataclass or similar
            if hasattr(config, "__dict__"):
                return self._serialize_config_dict(config.__dict__)
            # Fallback to basic attributes
            return {
                key: getattr(config, key)
                for key in dir(config)
                if not key.startswith("_") and not callable(getattr(config, key))
            }
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Failed to convert config to dict: %s", e)
            return {}

    def _serialize_config_dict(self, data: Any) -> Any:
        """Recursively serialize configuration data.

        Args:
            data: Data to serialize

        Returns:
            Serialized data

        """
        if hasattr(data, "model_dump"):
            return data.model_dump()
        if hasattr(data, "__dict__"):
            return {
                key: self._serialize_config_dict(value)
                for key, value in data.__dict__.items()
                if not key.startswith("_")
            }
        if isinstance(data, dict):
            return {
                key: self._serialize_config_dict(value) for key, value in data.items()
            }
        if isinstance(data, list | tuple):
            return [self._serialize_config_dict(item) for item in data]
        return data


# Global container manager instance
_container_manager = ContainerManager()


@lru_cache(maxsize=1)
def get_container() -> ApplicationContainer | None:
    """Get the global container instance.

    Returns:
        Container instance or None if not initialized

    """
    return _container_manager.container


async def initialize_container(config: Any) -> ApplicationContainer:
    """Initialize the global container.

    Args:
        config: Application configuration

    Returns:
        Initialized container

    """
    return await _container_manager.initialize(config)


async def shutdown_container() -> None:
    """Shutdown the global container."""
    await _container_manager.shutdown()
    get_container.cache_clear()


# Dependency injection decorators and functions for easy access
def inject_openai_provider():
    """Inject OpenAI client provider dependency."""
    return Provide[ApplicationContainer.openai_provider]


def inject_qdrant_provider():
    """Inject Qdrant client provider dependency."""
    return Provide[ApplicationContainer.qdrant_provider]


def inject_redis_provider():
    """Inject Redis client provider dependency."""
    return Provide[ApplicationContainer.redis_provider]


def inject_firecrawl_provider():
    """Inject Firecrawl client provider dependency."""
    return Provide[ApplicationContainer.firecrawl_provider]


def inject_http_provider():
    """Inject HTTP client provider dependency."""
    return Provide[ApplicationContainer.http_provider]


def inject_parallel_processing_system():
    """Inject parallel processing system dependency."""
    return Provide[ApplicationContainer.parallel_processing_system]


# Legacy raw client injection (for backward compatibility)
def inject_openai() -> AsyncOpenAI:
    """Inject raw OpenAI client dependency."""
    return Provide[ApplicationContainer.openai_client]


def inject_qdrant() -> AsyncQdrantClient:
    """Inject raw Qdrant client dependency."""
    return Provide[ApplicationContainer.qdrant_client]


def inject_redis() -> redis.Redis:
    """Inject raw Redis client dependency."""
    return Provide[ApplicationContainer.redis_client]


def inject_firecrawl() -> AsyncFirecrawlApp:
    """Inject raw Firecrawl client dependency."""
    return Provide[ApplicationContainer.firecrawl_client]


def inject_http() -> Any:
    """Inject raw HTTP client dependency."""
    return Provide[ApplicationContainer.http_client]


# Context manager for automatic dependency injection setup
class DependencyContext:
    """Context manager for dependency injection setup."""

    def __init__(self, config: Any):
        self.config = config
        self.container: ApplicationContainer | None = None

    async def __aenter__(self) -> ApplicationContainer:
        """Initialize dependencies."""
        self.container = await initialize_container(self.config)
        return self.container

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup dependencies."""
        await shutdown_container()


# Wire modules for automatic dependency injection
def wire_modules() -> None:
    """Wire modules for dependency injection."""
    container = get_container()
    if container:
        # Wire commonly used modules
        modules = [
            "src.services.embeddings",
            "src.services.vector_db",
            "src.services.crawling",
            "src.services.cache",
            "src.api.routers",
            "src.mcp_tools",
        ]
        container.wire(modules=modules)
        logger.info("Dependency injection wiring completed")
