"""FastAPI dependency injection functions."""

# pylint: disable=too-many-lines,no-else-return,no-else-raise,duplicate-code,cyclic-import

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends  # type: ignore[attr-defined]
from pydantic import BaseModel

from src.config import (
    AutoDetectedServices,
    CacheType,
    Config,
    DetectedEnvironment,
    Environment,
    get_config,
    get_config_with_auto_detection,
)
from src.infrastructure.client_manager import ClientManager
from src.services.auto_detection import (
    ConnectionPoolManager,
    EnvironmentDetector,
    HealthChecker,
    ServiceDiscovery,
)
from src.services.embeddings.manager import QualityTier
from src.services.errors import (
    CircuitBreakerRegistry,
    CrawlServiceError,
    EmbeddingServiceError,
    TaskQueueServiceError,
    circuit_breaker,
    tenacity_circuit_breaker,
)
from src.services.rag import RAGGenerator
from src.services.rag.models import RAGRequest as InternalRAGRequest
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)


#### HELPER FUNCTIONS & DECORATORS ####


def create_service_error_handler(
    service_name: str,
    error_class: type[Exception] = RuntimeError,
) -> Callable:
    """Create error handler decorator for service operations."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception("%s operation failed", service_name)
                msg = f"Failed {service_name} operation: {e}"
                raise error_class(msg) from e

        return wrapper

    return decorator


#### CONFIGURATION DEPENDENCIES ####

ConfigDep = Annotated[Config, Depends(get_config)]


async def get_auto_detected_config() -> Config:
    """Get configuration with auto-detection applied."""
    return await get_config_with_auto_detection()


AutoDetectedConfigDep = Annotated[Config, Depends(get_auto_detected_config)]


@lru_cache
def get_client_manager() -> ClientManager:
    """Get singleton ClientManager instance."""
    return ClientManager.from_unified_config()


async def get_auto_detected_client_manager() -> ClientManager:
    """Get ClientManager instance with auto-detection applied."""
    return await ClientManager.from_unified_config_with_auto_detection()


ClientManagerDep = Annotated[ClientManager, Depends(get_client_manager)]
AutoDetectedClientManagerDep = Annotated[
    ClientManager, Depends(get_auto_detected_client_manager)
]


# Auto-Detection Dependencies
@circuit_breaker(
    service_name="auto_detection_services",
    failure_threshold=3,
    recovery_timeout=30.0,
    enable_adaptive_timeout=True,
)
async def get_auto_detected_services(
    config: AutoDetectedConfigDep,
) -> Any:
    """Get auto-detected services information."""
    if (auto_detected := config.get_auto_detected_services()) is None:
        # Return empty services if auto-detection wasn't performed
        return AutoDetectedServices(
            environment=DetectedEnvironment(
                environment_type=Environment.DEVELOPMENT,
                is_containerized=False,
                is_kubernetes=False,
                detection_confidence=0.0,
                detection_time_ms=0.0,
            ),
            services=[],
            errors=["Auto-detection not performed"],
        )

    return auto_detected


AutoDetectedServicesDep = Annotated[Any, Depends(get_auto_detected_services)]


# Auto-Detection Service Access Dependencies
async def get_auto_detected_redis_client(
    client_manager: AutoDetectedClientManagerDep,
) -> Any:
    """Get Redis client using auto-detected configuration."""
    return await client_manager.get_redis_client()


async def get_auto_detected_qdrant_client(
    client_manager: AutoDetectedClientManagerDep,
) -> Any:
    """Get Qdrant client using auto-detected configuration."""
    return await client_manager.get_qdrant_client()


async def get_auto_detected_cache_manager(
    client_manager: AutoDetectedClientManagerDep,
) -> Any:
    """Get CacheManager using auto-detected Redis when available."""
    return await client_manager.get_cache_manager()


async def get_auto_detected_task_queue_manager(
    client_manager: AutoDetectedClientManagerDep,
) -> Any:
    """Get TaskQueueManager using auto-detected Redis when available."""
    return await client_manager.get_task_queue_manager()  # type: ignore[attr-defined]


AutoDetectedRedisDep = Annotated[Any, Depends(get_auto_detected_redis_client)]
AutoDetectedQdrantDep = Annotated[Any, Depends(get_auto_detected_qdrant_client)]
AutoDetectedCacheDep = Annotated[Any, Depends(get_auto_detected_cache_manager)]
AutoDetectedTaskQueueDep = Annotated[Any, Depends(get_auto_detected_task_queue_manager)]


@circuit_breaker(
    service_name="auto_detection_health",
    failure_threshold=2,
    recovery_timeout=15.0,
)
async def get_auto_detection_health_checker(
    config: AutoDetectedConfigDep,
) -> Any:
    """Get auto-detection health checker."""
    # HealthChecker imported at top-level

    health_checker = HealthChecker(config.auto_detection)

    # Initialize with auto-detected services if available
    auto_detected = config.get_auto_detected_services()
    if auto_detected and auto_detected.services:
        await health_checker.start_monitoring(auto_detected.services)

    return health_checker


AutoDetectionHealthDep = Annotated[Any, Depends(get_auto_detection_health_checker)]


@circuit_breaker(
    service_name="auto_detection_pools",
    failure_threshold=3,
    recovery_timeout=30.0,
)
async def get_auto_detection_connection_pools(
    config: AutoDetectedConfigDep,
) -> Any:
    """Get connection pool manager for auto-detected services."""
    # ConnectionPoolManager imported at top-level

    pool_manager = ConnectionPoolManager(config.auto_detection)

    # Initialize pools with auto-detected services if available
    auto_detected = config.get_auto_detected_services()
    if auto_detected and auto_detected.services:
        await pool_manager.initialize_pools(auto_detected.services)

    return pool_manager


AutoDetectionPoolsDep = Annotated[Any, Depends(get_auto_detection_connection_pools)]


class AutoDetectionRequest(BaseModel):
    """Auto-detection request model."""

    force_refresh: bool = False
    timeout_seconds: float | None = None
    enabled_services: list[str] | None = None


class AutoDetectionResponse(BaseModel):
    """Auto-detection response model."""

    services_found: int
    environment_type: str
    detection_time_ms: float
    services: list[dict[str, Any]] = []
    errors: list[str] = []
    is_cached: bool = False


@tenacity_circuit_breaker(
    service_name="perform_auto_detection",
    max_attempts=2,
    wait_multiplier=1.0,
    wait_max=30.0,
    failure_threshold=3,
    recovery_timeout=60.0,
)
async def perform_auto_detection(
    request: AutoDetectionRequest,
    config: ConfigDep,
) -> AutoDetectionResponse:
    """Perform service auto-detection (consolidated implementation)."""
    start_time = time.time()
    try:
        # Prepare detection config
        detection_config = config.auto_detection
        if request.timeout_seconds:
            detection_config.timeout_seconds = request.timeout_seconds

        # Perform environment detection and service discovery
        env_detector = EnvironmentDetector(config=detection_config)
        detected_env = await env_detector.detect()

        service_discovery = ServiceDiscovery(detection_config)
        discovery_result = await service_discovery.discover_all_services()

        # Filter services by type if requested
        services = discovery_result.services
        if request.enabled_services:
            services = [
                s
                for s in services
                if s.service_type in request.enabled_services  # type: ignore[attr-defined]
            ]

        # Format services for response
        formatted_services = [
            {
                "service_name": s.service_name,  # type: ignore[attr-defined]
                "service_type": s.service_type,  # type: ignore[attr-defined]
                "host": s.host,  # type: ignore[attr-defined]
                "port": s.port,  # type: ignore[attr-defined]
                "is_available": s.is_available,  # type: ignore[attr-defined]
                "connection_string": s.connection_string,  # type: ignore[attr-defined]
                "version": s.version,  # type: ignore[attr-defined]
                "supports_pooling": s.supports_pooling,  # type: ignore[attr-defined]
                "detection_time_ms": s.detection_time_ms,  # type: ignore[attr-defined]
                "metadata": s.metadata,  # type: ignore[attr-defined]
            }
            for s in services
        ]

        detection_time_ms = (time.time() - start_time) * 1000

        return AutoDetectionResponse(
            services_found=len(services),
            environment_type=detected_env.environment_type.value,
            detection_time_ms=detection_time_ms,
            services=formatted_services,
            errors=discovery_result.errors,
            is_cached=False,
        )
    except Exception as e:
        logger.exception("Auto-detection failed")
        return AutoDetectionResponse(
            services_found=0,
            environment_type="unknown",
            detection_time_ms=0.0,
            services=[],
            errors=[str(e)],
            is_cached=False,
        )


# Embedding Service Dependencies
class EmbeddingRequest(BaseModel):
    """Embedding generation request model."""

    texts: list[str]
    quality_tier: str | None = None
    provider_name: str | None = None
    max_cost: float | None = None
    speed_priority: bool = False
    auto_select: bool = True
    generate_sparse: bool = False


class EmbeddingResponse(BaseModel):
    """Embedding generation response model."""

    embeddings: list[list[float]]
    provider: str
    model: str
    cost: float
    latency_ms: float
    tokens: int
    reasoning: str
    quality_tier: str
    cache_hit: bool = False
    sparse_embeddings: list[dict[str, Any]] | None = None


@circuit_breaker(
    service_name="embedding_manager",
    failure_threshold=3,
    recovery_timeout=30.0,
    enable_adaptive_timeout=True,
)
async def get_embedding_manager(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized EmbeddingManager service."""
    return await client_manager.get_embedding_manager()


EmbeddingManagerDep = Annotated[Any, Depends(get_embedding_manager)]


@tenacity_circuit_breaker(
    service_name="generate_embeddings",
    max_attempts=3,
    wait_multiplier=1.0,
    wait_max=10.0,
    failure_threshold=3,
    recovery_timeout=30.0,
)
async def generate_embeddings(
    request: EmbeddingRequest,
    embedding_manager: EmbeddingManagerDep,
) -> EmbeddingResponse:
    """Generate embeddings with smart provider selection."""
    try:
        quality_tier = (
            QualityTier(request.quality_tier) if request.quality_tier else None
        )

        result = await embedding_manager.generate_embeddings(
            texts=request.texts,
            quality_tier=quality_tier,
            provider_name=request.provider_name,
            max_cost=request.max_cost,
            speed_priority=request.speed_priority,
            auto_select=request.auto_select,
            generate_sparse=request.generate_sparse,
        )

        return EmbeddingResponse(**result)
    except Exception as e:
        logger.exception("Embedding generation failed")
        msg = f"Failed to generate embeddings: {e}"
        raise EmbeddingServiceError(msg) from e


# Cache Service Dependencies
class CacheRequest(BaseModel):
    """Pydantic model for cache operations."""

    key: str
    value: Any | None = None
    cache_type: str = "crawl"
    ttl: int | None = None


@circuit_breaker(
    service_name="cache_manager",
    failure_threshold=2,
    recovery_timeout=10.0,
    enable_adaptive_timeout=True,
)
async def get_cache_manager(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized CacheManager service."""
    return await client_manager.get_cache_manager()


CacheManagerDep = Annotated[Any, Depends(get_cache_manager)]


async def cache_get(
    key: str,
    cache_type: str = "crawl",
    cache_manager: CacheManagerDep = None,
) -> Any:
    """Get value from cache."""
    try:
        return await cache_manager.get(key, CacheType(cache_type))
    except (OSError, ConnectionError, ImportError, ModuleNotFoundError):
        logger.exception("Cache get failed for key")
        return None


async def cache_set(
    key: str,
    value: Any,
    cache_type: str = "crawl",
    ttl: int | None = None,
    cache_manager: CacheManagerDep = None,
) -> bool:
    """Set value in cache."""
    try:
        return await cache_manager.set(key, value, CacheType(cache_type), ttl)
    except (OSError, ConnectionError, ImportError, ModuleNotFoundError):
        logger.exception("Cache set failed for key")
        return False


async def cache_delete(
    key: str,
    cache_type: str = "crawl",
    cache_manager: CacheManagerDep = None,
) -> bool:
    """Delete value from cache."""
    try:
        return await cache_manager.delete(key, CacheType(cache_type))
    except (OSError, ConnectionError, ImportError, ModuleNotFoundError):
        logger.exception("Cache delete failed for key")
        return False


# Crawl Service Dependencies
class CrawlRequest(BaseModel):
    """Pydantic model for crawling requests."""

    url: str
    preferred_provider: str | None = None
    max_pages: int = 50
    include_subdomains: bool = False


class CrawlResponse(BaseModel):
    """Pydantic model for crawling responses."""

    success: bool
    content: str = ""
    url: str
    title: str = ""
    metadata: dict[str, Any] = {}
    tier_used: str = "none"
    automation_time_ms: float = 0
    quality_score: float = 0.0
    error: str | None = None
    fallback_attempted: bool = False
    failed_tiers: list[str] = []


@circuit_breaker(
    service_name="crawl_manager",
    failure_threshold=5,
    recovery_timeout=60.0,
    enable_adaptive_timeout=True,
)
async def get_crawl_manager(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized CrawlManager service."""
    return await client_manager.get_crawl_manager()


CrawlManagerDep = Annotated[Any, Depends(get_crawl_manager)]


@tenacity_circuit_breaker(
    service_name="scrape_url",
    max_attempts=3,
    wait_multiplier=2.0,
    wait_max=30.0,
    failure_threshold=5,
    recovery_timeout=60.0,
)
async def scrape_url(
    request: CrawlRequest,
    crawl_manager: CrawlManagerDep,
) -> CrawlResponse:
    """Scrape URL with provider selection."""
    try:
        result = await crawl_manager.scrape_url(
            url=request.url, preferred_provider=request.preferred_provider
        )
    except Exception as e:
        logger.exception("URL scraping failed for %s", request.url)
        msg = f"Failed to scrape URL: {e}"
        raise CrawlServiceError(msg) from e
    else:
        return CrawlResponse(**result)


async def crawl_site(
    request: CrawlRequest,
    crawl_manager: CrawlManagerDep,
) -> dict[str, Any]:
    """Crawl entire website from starting URL."""
    try:
        result = await crawl_manager.crawl_site(
            url=request.url,
            max_pages=request.max_pages,
            preferred_provider=request.preferred_provider,
        )
    except Exception as e:
        logger.exception("Site crawling failed for %s", request.url)
        msg = f"Failed to crawl site: {e}"
        raise CrawlServiceError(msg) from e
    else:
        return result


# Task Queue Service Dependencies
class TaskRequest(BaseModel):
    """Pydantic model for task queue requests."""

    task_name: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    delay: int | None = None
    queue_name: str | None = None


async def get_task_queue_manager(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized TaskQueueManager service."""
    return await client_manager.get_task_queue_manager()  # type: ignore[attr-defined]


TaskQueueManagerDep = Annotated[Any, Depends(get_task_queue_manager)]


async def enqueue_task(
    request: TaskRequest,
    task_manager: TaskQueueManagerDep,
) -> str | None:
    """Enqueue a task for execution."""
    try:
        job_id = await task_manager.enqueue(
            request.task_name,
            *request.args,
            _delay=request.delay,
            _queue_name=request.queue_name,
            **request.kwargs,
        )
    except Exception as e:
        logger.exception("Task enqueue failed for %s", request.task_name)
        msg = f"Failed to enqueue task: {e}"
        raise TaskQueueServiceError(msg) from e
    else:
        return job_id


async def get_task_status(
    job_id: str,
    task_manager: TaskQueueManagerDep,
) -> dict[str, Any]:
    """Get status of a task."""
    try:
        status = await task_manager.get_job_status(job_id)
    except Exception as e:
        logger.exception("Task status check failed for")
        return {"status": "error", "message": str(e)}
    else:
        return status


# Database Dependencies
async def get_database_manager(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized DatabaseManager service."""
    return await client_manager.get_database_manager()


DatabaseManagerDep = Annotated[Any, Depends(get_database_manager)]


async def get_database_session(
    db_manager: DatabaseManagerDep,
) -> AsyncGenerator[Any]:
    """Get database session with automatic cleanup."""
    async with db_manager.get_session() as session:
        yield session


DatabaseSessionDep = Annotated[Any, Depends(get_database_session)]


# Vector Store Dependencies
@circuit_breaker(
    service_name="vector_store_service",
    failure_threshold=3,
    recovery_timeout=15.0,
    enable_adaptive_timeout=True,
)
async def get_vector_store_service(
    client_manager: ClientManagerDep,
) -> VectorStoreService:
    """Get initialized vector store service."""

    return await client_manager.get_vector_store_service()


VectorStoreServiceDep = Annotated[VectorStoreService, Depends(get_vector_store_service)]


# HyDE Engine Dependencies
async def get_hyde_engine(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized HyDEEngine service."""
    return await client_manager.get_hyde_engine()  # type: ignore[attr-defined]


HyDEEngineDep = Annotated[Any, Depends(get_hyde_engine)]


# Content Intelligence Dependencies
async def get_content_intelligence_service(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized ContentIntelligenceService."""
    return await client_manager.get_content_intelligence_service()  # type: ignore[attr-defined]


ContentIntelligenceServiceDep = Annotated[
    Any, Depends(get_content_intelligence_service)
]


# RAG Service Dependencies
class RAGRequest(BaseModel):
    """RAG answer generation request model."""

    query: str
    search_results: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    include_sources: bool = True
    require_high_confidence: bool = False
    max_context_results: int | None = None
    preferred_source_types: list[str] | None = None
    exclude_source_ids: list[str] | None = None


class RAGResponse(BaseModel):
    """RAG answer generation response model."""

    answer: str
    confidence_score: float
    sources_used: int
    generation_time_ms: float
    sources: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None
    follow_up_questions: list[str] | None = None
    cached: bool = False
    reasoning_trace: list[str] | None = None


@circuit_breaker(
    service_name="rag_generator",
    failure_threshold=3,
    recovery_timeout=30.0,
    enable_adaptive_timeout=True,
)
async def get_rag_generator(
    config: ConfigDep,
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized RAGGenerator service."""
    # RAGGenerator imported at top-level

    generator = RAGGenerator(config.rag, client_manager)  # type: ignore[arg-type]
    await generator.initialize()
    return generator


RAGGeneratorDep = Annotated[Any, Depends(get_rag_generator)]


def _convert_to_internal_rag_request(request: RAGRequest) -> InternalRAGRequest:
    """Convert external RAG request to internal format."""
    # Extract top_k from search_results length as a reasonable default
    if request.search_results:
        top_k = request.max_context_results or len(request.search_results)
    else:
        top_k = request.max_context_results

    # Extract filters from search_results metadata if available
    filters: dict[str, Any] | None = None
    if request.preferred_source_types:
        filters = {"source_types": request.preferred_source_types}

    return InternalRAGRequest(
        query=request.query,
        top_k=top_k,
        filters=filters,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        include_sources=request.include_sources,
    )


def _format_rag_sources(
    request: RAGRequest, result: Any
) -> list[dict[str, Any]] | None:
    """Format RAG sources for response."""
    if not (request.include_sources and result.sources):
        return None

    return [
        {
            "source_id": source.source_id,
            "title": source.title,
            "url": source.url,
            "relevance_score": source.score,
            "excerpt": source.excerpt,
        }
        for source in result.sources
    ]


def _format_rag_metrics(result: Any) -> dict[str, Any] | None:
    """Format RAG metrics for response."""
    if not result.metrics:
        return None

    return {
        "total_tokens": result.metrics.total_tokens,
        "prompt_tokens": result.metrics.prompt_tokens,
        "completion_tokens": result.metrics.completion_tokens,
        "generation_time_ms": result.metrics.generation_time_ms,
    }


@tenacity_circuit_breaker(
    service_name="generate_rag_answer",
    max_attempts=3,
    wait_multiplier=2.0,
    wait_max=30.0,
    failure_threshold=3,
    recovery_timeout=30.0,
)
async def generate_rag_answer(
    request: RAGRequest,
    rag_generator: RAGGeneratorDep,
) -> RAGResponse:
    """Generate contextual answer from search results using RAG."""
    try:
        internal_request = _convert_to_internal_rag_request(request)
        result = await rag_generator.generate_answer(internal_request)

        sources = _format_rag_sources(request, result)
        metrics = _format_rag_metrics(result)

        return RAGResponse(
            answer=result.answer,
            confidence_score=result.confidence_score or 0.0,
            sources_used=len(result.sources),
            generation_time_ms=result.generation_time_ms,
            sources=sources,
            metrics=metrics,
            follow_up_questions=None,
            cached=False,
            reasoning_trace=None,
        )
    except Exception as e:
        logger.exception("RAG answer generation failed")
        msg = f"Failed to generate RAG answer: {e}"
        raise EmbeddingServiceError(msg) from e


async def get_rag_metrics(
    rag_generator: RAGGeneratorDep,
) -> dict[str, Any]:
    """Get RAG service performance metrics."""
    try:
        metrics = rag_generator.get_metrics()
        # Convert RAGServiceMetrics to dict if it's a Pydantic model
        if hasattr(metrics, "model_dump"):
            return metrics.model_dump()
        return metrics
    except Exception as e:
        logger.exception("Failed to get RAG metrics")
        return {"error": str(e)}


async def clear_rag_cache(
    rag_generator: RAGGeneratorDep,
) -> dict[str, str]:
    """Clear RAG answer cache."""
    try:
        rag_generator.clear_cache()
        # RAGGenerator.clear_cache is a no-op (no cache to purge)
        return {
            "status": "noop",
            "message": "RAG generator operates without a local cache",
        }
    except Exception as e:
        logger.exception("Failed to clear RAG cache")
        return {"status": "error", "message": str(e)}


# Browser Automation Dependencies
async def get_browser_automation_router(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized BrowserAutomationRouter service."""
    return await client_manager.get_browser_automation_router()


BrowserAutomationRouterDep = Annotated[Any, Depends(get_browser_automation_router)]


# Project Storage Dependencies
async def get_project_storage(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized ProjectStorage service."""
    return await client_manager.get_project_storage()  # type: ignore[attr-defined]


ProjectStorageDep = Annotated[Any, Depends(get_project_storage)]


# Circuit Breaker Monitoring Functions
async def get_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all circuit breakers."""
    try:
        all_status = CircuitBreakerRegistry.get_all_status()

        # Calculate aggregate metrics
        total_circuits = len(all_status)
        open_circuits = sum(
            1 for status in all_status.values() if status["state"] == "open"
        )
        half_open_circuits = sum(
            1 for status in all_status.values() if status["state"] == "half_open"
        )

        return {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "summary": {
                "total_circuits": total_circuits,
                "open_circuits": open_circuits,
                "half_open_circuits": half_open_circuits,
                "closed_circuits": total_circuits - open_circuits - half_open_circuits,
                "health_percentage": (
                    (total_circuits - open_circuits) / total_circuits * 100
                )
                if total_circuits > 0
                else 100.0,
            },
            "circuits": all_status,
        }
    except Exception as e:
        logger.exception("Circuit breaker status check failed")
        return {
            "error": str(e),
            "summary": {
                "total_circuits": 0,
                "open_circuits": 0,
                "half_open_circuits": 0,
                "closed_circuits": 0,
                "health_percentage": 0.0,
            },
            "circuits": {},
        }


async def reset_circuit_breaker(service_name: str) -> dict[str, Any]:
    """Reset a specific circuit breaker."""

    try:
        if (breaker := CircuitBreakerRegistry.get(service_name)) is None:
            return {
                "success": False,
                "error": f"Circuit breaker not found for service: {service_name}",
                "available_services": CircuitBreakerRegistry.get_services(),
            }

        breaker.reset()
        return {
            "success": True,
            "message": f"Circuit breaker for {service_name} has been reset",
            "new_status": breaker.get_status(),
        }
    except Exception as e:
        logger.exception("Failed to reset circuit breaker for")
        return {
            "success": False,
            "error": str(e),
        }


async def reset_all_circuit_breakers() -> dict[str, Any]:
    """Reset all circuit breakers."""

    try:
        services = CircuitBreakerRegistry.get_services()
        results = {}

        for service_name in services:
            result = await reset_circuit_breaker(service_name)
            results[service_name] = result

        successful_resets = sum(1 for result in results.values() if result["success"])

        return {
            "success": True,
            "summary": {
                "total_services": len(services),
                "successful_resets": successful_resets,
                "failed_resets": len(services) - successful_resets,
            },
            "results": results,
        }
    except Exception as e:
        logger.exception("Failed to reset all circuit breakers")
        return {
            "success": False,
            "error": str(e),
        }


# Utility Functions for Service Health
async def get_service_health() -> dict[str, Any]:
    """Get health status of all services."""

    try:
        client_manager = get_client_manager()
        health_status = await client_manager.get_health_status()

        # Add circuit breaker status
        circuit_status = await get_circuit_breaker_status()

        return {
            "status": "healthy",
            "services": health_status,
            "circuit_breakers": circuit_status,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {},
            "circuit_breakers": {},
        }


async def get_auto_detected_service_health(
    health_checker: AutoDetectionHealthDep = None,
) -> dict[str, Any]:
    """Get health status of auto-detected services."""

    try:
        if health_checker is None:
            return {
                "status": "no_auto_detection",
                "message": "Auto-detection not configured or no services detected",
                "services": {},
            }

        # Get comprehensive health summary
        health_summary = await health_checker.check_all_services()

        # Get health trends
        trends = health_checker.get_health_trends()

        return {
            "status": "healthy"
            if health_summary.overall_health_score >= 0.8
            else "degraded",
            "summary": {
                "total_services": health_summary.total_services,
                "healthy_services": health_summary.healthy_services,
                "unhealthy_services": health_summary.unhealthy_services,
                "overall_health_score": health_summary.overall_health_score,
                "average_response_time_ms": health_summary.average_response_time_ms,
            },
            "services": {
                result.service_name: {
                    "is_healthy": result.is_healthy,
                    "response_time_ms": result.response_time_ms,
                    "status_code": result.status_code,
                    "error_message": result.error_message,
                    "metadata": result.metadata,
                    "uptime_1h": trends.get(result.service_name, {}).get(
                        "uptime_1h", 0.0
                    ),
                    "uptime_24h": trends.get(result.service_name, {}).get(
                        "uptime_24h", 0.0
                    ),
                }
                for result in health_summary.service_results
            },
            "trends": trends,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Auto-detected service health check failed")
        return {
            "status": "error",
            "error": str(e),
            "services": {},
            "trends": {},
        }


async def get_auto_detection_pool_metrics(
    pool_manager: AutoDetectionPoolsDep = None,
) -> dict[str, Any]:
    """Get metrics for auto-detected service connection pools."""
    try:
        if pool_manager is None:
            return {
                "status": "no_pools",
                "message": "Connection pools not initialized",
                "pools": {},
            }

        # Get comprehensive pool statistics
        pool_stats = pool_manager.get_pool_stats()

        # Get individual pool health metrics
        all_pool_health = pool_manager.get_all_pool_health()

        return {
            "status": "active",
            "summary": {
                "total_pools": pool_stats["total_pools"],
                "healthy_pools": pool_stats["healthy_pools"],
                "pool_types": list(pool_stats["pools"].keys()),
            },
            "pools": {
                pool_name: {
                    **pool_info,
                    "health_metrics": all_pool_health.get(pool_name, {}).model_dump()
                    if all_pool_health.get(pool_name)
                    else None,
                }
                for pool_name, pool_info in pool_stats["pools"].items()
            },
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("Auto-detection pool metrics failed")
        return {
            "status": "error",
            "error": str(e),
            "pools": {},
        }


async def get_auto_detection_summary(
    auto_detected: AutoDetectedServicesDep = None,
    health_checker: AutoDetectionHealthDep = None,
    pool_manager: AutoDetectionPoolsDep = None,
) -> dict[str, Any]:
    """Get comprehensive auto-detection summary with all metrics."""
    summary = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "auto_detection": {},
        "service_health": {},
        "connection_pools": {},
        "overall_status": "unknown",
    }

    try:
        # Inline auto-detection formatting
        if auto_detected:
            summary["auto_detection"] = {
                "environment": {
                    "type": auto_detected.environment.environment_type.value,
                    "is_containerized": auto_detected.environment.is_containerized,
                    "is_kubernetes": auto_detected.environment.is_kubernetes,
                    "cloud_provider": auto_detected.environment.cloud_provider,
                    "region": auto_detected.environment.region,
                    "detection_confidence": (
                        auto_detected.environment.detection_confidence
                    ),
                },
                "services": {
                    "total_discovered": len(auto_detected.services),
                    "services_by_type": {
                        service.service_type: {
                            "host": service.host,
                            "port": service.port,
                            "is_available": service.is_available,
                            "version": service.version,
                            "supports_pooling": service.supports_pooling,
                        }
                        for service in auto_detected.services
                    },
                },
                "detection_performance": {
                    "total_time_ms": auto_detected.total_detection_time_ms,
                    "started_at": auto_detected.detection_started_at,
                    "completed_at": auto_detected.detection_completed_at,
                },
                "errors": auto_detected.errors,
            }

        summary["service_health"] = await get_auto_detected_service_health(
            health_checker
        )
        summary["connection_pools"] = await get_auto_detection_pool_metrics(
            pool_manager
        )

        # Inline status determination
        health_status = summary["service_health"].get("status", "unknown")
        pool_status = summary["connection_pools"].get("status", "unknown")

        if health_status == "healthy" and pool_status in ["active", "no_pools"]:
            summary["overall_status"] = "healthy"
        elif health_status == "degraded" or pool_status == "error":
            summary["overall_status"] = "degraded"
        elif health_status in ["no_auto_detection", "error"]:
            summary["overall_status"] = "unavailable"
        else:
            summary["overall_status"] = "unknown"

        return summary
    except Exception as e:
        logger.exception("Auto-detection summary failed")
        return {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "overall_status": "error",
            "error": str(e),
            "auto_detection": {},
            "service_health": {},
            "connection_pools": {},
        }


# Service Performance Metrics
async def get_service_metrics() -> dict[str, Any]:
    """Get performance metrics for all services."""
    try:
        client_manager = get_client_manager()

        # Collect metrics from various services
        metrics = {
            "embedding_service": {},
            "cache_service": {},
            "crawl_service": {},
            "task_queue": {},
        }

        # Get cache metrics if available
        try:
            cache_manager = await client_manager.get_cache_manager()
            cache_stats = await cache_manager.get_performance_stats()
            metrics["cache_service"] = cache_stats
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.debug("Cache metrics unavailable", exc_info=e)

        # Get crawl metrics if available
        try:
            crawl_manager = await client_manager.get_crawl_manager()
            if crawl_manager:
                crawl_metrics = crawl_manager.get_tier_metrics()  # type: ignore[attr-defined]
                metrics["crawl_service"] = crawl_metrics
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.debug("Crawl metrics unavailable", exc_info=e)

    except Exception as e:
        logger.exception("Metrics collection failed")
        return {"error": str(e)}
    else:
        return metrics


# Cleanup Function
async def cleanup_services() -> None:
    """Cleanup all services and release resources."""
    try:
        client_manager = get_client_manager()
        await client_manager.cleanup()
        logger.info("All services cleaned up successfully")
    except (ConnectionError, OSError, PermissionError):
        logger.exception("Service cleanup failed")
        raise


# ============================================================================
# CRAWLING OPERATIONS
# ============================================================================
# Note: Redis operations removed - use cache_manager functions instead
class BulkCrawlRequest(BaseModel):
    """Pydantic model for bulk crawling requests."""

    urls: list[str]
    preferred_provider: str | None = None
    max_concurrent: int = 5


async def bulk_scrape_urls(
    request: BulkCrawlRequest,
    crawl_manager: CrawlManagerDep,
) -> list[dict[str, Any]]:
    """Scrape multiple URLs concurrently (consolidated implementation)."""
    try:
        # Try built-in bulk_scrape if available
        if hasattr(crawl_manager, "bulk_scrape"):
            return await crawl_manager.bulk_scrape(
                urls=request.urls,
                preferred_provider=request.preferred_provider,
                max_concurrent=request.max_concurrent,
            )

        # Fallback: individual scraping with semaphore
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def scrape_with_semaphore(url: str) -> dict[str, Any]:
            async with semaphore:
                return await crawl_manager.scrape_url(url, request.preferred_provider)

        results = await asyncio.gather(
            *[scrape_with_semaphore(url) for url in request.urls],
            return_exceptions=True,
        )

        # Convert exceptions to error dicts
        processed: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append(
                    {
                        "success": False,
                        "error": f"Scraping failed: {result}",
                        "url": request.urls[i],
                        "content": "",
                        "metadata": {},
                    }
                )
            else:
                processed.append(result)  # type: ignore[arg-type]
        return processed
    except Exception as e:
        logger.exception("Bulk scraping failed")
        msg = f"Failed to scrape URLs: {e}"
        raise CrawlServiceError(msg) from e


async def get_crawl_recommended_tool(
    url: str,
    crawl_manager: CrawlManagerDep,
) -> str:
    """Get recommended tool for a URL based on performance metrics."""
    try:
        return await crawl_manager.get_recommended_tool(url)
    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.warning("Tool recommendation failed for %s: %s", url, e)
        return "crawl4ai"  # Default fallback


async def map_website_urls(
    url: str,
    include_subdomains: bool = False,
    crawl_manager: CrawlManagerDep = None,
) -> dict[str, Any]:
    """Map a website to get list of URLs."""
    try:
        return await crawl_manager.map_url(url, include_subdomains)
    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.warning("URL mapping failed for %s: %s", url, e)
        return {
            "success": False,
            "error": f"URL mapping failed: {e}",
            "urls": [],
            "total": 0,
        }


async def get_crawl_tier_metrics(
    crawl_manager: CrawlManagerDep,
) -> dict[str, dict]:
    """Get performance metrics for all crawling tiers."""
    try:
        return crawl_manager.get_tier_metrics()
    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.warning("Failed to get tier metrics: %s", e)
        return {}


# Direct Monitoring Operations (replacing MonitoringManager)
class HealthCheckRequest(BaseModel):
    """Pydantic model for health check registration."""

    service_name: str
    check_interval: int = 30


class MetricRequest(BaseModel):
    """Pydantic model for metric recording."""

    metric_name: str
    value: float
    labels: dict[str, str] | None = None


# Simple health check tracking without complex Manager state
_health_checks: dict[str, dict[str, Any]] = {}


async def register_health_check_function(
    request: HealthCheckRequest,
    check_function: Callable[[], Any],
) -> dict[str, str]:
    """Register a health check for a service."""
    try:
        _health_checks[request.service_name] = {
            "check_function": check_function,
            "check_interval": request.check_interval,
            "last_check": time.time(),
            "consecutive_failures": 0,
            "last_error": None,
            "state": "healthy",
        }
        logger.info("Registered health check for %s", request.service_name)
    except Exception as e:
        logger.exception("Failed to register health check for %s", request.service_name)
        return {
            "status": "error",
            "message": str(e),
        }
    else:
        return {
            "status": "success",
            "message": f"Health check registered for {request.service_name}",
        }


def _update_health_success(health: dict[str, Any]) -> None:
    """Update health check state for successful check."""
    health["last_check"] = time.time()
    health["state"] = "healthy"
    health["consecutive_failures"] = 0
    health["last_error"] = None


def _update_health_failure(health: dict[str, Any], error_msg: str) -> None:
    """Update health check state for failed check."""
    health["last_check"] = time.time()
    health["last_error"] = error_msg
    health["consecutive_failures"] = health.get("consecutive_failures", 0) + 1
    health["state"] = "degraded" if health["consecutive_failures"] < 3 else "failed"


async def check_service_health_function(service_name: str) -> bool:
    """Check health of a specific service."""
    if service_name not in _health_checks:
        logger.warning("No health check registered for %s", service_name)
        return False

    health = _health_checks[service_name]

    if not health.get("check_function"):
        return False

    try:
        is_healthy = await health["check_function"]()
    except Exception as e:
        logger.exception("Health check failed for %s", service_name)
        _update_health_failure(health, str(e))
        return False
    else:
        if is_healthy:
            _update_health_success(health)
            return is_healthy
        _update_health_failure(health, "Health check returned false")
        return is_healthy


async def get_all_health_status() -> dict[str, dict[str, Any]]:
    """Get health status of all monitored services."""
    status = {}

    for service_name, health in _health_checks.items():
        status[service_name] = {
            "state": health.get("state", "unknown"),
            "last_check": health.get("last_check", 0),
            "last_error": health.get("last_error"),
            "consecutive_failures": health.get("consecutive_failures", 0),
            "is_healthy": health.get("state") == "healthy",
        }

    return status


async def get_overall_health_summary() -> dict[str, Any]:
    """Get overall system health summary."""
    health_status = await get_all_health_status()

    total_services = len(health_status)
    healthy_services = sum(
        1 for status in health_status.values() if status["is_healthy"]
    )
    failed_services = sum(
        1 for status in health_status.values() if status["state"] == "failed"
    )

    overall_healthy = failed_services == 0 and healthy_services == total_services

    return {
        "overall_healthy": overall_healthy,
        "total_services": total_services,
        "healthy_services": healthy_services,
        "failed_services": failed_services,
        "health_percentage": (healthy_services / max(total_services, 1)) * 100,
        "services": health_status,
    }


# Performance tracking without complex Manager state
async def track_operation_performance(
    operation_name: str,
    operation_func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Track performance of an operation."""
    start_time = time.time()

    try:
        result = await operation_func(*args, **kwargs)
        # Record success metrics (could integrate with metrics system)
        duration_ms = (time.time() - start_time) * 1000
        logger.debug("Operation %s completed in %.2fms", operation_name, duration_ms)
    except Exception:
        # Record failure metrics
        duration_ms = (time.time() - start_time) * 1000
        logger.exception("Operation %s failed in %.2fms", operation_name, duration_ms)
        raise
    else:
        return result


# Direct Embedding Operations (extending existing embedding dependencies)
class RerankerRequest(BaseModel):
    """Pydantic model for reranking requests."""

    query: str
    results: list[dict[str, Any]]


async def rerank_search_results(
    request: RerankerRequest,
    embedding_manager: EmbeddingManagerDep,
) -> list[dict[str, Any]]:
    """Rerank search results using BGE reranker."""
    try:
        return await embedding_manager.rerank_results(request.query, request.results)
    except Exception:
        logger.exception("Result reranking failed: %s")
        # Return original results on failure
        return request.results


async def estimate_embedding_cost(
    texts: list[str],
    provider_name: str | None = None,
    embedding_manager: EmbeddingManagerDep = None,
) -> dict[str, dict[str, float]] | dict[str, dict[str, str | float]]:
    """Estimate embedding generation cost."""
    try:
        return embedding_manager.estimate_cost(texts, provider_name)
    except Exception as e:
        logger.exception("Cost estimation failed")
        return {"error": {"message": str(e), "cost": 0.0}}


async def get_embedding_provider_info(
    embedding_manager: EmbeddingManagerDep,
) -> dict[str, dict[str, Any]]:
    """Get information about available embedding providers."""
    try:
        return embedding_manager.get_provider_info()
    except Exception as e:
        logger.exception("Provider info retrieval failed")
        return {"error": {"message": str(e)}}


async def get_optimal_embedding_provider(
    text_length: int,
    quality_required: bool = False,
    budget_limit: float | None = None,
    embedding_manager: EmbeddingManagerDep = None,
) -> str:
    """Select optimal provider based on constraints."""
    try:
        return await embedding_manager.get_optimal_provider(
            text_length, quality_required, budget_limit
        )
    except Exception:
        logger.exception("Optimal provider selection failed")
        # Return reasonable default
        return "fastembed" if text_length > 10000 else "openai"


class TextAnalysisRequest(BaseModel):
    """Pydantic model for text analysis requests."""

    texts: list[str]


async def analyze_text_characteristics(
    request: TextAnalysisRequest,
    embedding_manager: EmbeddingManagerDep,
) -> dict[str, Any]:
    """Analyze text characteristics for smart model selection."""
    try:
        analysis = embedding_manager.analyze_text_characteristics(request.texts)
        # Convert TextAnalysis to dict for service boundary
        analysis_result = {
            "total_length": analysis.total_length,
            "avg_length": analysis.avg_length,
            "complexity_score": analysis.complexity_score,
            "estimated_tokens": analysis.estimated_tokens,
            "text_type": analysis.text_type,
            "requires_high_quality": analysis.requires_high_quality,
        }
    except Exception as e:
        logger.exception("Text analysis failed")
        return {
            "total_length": sum(len(text) for text in request.texts),
            "avg_length": sum(len(text) for text in request.texts) / len(request.texts),
            "complexity_score": 0.5,
            "estimated_tokens": sum(len(text.split()) for text in request.texts),
            "text_type": "general",
            "requires_high_quality": False,
            "error": str(e),
        }
    else:
        return analysis_result


async def get_embedding_usage_report(
    embedding_manager: EmbeddingManagerDep,
) -> dict[str, Any]:
    """Get comprehensive embedding usage report."""
    try:
        return embedding_manager.get_usage_report()
    except Exception as e:
        logger.exception("Usage report generation failed")
        return {"error": str(e)}
