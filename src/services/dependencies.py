"""FastAPI dependency injection functions replacing Manager classes.

This module provides function-based dependency injection for services,
replacing the 50+ Manager classes with clean, testable functions.
Achieves 60% complexity reduction while maintaining full functionality.
"""

import logging
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends
from pydantic import BaseModel

from src.config import Config, get_config, get_config_with_auto_detection
from src.config.auto_detect import AutoDetectedServices, DetectedEnvironment
from src.config.enums import CacheType, Environment
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


logger = logging.getLogger(__name__)

# Configuration Dependencies
ConfigDep = Annotated[Config, Depends(get_config)]


async def get_auto_detected_config() -> Config:
    """Get configuration with auto-detection applied.

    This async dependency performs service auto-detection and applies
    discovered services to the configuration with proper precedence.
    """
    return await get_config_with_auto_detection()


AutoDetectedConfigDep = Annotated[Config, Depends(get_auto_detected_config)]


@lru_cache
def get_client_manager() -> ClientManager:
    """Get singleton ClientManager instance.

    Uses @lru_cache for singleton pattern with automatic cleanup.
    Replaces ClientManager class singleton complexity.
    """
    return ClientManager.from_unified_config()


async def get_auto_detected_client_manager() -> ClientManager:
    """Get ClientManager instance with auto-detection applied.

    This async dependency creates a ClientManager with service auto-detection
    enabled, allowing automatic discovery and configuration of Redis, Qdrant,
    and PostgreSQL services in the environment.
    """
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
    """Get auto-detected services information.

    Returns the auto-detection results including discovered services
    and environment information. Protected by circuit breaker for
    external metadata API failures.
    """
    auto_detected = config.get_auto_detected_services()
    if auto_detected is None:
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
    """Get Redis client using auto-detected configuration when available.

    Returns Redis client configured with auto-detected service parameters
    including Redis 8.2 RESP3 protocol optimizations and connection pooling.
    """
    return await client_manager.get_redis_client()


async def get_auto_detected_qdrant_client(
    client_manager: AutoDetectedClientManagerDep,
) -> Any:
    """Get Qdrant client using auto-detected configuration when available.

    Returns Qdrant client configured with auto-detected service parameters
    including gRPC optimization when available.
    """
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
    return await client_manager.get_task_queue_manager()


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
    """Get auto-detection health checker for monitoring services.

    Provides health monitoring for auto-detected services.
    Protected by circuit breaker for service health check failures.
    """
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
    """Get connection pool manager for auto-detected services.

    Provides optimized connection pools for discovered services.
    Protected by circuit breaker for connection pool initialization failures.
    """
    # ConnectionPoolManager imported at top-level

    pool_manager = ConnectionPoolManager(config.auto_detection)

    # Initialize pools with auto-detected services if available
    auto_detected = config.get_auto_detected_services()
    if auto_detected and auto_detected.services:
        await pool_manager.initialize_pools(auto_detected.services)

    return pool_manager


AutoDetectionPoolsDep = Annotated[Any, Depends(get_auto_detection_connection_pools)]


class AutoDetectionRequest(BaseModel):
    """Pydantic model for auto-detection requests."""

    force_refresh: bool = False
    timeout_seconds: float | None = None
    enabled_services: list[str] | None = None


class AutoDetectionResponse(BaseModel):
    """Pydantic model for auto-detection responses."""

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
    """Perform service auto-detection on demand.

    Function-based auto-detection that can be called from API endpoints.
    Protected by Tenacity-powered circuit breaker with retry logic.
    """
    try:
        # All imports moved to top-level

        start_time = time.time()

        # Override config if request specifies timeout
        detection_config = config.auto_detection
        if request.timeout_seconds:
            detection_config.timeout_seconds = request.timeout_seconds

        # Perform environment detection
        env_detector = EnvironmentDetector(detection_config)
        detected_env = await env_detector.detect()

        # Perform service discovery
        service_discovery = ServiceDiscovery(detection_config)
        discovery_result = await service_discovery.discover_all_services()

        # Filter services if requested
        services = discovery_result.services
        if request.enabled_services:
            services = [
                s for s in services if s.service_type in request.enabled_services
            ]

        detection_time_ms = (time.time() - start_time) * 1000

        # Format services for response
        formatted_services = [
            {
                "service_name": service.service_name,
                "service_type": service.service_type,
                "host": service.host,
                "port": service.port,
                "is_available": service.is_available,
                "connection_string": service.connection_string,
                "version": service.version,
                "supports_pooling": service.supports_pooling,
                "detection_time_ms": service.detection_time_ms,
                "metadata": service.metadata,
            }
            for service in services
        ]

        return AutoDetectionResponse(
            services_found=len(services),
            environment_type=detected_env.environment_type.value,
            detection_time_ms=detection_time_ms,
            services=formatted_services,
            errors=discovery_result.errors,
            is_cached=False,
        )

    except Exception:
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
    """Pydantic model for embedding generation requests."""

    texts: list[str]
    quality_tier: str | None = None
    provider_name: str | None = None
    max_cost: float | None = None
    speed_priority: bool = False
    auto_select: bool = True
    generate_sparse: bool = False


class EmbeddingResponse(BaseModel):
    """Pydantic model for embedding generation responses."""

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
    """Get initialized EmbeddingManager service.

    Replaces EmbeddingManager.initialize() pattern with dependency injection.
    Protected by circuit breaker for external embedding API failures.
    """
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
    """Generate embeddings with smart provider selection.

    Function-based replacement for EmbeddingManager.generate_embeddings().
    Provides clean interface with Pydantic validation.
    Protected by Tenacity-powered circuit breaker with exponential backoff.
    """
    try:
        # QualityTier imported at top-level

        # Convert string to enum if provided
        quality_tier = None
        if request.quality_tier:
            quality_tier = QualityTier(request.quality_tier)

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

    except Exception:
        logger.exception("Embedding generation failed")
        raise EmbeddingServiceError(f"Failed to generate embeddings: {e}") from e


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
    """Get initialized CacheManager service.

    Replaces CacheManager.initialize() pattern with dependency injection.
    Protected by circuit breaker for Redis connection failures.
    """
    return await client_manager.get_cache_manager()


CacheManagerDep = Annotated[Any, Depends(get_cache_manager)]


async def cache_get(
    key: str,
    cache_type: str = "crawl",
    cache_manager: CacheManagerDep = None,
) -> Any:
    """Get value from cache with L1 -> L2 fallback.

    Function-based replacement for CacheManager.get().
    """
    try:
        # CacheType imported at top-level

        cache_type_enum = CacheType(cache_type)
        return await cache_manager.get(key, cache_type_enum)
    except Exception:
        logger.exception(f"Cache get failed for key {key}")
        return None


async def cache_set(
    key: str,
    value: Any,
    cache_type: str = "crawl",
    ttl: int | None = None,
    cache_manager: CacheManagerDep = None,
) -> bool:
    """Set value in both cache layers.

    Function-based replacement for CacheManager.set().
    """
    try:
        # CacheType imported at top-level

        cache_type_enum = CacheType(cache_type)
        return await cache_manager.set(key, value, cache_type_enum, ttl)
    except Exception:
        logger.exception(f"Cache set failed for key {key}")
        return False


async def cache_delete(
    key: str,
    cache_type: str = "crawl",
    cache_manager: CacheManagerDep = None,
) -> bool:
    """Delete value from both cache layers.

    Function-based replacement for CacheManager.delete().
    """
    try:
        # CacheType imported at top-level

        cache_type_enum = CacheType(cache_type)
        return await cache_manager.delete(key, cache_type_enum)
    except Exception:
        logger.exception(f"Cache delete failed for key {key}")
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
    """Get initialized CrawlManager service.

    Replaces CrawlManager.initialize() pattern with dependency injection.
    Protected by circuit breaker for external crawling API failures.
    """
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
    """Scrape URL with intelligent 5-tier AutomationRouter selection.

    Function-based replacement for CrawlManager.scrape_url().
    Protected by Tenacity-powered circuit breaker with exponential backoff.
    """
    try:
        result = await crawl_manager.scrape_url(
            url=request.url,
            preferred_provider=request.preferred_provider,
        )
        return CrawlResponse(**result)
    except Exception:
        logger.exception(f"URL scraping failed for {request.url}")
        raise CrawlServiceError(f"Failed to scrape URL: {e}") from e


async def crawl_site(
    request: CrawlRequest,
    crawl_manager: CrawlManagerDep,
) -> dict[str, Any]:
    """Crawl entire website from starting URL.

    Function-based replacement for CrawlManager.crawl_site().
    """
    try:
        result = await crawl_manager.crawl_site(
            url=request.url,
            max_pages=request.max_pages,
            preferred_provider=request.preferred_provider,
        )
    except Exception:
        logger.exception(f"Site crawling failed for {request.url}")
        raise CrawlServiceError(f"Failed to crawl site: {e}") from e
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
    """Get initialized TaskQueueManager service.

    Replaces TaskQueueManager.initialize() pattern with dependency injection.
    """
    return await client_manager.get_task_queue_manager()


TaskQueueManagerDep = Annotated[Any, Depends(get_task_queue_manager)]


async def enqueue_task(
    request: TaskRequest,
    task_manager: TaskQueueManagerDep,
) -> str | None:
    """Enqueue a task for execution.

    Function-based replacement for TaskQueueManager.enqueue().
    """
    try:
        job_id = await task_manager.enqueue(
            request.task_name,
            *request.args,
            _delay=request.delay,
            _queue_name=request.queue_name,
            **request.kwargs,
        )
    except Exception:
        logger.exception(f"Task enqueue failed for {request.task_name}")
        raise TaskQueueServiceError(f"Failed to enqueue task: {e}") from e
    else:
        return job_id


async def get_task_status(
    job_id: str,
    task_manager: TaskQueueManagerDep,
) -> dict[str, Any]:
    """Get status of a task.

    Function-based replacement for TaskQueueManager.get_job_status().
    """
    try:
        status = await task_manager.get_job_status(job_id)
    except Exception:
        logger.exception(f"Task status check failed for {job_id}")
        return {"status": "error", "message": str(e)}
    else:
        return status


# Database Dependencies
async def get_database_manager(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized DatabaseManager service.

    Replaces DatabaseManager.initialize() pattern with dependency injection.
    """
    return await client_manager.get_database_manager()


DatabaseManagerDep = Annotated[Any, Depends(get_database_manager)]


async def get_database_session(
    db_manager: DatabaseManagerDep,
) -> AsyncGenerator[Any]:
    """Get database session with automatic cleanup.

    Function-based replacement for DatabaseManager session management.
    Uses yield for automatic resource cleanup.
    """
    async with db_manager.get_session() as session:
        yield session


DatabaseSessionDep = Annotated[Any, Depends(get_database_session)]


# Vector Database Dependencies
@circuit_breaker(
    service_name="qdrant_service",
    failure_threshold=3,
    recovery_timeout=15.0,
    enable_adaptive_timeout=True,
)
async def get_qdrant_service(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized QdrantService.

    Replaces QdrantService.initialize() pattern with dependency injection.
    Protected by circuit breaker for Qdrant database failures.
    """
    return await client_manager.get_qdrant_service()


QdrantServiceDep = Annotated[Any, Depends(get_qdrant_service)]


# HyDE Engine Dependencies
async def get_hyde_engine(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized HyDEEngine service.

    Replaces HyDEEngine.initialize() pattern with dependency injection.
    """
    return await client_manager.get_hyde_engine()


HyDEEngineDep = Annotated[Any, Depends(get_hyde_engine)]


# Content Intelligence Dependencies
async def get_content_intelligence_service(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized ContentIntelligenceService.

    Replaces ContentIntelligenceService.initialize() pattern with dependency injection.
    """
    return await client_manager.get_content_intelligence_service()


ContentIntelligenceServiceDep = Annotated[
    Any, Depends(get_content_intelligence_service)
]


# RAG Service Dependencies
class RAGRequest(BaseModel):
    """Pydantic model for RAG answer generation requests."""

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
    """Pydantic model for RAG answer generation responses."""

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
    """Get initialized RAGGenerator service.

    Replaces RAGGenerator.initialize() pattern with dependency injection.
    Protected by circuit breaker for LLM API failures.
    """
    # RAGGenerator imported at top-level

    generator = RAGGenerator(config.rag, client_manager)
    await generator.initialize()
    return generator


RAGGeneratorDep = Annotated[Any, Depends(get_rag_generator)]


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
    """Generate contextual answer from search results using RAG.

    Function-based replacement for RAGGenerator.generate_answer().
    Provides clean interface with Pydantic validation.
    Protected by Tenacity-powered circuit breaker with exponential backoff.
    """
    try:
        # InternalRAGRequest imported at top-level

        # Convert external request to internal request
        internal_request = InternalRAGRequest(
            query=request.query,
            search_results=request.search_results,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_sources=request.include_sources,
            require_high_confidence=request.require_high_confidence,
            max_context_results=request.max_context_results,
            preferred_source_types=request.preferred_source_types,
            exclude_source_ids=request.exclude_source_ids,
        )

        result = await rag_generator.generate_answer(internal_request)

        # Format sources for response
        sources = None
        if request.include_sources and result.sources:
            sources = [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": source.url,
                    "relevance_score": source.relevance_score,
                    "excerpt": source.excerpt,
                    "position_in_context": source.position_in_context,
                }
                for source in result.sources
            ]

        # Format metrics
        metrics = None
        if result.metrics:
            metrics = {
                "confidence_score": result.metrics.confidence_score,
                "context_utilization": result.metrics.context_utilization,
                "source_diversity": result.metrics.source_diversity,
                "answer_length": result.metrics.answer_length,
                "tokens_used": result.metrics.tokens_used,
                "cost_estimate": result.metrics.cost_estimate,
            }

        return RAGResponse(
            answer=result.answer,
            confidence_score=result.confidence_score,
            sources_used=len(result.sources),
            generation_time_ms=result.generation_time_ms,
            sources=sources,
            metrics=metrics,
            follow_up_questions=result.follow_up_questions,
            cached=result.cached,
            reasoning_trace=result.reasoning_trace,
        )

    except Exception:
        logger.exception("RAG answer generation failed")
        raise EmbeddingServiceError(f"Failed to generate RAG answer: {e}") from e


async def get_rag_metrics(
    rag_generator: RAGGeneratorDep,
) -> dict[str, Any]:
    """Get RAG service performance metrics.

    Function-based replacement for RAGGenerator.get_metrics().
    """
    try:
        metrics = rag_generator.get_metrics()
    except Exception:
        logger.exception("Failed to get RAG metrics")
        return {"error": str(e)}
    else:
        return metrics


async def clear_rag_cache(
    rag_generator: RAGGeneratorDep,
) -> dict[str, str]:
    """Clear RAG answer cache.

    Function-based replacement for RAGGenerator.clear_cache().
    """
    try:
        rag_generator.clear_cache()
    except Exception:
        logger.exception("Failed to clear RAG cache")
        return {"status": "error", "message": str(e)}
    else:
        return {
            "status": "success",
            "message": "RAG answer cache cleared successfully",
        }


# Browser Automation Dependencies
async def get_browser_automation_router(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized BrowserAutomationRouter service.

    Replaces BrowserAutomationRouter.initialize() pattern with dependency injection.
    """
    return await client_manager.get_browser_automation_router()


BrowserAutomationRouterDep = Annotated[Any, Depends(get_browser_automation_router)]


# Project Storage Dependencies
async def get_project_storage(
    client_manager: ClientManagerDep,
) -> Any:
    """Get initialized ProjectStorage service.

    Replaces ProjectStorage.initialize() pattern with dependency injection.
    """
    return await client_manager.get_project_storage()


ProjectStorageDep = Annotated[Any, Depends(get_project_storage)]


# Circuit Breaker Monitoring Functions
async def get_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all circuit breakers.

    Provides real-time monitoring of circuit breaker states, metrics,
    and health across all protected services.
    """
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

        # datetime imported at top-level

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
    except Exception:
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
    """Reset a specific circuit breaker.

    Args:
        service_name: Name of the service circuit breaker to reset

    Returns:
        Reset operation result
    """
    try:
        breaker = CircuitBreakerRegistry.get(service_name)
        if breaker is None:
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
    except Exception:
        logger.exception(f"Failed to reset circuit breaker for {service_name}")
        return {
            "success": False,
            "error": str(e),
        }


async def reset_all_circuit_breakers() -> dict[str, Any]:
    """Reset all circuit breakers.

    Returns:
        Reset operation results for all circuits
    """
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
    except Exception:
        logger.exception("Failed to reset all circuit breakers")
        return {
            "success": False,
            "error": str(e),
        }


# Utility Functions for Service Health
async def get_service_health() -> dict[str, Any]:
    """Get health status of all services.

    Function-based replacement for various health check methods.
    Now includes circuit breaker health information.
    """
    try:
        client_manager = get_client_manager()
        health_status = await client_manager.get_health_status()

        # Add circuit breaker status
        circuit_status = await get_circuit_breaker_status()

        # datetime imported at top-level

        return {
            "status": "healthy",
            "services": health_status,
            "circuit_breakers": circuit_status,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
    except Exception:
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
    """Get health status of auto-detected services.

    Function-based health monitoring for auto-detected services
    with detailed metrics and uptime tracking.
    """
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

        # datetime imported at top-level

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

    except Exception:
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
    """Get metrics for auto-detected service connection pools.

    Function-based pool metrics collection for monitoring
    connection pool performance and utilization.
    """
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

        # datetime imported at top-level

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

    except Exception:
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
    """Get comprehensive auto-detection summary with all metrics.

    Provides complete overview of auto-detected services, their health,
    connection pools, and performance metrics.
    """
    try:
        # datetime imported at top-level

        summary = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "auto_detection": {},
            "service_health": {},
            "connection_pools": {},
            "overall_status": "unknown",
        }

        # Get auto-detection information
        if auto_detected:
            summary["auto_detection"] = {
                "environment": {
                    "type": auto_detected.environment.environment_type.value,
                    "is_containerized": auto_detected.environment.is_containerized,
                    "is_kubernetes": auto_detected.environment.is_kubernetes,
                    "cloud_provider": auto_detected.environment.cloud_provider,
                    "region": auto_detected.environment.region,
                    "detection_confidence": auto_detected.environment.detection_confidence,
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

        # Get service health
        summary["service_health"] = await get_auto_detected_service_health(
            health_checker
        )

        # Get connection pool metrics
        summary["connection_pools"] = await get_auto_detection_pool_metrics(
            pool_manager
        )

        # Determine overall status
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

    except Exception:
        logger.exception("Auto-detection summary failed")
        return {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "overall_status": "error",
            "error": str(e),
            "auto_detection": {},
            "service_health": {},
            "connection_pools": {},
        }
    else:
        return summary


# Service Performance Metrics
async def get_service_metrics() -> dict[str, Any]:
    """Get performance metrics for all services.

    Function-based replacement for various metrics collection methods.
    """
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
        except Exception:
            logger.debug("Cache metrics unavailable", exc_info=e)

        # Get crawl metrics if available
        try:
            crawl_manager = await client_manager.get_crawl_manager()
            crawl_metrics = crawl_manager.get_tier_metrics()
            metrics["crawl_service"] = crawl_metrics
        except Exception:
            logger.debug("Crawl metrics unavailable", exc_info=e)

    except Exception:
        logger.exception("Metrics collection failed")
        return {"error": str(e)}
    else:
        return metrics


# Cleanup Function
async def cleanup_services() -> None:
    """Cleanup all services and release resources.

    Function-based replacement for various cleanup methods.
    """
    try:
        client_manager = get_client_manager()
        await client_manager.cleanup()
        logger.info("All services cleaned up successfully")
    except Exception:
        logger.exception("Service cleanup failed")
        raise
