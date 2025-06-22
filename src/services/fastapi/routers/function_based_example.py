"""Example FastAPI router demonstrating function-based dependency injection.

This router shows how to use the new function-based services
that replace the 50+ Manager classes.
"""

import logging
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status
from pydantic import BaseModel
from src.services.fastapi.dependencies import CacheManagerDep
from src.services.fastapi.dependencies import ConfigDep  # Direct Dependencies
from src.services.fastapi.dependencies import CrawlManagerDep
from src.services.fastapi.dependencies import CrawlRequest
from src.services.fastapi.dependencies import CrawlResponse
from src.services.fastapi.dependencies import EmbeddingManagerDep
from src.services.fastapi.dependencies import EmbeddingRequest  # Pydantic Models
from src.services.fastapi.dependencies import EmbeddingResponse
from src.services.fastapi.dependencies import TaskRequest
from src.services.fastapi.dependencies import cache_get
from src.services.fastapi.dependencies import cache_set
from src.services.fastapi.dependencies import crawl_site
from src.services.fastapi.dependencies import enqueue_task
from src.services.fastapi.dependencies import generate_embeddings  # Service Functions
from src.services.fastapi.dependencies import get_service_health
from src.services.fastapi.dependencies import get_service_metrics
from src.services.fastapi.dependencies import get_task_status
from src.services.fastapi.dependencies import scrape_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2", tags=["Function-Based Services"])


# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    services: dict[str, Any]
    timestamp: str | None = None
    error: str | None = None


class MetricsResponse(BaseModel):
    """Metrics response model."""
    embedding_service: dict[str, Any] = {}
    cache_service: dict[str, Any] = {}
    crawl_service: dict[str, Any] = {}
    task_queue: dict[str, Any] = {}
    error: str | None = None


class CacheOperationRequest(BaseModel):
    """Cache operation request model."""
    key: str
    value: Any | None = None
    cache_type: str = "crawl"
    ttl: int | None = None


class TaskStatusResponse(BaseModel):
    """Task status response model."""
    status: str
    job_id: str
    function: str | None = None
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    enqueue_time: str | None = None
    start_time: str | None = None
    finish_time: str | None = None
    result: Any | None = None
    error: str | None = None
    message: str | None = None


# Health and Metrics Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Get health status of all services.

    This endpoint demonstrates function-based health checking
    without needing Manager class instances.
    """
    try:
        health_data = await get_service_health()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {e}"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get performance metrics for all services.

    This endpoint demonstrates function-based metrics collection
    without needing Manager class instances.
    """
    try:
        metrics_data = await get_service_metrics()
        return MetricsResponse(**metrics_data)
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return MetricsResponse(error=str(e))


# Embedding Service Endpoints
@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces EmbeddingManager.generate_embeddings().
    """
    try:
        return await generate_embeddings(request)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {e}"
        )


@router.get("/embeddings/info")
async def get_embedding_info(
    embedding_manager: EmbeddingManagerDep,
) -> dict[str, Any]:
    """Get embedding provider information.

    This endpoint demonstrates direct dependency injection
    for accessing manager methods when needed.
    """
    try:
        provider_info = embedding_manager.get_provider_info()
        usage_report = embedding_manager.get_usage_report()

        return {
            "providers": provider_info,
            "usage": usage_report,
        }
    except Exception as e:
        logger.error(f"Failed to get embedding info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedding info: {e}"
        )


# Cache Service Endpoints
@router.get("/cache/{key}")
async def get_cache_value(
    key: str,
    cache_type: str = "crawl",
    cache_manager: CacheManagerDep = None,
) -> dict[str, Any]:
    """Get value from cache using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces CacheManager.get().
    """
    try:
        value = await cache_get(key, cache_type, cache_manager)
        return {
            "key": key,
            "value": value,
            "cache_type": cache_type,
            "found": value is not None,
        }
    except Exception as e:
        logger.error(f"Cache get failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache get failed: {e}"
        )


@router.post("/cache")
async def set_cache_value(
    request: CacheOperationRequest,
    cache_manager: CacheManagerDep = None,
) -> dict[str, Any]:
    """Set value in cache using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces CacheManager.set().
    """
    try:
        success = await cache_set(
            request.key,
            request.value,
            request.cache_type,
            request.ttl,
            cache_manager
        )
        return {
            "key": request.key,
            "cache_type": request.cache_type,
            "success": success,
        }
    except Exception as e:
        logger.error(f"Cache set failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache set failed: {e}"
        )


@router.get("/cache/stats")
async def get_cache_stats(
    cache_manager: CacheManagerDep,
) -> dict[str, Any]:
    """Get cache statistics.

    This endpoint demonstrates direct dependency injection
    for accessing manager methods when needed.
    """
    try:
        stats = await cache_manager.get_stats()
        performance_stats = await cache_manager.get_performance_stats()

        return {
            "stats": stats,
            "performance": performance_stats,
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {e}"
        )


# Crawl Service Endpoints
@router.post("/crawl/url", response_model=CrawlResponse)
async def scrape_single_url(request: CrawlRequest) -> CrawlResponse:
    """Scrape a single URL using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces CrawlManager.scrape_url().
    """
    try:
        return await scrape_url(request)
    except Exception as e:
        logger.error(f"URL scraping failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL scraping failed: {e}"
        )


@router.post("/crawl/site")
async def crawl_website(request: CrawlRequest) -> dict[str, Any]:
    """Crawl an entire website using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces CrawlManager.crawl_site().
    """
    try:
        return await crawl_site(request)
    except Exception as e:
        logger.error(f"Website crawling failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Website crawling failed: {e}"
        )


@router.get("/crawl/metrics")
async def get_crawl_metrics(
    crawl_manager: CrawlManagerDep,
) -> dict[str, Any]:
    """Get crawling metrics.

    This endpoint demonstrates direct dependency injection
    for accessing manager methods when needed.
    """
    try:
        metrics = crawl_manager.get_metrics()
        tier_metrics = crawl_manager.get_tier_metrics()
        provider_info = crawl_manager.get_provider_info()

        return {
            "metrics": metrics,
            "tier_metrics": tier_metrics,
            "provider_info": provider_info,
        }
    except Exception as e:
        logger.error(f"Failed to get crawl metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get crawl metrics: {e}"
        )


# Task Queue Endpoints
@router.post("/tasks")
async def create_task(request: TaskRequest) -> dict[str, Any]:
    """Enqueue a task using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces TaskQueueManager.enqueue().
    """
    try:
        job_id = await enqueue_task(request)
        if job_id:
            return {
                "job_id": job_id,
                "task_name": request.task_name,
                "status": "enqueued",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue task"
            )
    except Exception as e:
        logger.error(f"Task enqueue failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task enqueue failed: {e}"
        )


@router.get("/tasks/{job_id}", response_model=TaskStatusResponse)
async def get_task_status_endpoint(job_id: str) -> TaskStatusResponse:
    """Get task status using function-based dependency injection.

    This endpoint demonstrates the new function-based approach
    that replaces TaskQueueManager.get_job_status().
    """
    try:
        status_data = await get_task_status(job_id)
        return TaskStatusResponse(**status_data)
    except Exception as e:
        logger.error(f"Task status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task status check failed: {e}"
        )


# Configuration Endpoint
@router.get("/config")
async def get_current_config(config: ConfigDep) -> dict[str, Any]:
    """Get current configuration using function-based dependency injection.

    This endpoint demonstrates config injection without
    needing a ConfigManager class.
    """
    return {
        "embedding_provider": config.embedding_provider.value,
        "cache_enabled": config.cache.enable_caching,
        "crawl_providers": {
            "firecrawl_enabled": bool(config.firecrawl.api_key),
            "openai_enabled": bool(config.openai.api_key),
        },
        "performance": {
            "max_retries": config.performance.max_retries,
            "timeout_seconds": config.performance.timeout_seconds,
        },
    }


# Demonstration of complex function composition
@router.post("/demo/full-pipeline")
async def full_pipeline_demo(
    url: str,
    texts: list[str],
    cache_key: str,
    embedding_manager: EmbeddingManagerDep,
    cache_manager: CacheManagerDep,
    crawl_manager: CrawlManagerDep,
) -> dict[str, Any]:
    """Demonstrate complex pipeline using function-based dependency injection.

    This endpoint shows how multiple services can be composed
    without complex Manager class orchestration.
    """
    try:
        # Step 1: Scrape URL
        crawl_request = CrawlRequest(url=url)
        crawl_result = await scrape_url(crawl_request, crawl_manager)

        # Step 2: Generate embeddings for provided texts
        embedding_request = EmbeddingRequest(
            texts=texts,
            auto_select=True,
            speed_priority=False,
        )
        embedding_result = await generate_embeddings(embedding_request, embedding_manager)

        # Step 3: Cache the results
        cache_data = {
            "crawl_result": crawl_result.dict(),
            "embedding_result": embedding_result.dict(),
            "pipeline_success": True,
        }

        cache_success = await cache_set(
            cache_key,
            cache_data,
            "crawl",
            3600,  # 1 hour TTL
            cache_manager
        )

        return {
            "pipeline_id": cache_key,
            "steps_completed": 3,
            "crawl_success": crawl_result.success,
            "embedding_count": len(embedding_result.embeddings),
            "cache_success": cache_success,
            "total_cost": embedding_result.cost,
            "total_time_ms": crawl_result.automation_time_ms + embedding_result.latency_ms,
        }

    except Exception as e:
        logger.error(f"Pipeline demo failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline demo failed: {e}"
        )
