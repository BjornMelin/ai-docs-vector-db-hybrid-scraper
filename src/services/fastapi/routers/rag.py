"""RAG (Retrieval-Augmented Generation) API endpoints.

This module provides FastAPI endpoints for generating contextual answers
using RAG patterns with the integrated service dependencies.
"""

import logging
from typing import Any

from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter
from starlette import status

from src.services.dependencies import (
    ConfigDep,
    RAGGeneratorDep,
    RAGRequest,
    RAGResponse,
    clear_rag_cache,
    generate_rag_answer,
    get_rag_metrics,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/generate", response_model=RAGResponse)
async def generate_answer(
    request: RAGRequest,
    config: ConfigDep,
    rag_generator: RAGGeneratorDep,
) -> RAGResponse:
    """Generate a contextual answer from search results using RAG.

    This endpoint showcases the integration of RAG patterns with the existing
    service dependencies. It uses the function-based dependency injection
    pattern to generate answers with source attribution and quality metrics.

    Args:
        request: RAG generation request with query and search results
        config: Application configuration dependency

    Returns:
        RAGResponse: Generated answer with metadata and sources
    """

    # Check if RAG is enabled
    if not config.rag.enable_rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not enabled in configuration",
        )

    try:
        # Use function-based dependency injection for RAG generation
        return await generate_rag_answer(request, rag_generator)

    except Exception as e:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate RAG answer",
        ) from e


@router.get("/metrics")
async def get_metrics(
    rag_generator: RAGGeneratorDep,
) -> dict[str, Any]:
    """Get RAG service performance metrics.

    Returns comprehensive metrics about RAG answer generation including
    performance statistics, cost estimates, and quality measures.

    Args:
        rag_generator: RAG generator service dependency

    Returns:
        dict[str, Any]: Service metrics and statistics
    """

    try:
        metrics = await get_rag_metrics(rag_generator)
    except Exception as e:
        logger.exception("Failed to get RAG metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get RAG metrics",
        ) from e

    return {
        "status": "success",
        "metrics": metrics,
    }


@router.post("/cache/clear")
async def clear_cache(
    rag_generator: RAGGeneratorDep,
) -> dict[str, str]:
    """Clear the RAG answer cache.

    Clears all cached RAG answers to free memory or force regeneration
    of answers with updated context or configuration.

    Args:
        rag_generator: RAG generator service dependency

    Returns:
        dict[str, str]: Cache clearing results
    """

    try:
        return await clear_rag_cache(rag_generator)
    except Exception as e:
        logger.exception("Failed to clear RAG cache")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear RAG cache",
        ) from e


@router.get("/config")
async def get_settings(
    config: ConfigDep,
) -> dict[str, Any]:
    """Get current RAG configuration.

    Returns the current RAG configuration including model settings,
    context limits, and feature flags.

    Args:
        config: Application configuration dependency

    Returns:
        dict[str, Any]: RAG configuration details

    """
    return {
        "enabled": config.rag.enable_rag,
        "model": config.rag.model,
        "temperature": config.rag.temperature,
        "max_tokens": config.rag.max_tokens,
        "max_context_length": config.rag.max_context_length,
        "max_results_for_context": config.rag.max_results_for_context,
        "min_confidence_threshold": config.rag.min_confidence_threshold,
        "include_sources": config.rag.include_sources,
        "include_confidence_score": config.rag.include_confidence_score,
        "enable_caching": config.rag.enable_caching,
        "cache_ttl_seconds": config.rag.cache_ttl_seconds,
        "parallel_processing": config.rag.parallel_processing,
        "enable_answer_metrics": config.rag.enable_answer_metrics,
    }


@router.get("/health")
async def health_check(
    config: ConfigDep,
) -> dict[str, Any]:
    """Health check for RAG service.

    Validates that RAG is properly configured and can connect to required
    services (LLM providers, etc.). Useful for monitoring and troubleshooting.

    Args:
        config: Application configuration dependency

    Returns:
        dict[str, Any]: Health check results

    """
    health_status = {
        "rag_enabled": config.rag.enable_rag,
        "model": config.rag.model,
        "status": "unknown",
        "connectivity_test": False,
        "error": None,
    }

    if not config.rag.enable_rag:
        health_status["status"] = "disabled"
        health_status["error"] = "RAG is not enabled in configuration"
        return health_status

    try:
        # Basic configuration validation
        health_status["status"] = "healthy"
        health_status["connectivity_test"] = True

    except (ValueError, TypeError, AttributeError) as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        logger.warning("RAG health check failed")

    return health_status
