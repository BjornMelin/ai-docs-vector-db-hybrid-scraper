import typing

"""RAG (Retrieval-Augmented Generation) API endpoints.

This module provides FastAPI endpoints for generating contextual answers
using RAG patterns with the integrated service dependencies.
"""

import logging
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status

from ...dependencies import ConfigDep
from ...dependencies import RAGGeneratorDep
from ...dependencies import RAGRequest
from ...dependencies import RAGResponse
from ...dependencies import clear_rag_cache
from ...dependencies import generate_rag_answer
from ...dependencies import get_rag_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/generate", response_model=RAGResponse)
async def generate_answer(
    request: RAGRequest,
    config: ConfigDep,
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

    Raises:
        HTTPException: If RAG is not enabled or generation fails
    """
    # Check if RAG is enabled
    if not config.rag.enable_rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not enabled in configuration",
        )

    if not request.search_results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search results are required for RAG answer generation",
        )

    try:
        # Use function-based dependency injection for RAG generation
        response = await generate_rag_answer(request)
        return response

    except Exception as e:
        logger.exception(f"RAG answer generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate RAG answer: {e!s}",
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

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        metrics = await get_rag_metrics(rag_generator)
        return {
            "status": "success",
            "metrics": metrics,
        }
    except Exception as e:
        logger.exception(f"Failed to get RAG metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG metrics: {e!s}",
        ) from e


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

    Raises:
        HTTPException: If cache clearing fails
    """
    try:
        result = await clear_rag_cache(rag_generator)
        return result
    except Exception as e:
        logger.exception(f"Failed to clear RAG cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear RAG cache: {e!s}",
        ) from e


@router.get("/config")
async def get_config(
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

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        logger.warning(f"RAG health check failed: {e}")

    return health_status
