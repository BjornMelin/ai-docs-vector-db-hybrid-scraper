import typing

"""RAG (Retrieval-Augmented Generation) MCP tools.

This module provides MCP tools for generating contextual answers from search results
using Large Language Models. Portfolio-worthy implementation showcasing advanced
AI integration patterns.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field

from src.config import get_config

from ...services.rag import RAGGenerator
from ...services.rag.models import RAGRequest

logger = logging.getLogger(__name__)


class RAGAnswerRequest(BaseModel):
    """Request for RAG answer generation."""

    query: str = Field(..., min_length=1, description="User query to answer")
    search_results: list[dict[str, Any]] = Field(
        ..., description="Search results to use as context"
    )

    # Optional configuration
    max_tokens: int | None = Field(
        None, gt=0, le=4000, description="Maximum response tokens"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Generation temperature"
    )
    include_sources: bool = Field(True, description="Include source citations")
    require_high_confidence: bool = Field(
        False, description="Require high confidence answers"
    )
    max_context_results: int | None = Field(
        None, gt=0, le=20, description="Max results for context"
    )


class RAGAnswerResponse(BaseModel):
    """Response from RAG answer generation."""

    answer: str = Field(..., description="Generated contextual answer")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Answer confidence"
    )
    sources_used: int = Field(..., ge=0, description="Number of sources used")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time")

    # Optional detailed information
    sources: list[dict[str, Any]] | None = Field(None, description="Source details")
    metrics: dict[str, Any] | None = Field(None, description="Answer quality metrics")
    follow_up_questions: list[str] | None = Field(
        None, description="Suggested follow-up questions"
    )


class RAGMetricsResponse(BaseModel):
    """RAG service metrics response."""

    generation_count: int = Field(..., ge=0, description="Total generations")
    avg_generation_time: float = Field(
        ..., ge=0.0, description="Average generation time (ms)"
    )
    total_cost: float = Field(..., ge=0.0, description="Total estimated cost (USD)")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")

    # Performance metrics
    avg_confidence: float | None = Field(None, description="Average confidence score")
    avg_tokens_used: float | None = Field(
        None, description="Average tokens per generation"
    )


def register_tools(app: FastMCP) -> None:
    """Register RAG tools with FastMCP app."""

    @app.tool()
    async def generate_rag_answer(request: RAGAnswerRequest) -> RAGAnswerResponse:
        """Generate a contextual answer from search results using RAG.

        This tool uses advanced Large Language Model capabilities to generate
        contextual, accurate answers based on provided search results. Features
        include source attribution, confidence scoring, and quality metrics.

        Args:
            request: RAG answer generation request

        Returns:
            RAGAnswerResponse: Generated answer with metadata and sources

        Raises:
            ValueError: If request is invalid
            RuntimeError: If RAG generation fails
        """
        config = get_config()

        # Check if RAG is enabled
        if not config.rag.enable_rag:
            raise RuntimeError("RAG is not enabled in the configuration")

        if not request.search_results:
            raise ValueError("Search results are required for RAG answer generation")

        try:
            # Initialize RAG generator
            rag_generator = RAGGenerator(config.rag)
            await rag_generator.initialize()

            # Create RAG request
            rag_request = RAGRequest(
                query=request.query,
                search_results=request.search_results,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                include_sources=request.include_sources,
                require_high_confidence=request.require_high_confidence,
                max_context_results=request.max_context_results,
            )

            # Generate answer
            result = await rag_generator.generate_answer(rag_request)

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

            # Cleanup
            await rag_generator.cleanup()

            return RAGAnswerResponse(
                answer=result.answer,
                confidence_score=result.confidence_score,
                sources_used=len(result.sources),
                generation_time_ms=result.generation_time_ms,
                sources=sources,
                metrics=metrics,
                follow_up_questions=result.follow_up_questions,
            )

        except Exception as e:
            logger.exception(f"RAG answer generation failed: {e}")
            raise RuntimeError(f"Failed to generate RAG answer: {e}") from e

    @app.tool()
    async def get_rag_metrics() -> RAGMetricsResponse:
        """Get RAG service performance metrics.

        Returns comprehensive metrics about RAG answer generation including
        performance statistics, cost estimates, and quality measures.

        Returns:
            RAGMetricsResponse: Service metrics and statistics

        Raises:
            RuntimeError: If metrics retrieval fails
        """
        config = get_config()

        if not config.rag.enable_rag:
            raise RuntimeError("RAG is not enabled in the configuration")

        try:
            # Initialize RAG generator to get metrics
            rag_generator = RAGGenerator(config.rag)
            metrics = rag_generator.get_metrics()

            return RAGMetricsResponse(
                generation_count=metrics["generation_count"],
                avg_generation_time=metrics["avg_generation_time"],
                total_cost=metrics["total_cost"],
                cache_hit_rate=metrics["cache_hit_rate"],
                avg_confidence=None,  # Would need to track this separately
                avg_tokens_used=(
                    metrics["total_tokens_used"] / metrics["generation_count"]
                    if metrics["generation_count"] > 0
                    else 0.0
                ),
            )

        except Exception as e:
            logger.exception(f"Failed to get RAG metrics: {e}")
            raise RuntimeError(f"Failed to get RAG metrics: {e}") from e

    @app.tool()
    async def test_rag_configuration() -> dict[str, Any]:
        """Test RAG configuration and connectivity.

        Validates that RAG is properly configured and can connect to required
        services (LLM providers, etc.). Useful for troubleshooting and setup.

        Returns:
            dict[str, Any]: Configuration test results

        Raises:
            RuntimeError: If configuration test fails
        """
        config = get_config()

        results = {
            "rag_enabled": config.rag.enable_rag,
            "model": config.rag.model,
            "max_tokens": config.rag.max_tokens,
            "temperature": config.rag.temperature,
            "max_context_length": config.rag.max_context_length,
            "connectivity_test": False,
            "error": None,
        }

        if not config.rag.enable_rag:
            results["error"] = "RAG is not enabled in configuration"
            return results

        try:
            # Test RAG generator initialization
            rag_generator = RAGGenerator(config.rag)
            await rag_generator.initialize()

            results["connectivity_test"] = True
            results["openai_client_available"] = rag_generator._llm_client is not None

            # Cleanup
            await rag_generator.cleanup()

        except Exception as e:
            results["error"] = str(e)
            logger.exception(f"RAG configuration test failed: {e}")

        return results

    @app.tool()
    async def clear_rag_cache() -> dict[str, str]:
        """Clear the RAG answer cache.

        Clears all cached RAG answers to free memory or force regeneration
        of answers with updated context or configuration.

        Returns:
            dict[str, str]: Cache clearing results

        Raises:
            RuntimeError: If cache clearing fails
        """
        config = get_config()

        if not config.rag.enable_rag:
            raise RuntimeError("RAG is not enabled in the configuration")

        try:
            # Create RAG generator instance and clear cache
            rag_generator = RAGGenerator(config.rag)
            rag_generator.clear_cache()

            return {
                "status": "success",
                "message": "RAG answer cache cleared successfully",
            }

        except Exception as e:
            logger.exception(f"Failed to clear RAG cache: {e}")
            raise RuntimeError(f"Failed to clear RAG cache: {e}") from e

    logger.info("RAG MCP tools registered successfully")
