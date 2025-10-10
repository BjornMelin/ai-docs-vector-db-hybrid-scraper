"""Retrieval-augmented generation tools for the MCP server.

Provide structured endpoints that build answers from vector search output using the
configured language model stack.
"""

import logging
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.config.loader import Settings, get_settings
from src.services.dependencies import get_rag_generator
from src.services.rag.generator import RAGGenerator
from src.services.rag.models import RAGRequest, RAGResult


logger = logging.getLogger(__name__)


class RAGAnswerRequest(BaseModel):
    """Request for RAG answer generation."""

    query: str = Field(..., min_length=1, description="User query to answer")

    # Optional configuration
    top_k: int | None = Field(
        None, gt=0, le=50, description="Override number of documents retrieved"
    )
    filters: dict[str, Any] | None = Field(
        None, description="Optional metadata filters for retrieval"
    )
    max_tokens: int | None = Field(
        None, gt=0, le=4000, description="Maximum response tokens"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Generation temperature"
    )
    include_sources: bool | None = Field(
        None, description="Override default source citation behaviour"
    )


class RAGAnswerResponse(BaseModel):
    """Response from RAG answer generation."""

    answer: str = Field(..., description="Generated contextual answer")
    confidence_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Answer confidence"
    )
    sources_used: int = Field(..., ge=0, description="Number of sources used")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time")

    # Optional detailed information
    sources: list[dict[str, Any]] | None = Field(None, description="Source details")
    metrics: dict[str, Any] | None = Field(None, description="Answer quality metrics")


class RAGMetricsResponse(BaseModel):
    """RAG service metrics response."""

    generation_count: int = Field(..., ge=0, description="Total generations")
    avg_generation_time_ms: float | None = Field(
        None, ge=0.0, description="Average generation time (ms)"
    )
    total_generation_time_ms: float = Field(
        ..., ge=0.0, description="Aggregate generation time (ms)"
    )


async def _resolve_rag_generator(generator: RAGGenerator | None = None) -> RAGGenerator:
    """Return the RAG generator instance."""

    resolved = await get_rag_generator(generator)
    if not isinstance(resolved, RAGGenerator):
        raise RuntimeError("RAG generator is unavailable")
    return resolved


def _ensure_rag_enabled(config: Settings) -> None:
    """Validate that RAG is enabled in the active configuration."""

    if not config.rag.enable_rag:
        msg = "RAG is not enabled in the configuration"
        raise RuntimeError(msg)


def _build_rag_service_request(request: RAGAnswerRequest) -> RAGRequest:
    """Translate the MCP request model to the service-layer request."""

    return RAGRequest(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        include_sources=request.include_sources,
    )


def _build_rag_answer_response(result: RAGResult) -> RAGAnswerResponse:
    """Convert a RAG generator result into the MCP response schema."""

    sources = None
    if result.sources:
        sources = [source.model_dump() for source in result.sources]

    metrics = result.metrics.model_dump() if result.metrics else None

    return RAGAnswerResponse(
        answer=result.answer,
        confidence_score=result.confidence_score,
        sources_used=len(result.sources),
        generation_time_ms=result.generation_time_ms,
        sources=sources,
        metrics=metrics,
    )


def register_tools(
    app: FastMCP,
    rag_generator_override: RAGGenerator | None = None,
) -> None:
    """Register RAG tools with FastMCP app."""

    @app.tool()
    async def generate_rag_answer(request: RAGAnswerRequest) -> RAGAnswerResponse:
        """Generate a contextual answer from search results using RAG.

        The handler forwards the request to the configured RAG generator and
        collects the answer, source list, and metrics produced by the pipeline.

        Args:
            request: RAG answer generation request

        Returns:
            RAGAnswerResponse: Generated answer with metadata and sources
        """

        config = get_settings()

        _ensure_rag_enabled(config)

        try:
            generator = await _resolve_rag_generator(rag_generator_override)

            rag_request = _build_rag_service_request(request)
            result = await generator.generate_answer(rag_request)

            return _build_rag_answer_response(result)

        except Exception as e:  # pragma: no cover - runtime safety
            logger.exception("RAG answer generation failed")
            msg = "Failed to generate RAG answer"
            raise RuntimeError(msg) from e

    @app.tool()
    async def get_rag_metrics() -> RAGMetricsResponse:
        """Get RAG service performance metrics.

        Returns metrics about RAG answer generation including
        performance stats, cost estimates, and quality measures.

        Returns:
            RAGMetricsResponse: Service metrics and statistics

        Raises:
            RuntimeError: If metrics retrieval fails
        """

        config = get_settings()

        _ensure_rag_enabled(config)

        try:
            generator = await _resolve_rag_generator(rag_generator_override)
            service_metrics = generator.get_metrics()
            return RAGMetricsResponse(
                generation_count=service_metrics.generation_count,
                avg_generation_time_ms=service_metrics.avg_generation_time_ms,
                total_generation_time_ms=service_metrics.total_generation_time_ms,
            )

        except Exception as e:  # pragma: no cover - runtime safety
            logger.exception("Failed to get RAG metrics")
            msg = "Failed to get RAG metrics"
            raise RuntimeError(msg) from e

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

        config = get_settings()

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
            await _resolve_rag_generator(rag_generator_override)
            results["connectivity_test"] = True
        except Exception as e:  # pragma: no cover - runtime safety
            results["error"] = str(e)
            logger.exception("RAG configuration test failed")

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

        config = get_settings()

        if not config.rag.enable_rag:
            msg = "RAG is not enabled in the configuration"
            raise RuntimeError(msg)

        return {
            "status": "noop",
            "message": "RAG generator operates without a local cache",
        }

    logger.info("RAG MCP tools registered successfully")
