"""High-level helpers for interacting with the RAG generator service."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.services.errors import ExternalServiceError, NetworkError
from src.services.rag.models import RAGRequest as InternalRAGRequest
from src.services.service_resolver import get_rag_generator


logger = logging.getLogger(__name__)


class RAGRequest(BaseModel):
    """Public request model accepted by the RAG HTTP endpoints.

    Attributes:
        query: Natural language question submitted by the user.
        search_results: Optional search results that may be reused.
        max_tokens: Maximum number of completion tokens when overridden.
        temperature: Temperature override for the underlying model.
        include_sources: Whether to include source metadata in the response.
        max_context_results: Optional limit on context results considered.
        preferred_source_types: Filter describing preferred source categories.
    """

    query: str = Field(..., min_length=1, description="Natural language question.")
    search_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional search results supplied by the caller.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        le=4000,
        description="Override for the maximum completion tokens.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override for the sampling temperature.",
    )
    include_sources: bool | None = Field(
        default=None,
        description="Whether to include sources in the response.",
    )
    max_context_results: int | None = Field(
        default=None,
        gt=0,
        le=50,
        description="Limit on search results considered for context.",
    )
    preferred_source_types: list[str] | None = Field(
        default=None,
        description="Optional filter applied to selected sources.",
    )


class RAGResponse(BaseModel):
    """Response payload emitted by the RAG HTTP endpoints.

    Attributes:
        answer: Generated answer text returned to the caller.
        confidence_score: Confidence score reported by the generator.
        sources_used: Count of sources referenced in the response.
        generation_time_ms: Time in milliseconds spent generating the answer.
        sources: Optional list of formatted source descriptors.
        metrics: Optional structured metrics emitted by the generator.
        follow_up_questions: Suggested follow-up questions when available.
        cached: Whether the result originated from cache.
        reasoning_trace: Optional reasoning trace returned by the generator.
    """

    answer: str = Field(..., description="Model generated answer text.")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score derived from metadata."
    )
    sources_used: int = Field(
        ..., ge=0, description="Number of source documents referenced."
    )
    generation_time_ms: float = Field(
        ..., ge=0.0, description="Total time spent generating the answer."
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None, description="Formatted source attributions when requested."
    )
    metrics: dict[str, Any] | None = Field(
        default=None, description="Structured metrics reported by the generator."
    )
    follow_up_questions: list[str] | None = Field(
        default=None,
        description="Placeholder for potential follow-up questions.",
    )
    cached: bool = Field(
        default=False,
        description="Indicates whether the response originated from cache.",
    )
    reasoning_trace: list[str] | None = Field(
        default=None,
        description="Optional reasoning trace returned by the generator.",
    )


def _convert_to_internal_request(request: RAGRequest) -> InternalRAGRequest:
    """Translate the public request into the internal generator model.

    Args:
        request: Public request payload received by the API.

    Returns:
        InternalRAGRequest: Request formatted for the generator implementation.
    """
    if request.search_results:
        top_k = request.max_context_results or len(request.search_results)
    else:
        top_k = request.max_context_results

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


def _format_sources(request: RAGRequest, result: Any) -> list[dict[str, Any]] | None:
    """Format raw generator sources into an API-friendly representation.

    Args:
        request: Original request controlling source inclusion.
        result: Generator response containing raw source information.

    Returns:
        list[dict[str, Any]] | None: Formatted source metadata when available.
    """
    include_sources = request.include_sources
    if include_sources is not None and not include_sources:
        return None

    raw_sources = getattr(result, "sources", None) or []
    if not raw_sources:
        return None

    return [
        {
            "source_id": getattr(source, "source_id", ""),
            "title": getattr(source, "title", ""),
            "url": getattr(source, "url", None),
            "relevance_score": getattr(source, "score", None),
            "excerpt": getattr(source, "excerpt", None),
        }
        for source in raw_sources
    ]


def _format_metrics(result: Any) -> dict[str, Any] | None:
    """Convert generator metrics into primitive types.

    Args:
        result: Generator response containing aggregated metrics.

    Returns:
        dict[str, Any] | None: Metrics serialised into primitives if present.
    """
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return None

    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()

    return {
        "total_tokens": getattr(metrics, "total_tokens", None),
        "prompt_tokens": getattr(metrics, "prompt_tokens", None),
        "completion_tokens": getattr(metrics, "completion_tokens", None),
        "generation_time_ms": getattr(metrics, "generation_time_ms", None),
    }


async def generate_rag_answer(
    request: RAGRequest,
    rag_generator: Any | None = None,
) -> RAGResponse:
    """Generate contextual answers using the configured RAG generator.

    Args:
        request: Request payload describing the answer to generate.
        rag_generator: Optional generator instance for dependency injection.

    Returns:
        RAGResponse: Structured response including answer, sources, and metrics.

    Raises:
        ExternalServiceError: If the generator reports a failure.
        NetworkError: If a downstream network issue occurs.
    """
    try:
        generator = rag_generator or await get_rag_generator()
        internal_request = _convert_to_internal_request(request)
        result = await generator.generate_answer(internal_request)

        sources = _format_sources(request, result)
        metrics = _format_metrics(result)
        confidence = getattr(result, "confidence_score", None) or 0.0
        return RAGResponse(
            answer=str(getattr(result, "answer", "")),
            confidence_score=confidence,
            sources_used=len(getattr(result, "sources", []) or []),
            generation_time_ms=float(getattr(result, "generation_time_ms", 0.0)),
            sources=sources,
            metrics=metrics,
            follow_up_questions=None,
            cached=False,
            reasoning_trace=None,
        )
    except (ExternalServiceError, NetworkError):
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("RAG answer generation failed")
        msg = f"Failed to generate RAG answer: {exc}"
        raise ExternalServiceError(msg) from exc


async def get_rag_metrics(rag_generator: Any | None = None) -> dict[str, Any]:
    """Return aggregate metrics for the RAG generator.

    Args:
        rag_generator: Optional generator instance for dependency injection.

    Returns:
        dict[str, Any]: Metrics describing recent generator performance.
    """
    try:
        generator = rag_generator or await get_rag_generator()
        metrics = generator.get_metrics()
        if hasattr(metrics, "model_dump"):
            return metrics.model_dump()
        return metrics
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to retrieve RAG metrics")
        return {"error": str(exc)}


async def clear_rag_cache(rag_generator: Any | None = None) -> dict[str, str]:
    """Clear any cached responses maintained by the RAG generator.

    Args:
        rag_generator: Optional generator instance for dependency injection.

    Returns:
        dict[str, str]: Status payload describing the outcome of the cache clear.
    """
    try:
        generator = rag_generator or await get_rag_generator()
        generator.clear_cache()
        return {"status": "success", "message": "RAG cache cleared"}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to clear RAG cache")
        return {"status": "error", "message": str(exc)}


__all__ = [
    "RAGRequest",
    "RAGResponse",
    "clear_rag_cache",
    "generate_rag_answer",
    "get_rag_metrics",
]
