"""Query processing data models for the final retrieval pipeline.

This module defines the core data models used throughout the query processing
pipeline, including request and response structures for search operations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.contracts.retrieval import SearchRecord


class SearchRequest(BaseModel):
    """Parameters controlling a search invocation.

    This model encapsulates all parameters that can be used to customize
    search behavior, including query processing options, result limits,
    filtering, and features like RAG and personalization.

    Attributes:
        query: Search query text (minimum 1 character).
        collection: Target collection identifier (optional).
        limit: Maximum number of results to return (1-1000).
        filters: Structured payload filters (optional).
        enable_expansion: Whether to apply synonym-based expansion.
        enable_personalization: Whether to enable preference-based ranking.
        user_id: User identifier for personalization (optional).
        user_preferences: Preference boosts keyed by category (optional).
        normalize_scores: Whether to normalize result scores per request.
        enable_rag: Whether to generate retrieval-augmented response.
        rag_top_k: Number of documents to fetch for RAG context (optional).
        rag_max_tokens: Maximum tokens for generated answer (optional).
        group_by: Payload field used for server-side grouping.
        group_size: Maximum hits per group (1-10).
        overfetch_multiplier: Server-side overfetch multiplier for grouping.
    """

    query: str = Field(..., min_length=1, description="Search query text")
    collection: str | None = Field(
        default=None,
        min_length=1,
        description="Target collection identifier",
    )
    limit: int = Field(10, ge=1, le=1000, description="Result cap")
    filters: dict[str, Any] | None = Field(
        default=None, description="Structured payload filters"
    )
    enable_expansion: bool = Field(
        default=True, description="Apply synonym-based expansion"
    )
    enable_personalization: bool = Field(
        default=False, description="Enable preference-based ranking"
    )
    user_id: str | None = Field(
        default=None, description="User identifier for personalization"
    )
    user_preferences: dict[str, float] | None = Field(
        default=None, description="Preference boosts keyed by category"
    )
    normalize_scores: bool = Field(
        default=True, description="Normalize result scores per request"
    )
    enable_rag: bool = Field(
        default=False, description="Generate retrieval-augmented response"
    )
    rag_top_k: int | None = Field(
        default=None, ge=1, description="Documents to fetch for RAG context"
    )
    rag_max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens for generated answer"
    )
    group_by: str = Field(
        default="doc_id",
        description="Payload field used for server-side grouping",
    )
    group_size: int = Field(
        default=1, ge=1, le=10, description="Maximum hits per group"
    )
    overfetch_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        description="Server-side overfetch multiplier for grouping",
    )


class SearchResponse(BaseModel):
    """Canonical response structure emitted by the search orchestrator.

    This model contains the complete search response including results,
    metadata about the search process, and optional generated content.

    Attributes:
        records: List of search records returned.
        total_results: Number of records returned.
        query: Processed query text.
        processing_time_ms: Latency in milliseconds.
        expanded_query: Expanded query string when expansion applied (optional).
        features_used: List of features engaged during search.
        grouping_applied: Whether server-side grouping was used.
        generated_answer: Generated RAG answer when requested (optional).
        answer_confidence: Confidence score for generated answer (optional).
        answer_sources: Sources supporting the generated answer (optional).
    """

    records: list[SearchRecord]
    total_results: int = Field(
        default=0, ge=0, description="Number of records returned"
    )
    query: str = Field(..., description="Processed query text")
    processing_time_ms: float = Field(..., ge=0.0, description="Latency in ms")
    expanded_query: str | None = Field(
        default=None, description="Expanded query string when expansion applied"
    )
    features_used: list[str] = Field(
        default_factory=list, description="Features engaged during search"
    )
    grouping_applied: bool = Field(
        default=False, description="Whether server-side grouping was used"
    )
    generated_answer: str | None = Field(
        default=None, description="Generated RAG answer when requested"
    )
    answer_confidence: float | None = Field(
        default=None, description="Confidence score for generated answer"
    )
    answer_sources: list[dict[str, Any]] | None = Field(
        default=None, description="Sources supporting the generated answer"
    )


__all__ = ["SearchRequest", "SearchResponse"]
