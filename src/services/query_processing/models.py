"""Query processing data models for the retrieval pipeline.

This module defines the core data models used throughout the query processing
pipeline, including request and response structures for search operations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.contracts.retrieval import SearchRecord
from src.models import SearchRequest


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
