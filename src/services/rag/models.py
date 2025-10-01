"""Lightweight data models for the LangChain-backed RAG service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Runtime configuration for the LangChain RAG generator."""

    model: str = Field(
        default="gpt-4o-mini",
        description="ChatCompletion model identifier supported by langchain-openai.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to the chat model.",
    )
    max_tokens: int = Field(
        default=600,
        gt=0,
        le=4000,
        description="Upper bound on response tokens for the chat model.",
    )
    retriever_top_k: int = Field(
        default=5,
        gt=0,
        le=50,
        description="Default number of documents to fetch from the retriever.",
    )
    include_sources: bool = Field(
        default=True,
        description="Return structured source attributions alongside the answer.",
    )
    confidence_from_scores: bool = Field(
        default=True,
        description=(
            "If True, derive a heuristic confidence from retrieval scores whenever "
            "a score is supplied by the retriever."
        ),
    )


class SourceAttribution(BaseModel):
    """Source metadata surfaced with the generated answer."""

    source_id: str = Field(..., description="Identifier taken from the search result.")
    title: str = Field(..., description="Human readable source title if available.")
    url: str | None = Field(None, description="Optional source URL.")
    excerpt: str | None = Field(
        None, description="Relevant snippet included in context."
    )
    score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Normalized relevance score when provided by the retrieval layer.",
    )


class AnswerMetrics(BaseModel):
    """Lightweight telemetry returned with each generation."""

    total_tokens: int | None = Field(
        None, description="Total tokens reported by the model, if available."
    )
    prompt_tokens: int | None = Field(
        None, description="Prompt tokens reported by the model, if available."
    )
    completion_tokens: int | None = Field(
        None, description="Completion tokens reported by the model, if available."
    )
    generation_time_ms: float = Field(
        ..., ge=0.0, description="Wall clock generation latency in milliseconds."
    )


class RAGRequest(BaseModel):
    """Incoming request payload for the RAG generator."""

    query: str = Field(..., min_length=1, description="Natural language question.")
    top_k: int | None = Field(
        None,
        gt=0,
        le=50,
        description="Optional override for the number of documents to retrieve.",
    )
    filters: dict[str, Any] | None = Field(
        None,
        description="Optional metadata filters passed to the retriever.",
    )
    max_tokens: int | None = Field(
        None,
        gt=0,
        le=4000,
        description="Optional override for maximum completion tokens.",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Optional override for model sampling temperature.",
    )
    include_sources: bool | None = Field(
        None,
        description="Override default source attribution behaviour.",
    )


class RAGResult(BaseModel):
    """Structured response returned by the RAG generator."""

    answer: str = Field(..., description="Model generated answer text.")
    confidence_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Heuristic confidence derived from retrieval scores.",
    )
    sources: list[SourceAttribution] = Field(
        default_factory=list,
        description="Ordered list of sources referenced in the answer.",
    )
    generation_time_ms: float = Field(
        ..., ge=0.0, description="Total generation latency in milliseconds."
    )
    metrics: AnswerMetrics | None = Field(
        None, description="Optional token usage information from the chat model."
    )


class RAGServiceMetrics(BaseModel):
    """Aggregated metrics emitted by the generator instance."""

    generation_count: int = Field(
        ..., ge=0, description="Total number of answers generated in this instance."
    )
    avg_generation_time_ms: float | None = Field(
        None,
        ge=0.0,
        description="Mean generation latency in milliseconds when available.",
    )
    total_generation_time_ms: float = Field(
        ..., ge=0.0, description="Aggregate generation latency in milliseconds."
    )
