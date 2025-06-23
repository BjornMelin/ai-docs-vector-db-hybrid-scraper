"""RAG service models and configuration."""

from typing import Any

from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) service."""

    # LLM Configuration
    model: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: int = Field(
        default=1000, gt=0, le=4000, description="Maximum response tokens"
    )
    timeout_seconds: float = Field(default=30.0, gt=0, description="Generation timeout")

    # Answer Generation Configuration
    max_context_length: int = Field(
        default=4000, gt=0, description="Max context length in tokens"
    )
    max_results_for_context: int = Field(
        default=5, gt=0, le=20, description="Max search results to include"
    )
    min_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum confidence for answers"
    )

    # Answer Quality Configuration
    include_sources: bool = Field(default=True, description="Include source citations")
    include_confidence_score: bool = Field(
        default=True, description="Include confidence scoring"
    )
    enable_fact_checking: bool = Field(
        default=False, description="Enable basic fact checking"
    )

    # Performance Configuration
    enable_caching: bool = Field(default=True, description="Enable answer caching")
    cache_ttl_seconds: int = Field(default=3600, gt=0, description="Cache TTL")
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing"
    )

    # Portfolio Features
    enable_answer_metrics: bool = Field(
        default=True, description="Track answer quality metrics"
    )
    enable_source_attribution: bool = Field(
        default=True, description="Detailed source attribution"
    )


class SourceAttribution(BaseModel):
    """Source attribution for generated answers."""

    source_id: str = Field(..., description="Unique source identifier")
    title: str = Field(..., description="Source document title")
    url: str | None = Field(None, description="Source URL if available")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance to query"
    )
    excerpt: str = Field(..., description="Relevant excerpt from source")
    position_in_context: int = Field(
        ..., ge=0, description="Position in context window"
    )


class AnswerMetrics(BaseModel):
    """Metrics for generated answers."""

    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Answer confidence"
    )
    context_utilization: float = Field(
        ..., ge=0.0, le=1.0, description="How much context was used"
    )
    source_diversity: float = Field(
        ..., ge=0.0, le=1.0, description="Diversity of sources used"
    )
    answer_length: int = Field(..., ge=0, description="Answer length in characters")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time")
    tokens_used: int = Field(..., ge=0, description="Total tokens consumed")
    cost_estimate: float = Field(..., ge=0.0, description="Estimated cost in USD")


class RAGRequest(BaseModel):
    """Request for RAG answer generation."""

    query: str = Field(..., min_length=1, description="User query")
    search_results: list[dict[str, Any]] = Field(
        ..., description="Search results to use as context"
    )

    # Optional configuration overrides
    max_tokens: int | None = Field(
        None, gt=0, le=4000, description="Override max tokens"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Override temperature"
    )
    include_sources: bool = Field(True, description="Include source citations")

    # Context preferences
    preferred_source_types: list[str] | None = Field(
        None, description="Preferred source types"
    )
    exclude_source_ids: list[str] | None = Field(None, description="Sources to exclude")

    # Quality requirements
    require_high_confidence: bool = Field(
        False, description="Require high confidence answers"
    )
    max_context_results: int | None = Field(
        None, gt=0, le=20, description="Max results for context"
    )


class RAGResult(BaseModel):
    """Result of RAG answer generation."""

    # Generated content
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Answer confidence"
    )

    # Source information
    sources: list[SourceAttribution] = Field(
        default_factory=list, description="Source attributions"
    )
    context_used: str = Field(..., description="Context provided to LLM")

    # Metadata
    query_processed: str = Field(..., description="Processed query")
    generation_time_ms: float = Field(..., ge=0.0, description="Generation time")

    # Quality metrics
    metrics: AnswerMetrics | None = Field(None, description="Answer quality metrics")

    # Status flags
    truncated: bool = Field(False, description="Whether context was truncated")
    cached: bool = Field(False, description="Whether answer was cached")

    # Portfolio showcase features
    reasoning_trace: list[str] | None = Field(
        None, description="Step-by-step reasoning"
    )
    alternative_perspectives: list[str] | None = Field(
        None, description="Alternative viewpoints"
    )
    follow_up_questions: list[str] | None = Field(
        None, description="Suggested follow-up questions"
    )
