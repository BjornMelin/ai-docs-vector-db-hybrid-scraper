"""Advanced Query Processing Models.

This module defines all data models for the advanced query processing pipeline,
including query intent classification, processing requests/responses, and
configuration models for the centralized orchestrator.
"""

import math
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class QueryIntent(str, Enum):
    """Advanced query intent classification categories.

    Expanded from 4 basic categories to 14 total categories for comprehensive
    query understanding and strategy selection.
    """

    # Basic categories (existing)
    CONCEPTUAL = "conceptual"  # High-level understanding questions
    PROCEDURAL = "procedural"  # How-to and step-by-step queries
    FACTUAL = "factual"  # Specific facts and data queries
    TROUBLESHOOTING = "troubleshooting"  # Problem-solving queries

    # Advanced categories (new)
    COMPARATIVE = "comparative"  # Comparison between technologies/concepts
    ARCHITECTURAL = "architectural"  # System design and architecture queries
    PERFORMANCE = "performance"  # Optimization and performance-related queries
    SECURITY = "security"  # Security-focused questions and concerns
    INTEGRATION = "integration"  # API integration and compatibility queries
    BEST_PRACTICES = "best_practices"  # Recommended approaches and patterns
    CODE_REVIEW = "code_review"  # Code analysis and improvement suggestions
    MIGRATION = "migration"  # Upgrade and migration guidance
    DEBUGGING = "debugging"  # Error diagnosis and resolution
    CONFIGURATION = "configuration"  # Setup and configuration assistance


class QueryComplexity(str, Enum):
    """Query complexity levels for adaptive processing."""

    SIMPLE = "simple"  # Straightforward queries with single intent
    MODERATE = "moderate"  # Multi-faceted queries requiring some analysis
    COMPLEX = "complex"  # Advanced queries requiring comprehensive processing
    EXPERT = "expert"  # Highly technical queries for expert-level responses


class SearchStrategy(str, Enum):
    """Available search strategies for query processing."""

    SEMANTIC = "semantic"  # Pure semantic vector search
    HYBRID = "hybrid"  # Dense + sparse vector combination
    HYDE = "hyde"  # Hypothetical Document Embeddings
    MULTI_STAGE = "multi_stage"  # Multi-stage retrieval
    FILTERED = "filtered"  # Filtered search with payload constraints
    RERANKED = "reranked"  # Search with BGE reranking
    ADAPTIVE = "adaptive"  # Adaptive strategy based on query analysis


class MatryoshkaDimension(int, Enum):
    """Available Matryoshka embedding dimensions for dynamic selection."""

    SMALL = 512  # Quick searches, simple queries
    MEDIUM = 768  # Balanced performance/quality
    LARGE = 1536  # Full quality, complex queries


class QueryIntentClassification(BaseModel):
    """Results of advanced query intent classification."""

    primary_intent: QueryIntent = Field(
        ..., description="Primary intent classification"
    )
    secondary_intents: list[QueryIntent] = Field(
        default_factory=list, description="Secondary intents"
    )
    confidence_scores: dict[QueryIntent, float] = Field(
        default_factory=dict, description="Confidence scores for each intent"
    )
    complexity_level: QueryComplexity = Field(
        ..., description="Assessed query complexity"
    )
    domain_category: str | None = Field(
        default=None, description="Detected domain/technology category"
    )
    classification_reasoning: str = Field(
        default="", description="Explanation of classification decision"
    )
    requires_context: bool = Field(
        default=False, description="Whether query requires additional context"
    )
    suggested_followups: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )

    @field_validator("confidence_scores")
    @classmethod
    def validate_confidence_scores(
        cls, v: dict[QueryIntent, float]
    ) -> dict[QueryIntent, float]:
        """Validate confidence scores are between 0 and 1."""
        for intent, score in v.items():
            if not 0.0 <= score <= 1.0:
                msg = f"Confidence score for {intent} must be between 0.0 and 1.0"
                raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_intent_consistency(self) -> "QueryIntentClassification":
        """Ensure primary intent has the highest confidence."""
        if self.confidence_scores:
            primary_confidence = self.confidence_scores.get(self.primary_intent, 0.0)
            max_confidence = (
                max(self.confidence_scores.values()) if self.confidence_scores else 0.0
            )

            if primary_confidence != max_confidence:
                msg = "Primary intent must have the highest confidence score"
                raise ValueError(msg)

        return self

    @computed_field
    @property
    def total_confidence(self) -> float:
        """Sum of all confidence scores."""
        return round(sum(self.confidence_scores.values()), 3)

    @computed_field
    @property
    def confidence_distribution(self) -> dict[str, float]:
        """Normalized confidence distribution."""
        total = self.total_confidence
        if total == 0:
            return {}
        return {
            intent.value: round(score / total, 3)
            for intent, score in self.confidence_scores.items()
        }

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Check if primary intent has high confidence (>0.8)."""
        return self.confidence_scores.get(self.primary_intent, 0.0) > 0.8

    @computed_field
    @property
    def ambiguity_score(self) -> float:
        """Measure of classification ambiguity (entropy-based)."""
        if not self.confidence_scores:
            return 1.0

        scores = list(self.confidence_scores.values())
        total = sum(scores)
        if total == 0:
            return 1.0

        # Calculate entropy
        entropy = -sum(
            (score / total) * math.log2(score / total) for score in scores if score > 0
        )
        # Normalize by max possible entropy
        max_entropy = math.log2(len(scores)) if len(scores) > 1 else 1
        return round(entropy / max_entropy, 3)

    @computed_field
    @property
    def processing_strategy_hint(self) -> str:
        """Suggest processing strategy based on intent and complexity."""
        if self.complexity_level == QueryComplexity.SIMPLE:
            return "direct_search"
        if self.primary_intent in {QueryIntent.COMPARATIVE, QueryIntent.ARCHITECTURAL}:
            return "multi_stage_retrieval"
        if self.primary_intent in {QueryIntent.TROUBLESHOOTING, QueryIntent.DEBUGGING}:
            return "context_enhanced_search"
        if self.requires_context:
            return "hyde_enhanced_search"
        return "hybrid_search"

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class QueryPreprocessingResult(BaseModel):
    """Results of query preprocessing and enhancement."""

    original_query: str = Field(..., description="Original input query")
    processed_query: str = Field(..., description="Processed and enhanced query")
    corrections_applied: list[str] = Field(
        default_factory=list, description="Spelling/grammar corrections applied"
    )
    expansions_added: list[str] = Field(
        default_factory=list, description="Synonym expansions added"
    )
    normalization_applied: bool = Field(
        default=False, description="Whether text normalization was applied"
    )
    context_extracted: dict[str, Any] = Field(
        default_factory=dict, description="Extracted contextual information"
    )
    preprocessing_time_ms: float = Field(
        default=0.0, description="Time spent on preprocessing"
    )

    model_config = ConfigDict(extra="forbid")


class SearchStrategySelection(BaseModel):
    """Strategy selection results for query processing."""

    primary_strategy: SearchStrategy = Field(
        ..., description="Primary search strategy to use"
    )
    fallback_strategies: list[SearchStrategy] = Field(
        default_factory=list, description="Fallback strategies in order of preference"
    )
    matryoshka_dimension: MatryoshkaDimension = Field(
        ..., description="Selected embedding dimension"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in strategy selection"
    )
    reasoning: str = Field(default="", description="Reasoning for strategy selection")
    estimated_quality: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Estimated search quality"
    )
    estimated_latency_ms: float = Field(
        default=100.0, description="Estimated search latency"
    )

    @field_validator("confidence", "estimated_quality")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability values are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            msg = "Value must be between 0.0 and 1.0"
            raise ValueError(msg)
        return round(v, 3)

    @field_validator("estimated_latency_ms")
    @classmethod
    def validate_latency(cls, v: float) -> float:
        """Validate latency is reasonable."""
        if v < 0:
            msg = "Latency cannot be negative"
            raise ValueError(msg)
        if v > 30000:  # 30 seconds
            msg = "Estimated latency exceeds reasonable bounds (30s)"
            raise ValueError(msg)
        return round(v, 1)

    @computed_field
    @property
    def strategy_complexity_score(self) -> float:
        """Calculate complexity score based on strategy."""
        complexity_scores = {
            SearchStrategy.SEMANTIC: 0.2,
            SearchStrategy.HYBRID: 0.5,
            SearchStrategy.HYDE: 0.7,
            SearchStrategy.MULTI_STAGE: 0.8,
            SearchStrategy.FILTERED: 0.6,
            SearchStrategy.RERANKED: 0.9,
            SearchStrategy.ADAPTIVE: 1.0,
        }
        return complexity_scores.get(self.primary_strategy, 0.5)

    @computed_field
    @property
    def fallback_chain_length(self) -> int:
        """Number of fallback strategies available."""
        return len(self.fallback_strategies)

    @computed_field
    @property
    def estimated_accuracy_confidence(self) -> float:
        """Combined score for accuracy confidence."""
        return round((self.confidence * self.estimated_quality), 3)

    @computed_field
    @property
    def performance_category(self) -> str:
        """Categorize performance based on estimated metrics."""
        if self.estimated_latency_ms < 200:
            return "fast"
        if self.estimated_latency_ms < 1000:
            return "moderate"
        return "slow"

    @computed_field
    @property
    def resource_intensity(self) -> str:
        """Estimate resource requirements."""
        intensity = self.strategy_complexity_score
        if self.matryoshka_dimension == MatryoshkaDimension.LARGE:
            intensity += 0.3
        elif self.matryoshka_dimension == MatryoshkaDimension.MEDIUM:
            intensity += 0.1

        if intensity < 0.4:
            return "low"
        if intensity < 0.7:
            return "medium"
        return "high"

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "primary_strategy": "hybrid",
                    "fallback_strategies": ["semantic", "dense"],
                    "matryoshka_dimension": 768,
                    "confidence": 0.85,
                    "reasoning": "Query complexity suggests hybrid approach",
                    "estimated_quality": 0.9,
                    "estimated_latency_ms": 150.0,
                }
            ]
        },
    )


class QueryProcessingRequest(BaseModel):
    """Request for advanced query processing pipeline."""

    query: str = Field(..., description="User query to process", min_length=1)
    collection_name: str = Field(
        default="documents", description="Target collection name"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum results to return"
    )

    # Processing options
    enable_preprocessing: bool = Field(
        default=True, description="Enable query preprocessing"
    )
    enable_intent_classification: bool = Field(
        default=True, description="Enable query intent classification"
    )
    enable_strategy_selection: bool = Field(
        default=True, description="Enable automatic strategy selection"
    )
    enable_matryoshka_optimization: bool = Field(
        default=True, description="Enable Matryoshka dimension optimization"
    )

    # Manual overrides
    force_strategy: SearchStrategy | None = Field(
        default=None, description="Force specific search strategy"
    )
    force_dimension: MatryoshkaDimension | None = Field(
        default=None, description="Force specific embedding dimension"
    )

    # Context and constraints
    user_context: dict[str, Any] = Field(
        default_factory=dict, description="Additional user context"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Search filters to apply"
    )
    search_accuracy: str = Field(
        default="balanced", description="Search accuracy level"
    )
    max_processing_time_ms: int = Field(
        default=5000, description="Maximum processing time budget"
    )

    model_config = ConfigDict(extra="forbid")


class QueryProcessingResponse(BaseModel):
    """Response from advanced query processing pipeline."""

    success: bool = Field(default=True, description="Whether processing succeeded")

    # Core results
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(default=0, description="Total matching documents")

    # Processing insights
    intent_classification: QueryIntentClassification | None = Field(
        default=None, description="Query intent classification results"
    )
    preprocessing_result: QueryPreprocessingResult | None = Field(
        default=None, description="Query preprocessing results"
    )
    strategy_selection: SearchStrategySelection | None = Field(
        default=None, description="Strategy selection results"
    )

    # Performance metrics
    total_processing_time_ms: float = Field(
        default=0.0, description="Total processing time"
    )
    search_time_ms: float = Field(
        default=0.0, description="Time spent on actual search"
    )
    strategy_selection_time_ms: float = Field(
        default=0.0, description="Time spent on strategy selection"
    )

    # Quality indicators
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall confidence in results"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated result quality"
    )

    # Debugging and transparency
    processing_steps: list[str] = Field(
        default_factory=list, description="Steps taken during processing"
    )
    fallback_used: bool = Field(
        default=False, description="Whether fallback strategy was used"
    )
    cache_hit: bool = Field(default=False, description="Whether results were cached")

    # Error handling
    error: str | None = Field(default=None, description="Error message if failed")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")

    model_config = ConfigDict(extra="forbid")


class QueryAnalytics(BaseModel):
    """Analytics data for query processing optimization."""

    query_hash: str = Field(..., description="Hash of the query for tracking")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Processing timestamp"
    )

    # Query characteristics
    query_length: int = Field(default=0, description="Length of query in characters")
    query_word_count: int = Field(default=0, description="Number of words in query")
    detected_language: str | None = Field(
        default=None, description="Detected query language"
    )

    # Processing performance
    total_time_ms: float = Field(default=0.0, description="Total processing time")
    strategy_used: SearchStrategy = Field(..., description="Strategy used")
    dimension_used: MatryoshkaDimension = Field(..., description="Dimension used")

    # Results quality
    results_count: int = Field(default=0, description="Number of results returned")
    average_score: float = Field(default=0.0, description="Average result score")
    user_satisfaction: float | None = Field(
        default=None, description="User satisfaction score if available"
    )

    # A/B testing
    experimental_group: str | None = Field(
        default=None, description="Experimental group for A/B testing"
    )

    model_config = ConfigDict(extra="forbid")


# Export all models
__all__ = [
    "MatryoshkaDimension",
    "QueryAnalytics",
    "QueryComplexity",
    "QueryIntent",
    "QueryIntentClassification",
    "QueryPreprocessingResult",
    "QueryProcessingRequest",
    "QueryProcessingResponse",
    "SearchStrategy",
    "SearchStrategySelection",
]
