"""
Comprehensive Vector Search Models - Security-First Implementation

This module provides production-ready Pydantic v2 models for vector search operations
with comprehensive security validation, performance optimization, and type safety.

Key Features:
- Security-first design with DoS prevention and injection protection
- Full Pydantic v2 compliance with modern validation patterns
- Type-safe replacements for all dict[str, Any] patterns
- Async-ready design for Qdrant integration
- Performance-optimized with caching and validation strategies
"""

from __future__ import annotations

import math
import re
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


if TYPE_CHECKING:
    from datetime import datetime


# =============================================================================
# BASE CONFIGURATION & ENUMS
# =============================================================================


class SearchAccuracy(str, Enum):
    """Search accuracy levels with performance trade-offs."""

    HIGH = "high"  # Best quality, slower
    MEDIUM = "medium"  # Balanced quality/speed
    LOW = "low"  # Fastest, lower quality
    ADAPTIVE = "adaptive"  # Dynamic based on query


class VectorType(str, Enum):
    """Supported vector types for search operations."""

    DENSE = "dense"  # Standard dense vectors
    SPARSE = "sparse"  # Sparse vectors with indices
    HYBRID = "hybrid"  # Combined dense + sparse
    BINARY = "binary"  # Binary quantized vectors


class FusionAlgorithm(str, Enum):
    """Algorithms for combining multiple search results."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # Weighted score combination
    MAX_SCORE = "max_score"  # Maximum score selection
    ADAPTIVE = "adaptive"  # Context-aware fusion


# =============================================================================
# SECURITY FOUNDATION MODELS
# =============================================================================


class SecureBaseModel(BaseModel):
    """Base model with security-hardened defaults."""

    model_config = ConfigDict(
        # Security settings
        strict=True,
        extra="forbid",
        validate_assignment=True,
        # Performance optimization
        str_strip_whitespace=True,
        str_max_length=5000,  # DoS prevention
        # Error handling
        hide_input_in_errors=True,  # Security: don't leak sensitive data
        # Type safety
        use_enum_values=False,  # Keep enum types for validation
        arbitrary_types_allowed=False,
        frozen=False,  # Allow mutation for performance
    )


class VectorSearchError(Exception):
    """Base exception for vector search operations."""


class SecurityValidationError(VectorSearchError):
    """Raised when security validation fails."""


class DimensionError(VectorSearchError):
    """Raised when vector dimensions are invalid."""


class FilterValidationError(VectorSearchError):
    """Raised when filter validation fails."""


class SearchConfigurationError(VectorSearchError):
    """Raised when search configuration is invalid."""


# =============================================================================
# CORE VECTOR MODELS
# =============================================================================


class SecureVectorModel(SecureBaseModel):
    """Secure vector representation with DoS prevention."""

    values: list[float] = Field(
        ...,
        description="Vector values with security validation",
        min_length=1,
        max_length=4096,  # Industry standard maximum
    )

    @field_validator("values", mode="before")
    @classmethod
    def validate_vector_security(cls, v: list[float]) -> list[float]:
        """Comprehensive vector security validation."""
        if not isinstance(v, list):
            msg = "Vector must be a list of numbers"
            raise TypeError(msg)

        if not v:
            msg = "Vector cannot be empty"
            raise ValueError(msg)

        if len(v) > 4096:
            msg = f"Vector dimensions exceed maximum allowed (4096), got {len(v)} (security: DoS prevention)"
            raise DimensionError(msg)

        def _raise_error(message: str) -> None:
            """Helper to raise ValueError."""
            raise ValueError(message)

        # Convert and validate each value
        validated_values = []
        for i, val in enumerate(v):
            try:
                float_val = float(val)
                if not (-1e6 <= float_val <= 1e6):  # Reasonable bounds
                    _raise_error(
                        f"Vector value at index {i} out of bounds: {float_val}"
                    )

                if math.isnan(float_val):  # NaN check
                    _raise_error(f"Vector contains NaN at index {i}")
                if abs(float_val) == float("inf"):  # Infinity check
                    _raise_error(f"Vector contains infinite value at index {i}")
                validated_values.append(float_val)
            except (ValueError, TypeError) as e:
                _raise_error(f"Invalid vector value at index {i}: {val} - {e}")

        return validated_values

    @computed_field
    @cached_property
    def dimension(self) -> int:
        """Vector dimension (cached for performance)."""
        return len(self.values)

    @computed_field
    @cached_property
    def magnitude(self) -> float:
        """Vector magnitude (cached for performance)."""
        return sum(v * v for v in self.values) ** 0.5


class SecureSparseVectorModel(SecureBaseModel):
    """Secure sparse vector with index validation."""

    indices: list[int] = Field(
        ...,
        description="Sparse vector indices",
        min_length=1,
        max_length=4096,
    )
    values: list[float] = Field(
        ...,
        description="Sparse vector values",
        min_length=1,
        max_length=4096,
    )

    @model_validator(mode="after")
    def validate_sparse_vector(self) -> Self:
        """Validate sparse vector consistency and security."""
        if len(self.indices) != len(self.values):
            msg = "Indices and values must have the same length"
            raise ValueError(msg)

        # Check for duplicate indices
        if len(set(self.indices)) != len(self.indices):
            msg = "Duplicate indices not allowed in sparse vector"
            raise ValueError(msg)

        # Validate indices are non-negative and within bounds
        for _i, idx in enumerate(self.indices):
            if idx < 0:
                msg = f"Negative index not allowed: {idx}"
                raise ValueError(msg)
            if idx >= 100000:  # Reasonable upper bound
                msg = f"Index too large: {idx} (DoS prevention)"
                raise ValueError(msg)

        # Validate values
        for i, val in enumerate(self.values):
            if math.isnan(val):  # NaN check
                msg = f"Sparse vector contains NaN at position {i}"
                raise ValueError(msg)
            if abs(val) == float("inf"):
                msg = f"Sparse vector contains infinite value at position {i}"
                raise ValueError(msg)

        return self


# =============================================================================
# SECURE METADATA & FILTER MODELS
# =============================================================================


class SecureMetadataModel(SecureBaseModel):
    """Secure metadata model replacing dict[str, Any]."""

    title: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=1000)
    source: str | None = Field(None, max_length=500)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tags: list[str] = Field(default_factory=list, max_length=20)
    category: str | None = Field(None, max_length=100)
    version: str | None = Field(None, pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    language: str | None = Field(None, pattern=r"^[a-z]{2}(-[A-Z]{2})?$")

    @field_validator("tags", mode="after")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags for security."""
        for tag in v:
            if len(tag) > 50:
                msg = f"Tag too long: {tag[:20]}..."
                raise ValueError(msg)
            if not re.match(r"^[a-zA-Z0-9_-]+$", tag):
                msg = f"Invalid tag format: {tag}"
                raise ValueError(msg)
        return v


class SecureFilterModel(SecureBaseModel):
    """Secure filter model preventing NoSQL injection."""

    field: str = Field(
        ...,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_.]*$",  # Valid field names only
        max_length=50,
        description="Field name for filtering",
    )
    operator: Literal[
        "eq", "ne", "gt", "lt", "gte", "lte", "in", "nin", "range", "exists", "regex"
    ] = Field(..., description="Filter operator (whitelist only)")
    value: str | int | float | bool | list[str | int | float | bool] | None = Field(
        ..., description="Filter value with type validation"
    )

    @field_validator("value", mode="after")
    @classmethod
    def validate_filter_value(cls, v: Any) -> Any:
        """Validate filter value for security."""
        if v is None:
            return v

        # Handle list values
        if isinstance(v, list):
            if len(v) > 100:  # DoS prevention
                msg = "Filter list too long (max 100 items)"
                raise ValueError(msg)
            for item in v:
                if isinstance(item, str) and len(item) > 500:
                    msg = "Filter string value too long"
                    raise ValueError(msg)
            return v

        # Handle string values
        if isinstance(v, str):
            if len(v) > 500:
                msg = "Filter string value too long"
                raise ValueError(msg)
            # Basic SQL injection prevention
            dangerous_patterns = [
                ";",
                "--",
                "/*",
                "*/",
                "xp_",
                "sp_",
                "DROP",
                "DELETE",
                "INSERT",
                "UPDATE",
            ]
            for pattern in dangerous_patterns:
                if pattern.lower() in v.lower():
                    msg = f"Potentially dangerous pattern in filter value: {pattern}"
                    raise FilterValidationError(msg)

        return v


class SecureFilterGroupModel(SecureBaseModel):
    """Secure filter group for complex queries."""

    operator: Literal["and", "or", "not"] = Field(..., description="Logical operator")
    filters: list[SecureFilterModel | SecureFilterGroupModel] = Field(
        ...,
        min_length=1,
        max_length=50,  # DoS prevention
        description="Nested filters and filter groups",
    )

    @model_validator(mode="after")
    def validate_filter_depth(self) -> Self:
        """Prevent deeply nested filter groups (DoS prevention)."""

        def check_depth(obj: SecureFilterGroupModel, current_depth: int = 0) -> int:
            if current_depth > 10:  # Max nesting depth
                msg = "Filter nesting too deep (max 10 levels)"
                raise FilterValidationError(msg)

            max_child_depth = current_depth
            for f in obj.filters:
                if isinstance(f, SecureFilterGroupModel):
                    child_depth = check_depth(f, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        check_depth(self)
        return self


# =============================================================================
# SEARCH CONFIGURATION MODELS
# =============================================================================


class SecureSearchParamsModel(SecureBaseModel):
    """Secure search parameters with bounds validation."""

    # HNSW parameters
    ef_construct: int = Field(
        default=128, ge=16, le=512, description="HNSW ef_construct parameter"
    )
    ef_search: int = Field(
        default=128, ge=16, le=512, description="HNSW ef_search parameter"
    )
    m: int = Field(default=16, ge=4, le=64, description="HNSW M parameter")

    # Search parameters
    limit: int = Field(
        default=10, ge=1, le=1000, description="Maximum results to return"
    )
    offset: int = Field(
        default=0, ge=0, le=10000, description="Results offset for pagination"
    )
    threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )

    # Performance parameters
    timeout: int = Field(
        default=30, ge=1, le=300, description="Search timeout in seconds"
    )
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch processing size"
    )

    @model_validator(mode="after")
    def validate_hnsw_params(self) -> Self:
        """Validate HNSW parameter relationships."""
        if self.ef_search < self.limit:
            self.ef_search = max(self.limit, 64)  # Auto-adjust for performance

        return self


class SecurePayloadModel(SecureBaseModel):
    """Secure payload model with content validation."""

    content: str = Field(..., max_length=10000, description="Payload content")
    metadata: SecureMetadataModel = Field(default_factory=SecureMetadataModel)  # type: ignore[call-arg]
    vector_type: VectorType = Field(default=VectorType.DENSE)

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_security(cls, v: str) -> str:
        """Validate content for security (lazy import to avoid circular dependency)."""
        try:
            from src.services.security.ai_security import (
                AISecurityValidator,
            )

            validator = AISecurityValidator()
            if not validator.validate_search_query(v):
                msg = "Content failed AI security validation"
                raise SecurityValidationError(msg)
        except ImportError:
            # Fallback validation if AI security is not available
            dangerous_patterns = [
                "<script",
                "javascript:",
                "data:text/html",
                "vbscript:",
            ]
            content_lower = v.lower()
            for pattern in dangerous_patterns:
                if pattern in content_lower:
                    msg = f"Content contains dangerous pattern: {pattern}"
                    raise SecurityValidationError(msg) from None

        return v


# =============================================================================
# SEARCH REQUEST MODELS
# =============================================================================


class BasicSearchRequest(SecureBaseModel):
    """Basic vector search request."""

    query_vector: SecureVectorModel = Field(
        ..., description="Query vector for similarity search"
    )
    search_params: SecureSearchParamsModel = Field(
        default_factory=SecureSearchParamsModel  # type: ignore[call-arg]
    )
    filters: list[SecureFilterModel] = Field(default_factory=list, max_length=20)
    include_metadata: bool = Field(default=True)
    include_vectors: bool = Field(default=False)

    @model_validator(mode="before")
    def set_default_search_params(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        """Set default search_params if not provided."""
        if values.get("search_params") is None:
            values["search_params"] = SecureSearchParamsModel()
        return values


class AdvancedFilteredSearchRequest(BasicSearchRequest):
    """Search request with advanced filtering."""

    filter_groups: list[SecureFilterGroupModel] = Field(
        default_factory=list, max_length=5
    )
    exclude_filters: list[SecureFilterModel] = Field(
        default_factory=list, max_length=10
    )


class AdvancedHybridSearchRequest(AdvancedFilteredSearchRequest):
    """Hybrid search combining dense and sparse vectors."""

    sparse_vector: SecureSparseVectorModel | None = None
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    fusion_algorithm: FusionAlgorithm = Field(default=FusionAlgorithm.RRF)

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """Validate weight normalization."""
        total_weight = self.dense_weight + self.sparse_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            # Auto-normalize weights
            self.dense_weight = self.dense_weight / total_weight
            self.sparse_weight = self.sparse_weight / total_weight

        return self


class HyDESearchRequest(AdvancedFilteredSearchRequest):
    """Hypothetical Document Embeddings search request."""

    query_text: str = Field(..., max_length=2000, description="Natural language query")
    num_hypotheses: int = Field(default=3, ge=1, le=10)
    hypothesis_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query_text", mode="after")
    @classmethod
    def validate_query_security(cls, v: str) -> str:
        """Validate query text for security."""
        try:
            from src.services.security.ai_security import (
                AISecurityValidator,
            )

            validator = AISecurityValidator()
            if not validator.validate_search_query(v):
                msg = "Query failed AI security validation"
                raise SecurityValidationError(msg)
        except ImportError:
            # Basic fallback validation
            if len(v.strip()) < 3:
                msg = "Query too short for meaningful search"
                raise ValueError(msg) from None

        return v


# =============================================================================
# SEARCH RESPONSE MODELS
# =============================================================================


class SecureSearchResult(SecureBaseModel):
    """Individual search result with secure payload."""

    id: str = Field(..., description="Result identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    payload: SecurePayloadModel = Field(..., description="Result payload")
    vector: SecureVectorModel | None = None

    @computed_field
    @property
    def relevance_tier(self) -> Literal["high", "medium", "low"]:
        """Computed relevance tier based on score."""
        if self.score >= 0.8:
            return "high"
        if self.score >= 0.6:
            return "medium"
        return "low"


class SearchResponse(SecureBaseModel):
    """Complete search response with metadata."""

    results: list[SecureSearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., ge=0, description="Total available results")
    search_time_ms: int = Field(..., ge=0, description="Search execution time")
    accuracy: SearchAccuracy = Field(..., description="Achieved search accuracy")

    # Performance metrics
    vector_ops: int = Field(default=0, ge=0, description="Number of vector operations")
    cache_hits: int = Field(default=0, ge=0, description="Cache hit count")

    @computed_field
    @property
    def performance_score(self) -> float:
        """Computed performance score (0-1)."""
        # Score based on search time and cache efficiency
        time_score = max(0, 1 - (self.search_time_ms / 1000))  # Penalty after 1 second
        cache_score = self.cache_hits / max(1, self.vector_ops)  # Cache efficiency
        return (time_score * 0.7) + (cache_score * 0.3)


# =============================================================================
# SEARCH STAGE MODELS (for complex operations)
# =============================================================================


class SearchStage(SecureBaseModel):
    """Individual search stage for multi-stage search operations."""

    stage_name: str = Field(..., max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    query_vector: SecureVectorModel = Field(..., description="Stage query vector")
    search_params: SecureSearchParamsModel = Field(
        default_factory=SecureSearchParamsModel  # type: ignore[call-arg]
    )
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    def set_default_search_params(cls, values: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        """Set default search_params if not provided."""
        if values.get("search_params") is None:
            values["search_params"] = SecureSearchParamsModel()
        return values


class MultiStageSearchRequest(SecureBaseModel):
    """Multi-stage search request for complex operations."""

    stages: list[SearchStage] = Field(..., min_length=1, max_length=10)
    fusion_algorithm: FusionAlgorithm = Field(default=FusionAlgorithm.RRF)
    global_filters: list[SecureFilterModel] = Field(default_factory=list, max_length=20)
    final_limit: int = Field(default=10, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_stage_weights(self) -> Self:
        """Validate and normalize stage weights."""
        total_weight = sum(stage.weight for stage in self.stages)
        if total_weight <= 0:
            msg = "Total stage weight must be positive"
            raise ValueError(msg)

        # Normalize weights to sum to 1.0
        for stage in self.stages:
            stage.weight = stage.weight / total_weight

        return self


# =============================================================================
# ASYNC OPERATION MODELS
# =============================================================================


class AsyncSearchContext(SecureBaseModel):
    """Context for async search operations."""

    request_id: str = Field(..., description="Unique request identifier")
    client_id: str = Field(..., max_length=100, description="Client identifier")
    timeout: int = Field(default=30, ge=1, le=300)
    retry_count: int = Field(default=0, ge=0, le=5)

    @field_validator("client_id", mode="after")
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        """Validate client ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            msg = "Invalid client ID format"
            raise ValueError(msg)
        return v


class BatchSearchRequest(SecureBaseModel):
    """Batch search request for multiple queries."""

    queries: list[
        BasicSearchRequest | AdvancedHybridSearchRequest | HyDESearchRequest
    ] = Field(
        ...,
        min_length=1,
        max_length=100,  # DoS prevention
        description="Batch of search requests",
    )
    context: AsyncSearchContext | None = None
    parallel_execution: bool = Field(default=True)

    @computed_field
    @property
    def total_vectors(self) -> int:
        """Total number of vectors in batch."""
        return len(self.queries)


class BatchSearchResponse(SecureBaseModel):
    """Batch search response."""

    responses: list[SearchResponse] = Field(
        ..., description="Individual search responses"
    )
    batch_time_ms: int = Field(..., ge=0, description="Total batch execution time")
    successful_queries: int = Field(
        ..., ge=0, description="Number of successful queries"
    )
    failed_queries: int = Field(..., ge=0, description="Number of failed queries")

    @computed_field
    @property
    def success_rate(self) -> float:
        """Batch success rate (0-1)."""
        total = self.successful_queries + self.failed_queries
        return self.successful_queries / max(1, total)


# =============================================================================
# PREFETCH CONFIGURATION MODELS
# =============================================================================


class PrefetchConfig(SecureBaseModel):
    """Configuration for prefetch operations with optimized limits."""

    dense_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    sparse_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    hyde_multiplier: float = Field(default=2.5, ge=1.0, le=5.0)
    max_prefetch_limit: int = Field(default=1000, ge=10, le=5000)

    def calculate_prefetch_limit(
        self, vector_type: VectorType, final_limit: int
    ) -> int:
        """Calculate optimal prefetch limit based on vector type and final limit.

        Args:
            vector_type: Type of vector operation
            final_limit: Final result limit

        Returns:
            Optimal prefetch limit for the given vector type

        """
        if vector_type == VectorType.DENSE:
            multiplier = self.dense_multiplier
        elif vector_type == VectorType.SPARSE:
            multiplier = self.sparse_multiplier
        elif hasattr(VectorType, "HYDE") and vector_type == VectorType.HYDE:
            multiplier = self.hyde_multiplier
        else:
            multiplier = self.dense_multiplier  # Default fallback

        prefetch_limit = int(final_limit * multiplier)
        return min(prefetch_limit, self.max_prefetch_limit)


# =============================================================================
# EXPORT DEFINITIONS
# =============================================================================
# MISSING CLASSES FOR BACKWARD COMPATIBILITY
# =============================================================================


class QueryClassification(SecureBaseModel):
    """Query classification result."""

    query_type: str = Field(..., description="Type of query")
    complexity_level: str = Field(..., description="Complexity level")
    domain: str = Field(..., description="Query domain")
    programming_language: str | None = Field(None, description="Programming language")
    is_multimodal: bool = Field(False, description="Whether query is multimodal")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    features: dict[str, Any] = Field(default_factory=dict, description="Query features")



class SearchResult(SecureBaseModel):
    """Individual search result - alias for backward compatibility."""

    id: str = Field(..., description="Result identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    content: str = Field(..., description="Result content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    payload: dict[str, Any] = Field(default_factory=dict, description="Result payload")


class FusionConfig(SecureBaseModel):
    """Configuration for fusion algorithms."""

    algorithm: str = Field(default="rrf", description="Fusion algorithm name")
    weights: dict[str, float] = Field(
        default_factory=dict, description="Algorithm weights"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
    )


class ModelSelectionStrategy(SecureBaseModel):
    """Model selection strategy configuration."""

    primary_model: str = Field(..., description="Primary model identifier")
    model_type: str = Field(..., description="Type of model")
    fallback_models: list[str] = Field(
        default_factory=list, description="Fallback models"
    )
    selection_criteria: dict[str, Any] = Field(
        default_factory=dict, description="Selection criteria"
    )


class ABTestConfig(SecureBaseModel):
    """A/B testing configuration."""

    experiment_name: str = Field(..., description="Experiment name")
    variants: list[str] = Field(..., description="Test variants")
    allocation: dict[str, float] = Field(
        default_factory=dict, description="Variant allocation"
    )
    success_metrics: list[str] = Field(
        default_factory=list, description="Success metrics"
    )


class HybridSearchRequest(AdvancedHybridSearchRequest):
    """Hybrid search request - alias for backward compatibility."""

    query: str = Field(..., description="Search query text")
    collection_name: str = Field(..., description="Collection to search")
    enable_query_classification: bool = Field(
        True, description="Enable query classification"
    )
    enable_model_selection: bool = Field(True, description="Enable model selection")
    enable_adaptive_fusion: bool = Field(True, description="Enable adaptive fusion")
    enable_splade: bool = Field(False, description="Enable SPLADE")
    fusion_config: FusionConfig | None = Field(None, description="Fusion configuration")
    user_id: str | None = Field(None, description="User identifier")
    session_id: str | None = Field(None, description="Session identifier")


class RetrievalMetrics(SecureBaseModel):
    """Retrieval performance metrics."""

    query_vector_time_ms: float = Field(
        ..., ge=0.0, description="Time to generate query vector in milliseconds"
    )
    search_time_ms: float = Field(
        ..., ge=0.0, description="Time to execute search in milliseconds"
    )
    total_time_ms: float = Field(
        ..., ge=0.0, description="Total processing time in milliseconds"
    )
    results_count: int = Field(..., ge=0, description="Number of results found")
    filtered_count: int = Field(
        ..., ge=0, description="Number of results after filtering"
    )
    cache_hit: bool = Field(..., description="Whether cache was hit")
    hnsw_ef_used: int = Field(
        ..., ge=1, description="HNSW ef parameter used for search"
    )

    @model_validator(mode="after")
    def validate_timing_consistency(self) -> Self:
        """Validate that timing values are consistent."""
        if self.total_time_ms < self.search_time_ms:
            msg = "Total time cannot be less than search time"
            raise ValueError(msg)
        if self.total_time_ms < self.query_vector_time_ms:
            msg = "Total time cannot be less than query vector time"
            raise ValueError(msg)
        return self


class HybridSearchResponse(SecureBaseModel):
    """Hybrid search response."""

    results: list[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    query_classification: QueryClassification | None = Field(
        None, description="Query classification"
    )
    model_selection: ModelSelectionStrategy | None = Field(
        None, description="Model selection info"
    )
    fusion_weights: dict[str, float] | None = Field(None, description="Fusion weights")
    optimization_applied: bool = Field(
        False, description="Whether optimization was applied"
    )
    retrieval_metrics: RetrievalMetrics | None = Field(
        None, description="Retrieval metrics"
    )
    total_results: int = Field(0, description="Total number of results")
    execution_time_ms: float = Field(0.0, description="Execution time in milliseconds")
    effectiveness_score: float | None = Field(
        None, description="Search effectiveness score"
    )
    fallback_reason: str | None = Field(
        None, description="Reason for fallback if applicable"
    )


# =============================================================================

# Core models
__all__ = [
    "ABTestConfig",
    "AdvancedFilteredSearchRequest",
    "AdvancedHybridSearchRequest",
    # Async models
    "AsyncSearchContext",
    # Request models
    "BasicSearchRequest",
    "BatchSearchRequest",
    "BatchSearchResponse",
    "DimensionError",
    "FilterValidationError",
    "FusionAlgorithm",
    "FusionConfig",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "HyDESearchRequest",
    "ModelSelectionStrategy",
    "MultiStageSearchRequest",
    # Prefetch models
    "PrefetchConfig",
    "QueryClassification",
    "RetrievalMetrics",
    "SearchAccuracy",
    "SearchConfigurationError",
    "SearchResponse",
    "SearchResult",
    # Search stage models
    "SearchStage",
    # Base classes and enums
    "SecureBaseModel",
    "SecureFilterGroupModel",
    "SecureFilterModel",
    # Security models
    "SecureMetadataModel",
    "SecurePayloadModel",
    "SecureSearchParamsModel",
    # Response models
    "SecureSearchResult",
    "SecureSparseVectorModel",
    # Vector models
    "SecureVectorModel",
    "SecurityValidationError",
    # Exception classes
    "VectorSearchError",
    "VectorType",
]
