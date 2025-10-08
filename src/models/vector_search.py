"""Vector Search Models - Security-First Implementation

This module provides production-ready Pydantic v2 models for vector search
operations with security validation, performance optimization, and type safety.

Key Features:
- Security-first design with DoS prevention and injection protection
- Full Pydantic v2 compliance with modern validation patterns
- Type-safe replacements for all dict[str, Any] patterns
- Async-ready design for Qdrant integration
- Performance-optimized with caching and validation strategies
"""

from __future__ import annotations

import math
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

from src.config.models import FusionAlgorithm


if TYPE_CHECKING:
    pass


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

        if not v:
            msg = "Vector cannot be empty"
            raise ValueError(msg)

        if len(v) > 4096:
            msg = (
                f"Vector dimensions exceed maximum allowed (4096), got {len(v)} "
                "(security: DoS prevention)"
            )
            raise DimensionError(msg)

        def _raise_error(message: str) -> None:
            """Helper to raise ValueError."""
            raise ValueError(message)

        # Convert and validate each value
        validated_values = []
        for i, val in enumerate(v):
            try:
                float_val = float(val)
                if not -1e6 <= float_val <= 1e6:  # Reasonable bounds
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
    @classmethod
    def set_default_search_params(cls, values: dict[str, Any]) -> dict[str, Any]:
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


class SearchStage(SecureBaseModel):
    """Individual search stage for multi-stage search operations."""

    stage_name: str = Field(..., max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    query_vector: SecureVectorModel = Field(..., description="Stage query vector")
    search_params: SecureSearchParamsModel = Field(
        default_factory=SecureSearchParamsModel  # type: ignore[call-arg]
    )
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def set_default_search_params(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set default search_params if not provided."""
        if values.get("search_params") is None:
            values["search_params"] = SecureSearchParamsModel()
        return values


class PrefetchConfig(SecureBaseModel):
    """Configuration for prefetch operations with optimized limits."""

    dense_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    sparse_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    hyde_multiplier: float = Field(default=2.5, ge=1.0, le=5.0)
    max_prefetch_limit: int = Field(default=1000, ge=10, le=5000)

    def calculate_prefetch_limit(self, vector_type: str, final_limit: int) -> int:
        """Calculate optimal prefetch limit based on vector type and final limit.

        Args:
            vector_type: Type of vector operation
            final_limit: Final result limit

        Returns:
            Optimal prefetch limit for the given vector type
        """

        if vector_type == "dense":
            multiplier = self.dense_multiplier
        elif vector_type == "sparse":
            multiplier = self.sparse_multiplier
        elif vector_type == "hyde":
            multiplier = self.hyde_multiplier
        else:
            multiplier = self.dense_multiplier  # Default fallback

        prefetch_limit = int(final_limit * multiplier)
        return min(prefetch_limit, self.max_prefetch_limit)


class FusionConfig(SecureBaseModel):
    """Configuration for fusion algorithms."""

    algorithm: str = Field(default="rrf", description="Fusion algorithm name")
    weights: dict[str, float] = Field(
        default_factory=dict, description="Algorithm weights"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
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


__all__ = [
    "AdvancedFilteredSearchRequest",
    "AdvancedHybridSearchRequest",
    "BasicSearchRequest",
    "DimensionError",
    "FilterValidationError",
    "FusionConfig",
    "HybridSearchRequest",
    "PrefetchConfig",
    "SearchConfigurationError",
    "SearchStage",
    "SecureBaseModel",
    "SecureFilterGroupModel",
    "SecureFilterModel",
    "SecureSearchParamsModel",
    "SecureSparseVectorModel",
    "SecureVectorModel",
    "SecurityValidationError",
    "VectorSearchError",
]
