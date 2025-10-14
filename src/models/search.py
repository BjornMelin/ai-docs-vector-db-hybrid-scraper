"""Unified Search Request Model.

This module provides the single, comprehensive SearchRequest model that serves
all search operations across the entire codebase.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.models import (
    FusionAlgorithm,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)


class SearchRequest(BaseModel):
    """Unified search request model for all search operations.

    Attributes:
        # Core search parameters
        query: Search query text
        collection: Target collection name
        limit: Maximum results to return
        offset: Results offset for pagination

        # Vector search parameters
        query_vector: Vector representation for similarity search
        sparse_vector: Sparse vector for hybrid search
        vector_type: Type of vector operation

        # Search strategy and configuration
        search_strategy: Search approach (dense, sparse, hybrid, multi_stage)
        search_accuracy: Accuracy level (fast, balanced, accurate, exact)
        fusion_algorithm: Algorithm for combining results

        # Filtering and metadata
        filters: Payload filters for search
        filter_groups: Complex nested filter groups
        exclude_filters: Filters to exclude results
        metadata_filters: Metadata-specific filters
        temporal_filters: Time-based filters
        content_filters: Content type filters

        # Advanced features
        enable_expansion: Query expansion/synonym handling
        enable_personalization: User preference-based ranking
        enable_reranking: BGE reranking
        enable_hyde: Hypothetical Document Embeddings
        enable_rag: Retrieval-Augmented Generation
        enable_caching: Result caching
        enable_adaptive_fusion: Dynamic fusion algorithm selection

        # RAG parameters
        rag_top_k: Documents for RAG context
        rag_max_tokens: Maximum tokens for RAG response

        # User and context
        user_id: User identifier for personalization
        user_preferences: Preference boosts by category
        user_context: Additional user context
        session_id: Session tracking

        # Performance and optimization
        timeout: Search timeout in seconds
        batch_size: Processing batch size
        cache_ttl: Cache time-to-live
        max_processing_time_ms: Maximum processing time
        score_threshold: Minimum similarity score
        normalize_scores: Score normalization
        boost_recent: Recency boosting
        adaptive_threshold: Dynamic threshold adjustment

        # Grouping and aggregation
        group_by: Field for result grouping
        group_size: Maximum hits per group
        overfetch_multiplier: Overfetch factor for grouping

        # Output control
        include_metadata: Include result metadata
        include_vectors: Include vector data in results
        include_analytics: Include detailed analytics

        # Advanced search options
        stages: Multi-stage search configuration
        embedding_model: Specific embedding model
        domain: Domain hint for better processing
        num_hypotheses: Number of HyDE hypotheses
        generation_temperature: HyDE generation temperature
        max_generation_tokens: Maximum HyDE tokens
        force_strategy: Override search strategy
        force_dimension: Override vector dimension
        nested_logic: Allow nested filter logic
        optimize_order: Optimize filter evaluation order
    """

    # Core search parameters
    query: str = Field(..., min_length=1, description="Search query text")
    collection: str | None = Field(
        default="documentation", min_length=1, description="Target collection"
    )
    limit: int = Field(10, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Results offset for pagination")

    # Vector search parameters
    query_vector: list[float] | None = Field(
        default=None, description="Vector representation for similarity search"
    )
    sparse_vector: dict[int, float] | None = Field(
        default=None, description="Sparse vector for hybrid search"
    )
    vector_type: VectorType = Field(
        default=VectorType.DENSE, description="Type of vector operation"
    )

    # Search strategy and configuration
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.HYBRID, description="Search approach"
    )
    search_accuracy: SearchAccuracy = Field(
        default=SearchAccuracy.BALANCED, description="Accuracy level"
    )
    fusion_algorithm: FusionAlgorithm = Field(
        default=FusionAlgorithm.RRF, description="Result fusion algorithm"
    )

    # Filtering and metadata
    filters: dict[str, Any] | None = Field(default=None, description="Payload filters")
    filter_groups: list[dict[str, Any]] | None = Field(
        default=None, description="Complex nested filter groups"
    )
    exclude_filters: list[dict[str, Any]] | None = Field(
        default=None, description="Filters to exclude results"
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None, description="Metadata-specific filters"
    )
    temporal_filters: dict[str, Any] | None = Field(
        default=None, description="Time-based filters"
    )
    content_filters: dict[str, Any] | None = Field(
        default=None, description="Content type filters"
    )

    # Advanced features
    enable_expansion: bool = Field(
        default=True, description="Query expansion/synonym handling"
    )
    enable_personalization: bool = Field(
        default=False, description="User preference-based ranking"
    )
    enable_reranking: bool = Field(default=True, description="BGE reranking")
    enable_hyde: bool = Field(default=False, description="HyDE enhancement")
    enable_rag: bool = Field(default=False, description="RAG generation")
    enable_caching: bool = Field(default=True, description="Result caching")
    enable_adaptive_fusion: bool = Field(
        default=True, description="Dynamic fusion algorithm selection"
    )

    # RAG parameters
    rag_top_k: int | None = Field(
        default=None, ge=1, description="Documents for RAG context"
    )
    rag_max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens for RAG response"
    )

    # User and context
    user_id: str | None = Field(
        default=None, description="User identifier for personalization"
    )
    user_preferences: dict[str, float] | None = Field(
        default=None, description="Preference boosts by category"
    )
    user_context: dict[str, Any] | None = Field(
        default=None, description="Additional user context"
    )
    session_id: str | None = Field(default=None, description="Session tracking")

    # Performance and optimization
    timeout: int = Field(default=30, ge=1, le=300, description="Search timeout")
    batch_size: int = Field(
        default=32, ge=1, le=1000, description="Processing batch size"
    )
    cache_ttl: int | None = Field(default=None, description="Cache TTL in seconds")
    max_processing_time_ms: int | None = Field(
        default=None, description="Maximum processing time"
    )
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    normalize_scores: bool = Field(
        default=True, description="Score normalization per request"
    )
    boost_recent: bool = Field(default=False, description="Recency boosting")
    adaptive_threshold: bool = Field(
        default=False, description="Dynamic threshold adjustment"
    )

    # Grouping and aggregation
    group_by: str = Field(default="doc_id", description="Field for result grouping")
    group_size: int = Field(
        default=1, ge=1, le=10, description="Maximum hits per group"
    )
    overfetch_multiplier: float = Field(
        default=2.0, ge=1.0, description="Overfetch factor for grouping"
    )

    # Output control
    include_metadata: bool = Field(default=True, description="Include result metadata")
    include_vectors: bool = Field(default=False, description="Include vector data")
    include_analytics: bool = Field(default=False, description="Include analytics")

    # Advanced search options
    stages: list[dict[str, Any]] | None = Field(
        default=None, description="Multi-stage search configuration"
    )
    embedding_model: str | None = Field(
        default=None, description="Specific embedding model to use"
    )
    domain: str | None = Field(
        default=None, description="Domain hint for better processing"
    )
    num_hypotheses: int = Field(
        default=5, ge=1, le=10, description="Number of HyDE hypotheses"
    )
    generation_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="HyDE generation temperature"
    )
    max_generation_tokens: int = Field(
        default=200, ge=50, le=500, description="Maximum HyDE tokens"
    )
    force_strategy: str | None = Field(
        default=None, description="Force specific search strategy"
    )
    force_dimension: int | None = Field(
        default=None, description="Force specific vector dimension"
    )
    nested_logic: bool = Field(
        default=False, description="Allow nested boolean expressions"
    )
    optimize_order: bool = Field(
        default=True, description="Optimize filter evaluation order"
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        str_max_length=5000,
        validate_assignment=True,
    )

    _FILTER_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[a-zA-Z_][a-zA-Z0-9_.]*$"
    )
    _DANGEROUS_FILTER_PATTERNS: ClassVar[tuple[str, ...]] = (
        ";",
        "--",
        "/*",
        "*/",
        "xp_",
        "sp_",
        "drop",
        "delete",
        "insert",
        "update",
    )
    _MAX_FILTER_KEYS: ClassVar[int] = 50
    _MAX_FILTER_DEPTH: ClassVar[int] = 10
    _MAX_FILTER_LIST_ITEMS: ClassVar[int] = 100
    _MAX_FILTER_STRING_LENGTH: ClassVar[int] = 500
    _MAX_VECTOR_DIMENSION: ClassVar[int] = 4096
    _MAX_SPARSE_INDEX: ClassVar[int] = 100_000
    _VECTOR_VALUE_RANGE: ClassVar[tuple[float, float]] = (-1e6, 1e6)

    @field_validator(
        "filters",
        "metadata_filters",
        "temporal_filters",
        "content_filters",
        mode="after",
    )
    @classmethod
    def _validate_filter_mapping(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Validate filter mappings to prevent injection and DoS attacks."""
        if value is None:
            return None

        if not isinstance(value, Mapping):
            msg = "Filters must be supplied as a mapping"
            raise TypeError(msg)

        if len(value) > cls._MAX_FILTER_KEYS:
            msg = f"Too many filter keys supplied (max {cls._MAX_FILTER_KEYS})"
            raise ValueError(msg)

        validated: dict[str, Any] = {}
        for raw_key, raw_val in value.items():
            key = cls._ensure_valid_filter_key(raw_key)
            validated[key] = cls._validate_filter_value(raw_val)
        return validated

    @field_validator("exclude_filters", mode="after")
    @classmethod
    def _validate_exclude_filters(
        cls, value: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Validate exclude filter payloads."""
        if value is None:
            return None

        if len(value) > cls._MAX_FILTER_KEYS:
            msg = f"Too many exclude filters supplied (max {cls._MAX_FILTER_KEYS})"
            raise ValueError(msg)

        validated: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                msg = "Exclude filters must be mappings"
                raise TypeError(msg)
            validated.append(cls._validate_filter_mapping(dict(item)) or {})
        return validated

    @field_validator("filter_groups", mode="after")
    @classmethod
    def _validate_filter_groups(
        cls, value: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Validate nested filter groups with depth and size limits."""
        if value is None:
            return None

        if len(value) > cls._MAX_FILTER_KEYS:
            msg = f"Too many filter groups supplied (max {cls._MAX_FILTER_KEYS})"
            raise ValueError(msg)

        for group in value:
            cls._validate_filter_group(group, depth=0)
        return value

    @field_validator("query_vector", mode="after")
    @classmethod
    def _validate_query_vector(cls, value: list[float] | None) -> list[float] | None:
        """Validate dense query vectors for bounds and numeric stability."""
        if value is None:
            return None

        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            msg = "query_vector must be a sequence of floats"
            raise TypeError(msg)

        if not value:
            msg = "query_vector cannot be empty"
            raise ValueError(msg)

        if len(value) > cls._MAX_VECTOR_DIMENSION:
            msg = (
                "query_vector exceeds maximum allowed dimension "
                f"({cls._MAX_VECTOR_DIMENSION})"
            )
            raise ValueError(msg)

        validated: list[float] = []
        for index, item in enumerate(value):
            try:
                float_val = float(item)
            except (TypeError, ValueError) as exc:
                msg = f"Invalid vector value at index {index}: {item!r}"
                raise ValueError(msg) from exc
            if not math.isfinite(float_val):
                msg = f"Non-finite vector value at index {index}"
                raise ValueError(msg)
            minimum, maximum = cls._VECTOR_VALUE_RANGE
            if float_val < minimum or float_val > maximum:
                msg = (
                    "Vector value out of allowed range "
                    f"[{minimum}, {maximum}] at index {index}"
                )
                raise ValueError(msg)
            validated.append(float_val)
        return validated

    @field_validator("sparse_vector", mode="after")
    @classmethod
    def _validate_sparse_vector(
        cls, value: dict[int, float] | None
    ) -> dict[int, float] | None:
        """Validate sparse vector representations."""
        if value is None:
            return None

        if not isinstance(value, Mapping):
            msg = "sparse_vector must be a mapping of indices to weights"
            raise TypeError(msg)

        if not value:
            msg = "sparse_vector cannot be empty"
            raise ValueError(msg)

        if len(value) > cls._MAX_VECTOR_DIMENSION:
            msg = (
                "sparse_vector exceeds maximum allowed dimension "
                f"({cls._MAX_VECTOR_DIMENSION})"
            )
            raise ValueError(msg)

        validated: dict[int, float] = {}
        for raw_index, raw_weight in value.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                msg = f"Sparse vector index must be an integer: {raw_index!r}"
                raise ValueError(msg) from exc
            if index < 0 or index >= cls._MAX_SPARSE_INDEX:
                msg = (
                    "Sparse vector index out of bounds: "
                    f"{index} (max {cls._MAX_SPARSE_INDEX - 1})"
                )
                raise ValueError(msg)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                msg = f"Sparse vector weight must be numeric: {raw_weight!r}"
                raise ValueError(msg) from exc
            if not math.isfinite(weight):
                msg = f"Sparse vector weight not finite for index {index}"
                raise ValueError(msg)
            validated[index] = weight
        return validated

    @model_validator(mode="after")
    def _validate_strategy(self) -> SearchRequest:
        """Cross-field validation for strategy, vector type, and dimensions."""
        if (
            self.vector_type in {VectorType.SPARSE, VectorType.HYBRID}
            and not self.sparse_vector
        ):
            msg = f"sparse_vector is required when vector_type is {self.vector_type}"
            raise ValueError(msg)

        if (
            self.search_strategy == SearchStrategy.SPARSE
            and self.vector_type == VectorType.DENSE
        ):
            msg = "Sparse search_strategy requires a sparse-compatible vector_type"
            raise ValueError(msg)

        if (
            self.search_strategy == SearchStrategy.DENSE
            and self.vector_type == VectorType.SPARSE
        ):
            msg = "Dense search_strategy cannot be paired with sparse-only vector_type"
            raise ValueError(msg)

        if (
            self.force_dimension is not None
            and self.query_vector is not None
            and len(self.query_vector) != self.force_dimension
        ):
            msg = (
                "query_vector dimension does not match force_dimension "
                f"({self.force_dimension})"
            )
            raise ValueError(msg)

        return self

    @classmethod
    def from_input(
        cls,
        payload: SearchRequest | Mapping[str, Any] | str,
        **overrides: Any,
    ) -> SearchRequest:
        """Normalize heterogeneous search request payloads.

        Args:
            payload: An existing :class:`SearchRequest`, mapping, or raw query
                string describing the search.
            **overrides: Field overrides applied after normalisation.

        Returns:
            A validated :class:`SearchRequest` instance built from the provided
            payload.

        Raises:
            TypeError: If the payload type cannot be normalised.
        """
        if isinstance(payload, cls):
            if not overrides:
                return payload
            return payload.model_copy(update=overrides)

        if isinstance(payload, str):
            data: dict[str, Any] = {"query": payload}
            data.update(overrides)
            return cls.model_validate(data)

        if isinstance(payload, Mapping):
            data = dict(payload)
            data.update(overrides)
            return cls.model_validate(data)

        msg = f"Unsupported search request payload type: {type(payload)!r}"
        raise TypeError(msg)

    @classmethod
    def _ensure_valid_filter_key(cls, key: Any) -> str:
        """Validate a filter key name."""
        if not isinstance(key, str):
            msg = "Filter keys must be strings"
            raise TypeError(msg)

        if not cls._FILTER_KEY_PATTERN.match(key):
            msg = f"Invalid filter key: {key!r}"
            raise ValueError(msg)

        return key

    @classmethod
    def _validate_filter_value(cls, value: Any) -> Any:
        """Validate an individual filter value."""
        if value is None:
            return value

        if isinstance(value, str):
            if len(value) > cls._MAX_FILTER_STRING_LENGTH:
                msg = "Filter string value too long"
                raise ValueError(msg)
            lowered = value.lower()
            for pattern in cls._DANGEROUS_FILTER_PATTERNS:
                if pattern in lowered:
                    msg = "Potentially dangerous pattern in filter value"
                    raise ValueError(msg)
            return value

        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) > cls._MAX_FILTER_LIST_ITEMS:
                msg = "Filter list contains too many items"
                raise ValueError(msg)
            for item in value:
                cls._validate_filter_value(item)
            return list(value)

        if isinstance(value, Mapping):
            return cls._validate_filter_mapping(dict(value)) or {}

        return value

    @classmethod
    def _validate_filter_group(cls, group: Mapping[str, Any], *, depth: int) -> None:
        """Validate nested filter group structure."""
        if not isinstance(group, Mapping):
            msg = "Filter groups must be mappings"
            raise TypeError(msg)

        if depth > cls._MAX_FILTER_DEPTH:
            msg = "Filter group nesting too deep"
            raise ValueError(msg)

        operator = group.get("operator")
        if operator not in {"and", "or", "not"}:
            msg = "Filter group operator must be 'and', 'or', or 'not'"
            raise ValueError(msg)

        filters = group.get("filters")
        if not isinstance(filters, Sequence) or isinstance(filters, str | bytes):
            msg = "Filter group must contain a list of filters"
            raise TypeError(msg)
        if len(filters) == 0:
            msg = "Filter group cannot be empty"
            raise ValueError(msg)
        if len(filters) > cls._MAX_FILTER_KEYS:
            msg = "Filter group contains too many filters"
            raise ValueError(msg)

        for item in filters:
            if isinstance(item, Mapping):
                if "filters" in item:
                    cls._validate_filter_group(item, depth=depth + 1)
                else:
                    cls._validate_filter_mapping(dict(item))
            else:
                msg = "Filter group entries must be mappings"
                raise TypeError(msg)


__all__ = ["SearchRequest"]
