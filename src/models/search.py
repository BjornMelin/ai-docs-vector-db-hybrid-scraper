"""Unified Search Request Model.

This module provides the single, comprehensive SearchRequest model that serves
all search operations across the entire codebase.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

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
    sparse_vector: dict[str, Any] | None = Field(
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

    model_config = {"extra": "forbid"}

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


__all__ = ["SearchRequest"]
