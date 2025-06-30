import typing
"""Vector search models for Qdrant Query API and search operations.

This module consolidates all models related to vector search, HNSW configuration,
search stages, prefetch configurations, and search parameters.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from ..config import ABTestVariant
from ..config import FusionAlgorithm
from ..config import ModelType
from ..config import QueryComplexity
from ..config import QueryType
from ..config import SearchAccuracy
from ..config import VectorType


class SearchStage(BaseModel):
    """Configuration for a single stage in multi-stage retrieval."""

    query_vector: list[float] = Field(..., description="Vector for this stage")
    vector_name: str = Field(..., description="Vector field name (dense, sparse, etc.)")
    vector_type: VectorType = Field(..., description="Type of vector for optimization")
    limit: int = Field(..., description="Number of results to retrieve in this stage")
    filter: dict[str, Any] | None = Field(
        None, description="Optional filter for this stage"
    )
    search_params: dict[str, Any] | None = Field(
        None, description="Stage-specific search parameters"
    )


class PrefetchConfig(BaseModel):
    """Optimized prefetch configuration based on research findings."""

    # Optimal prefetch limits based on vector type
    sparse_multiplier: float = Field(
        5.0, description="Multiplier for sparse vectors (cast wider net)"
    )
    hyde_multiplier: float = Field(
        3.0, description="Multiplier for HyDE vectors (moderate expansion)"
    )
    dense_multiplier: float = Field(
        2.0, description="Multiplier for dense vectors (precision focus)"
    )

    # Maximum prefetch limits to prevent performance degradation
    max_sparse_limit: int = Field(500, description="Maximum sparse prefetch limit")
    max_dense_limit: int = Field(200, description="Maximum dense prefetch limit")
    max_hyde_limit: int = Field(150, description="Maximum HyDE prefetch limit")

    def calculate_prefetch_limit(
        self, vector_type: VectorType, final_limit: int
    ) -> int:
        """Calculate optimal prefetch limit for a given vector type."""
        if vector_type == VectorType.SPARSE:
            calculated = int(final_limit * self.sparse_multiplier)
            return min(calculated, self.max_sparse_limit)
        elif vector_type == VectorType.HYDE:
            calculated = int(final_limit * self.hyde_multiplier)
            return min(calculated, self.max_hyde_limit)
        else:  # DENSE
            calculated = int(final_limit * self.dense_multiplier)
            return min(calculated, self.max_dense_limit)


class SearchParams(BaseModel):
    """HNSW search parameters optimized for different accuracy levels."""

    accuracy_level: SearchAccuracy = Field(
        SearchAccuracy.BALANCED, description="Desired accuracy level"
    )
    hnsw_ef: int | None = Field(None, description="HNSW exploration factor")
    exact: bool = Field(False, description="Use exact search (disable HNSW)")

    @classmethod
    def from_accuracy_level(cls, accuracy_level: SearchAccuracy) -> "SearchParams":
        """Create SearchParams from accuracy level."""
        params_map = {
            SearchAccuracy.FAST: {"hnsw_ef": 50, "exact": False},
            SearchAccuracy.BALANCED: {"hnsw_ef": 100, "exact": False},
            SearchAccuracy.ACCURATE: {"hnsw_ef": 200, "exact": False},
            SearchAccuracy.EXACT: {"exact": True},
        }

        config = params_map.get(accuracy_level, params_map[SearchAccuracy.BALANCED])
        return cls(accuracy_level=accuracy_level, **config)


class FusionConfig(BaseModel):
    """Configuration for fusion algorithm selection."""

    algorithm: FusionAlgorithm = Field(
        FusionAlgorithm.RRF, description="Fusion algorithm to use"
    )
    auto_select: bool = Field(
        True, description="Automatically select fusion algorithm based on query type"
    )

    @classmethod
    def select_fusion_algorithm(cls, query_type: str) -> FusionAlgorithm:
        """Select optimal fusion algorithm based on query type."""
        fusion_map = {
            "hybrid": FusionAlgorithm.RRF,  # Best for combining dense+sparse
            "multi_stage": FusionAlgorithm.RRF,  # Good for multiple strategies
            "reranking": FusionAlgorithm.DBSF,  # Better for similar vectors
            "hyde": FusionAlgorithm.RRF,  # Good for hypothetical docs
        }

        return fusion_map.get(query_type, FusionAlgorithm.RRF)


class MultiStageSearchRequest(BaseModel):
    """Request configuration for multi-stage retrieval."""

    collection_name: str = Field(..., description="Target collection")
    stages: list[SearchStage] = Field(..., description="Search stages to execute")
    fusion_config: FusionConfig = Field(
        default_factory=FusionConfig, description="Fusion configuration"
    )
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    limit: int = Field(10, description="Final number of results to return")
    score_threshold: float = Field(0.0, description="Minimum score threshold")


class HyDESearchRequest(BaseModel):
    """Request configuration for HyDE (Hypothetical Document Embeddings) search."""

    collection_name: str = Field(..., description="Target collection")
    query: str = Field(..., description="Original query text")
    num_hypothetical_docs: int = Field(
        5, description="Number of hypothetical documents to generate"
    )
    limit: int = Field(10, description="Final number of results to return")
    fusion_config: FusionConfig = Field(
        default_factory=FusionConfig, description="Fusion configuration"
    )
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )


class FilteredSearchRequest(BaseModel):
    """Request configuration for filtered search with indexed payload fields."""

    collection_name: str = Field(..., description="Target collection")
    query_vector: list[float] = Field(..., description="Query vector")
    filters: dict[str, Any] = Field(..., description="Filters to apply")
    limit: int = Field(10, description="Number of results to return")
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    score_threshold: float = Field(0.0, description="Minimum score threshold")


class HybridSearchRequest(BaseModel):
    """Request configuration for hybrid dense+sparse vector search."""

    collection_name: str = Field(..., description="Target collection")
    dense_vector: list[float] = Field(..., description="Dense query vector")
    sparse_vector: dict[str, Any] | None = Field(
        None, description="Sparse query vector (indices and values)"
    )
    dense_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for dense vector results"
    )
    sparse_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for sparse vector results"
    )
    limit: int = Field(10, description="Final number of results to return")
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    score_threshold: float = Field(0.0, description="Minimum score threshold")


class SearchResult(BaseModel):
    """Standardized search result format."""

    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata and content"
    )
    vector: list[float] | None = Field(None, description="Document vector if requested")


class SearchResponse(BaseModel):
    """Standardized search response format."""

    results: list[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_count: int = Field(0, description="Total matching documents")
    query_time_ms: float = Field(
        0.0, description="Query execution time in milliseconds"
    )
    search_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used for search"
    )


class RetrievalMetrics(BaseModel):
    """Metrics for search and retrieval operations."""

    query_vector_time_ms: float = Field(
        0.0, description="Time to generate query vector"
    )
    search_time_ms: float = Field(0.0, description="Time for vector search")
    total_time_ms: float = Field(0.0, description="Total retrieval time")
    results_count: int = Field(0, description="Number of results returned")
    filtered_count: int = Field(0, description="Number of results after filtering")
    cache_hit: bool = Field(False, description="Whether result was cached")
    hnsw_ef_used: int | None = Field(None, description="HNSW ef parameter used")


class AdaptiveSearchParams(BaseModel):
    """Parameters for adaptive search optimization."""

    time_budget_ms: int = Field(
        default=100, gt=0, description="Maximum time budget for search"
    )
    min_results: int = Field(
        default=5, gt=0, description="Minimum number of results required"
    )
    max_ef: int = Field(default=200, gt=0, description="Maximum ef to try")
    min_ef: int = Field(default=50, gt=0, description="Minimum ef to try")
    ef_step: int = Field(default=25, gt=0, description="Step size for ef adjustment")


class IndexingRequest(BaseModel):
    """Request for creating payload indexes."""

    collection_name: str = Field(..., description="Target collection")
    field_name: str = Field(..., description="Field to index")
    field_type: str = Field(
        default="keyword",
        description="Index type (keyword, integer, float, text, geo, bool)",
    )
    wait: bool = Field(
        default=True, description="Wait for indexing operation to complete"
    )


class CollectionStats(BaseModel):
    """Statistics for a collection."""

    name: str = Field(..., description="Collection name")
    points_count: int = Field(0, description="Number of points in collection")
    vectors_count: int = Field(0, description="Number of vectors in collection")
    indexed_fields: list[str] = Field(
        default_factory=list, description="Fields with payload indexes"
    )
    status: str = Field("unknown", description="Collection status")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Collection configuration"
    )


class OptimizationRequest(BaseModel):
    """Request for collection optimization."""

    collection_name: str = Field(..., description="Target collection")
    optimization_type: str = Field(
        default="auto", description="Type of optimization (auto, indexing, hnsw)"
    )
    force: bool = Field(
        default=False, description="Force optimization even if not needed"
    )


class VectorSearchConfig(BaseModel):
    """Overall configuration for vector search operations."""

    default_prefetch: PrefetchConfig = Field(
        default_factory=PrefetchConfig, description="Default prefetch configuration"
    )
    default_search_params: SearchParams = Field(
        default_factory=SearchParams, description="Default search parameters"
    )
    default_fusion: FusionConfig = Field(
        default_factory=FusionConfig, description="Default fusion configuration"
    )
    enable_metrics: bool = Field(
        default=True, description="Enable search metrics collection"
    )
    enable_adaptive_search: bool = Field(
        default=True, description="Enable adaptive search optimization"
    )
    cache_search_results: bool = Field(default=True, description="Cache search results")
    result_cache_ttl: int = Field(
        default=300, description="Search result cache TTL in seconds"
    )


class QueryClassification(BaseModel):
    """Query classification results for adaptive search optimization."""

    query_type: QueryType = Field(..., description="Primary query type")
    complexity_level: QueryComplexity = Field(..., description="Query complexity level")
    domain: str = Field(
        ..., description="Technical domain (programming, general, etc.)"
    )
    programming_language: str | None = Field(
        None, description="Detected programming language if applicable"
    )
    is_multimodal: bool = Field(
        False, description="Whether query involves multiple modalities"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score"
    )
    features: dict[str, Any] = Field(
        default_factory=dict, description="Extracted query features"
    )


class EffectivenessScore(BaseModel):
    """Effectiveness scoring for fusion weight optimization."""

    dense_effectiveness: float = Field(
        ..., ge=0.0, le=1.0, description="Dense retrieval effectiveness score"
    )
    sparse_effectiveness: float = Field(
        ..., ge=0.0, le=1.0, description="Sparse retrieval effectiveness score"
    )
    hybrid_effectiveness: float = Field(
        ..., ge=0.0, le=1.0, description="Combined hybrid effectiveness score"
    )
    query_id: str = Field(..., description="Unique query identifier")
    timestamp: float = Field(..., description="Scoring timestamp")
    evaluation_method: str = Field(
        default="top_result", description="Method used for effectiveness evaluation"
    )


class AdaptiveFusionWeights(BaseModel):
    """Dynamic fusion weights based on query characteristics."""

    dense_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Weight for dense vector results"
    )
    sparse_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Weight for sparse vector results"
    )
    hybrid_weight: float = Field(
        default=1.0, ge=0.0, description="Overall hybrid fusion weight"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Weight assignment confidence"
    )
    learning_rate: float = Field(
        default=0.01, gt=0.0, description="Learning rate for weight adaptation"
    )
    query_classification: QueryClassification = Field(
        ..., description="Query classification that informed weights"
    )
    effectiveness_score: EffectivenessScore | None = Field(
        None, description="Effectiveness score used for weight calculation"
    )

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def validate_weights_sum_to_one(cls, v, info):
        """Ensure dense and sparse weights sum to approximately 1.0."""
        if info.data and "dense_weight" in info.data:
            total = info.data["dense_weight"] + v
            if not (0.95 <= total <= 1.05):  # Allow small tolerance
                raise ValueError("Dense and sparse weights should sum to ~1.0")
        return v


class ModelSelectionStrategy(BaseModel):
    """Strategy for selecting optimal embedding models based on query."""

    primary_model: str = Field(..., description="Primary embedding model to use")
    model_type: ModelType = Field(..., description="Type of the selected model")
    fallback_models: list[str] = Field(
        default_factory=list, description="Fallback models if primary fails"
    )
    model_weights: dict[str, float] = Field(
        default_factory=dict, description="Weights for ensemble model usage"
    )
    selection_rationale: str = Field(..., description="Reason for model selection")
    expected_performance: float = Field(
        ..., ge=0.0, le=1.0, description="Expected performance score"
    )
    cost_efficiency: float = Field(
        ..., ge=0.0, le=1.0, description="Cost efficiency rating"
    )
    query_classification: QueryClassification = Field(
        ..., description="Query classification that informed selection"
    )


class SPLADEConfig(BaseModel):
    """Configuration for SPLADE sparse vector generation."""

    model_name: str = Field(
        default="naver/splade-cocondenser-ensembledistil",
        description="SPLADE model identifier",
    )
    max_sequence_length: int = Field(
        default=512, description="Maximum input sequence length"
    )
    top_k_tokens: int = Field(
        default=256, description="Top-k tokens to keep in sparse vector"
    )
    alpha: float = Field(default=1.0, description="Sparsity regularization parameter")
    cache_embeddings: bool = Field(
        default=True, description="Whether to cache SPLADE embeddings"
    )
    batch_size: int = Field(default=32, description="Batch size for SPLADE inference")


class ABTestConfig(BaseModel):
    """Configuration for A/B testing different fusion strategies."""

    experiment_name: str = Field(..., description="Name of the A/B test experiment")
    variants: list[ABTestVariant] = Field(..., description="List of test variants")
    traffic_allocation: dict[str, float] = Field(
        ..., description="Traffic allocation per variant (must sum to 1.0)"
    )
    success_metrics: list[str] = Field(
        default_factory=lambda: ["ndcg@10", "mrr", "click_through_rate"],
        description="Metrics to track for success",
    )
    minimum_sample_size: int = Field(
        default=1000, description="Minimum samples per variant for significance"
    )
    max_duration_days: int = Field(
        default=30, description="Maximum experiment duration"
    )
    significance_threshold: float = Field(
        default=0.05, description="P-value threshold for statistical significance"
    )
    early_stopping: bool = Field(
        default=True, description="Enable early stopping for clear winners"
    )


class HybridSearchRequest(BaseModel):
    """Request for hybrid search with adaptive optimization."""

    collection_name: str = Field(..., description="Target collection")
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, description="Number of results to return")
    enable_adaptive_fusion: bool = Field(
        default=True, description="Enable adaptive fusion weight tuning"
    )
    enable_query_classification: bool = Field(
        default=True, description="Enable automatic query classification"
    )
    enable_model_selection: bool = Field(
        default=True, description="Enable dynamic model selection"
    )
    enable_splade: bool = Field(
        default=True, description="Enable SPLADE sparse vector generation"
    )
    fusion_config: FusionConfig = Field(
        default_factory=FusionConfig, description="Fusion configuration"
    )
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    splade_config: SPLADEConfig = Field(
        default_factory=SPLADEConfig, description="SPLADE configuration"
    )
    ab_test_config: ABTestConfig | None = Field(
        None, description="A/B testing configuration"
    )
    user_id: str | None = Field(
        None, description="User ID for personalization and A/B testing"
    )
    session_id: str | None = Field(None, description="Session ID for context tracking")
    score_threshold: float = Field(0.0, description="Minimum score threshold")


class HybridSearchResponse(BaseModel):
    """Response from hybrid search with optimization metadata."""

    results: list[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    query_classification: QueryClassification | None = Field(
        None, description="Query classification results"
    )
    fusion_weights: AdaptiveFusionWeights | None = Field(
        None, description="Fusion weights used"
    )
    model_selection: ModelSelectionStrategy | None = Field(
        None, description="Model selection strategy used"
    )
    effectiveness_score: EffectivenessScore | None = Field(
        None, description="Effectiveness evaluation"
    )
    ab_test_variant: ABTestVariant | None = Field(
        None, description="A/B test variant used"
    )
    retrieval_metrics: RetrievalMetrics = Field(
        default_factory=RetrievalMetrics, description="Performance metrics"
    )
    optimization_applied: bool = Field(
        default=False, description="Whether optimization was applied"
    )
    fallback_reason: str | None = Field(
        None, description="Reason if fallback strategy was used"
    )


class QueryFeatures(BaseModel):
    """Extracted features from query for classification and optimization."""

    query_length: int = Field(..., description="Number of words/tokens in query")
    has_code_keywords: bool = Field(
        default=False, description="Contains programming keywords"
    )
    has_function_names: bool = Field(
        default=False, description="Contains function/method names"
    )
    has_programming_syntax: bool = Field(
        default=False, description="Contains code syntax elements"
    )
    question_type: str | None = Field(
        None, description="Type of question (how, what, why, etc.)"
    )
    technical_depth: str = Field(
        default="medium", description="Technical depth (basic, medium, advanced)"
    )
    entity_mentions: list[str] = Field(
        default_factory=list, description="Detected entities in query"
    )
    programming_language_indicators: list[str] = Field(
        default_factory=list, description="Programming language indicators"
    )
    semantic_complexity: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic complexity score"
    )
    keyword_density: float = Field(
        ..., ge=0.0, le=1.0, description="Technical keyword density"
    )


# Advanced filtering integration models
class TemporalSearchCriteria(BaseModel):
    """Temporal search criteria for date-based filtering."""

    start_date: str | None = Field(None, description="Start date (ISO format)")
    end_date: str | None = Field(None, description="End date (ISO format)")
    time_window: str | None = Field(
        None, description="Relative time window (e.g., '7d', '1M')"
    )
    freshness_weight: float = Field(
        0.1, ge=0.0, le=1.0, description="Weight for content freshness"
    )
    enable_decay: bool = Field(True, description="Enable time decay scoring")

    @model_validator(mode="after")
    def validate_temporal_criteria(self) -> "TemporalSearchCriteria":
        """Validate temporal criteria consistency."""
        if self.time_window and (self.start_date or self.end_date):
            raise ValueError("Cannot specify both time_window and explicit dates")
        return self


class ContentTypeSearchCriteria(BaseModel):
    """Content type criteria for filtering by document type."""

    allowed_types: list[str] = Field(
        default_factory=list, description="Allowed content types"
    )
    excluded_types: list[str] = Field(
        default_factory=list, description="Excluded content types"
    )
    type_weights: dict[str, float] = Field(
        default_factory=dict, description="Weights for different content types"
    )
    semantic_classification: bool = Field(
        False, description="Enable semantic content classification"
    )
    confidence_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Confidence threshold for classification"
    )


class MetadataSearchCriteria(BaseModel):
    """Metadata search criteria with boolean logic support."""

    filters: dict[str, Any] = Field(
        default_factory=dict, description="Key-value filters"
    )
    boolean_operator: str = Field("AND", description="Boolean operator (AND, OR)")
    nested_conditions: list[dict[str, Any]] = Field(
        default_factory=list, description="Nested filter conditions"
    )
    field_boost: dict[str, float] = Field(
        default_factory=dict, description="Field-specific boost values"
    )
    enable_fuzzy_match: bool = Field(False, description="Enable fuzzy matching")
    fuzzy_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Fuzzy match threshold"
    )


class SimilarityThresholdCriteria(BaseModel):
    """Adaptive similarity threshold criteria."""

    base_threshold: float = Field(
        0.6, ge=0.0, le=1.0, description="Base similarity threshold"
    )
    adaptive_mode: str = Field(
        "dynamic", description="Adaptive mode (static, dynamic, auto)"
    )
    quality_target: float = Field(
        0.8, ge=0.0, le=1.0, description="Target quality score"
    )
    min_results: int = Field(5, ge=1, description="Minimum results to return")
    max_threshold_reduction: float = Field(
        0.3, ge=0.0, le=0.5, description="Maximum threshold reduction"
    )
    enable_feedback_learning: bool = Field(
        True, description="Enable feedback-based learning"
    )


class FilteredSearchRequest(BaseModel):
    """Filtered search request with advanced filtering capabilities."""

    collection_name: str = Field(..., description="Target collection")
    query_vector: list[float] = Field(..., description="Query vector")
    limit: int = Field(10, ge=1, le=1000, description="Number of results to return")

    # Advanced filtering criteria
    temporal_criteria: TemporalSearchCriteria | None = Field(
        None, description="Temporal filtering criteria"
    )
    content_type_criteria: ContentTypeSearchCriteria | None = Field(
        None, description="Content type filtering criteria"
    )
    metadata_criteria: MetadataSearchCriteria | None = Field(
        None, description="Metadata filtering criteria"
    )
    similarity_threshold_criteria: SimilarityThresholdCriteria | None = Field(
        None, description="Similarity threshold criteria"
    )

    # Filter composition
    filter_composition_logic: dict[str, Any] | None = Field(
        None, description="Advanced filter composition logic"
    )

    # Search parameters
    search_params: SearchParams = Field(
        default_factory=SearchParams, description="Search parameters"
    )
    enable_query_expansion: bool = Field(False, description="Enable query expansion")
    enable_result_clustering: bool = Field(
        False, description="Enable result clustering"
    )
    enable_personalized_ranking: bool = Field(
        False, description="Enable personalized ranking"
    )

    # User context
    user_id: str | None = Field(None, description="User ID for personalization")
    session_id: str | None = Field(None, description="Session ID for context")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class FilteredSearchResult(BaseModel):
    """Filtered search result with filtering and processing metadata."""

    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )

    # Enhanced metadata
    filter_scores: dict[str, float] = Field(
        default_factory=dict, description="Individual filter scores"
    )
    cluster_id: str | None = Field(None, description="Cluster assignment if clustered")
    cluster_label: str | None = Field(None, description="Cluster label if available")
    personalization_boost: float = Field(0.0, description="Personalization score boost")
    temporal_relevance: float = Field(
        1.0, ge=0.0, le=1.0, description="Temporal relevance score"
    )
    content_type_match: float = Field(
        1.0, ge=0.0, le=1.0, description="Content type match score"
    )

    # Processing metadata
    processing_pipeline: list[str] = Field(
        default_factory=list, description="Processing stages applied"
    )
    expansion_terms: list[str] = Field(
        default_factory=list, description="Query expansion terms used"
    )
    ranking_factors: dict[str, float] = Field(
        default_factory=dict, description="Ranking factors applied"
    )


class FilteredSearchResponse(BaseModel):
    """Filtered search response with comprehensive metadata."""

    results: list[FilteredSearchResult] = Field(
        default_factory=list, description="Filtered search results"
    )
    total_count: int = Field(0, description="Total matching documents")

    # Query processing metadata
    query_processed: str = Field(..., description="Processed query")
    query_expansion_applied: bool = Field(
        False, description="Whether query expansion was applied"
    )
    expanded_terms: list[str] = Field(
        default_factory=list, description="Expanded query terms"
    )

    # Filtering metadata
    filters_applied: list[str] = Field(
        default_factory=list, description="Filters applied"
    )
    filter_composition: dict[str, Any] | None = Field(
        None, description="Filter composition used"
    )
    filtered_count: int = Field(0, description="Count after filtering")

    # Clustering metadata
    clustering_applied: bool = Field(
        False, description="Whether clustering was applied"
    )
    clusters_found: int = Field(0, description="Number of clusters found")
    cluster_distribution: dict[str, int] = Field(
        default_factory=dict, description="Result distribution across clusters"
    )

    # Ranking metadata
    personalized_ranking_applied: bool = Field(
        False, description="Whether personalized ranking was applied"
    )
    ranking_strategy: str | None = Field(None, description="Ranking strategy used")

    # Performance metadata
    retrieval_metrics: RetrievalMetrics = Field(
        default_factory=RetrievalMetrics, description="Performance metrics"
    )
    quality_metrics: dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )

    # Optimization metadata
    optimization_applied: bool = Field(
        False, description="Whether optimization was applied"
    )
    adaptive_adjustments: dict[str, Any] = Field(
        default_factory=dict, description="Adaptive adjustments made"
    )


class VectorSearchIntegrationConfig(BaseModel):
    """Configuration for integrating vector search with advanced features."""

    # Feature enablement
    enable_advanced_filtering: bool = Field(
        True, description="Enable advanced filtering"
    )
    enable_query_processing: bool = Field(
        True, description="Enable query processing pipeline"
    )
    enable_adaptive_optimization: bool = Field(
        True, description="Enable adaptive optimization"
    )
    enable_federated_search: bool = Field(False, description="Enable federated search")

    # Integration settings
    default_pipeline: str = Field("balanced", description="Default processing pipeline")
    max_processing_time_ms: float = Field(5000.0, description="Maximum processing time")
    cache_integration_results: bool = Field(
        True, description="Cache integrated results"
    )

    # Quality thresholds
    min_quality_score: float = Field(
        0.6, ge=0.0, le=1.0, description="Minimum quality threshold"
    )
    diversity_factor: float = Field(
        0.1, ge=0.0, le=1.0, description="Result diversity factor"
    )

    # Performance optimization
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_concurrent_operations: int = Field(
        5, ge=1, description="Max concurrent operations"
    )
    adaptive_timeout: bool = Field(
        True, description="Enable adaptive timeout adjustment"
    )

    @model_validator(mode="after")
    def validate_config(self) -> "VectorSearchIntegrationConfig":
        """Validate integration configuration."""
        if self.enable_federated_search and not self.enable_query_processing:
            raise ValueError("Federated search requires query processing to be enabled")
        return self


class IntegratedSearchRequest(BaseModel):
    """Unified search request integrating all advanced capabilities."""

    # Core search parameters
    query: str = Field(..., description="Search query")
    collection_name: str | None = Field(None, description="Target collection")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset")

    # Vector search configuration
    vector_search_config: VectorSearchConfig = Field(
        default_factory=VectorSearchConfig, description="Vector search configuration"
    )

    # Advanced filtering
    temporal_criteria: TemporalSearchCriteria | None = Field(None)
    content_type_criteria: ContentTypeSearchCriteria | None = Field(None)
    metadata_criteria: MetadataSearchCriteria | None = Field(None)
    similarity_threshold_criteria: SimilarityThresholdCriteria | None = Field(None)

    # Query processing options
    enable_query_expansion: bool = Field(True)
    enable_result_clustering: bool = Field(False)
    enable_personalized_ranking: bool = Field(False)
    enable_federated_search: bool = Field(False)

    # Integration configuration
    integration_config: VectorSearchIntegrationConfig = Field(
        default_factory=VectorSearchIntegrationConfig,
        description="Integration configuration",
    )

    # User context
    user_id: str | None = Field(None)
    session_id: str | None = Field(None)
    context: dict[str, Any] = Field(default_factory=dict)

    # Performance controls
    time_budget_ms: float | None = Field(None, description="Time budget override")
    quality_threshold: float | None = Field(
        None, description="Quality threshold override"
    )


# Export commonly used types
__all__ = [
    "ABTestConfig",
    "AdaptiveFusionWeights",
    "AdaptiveSearchParams",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "CollectionStats",
    "ContentTypeSearchCriteria",
    "EffectivenessScore",
    "FilteredSearchRequest",
    "FilteredSearchResponse", 
    "FilteredSearchResult",
    "FusionConfig",
    "HyDESearchRequest",
    "HybridSearchRequest",
    "IndexingRequest",
    "IntegratedSearchRequest",
    "MetadataSearchCriteria",
    "ModelSelectionStrategy",
    "MultiStageSearchRequest",
    "OptimizationRequest",
    "PrefetchConfig",
    "QueryClassification",
    "QueryFeatures",
    "RetrievalMetrics",
    "SPLADEConfig",
    "SearchParams",
    "SearchResponse",
    "SearchResult",
    "SearchStage",
    "SimilarityThresholdCriteria",
    "TemporalSearchCriteria",
    "VectorSearchConfig",
    "VectorSearchIntegrationConfig",
]
