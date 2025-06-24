import typing


"""Centralized Pydantic v2 models for the AI Documentation Vector DB.

This module provides a unified import location for all Pydantic models used throughout
the application, organized by domain and purpose.

Usage:
    from models import Config, SearchRequest, VectorSearchConfig
    from models.configuration import QdrantConfig, CacheConfig
    from models.vector_search import SearchParams, PrefetchConfig
"""

# Configuration models
# API contract models
from ..config import (
    CacheConfig,
    ChunkingConfig,
    Config,
    Crawl4AIConfig,
    DocumentationSite,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    HyDEConfig,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    SecurityConfig,
    get_config,
    reset_config,
    set_config,
)


# Legacy compatibility - these were removed in consolidation
class CollectionHNSWConfigs:
    """Legacy compatibility - removed during config simplification."""
    pass

class HNSWConfig:
    """Legacy compatibility - removed during config simplification."""
    pass

class ModelBenchmark:
    """Legacy compatibility - removed during config simplification."""
    pass

class SmartSelectionConfig:
    """Legacy compatibility - removed during config simplification."""
    pass
from .api_contracts import (
    AdvancedSearchRequest,
    AnalyticsRequest,
    AnalyticsResponse,
    BulkDocumentRequest,
    BulkDocumentResponse,
    CacheRequest,
    CacheResponse,
    CollectionInfo,
    CollectionRequest,
    CollectionResponse,
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    HealthCheckResponse,
    ListCollectionsResponse,
    MCPRequest,
    MCPResponse,
    MetricData,
    SearchRequest,
    SearchResultItem,
    ValidationRequest,
    ValidationResponse,
)

# Document processing models
from .document_processing import (
    Chunk,
    ChunkType,
    CodeBlock,
    CodeLanguage,
    ContentFilter,
    DocumentBatch,
    DocumentMetadata,
    ProcessedDocument,
    ScrapingStats,
    VectorMetrics,
)

# Shared validators and utilities
from .validators import (
    CollectionName,
    NonNegativeInt,
    Percentage,
    PortNumber,
    PositiveInt,
    firecrawl_api_key_validator,
    openai_api_key_validator,
    url_validator,
    validate_api_key_common,
    validate_cache_ttl,
    validate_chunk_sizes,
    validate_collection_name,
    validate_embedding_model_name,
    validate_model_benchmark_consistency,
    validate_percentage,
    validate_positive_int,
    validate_rate_limit_config,
    validate_scoring_weights,
    validate_url_format,
    validate_vector_dimensions,
)

# Vector search models
from .vector_search import (
    AdaptiveSearchParams,
    CollectionStats,
    ContentTypeSearchCriteria,
    EnhancedFilteredSearchRequest,
    EnhancedSearchResponse,
    EnhancedSearchResult,
    FilteredSearchRequest,
    FusionConfig,
    HybridSearchRequest,
    HyDESearchRequest,
    IndexingRequest,
    IntegratedSearchRequest,
    MetadataSearchCriteria,
    MultiStageSearchRequest,
    OptimizationRequest,
    PrefetchConfig,
    RetrievalMetrics,
    SearchParams,
    SearchResponse,
    SearchResult,
    SearchStage,
    SimilarityThresholdCriteria,
    TemporalSearchCriteria,
    VectorSearchConfig,
    VectorSearchIntegrationConfig,
)


# Commonly used exports
__all__ = [
    # Vector Search
    "AdaptiveSearchParams",
    # API Contracts
    "AdvancedSearchRequest",
    "AnalyticsRequest",
    "AnalyticsResponse",
    "BulkDocumentRequest",
    "BulkDocumentResponse",
    # Configuration
    "CacheConfig",
    "CacheRequest",
    "CacheResponse",
    # Document Processing
    "Chunk",
    "ChunkType",
    "ChunkingConfig",
    "CodeBlock",
    "CodeLanguage",
    "CollectionHNSWConfigs",
    "CollectionInfo",
    # Validators and Utilities
    "CollectionName",
    "CollectionRequest",
    "CollectionResponse",
    "CollectionStats",
    "ContentFilter",
    "ContentTypeSearchCriteria",
    "Crawl4AIConfig",
    "DocumentBatch",
    "DocumentMetadata",
    "DocumentRequest",
    "DocumentResponse",
    "DocumentationSite",
    "EmbeddingConfig",
    "EnhancedFilteredSearchRequest",
    "EnhancedSearchResponse",
    "EnhancedSearchResult",
    "ErrorResponse",
    "FastEmbedConfig",
    "FilteredSearchRequest",
    "FirecrawlConfig",
    "FusionConfig",
    "HNSWConfig",
    "HealthCheckResponse",
    "HyDEConfig",
    "HyDESearchRequest",
    "HybridSearchRequest",
    "IndexingRequest",
    "IntegratedSearchRequest",
    "ListCollectionsResponse",
    "MCPRequest",
    "MCPResponse",
    "MetadataSearchCriteria",
    "MetricData",
    "ModelBenchmark",
    "MultiStageSearchRequest",
    "NonNegativeInt",
    "OpenAIConfig",
    "OptimizationRequest",
    "Percentage",
    "PerformanceConfig",
    "PortNumber",
    "PositiveInt",
    "PrefetchConfig",
    "ProcessedDocument",
    "QdrantConfig",
    "RetrievalMetrics",
    "ScrapingStats",
    "SearchParams",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchResultItem",
    "SearchStage",
    "SecurityConfig",
    "SimilarityThresholdCriteria",
    "SmartSelectionConfig",
    "TemporalSearchCriteria",
    "Config",
    "ValidationRequest",
    "ValidationResponse",
    "VectorMetrics",
    "VectorSearchConfig",
    "VectorSearchIntegrationConfig",
    "firecrawl_api_key_validator",
    "get_config",
    "openai_api_key_validator",
    "reset_config",
    "set_config",
    "url_validator",
    "validate_api_key_common",
    "validate_cache_ttl",
    "validate_chunk_sizes",
    "validate_collection_name",
    "validate_embedding_model_name",
    "validate_model_benchmark_consistency",
    "validate_percentage",
    "validate_positive_int",
    "validate_rate_limit_config",
    "validate_scoring_weights",
    "validate_url_format",
    "validate_vector_dimensions",
]
