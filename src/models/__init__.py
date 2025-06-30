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
from src.config import (
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

from .api_contracts import (
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
    FilteredSearchRequest,
    FilteredSearchResponse,
    FilteredSearchResult,
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


# Legacy compatibility - these were removed in consolidation
class CollectionHNSWConfigs:
    """Legacy compatibility - removed during config simplification."""


class HNSWConfig:
    """Legacy compatibility - removed during config simplification."""


class ModelBenchmark:
    """Legacy compatibility - removed during config simplification."""


class SmartSelectionConfig:
    """Legacy compatibility - removed during config simplification."""


# Commonly used exports
__all__ = [
    # Vector Search
    "AdaptiveSearchParams",
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
    "Config",
    "ContentFilter",
    "ContentTypeSearchCriteria",
    "Crawl4AIConfig",
    "DocumentBatch",
    "DocumentMetadata",
    "DocumentRequest",
    "DocumentResponse",
    "DocumentationSite",
    "EmbeddingConfig",
    "ErrorResponse",
    "FastEmbedConfig",
    "FilteredSearchRequest",
    "FilteredSearchResponse",
    "FilteredSearchResult",
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
    # API Contracts
    "SearchRequest",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchResultItem",
    "SearchStage",
    "SecurityConfig",
    "SimilarityThresholdCriteria",
    "SmartSelectionConfig",
    "TemporalSearchCriteria",
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
