"""Centralized Pydantic v2 models for the AI Documentation Vector DB.

This module provides a unified import location for all Pydantic models used throughout
the application, organized by domain and purpose.

Usage:
    from models import Config, SearchRequest, VectorSearchConfig
    from models.configuration import QdrantConfig, CacheConfig
    from models.vector_search import SecureSearchParamsModel, PrefetchConfig
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
    SearchResponse as ApiSearchResponse,
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
    collection_name_field,
    firecrawl_api_key_validator,
    non_negative_int,
    openai_api_key_validator,
    percentage,
    port_number,
    positive_int,
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
    # Request models
    AdvancedFilteredSearchRequest,
    AdvancedHybridSearchRequest,
    # Async models
    AsyncSearchContext,
    BasicSearchRequest,
    BatchSearchRequest,
    BatchSearchResponse,
    DimensionError,
    FilterValidationError,
    FusionAlgorithm,
    HyDESearchRequest,
    MultiStageSearchRequest,
    SearchAccuracy,
    SearchConfigurationError,
    SearchResponse,
    # Stage models
    SearchStage,
    # Base classes and enums
    SecureBaseModel,
    SecureFilterGroupModel,
    SecureFilterModel,
    # Security models
    SecureMetadataModel,
    SecurePayloadModel,
    SecureSearchParamsModel,
    # Response models
    SecureSearchResult,
    SecureSparseVectorModel,
    # Vector models
    SecureVectorModel,
    SecurityValidationError,
    # Exception classes
    VectorSearchError,
    VectorType,
)


# Commonly used exports
__all__ = [
    # Vector Search - Request models
    "AdvancedFilteredSearchRequest",
    "AdvancedHybridSearchRequest",
    # API Contracts
    "AnalyticsRequest",
    "AnalyticsResponse",
    # Vector Search - Async models
    "AsyncSearchContext",
    "BasicSearchRequest",
    "BatchSearchRequest",
    "BatchSearchResponse",
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
    "CollectionInfo",
    "CollectionRequest",
    "CollectionResponse",
    "Config",
    "ContentFilter",
    "Crawl4AIConfig",
    "DimensionError",
    "DocumentBatch",
    "DocumentMetadata",
    "DocumentRequest",
    "DocumentResponse",
    "DocumentationSite",
    "EmbeddingConfig",
    "ErrorResponse",
    "FastEmbedConfig",
    "FilterValidationError",
    "FirecrawlConfig",
    "FusionAlgorithm",
    "HealthCheckResponse",
    "HyDEConfig",
    "HyDESearchRequest",
    "ListCollectionsResponse",
    "MCPRequest",
    "MCPResponse",
    "MetricData",
    "MultiStageSearchRequest",
    "OpenAIConfig",
    "PerformanceConfig",
    "ProcessedDocument",
    "QdrantConfig",
    "ScrapingStats",
    "SearchAccuracy",
    "SearchConfigurationError",
    # API Contracts
    "SearchRequest",
    "SearchResponse",
    "ApiSearchResponse",
    "SearchResultItem",
    # Vector Search - Stage models
    "SearchStage",
    # Vector Search - Base classes and enums
    "SecureBaseModel",
    "SecureFilterGroupModel",
    "SecureFilterModel",
    # Vector Search - Security models
    "SecureMetadataModel",
    "SecurePayloadModel",
    "SecureSearchParamsModel",
    # Vector Search - Response models
    "SecureSearchResult",
    "SecureSparseVectorModel",
    # Vector Search - Vector models
    "SecureVectorModel",
    "SecurityConfig",
    "SecurityValidationError",
    "ValidationRequest",
    "ValidationResponse",
    "VectorMetrics",
    # Vector Search - Exception classes
    "VectorSearchError",
    "VectorType",
    # Validators and Utilities
    "collection_name_field",
    "firecrawl_api_key_validator",
    "get_config",
    "non_negative_int",
    "openai_api_key_validator",
    "percentage",
    "port_number",
    "positive_int",
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
