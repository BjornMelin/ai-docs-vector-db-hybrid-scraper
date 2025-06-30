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
    # Async models
    AsyncSearchContext,
    # Request models
    BasicSearchRequest,
    BatchSearchRequest,
    BatchSearchResponse,
    DimensionError,
    FilteredSearchRequest,
    FilterModel,
    FilterValidationError,
    FusionAlgorithm,
    HybridSearchRequest,
    HyDESearchRequest,
    MetadataModel,
    MultiStageSearchRequest,
    PayloadModel,
    SearchAccuracy,
    SearchConfigurationError,
    # Legacy aliases
    SearchParams,
    SearchResponse,
    # Response models
    SearchResult,
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
    SecureSparseVectorModel,
    # Vector models
    SecureVectorModel,
    SecurityValidationError,
    SparseVectorModel,
    VectorModel,
    # Exception classes
    VectorSearchError,
    VectorType,
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
    # API Contracts
    "AnalyticsRequest",
    "AnalyticsResponse",
    # Vector Search - Async models
    "AsyncSearchContext",
    # Vector Search - Request models
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
    "CollectionHNSWConfigs",
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
    "FilterModel",
    "FilterValidationError",
    "FilteredSearchRequest",
    "FirecrawlConfig",
    "FusionAlgorithm",
    "HNSWConfig",
    "HealthCheckResponse",
    "HyDEConfig",
    "HyDESearchRequest",
    "HybridSearchRequest",
    "ListCollectionsResponse",
    "MCPRequest",
    "MCPResponse",
    "MetadataModel",
    "MetricData",
    "ModelBenchmark",
    "MultiStageSearchRequest",
    "OpenAIConfig",
    "PayloadModel",
    "PerformanceConfig",
    "ProcessedDocument",
    "QdrantConfig",
    "ScrapingStats",
    "SearchAccuracy",
    "SearchConfigurationError",
    # Vector Search - Legacy aliases
    "SearchParams",
    # API Contracts
    "SearchRequest",
    "SearchResponse",
    # Vector Search - Response models
    "SearchResult",
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
    "SecureSparseVectorModel",
    # Vector Search - Vector models
    "SecureVectorModel",
    "SecurityConfig",
    "SecurityValidationError",
    "SmartSelectionConfig",
    "SparseVectorModel",
    "ValidationRequest",
    "ValidationResponse",
    "VectorMetrics",
    "VectorModel",
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
