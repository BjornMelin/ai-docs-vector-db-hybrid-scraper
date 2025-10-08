"""Centralized Pydantic v2 models for the AI Documentation Vector DB.

This module provides a unified import location for all Pydantic models used throughout
the application, organized by domain and purpose.

Usage:
    from models import Config
    from models.configuration import QdrantConfig, CacheConfig
    from models.vector_search import SecureSearchParamsModel, PrefetchConfig
"""

# Configuration models
# API contract models
from src.config import (
    CacheConfig,
    ChunkingConfig,
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
    get_settings,
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
    SearchResponse as ApiSearchResponse,
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
from .search import SearchRequest

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
    DimensionError,
    FilterValidationError,
    FusionConfig,
    # Stage models
    SearchStage,
    # Base classes and enums
    SecureBaseModel,
    SecureFilterGroupModel,
    SecureFilterModel,
    SecureSearchParamsModel,
    # Vector models
    SecureSparseVectorModel,
    SecureVectorModel,
    SecurityValidationError,
    # Exception classes
    VectorSearchError,
)


# Commonly used exports
__all__ = [  # noqa: RUF022 - organized by category for readability
    # Vector Search - Request models
    "SearchRequest",
    # API Contracts
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
    "CollectionInfo",
    "CollectionRequest",
    "CollectionResponse",
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
    "FusionConfig",
    "HealthCheckResponse",
    "HyDEConfig",
    "ListCollectionsResponse",
    "MCPRequest",
    "MCPResponse",
    "MetricData",
    "OpenAIConfig",
    "PerformanceConfig",
    "ProcessedDocument",
    "QdrantConfig",
    "ScrapingStats",
    # API Contracts
    "ApiSearchResponse",
    # Vector Search - Stage models
    "SearchStage",
    # Vector Search - Base classes and enums
    "SecureBaseModel",
    "SecureFilterGroupModel",
    "SecureFilterModel",
    "SecureSearchParamsModel",
    # Vector Search - Vector models
    "SecureSparseVectorModel",
    "SecureVectorModel",
    "SecurityConfig",
    "SecurityValidationError",
    "ValidationRequest",
    "ValidationResponse",
    "VectorMetrics",
    # Vector Search - Exception classes
    "VectorSearchError",
    # Validators and Utilities
    "collection_name_field",
    "firecrawl_api_key_validator",
    "get_settings",
    "non_negative_int",
    "openai_api_key_validator",
    "percentage",
    "port_number",
    "positive_int",
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
