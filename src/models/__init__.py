"""Centralized Pydantic v2 models for the AI Documentation Vector DB.

This module provides a unified import location for all Pydantic models used throughout
the application, organized by domain and purpose.

Usage:
    from models import UnifiedConfig, SearchRequest, VectorSearchConfig
    from models.configuration import QdrantConfig, CacheConfig
    from models.vector_search import SearchParams, PrefetchConfig
"""

# Configuration models
# API contract models
from .api_contracts import AdvancedSearchRequest
from .api_contracts import AnalyticsRequest
from .api_contracts import AnalyticsResponse
from .api_contracts import BulkDocumentRequest
from .api_contracts import BulkDocumentResponse
from .api_contracts import CacheRequest
from .api_contracts import CacheResponse
from .api_contracts import CollectionInfo
from .api_contracts import CollectionRequest
from .api_contracts import CollectionResponse
from .api_contracts import DocumentRequest
from .api_contracts import DocumentResponse
from .api_contracts import ErrorResponse
from .api_contracts import HealthCheckResponse
from .api_contracts import ListCollectionsResponse
from .api_contracts import MCPRequest
from .api_contracts import MCPResponse
from .api_contracts import MetricData
from .api_contracts import SearchRequest
from .api_contracts import SearchResultItem
from .api_contracts import ValidationRequest
from .api_contracts import ValidationResponse
from .configuration import CacheConfig
from .configuration import ChunkingConfig
from .configuration import CollectionHNSWConfigs
from .configuration import Crawl4AIConfig
from .configuration import DocumentationSite
from .configuration import EmbeddingConfig
from .configuration import FastEmbedConfig
from .configuration import FirecrawlConfig
from .configuration import HNSWConfig
from .configuration import HyDEConfig
from .configuration import ModelBenchmark
from .configuration import OpenAIConfig
from .configuration import PerformanceConfig
from .configuration import QdrantConfig
from .configuration import SecurityConfig
from .configuration import SmartSelectionConfig
from .configuration import UnifiedConfig
from .configuration import get_config
from .configuration import reset_config
from .configuration import set_config

# Document processing models
from .document_processing import Chunk
from .document_processing import ChunkingConfiguration
from .document_processing import ChunkType
from .document_processing import CodeBlock
from .document_processing import CodeLanguage
from .document_processing import ContentFilter
from .document_processing import DocumentBatch
from .document_processing import DocumentMetadata
from .document_processing import ProcessedDocument
from .document_processing import ScrapingStats
from .document_processing import VectorMetrics

# Shared validators and utilities
from .validators import CollectionName
from .validators import NonNegativeInt
from .validators import Percentage
from .validators import PortNumber
from .validators import PositiveInt
from .validators import firecrawl_api_key_validator
from .validators import openai_api_key_validator
from .validators import url_validator
from .validators import validate_api_key_common
from .validators import validate_cache_ttl
from .validators import validate_chunk_sizes
from .validators import validate_collection_name
from .validators import validate_embedding_model_name
from .validators import validate_model_benchmark_consistency
from .validators import validate_percentage
from .validators import validate_positive_int
from .validators import validate_rate_limit_config
from .validators import validate_scoring_weights
from .validators import validate_url_format
from .validators import validate_vector_dimensions

# Vector search models
from .vector_search import AdaptiveSearchParams
from .vector_search import CollectionStats
from .vector_search import FilteredSearchRequest
from .vector_search import FusionConfig
from .vector_search import HybridSearchRequest
from .vector_search import HyDESearchRequest
from .vector_search import IndexingRequest
from .vector_search import MultiStageSearchRequest
from .vector_search import OptimizationRequest
from .vector_search import PrefetchConfig
from .vector_search import RetrievalMetrics
from .vector_search import SearchParams
from .vector_search import SearchResponse
from .vector_search import SearchResult
from .vector_search import SearchStage
from .vector_search import VectorSearchConfig

# Commonly used exports
__all__ = [
    # Configuration
    "CacheConfig",
    "ChunkingConfig",
    "CollectionHNSWConfigs",
    "Crawl4AIConfig",
    "DocumentationSite",
    "EmbeddingConfig",
    "FastEmbedConfig",
    "FirecrawlConfig",
    "HNSWConfig",
    "HyDEConfig",
    "ModelBenchmark",
    "OpenAIConfig",
    "PerformanceConfig",
    "QdrantConfig",
    "SecurityConfig",
    "SmartSelectionConfig",
    "UnifiedConfig",
    "get_config",
    "reset_config",
    "set_config",
    # Vector Search
    "AdaptiveSearchParams",
    "CollectionStats",
    "FilteredSearchRequest",
    "FusionConfig",
    "HyDESearchRequest",
    "HybridSearchRequest",
    "IndexingRequest",
    "MultiStageSearchRequest",
    "OptimizationRequest",
    "PrefetchConfig",
    "RetrievalMetrics",
    "SearchParams",
    "SearchResponse",
    "SearchResult",
    "SearchStage",
    "VectorSearchConfig",
    # API Contracts
    "AdvancedSearchRequest",
    "AnalyticsRequest",
    "AnalyticsResponse",
    "BulkDocumentRequest",
    "BulkDocumentResponse",
    "CacheRequest",
    "CacheResponse",
    "CollectionInfo",
    "CollectionRequest",
    "CollectionResponse",
    "DocumentRequest",
    "DocumentResponse",
    "ErrorResponse",
    "HealthCheckResponse",
    "ListCollectionsResponse",
    "MCPRequest",
    "MCPResponse",
    "MetricData",
    "SearchRequest",
    "SearchResultItem",
    "ValidationRequest",
    "ValidationResponse",
    # Document Processing
    "Chunk",
    "ChunkingConfiguration",
    "ChunkType",
    "CodeBlock",
    "CodeLanguage",
    "ContentFilter",
    "DocumentBatch",
    "DocumentMetadata",
    "ProcessedDocument",
    "ScrapingStats",
    "VectorMetrics",
    # Validators and Utilities
    "CollectionName",
    "NonNegativeInt",
    "Percentage",
    "PortNumber",
    "PositiveInt",
    "firecrawl_api_key_validator",
    "openai_api_key_validator",
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
