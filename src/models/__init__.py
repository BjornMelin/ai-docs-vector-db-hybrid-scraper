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
from ..config import CacheConfig
from ..config import ChunkingConfig
from ..config import Crawl4AIConfig
from ..config import DocumentationSite
from ..config import EmbeddingConfig
from ..config import FastEmbedConfig
from ..config import FirecrawlConfig
from ..config import HyDEConfig
from ..config import OpenAIConfig
from ..config import PerformanceConfig
from ..config import QdrantConfig
from ..config import SecurityConfig
from ..config import UnifiedConfig
from ..config import get_config
from ..config import reset_config
from ..config import set_config

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

# Document processing models
from .document_processing import Chunk
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
from .vector_search import ContentTypeSearchCriteria
from .vector_search import EnhancedFilteredSearchRequest
from .vector_search import EnhancedSearchResponse
from .vector_search import EnhancedSearchResult
from .vector_search import FilteredSearchRequest
from .vector_search import FusionConfig
from .vector_search import HybridSearchRequest
from .vector_search import HyDESearchRequest
from .vector_search import IndexingRequest
from .vector_search import IntegratedSearchRequest
from .vector_search import MetadataSearchCriteria
from .vector_search import MultiStageSearchRequest
from .vector_search import OptimizationRequest
from .vector_search import PrefetchConfig
from .vector_search import RetrievalMetrics
from .vector_search import SearchParams
from .vector_search import SearchResponse
from .vector_search import SearchResult
from .vector_search import SearchStage
from .vector_search import SimilarityThresholdCriteria
from .vector_search import TemporalSearchCriteria
from .vector_search import VectorSearchConfig
from .vector_search import VectorSearchIntegrationConfig

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
    "UnifiedConfig",
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
