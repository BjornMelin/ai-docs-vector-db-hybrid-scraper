"""Consolidated enums for the AI Documentation Vector DB system.

This module contains all enums used across the application to ensure
consistency and avoid duplication.
"""

from enum import Enum


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""

    OPENAI = "openai"
    FASTEMBED = "fastembed"


class CrawlProvider(str, Enum):
    """Available crawling providers."""

    CRAWL4AI = "crawl4ai"
    FIRECRAWL = "firecrawl"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    AST = "ast"


class SearchStrategy(str, Enum):
    """Search strategy options."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class EmbeddingModel(str, Enum):
    """Advanced embedding models based on research findings."""

    # OpenAI Models (API-based)
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"  # Best cost-performance
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"  # Best OpenAI performance

    # FastEmbed Models (Local inference, research-backed)
    NV_EMBED_V2 = "nvidia/NV-Embed-v2"  # #1 on MTEB leaderboard
    BGE_SMALL_EN_V15 = "BAAI/bge-small-en-v1.5"  # Cost-effective open source
    BGE_LARGE_EN_V15 = "BAAI/bge-large-en-v1.5"  # Higher accuracy

    # Sparse Models for Hybrid Search
    SPLADE_PP_EN_V1 = "prithvida/Splade_PP_en_v1"  # SPLADE++ for keyword matching


class QualityTier(str, Enum):
    """Quality tiers for project configuration."""

    ECONOMY = "economy"
    BALANCED = "balanced"
    PREMIUM = "premium"


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CollectionStatus(str, Enum):
    """Vector collection status."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class FusionAlgorithm(str, Enum):
    """Fusion algorithms for combining multiple search results."""

    RRF = "rrf"  # Reciprocal Rank Fusion - best for hybrid search
    DBSF = "dbsf"  # Distribution-Based Score Fusion - best for similar vectors


class SearchAccuracy(str, Enum):
    """Search accuracy levels for HNSW parameter optimization."""

    FAST = "fast"  # HNSW EF=50, prioritize speed
    BALANCED = "balanced"  # HNSW EF=100, balance speed/accuracy
    ACCURATE = "accurate"  # HNSW EF=200, prioritize accuracy
    EXACT = "exact"  # Exact search, disable HNSW


class VectorType(str, Enum):
    """Vector types for multi-stage retrieval."""

    DENSE = "dense"  # Dense embedding vectors
    SPARSE = "sparse"  # Sparse keyword vectors (SPLADE)
    HYDE = "hyde"  # Hypothetical document embeddings


class HttpStatus(int, Enum):
    """HTTP status codes for API responses."""

    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


class CacheType(str, Enum):
    """Cache types for different data patterns."""

    EMBEDDINGS = "embeddings"
    CRAWL = "crawl"
    SEARCH = "search"
    HYDE = "hyde"


# Re-export commonly used enums for backward compatibility
__all__ = [
    "CacheType",
    "ChunkingStrategy",
    "CollectionStatus",
    "CrawlProvider",
    "DocumentStatus",
    "EmbeddingModel",
    "EmbeddingProvider",
    "Environment",
    "FusionAlgorithm",
    "HttpStatus",
    "LogLevel",
    "QualityTier",
    "SearchAccuracy",
    "SearchStrategy",
    "VectorType",
]
