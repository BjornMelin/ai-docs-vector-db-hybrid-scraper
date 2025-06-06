"""Request models for MCP server tools."""

from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ...config.enums import ChunkingStrategy
from ...config.enums import FusionAlgorithm
from ...config.enums import SearchAccuracy
from ...config.enums import SearchStrategy
from ...config.enums import VectorType


class SearchRequest(BaseModel):
    """Search request with advanced options"""

    query: str = Field(..., min_length=1, description="Search query")
    collection: str = Field(
        default="documentation", min_length=1, description="Collection to search"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    strategy: SearchStrategy = Field(
        default=SearchStrategy.HYBRID, description="Search strategy"
    )
    enable_reranking: bool = Field(default=True, description="Enable BGE reranking")
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    # New advanced search options
    fusion_algorithm: FusionAlgorithm = Field(
        default=FusionAlgorithm.RRF, description="Fusion algorithm for hybrid search"
    )
    search_accuracy: SearchAccuracy = Field(
        default=SearchAccuracy.BALANCED, description="Search accuracy level"
    )
    embedding_model: str | None = Field(
        default=None, description="Specific embedding model to use"
    )
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")
    rerank: bool = Field(default=True, description="Enable reranking")
    cache_ttl: int | None = Field(default=None, description="Cache TTL in seconds")


class EmbeddingRequest(BaseModel):
    """Embedding generation request"""

    texts: list[str] = Field(..., description="Texts to embed")
    model: str | None = Field(default=None, description="Specific model to use")
    batch_size: int = Field(default=32, ge=1, le=100, description="Batch size")
    generate_sparse: bool = Field(
        default=False, description="Generate sparse embeddings"
    )


class DocumentRequest(BaseModel):
    """Document processing request"""

    url: str = Field(..., min_length=1, description="Document URL")
    collection: str = Field(
        default="documentation", min_length=1, description="Target collection"
    )
    chunk_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    chunk_size: int = Field(default=1600, ge=100, le=4000, description="Chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Chunk overlap")
    extract_metadata: bool = Field(
        default=True, description="Extract document metadata"
    )


class BatchRequest(BaseModel):
    """Batch document processing request"""

    urls: list[str] = Field(..., description="Document URLs")
    collection: str = Field(default="documentation", description="Target collection")
    chunk_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Max concurrent")


class ProjectRequest(BaseModel):
    """Project creation request"""

    name: str = Field(..., description="Project name")
    description: str | None = Field(default=None, description="Project description")
    quality_tier: str = Field(
        default="balanced",
        description="Quality tier (economy/balanced/premium)",
        pattern="^(economy|balanced|premium)$",
    )
    urls: list[str] | None = Field(default=None, description="Initial URLs to process")


class CostEstimateRequest(BaseModel):
    """Cost estimation request"""

    texts: list[str] = Field(..., description="Texts to estimate")
    provider: str | None = Field(default=None, description="Specific provider")
    include_reranking: bool = Field(default=False, description="Include reranking cost")


class AnalyticsRequest(BaseModel):
    """Analytics request"""

    collection: str | None = Field(default=None, description="Specific collection")
    include_performance: bool = Field(
        default=True, description="Include performance metrics"
    )
    include_costs: bool = Field(default=True, description="Include cost analysis")


class HyDESearchRequest(BaseModel):
    """HyDE (Hypothetical Document Embeddings) search request"""

    query: str = Field(..., description="Search query")
    collection: str = Field(default="documentation", description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    domain: str | None = Field(
        default=None, description="Domain hint for better generation"
    )

    # HyDE-specific options
    num_generations: int = Field(
        default=5, ge=1, le=10, description="Number of hypothetical documents"
    )
    generation_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Generation temperature"
    )
    max_generation_tokens: int = Field(
        default=200, ge=50, le=500, description="Max tokens per generation"
    )

    # Search options
    enable_reranking: bool = Field(default=True, description="Enable BGE reranking")
    enable_caching: bool = Field(default=True, description="Enable HyDE caching")
    fusion_algorithm: FusionAlgorithm = Field(
        default=FusionAlgorithm.RRF, description="Fusion algorithm"
    )
    search_accuracy: SearchAccuracy = Field(
        default=SearchAccuracy.BALANCED, description="Search accuracy"
    )

    # Filters and metadata
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )

    # Performance options
    force_hyde: bool = Field(
        default=False, description="Force HyDE even if disabled globally"
    )
    fallback_on_error: bool = Field(
        default=True, description="Fallback to regular search on error"
    )


class MultiStageSearchRequest(BaseModel):
    """Multi-stage search request with Query API prefetch"""

    query: str = Field(..., description="Search query")
    collection: str = Field(default="documentation", description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")

    # Stage configuration
    stages: list[dict[str, Any]] = Field(..., description="Search stages configuration")
    fusion_algorithm: FusionAlgorithm = Field(
        default=FusionAlgorithm.RRF, description="Fusion algorithm"
    )
    search_accuracy: SearchAccuracy = Field(
        default=SearchAccuracy.BALANCED, description="Search accuracy"
    )

    # Options
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")


class FilteredSearchRequest(BaseModel):
    """Filtered search request using indexed payload fields"""

    query: str = Field(..., description="Search query")
    collection: str = Field(default="documentation", description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")

    # Filters
    filters: dict[str, Any] = Field(
        ..., description="Filters to apply using indexed fields"
    )
    search_accuracy: SearchAccuracy = Field(
        default=SearchAccuracy.BALANCED, description="Search accuracy"
    )

    # Options
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")


# Advanced Search Request Models for Query API


class SearchStageRequest(BaseModel):
    """Single stage configuration for multi-stage search"""

    query_vector: list[float] = Field(..., description="Vector for this stage")
    vector_name: str = Field(..., description="Vector field name (dense, sparse, etc.)")
    vector_type: VectorType = Field(..., description="Type of vector for optimization")
    limit: int = Field(..., description="Number of results to retrieve in this stage")
    filters: dict[str, Any] | None = Field(
        None, description="Optional filters for this stage"
    )
