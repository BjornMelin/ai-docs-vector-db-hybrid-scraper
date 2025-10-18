"""Response models for MCP server tools."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Generic / Utility models
# ---------------------------------------------------------------------------


class OperationStatus(BaseModel):
    """Generic operation status model used for simple success/failure responses."""

    status: str = Field(..., description="Operation status, e.g. 'success' or 'error'")
    message: str | None = Field(
        default=None, description="Optional human-friendly status or error message"
    )
    details: dict[str, Any] | None = Field(
        default=None, description="Optional additional details about the operation"
    )


# Fallback model for miscellaneous key/value responses where the exact schema
# is either dynamic or not yet formalized. Tools should migrate away from this
# over time, but it provides a typed envelope while that work is in progress.


class GenericDictResponse(BaseModel):
    """Arbitrary key/value response allowing forward-compatibility."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Analytics / Monitoring
# ---------------------------------------------------------------------------


class AnalyticsResponse(BaseModel):
    """Response model returned by the analytics.get_analytics tool."""

    timestamp: str
    collections: dict[str, Any]
    cache_metrics: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    costs: dict[str, Any] | None = None

    # Allow forward-compatibility for any extra keys that might be added later.
    model_config = ConfigDict(extra="allow")


class SystemHealthServiceStatus(BaseModel):
    """Status for an individual dependency/service."""

    status: str
    error: str | None = None
    # Additional arbitrary service-specific metrics
    model_config = ConfigDict(extra="allow")


class SystemHealthResponse(BaseModel):
    """Aggregated system health response returned by analytics.get_system_health."""

    status: str
    timestamp: str
    services: dict[str, SystemHealthServiceStatus]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class CacheClearResponse(BaseModel):
    """Response after clearing cache via cache.clear_cache."""

    status: str
    cleared_count: int
    pattern: str | None = None


class CacheStatsResponse(BaseModel):
    """Cache statistics as returned by cache.get_cache_stats."""

    hit_rate: float | None = None
    size: int | None = None
    total_requests: int | None = None

    # Carry through unknown vendor-specific metrics.
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CollectionInfo(BaseModel):
    """Metadata for a single Qdrant collection."""

    name: str
    vectors_count: int | None = None
    points_count: int | None = None
    status: str | None = None

    model_config = ConfigDict(extra="allow")


# The list_collections tool returns a list[CollectionInfo]


class CollectionOperationResponse(OperationStatus):
    """Status response for collection-level operations like delete/optimize."""

    collection: str | None = None


# ---------------------------------------------------------------------------
# Documents / Embeddings / Projects / Misc
# ---------------------------------------------------------------------------


class AddDocumentResponse(BaseModel):
    """Result of adding a single document."""

    url: str
    title: str | None = None
    chunks_created: int
    collection: str
    chunking_strategy: str
    embedding_dimensions: int

    model_config = ConfigDict(extra="allow")


class EmbeddingGenerationResponse(BaseModel):
    """Embeddings generation result."""

    embeddings: list[list[float]]
    sparse_embeddings: list[list[float]] | None = None
    model: str | None = None
    provider: str | None = None
    cost_estimate: float | None = None
    total_tokens: int | None = None
    model_config = ConfigDict(extra="allow")


class EmbeddingProviderInfo(BaseModel):
    """Metadata describing an embedding provider."""

    name: str
    dims: int | None = None
    context_length: int | None = None
    model_config = ConfigDict(extra="allow")


# list_embedding_providers returns list[EmbeddingProviderInfo]


class ReindexCollectionResponse(OperationStatus):
    """Response from payload_indexing.reindex_collection."""

    collection: str
    reindexed_count: int | None = None


class ProjectInfo(BaseModel):
    """Project metadata."""

    id: str
    name: str
    description: str | None = None
    created_at: str | None = None
    model_config = ConfigDict(extra="allow")


# create_project returns ProjectInfo
# list_projects returns list[ProjectInfo]


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


class DocumentBatchResponse(BaseModel):
    """Response for batch document ingestion."""

    successful: list[AddDocumentResponse]
    failed: list[str]
    total: int


# ---------------------------------------------------------------------------
# Content Intelligence
# ---------------------------------------------------------------------------


class ContentIntelligenceResult(BaseModel):
    """Response from content intelligence analysis."""

    success: bool = Field(default=True, description="Whether analysis succeeded")
    enriched_content: Any | None = Field(
        default=None, description="Enriched content with intelligence analysis"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Total processing time in milliseconds"
    )
    cache_hit: bool = Field(
        default=False, description="Whether result was retrieved from cache"
    )
    error: str | None = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(extra="allow")
