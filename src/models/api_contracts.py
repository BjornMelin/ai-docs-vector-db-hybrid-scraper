import typing
"""API contract models for MCP (Model Context Protocol) requests and responses.

This module consolidates all request and response models used in the MCP server
and API endpoints, providing clear contracts for external interfaces.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


# Base Models
class MCPRequest(BaseModel):
    """Base class for MCP requests."""

    model_config = ConfigDict(extra="forbid")


class MCPResponse(BaseModel):
    """Base class for MCP responses."""

    success: bool = Field(..., description="Whether the operation succeeded")
    timestamp: float = Field(..., description="Response timestamp")

    model_config = ConfigDict(extra="forbid")


class ErrorResponse(MCPResponse):
    """Standard error response format."""

    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_type: str = Field(default="general", description="Error category")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional error context"
    )


# Search API Models
class SearchRequest(MCPRequest):
    """Request model for search operations."""

    query: str = Field(..., description="Search query text", min_length=1)
    collection_name: str = Field(default="documents", description="Target collection")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum score"
    )
    enable_hyde: bool = Field(default=False, description="Enable HyDE enhancement")
    filters: dict[str, Any] | None = Field(default=None, description="Search filters")


class AdvancedSearchRequest(MCPRequest):
    """Request model for advanced search operations."""

    query: str = Field(..., description="Search query text", min_length=1)
    collection_name: str = Field(default="documents", description="Target collection")
    search_strategy: str = Field(
        default="hybrid", description="Search strategy (dense, hybrid, multi_stage)"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    accuracy_level: str = Field(
        default="balanced",
        description="Accuracy level (fast, balanced, accurate, exact)",
    )
    enable_reranking: bool = Field(default=False, description="Enable result reranking")
    hyde_config: dict[str, Any] | None = Field(
        default=None, description="HyDE configuration"
    )
    filters: dict[str, Any] | None = Field(default=None, description="Search filters")


class SearchResultItem(BaseModel):
    """Individual search result item."""

    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    title: str | None = Field(default=None, description="Document title")
    content: str | None = Field(default=None, description="Document content excerpt")
    url: str | None = Field(default=None, description="Document URL")
    doc_type: str | None = Field(default=None, description="Document type")
    language: str | None = Field(default=None, description="Document language")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResponse(MCPResponse):
    """Response model for search operations."""

    success: bool = Field(default=True)
    results: list[SearchResultItem] = Field(
        default_factory=list, description="Search results"
    )
    total_count: int = Field(default=0, description="Total matching documents")
    query_time_ms: float = Field(default=0.0, description="Query execution time")
    search_strategy: str = Field(default="unknown", description="Strategy used")
    cache_hit: bool = Field(default=False, description="Whether result was cached")


# Document Management Models
class DocumentRequest(MCPRequest):
    """Request model for document operations."""

    url: str = Field(..., description="Document URL to process")
    collection_name: str = Field(default="documents", description="Target collection")
    doc_type: str | None = Field(default=None, description="Document type")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    force_recrawl: bool = Field(
        default=False, description="Force recrawling existing document"
    )


class BulkDocumentRequest(MCPRequest):
    """Request model for bulk document operations."""

    urls: list[str] = Field(
        ..., description="List of URLs to process", min_length=1, max_length=100
    )
    collection_name: str = Field(default="documents", description="Target collection")
    doc_type: str | None = Field(default=None, description="Document type for all URLs")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata for all documents"
    )
    force_recrawl: bool = Field(
        default=False, description="Force recrawling existing documents"
    )
    max_concurrent: int = Field(
        default=5, ge=1, le=20, description="Max concurrent processing"
    )


class DocumentResponse(MCPResponse):
    """Response model for document operations."""

    success: bool = Field(default=True)
    document_id: str = Field(..., description="Processed document ID")
    url: str = Field(..., description="Document URL")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    status: str = Field(default="processed", description="Processing status")


class BulkDocumentResponse(MCPResponse):
    """Response model for bulk document operations."""

    success: bool = Field(default=True)
    processed_count: int = Field(
        default=0, description="Successfully processed documents"
    )
    failed_count: int = Field(default=0, description="Failed documents")
    total_chunks: int = Field(default=0, description="Total chunks created")
    processing_time_ms: float = Field(default=0.0, description="Total processing time")
    results: list[DocumentResponse] = Field(
        default_factory=list, description="Individual results"
    )
    errors: list[str] = Field(default_factory=list, description="Processing errors")


# Collection Management Models
class CollectionRequest(MCPRequest):
    """Request model for collection operations."""

    collection_name: str = Field(..., description="Collection name", min_length=1)
    vector_size: int | None = Field(default=None, ge=1, description="Vector dimensions")
    distance_metric: str = Field(default="Cosine", description="Distance metric")
    enable_hybrid: bool = Field(default=True, description="Enable hybrid search")
    hnsw_config: dict[str, Any] | None = Field(
        default=None, description="HNSW configuration"
    )


class CollectionInfo(BaseModel):
    """Collection information model."""

    name: str = Field(..., description="Collection name")
    points_count: int = Field(default=0, description="Number of points")
    vectors_count: int = Field(default=0, description="Number of vectors")
    indexed_fields: list[str] = Field(
        default_factory=list, description="Indexed payload fields"
    )
    status: str = Field(default="unknown", description="Collection status")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Collection configuration"
    )


class CollectionResponse(MCPResponse):
    """Response model for collection operations."""

    success: bool = Field(default=True)
    collection_name: str = Field(..., description="Collection name")
    operation: str = Field(..., description="Operation performed")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Operation details"
    )


class ListCollectionsResponse(MCPResponse):
    """Response model for listing collections."""

    success: bool = Field(default=True)
    collections: list[CollectionInfo] = Field(
        default_factory=list, description="Collection information"
    )
    total_count: int = Field(default=0, description="Total number of collections")


# Analytics Models
class AnalyticsRequest(MCPRequest):
    """Request model for analytics operations."""

    collection_name: str | None = Field(
        default=None, description="Specific collection to analyze"
    )
    time_range: str = Field(default="24h", description="Time range for analytics")
    metric_types: list[str] = Field(
        default_factory=list, description="Specific metrics to include"
    )


class MetricData(BaseModel):
    """Individual metric data point."""

    name: str = Field(..., description="Metric name")
    value: int | float | str = Field(..., description="Metric value")
    unit: str | None = Field(default=None, description="Metric unit")
    timestamp: float | None = Field(default=None, description="Metric timestamp")


class AnalyticsResponse(MCPResponse):
    """Response model for analytics operations."""

    success: bool = Field(default=True)
    metrics: list[MetricData] = Field(
        default_factory=list, description="Analytics metrics"
    )
    time_range: str = Field(..., description="Time range analyzed")
    generated_at: float = Field(..., description="Report generation time")


# Utilities Models
class HealthCheckResponse(MCPResponse):
    """Response model for health check."""

    success: bool = Field(default=True)
    status: str = Field(default="healthy", description="Overall health status")
    services: dict[str, str] = Field(
        default_factory=dict, description="Individual service status"
    )
    uptime_seconds: float = Field(default=0.0, description="Service uptime")
    version: str = Field(default="unknown", description="Service version")


class CacheRequest(MCPRequest):
    """Request model for cache operations."""

    operation: str = Field(..., description="Cache operation (clear, stats, warm)")
    cache_type: str | None = Field(default=None, description="Specific cache type")
    keys: list[str] | None = Field(default=None, description="Specific cache keys")


class CacheResponse(MCPResponse):
    """Response model for cache operations."""

    success: bool = Field(default=True)
    operation: str = Field(..., description="Operation performed")
    affected_keys: int = Field(default=0, description="Number of affected cache keys")
    cache_stats: dict[str, Any] = Field(
        default_factory=dict, description="Cache statistics"
    )


# Validation Models
class ValidationRequest(MCPRequest):
    """Request model for configuration validation."""

    config_section: str | None = Field(
        default=None, description="Specific config section"
    )
    validate_connections: bool = Field(
        default=True, description="Test external connections"
    )


class ValidationResponse(MCPResponse):
    """Response model for configuration validation."""

    success: bool = Field(default=True)
    valid: bool = Field(..., description="Whether configuration is valid")
    issues: list[str] = Field(
        default_factory=list, description="Validation issues found"
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    tested_services: list[str] = Field(
        default_factory=list, description="Services tested"
    )


# Export all models
__all__ = [
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
    "SearchResponse",
    "SearchResultItem",
    "ValidationRequest",
    "ValidationResponse",
]
