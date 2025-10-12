"""Document processing models for content ingestion and chunking.

This module contains all models related to document processing, chunking,
and content representation in the vector database system.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import ChunkingStrategy, DocumentStatus


class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""

    title: str | None = Field(default=None, description="Document title")
    url: str = Field(..., description="Document URL")
    doc_type: str | None = Field(default=None, description="Document type")
    language: str | None = Field(default=None, description="Programming language")
    crawled_at: datetime = Field(
        default_factory=datetime.now, description="Crawl timestamp"
    )
    last_modified: datetime | None = Field(
        default=None, description="Last modification time"
    )
    content_hash: str | None = Field(
        default=None, description="Content hash for deduplication"
    )
    author: str | None = Field(default=None, description="Document author")
    tags: list[str] = Field(default_factory=list, description="Document tags")

    # Content analysis
    word_count: int = Field(default=0, description="Word count")
    char_count: int = Field(default=0, description="Character count")
    has_code: bool = Field(default=False, description="Contains code blocks")
    estimated_reading_time: int = Field(
        default=0, description="Estimated reading time in minutes"
    )

    # Processing metadata
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy used"
    )
    total_chunks: int = Field(default=0, description="Total number of chunks")
    processing_time_ms: float = Field(default=0.0, description="Processing time")

    model_config = ConfigDict(extra="forbid")


class ProcessedDocument(BaseModel):
    """Represents a fully processed document ready for embedding."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks: list[dict[str, Any]] = Field(
        default_factory=list, description="Document chunks"
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Processing status"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )

    # Vector database fields
    dense_vector: list[float] | None = Field(
        default=None, description="Dense embedding vector"
    )
    sparse_vector: dict[str, Any] | None = Field(
        default=None, description="Sparse vector"
    )

    model_config = ConfigDict(extra="forbid")


class VectorMetrics(BaseModel):
    """Vector processing metrics."""

    total_documents: int = Field(default=0, description="Total documents processed")
    total_chunks: int = Field(default=0, description="Total chunks created")
    successful_embeddings: int = Field(default=0, description="Successful embeddings")
    failed_embeddings: int = Field(default=0, description="Failed embeddings")
    processing_time: float = Field(
        default=0.0, description="Processing time in seconds"
    )
    tokens_processed: int = Field(default=0, description="Total tokens processed")
    avg_chunk_size: float = Field(default=0.0, description="Average chunk size")

    model_config = ConfigDict(extra="forbid")


class ScrapingStats(BaseModel):
    """Comprehensive scraping statistics."""

    total_processed: int = Field(default=0, description="Total documents processed")
    successful_embeddings: int = Field(default=0, description="Successful embeddings")
    failed_crawls: int = Field(default=0, description="Failed crawl attempts")
    total_chunks: int = Field(default=0, description="Total chunks created")
    unique_urls: int = Field(default=0, description="Unique URLs processed")
    start_time: datetime | None = Field(
        default=None, description="Processing start time"
    )
    end_time: datetime | None = Field(default=None, description="Processing end time")

    # Performance metrics
    avg_processing_time: float = Field(
        default=0.0, description="Average processing time per document"
    )
    docs_per_minute: float = Field(
        default=0.0, description="Documents processed per minute"
    )
    total_size_mb: float = Field(default=0.0, description="Total content size in MB")

    model_config = ConfigDict(extra="forbid")


class ContentFilter(BaseModel):
    """Configuration for content filtering during processing."""

    min_content_length: int = Field(default=50, description="Minimum content length")
    max_content_length: int = Field(
        default=1_000_000, description="Maximum content length"
    )
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["text/html", "text/plain", "text/markdown"],
        description="Allowed MIME types",
    )
    blocked_domains: list[str] = Field(
        default_factory=list, description="Blocked domains"
    )
    content_patterns_to_exclude: list[str] = Field(
        default_factory=list, description="Regex patterns to exclude from content"
    )

    # Quality filters
    min_word_count: int = Field(default=10, description="Minimum word count")
    max_duplicate_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Maximum allowed duplicate content ratio",
    )

    model_config = ConfigDict(extra="forbid")


class DocumentBatch(BaseModel):
    """Batch of documents for processing."""

    id: str = Field(..., description="Batch identifier")
    documents: list[ProcessedDocument] = Field(
        default_factory=list, description="Documents in batch"
    )
    batch_size: int = Field(default=0, description="Number of documents in batch")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Batch creation time"
    )
    status: str = Field(default="pending", description="Batch processing status")

    # Processing results
    successful_count: int = Field(
        default=0, description="Successfully processed documents"
    )
    failed_count: int = Field(default=0, description="Failed documents")
    total_chunks: int = Field(default=0, description="Total chunks in batch")
    processing_time_ms: float = Field(default=0.0, description="Total processing time")

    model_config = ConfigDict(extra="forbid")


# Export all models
__all__ = [
    "ContentFilter",
    "DocumentBatch",
    "DocumentMetadata",
    "ProcessedDocument",
    "ScrapingStats",
    "VectorMetrics",
]
