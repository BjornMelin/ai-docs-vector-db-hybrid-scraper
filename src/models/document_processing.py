"""Document processing models for content ingestion and chunking.

This module contains all models related to document processing, chunking,
and content representation in the vector database system.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from src.config import ChunkingStrategy, DocumentStatus


class CodeLanguage(str, Enum):
    """Supported programming languages for AST parsing."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """Types of content chunks."""

    TEXT = "text"
    CODE = "code"
    MIXED = "mixed"
    HEADER = "header"
    METADATA = "metadata"


@dataclass
class CodeBlock:
    """Represents a code block found in content."""

    language: str
    content: str
    start_pos: int
    end_pos: int
    fence_type: str = "```"  # Could be ``` or ~~~


@dataclass
class Chunk:
    """Enhanced chunk with metadata for document processing."""

    content: str
    start_pos: int
    end_pos: int
    chunk_index: int
    total_chunks: int = 0  # Updated after all chunks created
    char_count: int = 0
    token_estimate: int = 0  # Rough estimate: chars / 4
    chunk_type: str = "text"  # text, code, mixed
    language: str | None = None
    has_code: bool = False
    metadata: dict[str, Any] | None = None


class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""

    title: str | None = Field(default=None, description="Document title")
    url: str = Field(..., description="Document URL")
    doc_type: str | None = Field(default=None, description="Document type")
    language: str | None = Field(default=None, description="Programming language")
    crawled_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Crawl timestamp"
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
    """Vector processing metrics with comprehensive analytics."""

    total_documents: int = Field(
        default=0, ge=0, description="Total documents processed"
    )
    total_chunks: int = Field(default=0, ge=0, description="Total chunks created")
    successful_embeddings: int = Field(
        default=0, ge=0, description="Successful embeddings"
    )
    failed_embeddings: int = Field(default=0, ge=0, description="Failed embeddings")
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Processing time in seconds"
    )
    tokens_processed: int = Field(default=0, ge=0, description="Total tokens processed")
    avg_chunk_size: float = Field(default=0.0, ge=0.0, description="Average chunk size")

    @field_validator("avg_chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: float) -> float:
        """Validate chunk size is reasonable."""
        if v > 50000:  # 50k characters seems excessive
            msg = "Average chunk size exceeds reasonable bounds"
            raise ValueError(msg)
        return round(v, 2)

    @model_validator(mode="after")
    def validate_embedding_consistency(self) -> "VectorMetrics":
        """Ensure embedding counts don't exceed chunk counts."""
        total_embeddings = self.successful_embeddings + self.failed_embeddings
        if total_embeddings > self.total_chunks:
            msg = "Total embeddings cannot exceed total chunks"
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def total_embeddings_attempted(self) -> int:
        """Total embedding attempts (successful + failed)."""
        return self.successful_embeddings + self.failed_embeddings

    @computed_field
    @property
    def success_rate(self) -> float:
        """Embedding success rate as percentage."""
        total = self.total_embeddings_attempted
        if total == 0:
            return 0.0
        return round((self.successful_embeddings / total) * 100, 2)

    @computed_field
    @property
    def failure_rate(self) -> float:
        """Embedding failure rate as percentage."""
        return round(100.0 - self.success_rate, 2)

    @computed_field
    @property
    def chunks_per_document(self) -> float:
        """Average chunks per document."""
        if self.total_documents == 0:
            return 0.0
        return round(self.total_chunks / self.total_documents, 2)

    @computed_field
    @property
    def tokens_per_chunk(self) -> float:
        """Average tokens per chunk."""
        if self.total_chunks == 0:
            return 0.0
        return round(self.tokens_processed / self.total_chunks, 1)

    @computed_field
    @property
    def processing_efficiency(self) -> str:
        """Categorize processing efficiency."""
        if self.total_documents == 0:
            return "unknown"

        docs_per_second = self.total_documents / max(self.processing_time, 1)

        if docs_per_second > 10:
            return "excellent"
        if docs_per_second > 5:
            return "good"
        if docs_per_second > 1:
            return "moderate"
        return "slow"

    @computed_field
    @property
    def quality_score(self) -> float:
        """Combined quality score based on success rate and efficiency."""
        success_component = self.success_rate / 100.0
        efficiency_scores = {
            "excellent": 1.0,
            "good": 0.8,
            "moderate": 0.6,
            "slow": 0.3,
            "unknown": 0.0,
        }
        efficiency_component = efficiency_scores.get(self.processing_efficiency, 0.0)

        return round((success_component * 0.7 + efficiency_component * 0.3), 3)

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "total_documents": 100,
                    "total_chunks": 500,
                    "successful_embeddings": 480,
                    "failed_embeddings": 20,
                    "processing_time": 45.2,
                    "tokens_processed": 125000,
                    "avg_chunk_size": 250.0,
                }
            ]
        },
    )


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
        default_factory=lambda: datetime.now(UTC), description="Batch creation time"
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
    "Chunk",
    "ChunkType",
    "CodeBlock",
    "CodeLanguage",
    "ContentFilter",
    "DocumentBatch",
    "DocumentMetadata",
    "ProcessedDocument",
    "ScrapingStats",
    "VectorMetrics",
]
