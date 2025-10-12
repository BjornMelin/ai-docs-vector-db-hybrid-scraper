"""Unit tests for document processing models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.config import ChunkingConfig, ChunkingStrategy, DocumentStatus
from src.models.document_processing import (
    ContentFilter,
    DocumentBatch,
    DocumentMetadata,
    ProcessedDocument,
    ScrapingStats,
    VectorMetrics,
)


class TestDocumentMetadata:
    """Tests for the DocumentMetadata model."""

    def test_required_fields(self) -> None:
        """URL is required for metadata."""

        metadata = DocumentMetadata(url="https://example.com")
        assert metadata.url == "https://example.com"

        with pytest.raises(ValidationError):
            DocumentMetadata()

    def test_defaults(self) -> None:
        """Default values should align with configuration."""

        before = datetime.now(tz=UTC)
        metadata = DocumentMetadata(url="https://example.com")
        after = datetime.now(tz=UTC)

        assert metadata.title is None
        assert metadata.chunking_strategy == ChunkingStrategy.ENHANCED
        assert metadata.total_chunks == 0
        assert before <= metadata.crawled_at <= after

    def test_custom_values(self) -> None:
        """Custom field values should persist."""

        crawled_at = datetime.now(tz=UTC)
        metadata = DocumentMetadata(
            title="Guide",
            url="https://example.com/guide",
            doc_type="howto",
            language="en",
            crawled_at=crawled_at,
            total_chunks=8,
            chunking_strategy=ChunkingStrategy.BASIC,
            processing_time_ms=120.5,
        )

        assert metadata.title == "Guide"
        assert metadata.language == "en"
        assert metadata.crawled_at == crawled_at
        assert metadata.total_chunks == 8
        assert metadata.chunking_strategy == ChunkingStrategy.BASIC
        assert metadata.processing_time_ms == 120.5


class TestProcessedDocument:
    """Tests for processed document envelopes."""

    def test_defaults(self) -> None:
        """ProcessedDocument should populate optional defaults."""

        metadata = DocumentMetadata(url="https://example.com")
        doc = ProcessedDocument(id="doc-1", content="payload", metadata=metadata)

        assert doc.status is DocumentStatus.PENDING
        assert doc.chunks == []
        assert doc.dense_vector is None

    def test_custom_state(self) -> None:
        """Explicit state assignments should persist."""

        metadata = DocumentMetadata(url="https://example.com")
        doc = ProcessedDocument(
            id="doc-2",
            content="payload",
            metadata=metadata,
            chunks=[{"content": "part", "index": 0}],
            status=DocumentStatus.COMPLETED,
            dense_vector=[0.1, 0.2],
        )

        assert doc.status is DocumentStatus.COMPLETED
        assert len(doc.chunks) == 1
        assert doc.dense_vector == [0.1, 0.2]


class TestChunkingConfig:
    """Tests for the chunking configuration model."""

    def test_default_values(self) -> None:
        """Default chunking config matches documented defaults."""

        config = ChunkingConfig()
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.strategy is ChunkingStrategy.ENHANCED
        assert config.preserve_code_blocks is True
        assert config.detect_language is True
        assert config.token_chunk_size == 600
        assert config.token_chunk_overlap == 120
        assert config.json_max_chars == 20000
        assert config.enable_semantic_html_segmentation is True
        assert config.normalize_html_text is True

    def test_validation_rules(self) -> None:
        """Chunking configuration enforces relationship constraints."""

        ChunkingConfig(chunk_size=600, chunk_overlap=120)

        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=50, chunk_overlap=120)
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_overlap=-1)
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=0)
        with pytest.raises(ValidationError):
            ChunkingConfig(max_chunk_size=0)
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=500, max_chunk_size=100)


class TestVectorMetrics:
    """Tests for ingestion metrics models."""

    def test_defaults(self) -> None:
        """Vector metrics default to zeroed counters."""

        metrics = VectorMetrics()
        assert metrics.total_documents == 0
        assert metrics.total_chunks == 0
        assert metrics.tokens_processed == 0

    def test_custom_values(self) -> None:
        """Custom metrics values should persist."""

        metrics = VectorMetrics(
            total_documents=5,
            total_chunks=20,
            successful_embeddings=18,
            failed_embeddings=2,
            processing_time=12.5,
            tokens_processed=4200,
            avg_chunk_size=250.0,
        )

        assert metrics.total_documents == 5
        assert metrics.total_chunks == 20
        assert metrics.avg_chunk_size == 250.0


class TestScrapingStats:
    """Tests for scraping statistics."""

    def test_defaults(self) -> None:
        """Scraping stats default to zeroed metrics."""

        stats = ScrapingStats()
        assert stats.total_processed == 0
        assert stats.total_chunks == 0
        assert stats.start_time is None

    def test_custom_payload(self) -> None:
        """Custom scraping stats should persist."""

        start_time = datetime.now(tz=UTC)
        end_time = datetime.now(tz=UTC)
        stats = ScrapingStats(
            total_processed=15,
            successful_embeddings=12,
            failed_crawls=3,
            total_chunks=75,
            unique_urls=10,
            start_time=start_time,
            end_time=end_time,
            avg_processing_time=1.5,
            docs_per_minute=4.0,
            total_size_mb=12.3,
        )

        assert stats.total_processed == 15
        assert stats.failed_crawls == 3
        assert stats.start_time == start_time
        assert stats.end_time == end_time


class TestContentFilter:
    """Tests for ContentFilter settings."""

    def test_defaults(self) -> None:
        """Default filters should allow typical documents."""

        config = ContentFilter()
        assert config.min_content_length == 50
        assert config.allowed_mime_types

    def test_custom_values(self) -> None:
        """Custom content filter values should persist."""

        config = ContentFilter(
            min_content_length=10,
            max_content_length=1000,
            allowed_mime_types=["text/plain"],
            blocked_domains=["example.com"],
        )

        assert config.max_content_length == 1000
        assert config.blocked_domains == ["example.com"]


class TestDocumentBatch:
    """Tests for batched document processing model."""

    def test_defaults(self) -> None:
        """DocumentBatch initializes counters to zero."""

        batch = DocumentBatch(id="batch-1")
        assert batch.documents == []
        assert batch.total_chunks == 0

    def test_custom_values(self) -> None:
        """Custom values should persist when provided."""

        doc = ProcessedDocument(
            id="doc-1",
            content="payload",
            metadata=DocumentMetadata(url="https://example.com"),
        )
        batch = DocumentBatch(
            id="batch-2",
            documents=[doc],
            batch_size=1,
            total_chunks=3,
        )

        assert batch.batch_size == 1
        assert batch.total_chunks == 3
        assert batch.documents[0].id == "doc-1"
