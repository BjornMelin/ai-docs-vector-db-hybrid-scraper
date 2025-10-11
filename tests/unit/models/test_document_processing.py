"""Unit tests for document processing models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.config import ChunkingConfig, ChunkingStrategy, DocumentStatus
from src.models.document_processing import (
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


class TestCodeLanguage:
    """Test CodeLanguage enum."""

    def test_language_values(self):
        """Test enum values."""
        assert CodeLanguage.PYTHON.value == "python"
        assert CodeLanguage.JAVASCRIPT.value == "javascript"
        assert CodeLanguage.TYPESCRIPT.value == "typescript"
        assert CodeLanguage.MARKDOWN.value == "markdown"
        assert CodeLanguage.UNKNOWN.value == "unknown"

    def test_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(CodeLanguage.PYTHON, str)
        assert CodeLanguage.PYTHON == "python"


class TestChunkType:
    """Test ChunkType enum."""

    def test_chunk_type_values(self):
        """Test enum values."""
        assert ChunkType.TEXT.value == "text"
        assert ChunkType.CODE.value == "code"
        assert ChunkType.MIXED.value == "mixed"
        assert ChunkType.HEADER.value == "header"
        assert ChunkType.METADATA.value == "metadata"


class TestCodeBlock:
    """Test CodeBlock dataclass."""

    def test_required_fields(self):
        """Test creating code block with required fields."""
        block = CodeBlock(
            language="python",
            content="def hello(): pass",
            start_pos=0,
            end_pos=17,
        )
        assert block.language == "python"
        assert block.content == "def hello(): pass"
        assert block.start_pos == 0
        assert block.end_pos == 17
        assert block.fence_type == "```"  # default

    def test_custom_fence_type(self):
        """Test custom fence type."""
        block = CodeBlock(
            language="javascript",
            content="console.log('test');",
            start_pos=10,
            end_pos=30,
            fence_type="~~~",
        )
        assert block.fence_type == "~~~"


class TestChunk:
    """Test Chunk dataclass."""

    def test_required_fields(self):
        """Test creating chunk with required fields."""
        chunk = Chunk(
            content="Test content",
            start_pos=0,
            end_pos=12,
            chunk_index=0,
        )
        assert chunk.content == "Test content"
        assert chunk.start_pos == 0
        assert chunk.end_pos == 12
        assert chunk.chunk_index == 0

    def test_default_values(self):
        """Test default field values."""
        chunk = Chunk(
            content="Test",
            start_pos=0,
            end_pos=4,
            chunk_index=0,
        )
        assert chunk.total_chunks == 0
        assert chunk.char_count == 0
        assert chunk.token_estimate == 0
        assert chunk.chunk_type == "text"
        assert chunk.language is None
        assert chunk.has_code is False
        assert chunk.metadata is None

    def test_all_fields(self):
        """Test creating chunk with all fields."""
        metadata = {"function_name": "test_func", "class_name": "TestClass"}
        chunk = Chunk(
            content="def test_func(): pass",
            start_pos=100,
            end_pos=121,
            chunk_index=2,
            total_chunks=5,
            char_count=21,
            token_estimate=5,
            chunk_type="code",
            language="python",
            has_code=True,
            metadata=metadata,
        )
        assert chunk.total_chunks == 5
        assert chunk.char_count == 21
        assert chunk.token_estimate == 5
        assert chunk.chunk_type == "code"
        assert chunk.language == "python"
        assert chunk.has_code is True
        assert chunk.metadata == metadata


class TestDocumentMetadata:
    """Test DocumentMetadata model."""

    def test_required_fields(self):
        """Test required fields."""
        metadata = DocumentMetadata(url="https://example.com")
        assert metadata.url == "https://example.com"

        # Missing required field
        with pytest.raises(ValidationError):
            DocumentMetadata()  # type: ignore[call-arg]

    def test_default_values(self):
        """Test default field values."""
        metadata = DocumentMetadata(url="https://example.com")

        assert metadata.title is None
        assert metadata.doc_type is None
        assert metadata.language is None
        assert isinstance(metadata.crawled_at, datetime)
        assert metadata.last_modified is None
        assert metadata.content_hash is None
        assert metadata.author is None
        assert metadata.tags == []
        assert metadata.word_count == 0
        assert metadata.char_count == 0
        assert metadata.has_code is False
        assert metadata.estimated_reading_time == 0
        assert metadata.chunking_strategy == ChunkingStrategy.ENHANCED
        assert metadata.total_chunks == 0
        assert metadata.processing_time_ms == 0.0

    def test_custom_values(self):
        """Test custom field values."""
        crawled_at = datetime.now(tz=UTC)
        last_modified = datetime.now(tz=UTC)
        metadata = DocumentMetadata(
            title="Python Tutorial",
            url="https://example.com/tutorial.py",
            doc_type="tutorial",
            language="python",
            crawled_at=crawled_at,
            last_modified=last_modified,
            content_hash="abc123",
            author="John Doe",
            tags=["python", "tutorial", "beginner"],
            word_count=500,
            char_count=3000,
            has_code=True,
            estimated_reading_time=3,
            chunking_strategy=ChunkingStrategy.AST_AWARE,
            total_chunks=10,
            processing_time_ms=150.5,
        )
        assert metadata.title == "Python Tutorial"
        assert metadata.doc_type == "tutorial"
        assert metadata.language == "python"
        assert metadata.crawled_at == crawled_at
        assert metadata.last_modified == last_modified
        assert metadata.content_hash == "abc123"
        assert metadata.author == "John Doe"
        assert metadata.tags == ["python", "tutorial", "beginner"]
        assert metadata.word_count == 500
        assert metadata.char_count == 3000
        assert metadata.has_code is True
        assert metadata.estimated_reading_time == 3
        assert metadata.chunking_strategy == ChunkingStrategy.AST_AWARE
        assert metadata.total_chunks == 10
        assert metadata.processing_time_ms == 150.5

    def test_forbids_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DocumentMetadata(url="https://example.com", extra_field="not allowed")  # type: ignore[call-arg]


class TestProcessedDocument:
    """Test ProcessedDocument model."""

    def test_required_fields(self):
        """Test required fields."""
        metadata = DocumentMetadata(url="https://example.com")
        doc = ProcessedDocument(
            id="doc123",
            content="Document content",
            metadata=metadata,
        )
        assert doc.id == "doc123"
        assert doc.content == "Document content"
        assert doc.metadata == metadata

    def test_default_values(self):
        """Test default field values."""
        metadata = DocumentMetadata(url="https://example.com")
        doc = ProcessedDocument(
            id="doc123",
            content="Test",
            metadata=metadata,
        )
        assert doc.chunks == []
        assert doc.status == DocumentStatus.PENDING
        assert doc.error_message is None
        assert doc.dense_vector is None
        assert doc.sparse_vector is None

    def test_with_chunks_and_vectors(self):
        """Test document with chunks and vectors."""
        metadata = DocumentMetadata(url="https://example.com")
        chunks = [
            {"content": "Chunk 1", "index": 0},
            {"content": "Chunk 2", "index": 1},
        ]
        dense_vector = [0.1, 0.2, 0.3, 0.4]
        sparse_vector = {"indices": [0, 5, 10], "values": [0.5, 0.8, 0.2]}

        doc = ProcessedDocument(
            id="doc123",
            content="Full content",
            metadata=metadata,
            chunks=chunks,
            status=DocumentStatus.COMPLETED,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        assert len(doc.chunks) == 2
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.dense_vector == dense_vector
        assert doc.sparse_vector == sparse_vector

    def test_failed_document(self):
        """Test failed document with error message."""
        metadata = DocumentMetadata(url="https://example.com")
        doc = ProcessedDocument(
            id="doc123",
            content="",
            metadata=metadata,
            status=DocumentStatus.FAILED,
            error_message="Failed to crawl URL",
        )
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "Failed to crawl URL"


class TestChunkingConfig:
    """Test ChunkingConfig model."""

    def test_default_values(self):
        """Test default field values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.strategy == ChunkingStrategy.ENHANCED
        assert config.enable_ast_chunking is True
        assert config.preserve_code_blocks is True
        assert config.max_function_chunk_size == 2000
        assert config.supported_languages == [
            "python",
            "javascript",
            "typescript",
            "markdown",
        ]
        assert config.detect_language is True
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000

    def test_chunk_size_constraints(self):
        """Test chunk_size constraints."""
        # Valid sizes (must be > chunk_overlap and <= max_chunk_size)
        ChunkingConfig(chunk_size=500, chunk_overlap=100)
        ChunkingConfig(
            chunk_size=2000, max_chunk_size=5000, max_function_chunk_size=5000
        )

        # Invalid size
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=0)
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=-1)

        # chunk_size must be > chunk_overlap
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=100, chunk_overlap=200)

        # chunk_size cannot exceed max_chunk_size
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=5000, max_chunk_size=3000)

    def test_chunk_overlap_constraints(self):
        """Test chunk_overlap constraints."""
        # Valid overlaps
        ChunkingConfig(chunk_overlap=0)
        ChunkingConfig(chunk_overlap=500)

        # Invalid overlap
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_overlap=-1)

    def test_max_function_chunk_size_constraints(self):
        """Test max_function_chunk_size constraints."""
        # Valid sizes (must be >= max_chunk_size which defaults to 3000)
        ChunkingConfig(max_function_chunk_size=3000)
        ChunkingConfig(max_function_chunk_size=10000)

        # Invalid size
        with pytest.raises(ValidationError):
            ChunkingConfig(max_function_chunk_size=0)

        # max_function_chunk_size can now be smaller than max_chunk_size;
        # ensure instance builds
        ChunkingConfig(max_function_chunk_size=1000, max_chunk_size=2000)

    def test_min_max_chunk_size_constraints(self):
        """Test min_chunk_size and max_chunk_size constraints."""
        # Valid sizes (max_function_chunk_size must be >= max_chunk_size)
        ChunkingConfig(
            min_chunk_size=1, max_chunk_size=5000, max_function_chunk_size=5000
        )

        # Invalid min size
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=0)

        # Invalid max size
        with pytest.raises(ValidationError):
            ChunkingConfig(max_chunk_size=0)

        # min_chunk_size must be < max_chunk_size
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=2000, max_chunk_size=1000)

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=2000,
            chunk_overlap=400,
            strategy=ChunkingStrategy.BASIC,
            enable_ast_chunking=False,
            preserve_code_blocks=False,
            max_function_chunk_size=5000,
            supported_languages=["python", "rust"],
            detect_language=False,
            min_chunk_size=50,
            max_chunk_size=4000,
        )
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400
        assert config.strategy == ChunkingStrategy.BASIC
        assert config.enable_ast_chunking is False
        assert config.supported_languages == ["python", "rust"]
        assert config.min_chunk_size == 50
        assert config.max_chunk_size == 4000


class TestVectorMetrics:
    """Test VectorMetrics model."""

    def test_default_values(self):
        """Test default field values."""
        metrics = VectorMetrics()
        assert metrics.total_documents == 0
        assert metrics.total_chunks == 0
        assert metrics.successful_embeddings == 0
        assert metrics.failed_embeddings == 0
        assert metrics.processing_time == 0.0
        assert metrics.tokens_processed == 0
        assert metrics.avg_chunk_size == 0.0

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = VectorMetrics(
            total_documents=100,
            total_chunks=500,
            successful_embeddings=480,
            failed_embeddings=20,
            processing_time=300.5,
            tokens_processed=150000,
            avg_chunk_size=300.0,
        )
        assert metrics.total_documents == 100
        assert metrics.total_chunks == 500
        assert metrics.successful_embeddings == 480
        assert metrics.failed_embeddings == 20
        assert metrics.processing_time == 300.5
        assert metrics.tokens_processed == 150000
        assert metrics.avg_chunk_size == 300.0


class TestScrapingStats:
    """Test ScrapingStats model."""

    def test_default_values(self):
        """Test default field values."""
        stats = ScrapingStats()
        assert stats.total_processed == 0
        assert stats.successful_embeddings == 0
        assert stats.failed_crawls == 0
        assert stats.total_chunks == 0
        assert stats.unique_urls == 0
        assert stats.start_time is None
        assert stats.end_time is None
        assert stats.avg_processing_time == 0.0
        assert stats.docs_per_minute == 0.0
        assert stats.total_size_mb == 0.0

    def test_with_timing_data(self):
        """Test stats with timing data."""
        start_time = datetime.now(tz=UTC)
        end_time = datetime.now(tz=UTC)

        stats = ScrapingStats(
            total_processed=50,
            successful_embeddings=45,
            failed_crawls=5,
            total_chunks=250,
            unique_urls=48,
            start_time=start_time,
            end_time=end_time,
            avg_processing_time=2.5,
            docs_per_minute=10.0,
            total_size_mb=125.5,
        )
        assert stats.total_processed == 50
        assert stats.successful_embeddings == 45
        assert stats.failed_crawls == 5
        assert stats.total_chunks == 250
        assert stats.unique_urls == 48
        assert stats.start_time == start_time
        assert stats.end_time == end_time
        assert stats.avg_processing_time == 2.5
        assert stats.docs_per_minute == 10.0
        assert stats.total_size_mb == 125.5


class TestContentFilter:
    """Test ContentFilter model."""

    def test_default_values(self):
        """Test default field values."""
        filter_config = ContentFilter()
        assert filter_config.min_content_length == 50
        assert filter_config.max_content_length == 1_000_000
        assert filter_config.allowed_mime_types == [
            "text/html",
            "text/plain",
            "text/markdown",
        ]
        assert filter_config.blocked_domains == []
        assert filter_config.content_patterns_to_exclude == []
        assert filter_config.min_word_count == 10
        assert filter_config.max_duplicate_ratio == 0.8

    def test_max_duplicate_ratio_constraints(self):
        """Test max_duplicate_ratio constraints."""
        # Valid ratios
        ContentFilter(max_duplicate_ratio=0.0)
        ContentFilter(max_duplicate_ratio=0.5)
        ContentFilter(max_duplicate_ratio=1.0)

        # Invalid ratios
        with pytest.raises(ValidationError):
            ContentFilter(max_duplicate_ratio=-0.1)
        with pytest.raises(ValidationError):
            ContentFilter(max_duplicate_ratio=1.1)

    def test_custom_filter(self):
        """Test custom filter configuration."""
        filter_config = ContentFilter(
            min_content_length=100,
            max_content_length=500_000,
            allowed_mime_types=["text/html", "application/json"],
            blocked_domains=["spam.com", "ads.com"],
            content_patterns_to_exclude=[r"<script.*?</script>", r"<style.*?</style>"],
            min_word_count=20,
            max_duplicate_ratio=0.6,
        )
        assert filter_config.min_content_length == 100
        assert filter_config.max_content_length == 500_000
        assert filter_config.allowed_mime_types == ["text/html", "application/json"]
        assert filter_config.blocked_domains == ["spam.com", "ads.com"]
        assert len(filter_config.content_patterns_to_exclude) == 2
        assert filter_config.min_word_count == 20
        assert filter_config.max_duplicate_ratio == 0.6


class TestDocumentBatch:
    """Test DocumentBatch model."""

    def test_required_fields(self):
        """Test required fields."""
        batch = DocumentBatch(id="batch123")
        assert batch.id == "batch123"

    def test_default_values(self):
        """Test default field values."""
        batch = DocumentBatch(id="batch123")

        assert batch.documents == []
        assert batch.batch_size == 0
        assert isinstance(batch.created_at, datetime)
        assert batch.status == "pending"
        assert batch.successful_count == 0
        assert batch.failed_count == 0
        assert batch.total_chunks == 0
        assert batch.processing_time_ms == 0.0

    def test_with_documents(self):
        """Test batch with documents."""
        metadata1 = DocumentMetadata(url="https://example1.com")
        metadata2 = DocumentMetadata(url="https://example2.com")

        doc1 = ProcessedDocument(
            id="doc1",
            content="Content 1",
            metadata=metadata1,
        )
        doc2 = ProcessedDocument(
            id="doc2",
            content="Content 2",
            metadata=metadata2,
        )

        batch = DocumentBatch(
            id="batch123",
            documents=[doc1, doc2],
            batch_size=2,
            status="completed",
            successful_count=2,
            failed_count=0,
            total_chunks=10,
            processing_time_ms=500.0,
        )
        assert len(batch.documents) == 2
        assert batch.batch_size == 2
        assert batch.status == "completed"
        assert batch.successful_count == 2
        assert batch.failed_count == 0
        assert batch.total_chunks == 10
        assert batch.processing_time_ms == 500.0

    def test_partial_success_batch(self):
        """Test batch with partial success."""
        batch = DocumentBatch(
            id="batch123",
            batch_size=10,
            status="partial_success",
            successful_count=7,
            failed_count=3,
            total_chunks=35,
            processing_time_ms=1500.0,
        )
        assert batch.successful_count == 7
        assert batch.failed_count == 3
        assert batch.successful_count + batch.failed_count == batch.batch_size
