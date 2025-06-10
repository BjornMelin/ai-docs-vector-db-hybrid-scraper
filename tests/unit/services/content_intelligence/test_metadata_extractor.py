"""Simple tests for metadata extractor functionality."""

from datetime import datetime

import pytest
from src.services.content_intelligence.metadata_extractor import MetadataExtractor
from src.services.content_intelligence.models import ContentMetadata


class TestMetadataExtractor:
    """Simple tests for metadata extraction."""

    @pytest.fixture
    def metadata_extractor(self):
        """Create metadata extractor instance."""
        return MetadataExtractor()

    def test_metadata_extractor_initialization(self, metadata_extractor):
        """Test that MetadataExtractor can be initialized."""
        assert metadata_extractor is not None
        assert isinstance(metadata_extractor, MetadataExtractor)

    @pytest.mark.asyncio
    async def test_extract_basic_metadata_simple(self, metadata_extractor):
        """Test basic metadata extraction with simple content."""
        content = "This is a simple test document with some content."
        url = "https://example.com/test"

        result = await metadata_extractor.extract_metadata(content, url)

        assert isinstance(result, ContentMetadata)
        assert result.crawled_at is not None
        assert isinstance(result.crawled_at, datetime)
        assert result.word_count > 0
        assert result.char_count > 0

    @pytest.mark.asyncio
    async def test_extract_metadata_with_empty_content(self, metadata_extractor):
        """Test metadata extraction with empty content."""
        content = ""
        url = "https://example.com/empty"

        result = await metadata_extractor.extract_metadata(content, url)

        assert isinstance(result, ContentMetadata)
        assert result.word_count == 0
        assert result.char_count == 0
        assert result.crawled_at is not None

    def test_content_metadata_model_creation(self):
        """Test ContentMetadata model can be created with basic fields."""
        metadata = ContentMetadata(
            title="Test Document",
            description="A test document for validation",
            word_count=50,
            char_count=300,
            paragraph_count=2,
        )

        assert metadata.title == "Test Document"
        assert metadata.description == "A test document for validation"
        assert metadata.word_count == 50
        assert metadata.char_count == 300
        assert metadata.paragraph_count == 2

    def test_content_metadata_model_defaults(self):
        """Test ContentMetadata model with default values."""
        metadata = ContentMetadata()

        assert metadata.title is None
        assert metadata.description is None
        assert metadata.author is None
        assert metadata.language is None
        assert metadata.published_date is None
        assert metadata.word_count == 0
        assert metadata.char_count == 0
