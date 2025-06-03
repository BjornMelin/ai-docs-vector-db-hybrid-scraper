"""Unit tests for MCP response models."""

from datetime import datetime

import pytest
from pydantic import ValidationError
from src.mcp.models.responses import CrawlResult
from src.mcp.models.responses import SearchResult


class TestSearchResult:
    """Test SearchResult model."""

    def test_minimal_valid_result(self):
        """Test minimal valid search result."""
        result = SearchResult(
            id="doc_123",
            content="This is the document content",
            score=0.95,
        )
        assert result.id == "doc_123"
        assert result.content == "This is the document content"
        assert result.score == 0.95
        assert result.url is None
        assert result.title is None
        assert result.metadata is None

    def test_all_fields(self):
        """Test search result with all fields."""
        metadata = {
            "author": "John Doe",
            "created_at": "2024-01-01",
            "tags": ["python", "tutorial"],
        }
        result = SearchResult(
            id="doc_456",
            content="Advanced Python tutorial content",
            score=0.87,
            url="https://example.com/tutorials/python",
            title="Advanced Python Tutorial",
            metadata=metadata,
        )
        assert result.id == "doc_456"
        assert result.url == "https://example.com/tutorials/python"
        assert result.title == "Advanced Python Tutorial"
        assert result.metadata == metadata
        assert result.metadata["tags"] == ["python", "tutorial"]

    def test_missing_required_fields(self):
        """Test that required fields must be present."""
        # Missing id
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(content="test", score=0.5)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("id",) for error in errors)

        # Missing content
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(id="123", score=0.5)
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("content",) for error in errors)

        # Missing score
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(id="123", content="test")
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("score",) for error in errors)

    def test_score_validation(self):
        """Test score field accepts various numeric values."""
        # Valid scores
        SearchResult(id="1", content="test", score=0.0)
        SearchResult(id="2", content="test", score=1.0)
        SearchResult(id="3", content="test", score=0.5)
        SearchResult(id="4", content="test", score=-0.1)  # Negative scores allowed
        SearchResult(id="5", content="test", score=1.5)  # Scores > 1 allowed

    def test_empty_content(self):
        """Test that empty content is allowed."""
        result = SearchResult(id="empty", content="", score=0.1)
        assert result.content == ""

    def test_metadata_flexibility(self):
        """Test that metadata can contain any structure."""
        # Simple metadata
        result1 = SearchResult(
            id="1",
            content="test",
            score=0.5,
            metadata={"key": "value"},
        )
        assert result1.metadata["key"] == "value"

        # Complex nested metadata
        complex_metadata = {
            "nested": {
                "level": 2,
                "items": [1, 2, 3],
            },
            "array": ["a", "b", "c"],
            "number": 42,
            "boolean": True,
            "null": None,
        }
        result2 = SearchResult(
            id="2",
            content="test",
            score=0.5,
            metadata=complex_metadata,
        )
        assert result2.metadata["nested"]["level"] == 2
        assert result2.metadata["array"] == ["a", "b", "c"]
        assert result2.metadata["boolean"] is True
        assert result2.metadata["null"] is None


class TestCrawlResult:
    """Test CrawlResult model."""

    def test_minimal_valid_result(self):
        """Test minimal valid crawl result."""
        result = CrawlResult(url="https://example.com")
        assert result.url == "https://example.com"
        assert result.title == ""
        assert result.content == ""
        assert result.word_count == 0
        assert result.success is False
        assert result.site_name == ""
        assert result.depth == 0
        assert isinstance(result.crawl_timestamp, str)
        assert result.links == []
        assert result.metadata == {}
        assert result.error is None

    def test_all_fields(self):
        """Test crawl result with all fields."""
        result = CrawlResult(
            url="https://docs.example.com/api/reference",
            title="API Reference Documentation",
            content="This is the API reference content with detailed examples...",
            word_count=150,
            success=True,
            site_name="Example Docs",
            depth=2,
            crawl_timestamp="2024-01-15T10:30:00",
            links=[
                "https://docs.example.com/api/getting-started",
                "https://docs.example.com/api/authentication",
            ],
            metadata={
                "language": "en",
                "last_modified": "2024-01-10",
                "author": "API Team",
            },
            error=None,
        )
        assert result.url == "https://docs.example.com/api/reference"
        assert result.title == "API Reference Documentation"
        assert result.word_count == 150
        assert result.success is True
        assert result.site_name == "Example Docs"
        assert result.depth == 2
        assert len(result.links) == 2
        assert result.metadata["language"] == "en"
        assert result.error is None

    def test_failed_crawl(self):
        """Test crawl result for a failed crawl."""
        result = CrawlResult(
            url="https://example.com/404",
            success=False,
            error="404 Not Found",
        )
        assert result.url == "https://example.com/404"
        assert result.success is False
        assert result.error == "404 Not Found"
        assert result.content == ""
        assert result.word_count == 0

    def test_default_crawl_timestamp(self):
        """Test that crawl_timestamp is automatically set to current time."""
        before = datetime.now().isoformat()
        result = CrawlResult(url="https://example.com")
        after = datetime.now().isoformat()

        # crawl_timestamp should be between before and after
        assert before <= result.crawl_timestamp <= after

    def test_custom_crawl_timestamp(self):
        """Test setting custom crawl_timestamp value."""
        custom_time = "2024-01-01T00:00:00"
        result = CrawlResult(
            url="https://example.com",
            crawl_timestamp=custom_time,
        )
        assert result.crawl_timestamp == custom_time

    def test_empty_lists_and_dicts(self):
        """Test that empty lists and dicts are properly initialized."""
        result = CrawlResult(url="https://example.com")
        assert result.links == []
        assert isinstance(result.links, list)
        assert result.metadata == {}
        assert isinstance(result.metadata, dict)

    def test_complex_metadata(self):
        """Test complex metadata structures."""
        metadata = {
            "headers": {
                "content-type": "text/html",
                "last-modified": "2024-01-01",
            },
            "extraction": {
                "method": "beautifulsoup",
                "duration_ms": 250,
                "selectors_used": [".content", "#main"],
            },
            "metrics": {
                "images": 5,
                "links": 20,
                "scripts": 3,
            },
        }
        result = CrawlResult(
            url="https://example.com",
            metadata=metadata,
        )
        assert result.metadata["headers"]["content-type"] == "text/html"
        assert result.metadata["extraction"]["duration_ms"] == 250
        assert len(result.metadata["extraction"]["selectors_used"]) == 2

    def test_missing_required_field(self):
        """Test that URL is required."""
        with pytest.raises(ValidationError) as exc_info:
            CrawlResult()
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("url",)
        assert errors[0]["type"] == "missing"

    def test_field_types(self):
        """Test field type validation."""
        # Valid types
        CrawlResult(
            url="https://example.com",
            word_count=0,
            depth=0,
            success=True,
        )

        # Test that boolean fields accept bool values
        result = CrawlResult(url="test", success=True)
        assert result.success is True
        result = CrawlResult(url="test", success=False)
        assert result.success is False

    def test_serialization(self):
        """Test model serialization."""
        result = CrawlResult(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            word_count=2,
            success=True,
            links=["https://example.com/other"],
            metadata={"key": "value"},
        )

        # Test dict serialization
        data = result.model_dump()
        assert data["url"] == "https://example.com"
        assert data["title"] == "Test Page"
        assert data["word_count"] == 2
        assert data["success"] is True
        assert data["links"] == ["https://example.com/other"]
        assert data["metadata"] == {"key": "value"}

        # Test JSON serialization
        json_str = result.model_dump_json()
        assert "https://example.com" in json_str
        assert "Test Page" in json_str
