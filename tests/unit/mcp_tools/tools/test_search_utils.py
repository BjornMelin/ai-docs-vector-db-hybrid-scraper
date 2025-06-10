"""Simple tests for search utils functionality."""

from src.config.enums import SearchAccuracy
from src.mcp_tools.models.responses import SearchResult


class TestSearchUtils:
    """Simple tests for search utilities."""

    def test_search_result_model_creation(self):
        """Test SearchResult model can be created with valid data."""
        result = SearchResult(
            id="test-doc-1",
            content="This is test content for the search result.",
            score=0.95,
            url="https://example.com/test-doc",
            title="Test Document",
        )

        assert result.id == "test-doc-1"
        assert result.content == "This is test content for the search result."
        assert result.score == 0.95
        assert result.url == "https://example.com/test-doc"
        assert result.title == "Test Document"

    def test_search_accuracy_enum_values(self):
        """Test SearchAccuracy enum has expected values."""
        assert SearchAccuracy.FAST == "fast"
        assert SearchAccuracy.BALANCED == "balanced"
        assert SearchAccuracy.ACCURATE == "accurate"

        # Test all enum values are available
        accuracy_values = [e.value for e in SearchAccuracy]
        assert "fast" in accuracy_values
        assert "balanced" in accuracy_values
        assert "accurate" in accuracy_values

    def test_search_result_with_metadata(self):
        """Test SearchResult with optional metadata."""
        metadata = {
            "document_type": "article",
            "author": "Test Author",
            "tags": ["machine-learning", "python"],
        }

        result = SearchResult(
            id="test-doc-2",
            content="Content with metadata",
            score=0.87,
            metadata=metadata,
        )

        assert result.metadata == metadata
        assert result.metadata["document_type"] == "article"

    def test_search_result_minimal_creation(self):
        """Test SearchResult with minimal required fields."""
        result = SearchResult(id="minimal-doc", content="Minimal content", score=0.5)

        assert result.id == "minimal-doc"
        assert result.content == "Minimal content"
        assert result.score == 0.5
        assert result.url is None
        assert result.title is None
        assert result.metadata is None
