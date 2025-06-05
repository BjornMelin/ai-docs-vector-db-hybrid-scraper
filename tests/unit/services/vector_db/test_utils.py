"""Tests for Qdrant utility functions."""

import pytest
from qdrant_client import models
from src.services.vector_db.utils import build_filter


class TestQdrantUtils:
    """Test cases for Qdrant utility functions."""

    def test_build_filter_none(self):
        """Test build_filter with None input."""
        result = build_filter(None)
        assert result is None

    def test_build_filter_empty_dict(self):
        """Test build_filter with empty dictionary."""
        result = build_filter({})
        assert result is None

    def test_build_filter_keyword_fields(self):
        """Test build_filter with keyword fields."""
        filters = {
            "doc_type": "api",
            "language": "python",
            "framework": "fastapi",
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 3

        # Check that all conditions are field conditions with match values
        for condition in result.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchValue)

    def test_build_filter_text_fields(self):
        """Test build_filter with text fields."""
        filters = {
            "title": "Getting Started",
            "content_preview": "This tutorial shows how to",
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 2

        # Check that all conditions are field conditions with text matches
        for condition in result.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchText)

    def test_build_filter_range_fields(self):
        """Test build_filter with range fields."""
        filters = {
            "created_after": 1640995200,  # 2022-01-01
            "created_before": 1672531199,  # 2022-12-31
            "min_word_count": 100,
            "max_word_count": 1000,
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 4

        # Check that all conditions are field conditions with ranges
        for condition in result.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.range, models.Range)

    def test_build_filter_exact_match_fields(self):
        """Test build_filter with exact match fields."""
        filters = {
            "chunk_index": 0,
            "depth": 2,
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 2

        # Check that all conditions are field conditions with match values
        for condition in result.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchValue)

    def test_build_filter_mixed_fields(self):
        """Test build_filter with mixed field types."""
        filters = {
            "doc_type": "guide",  # keyword
            "title": "Installation",  # text
            "min_word_count": 50,  # range
            "chunk_index": 1,  # exact match
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 4

    def test_build_filter_invalid_keyword_value(self):
        """Test build_filter with invalid keyword field value."""
        filters = {
            "doc_type": ["api", "guide"]  # Should be string, not list
        }

        with pytest.raises(
            ValueError, match="Filter value for doc_type must be a simple type"
        ):
            build_filter(filters)

    def test_build_filter_invalid_text_value(self):
        """Test build_filter with invalid text field value."""
        filters = {
            "title": 123  # Should be string, not int
        }

        with pytest.raises(
            ValueError, match="Text filter value for title must be a string"
        ):
            build_filter(filters)

    def test_build_filter_all_range_mappings(self):
        """Test build_filter with all supported range mappings."""
        filters = {
            "created_after": 1640995200,
            "created_before": 1672531199,
            "updated_after": 1640995200,
            "updated_before": 1672531199,
            "scraped_after": 1640995200,
            "scraped_before": 1672531199,
            "min_word_count": 100,
            "max_word_count": 1000,
            "min_char_count": 500,
            "max_char_count": 5000,
            "min_quality_score": 0.7,
            "max_quality_score": 1.0,
            "min_score": 0.5,
            "min_total_chunks": 1,
            "max_total_chunks": 10,
            "min_links_count": 0,
            "max_links_count": 50,
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == len(filters)

    def test_build_filter_all_keyword_fields(self):
        """Test build_filter with all supported keyword fields."""
        filters = {
            "doc_type": "api",
            "language": "python",
            "framework": "fastapi",
            "version": "3.0",
            "crawl_source": "crawl4ai",
            "site_name": "FastAPI Docs",
            "embedding_model": "text-embedding-ada-002",
            "embedding_provider": "openai",
            "url": "https://fastapi.tiangolo.com/",
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == len(filters)

    def test_build_filter_bool_keyword_value(self):
        """Test build_filter with boolean keyword values."""
        filters = {
            "doc_type": True,  # boolean should be allowed
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 1

    def test_build_filter_numeric_keyword_value(self):
        """Test build_filter with numeric keyword values."""
        filters = {
            "language": 123,  # int should be allowed
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 1
