"""Tests for Qdrant utility functions."""

import pytest
from qdrant_client import models

from src.services.vector_db.utils import build_filter


def _flatten_conditions(
    filter_obj: models.Filter | models.Condition | None,
) -> list[models.FieldCondition]:
    stack: list[models.Filter | models.Condition] = []
    if filter_obj is None:
        return []
    stack.append(filter_obj)
    collected: list[models.FieldCondition] = []

    while stack:
        item = stack.pop()
        if isinstance(item, models.FieldCondition):
            collected.append(item)
            continue
        if isinstance(item, models.Filter):
            for section in (item.must, item.should, item.must_not):
                if section:
                    if isinstance(section, list):
                        stack.extend(section)
                    else:
                        stack.append(section)

    return collected


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
        conditions = _flatten_conditions(result)
        assert len(conditions) == 3

        for condition in conditions:
            assert isinstance(condition.match, models.MatchValue)

    def test_build_filter_text_fields(self):
        """Test build_filter with text fields."""
        filters = {
            "title": "Getting Started",
            "content_preview": "This tutorial shows how to",
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 2

        for condition in conditions:
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
        conditions = _flatten_conditions(result)
        assert len(conditions) == 4

        for condition in conditions:
            assert isinstance(condition.range, models.Range)

    def test_build_filter_exact_match_fields(self):
        """Test build_filter with exact match fields."""
        filters = {
            "chunk_index": 0,
            "depth": 2,
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 2

        for condition in conditions:
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
        conditions = _flatten_conditions(result)
        assert len(conditions) == 4

    def test_build_filter_keyword_list(self):
        """Test build_filter with list value converted to MatchAny."""
        filters = {"doc_type": ["api", "guide"]}

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 1
        assert isinstance(conditions[0].match, models.MatchAny)

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
            "crawled_after": 1640995200,
            "crawled_before": 1672531199,
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
        conditions = _flatten_conditions(result)
        assert len(conditions) == len(filters)

    def test_build_filter_match_any(self):
        """Test build_filter with MatchAny specification."""
        filters = {"content_type": {"any": ["guide", "tutorial"]}}

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 1
        assert isinstance(conditions[0].match, models.MatchAny)

    def test_build_filter_temporal_window(self):
        """Test temporal filter shortcut with relative window."""
        filters = {"temporal": {"field": "created_at", "window": "7d"}}

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert conditions
        condition = conditions[0]
        assert isinstance(condition.range, models.Range)
        assert condition.range.gte is not None

    def test_build_filter_boolean_structure(self):
        """Test boolean structure using must and must_not clauses."""
        filters = {
            "must": [{"doc_type": "api"}],
            "must_not": [{"language": {"any": ["java", "php"]}}],
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        keys = {condition.key for condition in conditions}
        assert {"doc_type", "language"}.issubset(keys)

    def test_build_filter_field_conditions(self):
        """Test custom field condition injection."""
        filters = {
            "field_conditions": [
                {"title": {"text": "introduction"}},
                {"range": {"score": {"gte": 0.6}}},
            ]
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 2

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
        conditions = _flatten_conditions(result)
        assert len(conditions) == len(filters)

    def test_build_filter_bool_keyword_value(self):
        """Test build_filter with boolean keyword values."""
        filters = {
            "doc_type": True,  # boolean should be allowed
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 1

    def test_build_filter_numeric_keyword_value(self):
        """Test build_filter with numeric keyword values."""
        filters = {
            "language": 123,  # int should be allowed
        }

        result = build_filter(filters)

        assert isinstance(result, models.Filter)
        conditions = _flatten_conditions(result)
        assert len(conditions) == 1
