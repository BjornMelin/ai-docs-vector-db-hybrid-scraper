import typing
"""Shared utilities for Qdrant vector database operations."""

from typing import Any

from qdrant_client import models


def build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Build optimized Qdrant filter from filter dictionary using indexed fields.

    This function creates Qdrant filter conditions for all indexed payload fields
    to ensure optimal query performance. Used by both search and document operations.

    Args:
        filters: Filter conditions dictionary

    Returns:
        Qdrant Filter object or None if no filters provided

    Raises:
        ValueError: If filter values have incorrect types
    """
    if not filters:
        return None

    conditions = []

    # Keyword filters for exact matching
    keyword_fields = [
        "doc_type",
        "language",
        "framework",
        "version",
        "crawl_source",
        "site_name",
        "embedding_model",
        "embedding_provider",
        "url",
    ]
    for field in keyword_fields:
        if field in filters:
            if not isinstance(filters[field], str | int | float | bool):
                raise ValueError(f"Filter value for {field} must be a simple type")
            conditions.append(
                models.FieldCondition(
                    key=field, match=models.MatchValue(value=filters[field])
                )
            )

    # Text filters for partial matching
    for field in ["title", "content_preview"]:
        if field in filters:
            if not isinstance(filters[field], str):
                raise ValueError(f"Text filter value for {field} must be a string")
            conditions.append(
                models.FieldCondition(
                    key=field, match=models.MatchText(text=filters[field])
                )
            )

    # Range filters for timestamps and metrics
    range_mappings = [
        ("created_after", "created_at", "gte"),
        ("created_before", "created_at", "lte"),
        ("updated_after", "last_updated", "gte"),
        ("updated_before", "last_updated", "lte"),
        ("crawled_after", "crawl_timestamp", "gte"),
        ("crawled_before", "crawl_timestamp", "lte"),
        ("min_word_count", "word_count", "gte"),
        ("max_word_count", "word_count", "lte"),
        ("min_char_count", "char_count", "gte"),
        ("max_char_count", "char_count", "lte"),
        ("min_quality_score", "quality_score", "gte"),
        ("max_quality_score", "quality_score", "lte"),
        ("min_score", "score", "gte"),
        ("min_total_chunks", "total_chunks", "gte"),
        ("max_total_chunks", "total_chunks", "lte"),
        ("min_links_count", "links_count", "gte"),
        ("max_links_count", "links_count", "lte"),
    ]

    for filter_key, field_key, operator in range_mappings:
        if filter_key in filters:
            range_params = {operator: filters[filter_key]}
            conditions.append(
                models.FieldCondition(key=field_key, range=models.Range(**range_params))
            )

    # Exact match filters for structural fields
    for field in ["chunk_index", "depth"]:
        if field in filters:
            conditions.append(
                models.FieldCondition(
                    key=field, match=models.MatchValue(value=filters[field])
                )
            )

    return models.Filter(must=conditions) if conditions else None
