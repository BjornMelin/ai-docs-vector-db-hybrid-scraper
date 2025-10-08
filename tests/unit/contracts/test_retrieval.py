"""Unit tests for canonical retrieval contracts."""

from __future__ import annotations

from src.contracts.retrieval import SearchRecord
from src.services.vector_db.types import VectorMatch


def test_from_vector_match_normalizes_payload() -> None:
    """Ensure ``SearchRecord.from_vector_match`` maps vector matches correctly."""

    match = VectorMatch(
        id="123",
        score=0.42,
        raw_score=0.84,
        normalized_score=None,
        payload={
            "content": "hello world",
            "title": "greeting",
            "url": "https://example.test",
            "doc_id": "doc-7",
            "content_type": "text/plain",
            "quality_relevance": 0.9,
            "_grouping": {"rank": 2, "applied": True},
        },
        collection=None,
    )

    record = SearchRecord.from_vector_match(match, collection_name="primary")

    assert record.id == "123"
    assert record.collection == "primary"
    assert record.content == "hello world"
    assert record.title == "greeting"
    assert record.url == "https://example.test"
    assert record.group_id == "doc-7"
    assert record.group_rank == 2
    assert record.grouping_applied is True
    assert record.score == 0.84
    assert record.raw_score == 0.84
    assert (
        record.metadata is not None and record.metadata["content_type"] == "text/plain"
    )


def test_from_vector_match_prefers_normalized_score() -> None:
    """Normalized score should override raw score when present."""

    match = VectorMatch(
        id="abc",
        score=0.2,
        raw_score=0.2,
        normalized_score=0.75,
        payload={},
        collection="docs",
    )

    record = SearchRecord.from_vector_match(match, collection_name="fallback")

    assert record.collection == "docs"
    assert record.score == 0.75
    assert record.normalized_score == 0.75
    assert record.raw_score == 0.2
