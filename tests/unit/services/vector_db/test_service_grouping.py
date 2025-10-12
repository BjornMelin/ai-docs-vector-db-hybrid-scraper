"""Grouping behavior tests for :mod:`src.services.vector_db.service`."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.contracts.retrieval import SearchRecord
from src.services.vector_db.service import VectorStoreService


class _ConfigStub:
    """Minimal configuration stub required by :class:`VectorStoreService`."""

    def __init__(self) -> None:
        self.fastembed = SimpleNamespace(
            dense_model="stub-model",
            sparse_model="qdrant/bm25",
            retrieval_mode="hybrid",
        )
        self.qdrant = SimpleNamespace(enable_grouping=True)
        self.query_processing = SimpleNamespace(
            enable_score_normalization=False,
            score_normalization_strategy="min_max",
            score_normalization_epsilon=1e-6,
        )


class _EmbeddingStub:
    """Embedding provider stand-in that satisfies the service contract."""

    embedding_dimension = 3


@pytest.fixture()
def vector_service() -> VectorStoreService:
    """Provide a :class:`VectorStoreService` instance for grouping tests."""

    return VectorStoreService(
        config=_ConfigStub(),
        async_qdrant_client=AsyncMock(),
        embeddings_provider=_EmbeddingStub(),
    )


def _make_record(metadata: dict[str, Any] | None = None) -> SearchRecord:
    """Create a :class:`SearchRecord` for grouping tests."""

    return SearchRecord(
        id="rec-1",
        content="sample",
        score=1.0,
        metadata=metadata,
    )


def test_annotate_grouping_metadata_updates_record_fields(
    vector_service: VectorStoreService,
) -> None:
    """Server-side grouping metadata should populate record attributes."""

    record = _make_record(
        {
            "doc_id": "doc-99",
            "_grouping": {"applied": True, "rank": 7},
        }
    )

    result = vector_service._annotate_grouping_metadata(
        [record],
        group_by="doc_id",
        grouping_applied=True,
    )

    enriched = result[0]
    assert enriched.group_id == "doc-99"
    assert enriched.group_rank == 1
    assert enriched.grouping_applied is True
    assert enriched.metadata is not None
    grouping = enriched.metadata.get("_grouping", {})
    assert grouping["group_id"] == "doc-99"
    assert grouping["rank"] == 1
    assert grouping["applied"] is True


def test_group_client_side_sets_group_attributes(
    vector_service: VectorStoreService,
) -> None:
    """Client-side grouping should update metadata and record attributes."""

    records = [
        _make_record({"doc_id": "doc-1"}),
        SearchRecord(
            id="rec-2",
            content="another",
            score=0.5,
            metadata={"doc_id": "doc-1"},
        ),
    ]

    grouped = vector_service._group_client_side(
        records,
        group_by="doc_id",
        group_size=2,
        limit=10,
    )

    assert len(grouped) == 2
    for index, record in enumerate(grouped, start=1):
        assert record.group_id == "doc-1"
        assert record.group_rank == index
        assert record.grouping_applied is False
        assert record.metadata is not None
        grouping = record.metadata.get("_grouping", {})
        assert grouping["group_id"] == "doc-1"
        assert grouping["rank"] == index
        assert grouping["applied"] is False


def test_annotate_grouping_metadata_noop_when_group_by_absent(
    vector_service: VectorStoreService,
) -> None:
    """Annotation should not modify records when grouping is disabled."""

    record = _make_record()

    result = vector_service._annotate_grouping_metadata(
        [record],
        group_by=None,
        grouping_applied=False,
    )

    untouched = result[0]
    assert untouched.group_id is None
    assert untouched.group_rank is None
    assert untouched.grouping_applied is None
    assert untouched.metadata is None
