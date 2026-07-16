"""Tests for payload normalization utilities."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.services.vector_db.payload_schema import (
    PayloadValidationError,
    compute_content_hash,
    normalize_document,
    stable_point_id,
)


def test_normalize_document_creates_expected_fields() -> None:
    """Canonical payload should populate required fields with defaults."""
    canonical = normalize_document(
        Document(page_content="alpha", metadata={"topic": "testing"}),
        id_hint="doc-1",
    )

    assert canonical.metadata["doc_id"] == "doc-1"
    assert canonical.metadata["chunk_index"] == 0
    assert canonical.metadata["tenant"] == "default"
    assert canonical.metadata["source"] == "unknown"
    assert canonical.metadata["content_hash"] == compute_content_hash("alpha")
    assert canonical.id == stable_point_id(
        tenant="default", doc_id="doc-1", chunk_index=0
    )


def test_normalize_document_respects_existing_fields() -> None:
    """Existing canonical metadata should be preserved when valid."""
    metadata = {
        "doc_id": "doc-2",
        "chunk_index": 5,
        "tenant": "tenant-a",
        "source": "https://example.com",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    canonical = normalize_document(
        Document(page_content="beta", metadata=metadata),
        id_hint="ignored",
    )

    assert canonical.metadata["doc_id"] == "doc-2"
    assert canonical.metadata["chunk_index"] == 5
    assert canonical.metadata["tenant"] == "tenant-a"
    assert canonical.metadata["source"] == "https://example.com"
    assert canonical.metadata["created_at"] == "2024-01-01T00:00:00+00:00"


def test_point_identity_does_not_change_with_content() -> None:
    """Content updates should replace a stable point instead of duplicating it."""
    metadata = {"doc_id": "doc-2", "chunk_index": 5, "tenant": "tenant-a"}
    first = normalize_document(
        Document(page_content="before", metadata=metadata), id_hint="ignored"
    )
    second = normalize_document(
        Document(page_content="after", metadata=metadata), id_hint="ignored"
    )

    assert first.id == second.id
    assert first.metadata["content_hash"] != second.metadata["content_hash"]


def test_normalize_document_raises_for_invalid_strings() -> None:
    """Invalid string inputs should surface a validation error."""
    metadata = {"doc_id": None, "chunk_index": 0, "tenant": "", "source": ""}
    try:
        normalize_document(
            Document(page_content="gamma", metadata=metadata), id_hint=""
        )
    except PayloadValidationError as exc:
        assert "doc_id" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Validation error not raised")


@pytest.mark.parametrize(
    "chunk_index",
    [True, -1, 1.1, 1.9, "1.0", "-1", "01"],
)
def test_normalize_document_rejects_ambiguous_chunk_indexes(
    chunk_index: object,
) -> None:
    """Chunk positions must not collapse into the same stable point ID."""
    with pytest.raises(PayloadValidationError, match="non-negative integer"):
        normalize_document(
            Document(
                page_content="gamma",
                metadata={"doc_id": "doc-1", "chunk_index": chunk_index},
            ),
            id_hint="doc-1",
        )


def test_normalize_document_accepts_canonical_integer_string() -> None:
    """Canonical integer strings should normalize without identity drift."""
    canonical = normalize_document(
        Document(
            page_content="gamma",
            metadata={"doc_id": "doc-1", "chunk_index": "12"},
        ),
        id_hint="doc-1",
    )

    assert canonical.metadata["chunk_index"] == 12
