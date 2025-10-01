"""Tests for payload normalization utilities."""

from __future__ import annotations

from src.services.vector_db.payload_schema import (
    PayloadValidationError,
    compute_content_hash,
    ensure_canonical_payload,
)


def test_ensure_canonical_payload_creates_expected_fields() -> None:
    """Canonical payload should populate required fields with defaults."""

    metadata = {"topic": "testing"}
    canonical = ensure_canonical_payload(metadata, content="alpha", id_hint="doc-1")

    assert canonical.payload["doc_id"] == "doc-1"
    assert canonical.payload["chunk_id"] == 0
    assert canonical.payload["tenant"] == "default"
    assert canonical.payload["source"] == "unknown"
    assert canonical.payload["content_hash"] == compute_content_hash("alpha")
    assert len(canonical.point_id) == 32


def test_ensure_canonical_payload_respects_existing_fields() -> None:
    """Existing canonical metadata should be preserved when valid."""

    metadata = {
        "doc_id": "doc-2",
        "chunk_id": 5,
        "tenant": "tenant-a",
        "source": "https://example.com",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    canonical = ensure_canonical_payload(metadata, content="beta", id_hint="ignored")

    assert canonical.payload["doc_id"] == "doc-2"
    assert canonical.payload["chunk_id"] == 5
    assert canonical.payload["tenant"] == "tenant-a"
    assert canonical.payload["source"] == "https://example.com"
    assert canonical.payload["created_at"] == "2024-01-01T00:00:00+00:00"


def test_ensure_canonical_payload_raises_for_invalid_strings() -> None:
    """Invalid string inputs should surface a validation error."""

    metadata = {"doc_id": None, "chunk_id": 0, "tenant": "", "source": ""}
    try:
        ensure_canonical_payload(metadata, content="gamma", id_hint="")
    except PayloadValidationError as exc:
        assert "doc_id" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Validation error not raised")
