"""Canonical LangChain document metadata for Qdrant persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import blake2b
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document


__all__ = [
    "PayloadValidationError",
    "compute_content_hash",
    "normalize_document",
    "stable_point_id",
]


class PayloadValidationError(ValueError):
    """Raised when payload metadata cannot be coerced into canonical form."""


_OPTIONAL_TIMESTAMP_FIELDS = ("created_at", "updated_at")
_HASH_DIGEST_SIZE = 16


def compute_content_hash(content: str) -> str:
    """Return a deterministic blake2b hash for the supplied content."""
    normalized = content.encode("utf-8")
    return blake2b(normalized, digest_size=_HASH_DIGEST_SIZE).hexdigest()


def _coerce_string(value: Any, *, field: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    msg = f"Field '{field}' must be a non-empty string"
    raise PayloadValidationError(msg)


def _coerce_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        msg = f"Field '{field}' must be a non-negative integer"
        raise PayloadValidationError(msg)
    if isinstance(value, int):
        result = value
    elif (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and value == str(int(value))
    ):
        result = int(value)
    else:
        msg = f"Field '{field}' must be a non-negative integer"
        raise PayloadValidationError(msg)
    if result < 0:
        msg = f"Field '{field}' must be a non-negative integer"
        raise PayloadValidationError(msg)
    return result


def stable_point_id(*, tenant: str, doc_id: str, chunk_index: int) -> str:
    """Return the stable, Qdrant-safe identifier for one document chunk."""
    point_key = json.dumps(
        [tenant, doc_id, chunk_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, point_key))


def normalize_document(document: Document, *, id_hint: str) -> Document:
    """Return a document that follows the sole persisted metadata contract.

    Args:
        document: LangChain document supplied by an ingestion boundary.
        id_hint: Stable source identifier when ``doc_id`` is absent.

    Returns:
        A LangChain document with a deterministic Qdrant point ID and metadata
        that is stored under QdrantVectorStore's native ``metadata`` key.
    """
    metadata: dict[str, Any] = dict(document.metadata or {})
    for dependency_field in (
        "_collection_name",
        "_id",
        "content",
        "id",
        "page_content",
        "chunk_id",
        "chunk_hash",
        "content_hash_previous",
    ):
        metadata.pop(dependency_field, None)

    doc_id = _coerce_string(metadata.get("doc_id") or id_hint, field="doc_id")
    metadata["doc_id"] = doc_id

    chunk_index = _coerce_int(
        metadata.get("chunk_index", 0),
        field="chunk_index",
    )
    metadata["chunk_index"] = chunk_index

    metadata["tenant"] = _coerce_string(
        metadata.get("tenant") or "default",
        field="tenant",
    )
    tenant = metadata["tenant"]

    metadata["source"] = _coerce_string(
        metadata.get("source") or metadata.get("url") or "unknown",
        field="source",
    )

    created_at = metadata.get("created_at")
    if not created_at:
        created_at = datetime.now(UTC).isoformat()
    metadata["created_at"] = created_at

    if metadata.get("updated_at") is None and "updated_at" in metadata:
        metadata.pop("updated_at")

    metadata["content_hash"] = compute_content_hash(document.page_content)

    for field in _OPTIONAL_TIMESTAMP_FIELDS:
        if field in metadata and not isinstance(metadata[field], str):
            metadata[field] = str(metadata[field])

    return Document(
        id=stable_point_id(
            tenant=tenant,
            doc_id=doc_id,
            chunk_index=chunk_index,
        ),
        page_content=document.page_content,
        metadata=metadata,
    )
