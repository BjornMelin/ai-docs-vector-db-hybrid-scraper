"""Canonical payload validation and normalization utilities for vector storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import blake2b
from typing import Any


__all__ = [
    "CanonicalPayload",
    "PayloadValidationError",
    "compute_content_hash",
    "ensure_canonical_payload",
]


@dataclass(slots=True)
class CanonicalPayload:
    """Normalized payload bundle ready for persistence.

    Attributes:
        point_id: Deterministic identifier for the vector record.
        payload: Canonical payload mapping containing required metadata fields.
    """

    point_id: str
    payload: dict[str, Any]


class PayloadValidationError(ValueError):
    """Raised when payload metadata cannot be coerced into canonical form."""


_REQUIRED_STRING_FIELDS = ("doc_id", "tenant", "source")
_REQUIRED_INT_FIELDS = ("chunk_id",)
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
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        msg = f"Field '{field}' must be coercible to int"
        raise PayloadValidationError(msg) from exc


def ensure_canonical_payload(
    raw_metadata: Mapping[str, Any] | None,
    *,
    content: str,
    id_hint: str,
) -> CanonicalPayload:
    """Validate and normalize metadata for ingestion.

    Args:
        raw_metadata: Original metadata mapping supplied by callers.
        content: Document chunk content used to derive hashes.
        id_hint: Fallback identifier when `doc_id` is missing.

    Returns:
        CanonicalPayload comprised of a deterministic point identifier and
        the normalized payload mapping.
    """
    payload: dict[str, Any] = dict(raw_metadata or {})
    payload.setdefault("content", content)

    doc_id = _coerce_string(payload.get("doc_id") or id_hint, field="doc_id")
    payload["doc_id"] = doc_id

    chunk_id = _coerce_int(
        payload.get("chunk_id", payload.get("chunk_index", 0)),
        field="chunk_id",
    )
    payload["chunk_id"] = chunk_id
    payload.pop("chunk_index", None)

    payload["tenant"] = _coerce_string(
        payload.get("tenant") or "default",
        field="tenant",
    )
    tenant = payload["tenant"]

    payload["source"] = _coerce_string(
        payload.get("source") or payload.get("url") or "unknown",
        field="source",
    )

    created_at = payload.get("created_at")
    if not created_at:
        created_at = datetime.now(UTC).isoformat()
    payload["created_at"] = created_at

    if payload.get("updated_at") is None and "updated_at" in payload:
        payload.pop("updated_at")

    computed_hash = compute_content_hash(content)
    if payload.get("content_hash") and payload["content_hash"] != computed_hash:
        payload["content_hash_previous"] = payload["content_hash"]
    payload["content_hash"] = computed_hash

    for field in _OPTIONAL_TIMESTAMP_FIELDS:
        if field in payload and not isinstance(payload[field], str):
            payload[field] = str(payload[field])

    for field in _REQUIRED_STRING_FIELDS:
        _coerce_string(payload[field], field=field)
    for field in _REQUIRED_INT_FIELDS:
        _coerce_int(payload[field], field=field)

    point_key = f"{tenant}|{doc_id}|{chunk_id}|{computed_hash}"
    point_id = blake2b(
        point_key.encode("utf-8"),
        digest_size=_HASH_DIGEST_SIZE,
    ).hexdigest()

    return CanonicalPayload(point_id=point_id, payload=payload)
