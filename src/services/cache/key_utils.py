"""Utility helpers for cache key generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def build_embedding_cache_key(
    text: str,
    model: str,
    provider: str,
    dimensions: int | None = None,
) -> str:
    """Return a deterministic cache key for embedding payloads.

    Args:
        text: Raw text that produced the embedding.
        model: Embedding model identifier.
        provider: Provider name that generated the embedding.
        dimensions: Optional embedding dimensionality for disambiguation.

    Returns:
        Prefixed cache key scoped to embedding artefacts.
    """
    digest_source = f"{provider}:{model}:{text}"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    if dimensions is not None:
        return f"emb:{dimensions}:{digest}"
    return f"emb:{digest}"


def build_search_cache_key(query: str, filters: dict[str, Any] | None = None) -> str:
    """Return a deterministic cache key for search responses.

    Args:
        query: Search query text.
        filters: Optional structured filters used for the search request.

    Returns:
        Prefixed cache key scoped to search results.
    """
    serialized_filters = json.dumps(filters or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{query}:{serialized_filters}".encode()).hexdigest()
    return f"srch:{digest}"
