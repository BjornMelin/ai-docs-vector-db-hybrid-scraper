"""Tests for cache key helper utilities."""

from src.services.cache.key_utils import (
    build_embedding_cache_key,
    build_search_cache_key,
)


def test_embedding_key_considers_dimensions() -> None:
    """Test that embedding keys differ based on dimensions."""

    base = build_embedding_cache_key("text", "model", "provider")
    with_dims = build_embedding_cache_key("text", "model", "provider", dimensions=1024)

    assert base != with_dims
    assert base.startswith("emb:")
    assert with_dims.startswith("emb:1024:")


def test_search_key_is_deterministic() -> None:
    """Test that search keys are deterministic regardless of metadata order."""

    key_a = build_search_cache_key("query", {"a": 1, "b": 2})
    key_b = build_search_cache_key("query", {"b": 2, "a": 1})

    assert key_a == key_b
    assert key_a.startswith("srch:")
