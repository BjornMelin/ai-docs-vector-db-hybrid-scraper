"""Tests for cache key helper utilities."""

from src.services.cache.persistent_cache import PersistentCacheManager


def test_embedding_key_considers_dimensions() -> None:
    """Test that embedding keys differ based on dimensions."""

    base = PersistentCacheManager.embedding_key("text", "model", "provider")
    with_dims = PersistentCacheManager.embedding_key(
        "text", "model", "provider", dimensions=1024
    )

    assert base != with_dims
    assert base.startswith("emb:")
    assert with_dims.startswith("emb:1024:")


def test_search_key_is_deterministic() -> None:
    """Test that search keys are deterministic regardless of metadata order."""

    key_a = PersistentCacheManager.search_key("query", {"a": 1, "b": 2})
    key_b = PersistentCacheManager.search_key("query", {"b": 2, "a": 1})

    assert key_a == key_b
    assert key_a.startswith("srch:")
