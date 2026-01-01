"""Tests for centralized test doubles."""

import pytest

from src.infrastructure.project_storage import ProjectStorageError
from tests.doubles.services import FakeCache, FakeEmbeddingManager, FakeProjectStorage


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fake_cache_counts_hits_and_misses_with_none_values() -> None:
    """FakeCache treats stored None as a hit (not a miss)."""
    cache = FakeCache()
    await cache.set("stored_none", None)

    assert await cache.get("stored_none") is None
    assert cache.hit_count == 1
    assert cache.miss_count == 0

    assert await cache.get("missing") is None
    assert cache.hit_count == 1
    assert cache.miss_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fake_embedding_manager_exposes_call_count() -> None:
    """FakeEmbeddingManager exposes call_count for assertions."""
    embedder = FakeEmbeddingManager()
    await embedder.embed_query("hello")
    await embedder.embed_documents(["a", "b"])

    assert embedder.call_count == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fake_project_storage_update_missing_project_raises_error() -> None:
    """FakeProjectStorage mirrors production exception types."""
    storage = FakeProjectStorage()

    with pytest.raises(ProjectStorageError):
        await storage.update_project("missing", {"name": "test"})
