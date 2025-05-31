"""Comprehensive tests for EmbeddingCache service."""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.services.cache.embedding_cache import EmbeddingCache


@pytest.fixture
def mock_base_cache():
    """Create mock base cache instance."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    cache.scan_keys = AsyncMock(return_value=[])
    cache.mget = AsyncMock(return_value=[])
    cache.get_stats = MagicMock(
        return_value={
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
        }
    )
    return cache


@pytest.fixture
def embedding_cache(mock_base_cache):
    """Create EmbeddingCache instance for testing."""
    return EmbeddingCache(mock_base_cache)


class TestEmbeddingCacheInitialization:
    """Test embedding cache initialization."""

    def test_cache_initialization(self, embedding_cache, mock_base_cache):
        """Test basic cache initialization."""
        assert embedding_cache._cache == mock_base_cache
        assert embedding_cache._prefix == "embedding:"
        assert embedding_cache._metadata_prefix == "embedding_meta:"
        assert embedding_cache.stats["cache_hits"] == 0
        assert embedding_cache.stats["cache_misses"] == 0

    def test_custom_prefix(self, mock_base_cache):
        """Test initialization with custom prefix."""
        cache = EmbeddingCache(mock_base_cache, prefix="custom_emb:")
        assert cache._prefix == "custom_emb:"
        assert cache._metadata_prefix == "custom_emb:meta:"


class TestKeyGeneration:
    """Test cache key generation logic."""

    def test_generate_cache_key_basic(self, embedding_cache):
        """Test basic cache key generation."""
        text = "This is a test document"
        provider = "openai"
        model = "text-embedding-3-small"
        dimensions = 1536

        key = embedding_cache._generate_cache_key(text, provider, model, dimensions)

        # Key should include prefix and hash
        assert key.startswith(embedding_cache._prefix)
        assert provider in key
        assert model in key
        assert str(dimensions) in key

        # Same inputs should generate same key
        key2 = embedding_cache._generate_cache_key(text, provider, model, dimensions)
        assert key == key2

    def test_generate_cache_key_different_inputs(self, embedding_cache):
        """Test cache keys are different for different inputs."""
        base_text = "Test document"

        key1 = embedding_cache._generate_cache_key(base_text, "openai", "model1", 1536)
        key2 = embedding_cache._generate_cache_key(
            base_text, "fastembed", "model1", 1536
        )
        key3 = embedding_cache._generate_cache_key(base_text, "openai", "model2", 1536)
        key4 = embedding_cache._generate_cache_key(base_text, "openai", "model1", 768)
        key5 = embedding_cache._generate_cache_key(
            "Different text", "openai", "model1", 1536
        )

        # All keys should be different
        keys = [key1, key2, key3, key4, key5]
        assert len(set(keys)) == len(keys)

    def test_generate_cache_key_unicode(self, embedding_cache):
        """Test cache key generation with unicode text."""
        text = "This is a test with emojis 🚀 and unicode: 你好世界"
        key = embedding_cache._generate_cache_key(text, "openai", "model", 1536)

        # Should handle unicode properly
        assert isinstance(key, str)
        assert key.startswith(embedding_cache._prefix)

    def test_generate_cache_key_long_text(self, embedding_cache):
        """Test cache key generation with very long text."""
        long_text = "x" * 10000  # 10K characters
        key = embedding_cache._generate_cache_key(long_text, "openai", "model", 1536)

        # Key length should be reasonable despite long input
        assert len(key) < 200  # Reasonable key length

    def test_text_hash_consistency(self, embedding_cache):
        """Test text hashing is consistent."""
        text = "Consistent hashing test"

        hash1 = embedding_cache._hash_text(text)
        hash2 = embedding_cache._hash_text(text)

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256


class TestGetEmbedding:
    """Test embedding retrieval operations."""

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self, embedding_cache, mock_base_cache):
        """Test getting embedding with cache miss."""
        mock_base_cache.get.return_value = None

        result = await embedding_cache.get_embedding(
            text="Test text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        )

        assert result is None
        assert embedding_cache.stats["cache_misses"] == 1
        mock_base_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, embedding_cache, mock_base_cache):
        """Test getting embedding with cache hit."""
        cached_data = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 5,
                "cached_at": datetime.now(UTC).isoformat(),
            },
        }
        mock_base_cache.get.return_value = cached_data

        result = await embedding_cache.get_embedding(
            text="Test text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=5,
        )

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embedding_cache.stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_get_embedding_dimension_mismatch(
        self, embedding_cache, mock_base_cache
    ):
        """Test handling of dimension mismatch."""
        cached_data = {
            "embedding": [0.1, 0.2, 0.3],  # 3 dimensions
            "metadata": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 3,
                "cached_at": datetime.now(UTC).isoformat(),
            },
        }
        mock_base_cache.get.return_value = cached_data

        # Request with different dimensions
        result = await embedding_cache.get_embedding(
            text="Test text",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,  # Different from cached
        )

        # Should return None due to dimension mismatch
        assert result is None
        assert embedding_cache.stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_get_embedding_with_metadata(self, embedding_cache, mock_base_cache):
        """Test getting embedding with metadata."""
        cached_data = {
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {
                "provider": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimensions": 3,
                "cached_at": datetime.now(UTC).isoformat(),
                "generation_time_ms": 25.5,
                "custom_field": "test",
            },
        }
        mock_base_cache.get.return_value = cached_data

        embedding, metadata = await embedding_cache.get_embedding_with_metadata(
            text="Test text",
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            dimensions=3,
        )

        assert embedding == [0.1, 0.2, 0.3]
        assert metadata["provider"] == "fastembed"
        assert metadata["generation_time_ms"] == 25.5
        assert metadata["custom_field"] == "test"


class TestSetEmbedding:
    """Test embedding storage operations."""

    @pytest.mark.asyncio
    async def test_set_embedding_basic(self, embedding_cache, mock_base_cache):
        """Test basic embedding storage."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        success = await embedding_cache.set_embedding(
            text="Test text",
            model="text-embedding-3-small",
            embedding=embedding,
            provider="openai",
            dimensions=5,
        )

        assert success is True

        # Verify data structure stored
        call_args = mock_base_cache.set.call_args
        stored_data = call_args[0][1]

        assert stored_data["embedding"] == embedding
        assert stored_data["metadata"]["provider"] == "openai"
        assert stored_data["metadata"]["model"] == "text-embedding-3-small"
        assert stored_data["metadata"]["dimensions"] == 5
        assert "cached_at" in stored_data["metadata"]

        # Check TTL was set (default for embeddings)
        assert call_args[1]["ttl"] == 86400  # 1 day default

    @pytest.mark.asyncio
    async def test_set_embedding_custom_ttl(self, embedding_cache, mock_base_cache):
        """Test embedding storage with custom TTL."""
        embedding = [0.1, 0.2, 0.3]

        success = await embedding_cache.set_embedding(
            text="Test text",
            model="model",
            embedding=embedding,
            provider="provider",
            dimensions=3,
            ttl=7200,  # 2 hours
        )

        assert success is True

        # Verify custom TTL
        call_args = mock_base_cache.set.call_args
        assert call_args[1]["ttl"] == 7200

    @pytest.mark.asyncio
    async def test_set_embedding_with_metadata(self, embedding_cache, mock_base_cache):
        """Test embedding storage with additional metadata."""
        embedding = [0.1, 0.2, 0.3]
        extra_metadata = {
            "generation_time_ms": 15.2,
            "model_version": "v2",
            "preprocessing": "normalized",
        }

        success = await embedding_cache.set_embedding(
            text="Test text",
            model="model",
            embedding=embedding,
            provider="provider",
            dimensions=3,
            metadata=extra_metadata,
        )

        assert success is True

        # Verify metadata was merged
        call_args = mock_base_cache.set.call_args
        stored_metadata = call_args[0][1]["metadata"]

        assert stored_metadata["generation_time_ms"] == 15.2
        assert stored_metadata["model_version"] == "v2"
        assert stored_metadata["preprocessing"] == "normalized"
        # Standard fields should still be present
        assert stored_metadata["provider"] == "provider"
        assert stored_metadata["model"] == "model"


class TestBatchOperations:
    """Test batch embedding operations."""

    @pytest.mark.asyncio
    async def test_get_batch_embeddings(self, embedding_cache, mock_base_cache):
        """Test batch retrieval of embeddings."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Mock cache responses: hit, miss, hit
        cached_values = [
            {"embedding": [0.1, 0.2], "metadata": {"dimensions": 2}},
            None,
            {"embedding": [0.3, 0.4], "metadata": {"dimensions": 2}},
        ]
        mock_base_cache.mget.return_value = cached_values

        results = await embedding_cache.get_batch_embeddings(
            texts=texts, provider="openai", model="model", dimensions=2
        )

        assert len(results) == 3
        assert results[0] == [0.1, 0.2]
        assert results[1] is None
        assert results[2] == [0.3, 0.4]

        # Stats should reflect batch operation
        assert embedding_cache.stats["cache_hits"] == 2
        assert embedding_cache.stats["cache_misses"] == 1
        assert embedding_cache.stats["batch_operations"] == 1

    @pytest.mark.asyncio
    async def test_set_batch_embeddings(self, embedding_cache, mock_base_cache):
        """Test batch storage of embeddings."""
        embeddings_map = {
            "Text 1": [0.1, 0.2],
            "Text 2": [0.3, 0.4],
            "Text 3": [0.5, 0.6],
        }

        # Mock batch set to return all successful
        mock_base_cache.mset = AsyncMock(return_value=True)

        success = await embedding_cache.set_batch_embeddings(
            embeddings_map=embeddings_map,
            provider="fastembed",
            model="model",
            dimensions=2,
        )

        assert success is True
        assert embedding_cache.stats["batch_operations"] == 1

        # Verify batch set was called
        mock_base_cache.mset.assert_called_once()
        call_args = mock_base_cache.mset.call_args
        items = call_args[0][0]

        assert len(items) == 3
        # Each item should have proper structure
        for _key, value in items.items():
            assert "embedding" in value
            assert "metadata" in value


class TestCacheInvalidation:
    """Test cache invalidation operations."""

    @pytest.mark.asyncio
    async def test_invalidate_by_text(self, embedding_cache, mock_base_cache):
        """Test invalidating cache entry by text."""
        text = "Text to invalidate"

        # Mock that the key exists
        mock_base_cache.exists.return_value = True

        success = await embedding_cache.invalidate_embedding(
            text=text, provider="openai", model="model", dimensions=1536
        )

        assert success is True
        mock_base_cache.delete.assert_called_once()
        assert embedding_cache.stats["invalidations"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self, embedding_cache, mock_base_cache):
        """Test invalidating non-existent entry."""
        mock_base_cache.exists.return_value = False
        mock_base_cache.delete.return_value = False

        success = await embedding_cache.invalidate_embedding(
            text="Nonexistent", provider="openai", model="model", dimensions=1536
        )

        assert success is False
        assert embedding_cache.stats["invalidations"] == 0

    @pytest.mark.asyncio
    async def test_invalidate_by_provider(self, embedding_cache, mock_base_cache):
        """Test invalidating all embeddings for a provider."""
        # Mock finding keys for provider
        mock_base_cache.scan_keys.return_value = [
            "embedding:openai:model1:xxx",
            "embedding:openai:model2:yyy",
        ]

        count = await embedding_cache.invalidate_by_provider("openai")

        assert count == 2
        assert mock_base_cache.delete.call_count == 2
        mock_base_cache.scan_keys.assert_called_once_with("embedding:openai:*")

    @pytest.mark.asyncio
    async def test_invalidate_by_model(self, embedding_cache, mock_base_cache):
        """Test invalidating all embeddings for a specific model."""
        # Mock finding keys for model
        mock_base_cache.scan_keys.return_value = [
            "embedding:openai:text-embedding-3-small:1536:xxx",
            "embedding:openai:text-embedding-3-small:768:yyy",
        ]

        count = await embedding_cache.invalidate_by_model(
            provider="openai", model="text-embedding-3-small"
        )

        assert count == 2
        mock_base_cache.scan_keys.assert_called_once_with(
            "embedding:openai:text-embedding-3-small:*"
        )


class TestCacheStatistics:
    """Test cache statistics functionality."""

    def test_get_stats(self, embedding_cache, mock_base_cache):
        """Test getting cache statistics."""
        # Set up some stats
        embedding_cache.stats["cache_hits"] = 150
        embedding_cache.stats["cache_misses"] = 50
        embedding_cache.stats["batch_operations"] = 10
        embedding_cache.stats["invalidations"] = 5

        # Mock base cache stats
        mock_base_cache.get_stats.return_value = {
            "hits": 200,
            "misses": 100,
            "hit_rate": 0.667,
        }

        stats = embedding_cache.get_stats()

        assert stats["embedding_hits"] == 150
        assert stats["embedding_misses"] == 50
        assert stats["embedding_hit_rate"] == 0.75
        assert stats["batch_operations"] == 10
        assert stats["invalidations"] == 5
        assert stats["base_cache_stats"]["hits"] == 200

    def test_reset_stats(self, embedding_cache):
        """Test resetting statistics."""
        # Set some stats
        embedding_cache.stats["cache_hits"] = 100
        embedding_cache.stats["cache_misses"] = 50

        embedding_cache.reset_stats()

        assert embedding_cache.stats["cache_hits"] == 0
        assert embedding_cache.stats["cache_misses"] == 0
        assert all(v == 0 for v in embedding_cache.stats.values())


class TestAdvancedFeatures:
    """Test advanced embedding cache features."""

    @pytest.mark.asyncio
    async def test_warmup_cache(self, embedding_cache, mock_base_cache):
        """Test cache warmup functionality."""
        # Prepare warmup data
        warmup_texts = [
            "Frequently accessed text 1",
            "Frequently accessed text 2",
            "Frequently accessed text 3",
        ]
        warmup_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        # Mock batch set success
        mock_base_cache.mset = AsyncMock(return_value=True)

        success = await embedding_cache.warmup_cache(
            texts=warmup_texts,
            embeddings=warmup_embeddings,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
            ttl=86400,
        )

        assert success is True
        mock_base_cache.mset.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_cache_entries(self, embedding_cache, mock_base_cache):
        """Test exporting cache entries for backup/analysis."""
        # Mock scan and get operations
        mock_base_cache.scan_keys.return_value = [
            "embedding:openai:model:1536:hash1",
            "embedding:openai:model:1536:hash2",
        ]

        mock_base_cache.mget.return_value = [
            {
                "embedding": [0.1, 0.2],
                "metadata": {"text_hash": "hash1", "original_text": "Text 1"},
            },
            {
                "embedding": [0.3, 0.4],
                "metadata": {"text_hash": "hash2", "original_text": "Text 2"},
            },
        ]

        entries = await embedding_cache.export_entries(
            provider="openai", model="model", include_embeddings=True
        )

        assert len(entries) == 2
        assert entries[0]["text"] == "Text 1"
        assert entries[0]["embedding"] == [0.1, 0.2]
        assert entries[1]["text"] == "Text 2"
        assert entries[1]["embedding"] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_analyze_cache_usage(self, embedding_cache, mock_base_cache):
        """Test cache usage analysis."""
        # Mock finding keys and their metadata
        mock_base_cache.scan_keys.return_value = [
            "embedding:openai:model1:1536:hash1",
            "embedding:openai:model1:768:hash2",
            "embedding:fastembed:model2:384:hash3",
        ]

        # Mock TTL values
        mock_base_cache.ttl = AsyncMock(side_effect=[3600, 7200, -1])

        analysis = await embedding_cache.analyze_usage()

        assert analysis["total_entries"] == 3
        assert analysis["by_provider"]["openai"] == 2
        assert analysis["by_provider"]["fastembed"] == 1
        assert analysis["by_dimensions"]["1536"] == 1
        assert analysis["by_dimensions"]["768"] == 1
        assert analysis["by_dimensions"]["384"] == 1
        assert analysis["expiring_soon"] == 1  # TTL < 1 hour
        assert analysis["persistent"] == 1  # TTL = -1

    @pytest.mark.asyncio
    async def test_embedding_versioning(self, embedding_cache, mock_base_cache):
        """Test embedding versioning support."""
        text = "Versioned text"
        embedding_v1 = [0.1, 0.2, 0.3]
        embedding_v2 = [0.4, 0.5, 0.6]

        # Store different versions
        await embedding_cache.set_embedding_versioned(
            text=text,
            model="model-v1",
            embedding=embedding_v1,
            provider="openai",
            dimensions=3,
            version="1.0",
        )

        await embedding_cache.set_embedding_versioned(
            text=text,
            model="model-v2",
            embedding=embedding_v2,
            provider="openai",
            dimensions=3,
            version="2.0",
        )

        # Retrieve specific versions
        await embedding_cache.get_embedding_versioned(
            text=text, provider="openai", model="model-v1", dimensions=3, version="1.0"
        )

        await embedding_cache.get_embedding_versioned(
            text=text, provider="openai", model="model-v2", dimensions=3, version="2.0"
        )

        # Verify versioning works correctly
        assert mock_base_cache.set.call_count >= 2
        assert mock_base_cache.get.call_count >= 2
