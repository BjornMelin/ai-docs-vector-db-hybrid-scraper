"""Tests for embedding cache module."""

import hashlib
from unittest.mock import AsyncMock

import pytest
from src.services.cache.embedding_cache import EmbeddingCache


class TestEmbeddingCache:
    """Test the EmbeddingCache class."""

    @pytest.fixture
    def mock_dragonfly_cache(self):
        """Create a mock DragonflyCache for testing."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.delete_many.return_value = {}
        mock_cache.exists.return_value = False
        mock_cache.scan_keys.return_value = []
        mock_cache.mget.return_value = []
        mock_cache.mset.return_value = True
        mock_cache.size.return_value = 100
        return mock_cache

    @pytest.fixture
    def embedding_cache(self, mock_dragonfly_cache):
        """Create an EmbeddingCache instance for testing."""
        cache = EmbeddingCache(mock_dragonfly_cache, default_ttl=604800)  # 7 days
        return cache

    def test_embedding_cache_initialization(self, mock_dragonfly_cache):
        """Test EmbeddingCache initialization."""
        cache = EmbeddingCache(mock_dragonfly_cache, default_ttl=86400)

        assert cache.cache == mock_dragonfly_cache
        assert cache.default_ttl == 86400

    def test_embedding_cache_default_initialization(self, mock_dragonfly_cache):
        """Test EmbeddingCache initialization with defaults."""
        cache = EmbeddingCache(mock_dragonfly_cache)
        assert cache.default_ttl == 86400 * 7  # 7 days

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, embedding_cache, mock_dragonfly_cache):
        """Test getting embedding with cache hit."""
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_dragonfly_cache.get.return_value = expected_embedding

        result = await embedding_cache.get_embedding(
            text="test query",
            model="text-embedding-3-small",
            provider="openai",
            dimensions=5,
        )

        assert result == expected_embedding
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test getting embedding with cache miss."""
        mock_dragonfly_cache.get.return_value = None

        result = await embedding_cache.get_embedding(
            text="test query", model="text-embedding-3-small"
        )

        assert result is None
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_dimension_validation(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test dimension validation for cached embeddings."""
        cached_embedding = [0.1, 0.2, 0.3]  # 3 dimensions
        mock_dragonfly_cache.get.return_value = cached_embedding

        # Request 5 dimensions but cache has 3
        result = await embedding_cache.get_embedding(
            text="test query",
            model="text-embedding-3-small",
            dimensions=5,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_type_conversion(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test that cached embeddings are converted to floats."""
        cached_embedding = [1, 2, 3]  # Integers
        mock_dragonfly_cache.get.return_value = cached_embedding

        result = await embedding_cache.get_embedding(
            text="test query", model="text-embedding-3-small"
        )

        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_get_embedding_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test getting embedding with cache exception."""
        mock_dragonfly_cache.get.side_effect = Exception("Cache error")

        result = await embedding_cache.get_embedding(
            text="test query", model="text-embedding-3-small"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_set_embedding_success(self, embedding_cache, mock_dragonfly_cache):
        """Test setting embedding successfully."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_dragonfly_cache.set.return_value = True

        result = await embedding_cache.set_embedding(
            text="test query",
            model="text-embedding-3-small",
            embedding=embedding,
            provider="openai",
            dimensions=5,
            ttl=3600,
        )

        assert result is True
        mock_dragonfly_cache.set.assert_called_once()

        # Verify TTL
        call_args = mock_dragonfly_cache.set.call_args
        assert call_args[1]["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_set_embedding_default_ttl(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test setting embedding with default TTL."""
        embedding = [0.1, 0.2, 0.3]
        mock_dragonfly_cache.set.return_value = True

        await embedding_cache.set_embedding(
            text="test query",
            model="text-embedding-3-small",
            embedding=embedding,
        )

        call_args = mock_dragonfly_cache.set.call_args
        assert call_args[1]["ttl"] == 604800  # 7 days

    @pytest.mark.asyncio
    async def test_set_embedding_invalid_format(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test setting embedding with invalid format."""
        # Test empty embedding
        result = await embedding_cache.set_embedding(
            text="test query",
            model="text-embedding-3-small",
            embedding=[],
        )
        assert result is False

        # Test non-list embedding
        result = await embedding_cache.set_embedding(
            text="test query",
            model="text-embedding-3-small",
            embedding="invalid",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_set_embedding_type_normalization(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test that embeddings are normalized to floats."""
        embedding = [1, 2, 3]  # Integers
        mock_dragonfly_cache.set.return_value = True

        await embedding_cache.set_embedding(
            text="test query",
            model="text-embedding-3-small",
            embedding=embedding,
        )

        call_args = mock_dragonfly_cache.set.call_args
        normalized = call_args[0][1]  # The embedding value
        assert normalized == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in normalized)

    @pytest.mark.asyncio
    async def test_set_embedding_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test setting embedding with cache exception."""
        embedding = [0.1, 0.2, 0.3]
        mock_dragonfly_cache.set.side_effect = Exception("Cache error")

        result = await embedding_cache.set_embedding(
            text="test query",
            model="text-embedding-3-small",
            embedding=embedding,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_success(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding retrieval with mixed hits and misses."""
        texts = ["query1", "query2", "query3"]
        embeddings = [
            [0.1, 0.2, 0.3],  # query1 cached
            None,  # query2 not cached
            [0.4, 0.5, 0.6],  # query3 cached
        ]
        mock_dragonfly_cache.mget.return_value = embeddings

        cached, missing = await embedding_cache.get_batch_embeddings(
            texts=texts,
            model="text-embedding-3-small",
            provider="openai",
            dimensions=3,
        )

        expected_cached = {
            "query1": [0.1, 0.2, 0.3],
            "query3": [0.4, 0.5, 0.6],
        }
        expected_missing = ["query2"]

        assert cached == expected_cached
        assert missing == expected_missing
        mock_dragonfly_cache.mget.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_empty_texts(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding retrieval with empty text list."""
        cached, missing = await embedding_cache.get_batch_embeddings(
            texts=[], model="text-embedding-3-small"
        )

        assert cached == {}
        assert missing == []
        mock_dragonfly_cache.mget.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_dimension_validation(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test dimension validation in batch retrieval."""
        texts = ["query1", "query2"]
        embeddings = [
            [0.1, 0.2, 0.3],  # 3 dimensions
            [0.4, 0.5],  # Wrong dimensions
        ]
        mock_dragonfly_cache.mget.return_value = embeddings

        cached, missing = await embedding_cache.get_batch_embeddings(
            texts=texts,
            model="text-embedding-3-small",
            dimensions=3,  # Expected 3 dimensions
        )

        expected_cached = {"query1": [0.1, 0.2, 0.3]}
        expected_missing = ["query2"]

        assert cached == expected_cached
        assert missing == expected_missing

    @pytest.mark.asyncio
    async def test_get_batch_embeddings_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding retrieval with exception."""
        texts = ["query1", "query2"]
        mock_dragonfly_cache.mget.side_effect = Exception("Batch error")

        cached, missing = await embedding_cache.get_batch_embeddings(
            texts=texts, model="text-embedding-3-small"
        )

        assert cached == {}
        assert missing == texts

    @pytest.mark.asyncio
    async def test_set_batch_embeddings_success(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding caching successfully."""
        embeddings = {
            "query1": [0.1, 0.2, 0.3],
            "query2": [0.4, 0.5, 0.6],
            "query3": [0.7, 0.8, 0.9],
        }
        mock_dragonfly_cache.mset.return_value = True

        result = await embedding_cache.set_batch_embeddings(
            embeddings=embeddings,
            model="text-embedding-3-small",
            provider="openai",
            dimensions=3,
            ttl=3600,
        )

        assert result is True
        mock_dragonfly_cache.mset.assert_called_once()

        # Verify TTL
        call_args = mock_dragonfly_cache.mset.call_args
        assert call_args[1]["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_set_batch_embeddings_empty(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding caching with empty dict."""
        result = await embedding_cache.set_batch_embeddings(
            embeddings={}, model="text-embedding-3-small"
        )

        assert result is True
        mock_dragonfly_cache.mset.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_batch_embeddings_validation(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding caching with validation."""
        embeddings = {
            "valid": [0.1, 0.2, 0.3],
            "empty": [],  # Invalid
            "wrong_type": "not_a_list",  # Invalid
            "wrong_dimensions": [0.1, 0.2],  # Wrong dimensions
        }
        mock_dragonfly_cache.mset.return_value = True

        result = await embedding_cache.set_batch_embeddings(
            embeddings=embeddings,
            model="text-embedding-3-small",
            dimensions=3,
        )

        assert result is True

        # Verify only valid embedding was passed
        call_args = mock_dragonfly_cache.mset.call_args
        cached_data = call_args[0][0]
        assert len(cached_data) == 1  # Only one valid embedding

    @pytest.mark.asyncio
    async def test_set_batch_embeddings_no_valid_data(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding caching with no valid data."""
        embeddings = {
            "invalid1": [],
            "invalid2": "not_a_list",
        }

        result = await embedding_cache.set_batch_embeddings(
            embeddings=embeddings, model="text-embedding-3-small"
        )

        assert result is False
        mock_dragonfly_cache.mset.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_batch_embeddings_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test batch embedding caching with exception."""
        embeddings = {"query1": [0.1, 0.2, 0.3]}
        mock_dragonfly_cache.mset.side_effect = Exception("Batch error")

        result = await embedding_cache.set_batch_embeddings(
            embeddings=embeddings, model="text-embedding-3-small"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_warm_cache_all_missing(self, embedding_cache, mock_dragonfly_cache):
        """Test cache warming with all queries missing."""
        queries = ["query1", "query2", "query3"]
        mock_dragonfly_cache.exists.return_value = False

        missing = await embedding_cache.warm_cache(
            common_queries=queries,
            model="text-embedding-3-small",
            provider="openai",
            dimensions=3,
        )

        assert missing == queries
        assert mock_dragonfly_cache.exists.call_count == 3

    @pytest.mark.asyncio
    async def test_warm_cache_partial_cached(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test cache warming with some queries already cached."""
        queries = ["query1", "query2", "query3"]

        # Mock exists to return True for the second call (query2)
        call_count = 0

        def mock_exists(key):
            nonlocal call_count
            call_count += 1
            return call_count == 2  # Second call returns True (query2)

        mock_dragonfly_cache.exists.side_effect = mock_exists

        missing = await embedding_cache.warm_cache(
            common_queries=queries, model="text-embedding-3-small"
        )

        expected_missing = ["query1", "query3"]
        assert missing == expected_missing

    @pytest.mark.asyncio
    async def test_warm_cache_all_cached(self, embedding_cache, mock_dragonfly_cache):
        """Test cache warming with all queries already cached."""
        queries = ["query1", "query2"]
        mock_dragonfly_cache.exists.return_value = True

        missing = await embedding_cache.warm_cache(
            common_queries=queries, model="text-embedding-3-small"
        )

        assert missing == []

    @pytest.mark.asyncio
    async def test_warm_cache_empty_queries(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test cache warming with empty query list."""
        missing = await embedding_cache.warm_cache(
            common_queries=[], model="text-embedding-3-small"
        )

        assert missing == []
        mock_dragonfly_cache.exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_cache_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test cache warming with exception."""
        queries = ["query1", "query2"]
        mock_dragonfly_cache.exists.side_effect = Exception("Cache error")

        missing = await embedding_cache.warm_cache(
            common_queries=queries, model="text-embedding-3-small"
        )

        assert missing == queries  # All returned as missing on error

    @pytest.mark.asyncio
    async def test_invalidate_model_success(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test model invalidation successfully."""
        keys = [
            "emb:openai:text-embedding-3-small:hash1",
            "emb:openai:text-embedding-3-small:hash2",
            "emb:openai:text-embedding-3-small:hash3",
        ]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.delete_many.return_value = dict.fromkeys(keys, True)

        count = await embedding_cache.invalidate_model(
            model="text-embedding-3-small", provider="openai"
        )

        assert count == 3
        mock_dragonfly_cache.scan_keys.assert_called_once_with(
            "emb:openai:text-embedding-3-small:*"
        )
        mock_dragonfly_cache.delete_many.assert_called_once_with(keys)

    @pytest.mark.asyncio
    async def test_invalidate_model_no_keys(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test model invalidation with no matching keys."""
        mock_dragonfly_cache.scan_keys.return_value = []

        count = await embedding_cache.invalidate_model(
            model="text-embedding-3-small", provider="openai"
        )

        assert count == 0
        mock_dragonfly_cache.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_model_with_batches(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test model invalidation with large number of keys in batches."""
        # Create 250 keys to test batching (batch_size = 100)
        keys = [f"emb:openai:model:hash{i}" for i in range(250)]
        mock_dragonfly_cache.scan_keys.return_value = keys

        # Mock delete_many to return success for all keys
        def mock_delete_many(batch):
            return dict.fromkeys(batch, True)

        mock_dragonfly_cache.delete_many.side_effect = mock_delete_many

        count = await embedding_cache.invalidate_model(model="model", provider="openai")

        assert count == 250
        # Should be called 3 times (100, 100, 50)
        assert mock_dragonfly_cache.delete_many.call_count == 3

    @pytest.mark.asyncio
    async def test_invalidate_model_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test model invalidation with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        count = await embedding_cache.invalidate_model(
            model="text-embedding-3-small", provider="openai"
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self, embedding_cache, mock_dragonfly_cache):
        """Test getting cache statistics successfully."""
        keys = [
            "emb:openai:text-embedding-3-small:384:hash1",
            "emb:openai:text-embedding-ada-002:1536:hash2",
            "emb:fastembed:BAAI/bge-small-en-v1.5:384:hash3",
            "emb:openai:text-embedding-3-small:384:hash4",
        ]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.size.return_value = 1000

        stats = await embedding_cache.get_cache_stats()

        expected_stats = {
            "total_embeddings": 4,
            "cache_size": 1000,
            "by_provider": {"openai": 3, "fastembed": 1},
            "by_model": {
                "openai:text-embedding-3-small": 2,
                "openai:text-embedding-ada-002": 1,
                "fastembed:BAAI/bge-small-en-v1.5": 1,
            },
        }

        assert stats == expected_stats

    @pytest.mark.asyncio
    async def test_get_cache_stats_with_malformed_keys(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test getting cache statistics with malformed keys."""
        keys = [
            "emb:openai:text-embedding-3-small:384:hash1",  # Valid
            "malformed_key",  # Malformed
            "emb:openai:hash2",  # Short
        ]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.size.return_value = 500

        stats = await embedding_cache.get_cache_stats()

        # Should handle malformed keys gracefully
        assert stats["total_embeddings"] == 3
        assert stats["by_provider"] == {"openai": 1}
        assert stats["by_model"] == {"openai:text-embedding-3-small": 1}

    @pytest.mark.asyncio
    async def test_get_cache_stats_with_exception(
        self, embedding_cache, mock_dragonfly_cache
    ):
        """Test getting cache statistics with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        stats = await embedding_cache.get_cache_stats()

        assert "error" in stats

    def test_get_key_generation(self, embedding_cache):
        """Test cache key generation."""
        key = embedding_cache._get_key(
            text="test query",
            model="text-embedding-3-small",
            provider="openai",
            dimensions=384,
        )

        # Should have format: emb:{provider}:{model}:{dimensions}:{hash}
        parts = key.split(":")
        assert len(parts) == 5
        assert parts[0] == "emb"
        assert parts[1] == "openai"
        assert parts[2] == "text-embedding-3-small"
        assert parts[3] == "384"
        assert len(parts[4]) == 32  # MD5 hash length

    def test_get_key_without_dimensions(self, embedding_cache):
        """Test cache key generation without dimensions."""
        key = embedding_cache._get_key(
            text="test query",
            model="text-embedding-3-small",
            provider="openai",
        )

        # Should have format: emb:{provider}:{model}:{hash}
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0] == "emb"
        assert parts[1] == "openai"
        assert parts[2] == "text-embedding-3-small"
        assert len(parts[3]) == 32  # MD5 hash length

    def test_get_key_normalization(self, embedding_cache):
        """Test that key generation normalizes text."""
        key1 = embedding_cache._get_key(
            text="  Test Query  ",
            model="model1",
            provider="provider1",
        )

        key2 = embedding_cache._get_key(
            text="test query",
            model="model1",
            provider="provider1",
        )

        assert key1 == key2

    def test_get_key_consistency(self, embedding_cache):
        """Test cache key generation consistency."""
        key1 = embedding_cache._get_key(
            text="test query",
            model="text-embedding-3-small",
            provider="openai",
            dimensions=384,
        )

        key2 = embedding_cache._get_key(
            text="test query",
            model="text-embedding-3-small",
            provider="openai",
            dimensions=384,
        )

        assert key1 == key2

    def test_get_key_hash_verification(self, embedding_cache):
        """Test that key generation produces correct hash."""
        text = "test query"
        key = embedding_cache._get_key(
            text=text,
            model="model1",
            provider="provider1",
        )

        # Extract hash from key
        key_hash = key.split(":")[-1]

        # Generate expected hash
        normalized_text = text.lower().strip()
        expected_hash = hashlib.md5(normalized_text.encode()).hexdigest()

        assert key_hash == expected_hash

    def test_provider_required_parameter(self, embedding_cache):
        """Test that provider parameter is required."""
        # Test that provider is included in key generation
        key = embedding_cache._get_key(text="test", model="model1", provider="openai")

        # Should include provider "openai" in key
        assert ":openai:" in key

        # Test different provider
        key2 = embedding_cache._get_key(
            text="test", model="model1", provider="fastembed"
        )

        # Should include provider "fastembed" in key
        assert ":fastembed:" in key2

        # Keys should be different for different providers
        assert key != key2
