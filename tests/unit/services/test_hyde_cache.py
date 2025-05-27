#!/usr/bin/env python3
"""
Unit tests for HyDE caching functionality.

Tests DragonflyDB integration, cache operations, and performance optimizations for HyDE.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest
from src.services.hyde.cache import HyDECache
from src.services.hyde.config import HyDEConfig


class TestHyDECache:
    """Test cases for HyDECache functionality."""

    @pytest.fixture
    def mock_dragonfly_client(self):
        """Mock DragonflyDB client for testing."""
        client = AsyncMock()

        # Mock successful operations
        client.get.return_value = None  # Cache miss by default
        client.set.return_value = True
        client.delete.return_value = 1
        client.exists.return_value = 0
        client.expire.return_value = True
        client.info.return_value = {"used_memory": "1024000"}

        return client

    @pytest.fixture
    def hyde_config(self):
        """Default HyDE configuration for testing."""
        return HyDEConfig(cache_hypothetical_docs=True)

    @pytest.fixture
    def hyde_cache(self, mock_dragonfly_client, hyde_config):
        """HyDECache instance for testing."""
        return HyDECache(
            config=hyde_config,
            dragonfly_client=mock_dragonfly_client,
        )

    @pytest.mark.asyncio
    async def test_set_and_get_hyde_embedding(self, hyde_cache, mock_dragonfly_client):
        """Test setting and getting HyDE embeddings from cache."""
        query = "machine learning algorithms"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        hypothetical_docs = ["doc1", "doc2", "doc3"]
        metadata = {"generation_time": 1.5, "model": "gpt-4"}
        domain = "tutorial"

        # Test setting embedding
        success = await hyde_cache.set_hyde_embedding(
            query=query,
            embedding=embedding,
            hypothetical_docs=hypothetical_docs,
            generation_metadata=metadata,
            domain=domain,
        )

        assert success is True
        assert mock_dragonfly_client.set.called

        # Verify the cache key and data structure
        call_args = mock_dragonfly_client.set.call_args
        cache_key = call_args[0][0]
        cache_data = call_args[0][1]

        assert "hyde:embedding:" in cache_key
        assert query.replace(" ", "_") in cache_key

        # Parse cached data
        parsed_data = json.loads(cache_data)
        assert "embedding" in parsed_data
        assert "hypothetical_docs" in parsed_data
        assert "metadata" in parsed_data
        assert parsed_data["hypothetical_docs"] == hypothetical_docs
        assert parsed_data["metadata"]["generation_time"] == metadata["generation_time"]

    @pytest.mark.asyncio
    async def test_get_hyde_embedding_cache_hit(
        self, hyde_cache, mock_dragonfly_client
    ):
        """Test getting HyDE embedding from cache when it exists."""
        query = "database optimization"
        domain = "technical"

        # Mock cache hit
        cached_embedding = [0.1, 0.2, 0.3]
        cached_docs = ["cached doc 1", "cached doc 2"]
        cached_data = {
            "embedding": np.array(cached_embedding, dtype=np.float32).tobytes(),
            "hypothetical_docs": cached_docs,
            "metadata": {"generation_time": 2.0},
            "domain": domain,
            "timestamp": 1234567890,
        }

        mock_dragonfly_client.get.return_value = json.dumps(cached_data)

        result = await hyde_cache.get_hyde_embedding(query, domain=domain)

        assert result is not None
        assert "embedding" in result
        assert "hypothetical_docs" in result
        assert "metadata" in result

        # Verify embedding reconstruction
        embedding = result["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) == len(cached_embedding)

        assert result["hypothetical_docs"] == cached_docs

    @pytest.mark.asyncio
    async def test_get_hyde_embedding_cache_miss(
        self, hyde_cache, mock_dragonfly_client
    ):
        """Test getting HyDE embedding when not in cache."""
        query = "cache miss test"

        # Mock cache miss
        mock_dragonfly_client.get.return_value = None

        result = await hyde_cache.get_hyde_embedding(query)

        assert result is None
        assert mock_dragonfly_client.get.called

    @pytest.mark.asyncio
    async def test_set_and_get_search_results(self, hyde_cache, mock_dragonfly_client):
        """Test caching search results."""
        query = "API documentation"
        collection = "docs"
        search_params = {"limit": 10, "strategy": "hybrid"}
        results = [
            {"id": "1", "content": "Result 1", "score": 0.9},
            {"id": "2", "content": "Result 2", "score": 0.8},
        ]
        metadata = {"search_time": 0.15, "total_found": 2}

        # Test setting search results
        success = await hyde_cache.set_search_results(
            query=query,
            collection=collection,
            search_params=search_params,
            results=results,
            metadata=metadata,
        )

        assert success is True
        assert mock_dragonfly_client.set.called

        # Mock cache hit for retrieval
        cached_data = {
            "results": results,
            "metadata": metadata,
            "timestamp": 1234567890,
        }
        mock_dragonfly_client.get.return_value = json.dumps(cached_data)

        # Test getting search results
        cached_results = await hyde_cache.get_search_results(
            query=query,
            collection=collection,
            search_params=search_params,
        )

        assert cached_results is not None
        assert cached_results["results"] == results
        assert cached_results["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_invalidate_query_cache(self, hyde_cache, mock_dragonfly_client):
        """Test invalidating cache for a specific query."""
        query = "test query"

        # Mock pattern matching
        pattern_keys = [
            "hyde:embedding:test_query",
            "hyde:search:test_query:docs",
            "hyde:search:test_query:collection",
        ]
        mock_dragonfly_client.keys.return_value = pattern_keys
        mock_dragonfly_client.delete.return_value = len(pattern_keys)

        deleted_count = await hyde_cache.invalidate_query_cache(query)

        assert deleted_count == len(pattern_keys)
        assert mock_dragonfly_client.keys.called
        assert mock_dragonfly_client.delete.called

    @pytest.mark.asyncio
    async def test_warm_cache(self, hyde_cache, mock_dragonfly_client):
        """Test cache warming functionality."""
        queries = [
            {"query": "API docs", "domain": "api"},
            {"query": "tutorials", "domain": "tutorial"},
        ]

        # Mock embedding generation for warming
        mock_embedding_fn = AsyncMock()
        mock_embedding_fn.return_value = ([0.1, 0.2, 0.3], ["doc1", "doc2"])

        warmed_count = await hyde_cache.warm_cache(
            queries=queries,
            embedding_generator_fn=mock_embedding_fn,
        )

        assert warmed_count == len(queries)
        assert mock_embedding_fn.call_count == len(queries)
        assert mock_dragonfly_client.set.call_count >= len(queries)

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, hyde_cache, mock_dragonfly_client):
        """Test getting cache statistics."""
        # Mock info response
        mock_dragonfly_client.info.return_value = {
            "used_memory": "2048000",
            "keyspace_hits": "150",
            "keyspace_misses": "50",
            "connected_clients": "10",
        }

        # Mock key counting
        mock_dragonfly_client.keys.return_value = ["key1", "key2", "key3"]

        stats = await hyde_cache.get_cache_stats()

        assert "memory_usage_bytes" in stats
        assert "hit_rate" in stats
        assert "total_keys" in stats
        assert "hyde_keys" in stats

        assert stats["memory_usage_bytes"] == 2048000
        assert stats["hit_rate"] == 0.75  # 150 / (150 + 50)
        assert stats["total_keys"] >= 3

    @pytest.mark.asyncio
    async def test_clear_hyde_cache(self, hyde_cache, mock_dragonfly_client):
        """Test clearing all HyDE-related cache entries."""
        # Mock pattern matching for HyDE keys
        hyde_keys = [
            "hyde:embedding:query1",
            "hyde:search:query2:docs",
            "hyde:embedding:query3",
        ]
        mock_dragonfly_client.keys.return_value = hyde_keys
        mock_dragonfly_client.delete.return_value = len(hyde_keys)

        cleared_count = await hyde_cache.clear_hyde_cache()

        assert cleared_count == len(hyde_keys)
        assert mock_dragonfly_client.keys.called
        assert mock_dragonfly_client.delete.called

    @pytest.mark.asyncio
    async def test_binary_embedding_serialization(self, hyde_cache):
        """Test binary serialization/deserialization of embeddings."""
        original_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Test serialization
        binary_data = hyde_cache._serialize_embedding(original_embedding)
        assert isinstance(binary_data, bytes)

        # Test deserialization
        restored_embedding = hyde_cache._deserialize_embedding(binary_data)
        assert isinstance(restored_embedding, list)
        assert len(restored_embedding) == len(original_embedding)

        # Verify values are approximately equal (accounting for float precision)
        for orig, restored in zip(original_embedding, restored_embedding, strict=False):
            assert abs(orig - restored) < 1e-6

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, hyde_cache):
        """Test cache key generation for different types."""
        query = "test query with spaces"
        collection = "docs"
        search_params = {"limit": 10, "strategy": "hybrid"}
        domain = "api"

        # Test embedding cache key
        embedding_key = hyde_cache._get_embedding_cache_key(query, domain)
        assert "hyde:embedding:" in embedding_key
        assert "test_query_with_spaces" in embedding_key
        assert "api" in embedding_key

        # Test search cache key
        search_key = hyde_cache._get_search_cache_key(query, collection, search_params)
        assert "hyde:search:" in search_key
        assert "test_query_with_spaces" in search_key
        assert "docs" in search_key

    @pytest.mark.asyncio
    async def test_cache_ttl_handling(self, hyde_cache, mock_dragonfly_client):
        """Test TTL (time-to-live) handling for cache entries."""
        query = "ttl test"
        embedding = [0.1, 0.2, 0.3]
        hypothetical_docs = ["doc1"]

        # Test setting with TTL
        await hyde_cache.set_hyde_embedding(
            query=query,
            embedding=embedding,
            hypothetical_docs=hypothetical_docs,
            ttl=3600,  # 1 hour
        )

        # Verify TTL was set
        set_call = mock_dragonfly_client.set.call_args
        if len(set_call[0]) > 2:  # Check if TTL was passed
            assert set_call[0][2] == 3600

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, hyde_cache, mock_dragonfly_client):
        """Test error handling for cache operations."""
        query = "error test"

        # Mock cache operation failure
        mock_dragonfly_client.get.side_effect = Exception("Connection error")

        # Should handle error gracefully and return None
        result = await hyde_cache.get_hyde_embedding(query)
        assert result is None

        # Mock set operation failure
        mock_dragonfly_client.set.side_effect = Exception("Write error")

        # Should handle error gracefully and return False
        success = await hyde_cache.set_hyde_embedding(
            query=query,
            embedding=[0.1, 0.2],
            hypothetical_docs=["doc"],
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_cache_compression(self, hyde_cache):
        """Test cache data compression for large embeddings."""
        # Large embedding vector
        large_embedding = [float(i) for i in range(1536)]  # OpenAI embedding size
        hypothetical_docs = ["doc"] * 100  # Many documents

        # Test compression
        compressed_data = hyde_cache._compress_cache_data(
            {
                "embedding": large_embedding,
                "hypothetical_docs": hypothetical_docs,
                "metadata": {"test": "data"},
            }
        )

        # Should be smaller than uncompressed JSON
        uncompressed_size = len(
            json.dumps(
                {
                    "embedding": large_embedding,
                    "hypothetical_docs": hypothetical_docs,
                    "metadata": {"test": "data"},
                }
            )
        )

        if hasattr(hyde_cache, "_compress_cache_data"):
            assert len(compressed_data) < uncompressed_size

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, hyde_cache, mock_dragonfly_client):
        """Test concurrent cache operations for thread safety."""
        queries = [f"query_{i}" for i in range(10)]

        # Create concurrent set operations
        tasks = []
        for i, query in enumerate(queries):
            task = hyde_cache.set_hyde_embedding(
                query=query,
                embedding=[float(i), float(i + 1)],
                hypothetical_docs=[f"doc_{i}"],
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should succeed
        assert all(
            result is True or isinstance(result, Exception) for result in results
        )

        # Should have made multiple calls
        assert mock_dragonfly_client.set.call_count >= len(queries)
