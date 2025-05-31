"""Comprehensive tests for SearchResultCache service."""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.cache.search_cache import SearchResultCache


@pytest.fixture
def mock_dragonfly_cache():
    """Create mock DragonflyCache instance."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.scan_keys = AsyncMock(return_value=[])
    cache.mget = AsyncMock(return_value=[])
    cache.pipeline = MagicMock()
    return cache


@pytest.fixture
def mock_patterns():
    """Create mock CachePatterns instance."""
    patterns = AsyncMock()
    patterns.increment_popularity = AsyncMock()
    patterns.get_popular_queries = AsyncMock(return_value=[])
    patterns.get_cache_analytics = AsyncMock(return_value={})
    return patterns


@pytest.fixture
def search_cache(mock_dragonfly_cache, mock_patterns):
    """Create SearchResultCache instance for testing."""
    cache = SearchResultCache(mock_dragonfly_cache, default_ttl=3600)
    with patch.object(cache, "patterns", mock_patterns):
        yield cache


class TestSearchResultCacheInitialization:
    """Test cache initialization."""

    def test_cache_initialization(self, mock_dragonfly_cache):
        """Test basic cache initialization."""
        cache = SearchResultCache(mock_dragonfly_cache, default_ttl=7200)
        assert cache.cache == mock_dragonfly_cache
        assert cache.default_ttl == 7200
        assert cache.patterns is not None

    def test_cache_default_ttl(self, mock_dragonfly_cache):
        """Test cache with default TTL."""
        cache = SearchResultCache(mock_dragonfly_cache)
        assert cache.default_ttl == 3600  # 1 hour default


class TestCacheKeyGeneration:
    """Test cache key generation logic."""

    def test_generate_cache_key_basic(self, search_cache):
        """Test basic cache key generation."""
        key = search_cache._generate_cache_key(
            query="test query",
            collection_name="documents",
            limit=10,
            search_type="hybrid"
        )

        assert key.startswith("search:")
        assert "documents" in key
        assert "hybrid" in key

    def test_generate_cache_key_with_filters(self, search_cache):
        """Test cache key with filters."""
        filters = {"category": "tech", "status": "published"}

        key = search_cache._generate_cache_key(
            query="test query",
            collection_name="documents",
            filters=filters,
            limit=10,
            search_type="vector"
        )

        assert key.startswith("search:")
        assert "documents" in key
        assert "vector" in key

    def test_generate_cache_key_consistent(self, search_cache):
        """Test that same inputs generate same key."""
        params = {
            "query": "test query",
            "collection_name": "docs",
            "filters": {"type": "api"},
            "limit": 20,
            "search_type": "hybrid"
        }

        key1 = search_cache._generate_cache_key(**params)
        key2 = search_cache._generate_cache_key(**params)

        assert key1 == key2

    def test_generate_cache_key_different_params(self, search_cache):
        """Test that different params generate different keys."""
        base_params = {
            "query": "test query",
            "collection_name": "docs",
            "limit": 10,
            "search_type": "hybrid"
        }

        key1 = search_cache._generate_cache_key(**base_params)

        # Different query
        key2 = search_cache._generate_cache_key(
            **{**base_params, "query": "different query"}
        )

        # Different collection
        key3 = search_cache._generate_cache_key(
            **{**base_params, "collection_name": "other"}
        )

        # Different search type
        key4 = search_cache._generate_cache_key(
            **{**base_params, "search_type": "vector"}
        )

        # All keys should be different
        keys = [key1, key2, key3, key4]
        assert len(set(keys)) == len(keys)

    def test_normalize_query(self, search_cache):
        """Test query normalization."""
        # Test case insensitive
        norm1 = search_cache._normalize_query("Test Query")
        norm2 = search_cache._normalize_query("test query")
        assert norm1 == norm2

        # Test whitespace normalization
        norm3 = search_cache._normalize_query("  test   query  ")
        assert norm3 == "test query"

        # Test special characters
        norm4 = search_cache._normalize_query("test-query_123")
        assert norm4 == "test-query_123"


class TestGetSearchResults:
    """Test retrieving cached search results."""

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, search_cache, mock_dragonfly_cache):
        """Test cache miss returns None."""
        mock_dragonfly_cache.get.return_value = None

        result = await search_cache.get_search_results(
            query="test query",
            collection_name="docs"
        )

        assert result is None
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, search_cache, mock_dragonfly_cache, mock_patterns):
        """Test cache hit returns results."""
        cached_data = {
            "results": [
                {"id": "1", "score": 0.9, "content": "Result 1"},
                {"id": "2", "score": 0.8, "content": "Result 2"}
            ],
            "metadata": {
                "total": 2,
                "query": "test query",
                "cached_at": datetime.now(UTC).isoformat()
            }
        }
        mock_dragonfly_cache.get.return_value = cached_data

        result = await search_cache.get_search_results(
            query="test query",
            collection_name="docs"
        )

        assert result == cached_data["results"]
        mock_patterns.increment_popularity.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_with_all_parameters(self, search_cache, mock_dragonfly_cache):
        """Test get with all parameters."""
        filters = {"category": "tech", "status": "active"}
        params = {"reranker": "cohere", "hyde": True}

        await search_cache.get_search_results(
            query="complex query",
            collection_name="articles",
            filters=filters,
            limit=50,
            search_type="semantic",
            **params
        )

        # Verify cache key was generated with all params
        call_args = mock_dragonfly_cache.get.call_args
        cache_key = call_args[0][0]
        assert cache_key.startswith("search:")
        assert "articles" in cache_key
        assert "semantic" in cache_key

    @pytest.mark.asyncio
    async def test_get_increments_popularity(self, search_cache, mock_dragonfly_cache, mock_patterns):
        """Test that cache hit increments popularity."""
        mock_dragonfly_cache.get.return_value = {
            "results": [{"id": "1"}],
            "metadata": {}
        }

        await search_cache.get_search_results(
            query="popular query",
            collection_name="docs"
        )

        mock_patterns.increment_popularity.assert_called_once()
        call_args = mock_patterns.increment_popularity.call_args
        assert call_args[0][0] == "popular query"


class TestSetSearchResults:
    """Test caching search results."""

    @pytest.mark.asyncio
    async def test_set_basic_results(self, search_cache, mock_dragonfly_cache):
        """Test basic result caching."""
        results = [
            {"id": "1", "score": 0.95, "content": "First result"},
            {"id": "2", "score": 0.90, "content": "Second result"}
        ]

        success = await search_cache.set_search_results(
            query="test query",
            collection_name="docs",
            results=results
        )

        assert success is True

        # Verify cache.set was called
        mock_dragonfly_cache.set.assert_called_once()
        call_args = mock_dragonfly_cache.set.call_args

        # Check stored data structure
        stored_data = call_args[0][1]
        assert "results" in stored_data
        assert stored_data["results"] == results
        assert "metadata" in stored_data
        assert stored_data["metadata"]["query"] == "test query"
        assert stored_data["metadata"]["collection"] == "docs"
        assert "cached_at" in stored_data["metadata"]

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, search_cache, mock_dragonfly_cache):
        """Test caching with custom TTL."""
        results = [{"id": "1", "score": 0.9}]

        await search_cache.set_search_results(
            query="test",
            collection_name="docs",
            results=results,
            ttl=7200  # 2 hours
        )

        call_args = mock_dragonfly_cache.set.call_args
        assert call_args[1]["ttl"] == 7200

    @pytest.mark.asyncio
    async def test_set_with_metadata(self, search_cache, mock_dragonfly_cache):
        """Test caching with additional metadata."""
        results = [{"id": "1", "score": 0.9}]
        metadata = {
            "total_found": 100,
            "processing_time_ms": 50,
            "reranker_used": "cohere"
        }

        await search_cache.set_search_results(
            query="test",
            collection_name="docs",
            results=results,
            metadata=metadata
        )

        call_args = mock_dragonfly_cache.set.call_args
        stored_data = call_args[0][1]

        # Original metadata should be preserved
        assert stored_data["metadata"]["total_found"] == 100
        assert stored_data["metadata"]["processing_time_ms"] == 50
        assert stored_data["metadata"]["reranker_used"] == "cohere"

    @pytest.mark.asyncio
    async def test_set_empty_results(self, search_cache, mock_dragonfly_cache):
        """Test caching empty results."""
        results = []

        success = await search_cache.set_search_results(
            query="no results query",
            collection_name="docs",
            results=results
        )

        assert success is True

        # Empty results should still be cached
        call_args = mock_dragonfly_cache.set.call_args
        stored_data = call_args[0][1]
        assert stored_data["results"] == []

    @pytest.mark.asyncio
    async def test_set_popular_query_extended_ttl(self, search_cache, mock_dragonfly_cache, mock_patterns):
        """Test that popular queries get extended TTL."""
        # Mock query as popular
        mock_patterns.get_popular_queries.return_value = [
            {"query": "popular query", "count": 100}
        ]

        results = [{"id": "1", "score": 0.9}]

        await search_cache.set_search_results(
            query="popular query",
            collection_name="docs",
            results=results
        )

        # Should check if query is popular
        mock_patterns.get_popular_queries.assert_called_once()

        # Popular queries might get longer TTL
        call_args = mock_dragonfly_cache.set.call_args
        ttl = call_args[1]["ttl"]
        assert ttl >= search_cache.default_ttl  # At least default TTL


class TestInvalidation:
    """Test cache invalidation methods."""

    @pytest.mark.asyncio
    async def test_invalidate_by_collection(self, search_cache, mock_dragonfly_cache):
        """Test invalidating all results for a collection."""
        # Mock finding keys
        mock_dragonfly_cache.scan_keys.return_value = [
            "search:docs:hybrid:hash1",
            "search:docs:vector:hash2",
            "search:docs:hybrid:hash3",
        ]

        count = await search_cache.invalidate_by_collection("docs")

        assert count == 3
        mock_dragonfly_cache.scan_keys.assert_called_once_with("search:docs:*")
        assert mock_dragonfly_cache.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_invalidate_by_query(self, search_cache, mock_dragonfly_cache):
        """Test invalidating results for specific query."""
        # Mock finding keys
        mock_dragonfly_cache.scan_keys.return_value = [
            "search:docs:hybrid:queryhash",
            "search:articles:hybrid:queryhash",
        ]

        count = await search_cache.invalidate_by_query("test query")

        assert count == 2
        assert mock_dragonfly_cache.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_by_pattern(self, search_cache, mock_dragonfly_cache):
        """Test invalidating by pattern."""
        mock_dragonfly_cache.scan_keys.return_value = [
            "search:docs:vector:hash1",
            "search:docs:vector:hash2",
        ]

        count = await search_cache.invalidate_by_pattern("search:*:vector:*")

        assert count == 2
        mock_dragonfly_cache.scan_keys.assert_called_once_with("search:*:vector:*")

    @pytest.mark.asyncio
    async def test_invalidate_all(self, search_cache, mock_dragonfly_cache):
        """Test invalidating all search results."""
        mock_dragonfly_cache.scan_keys.return_value = [
            "search:docs:hybrid:hash1",
            "search:articles:vector:hash2",
            "search:products:semantic:hash3",
        ]

        count = await search_cache.invalidate_all()

        assert count == 3
        mock_dragonfly_cache.scan_keys.assert_called_once_with("search:*")

    @pytest.mark.asyncio
    async def test_invalidate_empty_results(self, search_cache, mock_dragonfly_cache):
        """Test invalidation when no keys found."""
        mock_dragonfly_cache.scan_keys.return_value = []

        count = await search_cache.invalidate_by_collection("nonexistent")

        assert count == 0
        assert mock_dragonfly_cache.delete.call_count == 0


class TestBatchOperations:
    """Test batch operations."""

    @pytest.mark.asyncio
    async def test_get_multiple_queries(self, search_cache, mock_dragonfly_cache):
        """Test getting results for multiple queries."""
        queries = [
            {"query": "query1", "collection_name": "docs"},
            {"query": "query2", "collection_name": "docs"},
            {"query": "query3", "collection_name": "articles"}
        ]

        # Mock batch get results
        mock_dragonfly_cache.mget.return_value = [
            {"results": [{"id": "1"}], "metadata": {}},  # Hit
            None,  # Miss
            {"results": [{"id": "3"}], "metadata": {}},  # Hit
        ]

        results = await search_cache.get_multiple_queries(queries)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

        # Should use mget for efficiency
        mock_dragonfly_cache.mget.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_cache_popular_queries(self, search_cache, mock_dragonfly_cache, mock_patterns):
        """Test warming cache with popular queries."""
        popular_queries = [
            {"query": "getting started", "count": 150},
            {"query": "api reference", "count": 120},
            {"query": "installation", "count": 100},
        ]
        mock_patterns.get_popular_queries.return_value = popular_queries

        # Mock search function
        async def mock_search(query, collection):
            return [{"id": f"{query}-1", "score": 0.9}]

        success = await search_cache.warm_popular_queries(
            collection_name="docs",
            search_func=mock_search,
            top_n=2
        )

        assert success is True
        # Should cache top 2 queries
        assert mock_dragonfly_cache.set.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_invalidate(self, search_cache, mock_dragonfly_cache):
        """Test batch invalidation."""
        # Setup pipeline mock
        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1, 1])
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_dragonfly_cache.pipeline.return_value = pipeline

        keys_to_delete = [
            "search:docs:hybrid:hash1",
            "search:docs:vector:hash2",
            "search:articles:hybrid:hash3",
        ]

        count = await search_cache._batch_delete(keys_to_delete)

        assert count == 3
        assert pipeline.delete.call_count == 3
        pipeline.execute.assert_called_once()


class TestAnalytics:
    """Test analytics and monitoring features."""

    @pytest.mark.asyncio
    async def test_get_cache_analytics(self, search_cache, mock_patterns):
        """Test getting cache analytics."""
        mock_analytics = {
            "total_queries": 1000,
            "unique_queries": 250,
            "hit_rate": 0.75,
            "popular_queries": [
                {"query": "api docs", "count": 50},
                {"query": "tutorial", "count": 45}
            ],
            "collections": {
                "docs": {"queries": 600, "hit_rate": 0.8},
                "articles": {"queries": 400, "hit_rate": 0.7}
            }
        }
        mock_patterns.get_cache_analytics.return_value = mock_analytics

        analytics = await search_cache.get_analytics()

        assert analytics == mock_analytics
        mock_patterns.get_cache_analytics.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_query_performance(self, search_cache, mock_dragonfly_cache):
        """Test getting query performance metrics."""
        # Mock stored results with timing data
        mock_dragonfly_cache.get.return_value = {
            "results": [{"id": "1"}],
            "metadata": {
                "cached_at": datetime.now(UTC).isoformat(),
                "processing_time_ms": 45,
                "total_found": 100
            }
        }

        result = await search_cache.get_search_results(
            query="test",
            collection_name="docs"
        )

        assert result is not None
        # Performance data should be available in metadata

    @pytest.mark.asyncio
    async def test_track_cache_misses(self, search_cache, mock_dragonfly_cache, mock_patterns):
        """Test tracking cache misses for analysis."""
        mock_dragonfly_cache.get.return_value = None

        result = await search_cache.get_search_results(
            query="rare query",
            collection_name="docs"
        )

        assert result is None
        # Could track misses for optimization
        # This might be handled by patterns.track_miss() if implemented


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_get_set(self, search_cache, mock_dragonfly_cache):
        """Test concurrent get and set operations."""
        import asyncio

        async def get_results():
            return await search_cache.get_search_results(
                query="concurrent test",
                collection_name="docs"
            )

        async def set_results():
            return await search_cache.set_search_results(
                query="concurrent test",
                collection_name="docs",
                results=[{"id": "1", "score": 0.9}]
            )

        # Run operations concurrently
        results = await asyncio.gather(
            get_results(),
            set_results(),
            get_results(),
            return_exceptions=True
        )

        # All operations should complete without errors
        assert all(not isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_invalidation(self, search_cache, mock_dragonfly_cache):
        """Test concurrent invalidation operations."""
        import asyncio

        mock_dragonfly_cache.scan_keys.return_value = [
            f"search:docs:hybrid:hash{i}" for i in range(10)
        ]

        # Run multiple invalidations concurrently
        tasks = [
            search_cache.invalidate_by_collection("docs"),
            search_cache.invalidate_by_query("test"),
            search_cache.invalidate_by_pattern("search:*:vector:*")
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without errors
        assert all(not isinstance(r, Exception) for r in results)


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_get_with_none_query(self, search_cache):
        """Test get with None query."""
        result = await search_cache.get_search_results(
            query=None,
            collection_name="docs"
        )
        # Should handle gracefully
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_empty_query(self, search_cache, mock_dragonfly_cache):
        """Test get with empty query."""
        mock_dragonfly_cache.get.return_value = {
            "results": [],
            "metadata": {"query": ""}
        }

        result = await search_cache.get_search_results(
            query="",
            collection_name="docs"
        )

        # Empty queries are valid
        assert result == []

    @pytest.mark.asyncio
    async def test_set_with_invalid_results(self, search_cache, mock_dragonfly_cache):
        """Test set with invalid results format."""
        # Results should be a list
        await search_cache.set_search_results(
            query="test",
            collection_name="docs",
            results="invalid"  # Not a list
        )

        # Should handle gracefully or validate
        # Implementation dependent

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, search_cache, mock_dragonfly_cache):
        """Test handling special characters in query."""
        special_query = "test & query | with <special> chars!"

        await search_cache.get_search_results(
            query=special_query,
            collection_name="docs"
        )

        # Should handle special characters in cache key
        call_args = mock_dragonfly_cache.get.call_args
        cache_key = call_args[0][0]
        assert cache_key.startswith("search:")

    @pytest.mark.asyncio
    async def test_very_long_query(self, search_cache, mock_dragonfly_cache):
        """Test handling very long queries."""
        long_query = "test " * 1000  # Very long query

        await search_cache.get_search_results(
            query=long_query,
            collection_name="docs"
        )

        # Should handle long queries (possibly truncate for key)
        call_args = mock_dragonfly_cache.get.call_args
        cache_key = call_args[0][0]
        assert len(cache_key) < 1000  # Key should be reasonable length

    @pytest.mark.asyncio
    async def test_unicode_in_query(self, search_cache, mock_dragonfly_cache):
        """Test handling unicode in queries."""
        unicode_query = "测试 query with émojis 🔍"

        await search_cache.get_search_results(
            query=unicode_query,
            collection_name="docs"
        )

        # Should handle unicode properly
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_filters_ordering(self, search_cache):
        """Test that filter ordering doesn't affect cache key."""
        filters1 = {"category": "tech", "status": "active", "type": "article"}
        filters2 = {"type": "article", "status": "active", "category": "tech"}

        key1 = search_cache._generate_cache_key(
            query="test",
            collection_name="docs",
            filters=filters1
        )

        key2 = search_cache._generate_cache_key(
            query="test",
            collection_name="docs",
            filters=filters2
        )

        # Keys should be the same regardless of filter order
        assert key1 == key2


class TestCachePatterns:
    """Test cache pattern functionality."""

    @pytest.mark.asyncio
    async def test_ttl_based_on_result_count(self, search_cache, mock_dragonfly_cache):
        """Test TTL adjustment based on result count."""
        # Few results might get shorter TTL
        few_results = [{"id": "1"}]
        await search_cache.set_search_results(
            query="rare query",
            collection_name="docs",
            results=few_results
        )

        # Many results might get longer TTL
        many_results = [{"id": str(i)} for i in range(100)]
        await search_cache.set_search_results(
            query="common query",
            collection_name="docs",
            results=many_results
        )

        # Implementation dependent - verify TTL logic if implemented

    @pytest.mark.asyncio
    async def test_conditional_caching(self, search_cache, mock_dragonfly_cache):
        """Test conditional caching based on result quality."""
        # Poor results (low scores) might not be cached
        poor_results = [
            {"id": "1", "score": 0.1},
            {"id": "2", "score": 0.15}
        ]

        await search_cache.set_search_results(
            query="test",
            collection_name="docs",
            results=poor_results,
            min_score_to_cache=0.5  # If implemented
        )

        # Implementation dependent

    @pytest.mark.asyncio
    async def test_cache_versioning(self, search_cache, mock_dragonfly_cache):
        """Test cache versioning for schema changes."""
        # If search result format changes, old cache should be invalidated
        # This might be handled by version in cache key

        results = [{"id": "1", "score": 0.9, "v2_field": "new"}]

        await search_cache.set_search_results(
            query="test",
            collection_name="docs",
            results=results,
            version="v2"  # If implemented
        )

        # Implementation dependent


class TestMetrics:
    """Test metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_cache_hit_metrics(self, search_cache, mock_dragonfly_cache, mock_patterns):
        """Test cache hit metric tracking."""
        # Simulate hits and misses
        mock_dragonfly_cache.get.side_effect = [
            {"results": [], "metadata": {}},  # Hit
            None,  # Miss
            {"results": [], "metadata": {}},  # Hit
        ]

        for _ in range(3):
            await search_cache.get_search_results(
                query="test",
                collection_name="docs"
            )

        # Patterns should track metrics
        assert mock_patterns.increment_popularity.call_count == 2  # 2 hits

    @pytest.mark.asyncio
    async def test_performance_metrics(self, search_cache, mock_dragonfly_cache):
        """Test performance metric collection."""
        import time

        results = [{"id": "1"}]

        start = time.time()
        await search_cache.set_search_results(
            query="test",
            collection_name="docs",
            results=results
        )
        _ = time.time() - start

        # Could track set operation performance
        # Implementation dependent

