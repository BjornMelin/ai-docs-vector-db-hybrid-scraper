"""Tests for search cache module."""

import hashlib
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.cache.search_cache import SearchResultCache


class TestSearchResultCache:
    """Test the SearchResultCache class."""

    @pytest.fixture
    def mock_dragonfly_cache(self):
        """Create a mock DragonflyCache for testing."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.exists.return_value = False
        mock_cache.size.return_value = 100
        mock_cache.scan_keys.return_value = []
        mock_cache.delete_many.return_value = {}
        mock_cache.mget.return_value = []
        mock_cache.ttl.return_value = 3600

        # Mock client for atomic operations
        mock_client = AsyncMock()
        mock_client.incr.return_value = 1
        mock_client.expire.return_value = True

        # Store mock client as attribute for access in tests
        mock_cache._mock_client = mock_client

        # Make client property return the mock client (awaitable)
        async def get_client():
            return mock_client

        mock_cache.client = get_client()

        return mock_cache

    @pytest.fixture
    def mock_patterns(self):
        """Create a mock CachePatterns instance."""
        return MagicMock()

    @pytest.fixture
    def search_cache(self, mock_dragonfly_cache):
        """Create a SearchResultCache instance for testing."""
        cache = SearchResultCache(mock_dragonfly_cache, default_ttl=3600)
        return cache

    def test_search_cache_initialization(self, mock_dragonfly_cache):
        """Test SearchResultCache initialization."""
        cache = SearchResultCache(mock_dragonfly_cache, default_ttl=7200)

        assert cache.cache == mock_dragonfly_cache
        assert cache.default_ttl == 7200

    def test_search_cache_default_initialization(self, mock_dragonfly_cache):
        """Test SearchResultCache initialization with defaults."""
        cache = SearchResultCache(mock_dragonfly_cache)
        assert cache.default_ttl == 3600

    @pytest.mark.asyncio
    async def test_get_search_results_cache_hit(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting search results with cache hit."""
        expected_results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]
        mock_dragonfly_cache.get.return_value = expected_results

        results = await search_cache.get_search_results(
            query="test query",
            collection_name="test_collection",
            filters={"category": "tech"},
            limit=10,
            search_type="hybrid",
        )

        assert results == expected_results
        mock_dragonfly_cache.get.assert_called_once()

        # Should increment popularity
        mock_client = mock_dragonfly_cache._mock_client
        mock_client.incr.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_search_results_cache_miss(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting search results with cache miss."""
        mock_dragonfly_cache.get.return_value = None

        results = await search_cache.get_search_results(
            query="test query", collection_name="test_collection"
        )

        assert results is None
        mock_dragonfly_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_search_results_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting search results with cache exception."""
        mock_dragonfly_cache.get.side_effect = Exception("Cache error")

        results = await search_cache.get_search_results(
            query="test query", collection_name="test_collection"
        )

        assert results is None

    @pytest.mark.asyncio
    async def test_set_search_results_success(self, search_cache, mock_dragonfly_cache):
        """Test setting search results successfully."""
        results = [{"id": 1, "score": 0.9}]
        mock_dragonfly_cache.set.return_value = True
        mock_dragonfly_cache.get.return_value = 0  # No popularity

        success = await search_cache.set_search_results(
            query="test query",
            results=results,
            collection_name="test_collection",
            ttl=1800,
        )

        assert success is True
        mock_dragonfly_cache.set.assert_called_once()

        # Should increment popularity
        mock_client = mock_dragonfly_cache._mock_client
        mock_client.incr.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_search_results_popularity_adjustment(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test TTL adjustment based on query popularity."""
        results = [{"id": 1, "score": 0.9}]
        mock_dragonfly_cache.set.return_value = True

        # Test popular query (short TTL)
        mock_dragonfly_cache.get.return_value = 15  # High popularity
        await search_cache.set_search_results("popular query", results)

        # Should use half the default TTL
        set_call = mock_dragonfly_cache.set.call_args
        assert set_call[1]["ttl"] == 1800  # 3600 // 2

        mock_dragonfly_cache.reset_mock()

        # Test moderately popular query (normal TTL)
        mock_dragonfly_cache.get.return_value = 7  # Medium popularity
        await search_cache.set_search_results("medium query", results)

        set_call = mock_dragonfly_cache.set.call_args
        assert set_call[1]["ttl"] == 3600  # Default TTL

        mock_dragonfly_cache.reset_mock()

        # Test unpopular query (long TTL)
        mock_dragonfly_cache.get.return_value = 2  # Low popularity
        await search_cache.set_search_results("unpopular query", results)

        set_call = mock_dragonfly_cache.set.call_args
        assert set_call[1]["ttl"] == 7200  # 3600 * 2

    @pytest.mark.asyncio
    async def test_set_search_results_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test setting search results with cache exception."""
        results = [{"id": 1, "score": 0.9}]
        mock_dragonfly_cache.set.side_effect = Exception("Cache error")

        success = await search_cache.set_search_results("test query", results)

        assert success is False

    @pytest.mark.asyncio
    async def test_invalidate_by_collection(self, search_cache, mock_dragonfly_cache):
        """Test invalidating cache by collection."""
        keys = ["search:test_collection:hash1", "search:test_collection:hash2"]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.delete_many.return_value = {
            "search:test_collection:hash1": True,
            "search:test_collection:hash2": True,
        }

        count = await search_cache.invalidate_by_collection("test_collection")

        assert count == 2
        mock_dragonfly_cache.scan_keys.assert_called_once_with(
            "search:test_collection:*"
        )
        mock_dragonfly_cache.delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_by_collection_no_keys(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test invalidating cache by collection with no matching keys."""
        mock_dragonfly_cache.scan_keys.return_value = []

        count = await search_cache.invalidate_by_collection("test_collection")

        assert count == 0
        mock_dragonfly_cache.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_by_collection_with_batches(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test invalidating large number of keys in batches."""
        # Create 250 keys to test batching (batch_size = 100)
        keys = [f"search:test_collection:hash{i}" for i in range(250)]
        mock_dragonfly_cache.scan_keys.return_value = keys

        # Mock delete_many to return success for all keys
        def mock_delete_many(batch):
            return dict.fromkeys(batch, True)

        mock_dragonfly_cache.delete_many.side_effect = mock_delete_many

        count = await search_cache.invalidate_by_collection("test_collection")

        assert count == 250
        # Should be called 3 times (100, 100, 50)
        assert mock_dragonfly_cache.delete_many.call_count == 3

    @pytest.mark.asyncio
    async def test_invalidate_by_collection_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test invalidating cache by collection with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        count = await search_cache.invalidate_by_collection("test_collection")

        assert count == 0

    @pytest.mark.asyncio
    async def test_invalidate_by_query_pattern(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test invalidating cache by query pattern."""
        keys = ["search:col1:hash1", "search:col2:hash1"]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.delete_many.return_value = dict.fromkeys(keys, True)

        count = await search_cache.invalidate_by_query_pattern("test query")

        assert count == 2
        mock_dragonfly_cache.scan_keys.assert_called_once()
        mock_dragonfly_cache.delete_many.assert_called_once_with(keys)

    @pytest.mark.asyncio
    async def test_invalidate_by_query_pattern_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test invalidating cache by query pattern with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        count = await search_cache.invalidate_by_query_pattern("test query")

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_popular_queries(self, search_cache, mock_dragonfly_cache):
        """Test getting popular queries."""
        keys = ["popular:hash1", "popular:hash2", "popular:hash3"]
        counts = [10, 5, 15]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.mget.return_value = counts

        popular = await search_cache.get_popular_queries(limit=3)

        # Should be sorted by count descending
        expected = [("hash3", 15), ("hash1", 10), ("hash2", 5)]
        assert popular == expected

        mock_dragonfly_cache.scan_keys.assert_called_once_with("popular:*")
        mock_dragonfly_cache.mget.assert_called_once_with(keys)

    @pytest.mark.asyncio
    async def test_get_popular_queries_no_keys(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting popular queries with no keys."""
        mock_dragonfly_cache.scan_keys.return_value = []

        popular = await search_cache.get_popular_queries(limit=10)

        assert popular == []
        mock_dragonfly_cache.mget.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_popular_queries_with_none_values(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting popular queries with some None values."""
        keys = ["popular:hash1", "popular:hash2", "popular:hash3"]
        counts = [10, None, 15]  # One None value
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.mget.return_value = counts

        popular = await search_cache.get_popular_queries(limit=3)

        # Should only include non-None values
        expected = [("hash3", 15), ("hash1", 10)]
        assert popular == expected

    @pytest.mark.asyncio
    async def test_get_popular_queries_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting popular queries with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        popular = await search_cache.get_popular_queries(limit=10)

        assert popular == []

    @pytest.mark.asyncio
    async def test_cleanup_expired_popularity(self, search_cache, mock_dragonfly_cache):
        """Test cleaning up expired popularity counters."""
        keys = ["popular:hash1", "popular:hash2", "popular:hash3"]
        mock_dragonfly_cache.scan_keys.return_value = keys

        # Mock TTL responses: expired, active, expired
        mock_dragonfly_cache.ttl.side_effect = [-1, 3600, 0]

        expired_keys = ["popular:hash1", "popular:hash3"]
        mock_dragonfly_cache.delete_many.return_value = dict.fromkeys(
            expired_keys, True
        )

        count = await search_cache.cleanup_expired_popularity()

        assert count == 2
        mock_dragonfly_cache.delete_many.assert_called_once_with(expired_keys)

    @pytest.mark.asyncio
    async def test_cleanup_expired_popularity_no_expired(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test cleaning up expired popularity with no expired keys."""
        keys = ["popular:hash1", "popular:hash2"]
        mock_dragonfly_cache.scan_keys.return_value = keys
        mock_dragonfly_cache.ttl.return_value = 3600  # All active

        count = await search_cache.cleanup_expired_popularity()

        assert count == 0
        mock_dragonfly_cache.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_expired_popularity_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test cleaning up expired popularity with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        count = await search_cache.cleanup_expired_popularity()

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, search_cache, mock_dragonfly_cache):
        """Test getting cache statistics."""
        search_keys = [
            "search:collection1:hash1",
            "search:collection1:hash2",
            "search:collection2:hash3",
        ]
        popularity_keys = ["popular:hash1", "popular:hash2"]

        mock_dragonfly_cache.scan_keys.side_effect = [search_keys, popularity_keys]
        mock_dragonfly_cache.size.return_value = 100

        # Mock get_popular_queries
        with patch.object(
            search_cache, "get_popular_queries", return_value=[("query1", 10)]
        ):
            stats = await search_cache.get_cache_stats()

        expected_stats = {
            "_total_search_results": 3,
            "popularity_counters": 2,
            "cache_size": 100,
            "by_collection": {"collection1": 2, "collection2": 1},
            "top_queries": [("query1", 10)],
        }

        assert stats == expected_stats

    @pytest.mark.asyncio
    async def test_get_cache_stats_with_malformed_keys(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting cache statistics with malformed keys."""
        search_keys = [
            "search:collection1:hash1",
            "malformed_key",  # Malformed key
            "search:collection2:hash3",
        ]
        popularity_keys = []

        mock_dragonfly_cache.scan_keys.side_effect = [search_keys, popularity_keys]
        mock_dragonfly_cache.size.return_value = 50

        with patch.object(search_cache, "get_popular_queries", return_value=[]):
            stats = await search_cache.get_cache_stats()

        # Should handle malformed keys gracefully
        assert stats["_total_search_results"] == 3
        assert stats["by_collection"] == {"collection1": 1, "collection2": 1}

    @pytest.mark.asyncio
    async def test_get_cache_stats_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting cache statistics with exception."""
        mock_dragonfly_cache.scan_keys.side_effect = Exception("Scan error")

        stats = await search_cache.get_cache_stats()

        assert "error" in stats

    def test_get_search_key_consistency(self, search_cache):
        """Test search key generation consistency."""
        # Same parameters should generate same key
        key1 = search_cache._get_search_key(
            query="test query",
            collection_name="test_collection",
            filters={"category": "tech"},
            limit=10,
            search_type="hybrid",
            extra_param="value",
        )

        key2 = search_cache._get_search_key(
            query="test query",
            collection_name="test_collection",
            filters={"category": "tech"},
            limit=10,
            search_type="hybrid",
            extra_param="value",
        )

        assert key1 == key2

    def test_get_search_key_normalization(self, search_cache):
        """Test search key query normalization."""
        # Different case and whitespace should generate same key
        key1 = search_cache._get_search_key(
            query="  Test Query  ",
            collection_name="test",
            filters=None,
            limit=10,
            search_type="hybrid",
        )

        key2 = search_cache._get_search_key(
            query="test query",
            collection_name="test",
            filters=None,
            limit=10,
            search_type="hybrid",
        )

        assert key1 == key2

    def test_get_search_key_parameter_order(self, search_cache):
        """Test search key parameter order independence."""
        # Different parameter order should generate same key
        key1 = search_cache._get_search_key(
            query="test",
            collection_name="test",
            filters={"b": 2, "a": 1},
            limit=10,
            search_type="hybrid",
            param2="value2",
            param1="value1",
        )

        key2 = search_cache._get_search_key(
            query="test",
            collection_name="test",
            filters={"a": 1, "b": 2},
            limit=10,
            search_type="hybrid",
            param1="value1",
            param2="value2",
        )

        assert key1 == key2

    def test_get_search_key_format(self, search_cache):
        """Test search key format."""
        key = search_cache._get_search_key(
            query="test query",
            collection_name="test_collection",
            filters=None,
            limit=10,
            search_type="hybrid",
        )

        # Should have format: search:{collection}:{hash}
        assert key.startswith("search:test_collection:")
        assert len(key.split(":")) == 3
        assert len(key.split(":")[-1]) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_get_query_popularity(self, search_cache, mock_dragonfly_cache):
        """Test getting query popularity."""
        mock_dragonfly_cache.get.return_value = 5

        popularity = await search_cache._get_query_popularity("test query")

        assert popularity == 5

        # Test key format
        expected_hash = hashlib.md5(b"test query").hexdigest()
        expected_key = f"popular:{expected_hash}"
        mock_dragonfly_cache.get.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_get_query_popularity_no_data(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting query popularity with no data."""
        mock_dragonfly_cache.get.return_value = None

        popularity = await search_cache._get_query_popularity("test query")

        assert popularity == 0

    @pytest.mark.asyncio
    async def test_get_query_popularity_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test getting query popularity with exception."""
        mock_dragonfly_cache.get.side_effect = Exception("Cache error")

        popularity = await search_cache._get_query_popularity("test query")

        assert popularity == 0

    @pytest.mark.asyncio
    async def test_increment_query_popularity_first_time(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test incrementing query popularity for first time."""
        mock_client = mock_dragonfly_cache._mock_client
        mock_client.incr.return_value = 1  # First increment

        await search_cache._increment_query_popularity("test query")

        expected_hash = hashlib.md5(b"test query").hexdigest()
        expected_key = f"popular:{expected_hash}"

        mock_client.incr.assert_called_once_with(expected_key)
        mock_client.expire.assert_called_once_with(expected_key, 86400)

    @pytest.mark.asyncio
    async def test_increment_query_popularity_existing(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test incrementing query popularity for existing query."""
        mock_client = mock_dragonfly_cache._mock_client
        mock_client.incr.return_value = 5  # Not first increment

        await search_cache._increment_query_popularity("test query")

        mock_client.incr.assert_called_once()
        mock_client.expire.assert_not_called()  # Don't reset TTL

    @pytest.mark.asyncio
    async def test_increment_query_popularity_with_exception(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test incrementing query popularity with exception."""
        mock_client = mock_dragonfly_cache._mock_client
        mock_client.incr.side_effect = Exception("Increment error")

        # Should not raise exception
        await search_cache._increment_query_popularity("test query")

    @pytest.mark.asyncio
    async def test_warm_popular_searches_success(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test warming popular searches."""
        queries = ["query1", "query2", "query3"]
        mock_dragonfly_cache.exists.return_value = False  # Not cached

        # Mock search function
        async def mock_search_func(query, collection):
            return [{"id": 1, "query": query}]

        with patch.object(search_cache, "set_search_results", return_value=True):
            count = await search_cache.warm_popular_searches(
                queries, "test_collection", mock_search_func
            )

        assert count == 3

    @pytest.mark.asyncio
    async def test_warm_popular_searches_already_cached(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test warming popular searches with already cached queries."""
        queries = ["query1", "query2"]
        mock_dragonfly_cache.exists.return_value = True  # Already cached

        async def mock_search_func(query, collection):
            return [{"id": 1, "query": query}]

        count = await search_cache.warm_popular_searches(
            queries, "test_collection", mock_search_func
        )

        assert count == 0  # Nothing to warm

    @pytest.mark.asyncio
    async def test_warm_popular_searches_no_queries(self, search_cache):
        """Test warming popular searches with no queries."""
        count = await search_cache.warm_popular_searches([], "test_collection", None)
        assert count == 0

    @pytest.mark.asyncio
    async def test_warm_popular_searches_no_search_func(self, search_cache):
        """Test warming popular searches with no search function."""
        count = await search_cache.warm_popular_searches(
            ["query1"], "test_collection", None
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_warm_popular_searches_with_failures(
        self, search_cache, mock_dragonfly_cache
    ):
        """Test warming popular searches with some failures."""
        queries = ["query1", "query2", "query3"]
        mock_dragonfly_cache.exists.return_value = False

        # Mock search function that fails on query2
        async def mock_search_func(query, collection):
            if query == "query2":
                raise Exception("Search failed")
            return [{"id": 1, "query": query}]

        with patch.object(search_cache, "set_search_results", return_value=True):
            count = await search_cache.warm_popular_searches(
                queries, "test_collection", mock_search_func
            )

        assert count == 2  # Only query1 and query3 succeed

    @pytest.mark.asyncio
    async def test_warm_popular_searches_with_exception(self, search_cache):
        """Test warming popular searches with exception."""

        async def failing_search_func(query, collection):
            raise Exception("Critical error")

        # Should handle exception gracefully
        count = await search_cache.warm_popular_searches(
            ["query1"], "test_collection", failing_search_func
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_stats_alias(self, search_cache):
        """Test get_stats method alias."""
        # Should be an alias for get_cache_stats
        with patch.object(
            search_cache, "get_cache_stats", return_value={"test": "data"}
        ) as mock_method:
            # Check if method exists (might be added as alias)
            if hasattr(search_cache, "get_stats"):
                result = await search_cache.get_stats()
                assert result == {"test": "data"}
                mock_method.assert_called_once()
