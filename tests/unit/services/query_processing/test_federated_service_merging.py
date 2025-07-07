"""Result merging tests for the federated search service.

Tests for result merging strategies, deduplication, quality metrics, and caching functionality.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.services.query_processing.federated import (
    CollectionMetadata,
    CollectionSearchResult,
    CollectionSelectionStrategy,
    FederatedSearchRequest,
    FederatedSearchResult,
    FederatedSearchService,
    ResultMergingStrategy,
    SearchMode,
)


class TestFederatedSearchServiceMerging:
    """Test result merging functionality of FederatedSearchService."""

    @pytest.fixture
    def service(self):
        """Create FederatedSearchService instance for testing."""
        return FederatedSearchService()

    @pytest.fixture
    def sample_metadata(self):
        """Sample collection metadata for testing."""
        return {
            "docs": CollectionMetadata(
                collection_name="docs",
                display_name="Documentation",
                document_count=1000,
                vector_size=768,
                content_types=["documentation", "guide"],
                domains=["programming", "tutorial"],
                avg_search_time_ms=150.0,
                availability_score=0.98,
                quality_score=0.90,
                priority=8,
            ),
            "api": CollectionMetadata(
                collection_name="api",
                display_name="API Reference",
                document_count=500,
                vector_size=768,
                content_types=["api"],
                domains=["reference"],
                avg_search_time_ms=100.0,
                availability_score=0.95,
                quality_score=0.85,
                priority=6,
            ),
            "tutorials": CollectionMetadata(
                collection_name="tutorials",
                display_name="Tutorials",
                document_count=200,
                vector_size=768,
                content_types=["tutorial"],
                domains=["learning"],
                avg_search_time_ms=200.0,
                availability_score=0.90,
                quality_score=0.75,
                priority=4,
            ),
        }

    def test_result_merging_score_based(self, service):
        """Test score-based result merging."""
        collection_results = [
            CollectionSearchResult(
                collection_name="docs",
                results=[
                    {"id": "1", "score": 0.9, "payload": {"content": "high score"}},
                    {"id": "2", "score": 0.7, "payload": {"content": "medium score"}},
                ],
                _total_hits=2,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="api",
                results=[
                    {"id": "3", "score": 0.95, "payload": {"content": "highest score"}},
                    {"id": "4", "score": 0.6, "payload": {"content": "low score"}},
                ],
                _total_hits=2,
                search_time_ms=120.0,
                confidence_score=0.85,
                coverage_score=0.8,
                query_used="test",
            ),
        ]

        request = FederatedSearchRequest(
            query="test",
            result_merging_strategy=ResultMergingStrategy.SCORE_BASED,
            enable_deduplication=False,
        )

        merged = service._merge_by_score(collection_results, request)

        # Should be sorted by score (descending)
        assert len(merged) == 4
        assert merged[0]["id"] == "3"  # Highest score (0.95)
        assert merged[1]["id"] == "1"  # Second highest (0.9)
        assert merged[2]["id"] == "2"  # Third highest (0.7)
        assert merged[3]["id"] == "4"  # Lowest score (0.6)

        # Should have collection metadata
        assert all("_collection" in item for item in merged)
        assert all("_collection_confidence" in item for item in merged)

    def test_result_merging_round_robin(self, service):
        """Test round-robin result merging."""
        collection_results = [
            CollectionSearchResult(
                collection_name="docs",
                results=[
                    {"id": "docs_1", "score": 0.9},
                    {"id": "docs_2", "score": 0.8},
                ],
                _total_hits=2,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="api",
                results=[
                    {"id": "api_1", "score": 0.85},
                    {"id": "api_2", "score": 0.75},
                ],
                _total_hits=2,
                search_time_ms=120.0,
                confidence_score=0.85,
                coverage_score=0.8,
                query_used="test",
            ),
        ]

        request = FederatedSearchRequest(
            query="test",
            result_merging_strategy=ResultMergingStrategy.ROUND_ROBIN,
            enable_deduplication=False,
        )

        merged = service._merge_round_robin(collection_results, request)

        # Should alternate between collections
        assert len(merged) == 4
        assert merged[0]["id"] == "docs_1"
        assert merged[1]["id"] == "api_1"
        assert merged[2]["id"] == "docs_2"
        assert merged[3]["id"] == "api_2"

    @pytest.mark.asyncio
    async def test_result_merging_by_priority(self, service, sample_metadata):
        """Test priority-based result merging."""
        # Register collections to set up priority mapping
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        collection_results = [
            CollectionSearchResult(
                collection_name="tutorials",  # Priority 4
                results=[{"id": "tut_1", "score": 0.8}],
                _total_hits=1,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="docs",  # Priority 8
                results=[{"id": "docs_1", "score": 0.7}],
                _total_hits=1,
                search_time_ms=120.0,
                confidence_score=0.85,
                coverage_score=0.8,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="api",  # Priority 6
                results=[{"id": "api_1", "score": 0.9}],
                _total_hits=1,
                search_time_ms=90.0,
                confidence_score=0.9,
                coverage_score=0.95,
                query_used="test",
            ),
        ]

        request = FederatedSearchRequest(
            query="test",
            result_merging_strategy=ResultMergingStrategy.COLLECTION_PRIORITY,
            enable_deduplication=False,
        )

        merged = service._merge_by_priority(collection_results, request)

        # Should be ordered by collection priority
        assert len(merged) == 3
        assert merged[0]["id"] == "docs_1"  # Highest priority (8)
        assert merged[1]["id"] == "api_1"  # Medium priority (6)
        assert merged[2]["id"] == "tut_1"  # Lowest priority (4)

        # Should have priority metadata
        assert all("_collection_priority" in item for item in merged)

    def test_result_merging_temporal(self, service):
        """Test temporal result merging."""
        test_time_1 = "2024-01-01T10:00:00"
        test_time_2 = "2024-01-01T11:00:00"
        test_time_3 = "2024-01-01T09:00:00"

        collection_results = [
            CollectionSearchResult(
                collection_name="docs",
                results=[
                    {"id": "1", "score": 0.9, "payload": {"timestamp": test_time_1}},
                    {"id": "2", "score": 0.8, "payload": {"timestamp": test_time_3}},
                ],
                _total_hits=2,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="api",
                results=[
                    {"id": "3", "score": 0.85, "payload": {"timestamp": test_time_2}}
                ],
                _total_hits=1,
                search_time_ms=120.0,
                confidence_score=0.85,
                coverage_score=0.8,
                query_used="test",
            ),
        ]

        request = FederatedSearchRequest(
            query="test",
            result_merging_strategy=ResultMergingStrategy.TEMPORAL,
            enable_deduplication=False,
        )

        merged = service._merge_by_time(collection_results, request)

        # Should be sorted by timestamp (most recent first)
        assert len(merged) == 3
        assert merged[0]["id"] == "3"  # 11:00:00 (most recent)
        assert merged[1]["id"] == "1"  # 10:00:00
        assert merged[2]["id"] == "2"  # 09:00:00 (oldest)

    @pytest.mark.asyncio
    async def test_result_merging_diversity_optimized(self, service):
        """Test diversity-optimized result merging."""
        collection_results = [
            CollectionSearchResult(
                collection_name="docs",
                results=[
                    {"id": "docs_1", "score": 0.9},
                    {"id": "docs_2", "score": 0.8},
                    {"id": "docs_3", "score": 0.7},
                ],
                _total_hits=3,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="api",
                results=[
                    {"id": "api_1", "score": 0.85},
                    {"id": "api_2", "score": 0.75},
                ],
                _total_hits=2,
                search_time_ms=120.0,
                confidence_score=0.85,
                coverage_score=0.8,
                query_used="test",
            ),
        ]

        request = FederatedSearchRequest(
            query="test",
            result_merging_strategy=ResultMergingStrategy.DIVERSITY_OPTIMIZED,
            enable_deduplication=False,
            limit=10,
        )

        merged = service._merge_for_diversity(collection_results, request)

        # Should prioritize diversity (alternating collections)
        assert len(merged) == 5

        # First few results should alternate between collections
        collections_seen = [item["_collection"] for item in merged[:4]]
        assert (
            len(set(collections_seen)) == 2
        )  # Should have both collections represented

    @pytest.mark.asyncio
    async def test_deduplication(self, service):
        """Test result deduplication."""
        results = [
            {
                "id": "1",
                "score": 0.9,
                "payload": {"content": "machine learning algorithms"},
            },
            {
                "id": "2",
                "score": 0.8,
                "payload": {"content": "machine learning algorithms"},
            },  # Duplicate
            {"id": "3", "score": 0.7, "payload": {"content": "deep neural networks"}},
            {
                "id": "4",
                "score": 0.6,
                "payload": {"content": "neural network architectures"},
            },  # Similar
        ]

        # High threshold (strict deduplication)
        deduplicated = service._deduplicate_results(results, threshold=0.8)

        # Should remove strict duplicates
        assert len(deduplicated) < len(results)

        # Low threshold (no deduplication)
        deduplicated = service._deduplicate_results(results, threshold=1.0)

        # Should keep all results
        assert len(deduplicated) == len(results)

    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, service):
        """Test quality metrics calculation."""
        collection_results = [
            CollectionSearchResult(
                collection_name="docs",
                results=[{"id": "1"}, {"id": "2"}],
                _total_hits=2,
                search_time_ms=100.0,
                confidence_score=0.9,
                coverage_score=0.8,
                query_used="test",
            ),
            CollectionSearchResult(
                collection_name="api",
                results=[{"id": "3"}],
                _total_hits=1,
                search_time_ms=120.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test",
            ),
        ]

        merged_results = [
            {"id": "1", "_collection": "docs"},
            {"id": "2", "_collection": "docs"},
            {"id": "3", "_collection": "api"},
        ]

        request = FederatedSearchRequest(query="test")

        # Register some collections for coverage calculation
        await service.register_collection(
            "docs",
            CollectionMetadata(
                collection_name="docs", document_count=100, vector_size=768
            ),
        )
        await service.register_collection(
            "api",
            CollectionMetadata(
                collection_name="api", document_count=50, vector_size=768
            ),
        )
        await service.register_collection(
            "tutorials",
            CollectionMetadata(
                collection_name="tutorials", document_count=25, vector_size=768
            ),
        )

        metrics = service._calculate_quality_metrics(
            collection_results, merged_results, request
        )

        assert "overall_confidence" in metrics
        assert "coverage_score" in metrics
        assert "diversity_score" in metrics
        assert "deduplication_stats" in metrics

        assert 0.0 <= metrics["overall_confidence"] <= 1.0
        assert 0.0 <= metrics["coverage_score"] <= 1.0
        assert 0.0 <= metrics["diversity_score"] <= 1.0

    def test_quality_metrics_empty_results(self, service):
        """Test quality metrics with empty results."""
        metrics = service._calculate_quality_metrics(
            [], [], FederatedSearchRequest(query="test")
        )

        assert metrics["overall_confidence"] == 0.0
        assert metrics["coverage_score"] == 0.0
        assert metrics["diversity_score"] == 0.0

    def test_performance_stats_tracking(self, service):
        """Test performance statistics tracking."""
        initial_stats = service.get_performance_stats()
        assert initial_stats["_total_searches"] == 0

        # Simulate collection performance update
        service._update_collection_performance("test_collection", 150.0, True)

        stats = service.collection_performance_stats["test_collection"]
        assert stats["_total_searches"] == 1
        assert stats["avg_response_time"] == 150.0
        assert stats["success_rate"] == 1.0

        # Update with failure
        service._update_collection_performance("test_collection", 200.0, False)

        stats = service.collection_performance_stats["test_collection"]
        assert stats["_total_searches"] == 2
        assert stats["avg_response_time"] == 175.0  # (150 + 200) / 2
        assert stats["success_rate"] == 0.5  # 1 success out of 2 attempts

    def test_load_balancing_stats(self, service):
        """Test load balancing statistics."""
        target_collections = ["docs", "api", "tutorials"]

        # Set some load scores
        service.collection_load_scores = {"docs": 0.3, "api": 0.1, "tutorials": 0.7}

        stats = service._get_load_balancing_stats(target_collections)

        assert stats["target_collections"] == target_collections
        assert stats["load_scores"]["docs"] == 0.3
        assert stats["load_scores"]["api"] == 0.1
        assert stats["load_scores"]["tutorials"] == 0.7
        assert stats["load_balancing_enabled"] is True

    def test_caching(self, service):
        """Test result caching."""
        request = FederatedSearchRequest(query="test query", limit=10)

        # Initially no cache
        cached = service._get_cached_result(request)
        assert cached is None
        assert service.cache_stats["misses"] == 1
        assert service.cache_stats["hits"] == 0

        # Create and cache a result
        result = FederatedSearchResult(
            results=[],
            _total_results=0,
            collection_results=[],
            collections_searched=[],
            search_strategy=CollectionSelectionStrategy.ALL,
            merging_strategy=ResultMergingStrategy.SCORE_BASED,
            search_mode=SearchMode.PARALLEL,
            _total_search_time_ms=100.0,
            fastest_collection_ms=100.0,
            slowest_collection_ms=100.0,
            overall_confidence=0.8,
            coverage_score=0.9,
            diversity_score=0.5,
        )

        service._cache_result(request, result)

        # Should now find cached result
        cached = service._get_cached_result(request)
        assert cached is not None
        assert cached == result
        assert service.cache_stats["hits"] == 1

    def test_cache_key_generation(self, service):
        """Test cache key generation."""
        request1 = FederatedSearchRequest(
            query="test query", limit=10, target_collections=["docs", "api"]
        )

        request2 = FederatedSearchRequest(
            query="test query",
            limit=10,
            target_collections=["api", "docs"],  # Different order
        )

        request3 = FederatedSearchRequest(
            query="different query", limit=10, target_collections=["docs", "api"]
        )

        key1 = service._generate_cache_key(request1)
        key2 = service._generate_cache_key(request2)
        key3 = service._generate_cache_key(request3)

        # Same collections in different order should generate same key
        assert key1 == key2

        # Different query should generate different key
        assert key1 != key3

    def test_cache_size_limit(self, service):
        """Test cache size limit enforcement."""
        service.cache_size = 2  # Small cache for testing

        # Fill cache to capacity
        for i in range(3):
            request = FederatedSearchRequest(query=f"query {i}")
            result = FederatedSearchResult(
                results=[],
                _total_results=0,
                collection_results=[],
                collections_searched=[],
                search_strategy=CollectionSelectionStrategy.ALL,
                merging_strategy=ResultMergingStrategy.SCORE_BASED,
                search_mode=SearchMode.PARALLEL,
                _total_search_time_ms=100.0,
                fastest_collection_ms=100.0,
                slowest_collection_ms=100.0,
                overall_confidence=0.8,
                coverage_score=0.9,
                diversity_score=0.5,
            )
            service._cache_result(request, result)

        # Cache should not exceed size limit
        assert len(service.federated_cache) <= service.cache_size

    def test_clear_cache(self, service):
        """Test cache clearing."""
        # Add something to cache
        request = FederatedSearchRequest(query="test")
        result = FederatedSearchResult(
            results=[],
            _total_results=0,
            collection_results=[],
            collections_searched=[],
            search_strategy=CollectionSelectionStrategy.ALL,
            merging_strategy=ResultMergingStrategy.SCORE_BASED,
            search_mode=SearchMode.PARALLEL,
            _total_search_time_ms=100.0,
            fastest_collection_ms=100.0,
            slowest_collection_ms=100.0,
            overall_confidence=0.8,
            coverage_score=0.9,
            diversity_score=0.5,
        )

        # First cause a cache miss
        service._get_cached_result(request)

        # Now cache the result
        service._cache_result(request, result)

        assert len(service.federated_cache) > 0
        assert service.cache_stats["misses"] > 0

        # Clear cache
        service.clear_cache()

        assert len(service.federated_cache) == 0
        assert service.cache_stats == {"hits": 0, "misses": 0}

    @pytest.mark.asyncio
    async def test_get_collection_registry(self, service, sample_metadata):
        """Test getting collection registry."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        registry = service.get_collection_registry()

        # Should return copy of registry
        assert registry == service.collection_registry
        assert registry is not service.collection_registry  # Should be a copy

        # Modifying returned registry should not affect original
        registry["new_collection"] = None
        assert "new_collection" not in service.collection_registry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
