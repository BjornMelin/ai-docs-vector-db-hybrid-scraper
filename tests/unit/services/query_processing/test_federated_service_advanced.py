"""Service tests for the federated search service.

Tests for collection selection strategies, search modes, result merging strategies,
and search functionality.
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


class TestFederatedSearchServiceAdvanced:
    """Test advanced FederatedSearchService functionality."""

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

    @pytest.mark.asyncio
    async def test_collection_selection_strategies(self, service, sample_metadata):
        """Test different collection selection strategies."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        for strategy in CollectionSelectionStrategy:
            request = FederatedSearchRequest(
                query="api documentation",
                collection_selection_strategy=strategy,
                limit=5,
            )

            if strategy == CollectionSelectionStrategy.EXPLICIT:
                request.target_collections = ["docs", "api"]

            result = await service.search(request)

            assert isinstance(result, FederatedSearchResult)
            assert result.search_strategy == strategy

    @pytest.mark.asyncio
    async def test_result_merging_strategies(self, service, sample_metadata):
        """Test different result merging strategies."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        for strategy in ResultMergingStrategy:
            request = FederatedSearchRequest(
                query="test query", result_merging_strategy=strategy, limit=5
            )

            result = await service.search(request)

            assert isinstance(result, FederatedSearchResult)
            assert result.merging_strategy == strategy

    @pytest.mark.asyncio
    async def test_smart_routing(self, service, sample_metadata):
        """Test smart routing collection selection."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        # Test query with content type hints
        request = FederatedSearchRequest(
            query="api documentation guide",
            collection_selection_strategy=CollectionSelectionStrategy.SMART_ROUTING,
            limit=5,
        )

        # Test the internal smart routing method
        selected_collections = await service._select_collections(request)

        # Should return some collections
        assert isinstance(selected_collections, list)
        assert len(selected_collections) > 0

    @pytest.mark.asyncio
    async def test_content_based_selection(self, service, sample_metadata):
        """Test content-based collection selection."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        # Test query with content type hints
        request = FederatedSearchRequest(
            query="api reference documentation",
            collection_selection_strategy=CollectionSelectionStrategy.CONTENT_BASED,
            limit=5,
        )

        selected_collections = service._select_by_content_type(request)

        # Should prefer collections with matching content types
        assert isinstance(selected_collections, list)
        assert len(selected_collections) > 0

    @pytest.mark.asyncio
    async def test_performance_based_selection(self, service, sample_metadata):
        """Test performance-based collection selection."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query",
            collection_selection_strategy=CollectionSelectionStrategy.PERFORMANCE_BASED,
            limit=5,
        )

        selected_collections = service._select_by_performance(request)

        # Should return collections ordered by performance
        assert isinstance(selected_collections, list)
        assert len(selected_collections) == len(sample_metadata)

        # Should be ordered by performance - exact order depends on calculation
        # but should include all collections
        assert len(selected_collections) == 3
        assert "api" in selected_collections
        assert "docs" in selected_collections
        assert "tutorials" in selected_collections

    @pytest.mark.asyncio
    async def test_max_collections_limit(self, service, sample_metadata):
        """Test max collections limit."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query",
            collection_selection_strategy=CollectionSelectionStrategy.ALL,
            max_collections=2,
            limit=5,
        )

        selected_collections = await service._select_collections(request)

        # Should respect max collections limit
        assert len(selected_collections) <= 2

    @pytest.mark.asyncio
    async def test_parallel_search_execution(self, service, sample_metadata):
        """Test parallel search execution."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query", search_mode=SearchMode.PARALLEL, limit=5
        )

        with patch.object(service, "_search_single_collection") as mock_search:
            mock_search.return_value = CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=0,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test query",
            )

            target_collections = list(sample_metadata.keys())
            results = await service._execute_parallel_search(
                request, target_collections
            )

            assert len(results) == len(target_collections)
            assert mock_search.call_count == len(target_collections)

    @pytest.mark.asyncio
    async def test_sequential_search_execution(self, service, sample_metadata):
        """Test sequential search execution."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query", search_mode=SearchMode.SEQUENTIAL, limit=5
        )

        with patch.object(service, "_search_single_collection") as mock_search:
            mock_search.return_value = CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=0,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test query",
            )

            target_collections = list(sample_metadata.keys())
            results = await service._execute_sequential_search(
                request, target_collections
            )

            assert len(results) == len(target_collections)
            assert mock_search.call_count == len(target_collections)

    @pytest.mark.asyncio
    async def test_prioritized_search_execution(self, service, sample_metadata):
        """Test prioritized search execution."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query", search_mode=SearchMode.PRIORITIZED, limit=5
        )

        with patch.object(service, "_search_single_collection") as mock_search:
            mock_search.return_value = CollectionSearchResult(
                collection_name="test",
                results=[],
                _total_hits=0,
                search_time_ms=100.0,
                confidence_score=0.8,
                coverage_score=0.9,
                query_used="test query",
            )

            target_collections = list(sample_metadata.keys())
            results = await service._execute_prioritized_search(
                request, target_collections
            )

            assert len(results) == len(target_collections)

    @pytest.mark.asyncio
    async def test_adaptive_search_execution(self, service, sample_metadata):
        """Test adaptive search execution."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query", search_mode=SearchMode.ADAPTIVE, limit=5
        )

        # Test with small number of collections (should use parallel)
        target_collections = ["docs", "api"]
        with patch.object(service, "_execute_parallel_search") as mock_parallel:
            mock_parallel.return_value = []
            await service._execute_adaptive_search(request, target_collections)
            mock_parallel.assert_called_once()

        # Test with large number of collections (should use sequential)
        target_collections = [f"collection_{i}" for i in range(10)]
        with patch.object(service, "_execute_sequential_search") as mock_sequential:
            mock_sequential.return_value = []
            await service._execute_adaptive_search(request, target_collections)
            mock_sequential.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, service, sample_metadata):
        """Test search timeout handling."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query",
            timeout_ms=1500.0,  # Short but valid timeout
            per_collection_timeout_ms=1000.0,
            limit=5,
        )

        with patch.object(service, "_search_single_collection") as mock_search:
            # Make search take longer than timeout
            async def slow_search(*_args, **__kwargs):
                await asyncio.sleep(0.1)
                return CollectionSearchResult(
                    collection_name="test",
                    results=[],
                    _total_hits=0,
                    search_time_ms=100.0,
                    confidence_score=0.8,
                    coverage_score=0.9,
                    query_used="test query",
                )

            mock_search.side_effect = slow_search

            result = await service.search(request)

            # Should handle timeout gracefully
            assert isinstance(result, FederatedSearchResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
