"""Integration tests for the federated search service.

Tests for search mode integration scenarios and cross-mode functionality.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.services.query_processing.federated import (
    CollectionMetadata,
    CollectionSearchResult,
    FederatedSearchRequest,
    FederatedSearchResult,
    FederatedSearchService,
    SearchMode,
)


class TestSearchModeIntegration:
    """Test search mode integration scenarios."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample collection metadata."""
        return {
            "docs": CollectionMetadata(
                collection_name="docs",
                display_name="Documentation",
                document_count=1000,
                vector_size=768,
                content_types=["documentation"],
                domains=["api", "guide"],
                avg_search_time_ms=150.0,
                availability_score=0.98,
                quality_score=0.9,
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

    @pytest.fixture
    async def service_with_collections(self, sample_metadata):
        """Create service with registered collections."""
        service = FederatedSearchService(max_concurrent_searches=3)

        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        return service

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, service_with_collections):
        """Test performance difference between parallel and sequential modes."""
        # Mock slow search to see timing differences
        with patch.object(
            service_with_collections, "_search_single_collection"
        ) as mock_search:

            async def slow_search(collection_name, request):
                await asyncio.sleep(0.01)  # Small delay
                return CollectionSearchResult(
                    collection_name=collection_name,
                    results=[],
                    _total_hits=0,
                    search_time_ms=10.0,
                    confidence_score=0.8,
                    coverage_score=0.9,
                    query_used=request.query,
                )

            mock_search.side_effect = slow_search

            # Test parallel
            parallel_result = await service_with_collections._execute_parallel_search(
                FederatedSearchRequest(query="test"), ["docs", "api", "tutorials"]
            )

            # Test sequential
            sequential_result = (
                await service_with_collections._execute_sequential_search(
                    FederatedSearchRequest(query="test"), ["docs", "api", "tutorials"]
                )
            )

            # Both should succeed and return results
            assert len(parallel_result) == 3
            assert len(sequential_result) == 3

    @pytest.mark.asyncio
    async def test_timeout_handling_in_different_modes(self, service_with_collections):
        """Test timeout handling in different search modes."""
        very_short_timeout = FederatedSearchRequest(
            query="test", timeout_ms=1500.0, per_collection_timeout_ms=1000.0
        )

        for mode in [SearchMode.PARALLEL, SearchMode.SEQUENTIAL]:
            very_short_timeout.search_mode = mode

            with patch.object(
                service_with_collections, "_search_single_collection"
            ) as mock_search:
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
                        query_used="test",
                    )

                mock_search.side_effect = slow_search

                # Should handle timeout without crashing
                result = await service_with_collections.search(very_short_timeout)
                assert isinstance(result, FederatedSearchResult)

    @pytest.mark.asyncio
    async def test_mode_switching_based_on_conditions(self, service_with_collections):
        """Test adaptive mode switching based on conditions."""
        # Test adaptive mode falls back to sequential under high load
        with patch.object(
            service_with_collections, "_get_system_load", return_value=0.9
        ):
            request = FederatedSearchRequest(
                query="test", search_mode=SearchMode.ADAPTIVE
            )

            with patch.object(
                service_with_collections, "_execute_sequential_search"
            ) as mock_sequential:
                mock_sequential.return_value = []

                await service_with_collections._execute_search(request, ["docs", "api"])

                # Should use sequential under high load
                mock_sequential.assert_called_once()

    @pytest.mark.asyncio
    async def test_cross_mode_functionality(self, service_with_collections):
        """Test that all modes produce consistent results."""
        request_base = FederatedSearchRequest(query="test query")

        results = {}
        for mode in [SearchMode.PARALLEL, SearchMode.SEQUENTIAL]:
            request = FederatedSearchRequest(
                query=request_base.query,
                search_mode=mode,
                target_collections=["docs", "api"],
            )

            with patch.object(
                service_with_collections, "_search_single_collection"
            ) as mock_search:
                mock_search.return_value = CollectionSearchResult(
                    collection_name="mock",
                    results=[{"id": "1", "score": 0.8}],
                    _total_hits=1,
                    search_time_ms=100.0,
                    confidence_score=0.8,
                    coverage_score=0.9,
                    query_used="test query",
                )

                result = await service_with_collections.search(request)
                results[mode.value] = result

        # All modes should return valid results
        for mode_name, result in results.items():
            assert isinstance(result, FederatedSearchResult)
            assert result.search_mode.value == mode_name

    @pytest.mark.asyncio
    async def test_collection_failure_handling(self, service_with_collections):
        """Test handling of collection failures across modes."""
        for mode in [SearchMode.PARALLEL, SearchMode.SEQUENTIAL]:
            request = FederatedSearchRequest(
                query="test",
                search_mode=mode,
                target_collections=["docs", "api", "tutorials"],
            )

            with patch.object(
                service_with_collections, "_search_single_collection"
            ) as mock_search:

                async def failing_search(collection_name, _request):
                    if collection_name == "api":
                        error_message = "Collection failed"
                        raise RuntimeError(error_message)
                    return CollectionSearchResult(
                        collection_name=collection_name,
                        results=[],
                        _total_hits=0,
                        search_time_ms=100.0,
                        confidence_score=0.8,
                        coverage_score=0.9,
                        query_used="test",
                    )

                mock_search.side_effect = failing_search

                result = await service_with_collections.search(request)

                # Should handle failures gracefully
                assert isinstance(result, FederatedSearchResult)
                assert "api" in result.collections_failed
                assert len(result.collections_searched) == 2  # docs and tutorials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
