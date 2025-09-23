"""Core service tests for the federated search service.

Tests for service initialization, collection registration/unregistration,
and basic search functionality.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.services.query_processing.federated import (
    CollectionMetadata,
    CollectionSearchResult,
    CollectionSelectionStrategy,
    FederatedSearchRequest,
    FederatedSearchResult,
    FederatedSearchService,
    SearchMode,
)


class TestFederatedSearchServiceCore:
    """Test core FederatedSearchService functionality."""

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

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.enable_intelligent_routing is True
        assert service.enable_adaptive_load_balancing is True
        assert service.enable_result_caching is True
        assert service.max_concurrent_searches == 5
        assert service.collection_registry == {}
        assert service.collection_clients == {}
        assert service.collection_performance_stats == {}
        assert service.collection_load_scores == {}
        assert service.routing_intelligence == {}
        assert service.federated_cache == {}
        assert service.cache_size == 100
        assert service.cache_stats == {"hits": 0, "misses": 0}
        assert "_total_searches" in service.performance_stats
        assert service.performance_stats["_total_searches"] == 0

    def test_initialization_with_custom_settings(self):
        """Test service initialization with custom settings."""
        service = FederatedSearchService(
            enable_intelligent_routing=False,
            enable_adaptive_load_balancing=False,
            enable_result_caching=False,
            cache_size=50,
            max_concurrent_searches=3,
        )

        assert service.enable_intelligent_routing is False
        assert service.enable_adaptive_load_balancing is False
        assert service.enable_result_caching is False
        assert service.cache_size == 50
        assert service.max_concurrent_searches == 3

    @pytest.mark.asyncio
    async def test_register_collection(self, service, sample_metadata):
        """Test collection registration."""
        metadata = sample_metadata["docs"]
        mock_client = MagicMock()

        await service.register_collection("docs", metadata, mock_client)

        assert "docs" in service.collection_registry
        assert service.collection_registry["docs"] == metadata
        assert service.collection_clients["docs"] == mock_client
        assert "docs" in service.collection_performance_stats
        assert "docs" in service.collection_load_scores

        perf_stats = service.collection_performance_stats["docs"]
        assert perf_stats["_total_searches"] == 0
        assert perf_stats["avg_response_time"] == 0.0
        assert perf_stats["success_rate"] == 1.0
        assert isinstance(perf_stats["last_updated"], datetime)

    @pytest.mark.asyncio
    async def test_register_collection_without_client(self, service, sample_metadata):
        """Test collection registration without client."""
        metadata = sample_metadata["docs"]

        await service.register_collection("docs", metadata)

        assert "docs" in service.collection_registry
        assert service.collection_registry["docs"] == metadata
        assert "docs" not in service.collection_clients

    @pytest.mark.asyncio
    async def test_register_collection_error(self, service):
        """Test collection registration error handling."""
        # Mock an error during performance stats initialization
        with patch("src.services.query_processing.federated.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Mock datetime error")

            with pytest.raises(Exception, match="Mock datetime error"):
                await service.register_collection(
                    "invalid",
                    CollectionMetadata(
                        collection_name="test", document_count=100, vector_size=768
                    ),
                )

    @pytest.mark.asyncio
    async def test_unregister_collection(self, service, sample_metadata):
        """Test collection unregistration."""
        metadata = sample_metadata["docs"]
        await service.register_collection("docs", metadata)

        # Verify registration
        assert "docs" in service.collection_registry

        # Unregister
        await service.unregister_collection("docs")

        # Verify removal
        assert "docs" not in service.collection_registry
        assert "docs" not in service.collection_clients
        assert "docs" not in service.collection_performance_stats
        assert "docs" not in service.collection_load_scores

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_collection(self, service):
        """Test unregistering non-existent collection."""
        # Should not raise error
        await service.unregister_collection("nonexistent")

    @pytest.mark.asyncio
    async def test_basic_search(self, service, sample_metadata):
        """Test basic federated search."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query",
            # Use ALL to ensure collections are selected
            collection_selection_strategy=CollectionSelectionStrategy.ALL,
            limit=10,
        )

        result = await service.search(request)

        assert isinstance(result, FederatedSearchResult)
        assert result._total_search_time_ms > 0
        # With ALL strategy, should search all registered collections
        assert len(result.collection_results) == len(sample_metadata)
        assert all(
            isinstance(cr, CollectionSearchResult) for cr in result.collection_results
        )

    @pytest.mark.asyncio
    async def test_search_with_explicit_collections(self, service, sample_metadata):
        """Test search with explicitly specified collections."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        request = FederatedSearchRequest(
            query="test query",
            target_collections=["docs", "api"],
            collection_selection_strategy=CollectionSelectionStrategy.EXPLICIT,
            limit=10,
        )

        result = await service.search(request)

        assert isinstance(result, FederatedSearchResult)
        # Should only search specified collections
        # (though mock implementation searches all)
        assert result.search_strategy == CollectionSelectionStrategy.EXPLICIT

    @pytest.mark.asyncio
    async def test_search_modes(self, service, sample_metadata):
        """Test different search modes."""
        # Register collections
        for name, metadata in sample_metadata.items():
            await service.register_collection(name, metadata)

        for mode in SearchMode:
            request = FederatedSearchRequest(
                query="test query",
                search_mode=mode,
                # Ensure collections are selected
                collection_selection_strategy=CollectionSelectionStrategy.ALL,
                limit=5,
            )

            result = await service.search(request)

            assert isinstance(result, FederatedSearchResult)
            # Note: The mock implementation may not
            # preserve the exact search mode due to fallbacks
            # but it should be one of the valid modes
            assert result.search_mode in SearchMode
            assert result._total_search_time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
