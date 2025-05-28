"""Tests for HNSW optimizer service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.config import UnifiedConfig
from src.services.hnsw_optimizer import HNSWOptimizer


@pytest.fixture
def mock_config():
    """Create mock unified config for testing."""
    config = MagicMock(spec=UnifiedConfig)

    # Set up nested mock attributes for search configuration
    config.search = MagicMock()
    config.search.hnsw = MagicMock()
    config.search.hnsw.enable_adaptive_ef = True
    config.search.hnsw.default_ef_construct = 200
    config.search.hnsw.default_m = 16

    # Set up collection-specific HNSW configs
    config.search.collection_hnsw_configs = MagicMock()
    config.search.collection_hnsw_configs.api_reference = MagicMock()
    config.search.collection_hnsw_configs.api_reference.m = 24
    config.search.collection_hnsw_configs.api_reference.ef_construct = 300

    config.search.collection_hnsw_configs.tutorials = MagicMock()
    config.search.collection_hnsw_configs.tutorials.m = 16
    config.search.collection_hnsw_configs.tutorials.ef_construct = 200

    return config


@pytest.fixture
def mock_qdrant_service():
    """Create mock QdrantService for testing."""
    service = MagicMock()
    service.search.return_value = AsyncMock()
    return service


@pytest.fixture
def optimizer(mock_config, mock_qdrant_service):
    """Create HNSWOptimizer instance for testing."""
    return HNSWOptimizer(mock_config, mock_qdrant_service)


class TestHNSWOptimizer:
    """Test cases for HNSWOptimizer."""

    def test_initialization(self, optimizer, mock_config, mock_qdrant_service):
        """Test optimizer initialization."""
        assert optimizer.config == mock_config
        assert optimizer.qdrant_service == mock_qdrant_service
        assert hasattr(optimizer, "performance_cache")
        assert hasattr(optimizer, "adaptive_ef_cache")

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_basic_functionality(self, optimizer):
        """Test basic adaptive ef retrieve functionality."""
        collection_name = "test_collection"
        query_vector = [0.1] * 768
        time_budget_ms = 100
        limit = 10

        # Mock search response
        mock_response = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]

        # Mock the qdrant client with async search
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.search.return_value = mock_response
        optimizer.qdrant_service._client = mock_client

        result = await optimizer.adaptive_ef_retrieve(
            collection_name, query_vector, time_budget_ms, limit
        )

        assert "results" in result
        assert "ef_used" in result
        assert "search_time_ms" in result

    @pytest.mark.asyncio
    async def test_get_collection_specific_hnsw_config(self, optimizer):
        """Test getting collection-specific HNSW configuration."""
        # Test API reference config
        api_config = optimizer.get_collection_specific_hnsw_config("api_reference")
        assert "m" in api_config
        assert "ef_construct" in api_config

        # Test tutorials config
        tutorial_config = optimizer.get_collection_specific_hnsw_config("tutorials")
        assert "m" in tutorial_config
        assert "ef_construct" in tutorial_config

    @pytest.mark.asyncio
    async def test_optimize_collection_hnsw(self, optimizer):
        """Test collection HNSW optimization."""
        collection_name = "test_collection"
        collection_type = "api_reference"

        # Mock the get_collection call with async
        from unittest.mock import AsyncMock

        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = 16
        mock_collection_info.config.params.vectors.dense.hnsw_config.ef_construct = 128

        mock_client = AsyncMock()
        mock_client.get_collection.return_value = mock_collection_info
        optimizer.qdrant_service._client = mock_client

        result = await optimizer.optimize_collection_hnsw(
            collection_name, collection_type
        )

        assert "current_config" in result
        assert "recommended_config" in result
        assert "needs_update" in result

    def test_get_performance_cache_stats(self, optimizer):
        """Test performance cache statistics."""
        # Add some cache entries
        optimizer.performance_cache["test_key_1"] = {"time": 100, "ef": 50}
        optimizer.performance_cache["test_key_2"] = {"time": 200, "ef": 100}

        stats = optimizer.get_performance_cache_stats()

        assert "adaptive_ef_cache_size" in stats
        assert "performance_cache_size" in stats
        assert stats["performance_cache_size"] >= 2

    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """Test optimizer cleanup."""
        # Add some cache entries
        optimizer.performance_cache["test_key"] = {"time": 100}
        optimizer.adaptive_ef_cache["test_key"] = {"ef": 50}

        await optimizer.cleanup()

        assert len(optimizer.performance_cache) == 0
        assert len(optimizer.adaptive_ef_cache) == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_adaptive_ef_retrieve(self, optimizer):
        """Test error handling in adaptive ef retrieve."""
        collection_name = "test_collection"
        query_vector = [0.1] * 768
        time_budget_ms = 100
        limit = 10

        # Mock search failure with async
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.search.side_effect = Exception("Search failed")
        optimizer.qdrant_service._client = mock_client

        # The method should handle the error gracefully and return results
        result = await optimizer.adaptive_ef_retrieve(
            collection_name, query_vector, time_budget_ms, limit
        )

        # Check that it returns a valid result even with search failures
        assert "results" in result
        assert "ef_used" in result


class TestHNSWOptimizerIntegration:
    """Integration tests for HNSW optimizer with QdrantService."""

    @pytest.fixture
    def mock_qdrant_service_with_hnsw(self):
        """Create mock QdrantService with HNSW methods."""
        service = MagicMock()

        # Mock search_with_adaptive_ef method
        async def mock_search_with_adaptive_ef(*args, **kwargs):
            return {
                "results": [{"id": "1", "score": 0.9}],
                "ef_used": 100,
                "time_ms": 85,
            }

        service.search_with_adaptive_ef = mock_search_with_adaptive_ef
        return service

    @pytest.mark.asyncio
    async def test_search_with_adaptive_ef_integration(self, mock_config):
        """Test integration with QdrantService search_with_adaptive_ef."""
        # This would be tested with actual QdrantService integration
        # For now, test the configuration flow

        optimizer = HNSWOptimizer(mock_config, MagicMock())

        # Test that optimizer can provide ef recommendations
        collection_name = "test_collection"
        query_vector = [0.1] * 768
        time_budget_ms = 100

        # Mock the qdrant service _client.search method
        mock_client = MagicMock()
        mock_client.search.return_value = [{"id": "1", "score": 0.9}]
        optimizer.qdrant_service._client = mock_client

        result = await optimizer.adaptive_ef_retrieve(
            collection_name, query_vector, time_budget_ms
        )

        assert "ef_used" in result
        assert result["ef_used"] >= 50
        assert result["ef_used"] <= 200


class TestHNSWConfigurationRecommendations:
    """Test HNSW configuration recommendations."""

    def test_collection_type_recommendations(self, optimizer):
        """Test configuration recommendations for different collection types."""
        # API reference collections should get higher parameters
        api_config = optimizer.config.search.collection_hnsw_configs.api_reference
        assert api_config.m >= 16
        assert api_config.ef_construct >= 200

        # Tutorial collections might use different parameters
        tutorial_config = optimizer.config.search.collection_hnsw_configs.tutorials
        assert tutorial_config.m >= 16
        assert tutorial_config.ef_construct >= 200

    @pytest.mark.asyncio
    async def test_get_collection_specific_config(self, optimizer):
        """Test getting collection-specific HNSW configuration."""
        collection_type = "api_reference"

        config = optimizer.get_collection_specific_hnsw_config(collection_type)

        assert "m" in config
        assert "ef_construct" in config
        assert "description" in config
        assert "ef_recommendations" in config

    def test_collection_specific_config_large_collections(self, optimizer):
        """Test configuration for different collection types."""
        # Test API reference - should have higher parameters
        api_config = optimizer.get_collection_specific_hnsw_config("api_reference")
        assert api_config["m"] >= 16
        assert api_config["ef_construct"] >= 200

        # Test blog posts - should have lower parameters for speed
        blog_config = optimizer.get_collection_specific_hnsw_config("blog_posts")
        assert blog_config["m"] <= 16
        assert blog_config["ef_construct"] <= 200

    def test_ef_recommendations_by_collection_type(self, optimizer):
        """Test EF recommendations for different collection types."""
        # Test that different collection types get appropriate EF recommendations
        api_config = optimizer.get_collection_specific_hnsw_config("api_reference")
        tutorial_config = optimizer.get_collection_specific_hnsw_config("tutorials")

        # API reference should have higher EF values for better accuracy
        assert (
            api_config["ef_recommendations"]["balanced_ef"]
            >= tutorial_config["ef_recommendations"]["balanced_ef"]
        )

        # Verify EF ranges are reasonable
        assert (
            api_config["ef_recommendations"]["min_ef"]
            <= api_config["ef_recommendations"]["balanced_ef"]
        )
        assert (
            api_config["ef_recommendations"]["balanced_ef"]
            <= api_config["ef_recommendations"]["max_ef"]
        )
