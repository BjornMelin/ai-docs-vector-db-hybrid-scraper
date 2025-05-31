"""Simple tests for HNSW optimizer to increase coverage."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.config.models import HNSWConfig
from src.config.models import QdrantConfig
from src.config.models import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.utilities.hnsw_optimizer import HNSWOptimizer


@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig(
        qdrant=QdrantConfig(
            url="http://localhost:6333",
            api_key="test-key",
        ),
        hnsw=HNSWConfig(
            m=16,
            ef_construct=200,
            ef=100,
        ),
    )


@pytest.fixture
def mock_qdrant_service():
    """Create mock QdrantService."""
    service = MagicMock()
    service._initialized = True
    service._client = MagicMock()
    return service


@pytest.fixture
def optimizer(config, mock_qdrant_service):
    """Create HNSW optimizer."""
    return HNSWOptimizer(config, mock_qdrant_service)


class TestHNSWOptimizer:
    """Tests for HNSW optimizer."""

    def test_initialization(self, config, mock_qdrant_service):
        """Test optimizer initialization."""
        optimizer = HNSWOptimizer(config, mock_qdrant_service)

        assert optimizer.config == config
        assert optimizer.qdrant_service == mock_qdrant_service
        assert optimizer.performance_cache == {}
        assert optimizer.adaptive_ef_cache == {}
        assert not optimizer._initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self, optimizer):
        """Test successful initialization."""
        await optimizer.initialize()

        assert optimizer._initialized

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, optimizer):
        """Test initialization when already initialized."""
        optimizer._initialized = True

        await optimizer.initialize()  # Should return early

        assert optimizer._initialized

    @pytest.mark.asyncio
    async def test_initialize_qdrant_not_initialized(self, optimizer):
        """Test initialization when QdrantService not initialized."""
        optimizer.qdrant_service._initialized = False

        with pytest.raises(
            QdrantServiceError, match="QdrantService must be initialized"
        ):
            await optimizer.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """Test cleanup clears caches."""
        optimizer._initialized = True
        optimizer.performance_cache = {"test": "data"}
        optimizer.adaptive_ef_cache = {"test": "data"}

        await optimizer.cleanup()

        assert not optimizer._initialized
        assert optimizer.performance_cache == {}
        assert optimizer.adaptive_ef_cache == {}

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_cached(self, optimizer):
        """Test adaptive EF retrieve with cached result."""
        collection_name = "test_collection"
        query_vector = [0.1] * 384

        # Set up cache
        cache_key = f"{collection_name}_50"
        optimizer.adaptive_ef_cache[cache_key] = {"optimal_ef": 150}

        # Mock Qdrant response
        mock_results = MagicMock()
        mock_results.points = [{"id": "1", "score": 0.95}]
        optimizer.qdrant_service._client.query_points = AsyncMock(
            return_value=mock_results
        )

        # Set optimizer as initialized
        optimizer._initialized = True

        result = await optimizer.adaptive_ef_retrieve(
            collection_name=collection_name,
            query_vector=query_vector,
            target_limit=50,
            time_budget_ms=100,
        )

        assert result["ef_used"] == 150
        assert result["source"] == "cache"
        assert result["results"] == mock_results.points
        assert "search_time_ms" in result

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_not_initialized(self, optimizer):
        """Test adaptive EF retrieve when not initialized."""
        with pytest.raises(ValueError, match="HNSWOptimizer not initialized"):
            await optimizer.adaptive_ef_retrieve(
                collection_name="test", query_vector=[0.1] * 384, target_limit=10
            )

    @pytest.mark.asyncio
    async def test_get_adaptive_search_params(self, optimizer):
        """Test getting adaptive search parameters."""
        # Mock collection info
        mock_collection = MagicMock()
        mock_collection.status = "green"
        mock_collection.points_count = 100000
        mock_collection.config.params.vectors.size = 384

        optimizer.qdrant_service.get_collection = AsyncMock(
            return_value=mock_collection
        )
        optimizer._initialized = True

        params = await optimizer.get_adaptive_search_params(
            collection_name="test_collection", target_recall=0.95
        )

        assert "hnsw_ef" in params
        assert params["hnsw_ef"] >= 100  # Should be at least the base EF
        assert params["exact"] is False

    @pytest.mark.asyncio
    async def test_update_performance_cache(self, optimizer):
        """Test updating performance cache."""
        await optimizer.update_performance_cache(
            collection_name="test_collection",
            ef_value=150,
            search_time_ms=25.5,
            recall_estimate=0.95,
        )

        cache_key = "test_collection_150"
        assert cache_key in optimizer.performance_cache
        assert optimizer.performance_cache[cache_key]["search_time_ms"] == 25.5
        assert optimizer.performance_cache[cache_key]["recall_estimate"] == 0.95

    @pytest.mark.asyncio
    async def test_estimate_recall_small_ef(self, optimizer):
        """Test recall estimation for small EF values."""
        # Small collection with low EF
        recall = await optimizer.estimate_recall(
            ef=50, collection_size=1000, vector_dim=384
        )

        # Should have lower recall
        assert 0.5 <= recall <= 0.9

    @pytest.mark.asyncio
    async def test_estimate_recall_large_ef(self, optimizer):
        """Test recall estimation for large EF values."""
        # Large EF relative to collection
        recall = await optimizer.estimate_recall(
            ef=500, collection_size=10000, vector_dim=384
        )

        # Should have higher recall
        assert recall >= 0.95

    @pytest.mark.asyncio
    async def test_estimate_recall_exact_search(self, optimizer):
        """Test recall estimation when EF exceeds collection size."""
        # EF larger than collection
        recall = await optimizer.estimate_recall(
            ef=1000, collection_size=500, vector_dim=384
        )

        # Should be perfect recall
        assert recall == 1.0
