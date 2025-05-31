"""Additional tests for HNSW optimizer coverage."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from src.config.models import HNSWConfig
from src.config.models import QdrantConfig
from src.config.models import UnifiedConfig
from src.services.utilities.hnsw_optimizer import HNSWOptimizer


class OptimizationResult(BaseModel):
    """Mock optimization result for testing."""

    collection_name: str
    optimal_ef: int
    optimal_m: int
    baseline_recall: float
    optimized_recall: float
    baseline_latency_ms: float
    optimized_latency_ms: float
    optimization_time_s: float

    @property
    def recall_improvement(self) -> float:
        return self.optimized_recall - self.baseline_recall

    @property
    def latency_improvement(self) -> float:
        return self.baseline_latency_ms - self.optimized_latency_ms


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
            full_scan_threshold=10000,
            max_indexing_threads=0,
            on_disk=False,
            payload_m=None,
            enable_collection_optimization=True,
            optimization_timeout=300,
        ),
    )


@pytest.fixture
def mock_qdrant_service():
    """Create mock QdrantService."""
    service = MagicMock()
    service._initialized = True
    return service


@pytest.fixture
def optimizer(config, mock_qdrant_service):
    """Create HNSW optimizer."""
    return HNSWOptimizer(config, mock_qdrant_service)


class TestHNSWOptimizerCoverage:
    """Additional tests for HNSW optimizer."""

    def test_initialization(self, config, mock_qdrant_service):
        """Test optimizer initialization."""
        optimizer = HNSWOptimizer(config, mock_qdrant_service)

        assert optimizer.config == config
        assert optimizer.qdrant_service == mock_qdrant_service
        assert optimizer.performance_cache == {}
        assert optimizer.adaptive_ef_cache == {}

    def test_get_collection_config_custom(self, optimizer):
        """Test getting custom collection configuration."""
        custom_config = {
            "m": 32,
            "ef_construct": 400,
            "ef": 200,
            "max_indexing_threads": 4,
        }

        config = optimizer.get_collection_config("test_collection", custom_config)

        assert config["hnsw_config"]["m"] == 32
        assert config["hnsw_config"]["ef_construct"] == 400
        assert config["hnsw_config"]["max_indexing_threads"] == 4
        assert config["optimizers_config"]["default_segment_number"] == 0

    def test_get_collection_config_defaults(self, optimizer):
        """Test getting collection config with defaults."""
        config = optimizer.get_collection_config("test_collection")

        assert config["hnsw_config"]["m"] == 16
        assert config["hnsw_config"]["ef_construct"] == 200
        assert config["hnsw_config"]["full_scan_threshold"] == 10000
        assert config["hnsw_config"]["on_disk"] is False

    @pytest.mark.asyncio
    async def test_calculate_adaptive_ef_small_collection(self, optimizer):
        """Test adaptive EF calculation for small collections."""
        # Small collection
        ef = await optimizer._calculate_adaptive_ef(
            num_points=100, target_recall=0.95, collection_name="small_collection"
        )
        assert ef == 100  # Should use base EF for small collections

    @pytest.mark.asyncio
    async def test_calculate_adaptive_ef_medium_collection(self, optimizer):
        """Test adaptive EF calculation for medium collections."""
        # Medium collection
        ef = await optimizer._calculate_adaptive_ef(
            num_points=50000, target_recall=0.95, collection_name="medium_collection"
        )
        # Should be between base and increased value
        assert 100 < ef <= 150

    @pytest.mark.asyncio
    async def test_calculate_adaptive_ef_large_collection(self, optimizer):
        """Test adaptive EF calculation for large collections."""
        # Large collection
        ef = await optimizer._calculate_adaptive_ef(
            num_points=500000, target_recall=0.99, collection_name="large_collection"
        )
        # Should be higher for large collections with high recall
        assert ef > 150

    @pytest.mark.asyncio
    async def test_calculate_adaptive_ef_cached(self, optimizer):
        """Test adaptive EF uses cache."""
        collection_name = "cached_collection"

        # First call
        ef1 = await optimizer._calculate_adaptive_ef(
            num_points=100000, target_recall=0.95, collection_name=collection_name
        )

        # Second call should use cache
        ef2 = await optimizer._calculate_adaptive_ef(
            num_points=100000, target_recall=0.95, collection_name=collection_name
        )

        assert ef1 == ef2
        assert collection_name in optimizer._optimization_cache

    @pytest.mark.asyncio
    async def test_benchmark_collection_not_initialized(self, optimizer):
        """Test benchmarking when not initialized."""
        with pytest.raises(ValueError, match="not initialized"):
            await optimizer.benchmark_collection("test_collection")

    @pytest.mark.asyncio
    async def test_optimize_hnsw_parameters(self, optimizer):
        """Test HNSW parameter optimization."""
        mock_qdrant = MagicMock()
        mock_qdrant.get_collection = AsyncMock(
            return_value=MagicMock(
                status="green",
                points_count=100000,
                config=MagicMock(params=MagicMock(vectors=MagicMock(size=384))),
            )
        )

        with patch.object(optimizer, "_qdrant_client", mock_qdrant):
            with patch.object(optimizer, "_initialized", True):
                result = await optimizer.optimize_hnsw_parameters(
                    collection_name="test_collection",
                    sample_queries=[[0.1] * 384, [0.2] * 384],
                    target_recall=0.95,
                )

                assert isinstance(result, OptimizationResult)
                assert result.collection_name == "test_collection"
                assert result.optimal_ef >= 100

    @pytest.mark.asyncio
    async def test_optimize_for_workload_search_focused(self, optimizer):
        """Test optimization for search-focused workload."""
        params = await optimizer.optimize_for_workload(
            workload_type="search_focused", expected_points=100000
        )

        # Search-focused should have higher EF
        assert params["ef"] >= 200
        assert params["m"] == 16

    @pytest.mark.asyncio
    async def test_optimize_for_workload_index_focused(self, optimizer):
        """Test optimization for index-focused workload."""
        params = await optimizer.optimize_for_workload(
            workload_type="index_focused", expected_points=100000
        )

        # Index-focused should have lower EF_construct
        assert params["ef_construct"] <= 128
        assert params["m"] == 16

    @pytest.mark.asyncio
    async def test_optimize_for_workload_balanced(self, optimizer):
        """Test optimization for balanced workload."""
        params = await optimizer.optimize_for_workload(
            workload_type="balanced", expected_points=50000
        )

        # Balanced should have moderate values
        assert 100 <= params["ef"] <= 200
        assert 128 <= params["ef_construct"] <= 256

    @pytest.mark.asyncio
    async def test_optimize_for_workload_unknown_type(self, optimizer):
        """Test optimization with unknown workload type."""
        params = await optimizer.optimize_for_workload(
            workload_type="unknown", expected_points=10000
        )

        # Should fall back to balanced
        assert params is not None
        assert "ef" in params

    @pytest.mark.asyncio
    async def test_get_optimization_stats_empty(self, optimizer):
        """Test getting stats with no optimizations."""
        stats = await optimizer.get_optimization_stats()

        assert stats["total_optimizations"] == 0
        assert stats["collections_optimized"] == []
        assert stats["average_recall_improvement"] == 0
        assert stats["average_latency_improvement"] == 0

    @pytest.mark.asyncio
    async def test_get_optimization_stats_with_data(self, optimizer):
        """Test getting stats with optimization data."""
        # Add some fake optimization results
        optimizer._optimization_history = [
            OptimizationResult(
                collection_name="col1",
                optimal_ef=150,
                optimal_m=16,
                baseline_recall=0.90,
                optimized_recall=0.95,
                baseline_latency_ms=10.0,
                optimized_latency_ms=8.0,
                optimization_time_s=5.0,
            ),
            OptimizationResult(
                collection_name="col2",
                optimal_ef=200,
                optimal_m=32,
                baseline_recall=0.85,
                optimized_recall=0.93,
                baseline_latency_ms=15.0,
                optimized_latency_ms=12.0,
                optimization_time_s=7.0,
            ),
        ]

        stats = await optimizer.get_optimization_stats()

        assert stats["total_optimizations"] == 2
        assert set(stats["collections_optimized"]) == {"col1", "col2"}
        assert stats["average_recall_improvement"] > 0
        assert stats["average_latency_improvement"] > 0

    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """Test cleanup clears caches."""
        optimizer._optimization_cache = {"test": "data"}
        optimizer._optimization_history = [MagicMock()]
        optimizer._qdrant_client = MagicMock()

        await optimizer.cleanup()

        assert optimizer._optimization_cache == {}
        assert optimizer._optimization_history == []
        assert optimizer._qdrant_client is None
        assert not optimizer._initialized

    def test_optimization_result_creation(self):
        """Test OptimizationResult model."""
        result = OptimizationResult(
            collection_name="test",
            optimal_ef=150,
            optimal_m=16,
            baseline_recall=0.90,
            optimized_recall=0.95,
            baseline_latency_ms=10.0,
            optimized_latency_ms=8.0,
            optimization_time_s=5.0,
        )

        assert result.collection_name == "test"
        assert result.optimal_ef == 150
        assert result.recall_improvement == pytest.approx(0.05, rel=1e-3)
        assert result.latency_improvement == pytest.approx(2.0, rel=1e-3)
