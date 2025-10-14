"""Tests for HNSW parameter optimization utilities."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.services.errors import QdrantServiceError
from src.services.utilities.hnsw_optimizer import AdaptiveEfConfig, HNSWOptimizer


class TestError(Exception):
    """Custom exception for this module."""


class TestHNSWOptimizer:
    """Tests for HNSWOptimizer class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock unified config."""
        return MagicMock(spec=Settings)

    @pytest.fixture
    def _mock_qdrant_service(self):
        """Create mock Qdrant service."""
        service = MagicMock()
        service._initialized = True
        service._client = MagicMock()
        service._client.query_points = AsyncMock()
        service._client.get_collection = AsyncMock()
        service.get_client = AsyncMock(return_value=service._client)
        service.is_initialized = MagicMock(
            side_effect=lambda: bool(service._initialized)
        )
        return service

    @pytest.fixture
    def optimizer(self, mock_config, _mock_qdrant_service):
        """Create HNSWOptimizer instance."""
        return HNSWOptimizer(config=mock_config, qdrant_service=_mock_qdrant_service)

    def test_init(self, mock_config, _mock_qdrant_service):
        """Test optimizer initialization."""
        optimizer = HNSWOptimizer(
            config=mock_config, qdrant_service=_mock_qdrant_service
        )

        assert optimizer.config == mock_config
        assert optimizer.qdrant_service == _mock_qdrant_service
        assert optimizer.performance_cache == {}
        assert optimizer.adaptive_ef_cache == {}
        assert optimizer._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, optimizer, _mock_qdrant_service):
        """Test successful optimizer initialization."""
        await optimizer.initialize()

        assert optimizer._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, optimizer):
        """Test initialization when already initialized."""
        optimizer._initialized = True

        await optimizer.initialize()

        # Should remain initialized
        assert optimizer._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_qdrant_not_initialized(
        self, optimizer, _mock_qdrant_service
    ):
        """Test initialization when Qdrant service not initialized."""
        _mock_qdrant_service._initialized = False

        with pytest.raises(QdrantServiceError) as exc_info:
            await optimizer.initialize()

        assert "QdrantService must be initialized" in str(exc_info.value)
        assert optimizer._initialized is False

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_success(self, optimizer, _mock_qdrant_service):
        """Test successful adaptive ef retrieval."""
        optimizer._initialized = True

        # Mock query results
        mock_result = MagicMock()
        mock_result.points = [{"id": "doc1", "score": 0.9}]
        _mock_qdrant_service._client.query_points.return_value = mock_result

        query_vector = [0.1, 0.2, 0.3]
        config = AdaptiveEfConfig(
            time_budget_ms=100,
            min_ef=50,
            max_ef=200,
            target_limit=10,
        )
        result = await optimizer.adaptive_ef_retrieve(
            collection_name="test_collection",
            query_vector=query_vector,
            config=config,
        )

        assert "results" in result
        assert "ef_used" in result
        assert "search_time_ms" in result
        assert "time_budget_ms" in result
        assert result["time_budget_ms"] == 100
        assert result["source"] == "adaptive"
        assert len(result["ef_progression"]) > 0

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_with_cache_hit(
        self, optimizer, _mock_qdrant_service
    ):
        """Test adaptive ef retrieval with cache hit."""
        optimizer._initialized = True

        # Pre-populate cache
        cache_key = "test_collection:100:50:200"
        optimizer.adaptive_ef_cache[cache_key] = {
            "optimal_ef": 75,
            "expected_time_ms": 50,
            "timestamp": time.time(),
        }

        # Mock query result
        mock_result = MagicMock()
        mock_result.points = [{"id": "doc1", "score": 0.9}]
        _mock_qdrant_service._client.query_points.return_value = mock_result

        query_vector = [0.1, 0.2, 0.3]
        config = AdaptiveEfConfig(time_budget_ms=100, min_ef=50, max_ef=200)
        result = await optimizer.adaptive_ef_retrieve(
            collection_name="test_collection",
            query_vector=query_vector,
            config=config,
        )

        assert result["ef_used"] == 75
        assert result["source"] == "cache"
        _mock_qdrant_service._client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_time_budget_exceeded(
        self, optimizer, _mock_qdrant_service
    ):
        """Test adaptive ef retrieval when time budget is exceeded."""
        optimizer._initialized = True

        # Mock slow query that exceeds budget
        async def slow_query(*_args, **__kwargs):
            await asyncio.sleep(0.1)  # Simulate 100ms query
            mock_result = MagicMock()
            mock_result.points = [{"id": "doc1", "score": 0.9}]
            return mock_result

        _mock_qdrant_service._client.query_points.side_effect = slow_query

        query_vector = [0.1, 0.2, 0.3]
        config = AdaptiveEfConfig(
            time_budget_ms=50,  # Small budget
            min_ef=50,
            max_ef=200,
            step_size=25,
        )
        result = await optimizer.adaptive_ef_retrieve(
            collection_name="test_collection",
            query_vector=query_vector,
            config=config,
        )

        # Should stop early due to budget
        assert result["ef_used"] == 50  # Should use minimum EF
        assert result["budget_utilized_percent"] > 80  # Should be close to budget

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_query_error(
        self, optimizer, _mock_qdrant_service
    ):
        """Test adaptive ef retrieval with query error."""
        optimizer._initialized = True

        _mock_qdrant_service._client.query_points.side_effect = Exception(
            "Query failed"
        )

        query_vector = [0.1, 0.2, 0.3]
        result = await optimizer.adaptive_ef_retrieve(
            collection_name="test_collection",
            query_vector=query_vector,
        )

        # Should handle error gracefully
        assert result["results"] == []
        assert "ef_used" in result

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_cache_size_limit(
        self, optimizer, _mock_qdrant_service
    ):
        """Test cache size limiting in adaptive ef retrieval."""
        optimizer._initialized = True

        # Fill cache close to limit
        for i in range(99):  # Just under 100 limit
            cache_key = f"collection_{i}:100:50:200"
            optimizer.adaptive_ef_cache[cache_key] = {
                "optimal_ef": 75,
                "expected_time_ms": 50,
                "timestamp": time.time() - i,  # Different timestamps
            }

        # Mock query result
        mock_result = MagicMock()
        mock_result.points = [{"id": "doc1", "score": 0.9}]
        _mock_qdrant_service._client.query_points.return_value = mock_result

        query_vector = [0.1, 0.2, 0.3]
        await optimizer.adaptive_ef_retrieve(
            collection_name="new_collection",
            query_vector=query_vector,
        )

        # Should have added new entry and cleaned up if needed
        assert len(optimizer.adaptive_ef_cache) <= 100

    def test_get_collection_specific_hnsw_config_api_reference(self, optimizer):
        """Test getting HNSW config for API reference collection."""
        config = optimizer.get_collection_specific_hnsw_config("api_reference")

        assert config["m"] == 20
        assert config["ef_construct"] == 300
        assert config["full_scan_threshold"] == 5000
        assert config["description"] == "High accuracy for API documentation"
        assert "ef_recommendations" in config
        assert config["ef_recommendations"]["min_ef"] == 100

    def test_get_collection_specific_hnsw_config_tutorials(self, optimizer):
        """Test getting HNSW config for tutorials collection."""
        config = optimizer.get_collection_specific_hnsw_config("tutorials")

        assert config["m"] == 16
        assert config["ef_construct"] == 200
        assert config["full_scan_threshold"] == 10000
        assert config["ef_recommendations"]["balanced_ef"] == 100

    def test_get_collection_specific_hnsw_config_blog_posts(self, optimizer):
        """Test getting HNSW config for blog posts collection."""
        config = optimizer.get_collection_specific_hnsw_config("blog_posts")

        assert config["m"] == 12
        assert config["ef_construct"] == 150
        assert config["full_scan_threshold"] == 20000
        assert config["ef_recommendations"]["max_ef"] == 100

    def test_get_collection_specific_hnsw_config_code_examples(self, optimizer):
        """Test getting HNSW config for code examples collection."""
        config = optimizer.get_collection_specific_hnsw_config("code_examples")

        assert config["m"] == 18
        assert config["ef_construct"] == 250
        assert config["full_scan_threshold"] == 8000
        assert config["ef_recommendations"]["min_ef"] == 100

    def test_get_collection_specific_hnsw_config_unknown(self, optimizer):
        """Test getting HNSW config for unknown collection type."""
        config = optimizer.get_collection_specific_hnsw_config("unknown_type")

        # Should return general config
        assert config["m"] == 16
        assert config["ef_construct"] == 200
        assert config["description"] == "Default balanced configuration"

    def test_get_collection_specific_hnsw_config_general(self, optimizer):
        """Test getting HNSW config for general collection."""
        config = optimizer.get_collection_specific_hnsw_config("general")

        assert config["m"] == 16
        assert config["ef_construct"] == 200
        assert config["full_scan_threshold"] == 10000
        assert config["target_use_case"] == "General purpose documentation"

    @pytest.mark.asyncio
    async def test_optimize_collection_hnsw_success(
        self, optimizer, _mock_qdrant_service
    ):
        """Test successful collection HNSW optimization."""
        optimizer._initialized = True

        # Mock collection info
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = 12
        mock_collection_info.config.params.vectors.dense.hnsw_config.ef_construct = 128
        _mock_qdrant_service._client.get_collection.return_value = mock_collection_info

        result = await optimizer.optimize_collection_hnsw(
            collection_name="test_collection",
            collection_type="api_reference",
        )

        assert result["collection_name"] == "test_collection"
        assert result["collection_type"] == "api_reference"
        assert "current_config" in result
        assert "recommended_config" in result
        assert "needs_update" in result
        assert "optimization_timestamp" in result

    @pytest.mark.asyncio
    async def test_optimize_collection_hnsw_with_test_queries(
        self, optimizer, _mock_qdrant_service
    ):
        """Test collection optimization with test queries."""
        optimizer._initialized = True

        # Mock collection info with proper structure
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = 16
        mock_collection_info.config.params.vectors.dense.hnsw_config.ef_construct = 200
        hnsw_config = mock_collection_info.config.params.vectors.dense.hnsw_config
        hnsw_config.full_scan_threshold = 10000
        _mock_qdrant_service._client.get_collection.return_value = mock_collection_info

        # Mock test performance
        with patch.object(optimizer, "_test_search_performance") as mock_test:
            mock_test.return_value = {
                "avg_search_time_ms": 50.0,
                "p95_search_time_ms": 80.0,
                "queries_tested": 5,
            }

            test_queries = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            result = await optimizer.optimize_collection_hnsw(
                collection_name="test_collection",
                collection_type="tutorials",
                test_queries=test_queries,
            )

            assert result["current_performance"] is not None
            assert result["current_performance"]["avg_search_time_ms"] == 50.0
            mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_collection_hnsw_needs_update(
        self, optimizer, _mock_qdrant_service
    ):
        """Test optimization when collection needs update."""
        optimizer._initialized = True

        # Mock collection with suboptimal config
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = (
            8  # Low m value
        )
        mock_collection_info.config.params.vectors.dense.hnsw_config.ef_construct = (
            50  # Low ef
        )
        _mock_qdrant_service._client.get_collection.return_value = mock_collection_info

        result = await optimizer.optimize_collection_hnsw(
            collection_name="test_collection",
            collection_type="api_reference",  # Recommends m=20, ef=300
        )

        assert result["needs_update"] is True
        assert result["update_recommendation"]["action"] == "recreate_collection"
        assert "estimated_improvement" in result["update_recommendation"]

    @pytest.mark.asyncio
    async def test_optimize_collection_hnsw_no_update_needed(
        self, optimizer, _mock_qdrant_service
    ):
        """Test optimization when no update is needed."""
        optimizer._initialized = True

        # Mock collection with good config
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = 20
        mock_collection_info.config.params.vectors.dense.hnsw_config.ef_construct = 300
        _mock_qdrant_service._client.get_collection.return_value = mock_collection_info

        result = await optimizer.optimize_collection_hnsw(
            collection_name="test_collection",
            collection_type="api_reference",
        )

        assert result["needs_update"] is False
        assert result["update_recommendation"]["action"] == "no_update_needed"

    def test_extract_current_hnsw_config_success(self, optimizer):
        """Test successful HNSW config extraction."""
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = 18
        mock_collection_info.config.params.vectors.dense.hnsw_config.ef_construct = 250
        hnsw_config = mock_collection_info.config.params.vectors.dense.hnsw_config
        hnsw_config.full_scan_threshold = 8000

        config = optimizer._extract_current_hnsw_config(mock_collection_info)

        assert config["m"] == 18
        assert config["ef_construct"] == 250
        assert config["full_scan_threshold"] == 8000

    def test_extract_current_hnsw_config_missing_attributes(self, optimizer):
        """Test HNSW config extraction with missing attributes."""
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.dense.hnsw_config.m = 16

        # Simulate missing attributes by making getattr return None for missing ones
        def mock_getattr(_obj, attr, default=None):
            if attr == "m":
                return 16
            return default

        with patch("builtins.getattr", side_effect=mock_getattr):
            config = optimizer._extract_current_hnsw_config(mock_collection_info)

        assert config["m"] == 16
        assert config["ef_construct"] == 128  # Default
        assert config["full_scan_threshold"] == 10000  # Default

    def test_extract_current_hnsw_config_extraction_error(self, optimizer):
        """Test HNSW config extraction with error."""
        mock_collection_info = MagicMock()
        # Simulate extraction error
        mock_collection_info.config.params.vectors.dense = None

        config = optimizer._extract_current_hnsw_config(mock_collection_info)

        # Should return defaults
        assert config["m"] == 16
        assert config["ef_construct"] == 128
        assert config["full_scan_threshold"] == 10000

    def test_compare_hnsw_configs_significant_difference(self, optimizer):
        """Test config comparison with significant differences."""
        current = {"m": 12, "ef_construct": 100}
        recommended = {"m": 20, "ef_construct": 200}

        needs_update = optimizer._compare_hnsw_configs(current, recommended)

        assert needs_update is True  # Both m and ef differences are significant

    def test_compare_hnsw_configs_minor_difference(self, optimizer):
        """Test config comparison with minor differences."""
        current = {"m": 16, "ef_construct": 180}
        recommended = {"m": 18, "ef_construct": 200}

        needs_update = optimizer._compare_hnsw_configs(current, recommended)

        assert needs_update is False  # Differences are within thresholds

    def test_compare_hnsw_configs_m_difference_only(self, optimizer):
        """Test config comparison with only m difference."""
        current = {"m": 12, "ef_construct": 200}
        recommended = {"m": 20, "ef_construct": 200}

        needs_update = optimizer._compare_hnsw_configs(current, recommended)

        assert needs_update is True  # m difference of 8 is >= 4

    def test_compare_hnsw_configs_ef_difference_only(self, optimizer):
        """Test config comparison with only ef_construct difference."""
        current = {"m": 16, "ef_construct": 100}
        recommended = {"m": 16, "ef_construct": 200}

        needs_update = optimizer._compare_hnsw_configs(current, recommended)

        assert needs_update is True  # ef difference of 100 is >= 50

    def test_estimate_performance_improvement(self, optimizer):
        """Test performance improvement estimation."""
        current = {"m": 12, "ef_construct": 128}
        recommended = {"m": 20, "ef_construct": 300}

        improvement = optimizer._estimate_performance_improvement(current, recommended)

        assert improvement["estimated_recall_improvement_percent"] > 0
        # Note: The test implementation shows latency can be negative due to
        # better quality
        # Higher m increases latency (+2% per m = +16%) but higher ef can
        # reduce it slightly (-0.1% per ef = -17.2%)
        # Net effect: +16% - 17.2% = -1.2%
        assert "estimated_latency_change_percent" in improvement
        assert (
            improvement["estimated_memory_change_percent"] > 0
        )  # Higher m increases memory
        assert (
            improvement["build_time_change_percent"] > 0
        )  # Higher ef increases build time
        assert improvement["confidence"] == "medium"

    def test_estimate_performance_improvement_same_config(self, optimizer):
        """Test performance improvement estimation with same config."""
        config = {"m": 16, "ef_construct": 200}

        improvement = optimizer._estimate_performance_improvement(config, config)

        assert improvement["estimated_recall_improvement_percent"] == 0
        assert improvement["estimated_latency_change_percent"] == 0
        assert improvement["estimated_memory_change_percent"] == 0
        assert improvement["build_time_change_percent"] == 0

    @pytest.mark.asyncio
    async def test_test_search_performance_success(
        self, optimizer, _mock_qdrant_service
    ):
        """Test search performance testing."""
        optimizer._initialized = True

        # Mock query results with varying times
        _mock_qdrant_service._client.query_points = AsyncMock()

        test_queries = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        with patch(
            "time.time", side_effect=[0, 0.05, 0.05, 0.08, 0.08, 0.12]
        ):  # Mock timing
            performance = await optimizer._test_search_performance(
                collection_name="test_collection",
                test_queries=test_queries,
                ef_value=100,
            )

        assert "avg_search_time_ms" in performance
        assert "p95_search_time_ms" in performance
        assert "min_search_time_ms" in performance
        assert "max_search_time_ms" in performance
        assert performance["queries_tested"] == 3
        assert performance["ef_used"] == 100

    @pytest.mark.asyncio
    async def test_test_search_performance_with_errors(
        self, optimizer, _mock_qdrant_service
    ):
        """Test search performance testing with some query errors."""
        optimizer._initialized = True

        # Mock some successful and some failing queries
        call_count = 0

        async def mock_query(*_args, **__kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First 2 calls succeed
                return MagicMock()
            return None

        _mock_qdrant_service._client.query_points.side_effect = mock_query

        test_queries = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]

        performance = await optimizer._test_search_performance(
            collection_name="test_collection",
            test_queries=test_queries,
            ef_value=100,
        )

        assert performance["queries_tested"] == 2  # Only 2 successful
        assert performance["ef_used"] == 100

    @pytest.mark.asyncio
    async def test_test_search_performance_no_successful_queries(
        self, optimizer, _mock_qdrant_service
    ):
        """Test search performance testing with no successful queries."""
        optimizer._initialized = True

        _mock_qdrant_service._client.query_points.side_effect = Exception(
            "All queries failed"
        )

        test_queries = [[0.1, 0.2, 0.3]]

        performance = await optimizer._test_search_performance(
            collection_name="test_collection",
            test_queries=test_queries,
            ef_value=100,
        )

        assert "error" in performance
        assert performance["queries_tested"] == 0
        assert performance["ef_used"] == 100

    def test_get_performance_cache_stats(self, optimizer):
        """Test getting performance cache statistics."""
        # Add some cache entries
        optimizer.adaptive_ef_cache["test1"] = {"optimal_ef": 75}
        optimizer.adaptive_ef_cache["test2"] = {"optimal_ef": 100}
        optimizer.performance_cache["perf1"] = {"avg_time": 50}

        stats = optimizer.get_performance_cache_stats()

        assert stats["adaptive_ef_cache_size"] == 2
        assert stats["performance_cache_size"] == 1
        assert "test1" in stats["cache_entries"]
        assert "test2" in stats["cache_entries"]

    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """Test optimizer cleanup."""
        optimizer._initialized = True
        optimizer.adaptive_ef_cache["test"] = {"data": "value"}
        optimizer.performance_cache["test"] = {"data": "value"}

        await optimizer.cleanup()

        assert optimizer._initialized is False
        assert len(optimizer.adaptive_ef_cache) == 0
        assert len(optimizer.performance_cache) == 0

    @pytest.mark.asyncio
    async def test_adaptive_ef_retrieve_step_size_adjustment(
        self, optimizer, _mock_qdrant_service
    ):
        """Test step size adjustment based on timing in adaptive ef retrieval."""
        optimizer._initialized = True

        # Mock query results with different timing patterns
        call_count = 0

        async def mock_query_with_timing(*_args, **__kwargs):
            nonlocal call_count
            call_count += 1

            # First call: fast (< 50% budget)
            if call_count == 1:
                await asyncio.sleep(0.02)  # 20ms
            # Second call: moderate (between 50% and 80% budget)
            elif call_count == 2:
                await asyncio.sleep(0.06)  # 60ms
            # Third call: close to budget (>= 80% budget)
            else:
                await asyncio.sleep(0.09)  # 90ms

            mock_result = MagicMock()
            mock_result.points = [{"id": f"doc{call_count}", "score": 0.9}]
            return mock_result

        _mock_qdrant_service._client.query_points.side_effect = mock_query_with_timing

        query_vector = [0.1, 0.2, 0.3]
        config = AdaptiveEfConfig(
            time_budget_ms=100,
            min_ef=50,
            max_ef=200,
            step_size=50,
        )
        result = await optimizer.adaptive_ef_retrieve(
            collection_name="test_collection",
            query_vector=query_vector,
            config=config,
        )

        # Should have stopped before reaching max_ef due to time budget
        assert result["ef_used"] < 200
        assert len(result["ef_progression"]) >= 2
