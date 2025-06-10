"""Comprehensive tests for monitoring metrics functionality."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client.registry import CollectorRegistry
from src.services.monitoring.metrics import MetricsConfig, MetricsRegistry


class TestMetricsConfig:
    """Test MetricsConfig model validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.export_port == 8000
        assert config.namespace == "ml_app"
        assert config.include_system_metrics is True
        assert config.collection_interval == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricsConfig(
            enabled=False,
            export_port=9090,
            namespace="custom_app",
            include_system_metrics=False,
            collection_interval=60.0,
        )
        assert config.enabled is False
        assert config.export_port == 9090
        assert config.namespace == "custom_app"
        assert config.include_system_metrics is False
        assert config.collection_interval == 60.0


class TestMetricsRegistry:
    """Test MetricsRegistry functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MetricsConfig(enabled=True, namespace="test_metrics")

    @pytest.fixture
    def test_registry(self):
        """Create isolated test registry."""
        return CollectorRegistry()

    @pytest.fixture
    def registry(self, config, test_registry):
        """Create test metrics registry."""
        return MetricsRegistry(config, test_registry)

    def test_initialization(self, registry):
        """Test registry initialization."""
        assert registry.config.enabled is True
        assert registry._metrics is not None
        assert len(registry._metrics) > 0

    def test_metrics_creation(self, registry):
        """Test that all expected metrics are created."""
        expected_metrics = [
            "search_requests",
            "search_duration", 
            "search_concurrent",
            "embedding_requests",
            "embedding_duration",
            "embedding_cost",
            "cache_hits",
            "cache_misses",
            "service_health",
        ]

        for metric_name in expected_metrics:
            assert metric_name in registry._metrics

    @pytest.mark.asyncio
    async def test_monitor_search_performance_decorator(self, registry):
        """Test search performance monitoring decorator."""

        @registry.monitor_search_performance(collection="test", query_type="semantic")
        async def mock_search():
            await asyncio.sleep(0.01)
            return {"results": []}

        result = await mock_search()
        assert result == {"results": []}

    @pytest.mark.asyncio
    async def test_monitor_embedding_generation_decorator(self, registry):
        """Test embedding generation monitoring decorator."""

        @registry.monitor_embedding_generation(
            provider="openai", model="text-embedding-ada-002"
        )
        async def mock_embedding():
            await asyncio.sleep(0.01)
            return []

        result = await mock_embedding()
        assert result == []

    def test_record_embedding_cost(self, registry):
        """Test recording embedding cost metrics."""
        registry.record_embedding_cost(
            provider="openai", model="text-embedding-ada-002", cost=0.0001
        )
        # Verify metric exists
        assert "embedding_cost" in registry._metrics

    def test_update_cache_stats(self, registry):
        """Test updating cache statistics."""
        mock_cache_manager = MagicMock()
        mock_cache_manager.get_stats = AsyncMock(return_value={
            "local_cache": {"hits": 10, "misses": 2, "size": 1024},
            "embedding_cache": {"hits": 5, "misses": 1, "size": 512}
        })
        
        registry.update_cache_stats(mock_cache_manager)
        assert "cache_hits" in registry._metrics
        assert "cache_misses" in registry._metrics

    def test_update_service_health(self, registry):
        """Test updating service health metrics."""
        registry.update_service_health(service="vector_search", healthy=True)
        registry.update_service_health(service="embeddings", healthy=False)
        assert "service_health" in registry._metrics

    def test_update_system_metrics(self, registry):
        """Test system metrics collection."""
        registry.update_system_metrics()
        # Just verify the method can be called without error
        # System metrics are created conditionally based on config

    def test_record_cache_operations(self, registry):
        """Test cache operation recording."""
        registry.record_cache_hit("local", "search")
        registry.record_cache_miss("embedding")
        assert "cache_hits" in registry._metrics
        assert "cache_misses" in registry._metrics

    def test_update_qdrant_metrics(self, registry):
        """Test Qdrant-specific metrics."""
        registry.update_qdrant_metrics("test_collection", size=1000, memory_usage=1024000)
        registry.record_qdrant_operation("search", "test_collection", success=True)
        assert "qdrant_collection_size" in registry._metrics
        assert "qdrant_operations" in registry._metrics

    def test_record_task_metrics(self, registry):
        """Test task queue metrics."""
        registry.record_task_queue_size("embeddings", "pending", 25)
        registry.record_task_execution("embeddings", duration_seconds=2.5, success=True)
        registry.update_worker_count("embeddings", 3)
        
        assert "task_queue_size" in registry._metrics
        assert "task_execution_duration" in registry._metrics
        assert "worker_active" in registry._metrics

    def test_browser_metrics(self, registry):
        """Test browser automation metrics."""
        registry.record_browser_request("premium", duration_seconds=1.2, success=True)
        registry.update_browser_tier_health("basic", healthy=True)
        
        assert "browser_response_time" in registry._metrics
        assert "browser_tier_health" in registry._metrics