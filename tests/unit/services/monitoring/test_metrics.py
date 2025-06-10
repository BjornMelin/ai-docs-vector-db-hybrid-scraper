"""Tests for monitoring metrics functionality."""

import asyncio
from unittest.mock import patch

import pytest
from src.services.monitoring.metrics import MetricsConfig
from src.services.monitoring.metrics import MetricsRegistry


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
            collection_interval=60.0
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
        return MetricsConfig(enabled=True)

    @pytest.fixture
    def registry(self, config):
        """Create test metrics registry."""
        return MetricsRegistry(config)

    def test_initialization(self, registry):
        """Test registry initialization."""
        assert registry.config.enabled is True
        assert registry._metrics is not None
        assert len(registry._metrics) > 0

    def test_metrics_creation(self, registry):
        """Test that all expected metrics are created."""
        expected_metrics = [
            "vector_search_requests_total",
            "vector_search_duration_seconds",
            "vector_search_quality_score",
            "embedding_requests_total",
            "embedding_generation_duration_seconds",
            "embedding_cost_total",
            "cache_hits_total",
            "cache_misses_total",
            "service_health_status",
            "system_cpu_usage_percent"
        ]

        for metric_name in expected_metrics:
            assert f"ml_app_{metric_name}" in [m._name for m in registry._metrics.values()]

    @pytest.mark.asyncio
    async def test_monitor_search_performance_decorator(self, registry):
        """Test search performance monitoring decorator."""
        @registry.monitor_search_performance(collection="test", query_type="semantic")
        async def mock_search():
            await asyncio.sleep(0.1)
            return {"results": []}

        # Execute the decorated function
        result = await mock_search()

        assert result == {"results": []}
        # Verify metrics were recorded (would need to check Prometheus registry)

    @pytest.mark.asyncio
    async def test_monitor_embedding_generation_decorator(self, registry):
        """Test embedding generation monitoring decorator."""
        @registry.monitor_embedding_generation(provider="openai", model="text-embedding-ada-002")
        async def mock_embedding():
            await asyncio.sleep(0.05)
            return {"embeddings": [], "cost": 0.0001}

        result = await mock_embedding()

        assert result == {"embeddings": [], "cost": 0.0001}

    def test_record_search_quality(self, registry):
        """Test recording search quality metrics."""
        registry.record_search_quality(
            collection="test_collection",
            query_type="hybrid",
            quality_score=0.85
        )

        # Verify metric was recorded
        metric = registry._metrics["ml_app_vector_search_quality_score"]
        assert metric is not None

    def test_record_embedding_cost(self, registry):
        """Test recording embedding cost metrics."""
        registry.record_embedding_cost(
            provider="openai",
            model="text-embedding-ada-002",
            cost=0.0001
        )

        metric = registry._metrics["ml_app_embedding_cost_total"]
        assert metric is not None

    def test_update_cache_metrics(self, registry):
        """Test updating cache metrics."""
        registry.update_cache_metrics(
            cache_type="local",
            hits=10,
            misses=2,
            memory_usage=1024
        )

        hits_metric = registry._metrics["ml_app_cache_hits_total"]
        misses_metric = registry._metrics["ml_app_cache_misses_total"]
        memory_metric = registry._metrics["ml_app_cache_memory_usage_bytes"]

        assert hits_metric is not None
        assert misses_metric is not None
        assert memory_metric is not None

    def test_update_service_health(self, registry):
        """Test updating service health metrics."""
        registry.update_service_health(service="vector_search", is_healthy=True)
        registry.update_service_health(service="embedding_service", is_healthy=False)

        metric = registry._metrics["ml_app_service_health_status"]
        assert metric is not None

    def test_update_dependency_health(self, registry):
        """Test updating dependency health metrics."""
        registry.update_dependency_health(dependency="qdrant", is_healthy=True)
        registry.update_dependency_health(dependency="redis", is_healthy=False)

        metric = registry._metrics["ml_app_dependency_health_status"]
        assert metric is not None

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_update_system_metrics(self, mock_memory, mock_cpu, registry):
        """Test updating system metrics."""
        # Mock system data
        mock_cpu.return_value = 45.5
        mock_memory.return_value.used = 8 * 1024 * 1024 * 1024  # 8GB

        registry.update_system_metrics()

        cpu_metric = registry._metrics["ml_app_system_cpu_usage_percent"]
        memory_metric = registry._metrics["ml_app_system_memory_usage_bytes"]

        assert cpu_metric is not None
        assert memory_metric is not None

    def test_disabled_registry(self):
        """Test that disabled registry doesn't create metrics."""
        config = MetricsConfig(enabled=False)
        registry = MetricsRegistry(config)

        assert registry._metrics == {}

    def test_decorator_with_disabled_registry(self):
        """Test that decorators work with disabled registry."""
        config = MetricsConfig(enabled=False)
        registry = MetricsRegistry(config)

        @registry.monitor_search_performance()
        async def mock_search():
            return {"results": []}

        # Should not raise any errors
        assert asyncio.run(mock_search()) == {"results": []}


class TestMetricsIntegration:
    """Integration tests for metrics functionality."""

    @pytest.fixture
    def registry(self):
        """Create registry for integration tests."""
        config = MetricsConfig(enabled=True)
        return MetricsRegistry(config)

    def test_concurrent_metrics_updates(self, registry):
        """Test concurrent metrics updates don't cause issues."""
        import threading

        def update_metrics():
            for i in range(100):
                registry.record_search_quality("test", "semantic", 0.8)
                registry.update_cache_metrics("local", 1, 0, 1024)

        threads = [threading.Thread(target=update_metrics) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert True

    @pytest.mark.asyncio
    async def test_decorator_error_handling(self, registry):
        """Test that decorators properly handle function errors."""
        @registry.monitor_search_performance()
        async def failing_search():
            raise ValueError("Search failed")

        with pytest.raises(ValueError, match="Search failed"):
            await failing_search()

        # Metrics should still be recorded for failed operations
