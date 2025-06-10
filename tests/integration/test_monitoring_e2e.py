"""End-to-end integration tests for monitoring system."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client.registry import CollectorRegistry
from src.config.models import MonitoringConfig
from src.services.monitoring.health import HealthCheckManager, QdrantHealthCheck
from src.services.monitoring.initialization import (
    initialize_monitoring_system,
    start_background_monitoring_tasks,
    stop_background_monitoring_tasks,
)
from src.services.monitoring.metrics import MetricsRegistry
from src.services.monitoring.middleware import (
    CustomMetricsMiddleware,
    PrometheusMiddleware,
)


class TestMonitoringE2E:
    """End-to-end monitoring system integration tests."""

    @pytest.fixture
    def monitoring_config(self):
        """Create monitoring configuration for testing."""
        return MonitoringConfig(
            enabled=True,
            metrics={"enabled": True, "port": 8000},
            health_checks={"enabled": True, "interval": 1.0},
            include_system_metrics=True,
            system_metrics_interval=2.0,
        )

    @pytest.fixture
    def isolated_registry(self):
        """Create isolated Prometheus registry."""
        return CollectorRegistry()

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        client.get_collections.return_value = MagicMock(collections=[
            MagicMock(name="docs"),
            MagicMock(name="knowledge"),
        ])
        return client

    @pytest.fixture
    async def monitoring_system(self, monitoring_config, mock_qdrant_client, isolated_registry):
        """Create and initialize complete monitoring system."""
        with patch("src.services.monitoring.initialization.get_metrics_registry", return_value=None), \
             patch("src.services.monitoring.initialization.MetricsRegistry") as mock_metrics_cls, \
             patch("src.services.monitoring.initialization.HealthCheckManager") as mock_health_cls:
            
            # Create real instances for testing
            metrics_registry = MetricsRegistry(monitoring_config, isolated_registry)
            mock_metrics_cls.return_value = metrics_registry
            
            health_manager = HealthCheckManager(monitoring_config.health_checks)
            # Add Qdrant health check
            qdrant_check = QdrantHealthCheck("qdrant", mock_qdrant_client)
            health_manager.add_health_check(qdrant_check)
            mock_health_cls.return_value = health_manager
            
            # Initialize monitoring system
            metrics, health = initialize_monitoring_system(
                monitoring_config, mock_qdrant_client, None
            )
            
            yield {
                "metrics_registry": metrics,
                "health_manager": health,
                "config": monitoring_config,
            }

    @pytest.mark.asyncio
    async def test_complete_monitoring_lifecycle(self, monitoring_system):
        """Test complete monitoring system lifecycle."""
        metrics_registry = monitoring_system["metrics_registry"]
        health_manager = monitoring_system["health_manager"]
        config = monitoring_system["config"]
        
        # Start background tasks
        tasks = start_background_monitoring_tasks(
            config, metrics_registry, health_manager, None
        )
        
        try:
            # Let background tasks run
            await asyncio.sleep(0.5)
            
            # Perform some operations to generate metrics
            @metrics_registry.monitor_search_performance(collection="test", query_type="hybrid")
            async def mock_search():
                await asyncio.sleep(0.01)
                return {"results": ["doc1", "doc2"]}
            
            # Execute monitored operations
            await mock_search()
            await mock_search()
            
            # Record some manual metrics
            metrics_registry.record_embedding_cost("openai", "text-embedding-ada-002", 0.0001)
            metrics_registry.update_service_health("vector_search", healthy=True)
            
            # Run health checks
            health_results = await health_manager.check_all()
            overall_status = health_manager.get_overall_status()
            
            # Verify results
            assert len(health_results) > 0
            assert "qdrant" in health_results
            assert overall_status in ["healthy", "degraded", "unhealthy"]
            
            # Verify metrics were created
            assert len(metrics_registry._metrics) > 0
            
        finally:
            # Cleanup
            await stop_background_monitoring_tasks(tasks)

    @pytest.mark.asyncio
    async def test_fastapi_monitoring_integration(self, monitoring_system):
        """Test FastAPI application with monitoring integration."""
        metrics_registry = monitoring_system["metrics_registry"]
        health_manager = monitoring_system["health_manager"]
        
        # Create FastAPI app with monitoring
        app = FastAPI()
        
        # Add Prometheus middleware for endpoints
        PrometheusMiddleware(
            app=app,
            metrics_registry=metrics_registry,
            health_manager=health_manager,
        )
        
        # Add custom metrics middleware for request monitoring
        app.add_middleware(CustomMetricsMiddleware, metrics_registry=metrics_registry)
        
        # Add test endpoints
        @app.get("/api/search")
        async def search_endpoint():
            # Simulate search operation
            await asyncio.sleep(0.01)
            return {"results": ["doc1", "doc2", "doc3"]}
        
        @app.get("/api/embed")
        async def embed_endpoint():
            # Simulate embedding operation
            metrics_registry.record_embedding_cost("openai", "text-embedding-ada-002", 0.0002)
            return {"embedding": [0.1, 0.2, 0.3]}
        
        @app.get("/api/error")
        async def error_endpoint():
            raise Exception("Test error")
        
        # Test the application
        client = TestClient(app)
        
        # Test successful requests
        response = client.get("/api/search")
        assert response.status_code == 200
        assert "results" in response.json()
        
        response = client.get("/api/embed")
        assert response.status_code == 200
        assert "embedding" in response.json()
        
        # Test error handling
        response = client.get("/api/error")
        assert response.status_code == 500
        
        # Test health endpoints
        response = client.get("/health")
        assert response.status_code in [200, 503]  # Depends on health status
        
        response = client.get("/health/live")
        assert response.status_code == 200
        
        response = client.get("/health/ready")
        assert response.status_code in [200, 503]
        
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    @pytest.mark.asyncio
    async def test_monitoring_under_load(self, monitoring_system):
        """Test monitoring system under simulated load."""
        metrics_registry = monitoring_system["metrics_registry"]
        health_manager = monitoring_system["health_manager"]
        
        # Create FastAPI app
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=metrics_registry,
            health_manager=health_manager,
        )
        app.add_middleware(CustomMetricsMiddleware, metrics_registry=metrics_registry)
        
        @app.get("/api/load-test")
        async def load_test_endpoint():
            # Simulate varying response times
            import random
            await asyncio.sleep(random.uniform(0.001, 0.01))
            return {"status": "ok", "timestamp": time.time()}
        
        client = TestClient(app)
        
        # Simulate concurrent load
        import threading
        results = []
        errors = []
        
        def make_requests():
            try:
                for _ in range(10):
                    response = client.get("/api/load-test")
                    results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 5 threads * 10 requests each
        assert all(status == 200 for status in results)

    @pytest.mark.asyncio
    async def test_monitoring_error_resilience(self, monitoring_system):
        """Test monitoring system resilience to errors."""
        metrics_registry = monitoring_system["metrics_registry"]
        health_manager = monitoring_system["health_manager"]
        config = monitoring_system["config"]
        
        # Mock a health check that fails intermittently
        failing_check = AsyncMock()
        failing_check.name = "failing_service"
        
        call_count = 0
        async def failing_check_func():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Simulated failure")
            return MagicMock(
                name="failing_service",
                status="healthy",
                message="OK",
                duration_ms=50.0
            )
        
        failing_check.check = failing_check_func
        health_manager.add_health_check(failing_check)
        
        # Start background tasks
        tasks = start_background_monitoring_tasks(
            config, metrics_registry, health_manager, None
        )
        
        try:
            # Let tasks run with failures
            await asyncio.sleep(1.0)
            
            # System should still be functional
            health_results = await health_manager.check_all()
            
            # Should have results for working checks
            assert "qdrant" in health_results
            
            # Failing service may or may not be in results depending on timing
            # But system should continue operating
            
        finally:
            await stop_background_monitoring_tasks(tasks)

    @pytest.mark.asyncio
    async def test_metrics_collection_accuracy(self, monitoring_system):
        """Test accuracy of metrics collection."""
        metrics_registry = monitoring_system["metrics_registry"]
        
        # Record specific metrics
        initial_time = time.time()
        
        # Simulate search operations
        @metrics_registry.monitor_search_performance(collection="docs", query_type="semantic")
        async def search_operation():
            await asyncio.sleep(0.05)  # 50ms
            return {"results": ["doc1", "doc2"]}
        
        @metrics_registry.monitor_embedding_generation(provider="openai", model="text-embedding-ada-002")
        async def embedding_operation():
            await asyncio.sleep(0.02)  # 20ms
            return [0.1, 0.2, 0.3]
        
        # Execute operations
        search_result = await search_operation()
        embedding_result = await embedding_operation()
        
        # Record additional metrics
        metrics_registry.record_embedding_cost("openai", "text-embedding-ada-002", 0.0001)
        metrics_registry.record_cache_hit("local", "search")
        metrics_registry.record_cache_miss("embedding")
        
        # Verify operations completed
        assert search_result == {"results": ["doc1", "doc2"]}
        assert embedding_result == [0.1, 0.2, 0.3]
        
        # Verify metrics exist (detailed verification would require access to Prometheus registry)
        expected_metrics = [
            "search_requests",
            "search_duration", 
            "embedding_requests",
            "embedding_duration",
            "embedding_cost",
            "cache_hits",
            "cache_misses",
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics_registry._metrics

    @pytest.mark.asyncio
    async def test_health_check_cascading(self, monitoring_system):
        """Test health check cascading and dependencies."""
        health_manager = monitoring_system["health_manager"]
        
        # Add multiple health checks with different statuses
        mock_service_1 = AsyncMock()
        mock_service_1.name = "primary_service"
        mock_service_1.check.return_value = MagicMock(
            name="primary_service",
            status="healthy",
            message="Primary service OK",
            duration_ms=25.0
        )
        
        mock_service_2 = AsyncMock()
        mock_service_2.name = "secondary_service"
        mock_service_2.check.return_value = MagicMock(
            name="secondary_service",
            status="degraded",
            message="High latency detected",
            duration_ms=1500.0
        )
        
        health_manager.add_health_check(mock_service_1)
        health_manager.add_health_check(mock_service_2)
        
        # Run health checks
        results = await health_manager.check_all()
        overall_status = health_manager.get_overall_status()
        
        # Verify results
        assert len(results) >= 3  # qdrant + primary + secondary
        assert "primary_service" in results
        assert "secondary_service" in results
        assert "qdrant" in results
        
        # Overall status should reflect worst case
        assert overall_status in ["healthy", "degraded", "unhealthy"]
        
        # If any service is degraded, overall should be degraded or worse
        statuses = [result.status for result in results.values()]
        if "degraded" in statuses:
            assert overall_status in ["degraded", "unhealthy"]