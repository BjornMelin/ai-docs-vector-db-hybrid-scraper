"""Tests for monitoring middleware functionality."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.testclient import TestClient
from src.services.monitoring.health import HealthCheckConfig
from src.services.monitoring.health import HealthCheckManager
from src.services.monitoring.metrics import MetricsConfig
from src.services.monitoring.metrics import MetricsRegistry
from src.services.monitoring.middleware import PrometheusMiddleware


class TestPrometheusMiddleware:
    """Test Prometheus middleware functionality."""

    @pytest.fixture
    def metrics_config(self):
        """Create metrics configuration."""
        return MetricsConfig(enabled=True)

    @pytest.fixture
    def health_config(self):
        """Create health check configuration."""
        return HealthCheckConfig(enabled=True)

    @pytest.fixture
    def metrics_registry(self, metrics_config):
        """Create metrics registry."""
        return MetricsRegistry(metrics_config)

    @pytest.fixture
    def health_manager(self, health_config):
        """Create health check manager."""
        return HealthCheckManager(health_config)

    @pytest.fixture
    def app(self, metrics_registry, health_manager):
        """Create FastAPI app with middleware."""
        app = FastAPI()

        # Add middleware
        middleware = PrometheusMiddleware(
            metrics_registry=metrics_registry,
            health_manager=health_manager
        )
        app.add_middleware(PrometheusMiddleware,
                          metrics_registry=metrics_registry,
                          health_manager=health_manager)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        # Check for expected metric names
        content = response.text
        assert "ml_app_" in content

    def test_health_endpoint(self, client):
        """Test basic health endpoint."""
        with patch.object(client.app.user_middleware[0].cls, 'health_manager') as mock_manager:
            mock_manager.get_overall_health = AsyncMock(return_value=(
                "healthy",
                {"overall_status": "healthy", "services": {}}
            ))

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_health_live_endpoint(self, client):
        """Test liveness endpoint."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_health_ready_endpoint(self, client):
        """Test readiness endpoint."""
        with patch.object(client.app.user_middleware[0].cls, 'health_manager') as mock_manager:
            mock_manager.get_overall_health = AsyncMock(return_value=(
                "healthy",
                {"overall_status": "healthy", "services": {}}
            ))

            response = client.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    def test_health_ready_endpoint_unhealthy(self, client):
        """Test readiness endpoint when unhealthy."""
        with patch.object(client.app.user_middleware[0].cls, 'health_manager') as mock_manager:
            mock_manager.get_overall_health = AsyncMock(return_value=(
                "unhealthy",
                {"overall_status": "unhealthy", "services": {"qdrant": {"status": "unhealthy"}}}
            ))

            response = client.get("/health/ready")

            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not_ready"

    def test_request_metrics_collection(self, client):
        """Test that request metrics are collected."""
        # Make a successful request
        response = client.get("/test")
        assert response.status_code == 200

        # Check metrics endpoint includes request data
        metrics_response = client.get("/metrics")
        content = metrics_response.text

        # Should include HTTP request metrics from prometheus-fastapi-instrumentator
        assert "http_requests_total" in content or "http_request" in content

    def test_error_handling_in_middleware(self, client):
        """Test middleware handles errors gracefully."""
        # Make request to error endpoint
        response = client.get("/error")
        assert response.status_code == 500

        # Metrics endpoint should still work
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200

    def test_middleware_with_disabled_metrics(self):
        """Test middleware behavior with disabled metrics."""
        disabled_config = MetricsConfig(enabled=False)
        disabled_registry = MetricsRegistry(disabled_config)

        app = FastAPI()
        middleware = PrometheusMiddleware(
            metrics_registry=disabled_registry,
            health_manager=None
        )

        # Should not raise errors
        assert middleware.metrics_registry._metrics == {}

    @pytest.mark.asyncio
    async def test_middleware_dispatch(self, metrics_registry, health_manager):
        """Test middleware dispatch method."""
        middleware = PrometheusMiddleware(
            metrics_registry=metrics_registry,
            health_manager=health_manager
        )

        # Mock request and call next
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"

        call_next = AsyncMock(return_value=Response(content="test", status_code=200))

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 200
        call_next.assert_called_once_with(request)

    def test_custom_metrics_path(self):
        """Test custom metrics path configuration."""
        config = MetricsConfig(path="/custom-metrics")
        registry = MetricsRegistry(config)

        app = FastAPI()
        app.add_middleware(PrometheusMiddleware,
                          metrics_registry=registry,
                          health_manager=None)

        client = TestClient(app)

        # Default path should not work
        response = client.get("/metrics")
        assert response.status_code == 404

        # Custom path should work
        response = client.get("/custom-metrics")
        assert response.status_code == 200


class TestMiddlewareIntegration:
    """Integration tests for middleware with full application."""

    @pytest.fixture
    def full_app(self):
        """Create full application with all middleware."""
        app = FastAPI(title="Test ML App")

        # Create monitoring components
        metrics_config = MetricsConfig(enabled=True)
        health_config = HealthCheckConfig(enabled=True)

        metrics_registry = MetricsRegistry(metrics_config)
        health_manager = HealthCheckManager(health_config)

        # Add middleware
        app.add_middleware(PrometheusMiddleware,
                          metrics_registry=metrics_registry,
                          health_manager=health_manager)

        # Add some test routes
        @app.get("/")
        async def root():
            return {"message": "Hello World"}

        @app.get("/search")
        async def search():
            # Simulate search operation
            return {"results": []}

        @app.post("/embeddings")
        async def embeddings():
            # Simulate embedding generation
            return {"embeddings": []}

        return app

    def test_full_application_monitoring(self, full_app):
        """Test monitoring with full application simulation."""
        client = TestClient(full_app)

        # Make various requests
        client.get("/")
        client.get("/search")
        client.post("/embeddings")
        client.get("/nonexistent")  # 404

        # Check metrics collection
        response = client.get("/metrics")
        assert response.status_code == 200

        content = response.text
        # Should have HTTP metrics
        assert "http" in content.lower()

        # Check health endpoints
        health_response = client.get("/health")
        assert health_response.status_code == 200

        live_response = client.get("/health/live")
        assert live_response.status_code == 200

    def test_concurrent_requests_monitoring(self, full_app):
        """Test monitoring under concurrent load."""
        import threading
        import time

        client = TestClient(full_app)
        results = []

        def make_requests():
            for _ in range(10):
                response = client.get("/search")
                results.append(response.status_code)
                time.sleep(0.01)

        # Create multiple threads
        threads = [threading.Thread(target=make_requests) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)

        # Metrics should still be accessible
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200

    def test_metrics_persistence_across_requests(self, full_app):
        """Test that metrics persist and accumulate across requests."""
        client = TestClient(full_app)

        # Make initial requests
        for _ in range(5):
            client.get("/search")

        metrics_1 = client.get("/metrics").text

        # Make more requests
        for _ in range(3):
            client.get("/search")

        metrics_2 = client.get("/metrics").text

        # Metrics should have accumulated
        assert len(metrics_2) >= len(metrics_1)

    def test_health_check_integration(self, full_app):
        """Test health check integration with real dependencies."""
        client = TestClient(full_app)

        # Mock health manager for consistent testing
        with patch.object(full_app.user_middleware[0].cls, 'health_manager') as mock_manager:
            # Test healthy state
            mock_manager.get_overall_health = AsyncMock(return_value=(
                "healthy",
                {
                    "overall_status": "healthy",
                    "services": {
                        "qdrant": {"status": "healthy"},
                        "redis": {"status": "healthy"}
                    }
                }
            ))

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

            # Test degraded state
            mock_manager.get_overall_health = AsyncMock(return_value=(
                "degraded",
                {
                    "overall_status": "degraded",
                    "services": {
                        "qdrant": {"status": "healthy"},
                        "redis": {"status": "degraded"}
                    }
                }
            ))

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
