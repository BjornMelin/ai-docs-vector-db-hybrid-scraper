"""Comprehensive tests for monitoring middleware functionality."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from src.services.monitoring.middleware import (
    CustomMetricsMiddleware,
    PrometheusMiddleware,
)


class TestPrometheusMiddleware:
    """Test PrometheusMiddleware functionality."""

    @pytest.fixture
    def mock_metrics_registry(self):
        """Create mock metrics registry."""
        registry = MagicMock()
        registry.get_prometheus_registry.return_value = MagicMock()
        return registry

    @pytest.fixture
    def mock_health_manager(self):
        """Create mock health manager."""
        manager = MagicMock()
        manager.check_all = AsyncMock(return_value={
            "qdrant": MagicMock(status="healthy", message="OK", duration_ms=50.0),
            "redis": MagicMock(status="healthy", message="OK", duration_ms=25.0),
        })
        manager.get_overall_status.return_value = "healthy"
        return manager

    def test_prometheus_middleware_initialization(self, mock_metrics_registry, mock_health_manager):
        """Test PrometheusMiddleware initialization."""
        app = FastAPI()
        middleware = PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        assert middleware.app is app
        assert middleware.metrics_registry is mock_metrics_registry
        assert middleware.health_manager is mock_health_manager

    def test_prometheus_middleware_endpoints_added(self, mock_metrics_registry, mock_health_manager):
        """Test that PrometheusMiddleware adds endpoints."""
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        # Check that endpoints were added
        routes = [route.path for route in app.routes]
        assert "/metrics" in routes
        assert "/health" in routes
        assert "/health/live" in routes
        assert "/health/ready" in routes

    def test_prometheus_middleware_without_health_manager(self, mock_metrics_registry):
        """Test PrometheusMiddleware without health manager."""
        app = FastAPI()
        middleware = PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=None,
        )
        
        assert middleware.health_manager is None
        # Should still add metrics endpoint
        routes = [route.path for route in app.routes]
        assert "/metrics" in routes

    def test_health_endpoint_functionality(self, mock_metrics_registry, mock_health_manager):
        """Test health endpoint functionality."""
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_metrics_endpoint_functionality(self, mock_metrics_registry, mock_health_manager):
        """Test metrics endpoint functionality."""
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_liveness_endpoint(self, mock_metrics_registry, mock_health_manager):
        """Test liveness endpoint."""
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        client = TestClient(app)
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_endpoint(self, mock_metrics_registry, mock_health_manager):
        """Test readiness endpoint."""
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        client = TestClient(app)
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_unhealthy_service_response(self, mock_metrics_registry, mock_health_manager):
        """Test response when services are unhealthy."""
        # Configure unhealthy service
        mock_health_manager.check_all = AsyncMock(return_value={
            "qdrant": MagicMock(
                status="unhealthy",
                message="Connection failed",
                duration_ms=5000.0,
            ),
        })
        mock_health_manager.get_overall_status.return_value = "unhealthy"
        
        app = FastAPI()
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 503  # Service Unavailable
        data = response.json()
        assert data["status"] == "unhealthy"


class TestCustomMetricsMiddleware:
    """Test CustomMetricsMiddleware functionality."""

    @pytest.fixture
    def mock_metrics_registry(self):
        """Create mock metrics registry."""
        registry = MagicMock()
        registry.record_request = MagicMock()
        registry.record_response = MagicMock()
        registry.record_error = MagicMock()
        return registry

    @pytest.fixture
    def app_with_custom_middleware(self, mock_metrics_registry):
        """Create FastAPI app with custom metrics middleware."""
        app = FastAPI()
        app.add_middleware(CustomMetricsMiddleware, metrics_registry=mock_metrics_registry)
        
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
        
        @app.get("/error")
        async def error_endpoint():
            raise HTTPException(status_code=500, detail="Test error")
        
        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(0.01)
            return {"message": "Slow response"}
        
        return app

    def test_custom_middleware_initialization(self, mock_metrics_registry):
        """Test CustomMetricsMiddleware initialization."""
        app = FastAPI()
        middleware = CustomMetricsMiddleware(app, metrics_registry=mock_metrics_registry)
        
        assert middleware.metrics_registry is mock_metrics_registry

    def test_successful_request_monitoring(self, app_with_custom_middleware, mock_metrics_registry):
        """Test monitoring of successful requests."""
        client = TestClient(app_with_custom_middleware)
        
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}

    def test_error_request_monitoring(self, app_with_custom_middleware, mock_metrics_registry):
        """Test monitoring of error requests."""
        client = TestClient(app_with_custom_middleware)
        
        response = client.get("/error")
        
        assert response.status_code == 500

    def test_slow_request_monitoring(self, app_with_custom_middleware, mock_metrics_registry):
        """Test monitoring of slow requests."""
        client = TestClient(app_with_custom_middleware)
        
        start_time = time.time()
        response = client.get("/slow")
        end_time = time.time()
        
        assert response.status_code == 200
        assert response.json() == {"message": "Slow response"}
        
        # Verify request took some time
        duration = end_time - start_time
        assert duration >= 0.01  # At least 10ms

    def test_middleware_without_metrics_registry(self):
        """Test middleware functionality without metrics registry."""
        app = FastAPI()
        middleware = CustomMetricsMiddleware(app, metrics_registry=None)
        
        assert middleware.metrics_registry is None
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Should still work without metrics
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMiddlewareIntegration:
    """Test middleware integration scenarios."""

    @pytest.fixture
    def full_monitoring_app(self, mock_metrics_registry, mock_health_manager):
        """Create app with full monitoring integration."""
        app = FastAPI()
        
        # Add PrometheusMiddleware for endpoints
        PrometheusMiddleware(
            app=app,
            metrics_registry=mock_metrics_registry,
            health_manager=mock_health_manager,
        )
        
        # Add CustomMetricsMiddleware for request monitoring
        app.add_middleware(CustomMetricsMiddleware, metrics_registry=mock_metrics_registry)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        @app.post("/api/data")
        async def data_endpoint(data: dict):
            return {"received": data}
        
        return app

    @pytest.fixture
    def mock_metrics_registry(self):
        """Create mock metrics registry."""
        registry = MagicMock()
        registry.get_prometheus_registry.return_value = MagicMock()
        return registry

    @pytest.fixture
    def mock_health_manager(self):
        """Create mock health manager."""
        manager = MagicMock()
        manager.check_all = AsyncMock(return_value={
            "qdrant": MagicMock(status="healthy", message="OK", duration_ms=50.0),
        })
        manager.get_overall_status.return_value = "healthy"
        return manager

    def test_full_monitoring_integration(self, full_monitoring_app, mock_health_manager):
        """Test complete monitoring integration."""
        client = TestClient(full_monitoring_app)
        
        # Test API endpoint
        response = client.get("/api/test")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Verify health checks were called
        mock_health_manager.check_all.assert_called()

    def test_multiple_request_monitoring(self, full_monitoring_app):
        """Test monitoring across multiple requests."""
        client = TestClient(full_monitoring_app)
        
        # Make multiple requests
        for i in range(5):
            response = client.get("/api/test")
            assert response.status_code == 200
        
        # POST request
        response = client.post("/api/data", json={"test": "data"})
        assert response.status_code == 200

    def test_concurrent_request_monitoring(self, full_monitoring_app):
        """Test monitoring under concurrent load."""
        import threading
        
        client = TestClient(full_monitoring_app)
        results = []
        
        def make_request():
            response = client.get("/api/test")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        assert all(status == 200 for status in results)
        assert len(results) == 10