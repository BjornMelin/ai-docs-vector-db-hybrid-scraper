"""Tests for observability dependencies module.

Tests FastAPI dependency injection patterns for observability services,
including setup, initialization, and lifecycle management.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from src.services.observability.dependencies import (
    get_observability_config,
    get_observability_service,
    get_ai_tracer,
    get_service_meter,
    create_span_context,
    record_ai_operation_metrics,
    track_ai_cost_metrics,
    get_observability_health,
    ObservabilityConfigDep,
    ObservabilityServiceDep,
    AITracerDep,
    ServiceMeterDep,
)
from src.services.observability.config import ObservabilityConfig


class TestObservabilityServiceDependency:
    """Test main observability service dependency."""

    @patch('src.services.observability.dependencies.get_observability_config')
    @patch('src.services.observability.dependencies.is_observability_enabled')
    @patch('src.services.observability.dependencies.get_tracer')
    @patch('src.services.observability.dependencies.get_meter')
    def test_get_observability_service_enabled(self, mock_get_meter, mock_get_tracer, 
                                               mock_is_enabled, mock_get_config):
        """Test observability service when enabled and initialized."""
        # Clear cache
        get_observability_service.cache_clear()
        
        mock_config = ObservabilityConfig(enabled=True)
        mock_get_config.return_value = mock_config
        mock_is_enabled.return_value = True
        
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        
        service = get_observability_service()
        
        assert service["config"] == mock_config
        assert service["tracer"] == mock_tracer
        assert service["meter"] == mock_meter
        assert service["enabled"] is True
        
        mock_get_tracer.assert_called_once_with("ai-docs-service")
        mock_get_meter.assert_called_once_with("ai-docs-service")

    @patch('src.services.observability.dependencies.get_observability_config')
    @patch('src.services.observability.dependencies.is_observability_enabled')
    @patch('src.services.observability.dependencies.initialize_observability')
    @patch('src.services.observability.dependencies.get_tracer')
    @patch('src.services.observability.dependencies.get_meter')
    def test_get_observability_service_initialization(self, mock_get_meter, mock_get_tracer,
                                                     mock_initialize, mock_is_enabled, mock_get_config):
        """Test observability service initialization when not yet enabled."""
        # Clear cache
        get_observability_service.cache_clear()
        
        mock_config = ObservabilityConfig(enabled=True)
        mock_get_config.return_value = mock_config
        mock_is_enabled.return_value = False  # Not yet enabled
        mock_initialize.return_value = True
        
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        
        service = get_observability_service()
        
        # Should call initialize_observability
        mock_initialize.assert_called_once_with(mock_config)
        assert service["config"] == mock_config
        assert service["tracer"] == mock_tracer
        assert service["meter"] == mock_meter

    @patch('src.services.observability.dependencies.get_observability_config')
    @patch('src.services.observability.dependencies.is_observability_enabled')
    def test_get_observability_service_disabled(self, mock_is_enabled, mock_get_config):
        """Test observability service when disabled."""
        # Clear cache
        get_observability_service.cache_clear()
        
        mock_config = ObservabilityConfig(enabled=False)
        mock_get_config.return_value = mock_config
        mock_is_enabled.return_value = False
        
        service = get_observability_service()
        
        assert service["config"] == mock_config
        assert service["enabled"] is False

    @patch('src.services.observability.dependencies.get_observability_config')
    def test_get_observability_service_exception_handling(self, mock_get_config):
        """Test observability service exception handling."""
        # Clear cache
        get_observability_service.cache_clear()
        
        mock_get_config.side_effect = Exception("Config error")
        
        service = get_observability_service()
        
        assert isinstance(service["config"], ObservabilityConfig)
        assert service["tracer"] is None
        assert service["meter"] is None
        assert service["enabled"] is False

    def test_get_observability_service_caching(self):
        """Test that observability service is cached."""
        # Clear cache first
        get_observability_service.cache_clear()
        
        with patch('src.services.observability.dependencies.get_observability_config') as mock_get_config:
            mock_config = ObservabilityConfig(enabled=False)
            mock_get_config.return_value = mock_config
            
            service1 = get_observability_service()
            service2 = get_observability_service()
            
            # Should be the same cached instance
            assert service1 is service2
            # Config should only be called once due to caching
            mock_get_config.assert_called_once()


class TestAITracerDependency:
    """Test AI tracer dependency."""

    def test_get_ai_tracer_enabled(self):
        """Test AI tracer when observability is enabled."""
        mock_service = {
            "enabled": True,
            "tracer": Mock(),
        }
        
        with patch('src.services.observability.dependencies.get_tracer') as mock_get_tracer:
            mock_tracer = Mock()
            mock_get_tracer.return_value = mock_tracer
            
            tracer = get_ai_tracer(mock_service)
            
            assert tracer == mock_tracer
            mock_get_tracer.assert_called_once_with("ai-operations")

    def test_get_ai_tracer_disabled(self):
        """Test AI tracer when observability is disabled."""
        mock_service = {
            "enabled": False,
            "tracer": None,
        }
        
        with patch('src.services.observability.tracking._NoOpTracer') as mock_noop:
            mock_noop_instance = Mock()
            mock_noop.return_value = mock_noop_instance
            
            tracer = get_ai_tracer(mock_service)
            
            assert tracer == mock_noop_instance

    def test_get_ai_tracer_no_tracer(self):
        """Test AI tracer when enabled but no tracer available."""
        mock_service = {
            "enabled": True,
            "tracer": None,
        }
        
        with patch('src.services.observability.tracking._NoOpTracer') as mock_noop:
            mock_noop_instance = Mock()
            mock_noop.return_value = mock_noop_instance
            
            tracer = get_ai_tracer(mock_service)
            
            assert tracer == mock_noop_instance


class TestServiceMeterDependency:
    """Test service meter dependency."""

    def test_get_service_meter_enabled(self):
        """Test service meter when observability is enabled."""
        mock_service = {
            "enabled": True,
            "meter": Mock(),
        }
        
        with patch('src.services.observability.dependencies.get_meter') as mock_get_meter:
            mock_meter = Mock()
            mock_get_meter.return_value = mock_meter
            
            meter = get_service_meter(mock_service)
            
            assert meter == mock_meter
            mock_get_meter.assert_called_once_with("service-metrics")

    def test_get_service_meter_disabled(self):
        """Test service meter when observability is disabled."""
        mock_service = {
            "enabled": False,
            "meter": None,
        }
        
        with patch('src.services.observability.tracking._NoOpMeter') as mock_noop:
            mock_noop_instance = Mock()
            mock_noop.return_value = mock_noop_instance
            
            meter = get_service_meter(mock_service)
            
            assert meter == mock_noop_instance

    def test_get_service_meter_no_meter(self):
        """Test service meter when enabled but no meter available."""
        mock_service = {
            "enabled": True,
            "meter": None,
        }
        
        with patch('src.services.observability.tracking._NoOpMeter') as mock_noop:
            mock_noop_instance = Mock()
            mock_noop.return_value = mock_noop_instance
            
            meter = get_service_meter(mock_service)
            
            assert meter == mock_noop_instance


class TestSpanContext:
    """Test span context creation."""

    def test_create_span_context(self):
        """Test span context creation."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value = mock_span
        
        operation_name = "test_operation"
        context = create_span_context(operation_name, mock_tracer)
        
        assert context == mock_span
        mock_tracer.start_as_current_span.assert_called_once_with(operation_name)

    def test_create_span_context_with_complex_name(self):
        """Test span context creation with complex operation name."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value = mock_span
        
        operation_name = "ai.embedding.generate_batch"
        context = create_span_context(operation_name, mock_tracer)
        
        assert context == mock_span
        mock_tracer.start_as_current_span.assert_called_once_with(operation_name)


class TestAIOperationMetrics:
    """Test AI operation metrics recording."""

    @pytest.mark.asyncio
    async def test_record_ai_operation_metrics_success(self):
        """Test successful AI operation metrics recording."""
        mock_meter = Mock()
        
        with patch('src.services.observability.tracking.record_ai_operation') as mock_record:
            await record_ai_operation_metrics(
                operation_type="embedding",
                provider="openai",
                success=True,
                duration=0.5,
                meter=mock_meter,
                model="text-embedding-3-small",
                tokens=100
            )
            
            mock_record.assert_called_once_with(
                operation_type="embedding",
                provider="openai",
                success=True,
                duration=0.5,
                model="text-embedding-3-small",
                tokens=100
            )

    @pytest.mark.asyncio
    async def test_record_ai_operation_metrics_failure(self):
        """Test AI operation metrics recording with failure."""
        mock_meter = Mock()
        
        with patch('src.services.observability.tracking.record_ai_operation') as mock_record:
            await record_ai_operation_metrics(
                operation_type="chat",
                provider="anthropic",
                success=False,
                duration=1.2,
                meter=mock_meter,
                error="rate_limit"
            )
            
            mock_record.assert_called_once_with(
                operation_type="chat",
                provider="anthropic",
                success=False,
                duration=1.2,
                error="rate_limit"
            )

    @pytest.mark.asyncio
    async def test_record_ai_operation_metrics_exception(self):
        """Test AI operation metrics recording with exception."""
        mock_meter = Mock()
        
        with patch('src.services.observability.tracking.record_ai_operation') as mock_record:
            mock_record.side_effect = Exception("Metrics recording failed")
            
            # Should not raise exception
            await record_ai_operation_metrics(
                operation_type="embedding",
                provider="openai",
                success=True,
                duration=0.5,
                meter=mock_meter
            )
            
            mock_record.assert_called_once()


class TestAICostTracking:
    """Test AI cost tracking."""

    @pytest.mark.asyncio
    async def test_track_ai_cost_metrics_success(self):
        """Test successful AI cost tracking."""
        mock_meter = Mock()
        
        with patch('src.services.observability.tracking.track_cost') as mock_track:
            await track_ai_cost_metrics(
                operation_type="embedding",
                provider="openai",
                cost_usd=0.001,
                meter=mock_meter,
                model="text-embedding-3-small",
                tokens=100
            )
            
            mock_track.assert_called_once_with(
                operation_type="embedding",
                provider="openai",
                cost_usd=0.001,
                model="text-embedding-3-small",
                tokens=100
            )

    @pytest.mark.asyncio
    async def test_track_ai_cost_metrics_zero_cost(self):
        """Test AI cost tracking with zero cost."""
        mock_meter = Mock()
        
        with patch('src.services.observability.tracking.track_cost') as mock_track:
            await track_ai_cost_metrics(
                operation_type="embedding",
                provider="local",
                cost_usd=0.0,
                meter=mock_meter
            )
            
            mock_track.assert_called_once_with(
                operation_type="embedding",
                provider="local",
                cost_usd=0.0
            )

    @pytest.mark.asyncio
    async def test_track_ai_cost_metrics_exception(self):
        """Test AI cost tracking with exception."""
        mock_meter = Mock()
        
        with patch('src.services.observability.tracking.track_cost') as mock_track:
            mock_track.side_effect = Exception("Cost tracking failed")
            
            # Should not raise exception
            await track_ai_cost_metrics(
                operation_type="embedding",
                provider="openai",
                cost_usd=0.001,
                meter=mock_meter
            )
            
            mock_track.assert_called_once()


class TestObservabilityHealth:
    """Test observability health status."""

    @pytest.mark.asyncio
    async def test_get_observability_health_enabled(self):
        """Test observability health when enabled."""
        mock_config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            otlp_endpoint="http://localhost:4317",
            instrument_fastapi=True,
            instrument_httpx=True,
            instrument_redis=False,
            instrument_sqlalchemy=True,
            track_ai_operations=True,
            track_costs=True
        )
        
        mock_service = {
            "config": mock_config,
            "enabled": True
        }
        
        health = await get_observability_health(mock_service)
        
        assert health["enabled"] is True
        assert health["service_name"] == "test-service"
        assert health["otlp_endpoint"] == "http://localhost:4317"
        assert health["instrumentation"]["fastapi"] is True
        assert health["instrumentation"]["httpx"] is True
        assert health["instrumentation"]["redis"] is False
        assert health["instrumentation"]["sqlalchemy"] is True
        assert health["ai_tracking"]["operations"] is True
        assert health["ai_tracking"]["costs"] is True
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_observability_health_disabled(self):
        """Test observability health when disabled."""
        mock_config = ObservabilityConfig(enabled=False)
        
        mock_service = {
            "config": mock_config,
            "enabled": False
        }
        
        health = await get_observability_health(mock_service)
        
        assert health["enabled"] is False
        assert health["otlp_endpoint"] is None
        assert health["instrumentation"]["fastapi"] is False
        assert health["instrumentation"]["httpx"] is False
        assert health["instrumentation"]["redis"] is False
        assert health["instrumentation"]["sqlalchemy"] is False
        assert health["ai_tracking"]["operations"] is False
        assert health["ai_tracking"]["costs"] is False
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_get_observability_health_exception(self):
        """Test observability health with exception."""
        mock_service = {}  # Missing required keys
        
        health = await get_observability_health(mock_service)
        
        assert health["enabled"] is False
        assert health["status"] == "error"
        assert "error" in health


class TestFastAPIIntegration:
    """Test FastAPI dependency injection integration."""

    def test_observability_config_dependency_annotation(self):
        """Test ObservabilityConfigDep annotation."""
        app = FastAPI()
        
        @app.get("/config")
        async def get_config_endpoint(config: ObservabilityConfigDep):
            return {"service_name": config.service_name}
        
        with TestClient(app) as client:
            response = client.get("/config")
            assert response.status_code == 200
            assert "service_name" in response.json()

    def test_observability_service_dependency_annotation(self):
        """Test ObservabilityServiceDep annotation."""
        app = FastAPI()
        
        @app.get("/service")
        async def get_service_endpoint(service: ObservabilityServiceDep):
            return {"enabled": service["enabled"]}
        
        with TestClient(app) as client:
            response = client.get("/service")
            assert response.status_code == 200
            assert "enabled" in response.json()

    def test_ai_tracer_dependency_annotation(self):
        """Test AITracerDep annotation."""
        app = FastAPI()
        
        @app.get("/tracer")
        async def get_tracer_endpoint(tracer: AITracerDep):
            return {"tracer_available": tracer is not None}
        
        with TestClient(app) as client:
            response = client.get("/tracer")
            assert response.status_code == 200
            assert "tracer_available" in response.json()

    def test_service_meter_dependency_annotation(self):
        """Test ServiceMeterDep annotation."""
        app = FastAPI()
        
        @app.get("/meter")
        async def get_meter_endpoint(meter: ServiceMeterDep):
            return {"meter_available": meter is not None}
        
        with TestClient(app) as client:
            response = client.get("/meter")
            assert response.status_code == 200
            assert "meter_available" in response.json()

    def test_full_dependency_chain(self):
        """Test full dependency chain in FastAPI endpoint."""
        app = FastAPI()
        
        @app.post("/ai/embed")
        async def embed_endpoint(
            config: ObservabilityConfigDep,
            service: ObservabilityServiceDep,
            tracer: AITracerDep,
            meter: ServiceMeterDep,
        ):
            with create_span_context("embedding_operation", tracer):
                await record_ai_operation_metrics(
                    operation_type="embedding",
                    provider="openai",
                    success=True,
                    duration=0.5,
                    meter=meter
                )
                return {
                    "success": True,
                    "observability_enabled": service["enabled"],
                    "service_name": config.service_name
                }
        
        with TestClient(app) as client:
            response = client.post("/ai/embed")
            assert response.status_code == 200
            data = response.json()
            assert "success" in data
            assert "observability_enabled" in data
            assert "service_name" in data

    def test_health_endpoint_integration(self):
        """Test health endpoint with observability dependencies."""
        app = FastAPI()
        
        @app.get("/health/observability")
        async def observability_health_endpoint(
            service: ObservabilityServiceDep,
        ):
            return await get_observability_health(service)
        
        with TestClient(app) as client:
            response = client.get("/health/observability")
            assert response.status_code == 200
            data = response.json()
            assert "enabled" in data
            assert "status" in data


class TestDependencyErrorHandling:
    """Test dependency error handling and resilience."""

    def test_dependency_resilience_with_service_errors(self):
        """Test that dependencies are resilient to service errors."""
        # Clear cache
        get_observability_service.cache_clear()
        
        with patch('src.services.observability.dependencies.get_observability_config') as mock_get_config:
            mock_get_config.side_effect = Exception("Service unavailable")
            
            # Should not raise exception
            service = get_observability_service()
            
            assert service["enabled"] is False
            assert service["tracer"] is None
            assert service["meter"] is None

    def test_ai_tracer_with_get_tracer_exception(self):
        """Test AI tracer behavior when get_tracer raises exception."""
        mock_service = {
            "enabled": True,
            "tracer": Mock(),
        }
        
        with patch('src.services.observability.dependencies.get_tracer') as mock_get_tracer:
            mock_get_tracer.side_effect = Exception("Tracer creation failed")
            
            # Function doesn't handle exceptions from get_tracer, so it should propagate
            with pytest.raises(Exception, match="Tracer creation failed"):
                get_ai_tracer(mock_service)

    @pytest.mark.asyncio
    async def test_metrics_recording_resilience(self):
        """Test metrics recording resilience to errors."""
        mock_meter = Mock()
        
        # Should not raise exception even if tracking fails
        await record_ai_operation_metrics(
            operation_type="embedding",
            provider="openai",
            success=True,
            duration=0.5,
            meter=mock_meter
        )
        
        await track_ai_cost_metrics(
            operation_type="embedding",
            provider="openai",
            cost_usd=0.001,
            meter=mock_meter
        )


class TestDependencyPerformance:
    """Test dependency performance characteristics."""

    def test_service_caching_performance(self):
        """Test that service caching improves performance."""
        # Clear cache
        get_observability_service.cache_clear()
        
        with patch('src.services.observability.dependencies.get_observability_config') as mock_get_config:
            mock_config = ObservabilityConfig(enabled=False)
            mock_get_config.return_value = mock_config
            
            # First call
            service1 = get_observability_service()
            call_count_1 = mock_get_config.call_count
            
            # Second call should use cache
            service2 = get_observability_service()
            call_count_2 = mock_get_config.call_count
            
            # Should be same object and no additional config calls
            assert service1 is service2
            assert call_count_1 == call_count_2

    def test_noop_objects_lightweight(self):
        """Test that NoOp objects are lightweight."""
        mock_service = {
            "enabled": False,
            "tracer": None,
            "meter": None,
        }
        
        with patch('src.services.observability.tracking._NoOpTracer') as mock_noop_tracer, \
             patch('src.services.observability.tracking._NoOpMeter') as mock_noop_meter:
            
            mock_noop_tracer.return_value = Mock()
            mock_noop_meter.return_value = Mock()
            
            tracer = get_ai_tracer(mock_service)
            meter = get_service_meter(mock_service)
            
            # Should create NoOp objects efficiently
            mock_noop_tracer.assert_called_once()
            mock_noop_meter.assert_called_once()
            assert tracer is not None
            assert meter is not None