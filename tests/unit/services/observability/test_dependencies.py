"""Tests for observability dependency injection."""

from unittest.mock import MagicMock, patch

import pytest

from src.services.observability.config import ObservabilityConfig
from src.services.observability.dependencies import (
    create_span_context,
    get_ai_tracer,
    get_observability_health,
    get_observability_service,
    get_service_meter,
    record_ai_operation_metrics,
    track_ai_cost_metrics,
)
from src.services.observability.tracking import _NoOpMeter, _NoOpTracer


class TestObservabilityDependencies:
    """Test observability dependency injection functions."""

    def setup_method(self):
        """Setup for each test."""
        # Clear cache for get_observability_service
        get_observability_service.cache_clear()

    @patch("src.services.observability.dependencies.get_observability_config")
    @patch("src.services.observability.dependencies.is_observability_enabled")
    @patch("src.services.observability.dependencies.initialize_observability")
    @patch("src.services.observability.dependencies.get_tracer")
    @patch("src.services.observability.dependencies.get_meter")
    def test_get_observability_service_enabled(
        self,
        mock_get_meter,
        mock_get_tracer,
        mock_initialize,
        mock_is_enabled,
        mock_get_config,
    ):
        """Test getting observability service when enabled."""
        # Setup mocks
        config = ObservabilityConfig(enabled=True, service_name="test-service")
        mock_get_config.return_value = config
        mock_is_enabled.return_value = False  # Not initialized yet
        mock_initialize.return_value = True

        mock_tracer = MagicMock()
        mock_meter = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter

        # Clear cache to ensure fresh call
        get_observability_service.cache_clear()

        service = get_observability_service()

        assert service["config"] == config
        assert service["tracer"] == mock_tracer
        assert service["meter"] == mock_meter
        # Note: The enabled flag is determined by is_observability_enabled(),
        # which after init should return True
        mock_is_enabled.return_value = True  # After initialization
        get_observability_service.cache_clear()  # Clear cache to get updated state
        service = get_observability_service()
        assert service["enabled"] is True

        # Verify initialization was called
        mock_initialize.assert_called_once_with(config)

    @patch("src.services.observability.dependencies.get_observability_config")
    @patch("src.services.observability.dependencies.is_observability_enabled")
    @patch("src.services.observability.dependencies.get_tracer")
    @patch("src.services.observability.dependencies.get_meter")
    def test_get_observability_service_already_enabled(
        self,
        mock_get_meter,
        mock_get_tracer,
        mock_is_enabled,
        mock_get_config,
    ):
        """Test getting observability service when already enabled."""
        config = ObservabilityConfig(enabled=True, service_name="test-service")
        mock_get_config.return_value = config
        mock_is_enabled.return_value = True  # Already initialized

        mock_tracer = MagicMock()
        mock_meter = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter

        get_observability_service.cache_clear()

        service = get_observability_service()

        assert service["enabled"] is True
        assert service["tracer"] == mock_tracer
        assert service["meter"] == mock_meter

    @patch("src.services.observability.dependencies.get_observability_config")
    def test_get_observability_service_disabled(self, mock_get_config):
        """Test getting observability service when disabled."""
        config = ObservabilityConfig(enabled=False)
        mock_get_config.return_value = config

        get_observability_service.cache_clear()

        service = get_observability_service()

        assert service["enabled"] is False
        # When disabled, we get NoOp implementations, not None

        assert isinstance(service["tracer"], _NoOpTracer)
        assert isinstance(service["meter"], _NoOpMeter)

    @patch("src.services.observability.dependencies.get_observability_config")
    def test_get_observability_service_exception_handling(self, mock_get_config):
        """Test observability service exception handling."""
        mock_get_config.side_effect = Exception("Config error")

        get_observability_service.cache_clear()

        service = get_observability_service()

        assert service["enabled"] is False
        assert service["tracer"] is None
        assert service["meter"] is None
        assert isinstance(service["config"], ObservabilityConfig)

    @patch("src.services.observability.dependencies.get_tracer")
    def test_get_ai_tracer_enabled(self, mock_get_tracer):
        """Test getting AI tracer when enabled."""
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        observability_service = {
            "enabled": True,
            "tracer": mock_tracer,
        }

        tracer = get_ai_tracer(observability_service)

        assert tracer == mock_tracer
        mock_get_tracer.assert_called_once_with("ai-operations")

    def test_get_ai_tracer_disabled(self):
        """Test getting AI tracer when disabled."""
        observability_service = {
            "enabled": False,
            "tracer": None,
        }

        tracer = get_ai_tracer(observability_service)

        # Should return NoOp tracer

        assert isinstance(tracer, _NoOpTracer)

    def test_get_ai_tracer_no_tracer(self):
        """Test getting AI tracer when enabled but no tracer available."""
        observability_service = {
            "enabled": True,
            "tracer": None,
        }

        tracer = get_ai_tracer(observability_service)

        assert isinstance(tracer, _NoOpTracer)

    @patch("src.services.observability.dependencies.get_meter")
    def test_get_service_meter_enabled(self, mock_get_meter):
        """Test getting service meter when enabled."""
        mock_meter = MagicMock()
        mock_get_meter.return_value = mock_meter

        observability_service = {
            "enabled": True,
            "meter": mock_meter,
        }

        meter = get_service_meter(observability_service)

        assert meter == mock_meter
        mock_get_meter.assert_called_once_with("service-metrics")

    def test_get_service_meter_disabled(self):
        """Test getting service meter when disabled."""
        observability_service = {
            "enabled": False,
            "meter": None,
        }

        meter = get_service_meter(observability_service)

        assert isinstance(meter, _NoOpMeter)

    def test_create_span_context(self):
        """Test creating span context."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        span_context = create_span_context("test-operation", mock_tracer)

        assert span_context == mock_span
        mock_tracer.start_as_current_span.assert_called_once_with("test-operation")

    @pytest.mark.asyncio
    async def test_record_ai_operation_metrics(self):
        """Test recording AI operation metrics."""
        mock_meter = MagicMock()

        with patch(
            "src.services.observability.tracking.record_ai_operation"
        ) as mock_record:
            await record_ai_operation_metrics(
                operation_type="embedding",
                provider="openai",
                success=True,
                duration=1.5,
                meter=mock_meter,
                model="text-embedding-3-small",
            )

            mock_record.assert_called_once_with(
                operation_type="embedding",
                provider="openai",
                success=True,
                duration=1.5,
                model="text-embedding-3-small",
            )

    @pytest.mark.asyncio
    async def test_record_ai_operation_metrics_exception_handling(self):
        """Test AI operation metrics recording handles exceptions."""
        mock_meter = MagicMock()

        with patch(
            "src.services.observability.tracking.record_ai_operation"
        ) as mock_record:
            mock_record.side_effect = Exception("Metrics error")

            # Should not raise exception
            await record_ai_operation_metrics(
                operation_type="embedding",
                provider="openai",
                success=True,
                duration=1.5,
                meter=mock_meter,
            )

    @pytest.mark.asyncio
    async def test_track_ai_cost_metrics(self):
        """Test tracking AI cost metrics."""
        mock_meter = MagicMock()

        with patch("src.services.observability.tracking.track_cost") as mock_track_cost:
            await track_ai_cost_metrics(
                operation_type="completion",
                provider="openai",
                cost_usd=0.05,
                meter=mock_meter,
                model="gpt-3.5-turbo",
            )

            mock_track_cost.assert_called_once_with(
                operation_type="completion",
                provider="openai",
                cost_usd=0.05,
                model="gpt-3.5-turbo",
            )

    @pytest.mark.asyncio
    async def test_track_ai_cost_metrics_exception_handling(self):
        """Test AI cost metrics tracking handles exceptions."""
        mock_meter = MagicMock()

        with patch("src.services.observability.tracking.track_cost") as mock_track_cost:
            mock_track_cost.side_effect = Exception("Cost tracking error")

            # Should not raise exception
            await track_ai_cost_metrics(
                operation_type="completion",
                provider="openai",
                cost_usd=0.05,
                meter=mock_meter,
            )

    @pytest.mark.asyncio
    async def test_get_observability_health_enabled(self):
        """Test getting observability health when enabled."""
        config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            otlp_endpoint="http://test.example.com:4317",
            track_ai_operations=True,
            track_costs=True,
        )

        observability_service = {
            "config": config,
            "enabled": True,
        }

        health = await get_observability_health(observability_service)

        assert health["enabled"] is True
        assert health["service_name"] == "test-service"
        assert health["otlp_endpoint"] == "http://test.example.com:4317"
        assert health["status"] == "healthy"
        assert health["instrumentation"]["fastapi"] is True
        assert health["ai_tracking"]["operations"] is True
        assert health["ai_tracking"]["costs"] is True

    @pytest.mark.asyncio
    async def test_get_observability_health_disabled(self):
        """Test getting observability health when disabled."""
        config = ObservabilityConfig(enabled=False)

        observability_service = {
            "config": config,
            "enabled": False,
        }

        health = await get_observability_health(observability_service)

        assert health["enabled"] is False
        assert health["status"] == "disabled"
        assert health["otlp_endpoint"] is None
        assert health["instrumentation"]["fastapi"] is False
        assert health["ai_tracking"]["operations"] is False

    @pytest.mark.asyncio
    async def test_get_observability_health_exception_handling(self):
        """Test observability health check handles exceptions."""
        observability_service = {"config": None}  # Invalid service

        health = await get_observability_health(observability_service)

        assert health["enabled"] is False
        assert health["status"] == "error"
        assert "error" in health


class TestObservabilityDependenciesCaching:
    """Test observability dependencies caching behavior."""

    def setup_method(self):
        """Setup for each test."""
        get_observability_service.cache_clear()

    @patch("src.services.observability.dependencies.get_observability_config")
    @patch("src.services.observability.dependencies.is_observability_enabled")
    @patch("src.services.observability.dependencies.get_tracer")
    @patch("src.services.observability.dependencies.get_meter")
    def test_observability_service_caching(
        self,
        mock_get_meter,
        mock_get_tracer,
        mock_is_enabled,
        mock_get_config,
    ):
        """Test that observability service is cached properly."""
        config = ObservabilityConfig(enabled=False)
        mock_get_config.return_value = config
        mock_is_enabled.return_value = False

        mock_tracer = MagicMock()
        mock_meter = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter

        # Clear cache and call twice
        get_observability_service.cache_clear()

        service1 = get_observability_service()
        service2 = get_observability_service()

        # Should be the same instance due to caching
        assert service1 is service2

        # Config should only be called once due to caching
        assert mock_get_config.call_count == 1

    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        get_observability_service.cache_clear()

        with patch(
            "src.services.observability.dependencies.get_observability_config"
        ) as mock_get_config:
            config = ObservabilityConfig(enabled=False)
            mock_get_config.return_value = config

            # Call service
            service1 = get_observability_service()

            # Clear cache and call again
            get_observability_service.cache_clear()
            service2 = get_observability_service()

            # Should be different instances after cache clear
            assert service1 is not service2

            # Config should be called twice (once per cache miss)
            assert mock_get_config.call_count == 2
