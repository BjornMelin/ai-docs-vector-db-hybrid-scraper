"""Integration tests for OpenTelemetry observability."""

import asyncio
import importlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.core import get_config, reset_config
from src.services.observability.config import ObservabilityConfig
from src.services.observability.dependencies import (
    get_ai_tracer,
    get_observability_health,
    get_observability_service,
    get_service_meter,
)
from src.services.observability.init import (
    initialize_observability,
    is_observability_enabled,
    shutdown_observability,
)
from src.services.observability.middleware import FastAPIObservabilityMiddleware
from src.services.observability.tracking import (
    _NoOpMeter,
    _NoOpTracer,
    get_meter,
    get_tracer,
    instrument_function,
    record_ai_operation,
    track_cost,
)


class TestObservabilityIntegration:
    """Test complete observability integration."""

    def setup_method(self):
        """Setup for each test."""
        reset_config()
        shutdown_observability()
        get_observability_service.cache_clear()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_observability()
        reset_config()

    def test_observability_disabled_by_default(self):
        """Test that observability is disabled by default."""
        config = get_config()

        assert config.observability.enabled is False
        assert is_observability_enabled() is False

    @patch.dict("os.environ", {"AI_DOCS_OBSERVABILITY__ENABLED": "true"})
    def test_observability_enabled_via_environment(self):
        """Test enabling observability via environment variables."""
        reset_config()

        config = get_config()
        assert config.observability.enabled is True

    @patch("opentelemetry.trace")
    @patch("opentelemetry.metrics")
    @patch("opentelemetry.sdk.trace.TracerProvider")
    @patch("opentelemetry.sdk.metrics.MeterProvider")
    @patch("opentelemetry.sdk.trace.export.BatchSpanProcessor")
    @patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter")
    @patch("opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter")
    @patch("opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader")
    @patch("opentelemetry.sdk.resources.Resource")
    @patch("src.services.observability.init._setup_auto_instrumentation")
    def test_full_observability_initialization(
        self,
        _mock_auto_instrumentation,
        mock_resource,
        mock_metric_reader,
        _mock_metric_exporter,
        _mock_span_exporter,
        mock_span_processor,
        mock_meter_provider,
        mock_tracer_provider,
        mock_metrics,
        mock_trace,
    ):
        """Test complete observability initialization flow."""
        # Setup mocks
        mock_resource.create.return_value = MagicMock()
        mock_tracer_provider.return_value = MagicMock()
        mock_meter_provider.return_value = MagicMock()
        mock_span_processor.return_value = MagicMock()
        mock_metric_reader.return_value = MagicMock()

        config = ObservabilityConfig(
            enabled=True,
            service_name="integration-test",
            otlp_endpoint="http://test.example.com:4317",
        )

        # Initialize observability
        result = initialize_observability(config)

        assert result is True
        assert is_observability_enabled() is True

        # Verify providers were set up
        mock_trace.set_tracer_provider.assert_called_once()
        mock_metrics.set_meter_provider.assert_called_once()

        # Test shutdown
        shutdown_observability()
        assert is_observability_enabled() is False

    def test_observability_service_dependency_integration(self):
        """Test observability service dependency integration."""
        # Test with disabled observability
        service = get_observability_service()

        assert service["enabled"] is False
        # When disabled, we get NoOp implementations
        assert isinstance(service["tracer"], _NoOpTracer)
        assert isinstance(service["meter"], _NoOpMeter)

    @patch("src.services.observability.dependencies.initialize_observability")
    @patch("src.services.observability.dependencies.is_observability_enabled")
    @patch("src.services.observability.dependencies.get_tracer")
    @patch("src.services.observability.dependencies.get_meter")
    def test_observability_service_initialization_flow(
        self,
        mock_get_meter,
        mock_get_tracer,
        mock_is_enabled,
        mock_initialize,
    ):
        """Test observability service initialization flow."""
        # Setup config to be enabled via monitoring.enable_metrics
        with patch.dict("os.environ", {"AI_DOCS_MONITORING__ENABLE_METRICS": "true"}):
            reset_config()
            get_observability_service.cache_clear()

            # Setup mocks
            mock_is_enabled.side_effect = [
                False,
                True,
            ]  # First not initialized, then initialized
            mock_initialize.return_value = True
            mock_get_tracer.return_value = MagicMock()
            mock_get_meter.return_value = MagicMock()

            service = get_observability_service()

            assert service["enabled"] is True
            assert service["tracer"] is not None
            assert service["meter"] is not None

            # Verify initialization was called
            mock_initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_observability_health_check_integration(self):
        """Test observability health check integration."""
        service = get_observability_service()
        health = await get_observability_health(service)

        assert "enabled" in health
        assert "status" in health
        assert health["status"] in ["healthy", "disabled", "error"]

    def test_dependency_injection_integration(self):
        """Test dependency injection integration."""
        service = get_observability_service()

        # Test AI tracer dependency
        tracer = get_ai_tracer(service)
        assert tracer is not None

        # Test service meter dependency
        meter = get_service_meter(service)
        assert meter is not None

        # With disabled observability, should get NoOp implementations
        assert isinstance(tracer, _NoOpTracer)
        assert isinstance(meter, _NoOpMeter)


class TestFastAPIObservabilityIntegration:
    """Test FastAPI observability middleware integration."""

    def setup_method(self):
        """Setup for each test."""
        self.app = FastAPI(title="Test API")

        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        @self.app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

    def test_middleware_integration_disabled(self):
        """Test middleware integration when observability is disabled."""
        with (
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.metrics": MagicMock(),
                },
            ),
            patch("src.services.observability.tracking.get_tracer") as mock_get_tracer,
        ):
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            # Add middleware
            self.app.add_middleware(
                FastAPIObservabilityMiddleware,
                **{
                    "service_name": "test-service",
                    "record_request_metrics": False,
                },
            )

            client = TestClient(self.app)

            # Make request
            response = client.get("/test")

            assert response.status_code == 200
            assert response.json() == {"message": "test"}

    def test_middleware_integration_with_metrics(self):
        """Test middleware integration with metrics enabled."""

        # Setup mocks
        mock_tracer = MagicMock()
        mock_meter = MagicMock()
        mock_span = MagicMock()

        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        # Setup metric mocks
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_up_down_counter = MagicMock()

        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_up_down_counter.return_value = mock_up_down_counter

        # Mock the tracking module before importing middleware
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.trace": MagicMock(),
                "opentelemetry.metrics": MagicMock(),
            },
        ):
            # Patch tracking functions
            with (
                patch(
                    "src.services.observability.tracking.get_tracer",
                    return_value=mock_tracer,
                ),
                patch(
                    "src.services.observability.tracking.get_meter",
                    return_value=mock_meter,
                ),
            ):
                # Force reload of middleware module to pick up mocked functions
                import src.services.observability.middleware

                importlib.reload(src.services.observability.middleware)

                from src.services.observability.middleware import (
                    FastAPIObservabilityMiddleware,
                )

                # Add middleware with metrics
                self.app.add_middleware(
                    FastAPIObservabilityMiddleware,
                    **{
                        "service_name": "test-service",
                        "record_request_metrics": True,
                    },
                )

                client = TestClient(self.app)

                # Make request
                response = client.get("/test")

                assert response.status_code == 200

                # Verify span was created during request
                mock_tracer.start_as_current_span.assert_called()

                # Verify metrics were recorded
                mock_up_down_counter.add.assert_called()  # Active requests
                mock_histogram.record.assert_called()  # Duration
                mock_counter.add.assert_called()  # Request count

    def test_middleware_integration_error_handling(self):
        """Test middleware error handling integration."""

        # Setup mocks
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        # Mock the tracking module before importing middleware
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.trace": MagicMock(),
                "opentelemetry.metrics": MagicMock(),
            },
        ):
            # Patch tracking functions
            with patch(
                "src.services.observability.tracking.get_tracer",
                return_value=mock_tracer,
            ):
                # Force reload of middleware module to pick up mocked functions
                import src.services.observability.middleware

                importlib.reload(src.services.observability.middleware)

                from src.services.observability.middleware import (
                    FastAPIObservabilityMiddleware,
                )

                # Add middleware
                self.app.add_middleware(
                    FastAPIObservabilityMiddleware,
                    **{
                        "service_name": "test-service",
                        "record_request_metrics": False,
                    },
                )

                client = TestClient(self.app)

                # Make request to error endpoint
                response = client.get("/error")

                assert response.status_code == 500

                # Verify error was recorded in span
                mock_span.record_exception.assert_called()
                mock_span.set_status.assert_called()

    def test_middleware_ai_context_detection(self):
        """Test middleware AI context detection."""

        # Setup mocks
        mock_tracer = MagicMock()
        mock_span = MagicMock()

        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        # Mock the tracking module before importing middleware
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.trace": MagicMock(),
                "opentelemetry.metrics": MagicMock(),
            },
        ):
            # Patch tracking functions
            with patch(
                "src.services.observability.tracking.get_tracer",
                return_value=mock_tracer,
            ):
                # Force reload of middleware module to pick up mocked functions
                import src.services.observability.middleware

                importlib.reload(src.services.observability.middleware)

                from src.services.observability.middleware import (
                    FastAPIObservabilityMiddleware,
                )

                # Add AI endpoint
                @self.app.post("/api/search")
                async def search_endpoint():
                    return {"results": []}

                # Add middleware with AI context
                self.app.add_middleware(
                    FastAPIObservabilityMiddleware,
                    **{
                        "service_name": "test-service",
                        "record_ai_context": True,
                        "record_request_metrics": False,
                    },
                )

                client = TestClient(self.app)

                # Make request to AI endpoint
                response = client.post("/api/search?model=embedding&provider=openai")

                assert response.status_code == 200

                # Verify AI context attributes were set
                mock_span.set_attribute.assert_any_call("ai.operation.type", "search")
                mock_span.set_attribute.assert_any_call("ai.model", "embedding")
                mock_span.set_attribute.assert_any_call("ai.provider", "openai")


class TestObservabilityConfigurationIntegration:
    """Test observability configuration integration."""

    def setup_method(self):
        """Setup for each test."""
        reset_config()

    def teardown_method(self):
        """Cleanup after each test."""
        reset_config()

    def test_configuration_hierarchy(self):
        """Test configuration hierarchy and inheritance."""
        config = get_config()

        # Test default configuration
        assert hasattr(config, "observability")
        assert isinstance(config.observability, ObservabilityConfig)
        assert config.observability.enabled is False

    @patch.dict(
        "os.environ",
        {
            "AI_DOCS_OBSERVABILITY__ENABLED": "true",
            "AI_DOCS_OBSERVABILITY__SERVICE_NAME": "integration-test",
            "AI_DOCS_OBSERVABILITY__TRACE_SAMPLE_RATE": "0.5",
        },
    )
    def test_environment_variable_configuration(self):
        """Test configuration via environment variables."""
        reset_config()

        config = get_config()

        assert config.observability.enabled is True
        assert config.observability.service_name == "integration-test"
        assert config.observability.trace_sample_rate == 0.5

    def test_programmatic_configuration(self):
        """Test programmatic configuration override."""
        config = get_config()

        # Modify observability config
        config.observability.enabled = True
        config.observability.service_name = "programmatic-test"

        assert config.observability.enabled is True
        assert config.observability.service_name == "programmatic-test"

    def test_configuration_validation_integration(self):
        """Test configuration validation integration."""
        # Test valid configuration
        config = ObservabilityConfig(
            enabled=True,
            trace_sample_rate=0.5,
        )
        assert config.trace_sample_rate == 0.5

        # Test invalid configuration should raise validation error
        with pytest.raises(ValueError):
            ObservabilityConfig(trace_sample_rate=1.5)


class TestEndToEndObservabilityFlow:
    """Test end-to-end observability flow."""

    def setup_method(self):
        """Setup for each test."""
        reset_config()
        shutdown_observability()
        get_observability_service.cache_clear()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_observability()
        reset_config()

    @patch("opentelemetry.trace")
    @patch("opentelemetry.metrics")
    @patch("opentelemetry.sdk.trace.TracerProvider")
    @patch("opentelemetry.sdk.metrics.MeterProvider")
    @patch("opentelemetry.sdk.trace.export.BatchSpanProcessor")
    @patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter")
    @patch("opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter")
    @patch("opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader")
    @patch("opentelemetry.sdk.resources.Resource")
    @patch("src.services.observability.init._setup_auto_instrumentation")
    def test_complete_observability_flow(
        self,
        _mock_auto_instrumentation,
        mock_resource,
        mock_metric_reader,
        _mock_metric_exporter,
        _mock_span_exporter,
        mock_span_processor,
        mock_meter_provider,
        mock_tracer_provider,
        _mock_metrics,
        _mock_trace,
    ):
        """Test complete observability flow from configuration to shutdown."""
        # Setup comprehensive mocks
        mock_resource.create.return_value = MagicMock()

        mock_tracer_provider_instance = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_provider_instance

        mock_meter_provider_instance = MagicMock()
        mock_meter_provider.return_value = mock_meter_provider_instance

        mock_span_processor_instance = MagicMock()
        mock_span_processor.return_value = mock_span_processor_instance

        mock_metric_reader_instance = MagicMock()
        mock_metric_reader.return_value = mock_metric_reader_instance

        # Step 1: Enable observability via environment
        with patch.dict("os.environ", {"AI_DOCS_OBSERVABILITY__ENABLED": "true"}):
            reset_config()
            get_observability_service.cache_clear()

            # Step 2: Get observability service (should trigger initialization)
            service = get_observability_service()

            assert service["enabled"] is True
            assert service["tracer"] is not None
            assert service["meter"] is not None

            # Step 3: Test dependency injection
            tracer = get_ai_tracer(service)
            meter = get_service_meter(service)

            assert tracer is not None
            assert meter is not None

            # Step 4: Test health check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                health = loop.run_until_complete(get_observability_health(service))

                assert health["enabled"] is True
                assert health["status"] == "healthy"

            finally:
                loop.close()

            # Step 5: Test shutdown
            shutdown_observability()
            assert is_observability_enabled() is False

            # Verify shutdown was called on providers
            mock_tracer_provider_instance.shutdown.assert_called_once()
            mock_meter_provider_instance.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_observability_with_ai_operations(self):
        """Test observability integration with AI operations."""

        # Test AI operation recording (should work even when disabled)
        record_ai_operation(
            operation_type="embedding",
            provider="test-provider",
            success=True,
            duration=0.5,
        )

        # Test cost tracking
        track_cost(
            operation_type="completion",
            provider="test-provider",
            cost_usd=0.01,
        )

        # Test function instrumentation
        @instrument_function(operation_type="test_function")
        async def test_ai_function():
            await asyncio.sleep(0.001)
            return "success"

        result = await test_ai_function()
        assert result == "success"

    def test_observability_graceful_degradation(self):
        """Test observability graceful degradation when packages unavailable."""
        # Test that observability functions work even when OpenTelemetry is unavailable

        with patch("src.services.observability.tracking.trace") as mock_trace:
            mock_trace.side_effect = ImportError("OpenTelemetry not available")

            tracer = get_tracer("test-service")
            assert isinstance(tracer, _NoOpTracer)

        with patch("src.services.observability.tracking.metrics") as mock_metrics:
            mock_metrics.side_effect = ImportError("OpenTelemetry not available")

            meter = get_meter("test-service")
            assert isinstance(meter, _NoOpMeter)
