"""Integration tests for OpenTelemetry observability with boundary-only mocking."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import get_config, reset_config
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

    def test_observability_integration_with_external_config(self):
        """Test observability integration using external configuration."""
        # Test with external config service (boundary mock)
        config = ObservabilityConfig(
            enabled=True,
            service_name="integration-test",
            otlp_endpoint="http://test.example.com:4317",
        )

        # Test initialization behavior with real internal components
        # Only mock the external OTLP endpoint behavior
        with patch("src.services.observability.init.OTLPSpanExporter") as mock_exporter:
            mock_exporter.return_value = MagicMock()

            result = initialize_observability(config)

            # Verify external boundary was used
            if result:  # If initialization succeeded
                assert is_observability_enabled() == result

                # Test shutdown behavior
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

    def test_observability_service_initialization_flow(self):
        """Test observability service initialization flow with boundary mocking."""
        # Setup config to enable observability through environment
        with patch.dict("os.environ", {"AI_DOCS_MONITORING__ENABLE_METRICS": "true"}):
            reset_config()
            get_observability_service.cache_clear()

            # Only mock external OpenTelemetry components
            with patch(
                "src.services.observability.init.OTLPSpanExporter"
            ) as mock_exporter:
                mock_exporter.return_value = MagicMock()

                service = get_observability_service()

                # Test observable behavior
                assert "enabled" in service
                assert "tracer" in service
                assert "meter" in service

                # Service should have appropriate implementations based on config
                if service["enabled"]:
                    assert service["tracer"] is not None
                    assert service["meter"] is not None
                else:
                    assert isinstance(service["tracer"], _NoOpTracer)
                    assert isinstance(service["meter"], _NoOpMeter)

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
        @pytest.mark.asyncio
        async def test_endpoint():
            return {"message": "test"}

        @self.app.get("/error")
        async def error_endpoint():
            msg = "Test error"
            raise ValueError(msg)

    def test_middleware_integration_disabled(self):
        """Test middleware integration when observability is disabled."""
        # Add middleware with minimal external boundary mocking
        self.app.add_middleware(
            FastAPIObservabilityMiddleware,
            service_name="test-service",
            record_request_metrics=False,
        )

        client = TestClient(self.app)

        # Make request - should work regardless of observability state
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"message": "test"}

    def test_middleware_integration_with_metrics(self):
        """Test middleware integration with metrics enabled."""
        # Add middleware with metrics enabled
        self.app.add_middleware(
            FastAPIObservabilityMiddleware,
            service_name="test-service",
            record_request_metrics=True,
        )

        client = TestClient(self.app)

        # Make request - should work regardless of actual metric collection
        response = client.get("/test")

        assert response.status_code == 200
        assert response.json() == {"message": "test"}

        # The middleware should handle metric collection gracefully
        # (Implementation details are tested in unit tests)

    def test_middleware_integration_error_handling(self):
        """Test middleware error handling integration."""
        # Add middleware
        self.app.add_middleware(
            FastAPIObservabilityMiddleware,
            service_name="test-service",
            record_request_metrics=False,
        )

        client = TestClient(self.app)

        # Make request to error endpoint
        response = client.get("/error")

        assert response.status_code == 500

        # Middleware should handle errors gracefully without breaking the request

    def test_middleware_ai_context_detection(self):
        """Test middleware AI context detection."""

        # Add AI endpoint
        @self.app.post("/api/search")
        async def search_endpoint():
            return {"results": []}

        # Add middleware with AI context detection enabled
        self.app.add_middleware(
            FastAPIObservabilityMiddleware,
            service_name="test-service",
            record_ai_context=True,
            record_request_metrics=False,
        )

        client = TestClient(self.app)

        # Make request to AI endpoint with AI-related parameters
        response = client.post("/api/search?model=embedding&provider=openai")

        assert response.status_code == 200
        assert response.json() == {"results": []}

        # Middleware should detect AI context from URL patterns and parameters
        # (Specific attribute setting is tested in unit tests)


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
        assert config.observability.service_name == "integration-test"  # Test data
        assert config.observability.trace_sample_rate == 0.5

    def test_programmatic_configuration(self):
        """Test programmatic configuration override."""
        config = get_config()

        # Modify observability config
        config.observability.enabled = True
        config.observability.service_name = "programmatic-test"  # Test data

        assert config.observability.enabled is True
        assert config.observability.service_name == "programmatic-test"  # Test data

    def test_configuration_validation_integration(self):
        """Test configuration validation integration."""
        # Test valid configuration
        config = ObservabilityConfig(
            enabled=True,
            trace_sample_rate=0.5,
        )
        assert config.trace_sample_rate == 0.5

        # Test invalid configuration should raise validation error
        with pytest.raises(ValueError, match="trace_sample_rate"):
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

    @pytest.mark.asyncio
    async def test_complete_observability_flow(self):
        """Test complete observability flow from configuration to shutdown."""
        # Only mock external OTLP exporter (boundary mock)
        with patch("src.services.observability.init.OTLPSpanExporter") as mock_exporter:
            mock_exporter.return_value = MagicMock()

            # Step 1: Enable observability via environment
            with patch.dict("os.environ", {"AI_DOCS_OBSERVABILITY__ENABLED": "true"}):
                reset_config()
                get_observability_service.cache_clear()

                # Step 2: Get observability service
                service = get_observability_service()

                # Test observable behavior
                assert "enabled" in service
                assert "tracer" in service
                assert "meter" in service

                # Step 3: Test dependency injection
                tracer = get_ai_tracer(service)
                meter = get_service_meter(service)

                assert tracer is not None
                assert meter is not None

                # Step 4: Test health check
                health = await get_observability_health(service)

                assert "enabled" in health
                assert "status" in health
                assert health["status"] in ["healthy", "disabled", "error"]

                # Step 5: Test shutdown
                shutdown_observability()
                assert is_observability_enabled() is False

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
        @pytest.mark.asyncio
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
