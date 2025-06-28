"""Tests for OpenTelemetry initialization."""

from unittest.mock import MagicMock, patch

import src.services.observability.init as init_module
from src.services.observability.config import ObservabilityConfig
from src.services.observability.init import (
    _setup_auto_instrumentation,
    initialize_observability,
    is_observability_enabled,
    shutdown_observability,
)


class TestObservabilityInitialization:
    """Test OpenTelemetry initialization functionality."""

    def setup_method(self):
        """Setup for each test."""
        # Reset global state

        init_module._tracer_provider = None
        init_module._meter_provider = None

    def test_initialize_observability_disabled(self):
        """Test initialization when observability is disabled."""
        config = ObservabilityConfig(enabled=False)

        result = initialize_observability(config)

        assert result is False
        assert is_observability_enabled() is False

    def test_initialize_observability_missing_packages(self):
        """Test initialization when OpenTelemetry packages are missing."""
        config = ObservabilityConfig(enabled=True)

        # Mock the import to raise ImportError when trying to access OpenTelemetry
        with patch.dict("sys.modules", {"opentelemetry.trace": None}):
            result = initialize_observability(config)

            assert result is False
            assert is_observability_enabled() is False

    def test_initialize_observability_success(self):
        """Test successful OpenTelemetry initialization."""
        config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            otlp_endpoint="http://test.example.com:4317",
        )

        # Mock all the OpenTelemetry imports using sys.modules approach
        mock_modules = {
            "opentelemetry": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
        }

        # Setup the mock objects to have the expected attributes
        mock_tracer_provider = MagicMock()
        mock_meter_provider = MagicMock()

        mock_modules["opentelemetry.sdk.trace"].TracerProvider = mock_tracer_provider
        mock_modules["opentelemetry.sdk.metrics"].MeterProvider = mock_meter_provider
        mock_modules["opentelemetry.sdk.resources"].Resource = MagicMock()
        mock_modules["opentelemetry.sdk.trace.export"].BatchSpanProcessor = MagicMock()
        mock_modules[
            "opentelemetry.sdk.metrics.export"
        ].PeriodicExportingMetricReader = MagicMock()
        mock_modules[
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        ].OTLPSpanExporter = MagicMock()
        mock_modules[
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter"
        ].OTLPMetricExporter = MagicMock()

        with (
            patch.dict("sys.modules", mock_modules),
            patch(
                "src.services.observability.init._setup_auto_instrumentation"
            ) as mock_auto_instrumentation,
        ):
            result = initialize_observability(config)

            assert result is True
            assert is_observability_enabled() is True

            # Verify auto-instrumentation setup was called
            mock_auto_instrumentation.assert_called_once_with(config)

    def test_initialize_observability_with_console_exporter(self):
        """Test initialization with console exporter enabled."""
        config = ObservabilityConfig(
            enabled=True,
            console_exporter=True,
        )

        # Mock OpenTelemetry modules
        mock_modules = {
            "opentelemetry": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.metrics": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.metrics": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(),
            "opentelemetry.sdk.metrics.export": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": MagicMock(),
        }

        # Setup console exporter mock
        mock_console_exporter = MagicMock()
        mock_modules[
            "opentelemetry.sdk.trace.export"
        ].ConsoleSpanExporter = mock_console_exporter

        # Setup other required mocks
        mock_tracer_provider_instance = MagicMock()
        mock_modules["opentelemetry.sdk.trace"].TracerProvider = MagicMock(
            return_value=mock_tracer_provider_instance
        )
        mock_modules["opentelemetry.sdk.metrics"].MeterProvider = MagicMock()
        mock_modules["opentelemetry.sdk.resources"].Resource = MagicMock()
        mock_modules["opentelemetry.sdk.trace.export"].BatchSpanProcessor = MagicMock()
        mock_modules[
            "opentelemetry.sdk.metrics.export"
        ].PeriodicExportingMetricReader = MagicMock()
        mock_modules[
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        ].OTLPSpanExporter = MagicMock()
        mock_modules[
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter"
        ].OTLPMetricExporter = MagicMock()

        with patch.dict("sys.modules", mock_modules):
            with patch("src.services.observability.init._setup_auto_instrumentation"):
                result = initialize_observability(config)

                assert result is True

                # Verify console exporter was configured (two span processors: OTLP + Console)
                assert mock_tracer_provider_instance.add_span_processor.call_count == 2

    def test_initialize_observability_exception_handling(self):
        """Test initialization handles exceptions gracefully."""
        config = ObservabilityConfig(enabled=True)

        # Mock the import to cause an exception during initialization
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **_kwargs):
                if "opentelemetry" in name:
                    msg = "Initialization failed"
                    raise ImportError(msg)
                return __import__(name, *args, **_kwargs)

            mock_import.side_effect = side_effect

            result = initialize_observability(config)

            assert result is False
            assert is_observability_enabled() is False

    def test_shutdown_observability(self):
        """Test observability shutdown."""
        # Setup providers

        mock_tracer_provider = MagicMock()
        mock_meter_provider = MagicMock()

        init_module._tracer_provider = mock_tracer_provider
        init_module._meter_provider = mock_meter_provider

        shutdown_observability()

        # Verify shutdown was called
        mock_tracer_provider.shutdown.assert_called_once()
        mock_meter_provider.shutdown.assert_called_once()

        # Verify providers were reset
        assert init_module._tracer_provider is None
        assert init_module._meter_provider is None

    def test_shutdown_observability_with_exceptions(self):
        """Test shutdown handles exceptions gracefully."""
        # Setup providers that raise exceptions

        mock_tracer_provider = MagicMock()
        mock_tracer_provider.shutdown.side_effect = Exception("Shutdown failed")

        mock_meter_provider = MagicMock()
        mock_meter_provider.shutdown.side_effect = Exception("Shutdown failed")

        init_module._tracer_provider = mock_tracer_provider
        init_module._meter_provider = mock_meter_provider

        # Should not raise exception
        shutdown_observability()

        # Verify providers were reset despite exceptions
        # Note: The implementation catches exceptions but still sets providers to None
        assert init_module._tracer_provider is None
        assert init_module._meter_provider is None

    def test_is_observability_enabled(self):
        """Test observability enabled check."""

        # Initially disabled
        assert is_observability_enabled() is False

        # Set tracer provider
        init_module._tracer_provider = MagicMock()
        assert is_observability_enabled() is True

        # Reset
        init_module._tracer_provider = None
        assert is_observability_enabled() is False


class TestAutoInstrumentation:
    """Test automatic instrumentation setup."""

    def test_setup_fastapi_instrumentation(self):
        """Test FastAPI auto-instrumentation setup."""
        config = ObservabilityConfig(instrument_fastapi=True)

        # Mock the FastAPI instrumentor module
        mock_instrumentor_instance = MagicMock()
        mock_fastapi_instrumentor = MagicMock(return_value=mock_instrumentor_instance)

        mock_modules = {
            "opentelemetry": MagicMock(),
            "opentelemetry.instrumentation": MagicMock(),
            "opentelemetry.instrumentation.fastapi": MagicMock(),
        }
        mock_modules[
            "opentelemetry.instrumentation.fastapi"
        ].FastAPIInstrumentor = mock_fastapi_instrumentor

        with patch.dict("sys.modules", mock_modules):
            _setup_auto_instrumentation(config)

            mock_instrumentor_instance.instrument.assert_called_once()

    def test_setup_fastapi_instrumentation_import_error(self):
        """Test FastAPI instrumentation with import error."""
        config = ObservabilityConfig(instrument_fastapi=True)

        # Without OpenTelemetry packages installed, ImportError should be handled gracefully
        # Should not raise exception
        _setup_auto_instrumentation(config)

    def test_setup_httpx_instrumentation(self):
        """Test HTTPX auto-instrumentation setup."""
        config = ObservabilityConfig(instrument_httpx=True)

        # Test that the function handles missing packages gracefully
        _setup_auto_instrumentation(config)

    def test_setup_redis_instrumentation(self):
        """Test Redis auto-instrumentation setup."""
        config = ObservabilityConfig(instrument_redis=True)

        # Test that the function handles missing packages gracefully
        _setup_auto_instrumentation(config)

    def test_setup_sqlalchemy_instrumentation(self):
        """Test SQLAlchemy auto-instrumentation setup."""
        config = ObservabilityConfig(instrument_sqlalchemy=True)

        # Test that the function handles missing packages gracefully
        _setup_auto_instrumentation(config)

    def test_setup_auto_instrumentation_disabled(self):
        """Test auto-instrumentation when all are disabled."""
        config = ObservabilityConfig(
            instrument_fastapi=False,
            instrument_httpx=False,
            instrument_redis=False,
            instrument_sqlalchemy=False,
        )

        # Should not raise any exceptions
        _setup_auto_instrumentation(config)

    def test_setup_auto_instrumentation_exception_handling(self):
        """Test auto-instrumentation handles exceptions gracefully."""
        config = ObservabilityConfig(instrument_fastapi=True)

        # Without OpenTelemetry packages, this should handle gracefully
        # Should not raise exception
        _setup_auto_instrumentation(config)
