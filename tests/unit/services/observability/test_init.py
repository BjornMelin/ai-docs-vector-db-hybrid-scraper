"""Tests for the simplified observability initialisation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services.observability import init as init_module
from src.services.observability.config import ObservabilityConfig
from src.services.observability.init import (
    initialize_observability,
    is_observability_enabled,
    shutdown_observability,
)


def _otel_modules(resource_factory: MagicMock | None = None) -> dict[str, object]:
    """Build mock modules for OpenTelemetry imports."""
    # pylint: disable=too-many-locals
    resource_factory = resource_factory or MagicMock(
        create=MagicMock(return_value=MagicMock())
    )

    trace_module = SimpleNamespace(set_tracer_provider=MagicMock())
    metrics_module = SimpleNamespace(set_meter_provider=MagicMock())

    exporter_trace = SimpleNamespace(OTLPSpanExporter=MagicMock())
    exporter_metric = SimpleNamespace(OTLPMetricExporter=MagicMock())
    exporter_grpc = SimpleNamespace(
        trace_exporter=exporter_trace,
        metric_exporter=exporter_metric,
    )
    exporter_proto = SimpleNamespace(grpc=exporter_grpc)
    exporter_otlp = SimpleNamespace(proto=exporter_proto)
    exporter_module = SimpleNamespace(otlp=exporter_otlp)

    sdk_trace_export = SimpleNamespace(
        BatchSpanProcessor=MagicMock(),
        ConsoleSpanExporter=MagicMock(),
    )
    sdk_trace = SimpleNamespace(TracerProvider=MagicMock(), export=sdk_trace_export)
    sdk_metrics_export = SimpleNamespace(PeriodicExportingMetricReader=MagicMock())
    sdk_metrics = SimpleNamespace(MeterProvider=MagicMock(), export=sdk_metrics_export)
    sdk_resources = SimpleNamespace(Resource=resource_factory)
    sdk_module = SimpleNamespace(
        trace=sdk_trace,
        metrics=sdk_metrics,
        resources=sdk_resources,
    )

    root_module = SimpleNamespace(
        trace=trace_module,
        metrics=metrics_module,
        exporter=exporter_module,
        sdk=sdk_module,
    )

    return {
        "opentelemetry": root_module,
        "opentelemetry.trace": trace_module,
        "opentelemetry.metrics": metrics_module,
        "opentelemetry.exporter": exporter_module,
        "opentelemetry.exporter.otlp": exporter_otlp,
        "opentelemetry.exporter.otlp.proto": exporter_proto,
        "opentelemetry.exporter.otlp.proto.grpc": exporter_grpc,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": exporter_trace,
        "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": exporter_metric,
        "opentelemetry.sdk": sdk_module,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.trace.export": sdk_trace_export,
        "opentelemetry.sdk.metrics": sdk_metrics,
        "opentelemetry.sdk.metrics.export": sdk_metrics_export,
        "opentelemetry.sdk.resources": sdk_resources,
    }


class TestInitializeObservability:
    def setup_method(self) -> None:
        """Reset the observability state between test cases."""

        init_module._STATE.tracer_provider = None
        init_module._STATE.meter_provider = None

    def test_disabled_config_short_circuits(self) -> None:
        """Verify disabled configurations skip initialisation logic."""

        assert initialize_observability(ObservabilityConfig(enabled=False)) is False
        assert is_observability_enabled() is False

    def test_successful_initialisation(self) -> None:
        """Ensure a valid configuration wires OpenTelemetry components."""

        config = ObservabilityConfig(service_name="tests", enabled=True)
        resource_mock = MagicMock()
        resource_mock.create.return_value = MagicMock()

        with (
            patch.object(init_module, "_configure_instrumentations") as configure,
            patch.dict(
                "sys.modules",
                _otel_modules(resource_factory=resource_mock),
            ),
        ):
            assert initialize_observability(config) is True
            assert is_observability_enabled() is True
            configure.assert_called_once_with(config.instrumentations)

    def test_initialisation_from_settings(self) -> None:
        """Settings objects should be coerced into runtime observability configs."""

        settings = SimpleNamespace(
            app_name="Test Application",
            version="9.9.9",
            environment=SimpleNamespace(value="staging"),
            observability=SimpleNamespace(
                enabled=True,
                service_name="override-service",
                service_version="2.0.0",
                otlp_endpoint="http://collector:4317",
                otlp_headers={"authorization": "Bearer token"},
                otlp_insecure=False,
                track_ai_operations=True,
                track_costs=False,
                instrument_fastapi=True,
                instrument_httpx=False,
                console_exporter=True,
            ),
        )

        resource_mock = MagicMock()
        resource_mock.create.return_value = MagicMock()

        with (
            patch.object(init_module, "_configure_instrumentations") as configure,
            patch.dict(
                "sys.modules",
                _otel_modules(resource_factory=resource_mock),
            ),
        ):
            assert initialize_observability(settings) is True

        configure.assert_called_once()
        instrumentation = tuple(configure.call_args.args[0])
        assert instrumentation == ("fastapi", "logging")

    def test_reinitialisation_is_idempotent(self) -> None:
        """Confirm repeated initialisation calls keep state stable."""

        with patch.dict("sys.modules", _otel_modules()):
            initialize_observability(ObservabilityConfig())
            initialize_observability(ObservabilityConfig())

    def test_shutdown_resets_state(self) -> None:
        """Check shutdown clears providers and invokes resource cleanup."""

        tracer_provider = MagicMock()
        meter_provider = MagicMock()
        init_module._STATE.tracer_provider = tracer_provider
        init_module._STATE.meter_provider = meter_provider

        shutdown_observability()

        tracer_provider.shutdown.assert_called_once()
        meter_provider.shutdown.assert_called_once()
        assert init_module._STATE.tracer_provider is None
        assert init_module._STATE.meter_provider is None
