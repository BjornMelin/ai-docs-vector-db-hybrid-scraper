"""Tests for observability bootstrap logic."""

# ruff: noqa: UP037  # Forward references stay quoted for pylint compatibility.

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

from src.services.observability import init as init_module
from src.services.observability.config import (
    DEFAULT_INSTRUMENTATIONS,
    ObservabilityConfig,
)


@dataclass
class FakeOtelContext:  # pylint: disable=too-many-instance-attributes
    """Container exposing fake OpenTelemetry components for assertions."""

    trace_api: _TraceAPI
    metrics_api: _MetricsAPI
    tracer_provider_cls: type
    meter_provider_cls: type
    span_exporter_cls: type
    metric_exporter_cls: type
    batch_processor_cls: type
    console_exporter_cls: type
    metric_reader_cls: type
    resource_factory: type
    instrumentation_calls: dict[str, list[dict[str, Any]]]


class _TraceAPI:
    """Capture tracer provider registrations."""

    def __init__(self) -> None:
        """Initialize the trace API."""
        self.providers: list[Any] = []
        self.provider: Any | None = None

    def set_tracer_provider(self, provider: Any) -> None:
        """Set the tracer provider."""
        self.providers.append(provider)
        self.provider = provider  # type: ignore[attr-defined]


class _MetricsAPI:
    """Capture meter provider registrations."""

    def __init__(self) -> None:
        """Initialize the metrics API."""
        self.providers: list[Any] = []
        self.provider: Any | None = None

    def set_meter_provider(self, provider: Any) -> None:
        """Set the meter provider."""
        self.providers.append(provider)
        self.provider = provider  # type: ignore[attr-defined]


def _create_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


@pytest.fixture(autouse=True)
def _reset_state() -> Generator[None, None, None]:
    """Ensure observability state is isolated across tests."""
    init_module._STATE.tracer_provider = None
    init_module._STATE.meter_provider = None
    yield
    init_module._STATE.tracer_provider = None
    init_module._STATE.meter_provider = None


@pytest.fixture(name="fake_otel")
def fixture_fake_otel(  # pylint: disable=too-many-locals,too-many-statements
    monkeypatch: pytest.MonkeyPatch,
) -> FakeOtelContext:
    """Provide fake OpenTelemetry modules so imports remain local."""
    trace_api = _TraceAPI()
    metrics_api = _MetricsAPI()

    instrumentation_calls: dict[str, list[dict[str, Any]]] = {
        "fastapi": [],
        "httpx": [],
        "logging": [],
        "requests": [],
    }

    def _instrumentor(name: str) -> type:
        calls = instrumentation_calls[name]

        class _Instrumentor:  # pylint: disable=too-few-public-methods
            def instrument(self, **kwargs: Any) -> None:
                calls.append(kwargs)

        return _Instrumentor

    class _FakeResource:
        def __init__(self, attributes: Any) -> None:
            self.attributes = dict(attributes)

    class _FakeResourceFactory:
        last_attributes: dict[str, Any] | None = None

        @classmethod
        def create(cls, attributes: Any) -> _FakeResource:
            cls.last_attributes = dict(attributes)
            return _FakeResource(attributes)

    class _FakeBatchSpanProcessor:
        instances: ClassVar[list["_FakeBatchSpanProcessor"]] = []

        def __init__(self, exporter: Any) -> None:
            self.exporter = exporter
            self.__class__.instances.append(self)

    class _FakeConsoleSpanExporter:
        instances: ClassVar[list["_FakeConsoleSpanExporter"]] = []

        def __init__(self) -> None:
            self.__class__.instances.append(self)

    class _FakeTracerProvider:
        instances: ClassVar[list["_FakeTracerProvider"]] = []

        def __init__(self, resource: Any) -> None:
            self.resource = resource
            self.processors: list[Any] = []
            self.shutdown_called = False
            self.__class__.instances.append(self)

        def add_span_processor(self, processor: Any) -> None:
            self.processors.append(processor)

        def shutdown(self) -> None:
            self.shutdown_called = True

    class _FakeMeterProvider:
        instances: ClassVar[list["_FakeMeterProvider"]] = []

        def __init__(self, resource: Any, metric_readers: list[Any]) -> None:
            self.resource = resource
            self.metric_readers = list(metric_readers)
            self.shutdown_called = False
            self.__class__.instances.append(self)

        def shutdown(self) -> None:
            self.shutdown_called = True

    class _FakePeriodicExportingMetricReader:
        instances: ClassVar[list["_FakePeriodicExportingMetricReader"]] = []

        def __init__(self, exporter: Any) -> None:
            self.exporter = exporter
            self.__class__.instances.append(self)

    class _FakeOTLPSpanExporter:
        instances: ClassVar[list["_FakeOTLPSpanExporter"]] = []
        should_fail: ClassVar[bool] = False

        def __init__(
            self, *, endpoint: str, headers: dict[str, str], insecure: bool
        ) -> None:
            if self.__class__.should_fail:
                raise RuntimeError("span exporter failure")
            self.endpoint = endpoint
            self.headers = dict(headers)
            self.insecure = insecure
            self.__class__.instances.append(self)

    class _FakeOTLPMetricExporter:
        instances: ClassVar[list["_FakeOTLPMetricExporter"]] = []

        def __init__(
            self, *, endpoint: str, headers: dict[str, str], insecure: bool
        ) -> None:
            self.endpoint = endpoint
            self.headers = dict(headers)
            self.insecure = insecure
            self.__class__.instances.append(self)

    fastapi_module = _create_module(
        "opentelemetry.instrumentation.fastapi",
        FastAPIInstrumentor=_instrumentor("fastapi"),
    )
    httpx_module = _create_module(
        "opentelemetry.instrumentation.httpx",
        HTTPXClientInstrumentor=_instrumentor("httpx"),
    )
    requests_module = _create_module(
        "opentelemetry.instrumentation.requests",
        RequestsInstrumentor=_instrumentor("requests"),
    )
    logging_module = _create_module(
        "opentelemetry.instrumentation.logging",
        LoggingInstrumentor=_instrumentor("logging"),
    )

    metric_exporter_module = _create_module(
        "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
        OTLPMetricExporter=_FakeOTLPMetricExporter,
    )
    trace_exporter_module = _create_module(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        OTLPSpanExporter=_FakeOTLPSpanExporter,
    )
    grpc_module = _create_module(
        "opentelemetry.exporter.otlp.proto.grpc",
        metric_exporter=metric_exporter_module,
        trace_exporter=trace_exporter_module,
    )
    proto_module = _create_module(
        "opentelemetry.exporter.otlp.proto",
        grpc=grpc_module,
    )
    otlp_module = _create_module(
        "opentelemetry.exporter.otlp",
        proto=proto_module,
    )
    exporter_module = _create_module(
        "opentelemetry.exporter",
        otlp=otlp_module,
    )

    sdk_trace_export_module = _create_module(
        "opentelemetry.sdk.trace.export",
        BatchSpanProcessor=_FakeBatchSpanProcessor,
        ConsoleSpanExporter=_FakeConsoleSpanExporter,
    )
    sdk_trace_module = _create_module(
        "opentelemetry.sdk.trace",
        TracerProvider=_FakeTracerProvider,
        export=sdk_trace_export_module,
    )
    sdk_metrics_export_module = _create_module(
        "opentelemetry.sdk.metrics.export",
        PeriodicExportingMetricReader=_FakePeriodicExportingMetricReader,
    )
    sdk_metrics_module = _create_module(
        "opentelemetry.sdk.metrics",
        MeterProvider=_FakeMeterProvider,
        export=sdk_metrics_export_module,
    )
    sdk_resources_module = _create_module(
        "opentelemetry.sdk.resources",
        Resource=_FakeResourceFactory,
    )
    sdk_module = _create_module(
        "opentelemetry.sdk",
        trace=sdk_trace_module,
        metrics=sdk_metrics_module,
        resources=sdk_resources_module,
    )

    trace_module = _create_module(
        "opentelemetry.trace", set_tracer_provider=trace_api.set_tracer_provider
    )
    metrics_module = _create_module(
        "opentelemetry.metrics",
        set_meter_provider=metrics_api.set_meter_provider,
    )
    root_module = _create_module(
        "opentelemetry",
        trace=trace_module,
        metrics=metrics_module,
        exporter=exporter_module,
        sdk=sdk_module,
    )

    modules = {
        "opentelemetry": root_module,
        "opentelemetry.trace": trace_module,
        "opentelemetry.metrics": metrics_module,
        "opentelemetry.exporter": exporter_module,
        "opentelemetry.exporter.otlp": otlp_module,
        "opentelemetry.exporter.otlp.proto": proto_module,
        "opentelemetry.exporter.otlp.proto.grpc": grpc_module,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": trace_exporter_module,
        "opentelemetry.exporter.otlp.proto.grpc.metric_exporter": (
            metric_exporter_module
        ),
        "opentelemetry.sdk": sdk_module,
        "opentelemetry.sdk.trace": sdk_trace_module,
        "opentelemetry.sdk.trace.export": sdk_trace_export_module,
        "opentelemetry.sdk.metrics": sdk_metrics_module,
        "opentelemetry.sdk.metrics.export": sdk_metrics_export_module,
        "opentelemetry.sdk.resources": sdk_resources_module,
        "opentelemetry.instrumentation.fastapi": fastapi_module,
        "opentelemetry.instrumentation.httpx": httpx_module,
        "opentelemetry.instrumentation.requests": requests_module,
        "opentelemetry.instrumentation.logging": logging_module,
    }

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return FakeOtelContext(
        trace_api=trace_api,
        metrics_api=metrics_api,
        tracer_provider_cls=_FakeTracerProvider,
        meter_provider_cls=_FakeMeterProvider,
        span_exporter_cls=_FakeOTLPSpanExporter,
        metric_exporter_cls=_FakeOTLPMetricExporter,
        batch_processor_cls=_FakeBatchSpanProcessor,
        console_exporter_cls=_FakeConsoleSpanExporter,
        metric_reader_cls=_FakePeriodicExportingMetricReader,
        resource_factory=_FakeResourceFactory,
        instrumentation_calls=instrumentation_calls,
    )


class TestConfigurationCoercion:
    """Validate configuration coercion helpers."""

    def test_coerce_config_rejects_unknown_type(self) -> None:
        """Only supported configuration inputs are accepted."""
        with pytest.raises(TypeError):
            init_module._coerce_config(object())  # type: ignore[arg-type]

    def test_coerce_settings_with_defaults(self) -> None:
        """Settings lacking explicit toggles fall back to defaults."""
        settings = cast(
            init_module.SettingsLike,
            SimpleNamespace(
                app_name="docs",
                version="1.2.3",
                environment=SimpleNamespace(value="prod"),
                observability=SimpleNamespace(enabled=True),
            ),
        )

        coerced = init_module._coerce_config(settings)
        assert coerced.instrumentations == DEFAULT_INSTRUMENTATIONS
        assert coerced.environment == "prod"
        assert coerced.service_name == "docs"
        assert coerced.service_version == "1.2.3"

    def test_coerce_settings_respects_explicit_disables(self) -> None:
        """Explicit instrumentation opt-outs prevent defaults from reapplying."""
        observed = SimpleNamespace(
            enabled=True,
            instrument_fastapi=False,
            instrument_httpx=False,
            track_ai_operations=False,
            track_costs=False,
        )
        settings = cast(
            init_module.SettingsLike,
            SimpleNamespace(
                app_name="docs",
                version="1.2.3",
                environment="staging",
                observability=observed,
            ),
        )

        coerced = init_module._coerce_config(settings)
        assert coerced.instrumentations == ()

    def test_coerce_settings_enables_logging_when_tracking(self) -> None:
        """Tracking flags ensure logging instrumentation is configured."""
        observed = SimpleNamespace(
            enabled=True,
            instrument_fastapi=True,
            instrument_httpx=True,
            track_ai_operations=True,
            track_costs=False,
        )
        settings = cast(
            init_module.SettingsLike,
            SimpleNamespace(
                app_name="docs",
                version="1.2.3",
                environment="staging",
                observability=observed,
            ),
        )

        coerced = init_module._coerce_config(settings)
        assert coerced.instrumentations == ("fastapi", "httpx", "logging")


class TestInitializeObservability:
    """Exercise initialization behaviour across branches."""

    def test_disabled_configuration_short_circuits(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Disabled configs log an informational skip message."""
        caplog.set_level(logging.INFO)
        config = ObservabilityConfig(enabled=False)

        assert init_module.initialize_observability(config) is False
        assert "Observability disabled" in caplog.text
        assert init_module._STATE.tracer_provider is None
        assert init_module._STATE.meter_provider is None

    def test_successful_initialization_wires_components(
        self, fake_otel: FakeOtelContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Providers, exporters, and instrumentations are configured once."""
        caplog.set_level(logging.INFO)
        config = ObservabilityConfig(
            enabled=True,
            service_name="docs",
            service_version="2.0.0",
            otlp_endpoint="http://collector:4317",
            otlp_headers={"authorization": "token"},
            insecure_transport=False,
            metrics_enabled=True,
            console_exporter=True,
            instrumentations=("fastapi", "logging"),
        )

        assert init_module.initialize_observability(config) is True
        assert (
            init_module._STATE.tracer_provider
            is fake_otel.tracer_provider_cls.instances[0]
        )
        assert fake_otel.trace_api.providers == [
            fake_otel.tracer_provider_cls.instances[0]
        ]
        assert len(fake_otel.batch_processor_cls.instances) == 2
        exporters = [
            processor.exporter for processor in fake_otel.batch_processor_cls.instances
        ]
        span_exporter = fake_otel.span_exporter_cls.instances[0]
        assert span_exporter.endpoint == "http://collector:4317"
        assert span_exporter.headers == {"authorization": "token"}
        assert span_exporter.insecure is False
        assert span_exporter in exporters
        assert any(isinstance(exp, fake_otel.console_exporter_cls) for exp in exporters)
        assert len(fake_otel.console_exporter_cls.instances) == 1

        assert (
            init_module._STATE.meter_provider
            is fake_otel.meter_provider_cls.instances[0]
        )
        assert fake_otel.metrics_api.providers == [
            fake_otel.meter_provider_cls.instances[0]
        ]
        metric_reader = fake_otel.metric_reader_cls.instances[0]
        metric_exporter = fake_otel.metric_exporter_cls.instances[0]
        assert metric_reader.exporter is metric_exporter
        assert metric_exporter.endpoint == "http://collector:4317"
        assert (
            fake_otel.resource_factory.last_attributes == config.resource_attributes()
        )

        assert fake_otel.instrumentation_calls["fastapi"] == [{}]
        assert fake_otel.instrumentation_calls["logging"] == [
            {"set_logging_format": True}
        ]
        assert "Observability initialized" in caplog.text

    def test_metrics_disabled_skips_meter_provider(
        self, fake_otel: FakeOtelContext
    ) -> None:
        """Metrics disabled configuration avoids meter provider setup."""
        config = ObservabilityConfig(
            enabled=True,
            metrics_enabled=False,
            instrumentations=(),
        )

        assert init_module.initialize_observability(config) is True
        assert init_module._STATE.meter_provider is None
        assert fake_otel.metrics_api.providers == []

    def test_initialization_is_idempotent(self, fake_otel: FakeOtelContext) -> None:
        """Subsequent calls do not recreate exporters or providers."""
        config = ObservabilityConfig(
            enabled=True,
            instrumentations=("fastapi",),
            metrics_enabled=True,
        )

        assert init_module.initialize_observability(config) is True
        first_tracer_instances = list(fake_otel.tracer_provider_cls.instances)
        first_span_exporters = list(fake_otel.span_exporter_cls.instances)
        first_metric_exporters = list(fake_otel.metric_exporter_cls.instances)
        first_instrument_calls = {
            name: list(calls) for name, calls in fake_otel.instrumentation_calls.items()
        }

        assert init_module.initialize_observability(config) is True
        assert fake_otel.tracer_provider_cls.instances == first_tracer_instances
        assert fake_otel.span_exporter_cls.instances == first_span_exporters
        assert fake_otel.metric_exporter_cls.instances == first_metric_exporters
        assert {
            name: list(calls) for name, calls in fake_otel.instrumentation_calls.items()
        } == first_instrument_calls

    def test_exporter_failure_logs_and_cleans_up(
        self, fake_otel: FakeOtelContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exporter constructor errors log and leave providers unset."""
        fake_otel.span_exporter_cls.should_fail = True
        caplog.set_level(logging.ERROR)
        config = ObservabilityConfig(
            enabled=True,
            instrumentations=(),
            metrics_enabled=True,
        )

        try:
            assert init_module.initialize_observability(config) is False
        finally:
            fake_otel.span_exporter_cls.should_fail = False

        assert init_module._STATE.tracer_provider is None
        assert init_module._STATE.meter_provider is None
        assert fake_otel.trace_api.providers == []
        assert "Failed to initialize observability" in caplog.text

    def test_shutdown_resets_state(self, fake_otel: FakeOtelContext) -> None:
        """Shutdown flushes providers and clears cached state."""
        config = ObservabilityConfig(enabled=True, instrumentations=())
        assert init_module.initialize_observability(config) is True

        tracer_provider = cast(Any, init_module._STATE.tracer_provider)
        meter_provider = init_module._STATE.meter_provider

        init_module.shutdown_observability()

        assert tracer_provider is not None
        assert tracer_provider.shutdown_called is True
        if meter_provider is not None:
            assert cast(Any, meter_provider).shutdown_called is True
        assert init_module._STATE.tracer_provider is None
        assert init_module._STATE.meter_provider is None

    def test_missing_instrumentation_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing optional instrumentation surfaces a warning without raising."""
        caplog.set_level(logging.WARNING)
        original_import = __import__

        def _fake_import(
            name: str,
            globalns: Any = None,
            localns: Any = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name.startswith("opentelemetry.instrumentation.requests"):
                raise ImportError("missing module")
            return original_import(name, globalns, localns, fromlist, level)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        init_module._configure_instrumentations(["requests"])

        assert "Instrumentation 'requests' not installed" in caplog.text

    def test_unknown_instrumentation_is_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown instrumentation names emit a debug hint."""
        caplog.set_level(logging.DEBUG)
        init_module._configure_instrumentations(["custom"])
        assert "Unknown instrumentation" in caplog.text
