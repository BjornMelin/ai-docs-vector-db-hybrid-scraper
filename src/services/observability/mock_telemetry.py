"""Mock telemetry classes for when OpenTelemetry is not available."""

import logging
from typing import Any


logger = logging.getLogger(__name__)


class OTLPSpanExporter:
    """Mock OTLP span exporter."""

    def __init__(self, *args, **kwargs):
        pass


class MeterProvider:
    """Mock meter provider."""

    def __init__(self, *args, **kwargs):
        pass


class PeriodicExportingMetricReader:
    """Mock periodic exporting metric reader."""

    def __init__(self, *args, **kwargs):
        pass


class Resource:
    """Mock resource class."""

    @staticmethod
    def create(*_args, **_kwargs):
        return None


class TracerProvider:
    """Mock tracer provider."""

    def __init__(self, *args, **kwargs):
        pass


class BatchSpanProcessor:
    """Mock batch span processor."""

    def __init__(self, *args, **kwargs):
        pass


class MockMetrics:
    """Mock metrics API."""

    def __init__(self) -> None:
        self._provider = None

    def set_meter_provider(self, provider, *args, **kwargs):
        self._provider = provider

    def get_meter_provider(self):
        return self._provider


class MockTrace:
    """Mock trace API."""

    def __init__(self) -> None:
        self._provider = None

    def set_tracer_provider(self, provider, *args, **kwargs):
        self._provider = provider

    def get_tracer_provider(self):
        return self._provider


def create_mock_telemetry() -> tuple[Any, ...]:
    """Create mock telemetry components.

    Returns:
        Tuple of mock telemetry components
    """
    return (
        OTLPSpanExporter,
        MeterProvider,
        PeriodicExportingMetricReader,
        Resource,
        TracerProvider,
        BatchSpanProcessor,
        MockMetrics(),
        MockTrace(),
    )
