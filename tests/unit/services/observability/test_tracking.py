"""Tests for OpenTelemetry tracking utilities."""

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.services.observability.tracking import _initialize_metrics
from src.services.observability.tracking import _NoOpCounter
from src.services.observability.tracking import _NoOpHistogram
from src.services.observability.tracking import _NoOpMeter
from src.services.observability.tracking import _NoOpSpan
from src.services.observability.tracking import _NoOpTracer
from src.services.observability.tracking import get_meter
from src.services.observability.tracking import get_tracer
from src.services.observability.tracking import instrument_function
from src.services.observability.tracking import record_ai_operation
from src.services.observability.tracking import track_cost


class TestTracerAndMeter:
    """Test tracer and meter acquisition."""

    @patch("src.services.observability.tracking.trace")
    def test_get_tracer_success(self, mock_trace):
        """Test successful tracer acquisition."""
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        tracer = get_tracer("test-service")

        assert tracer == mock_tracer
        mock_trace.get_tracer.assert_called_once_with("test-service")

    def test_get_tracer_import_error(self):
        """Test tracer acquisition with import error."""
        with patch("src.services.observability.tracking.trace") as mock_trace:
            mock_trace.side_effect = ImportError("OpenTelemetry not available")

            tracer = get_tracer("test-service")

            assert isinstance(tracer, _NoOpTracer)

    @patch("src.services.observability.tracking.metrics")
    def test_get_meter_success(self, mock_metrics):
        """Test successful meter acquisition."""
        mock_meter = MagicMock()
        mock_metrics.get_meter.return_value = mock_meter

        meter = get_meter("test-service")

        assert meter == mock_meter
        mock_metrics.get_meter.assert_called_once_with("test-service")

    def test_get_meter_import_error(self):
        """Test meter acquisition with import error."""
        with patch("src.services.observability.tracking.metrics") as mock_metrics:
            mock_metrics.side_effect = ImportError("OpenTelemetry not available")

            meter = get_meter("test-service")

            assert isinstance(meter, _NoOpMeter)


class TestInstrumentFunction:
    """Test function instrumentation decorator."""

    @patch("src.services.observability.tracking.get_tracer")
    def test_instrument_async_function(self, mock_get_tracer):
        """Test instrumenting async function."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        @instrument_function(operation_type="test_operation")
        async def test_async_function(arg1, arg2=None):
            return f"result-{arg1}-{arg2}"

        # Test function execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                test_async_function("value1", arg2="value2")
            )

            assert result == "result-value1-value2"

            # Verify span attributes
            mock_span.set_attribute.assert_any_call("operation.type", "test_operation")
            mock_span.set_attribute.assert_any_call(
                "function.name", "test_async_function"
            )
            mock_span.set_attribute.assert_any_call("function.success", True)

        finally:
            loop.close()

    @patch("src.services.observability.tracking.get_tracer")
    def test_instrument_sync_function(self, mock_get_tracer):
        """Test instrumenting sync function."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        @instrument_function(operation_type="test_operation")
        def test_sync_function(arg1, arg2=None):
            return f"result-{arg1}-{arg2}"

        result = test_sync_function("value1", arg2="value2")

        assert result == "result-value1-value2"

        # Verify span attributes
        mock_span.set_attribute.assert_any_call("operation.type", "test_operation")
        mock_span.set_attribute.assert_any_call("function.name", "test_sync_function")
        mock_span.set_attribute.assert_any_call("function.success", True)

    @patch("src.services.observability.tracking.get_tracer")
    def test_instrument_function_with_args_recording(self, mock_get_tracer):
        """Test function instrumentation with argument recording."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        @instrument_function(record_args=True)
        def test_function(arg1: str, arg2: int, complex_arg: dict):
            return "success"

        test_function("simple", 42, {"complex": "data"})

        # Verify simple arguments were recorded
        mock_span.set_attribute.assert_any_call("function.arg.0", "simple")
        mock_span.set_attribute.assert_any_call("function.arg.1", "42")
        # Complex arguments should not be recorded

    @patch("src.services.observability.tracking.get_tracer")
    def test_instrument_function_with_result_recording(self, mock_get_tracer):
        """Test function instrumentation with result recording."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        @instrument_function(record_result=True)
        def test_function():
            return "simple_result"

        result = test_function()

        assert result == "simple_result"
        mock_span.set_attribute.assert_any_call("function.result", "simple_result")

    @patch("src.services.observability.tracking.get_tracer")
    def test_instrument_function_exception_handling(self, mock_get_tracer):
        """Test function instrumentation with exception handling."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_get_tracer.return_value = mock_tracer

        @instrument_function()
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()

        # Verify error attributes
        mock_span.set_attribute.assert_any_call("function.success", False)
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")
        mock_span.record_exception.assert_called_once()

    @patch("src.services.observability.tracking.get_tracer")
    def test_instrument_function_custom_span_name(self, mock_get_tracer):
        """Test function instrumentation with custom span name."""
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        @instrument_function(span_name="custom.operation")
        def test_function():
            return "result"

        test_function()

        mock_tracer.start_as_current_span.assert_called_with("custom.operation")


class TestMetricsTracking:
    """Test AI operation and cost tracking."""

    def setup_method(self):
        """Reset global metrics before each test."""
        import src.services.observability.tracking as tracking_module

        tracking_module._ai_operation_duration = None
        tracking_module._ai_operation_counter = None
        tracking_module._ai_cost_counter = None
        tracking_module._ai_token_counter = None

    @patch("src.services.observability.tracking.get_meter")
    def test_initialize_metrics(self, mock_get_meter):
        """Test metrics initialization."""
        mock_meter = MagicMock()
        mock_get_meter.return_value = mock_meter

        # Setup mock metrics
        mock_histogram = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_counter.return_value = mock_counter

        _initialize_metrics()

        # Verify metrics were created
        assert mock_meter.create_histogram.call_count == 1
        assert mock_meter.create_counter.call_count == 3

    @patch("src.services.observability.tracking._initialize_metrics")
    @patch("src.services.observability.tracking._ai_operation_counter", new=MagicMock())
    @patch(
        "src.services.observability.tracking._ai_operation_duration", new=MagicMock()
    )
    @patch("src.services.observability.tracking._ai_token_counter", new=MagicMock())
    def test_record_ai_operation(self, mock_init_metrics):
        """Test recording AI operation metrics."""
        import src.services.observability.tracking as tracking_module

        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_token_counter = MagicMock()

        tracking_module._ai_operation_counter = mock_counter
        tracking_module._ai_operation_duration = mock_histogram
        tracking_module._ai_token_counter = mock_token_counter

        record_ai_operation(
            operation_type="embedding",
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=100,
            output_tokens=50,
            duration=1.5,
            success=True,
        )

        # Verify operation counter was called
        expected_attrs = {
            "operation_type": "embedding",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "success": "True",
        }
        mock_counter.add.assert_called_once_with(1, expected_attrs)

        # Verify duration was recorded
        mock_histogram.record.assert_called_once_with(1.5, expected_attrs)

        # Verify token counts
        input_attrs = {**expected_attrs, "token_type": "input"}
        output_attrs = {**expected_attrs, "token_type": "output"}
        assert mock_token_counter.add.call_count == 2
        mock_token_counter.add.assert_any_call(100, input_attrs)
        mock_token_counter.add.assert_any_call(50, output_attrs)

    @patch("src.services.observability.tracking._initialize_metrics")
    @patch("src.services.observability.tracking._ai_cost_counter", new=MagicMock())
    def test_track_cost(self, mock_init_metrics):
        """Test cost tracking."""
        import src.services.observability.tracking as tracking_module

        mock_cost_counter = MagicMock()
        tracking_module._ai_cost_counter = mock_cost_counter

        track_cost(
            operation_type="completion",
            provider="openai",
            cost_usd=0.05,
            model="gpt-3.5-turbo",
            custom_attribute="value",
        )

        expected_attrs = {
            "operation_type": "completion",
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "custom_attribute": "value",
        }
        mock_cost_counter.add.assert_called_once_with(0.05, expected_attrs)

    def test_record_ai_operation_exception_handling(self):
        """Test AI operation recording handles exceptions gracefully."""
        # This should not raise an exception even if metrics are not initialized
        record_ai_operation(
            operation_type="embedding",
            provider="openai",
            success=True,
        )

    def test_track_cost_exception_handling(self):
        """Test cost tracking handles exceptions gracefully."""
        # This should not raise an exception even if metrics are not initialized
        track_cost(
            operation_type="completion",
            provider="openai",
            cost_usd=0.05,
        )


class TestNoOpImplementations:
    """Test NoOp implementations for when OpenTelemetry is not available."""

    def test_noop_tracer(self):
        """Test NoOp tracer implementation."""
        tracer = _NoOpTracer()

        span = tracer.start_as_current_span("test-span")
        assert isinstance(span, _NoOpSpan)

    def test_noop_span(self):
        """Test NoOp span implementation."""
        span = _NoOpSpan()

        # Context manager should work
        with span:
            pass

        # Methods should not raise exceptions
        span.set_attribute("key", "value")
        span.record_exception(Exception("test"))

    def test_noop_meter(self):
        """Test NoOp meter implementation."""
        meter = _NoOpMeter()

        histogram = meter.create_histogram("test-histogram")
        assert isinstance(histogram, _NoOpHistogram)

        counter = meter.create_counter("test-counter")
        assert isinstance(counter, _NoOpCounter)

    def test_noop_histogram(self):
        """Test NoOp histogram implementation."""
        histogram = _NoOpHistogram()

        # Should not raise exception
        histogram.record(1.5, {"key": "value"})

    def test_noop_counter(self):
        """Test NoOp counter implementation."""
        counter = _NoOpCounter()

        # Should not raise exception
        counter.add(1, {"key": "value"})
        counter.add(1.5, {"key": "value"})


class TestMetricsInitializationExceptionHandling:
    """Test metrics initialization exception handling."""

    def setup_method(self):
        """Reset global metrics before each test."""
        import src.services.observability.tracking as tracking_module

        tracking_module._ai_operation_duration = None
        tracking_module._ai_operation_counter = None
        tracking_module._ai_cost_counter = None
        tracking_module._ai_token_counter = None

    @patch("src.services.observability.tracking.get_meter")
    def test_initialize_metrics_exception_handling(self, mock_get_meter):
        """Test metrics initialization handles exceptions gracefully."""
        mock_get_meter.side_effect = Exception("Meter creation failed")

        # Should not raise exception
        _initialize_metrics()

        # Verify metrics remain None
        import src.services.observability.tracking as tracking_module

        assert tracking_module._ai_operation_duration is None
        assert tracking_module._ai_operation_counter is None
        assert tracking_module._ai_cost_counter is None
        assert tracking_module._ai_token_counter is None
