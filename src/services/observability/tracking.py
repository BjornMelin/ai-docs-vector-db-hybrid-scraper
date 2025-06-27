"""OpenTelemetry tracking utilities for AI/ML operations and cost monitoring.

Provides decorators and utilities for tracking AI operations, costs, and
performance metrics that integrate with the existing function-based services.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

# Optional OpenTelemetry imports - handled at runtime
try:
    from opentelemetry import trace, metrics
except ImportError:
    trace = None
    metrics = None


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Global metrics for AI operations
_ai_operation_duration: Any = None
_ai_operation_counter: Any = None
_ai_cost_counter: Any = None
_ai_token_counter: Any = None


def get_tracer(name: str = "ai-docs-vector-db") -> Any:
    """Get a tracer instance for creating spans.

    Args:
        name: Tracer name (typically module or component name)

    Returns:
        OpenTelemetry Tracer instance or NoOpTracer if not initialized
    """
    try:
        if trace is None:
            raise ImportError("OpenTelemetry trace not available")
        return trace.get_tracer(name)
    except ImportError:
        logger.warning("OpenTelemetry not available, returning NoOp tracer")
        return _NoOpTracer()


def get_meter(name: str = "ai-docs-vector-db") -> Any:
    """Get a meter instance for creating metrics.

    Args:
        name: Meter name (typically module or component name)

    Returns:
        OpenTelemetry Meter instance or NoOpMeter if not initialized
    """
    try:
        if metrics is None:
            raise ImportError("OpenTelemetry metrics not available")
        return metrics.get_meter(name)
    except ImportError:
        logger.warning("OpenTelemetry not available, returning NoOp meter")
        return _NoOpMeter()


def _initialize_metrics() -> None:
    """Initialize AI-specific metrics."""
    global \
        _ai_operation_duration, \
        _ai_operation_counter, \
        _ai_cost_counter, \
        _ai_token_counter

    try:
        meter = get_meter("ai-operations")

        _ai_operation_duration = meter.create_histogram(
            "ai_operation_duration_seconds",
            description="Duration of AI operations (embeddings, LLM calls)",
            unit="s",
        )

        _ai_operation_counter = meter.create_counter(
            "ai_operations_total",
            description="Total number of AI operations",
        )

        _ai_cost_counter = meter.create_counter(
            "ai_operation_cost_total",
            description="Total cost of AI operations",
            unit="USD",
        )

        _ai_token_counter = meter.create_counter(
            "ai_tokens_total",
            description="Total tokens processed by AI operations",
        )

    except Exception as e:
        logger.warning(f"Failed to initialize AI metrics: {e}")


def instrument_function(
    operation_type: str = "function",
    span_name: str | None = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to instrument functions with OpenTelemetry tracing.

    Args:
        operation_type: Type of operation for labeling
        span_name: Custom span name (defaults to function name)
        record_args: Whether to record function arguments as span attributes
        record_result: Whether to record return value (use carefully)

    Returns:
        Decorated function with tracing
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(name) as span:
                # Add function metadata
                span.set_attribute("operation.type", operation_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Record arguments if requested
                if record_args:
                    try:
                        # Only record simple types to avoid large attributes
                        for i, arg in enumerate(args[:3]):  # Limit to first 3 args
                            if isinstance(arg, str | int | float | bool):
                                span.set_attribute(f"function.arg.{i}", str(arg)[:100])

                        for key, value in list(kwargs.items())[:5]:  # Limit to 5 kwargs
                            if isinstance(value, str | int | float | bool):
                                span.set_attribute(
                                    f"function.kwarg.{key}", str(value)[:100]
                                )
                    except Exception as e:
                        logger.debug(f"Failed to record function arguments: {e}")

                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)

                    # Record result if requested (be careful with large objects)
                    if record_result and isinstance(result, str | int | float | bool):
                        span.set_attribute("function.result", str(result)[:100])

                    return result

                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e)[:200])
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(name) as span:
                # Add function metadata
                span.set_attribute("operation.type", operation_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Record arguments if requested
                if record_args:
                    try:
                        for i, arg in enumerate(args[:3]):
                            if isinstance(arg, str | int | float | bool):
                                span.set_attribute(f"function.arg.{i}", str(arg)[:100])

                        for key, value in list(kwargs.items())[:5]:
                            if isinstance(value, str | int | float | bool):
                                span.set_attribute(
                                    f"function.kwarg.{key}", str(value)[:100]
                                )
                    except Exception as e:
                        logger.debug(f"Failed to record function arguments: {e}")

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)

                    if record_result and isinstance(result, str | int | float | bool):
                        span.set_attribute("function.result", str(result)[:100])

                    return result

                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e)[:200])
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def record_ai_operation(
    operation_type: str,
    provider: str,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    duration: float | None = None,
    success: bool = True,
    **attributes,
) -> None:
    """Record metrics for an AI operation (embedding, LLM call, etc.).

    Args:
        operation_type: Type of AI operation (embedding, completion, etc.)
        provider: AI service provider (openai, fastembed, etc.)
        model: Model name if applicable
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        duration: Operation duration in seconds
        success: Whether operation succeeded
        **attributes: Additional attributes to record
    """
    # Initialize metrics if not done
    if _ai_operation_counter is None:
        _initialize_metrics()

    try:
        # Common attributes
        attrs = {
            "operation_type": operation_type,
            "provider": provider,
            "success": str(success),
        }

        if model:
            attrs["model"] = model

        # Add custom attributes
        attrs.update(attributes)

        # Record operation count
        if _ai_operation_counter:
            _ai_operation_counter.add(1, attrs)

        # Record duration
        if duration and _ai_operation_duration:
            _ai_operation_duration.record(duration, attrs)

        # Record token usage
        if _ai_token_counter:
            if input_tokens:
                token_attrs = {**attrs, "token_type": "input"}
                _ai_token_counter.add(input_tokens, token_attrs)

            if output_tokens:
                token_attrs = {**attrs, "token_type": "output"}
                _ai_token_counter.add(output_tokens, token_attrs)

    except Exception as e:
        logger.warning(f"Failed to record AI operation metrics: {e}")


def track_cost(
    operation_type: str,
    provider: str,
    cost_usd: float,
    model: str | None = None,
    **attributes,
) -> None:
    """Track the cost of an AI operation.

    Args:
        operation_type: Type of AI operation
        provider: AI service provider
        cost_usd: Cost in USD
        model: Model name if applicable
        **attributes: Additional attributes
    """
    if _ai_cost_counter is None:
        _initialize_metrics()

    try:
        attrs = {
            "operation_type": operation_type,
            "provider": provider,
        }

        if model:
            attrs["model"] = model

        attrs.update(attributes)

        if _ai_cost_counter:
            _ai_cost_counter.add(cost_usd, attrs)

    except Exception as e:
        logger.warning(f"Failed to record AI cost: {e}")


# NoOp implementations for when OpenTelemetry is not available
class _NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, _name: str, **_kwargs):
        return _NoOpSpan()


class _NoOpSpan:
    """No-op span when OpenTelemetry is not available."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


class _NoOpMeter:
    """No-op meter when OpenTelemetry is not available."""

    def create_histogram(self, _name: str, **_kwargs):
        return _NoOpHistogram()

    def create_counter(self, _name: str, **_kwargs):
        return _NoOpCounter()


class _NoOpHistogram:
    """No-op histogram when OpenTelemetry is not available."""

    def record(self, value: float, attributes: dict | None = None) -> None:
        pass


class _NoOpCounter:
    """No-op counter when OpenTelemetry is not available."""

    def add(self, value: float | int, attributes: dict | None = None) -> None:
        pass
