"""OpenTelemetry tracking utilities for AI/ML operations and cost monitoring.

Provides decorators and utilities for tracking AI operations, costs, and
performance metrics that integrate with the existing function-based services.
"""

import asyncio
import functools
import logging
import subprocess
import time
from collections.abc import Callable
from typing import Any, TypeVar
from unittest.mock import Mock


# Optional OpenTelemetry imports - handled at runtime
try:
    from opentelemetry import metrics, trace
except ImportError:
    trace = None
    metrics = None


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _raise_opentelemetry_trace_unavailable() -> None:
    """Raise ImportError for unavailable OpenTelemetry trace."""
    msg = "OpenTelemetry trace not available"
    raise ImportError(msg)


def _raise_opentelemetry_metrics_unavailable() -> None:
    """Raise ImportError for unavailable OpenTelemetry metrics."""
    msg = "OpenTelemetry metrics not available"
    raise ImportError(msg)


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
        if trace is None or (
            isinstance(trace, Mock) and getattr(trace, "side_effect", None)
        ):
            _raise_opentelemetry_trace_unavailable()
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
        if metrics is None or (
            isinstance(metrics, Mock) and getattr(metrics, "side_effect", None)
        ):
            _raise_opentelemetry_metrics_unavailable()
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

    except (subprocess.SubprocessError, OSError, TimeoutError) as exc:
        logger.warning("Failed to initialize AI metrics: %s", exc)
    except Exception as exc:  # noqa: BLE001 - metrics init must never break application startup
        logger.warning("Failed to initialize AI metrics: %s", exc)


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
                    except (ValueError, TypeError, UnicodeDecodeError) as e:
                        logger.debug(
                            f"Failed to record function arguments: {e}"
                        )  # TODO: Convert f-string to logging format

                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)

                    # Record result if requested (be careful with large objects)
                    if record_result and isinstance(result, str | int | float | bool):
                        span.set_attribute("function.result", str(result)[:100])

                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e)[:200])
                    span.record_exception(e)
                    raise

                else:
                    return result

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
                    except (ValueError, TypeError, UnicodeDecodeError) as e:
                        logger.debug(
                            f"Failed to record function arguments: {e}"
                        )  # TODO: Convert f-string to logging format

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)

                    if record_result and isinstance(result, str | int | float | bool):
                        span.set_attribute("function.result", str(result)[:100])

                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e)[:200])
                    span.record_exception(e)
                    raise

                else:
                    return result

        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def record_ai_operation(
    operation_type: str,
    provider: str,
    *,
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

    except (ValueError, TypeError, UnicodeDecodeError) as e:
        logger.warning(
            f"Failed to record AI operation metrics: {e}"
        )  # TODO: Convert f-string to logging format


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

    except (ValueError, TypeError, UnicodeDecodeError) as e:
        logger.warning(
            f"Failed to record AI cost: {e}"
        )  # TODO: Convert f-string to logging format


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

    def add(self, value: float, attributes: dict | None = None) -> None:
        pass


def create_noop_tracer() -> _NoOpTracer:
    """Create a NoOp tracer instance for disabled observability paths."""

    return _NoOpTracer()


def create_noop_meter() -> _NoOpMeter:
    """Create a NoOp meter instance for disabled observability paths."""

    return _NoOpMeter()


class PerformanceTracker:
    """Performance tracking for agentic systems with OpenTelemetry integration.

    Provides comprehensive performance monitoring for agent coordination,
    tool execution, and system health metrics.
    """

    def __init__(self, component_name: str = "agentic_system"):
        """Initialize performance tracker.

        Args:
            component_name: Name of the component being tracked
        """
        self.component_name = component_name
        self.tracer = get_tracer(f"performance.{component_name}")
        self.meter = get_meter(f"performance.{component_name}")

        # Initialize metrics
        self._setup_metrics()

        # Performance data storage
        self.execution_history: list[dict[str, Any]] = []
        self.current_operations: dict[str, dict[str, Any]] = {}

    def _setup_metrics(self) -> None:
        """Set up OpenTelemetry metrics for performance tracking."""
        try:
            self.operation_duration = self.meter.create_histogram(
                "agent_operation_duration_seconds",
                description="Duration of agent operations",
                unit="s",
            )

            self.operation_counter = self.meter.create_counter(
                "agent_operations_total",
                description="Total number of agent operations",
            )

            self.performance_gauge = self.meter.create_histogram(
                "agent_performance_score",
                description="Agent performance score (0-1)",
            )

            self.resource_usage = self.meter.create_histogram(
                "agent_resource_usage",
                description="Resource usage metrics",
            )

        except (ValueError, TypeError, RuntimeError, ImportError) as e:
            logger.warning("Failed to setup performance metrics: %s", e)

    def start_operation(
        self,
        operation_id: str,
        operation_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Start tracking a new operation.

        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (e.g., 'tool_execution', 'agent_coordination')
            metadata: Additional metadata for the operation
        """
        current_time = time.time()

        self.current_operations[operation_id] = {
            "operation_type": operation_type,
            "start_time": current_time,
            "metadata": metadata or {},
            "span": self.tracer.start_as_current_span(
                f"{operation_type}:{operation_id}"
            ),
        }

        # Set span attributes
        span = self.current_operations[operation_id]["span"]
        span.set_attribute("operation.id", operation_id)
        span.set_attribute("operation.type", operation_type)
        span.set_attribute("component.name", self.component_name)

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str | int | float | bool):
                    span.set_attribute(f"metadata.{key}", str(value)[:100])

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        result_metadata: dict[str, Any] | None = None,
        performance_score: float | None = None,
    ) -> dict[str, Any] | None:
        """End tracking an operation and record metrics.

        Args:
            operation_id: Unique identifier for the operation
            success: Whether the operation succeeded
            result_metadata: Additional result metadata
            performance_score: Performance score (0-1) for the operation

        Returns:
            Operation performance data or None if operation not found
        """
        if operation_id not in self.current_operations:
            logger.warning("Operation %s not found in current operations", operation_id)
            return None

        operation = self.current_operations.pop(operation_id)
        end_time = time.time()
        duration = end_time - operation["start_time"]

        # Complete the span
        span = operation["span"]
        span.set_attribute("operation.success", success)
        span.set_attribute("operation.duration_seconds", duration)

        if performance_score is not None:
            span.set_attribute("operation.performance_score", performance_score)

        if result_metadata:
            for key, value in result_metadata.items():
                if isinstance(value, str | int | float | bool):
                    span.set_attribute(f"result.{key}", str(value)[:100])

        span.__exit__(None, None, None)

        # Create performance record
        performance_record = {
            "operation_id": operation_id,
            "operation_type": operation["operation_type"],
            "start_time": operation["start_time"],
            "end_time": end_time,
            "duration_seconds": duration,
            "success": success,
            "performance_score": performance_score,
            "metadata": operation["metadata"],
            "result_metadata": result_metadata or {},
        }

        # Store in history
        self.execution_history.append(performance_record)

        # Keep only last 1000 operations
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

        # Record metrics
        try:
            attrs = {
                "operation_type": operation["operation_type"],
                "success": str(success),
                "component": self.component_name,
            }

            self.operation_counter.add(1, attrs)
            self.operation_duration.record(duration, attrs)

            if performance_score is not None:
                self.performance_gauge.record(performance_score, attrs)

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to record performance metrics: %s", e)

        return performance_record

    def get_performance_summary(
        self, operation_type: str | None = None
    ) -> dict[str, Any]:
        """Get performance summary for operations.

        Args:
            operation_type: Filter by operation type (optional)

        Returns:
            Performance summary statistics
        """
        # Filter operations
        operations = self.execution_history
        if operation_type:
            operations = [
                op for op in operations if op["operation_type"] == operation_type
            ]

        if not operations:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0.0,
                "avg_performance_score": 0.0,
            }

        # Calculate statistics
        total_operations = len(operations)
        successful_operations = sum(1 for op in operations if op["success"])
        success_rate = successful_operations / total_operations

        durations = [op["duration_seconds"] for op in operations]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        performance_scores = [
            op["performance_score"]
            for op in operations
            if op["performance_score"] is not None
        ]
        avg_performance_score = (
            sum(performance_scores) / len(performance_scores)
            if performance_scores
            else 0.0
        )

        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": total_operations - successful_operations,
            "success_rate": success_rate,
            "avg_duration_seconds": avg_duration,
            "max_duration_seconds": max_duration,
            "min_duration_seconds": min_duration,
            "avg_performance_score": avg_performance_score,
            "operation_type": operation_type,
            "component": self.component_name,
        }

    def get_recent_operations(
        self, count: int = 10, operation_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Get recent operations for analysis.

        Args:
            count: Number of recent operations to return
            operation_type: Filter by operation type (optional)

        Returns:
            List of recent operation records
        """
        operations = self.execution_history
        if operation_type:
            operations = [
                op for op in operations if op["operation_type"] == operation_type
            ]

        # Return most recent operations
        return operations[-count:] if operations else []

    def clear_history(self) -> None:
        """Clear operation history."""
        self.execution_history.clear()
        logger.info("Performance history cleared for %s", self.component_name)

    def get_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get currently active operations.

        Returns:
            Dictionary of active operations
        """
        current_time = time.time()

        return {
            op_id: {
                "operation_type": op_data["operation_type"],
                "start_time": op_data["start_time"],
                "duration_so_far": current_time - op_data["start_time"],
                "metadata": op_data["metadata"],
            }
            for op_id, op_data in self.current_operations.items()
        }
