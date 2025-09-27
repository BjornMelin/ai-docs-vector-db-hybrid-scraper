"""OpenTelemetry instrumentation for AI/ML operations and custom business logic.

This module provides auto-instrumentation that builds upon OpenTelemetry's
standard instrumentors with AI/ML specific monitoring for vector operations, embeddings,
LLM calls, and complex search pipelines.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any, TypeVar, cast

from opentelemetry import baggage, trace
from opentelemetry.trace import Status, StatusCode

from .span_utils import record_llm_usage, span_context, span_context_async


logger = logging.getLogger(__name__)

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def get_tracer() -> trace.Tracer:
    """Get OpenTelemetry tracer for this module.

    Returns:
        OpenTelemetry tracer instance

    """
    return trace.get_tracer(__name__)


def _record_arguments(
    span: trace.Span, include_args: bool, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    if not include_args:
        return

    if args:
        span.set_attribute("function.args.count", len(args))
    if kwargs:
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"function.args.{key}", value)
            else:
                span.set_attribute(f"function.args.{key}", str(value))


def _record_result(span: trace.Span, include_result: bool, result: Any) -> None:
    if not include_result or result is None:
        return

    if hasattr(result, "__len__") and not isinstance(result, (str, bytes)):
        with suppress(TypeError):
            span.set_attribute("result.count", len(result))

    if isinstance(result, dict):
        status_value = result.get("status")
        if isinstance(status_value, (str, int, float, bool)):
            span.set_attribute("result.status", status_value)


def instrument_function(
    span_name: str | None = None,
    operation_type: str = "operation",
    include_args: bool = False,
    include_result: bool = False,
    baggage_context: dict[str, str] | None = None,
) -> Callable[[FuncT], FuncT]:
    """Decorator to instrument functions with OpenTelemetry tracing."""

    def decorator(func: FuncT) -> FuncT:
        effective_span_name = span_name or f"{func.__module__}.{func.__name__}"

        def _apply_common_attributes(span: trace.Span) -> None:
            span.set_attribute("operation.type", operation_type)
            span.set_attribute("function.name", func.__name__)
            span.set_attribute("function.module", func.__module__)
            if baggage_context:
                for key, value in baggage_context.items():
                    baggage.set_baggage(key, value)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(effective_span_name) as span:
                    _apply_common_attributes(span)
                    _record_arguments(span, include_args, args, kwargs)
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001 - re-raise after recording
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise
                    else:
                        span.set_status(Status(StatusCode.OK))
                        _record_result(span, include_result, result)
                        return result
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("operation.duration_ms", duration_ms)

            return cast(FuncT, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(effective_span_name) as span:
                _apply_common_attributes(span)
                _record_arguments(span, include_args, args, kwargs)
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raise after recording
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                else:
                    span.set_status(Status(StatusCode.OK))
                    _record_result(span, include_result, result)
                    return result
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("operation.duration_ms", duration_ms)

        return cast(FuncT, sync_wrapper)

    return decorator


def instrument_vector_search(
    collection_name: str = "default",
    query_type: str = "semantic",
) -> Callable[[FuncT], FuncT]:
    """Decorator to instrument vector search operations.

    Args:
        collection_name: Name of the vector collection
        query_type: Type of vector search (semantic, hybrid, keyword)

    Returns:
        Decorated function with vector search instrumentation

    """

    def decorator(func: FuncT) -> FuncT:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span("vector_search") as span:
                # Set semantic attributes following OpenTelemetry conventions
                span.set_attribute("vector.operation", "search")
                span.set_attribute("vector.collection", collection_name)
                span.set_attribute("vector.query_type", query_type)

                # Extract query information if available
                if args and hasattr(args[0], "__len__"):
                    span.set_attribute("vector.query_dimensions", len(args[0]))

                # Add baggage for vector search context
                baggage.set_baggage("search.collection", collection_name)
                baggage.set_baggage("search.type", query_type)

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)

                    # Extract search result metrics
                    if result:
                        if hasattr(result, "points") and result.points:
                            span.set_attribute(
                                "vector.results_count", len(result.points)
                            )
                            if result.points[0].score is not None:
                                span.set_attribute(
                                    "vector.top_score", result.points[0].score
                                )
                                span.set_attribute(
                                    "vector.min_score", result.points[-1].score
                                )

                        if hasattr(result, "scoring_time"):
                            span.set_attribute(
                                "vector.scoring_time_ms", result.scoring_time * 1000
                            )

                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                else:
                    return result

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("vector.search_duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span("vector_search") as span:
                # Set semantic attributes
                span.set_attribute("vector.operation", "search")
                span.set_attribute("vector.collection", collection_name)
                span.set_attribute("vector.query_type", query_type)

                # Add baggage for vector search context
                baggage.set_baggage("search.collection", collection_name)
                baggage.set_baggage("search.type", query_type)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                else:
                    return result

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("vector.search_duration_ms", duration * 1000)

        if asyncio.iscoroutinefunction(func):
            return cast(FuncT, async_wrapper)
        return cast(FuncT, sync_wrapper)

    return decorator


def instrument_embedding_generation(
    provider: str = "default",
    model: str = "default",
) -> Callable[[FuncT], FuncT]:
    """Decorator to instrument embedding generation operations.

    Args:
        provider: Embedding provider (openai, fastembed, cohere, etc.)
        model: Model name

    Returns:
        Decorated function with embedding instrumentation

    """

    def decorator(func: FuncT) -> FuncT:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span("embedding_generation") as span:
                # Set semantic attributes for AI/ML operations
                span.set_attribute("ai.operation", "embedding_generation")
                span.set_attribute("ai.model.provider", provider)
                span.set_attribute("ai.model.name", model)

                # Extract input information
                if args:
                    input_data = args[0]
                    if isinstance(input_data, list):
                        span.set_attribute("ai.input.batch_size", len(input_data))
                        span.set_attribute("ai.input.type", "batch")
                    elif isinstance(input_data, str):
                        span.set_attribute("ai.input.length", len(input_data))
                        span.set_attribute("ai.input.type", "single")

                # Add baggage for embedding context
                baggage.set_baggage("ai.provider", provider)
                baggage.set_baggage("ai.model", model)

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)

                    # Extract embedding result metrics
                    if result:
                        if isinstance(result, list) and result:
                            span.set_attribute("ai.output.embedding_count", len(result))
                            if hasattr(result[0], "__len__"):
                                span.set_attribute(
                                    "ai.output.embedding_dimensions", len(result[0])
                                )
                        elif hasattr(result, "__len__"):
                            span.set_attribute(
                                "ai.output.embedding_dimensions", len(result)
                            )

                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                else:
                    return result

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("ai.operation.duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()

            with tracer.start_as_current_span("embedding_generation") as span:
                # Set semantic attributes
                span.set_attribute("ai.operation", "embedding_generation")
                span.set_attribute("ai.model.provider", provider)
                span.set_attribute("ai.model.name", model)

                # Add baggage for embedding context
                baggage.set_baggage("ai.provider", provider)
                baggage.set_baggage("ai.model", model)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                else:
                    return result

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("ai.operation.duration_ms", duration * 1000)

        if asyncio.iscoroutinefunction(func):
            return cast(FuncT, async_wrapper)
        return cast(FuncT, sync_wrapper)

    return decorator


def instrument_llm_call(
    provider: str = "default",
    model: str = "default",
    operation: str = "completion",
) -> Callable[[FuncT], FuncT]:
    """Decorator to instrument LLM API calls.

    Args:
        provider: LLM provider (openai, anthropic, cohere, etc.)
        model: Model name
        operation: Type of LLM operation (completion, chat, embedding, etc.)

    Returns:
        Decorated function with LLM instrumentation

    """

    def decorator(func: FuncT) -> FuncT:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span_attributes: dict[str, Any] = {
                "llm.operation": operation,
                "llm.provider": provider,
                "llm.model": model,
            }

            if kwargs:
                if "temperature" in kwargs:
                    span_attributes["llm.temperature"] = kwargs["temperature"]
                if "max_tokens" in kwargs:
                    span_attributes["llm.max_tokens"] = kwargs["max_tokens"]
                if "messages" in kwargs and isinstance(kwargs["messages"], list):
                    span_attributes["llm.message_count"] = len(kwargs["messages"])

            baggage_entries = {"llm.provider": provider, "llm.model": model}

            async with span_context_async(
                tracer,
                "llm_call",
                attributes=span_attributes,
                baggage_entries=baggage_entries,
            ) as span:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    if result and hasattr(result, "usage"):
                        record_llm_usage(span, result.usage)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000
                    span.set_attribute("llm.call.duration_ms", duration)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            span_attributes: dict[str, Any] = {
                "llm.operation": operation,
                "llm.provider": provider,
                "llm.model": model,
            }

            if kwargs:
                if "temperature" in kwargs:
                    span_attributes["llm.temperature"] = kwargs["temperature"]
                if "max_tokens" in kwargs:
                    span_attributes["llm.max_tokens"] = kwargs["max_tokens"]
                if "messages" in kwargs and isinstance(kwargs["messages"], list):
                    span_attributes["llm.message_count"] = len(kwargs["messages"])

            baggage_entries = {"llm.provider": provider, "llm.model": model}

            with span_context(
                tracer,
                "llm_call",
                attributes=span_attributes,
                baggage_entries=baggage_entries,
            ) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    if result and hasattr(result, "usage"):
                        record_llm_usage(span, result.usage)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000
                    span.set_attribute("llm.call.duration_ms", duration)

        if asyncio.iscoroutinefunction(func):
            return cast(FuncT, async_wrapper)
        return cast(FuncT, sync_wrapper)

    return decorator


@contextmanager
def trace_operation(operation_name: str, operation_type: str = "custom", **attributes):
    """Context manager for manual span creation with automatic error handling."""

    tracer = get_tracer()
    base_attributes = {"operation.type": operation_type}
    merged_attributes = {**base_attributes, **attributes}

    with span_context(tracer, operation_name, attributes=merged_attributes) as span:
        yield span


@asynccontextmanager
async def trace_async_operation(
    operation_name: str, operation_type: str = "custom", **attributes
):
    """Async context manager mirroring :func:`trace_operation`."""

    tracer = get_tracer()
    base_attributes = {"operation.type": operation_type}
    merged_attributes = {**base_attributes, **attributes}

    async with span_context_async(
        tracer, operation_name, attributes=merged_attributes
    ) as span:
        yield span


def add_span_attribute(key: str, value: Any) -> None:
    """Add attribute to current active span if available.

    Args:
        key: Attribute key
        value: Attribute value

    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add event to current active span if available.

    Args:
        name: Event name
        attributes: Optional event attributes

    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})


def set_user_context(user_id: str, session_id: str | None = None) -> None:
    """Set user context in baggage for request tracing.

    Args:
        user_id: User identifier
        session_id: Optional session identifier

    """
    baggage.set_baggage("user.id", user_id)
    if session_id:
        baggage.set_baggage("session.id", session_id)


def set_business_context(query_type: str, operation_context: str | None = None) -> None:
    """Set business context in baggage for operation tracing.

    Args:
        query_type: Type of query or operation
        operation_context: Additional operation context

    """
    baggage.set_baggage("business.query_type", query_type)
    if operation_context:
        baggage.set_baggage("business.context", operation_context)


def get_current_trace_id() -> str | None:
    """Get current trace ID if available.

    Returns:
        Trace ID as string or None

    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        return format(current_span.get_span_context().trace_id, "032x")
    return None


def get_current_span_id() -> str | None:
    """Get current span ID if available.

    Returns:
        Span ID as string or None

    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        return format(current_span.get_span_context().span_id, "016x")
    return None
