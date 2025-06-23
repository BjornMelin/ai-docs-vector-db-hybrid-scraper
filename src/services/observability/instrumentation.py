"""Advanced OpenTelemetry instrumentation for AI/ML operations and custom business logic.

This module provides comprehensive auto-instrumentation that builds upon OpenTelemetry's
standard instrumentors with AI/ML specific monitoring for vector operations, embeddings,
LLM calls, and complex search pipelines.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from opentelemetry import baggage, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def get_tracer() -> trace.Tracer:
    """Get OpenTelemetry tracer for this module.

    Returns:
        OpenTelemetry tracer instance
    """
    return trace.get_tracer(__name__)


def instrument_function(
    span_name: str | None = None,
    operation_type: str = "operation",
    include_args: bool = False,
    include_result: bool = False,
    baggage_context: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """Decorator to instrument functions with OpenTelemetry tracing.

    Args:
        span_name: Custom span name (defaults to function name)
        operation_type: Type of operation for categorization
        include_args: Whether to include function arguments as attributes
        include_result: Whether to include result information as attributes
        baggage_context: Additional baggage context to propagate

    Returns:
        Decorated function with OpenTelemetry instrumentation
    """

    def decorator(func: F) -> F:
        effective_span_name = span_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span(effective_span_name) as span:
                # Set span attributes
                span.set_attribute("operation.type", operation_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Add baggage context
                if baggage_context:
                    for key, value in baggage_context.items():
                        baggage.set_baggage(key, value)

                # Include arguments if requested
                if include_args and args:
                    span.set_attribute("function.args.count", len(args))
                if include_args and kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, str | int | float | bool):
                            span.set_attribute(f"function.args.{key}", str(value))

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)

                    # Set success status
                    span.set_status(Status(StatusCode.OK))

                    # Include result information if requested
                    if include_result and result is not None:
                        if hasattr(result, "__len__"):
                            span.set_attribute("result.count", len(result))
                        if isinstance(result, dict) and "status" in result:
                            span.set_attribute("result.status", result["status"])

                    return result

                except Exception as e:
                    # Record exception in span
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    # Add performance metrics
                    duration = time.time() - start_time
                    span.set_attribute("operation.duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span(effective_span_name) as span:
                # Set span attributes
                span.set_attribute("operation.type", operation_type)
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Add baggage context
                if baggage_context:
                    for key, value in baggage_context.items():
                        baggage.set_baggage(key, value)

                # Include arguments if requested
                if include_args and args:
                    span.set_attribute("function.args.count", len(args))
                if include_args and kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, str | int | float | bool):
                            span.set_attribute(f"function.args.{key}", str(value))

                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)

                    # Set success status
                    span.set_status(Status(StatusCode.OK))

                    # Include result information if requested
                    if include_result and result is not None:
                        if hasattr(result, "__len__"):
                            span.set_attribute("result.count", len(result))
                        if isinstance(result, dict) and "status" in result:
                            span.set_attribute("result.status", result["status"])

                    return result

                except Exception as e:
                    # Record exception in span
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    # Add performance metrics
                    duration = time.time() - start_time
                    span.set_attribute("operation.duration_ms", duration * 1000)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def instrument_vector_search(
    collection_name: str = "default",
    query_type: str = "semantic",
) -> Callable[[F], F]:
    """Decorator to instrument vector search operations.

    Args:
        collection_name: Name of the vector collection
        query_type: Type of vector search (semantic, hybrid, keyword)

    Returns:
        Decorated function with vector search instrumentation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
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

                try:
                    start_time = time.time()
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
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("vector.search_duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span("vector_search") as span:
                # Set semantic attributes
                span.set_attribute("vector.operation", "search")
                span.set_attribute("vector.collection", collection_name)
                span.set_attribute("vector.query_type", query_type)

                # Add baggage for vector search context
                baggage.set_baggage("search.collection", collection_name)
                baggage.set_baggage("search.type", query_type)

                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("vector.search_duration_ms", duration * 1000)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def instrument_embedding_generation(
    provider: str = "default",
    model: str = "default",
) -> Callable[[F], F]:
    """Decorator to instrument embedding generation operations.

    Args:
        provider: Embedding provider (openai, fastembed, cohere, etc.)
        model: Model name

    Returns:
        Decorated function with embedding instrumentation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
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

                try:
                    start_time = time.time()
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
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("ai.operation.duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span("embedding_generation") as span:
                # Set semantic attributes
                span.set_attribute("ai.operation", "embedding_generation")
                span.set_attribute("ai.model.provider", provider)
                span.set_attribute("ai.model.name", model)

                # Add baggage for embedding context
                baggage.set_baggage("ai.provider", provider)
                baggage.set_baggage("ai.model", model)

                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("ai.operation.duration_ms", duration * 1000)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def instrument_llm_call(
    provider: str = "default",
    model: str = "default",
    operation: str = "completion",
) -> Callable[[F], F]:
    """Decorator to instrument LLM API calls.

    Args:
        provider: LLM provider (openai, anthropic, cohere, etc.)
        model: Model name
        operation: Type of LLM operation (completion, chat, embedding, etc.)

    Returns:
        Decorated function with LLM instrumentation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span("llm_call") as span:
                # Set semantic attributes for LLM operations
                span.set_attribute("llm.operation", operation)
                span.set_attribute("llm.provider", provider)
                span.set_attribute("llm.model", model)

                # Extract request parameters if available
                if kwargs:
                    if "temperature" in kwargs:
                        span.set_attribute("llm.temperature", kwargs["temperature"])
                    if "max_tokens" in kwargs:
                        span.set_attribute("llm.max_tokens", kwargs["max_tokens"])
                    if "messages" in kwargs and isinstance(kwargs["messages"], list):
                        span.set_attribute("llm.message_count", len(kwargs["messages"]))

                # Add baggage for LLM context
                baggage.set_baggage("llm.provider", provider)
                baggage.set_baggage("llm.model", model)

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)

                    # Extract usage and cost information
                    if result and hasattr(result, "usage"):
                        usage = result.usage
                        if hasattr(usage, "prompt_tokens"):
                            span.set_attribute(
                                "llm.usage.prompt_tokens", usage.prompt_tokens
                            )
                        if hasattr(usage, "completion_tokens"):
                            span.set_attribute(
                                "llm.usage.completion_tokens", usage.completion_tokens
                            )
                        if hasattr(usage, "total_tokens"):
                            span.set_attribute(
                                "llm.usage.total_tokens", usage.total_tokens
                            )

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("llm.call.duration_ms", duration * 1000)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span("llm_call") as span:
                # Set semantic attributes
                span.set_attribute("llm.operation", operation)
                span.set_attribute("llm.provider", provider)
                span.set_attribute("llm.model", model)

                # Add baggage for LLM context
                baggage.set_baggage("llm.provider", provider)
                baggage.set_baggage("llm.model", model)

                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                finally:
                    duration = time.time() - start_time
                    span.set_attribute("llm.call.duration_ms", duration * 1000)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextmanager
def trace_operation(operation_name: str, operation_type: str = "custom", **attributes):
    """Context manager for manual span creation with automatic error handling.

    Args:
        operation_name: Name of the operation
        operation_type: Type of operation
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span instance
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(operation_name) as span:
        # Set basic attributes
        span.set_attribute("operation.type", operation_type)

        # Set additional attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@asynccontextmanager
async def trace_async_operation(
    operation_name: str, operation_type: str = "custom", **attributes
):
    """Async context manager for manual span creation with automatic error handling.

    Args:
        operation_name: Name of the operation
        operation_type: Type of operation
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span instance
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(operation_name) as span:
        # Set basic attributes
        span.set_attribute("operation.type", operation_type)

        # Set additional attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


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
