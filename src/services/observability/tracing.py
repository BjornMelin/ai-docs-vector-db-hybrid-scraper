"""Lightweight OpenTelemetry tracing helpers.

This module exposes thin wrappers around OpenTelemetry to keep tracing usage
simple and explicit across the code-base. It intentionally avoids the large
custom decorator stack that previously duplicated SDK behaviour.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from typing import Any, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode


LOGGER = logging.getLogger(__name__)
DEFAULT_TRACER_NAME = "ai-docs.observability"

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def get_tracer(name: str | None = None) -> trace.Tracer:
    """Return a tracer using a consistent default namespace."""

    tracer_name = name or DEFAULT_TRACER_NAME
    return trace.get_tracer(tracer_name)


def _apply_attributes(span: Span, attributes: Mapping[str, Any] | None) -> None:
    """Apply attributes to a span if recording."""

    if not attributes or not span.is_recording():
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)


@contextmanager
def span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    tracer_name: str | None = None,
) -> Iterator[Span]:
    """Create a span context manager with consistent error handling."""

    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(
        name,
        record_exception=True,
        set_status_on_exception=True,
    ) as created_span:
        _apply_attributes(created_span, attributes)
        yield created_span
        if created_span.is_recording():
            created_span.set_status(Status(StatusCode.OK))


def trace_function(
    name: str | None = None,
    *,
    attributes: Mapping[str, Any] | None = None,
    tracer_name: str | None = None,
) -> Callable[[FuncT], FuncT]:
    """Decorate sync or async callables with a tracing span."""

    def decorator(func: FuncT) -> FuncT:
        span_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(
                    span_name, attributes=attributes, tracer_name=tracer_name
                ) as current_span:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001 - re-raised after recording
                        if current_span.is_recording():
                            current_span.record_exception(exc)
                            current_span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            return cast(FuncT, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(
                span_name, attributes=attributes, tracer_name=tracer_name
            ) as current_span:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised after recording
                    if current_span.is_recording():
                        current_span.record_exception(exc)
                        current_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return cast(FuncT, sync_wrapper)

    return decorator


class ConfigOperationType(str, Enum):
    """Supported configuration operation types."""

    LOAD = "load"
    UPDATE = "update"
    VALIDATE = "validate"
    ROLLBACK = "rollback"
    AUTO_DETECT = "auto_detect"


def instrument_config_operation(
    *,
    operation_type: ConfigOperationType,
    operation_name: str | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> Callable[[FuncT], FuncT]:
    """Trace configuration operations with standard attributes."""

    attributes = {"config.operation_type": operation_type.value}
    if extra_attributes:
        attributes.update(extra_attributes)

    return trace_function(operation_name, attributes=attributes)


def set_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Attach attributes to the active span when recording."""

    current_span = trace.get_current_span()
    if not current_span or not current_span.is_recording():
        LOGGER.debug("No active span when attempting to set attributes")
        return
    _apply_attributes(current_span, attributes)
