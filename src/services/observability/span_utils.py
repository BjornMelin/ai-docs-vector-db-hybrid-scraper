"""Shared helpers for span management and telemetry metadata."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from opentelemetry import baggage
from opentelemetry.trace import Span, Status, StatusCode, Tracer


def _apply_span_attributes(span: Span, attributes: Mapping[str, Any] | None) -> None:
    """Apply attributes to a span if it is recording."""
    if not attributes or not span.is_recording():
        return

    for key, value in attributes.items():
        span.set_attribute(key, value)


@contextmanager
def span_context(
    tracer: Tracer,
    span_name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    baggage_entries: Mapping[str, str] | None = None,
) -> Iterator[Span]:
    """Context manager that standardizes span attribute/status handling."""

    with tracer.start_as_current_span(span_name) as span:
        _apply_span_attributes(span, attributes)

        if baggage_entries:
            for key, value in baggage_entries.items():
                baggage.set_baggage(key, value)

        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - propagate upstream after recording
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        span.set_status(Status(StatusCode.OK))


@asynccontextmanager
async def span_context_async(
    tracer: Tracer,
    span_name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    baggage_entries: Mapping[str, str] | None = None,
) -> AsyncIterator[Span]:
    """Async variant of ``span_context`` to share the same semantics."""

    with span_context(
        tracer,
        span_name,
        attributes=attributes,
        baggage_entries=baggage_entries,
    ) as span:
        yield span


def record_llm_usage(span: Span, usage: Any) -> None:
    """Record token usage attributes on a span when present."""

    if not span.is_recording() or usage is None:
        return

    if hasattr(usage, "prompt_tokens"):
        span.set_attribute("llm.usage.prompt_tokens", usage.prompt_tokens)
    if hasattr(usage, "completion_tokens"):
        span.set_attribute("llm.usage.completion_tokens", usage.completion_tokens)
    if hasattr(usage, "total_tokens"):
        span.set_attribute("llm.usage.total_tokens", usage.total_tokens)


__all__ = [
    "span_context",
    "span_context_async",
    "record_llm_usage",
    "instrumented_span",
    "instrumented_span_async",
]


@contextmanager
def instrumented_span(
    tracer: Tracer,
    span_name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    baggage_entries: Mapping[str, str] | None = None,
    duration_attribute: str | None = None,
) -> Iterator[Span]:
    """Wrapper around :func:`span_context` that records duration automatically."""

    with span_context(
        tracer,
        span_name,
        attributes=attributes,
        baggage_entries=baggage_entries,
    ) as span:
        start_time = time.time()
        try:
            yield span
        finally:
            if duration_attribute is not None:
                span.set_attribute(
                    duration_attribute, (time.time() - start_time) * 1000
                )


@asynccontextmanager
async def instrumented_span_async(
    tracer: Tracer,
    span_name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    baggage_entries: Mapping[str, str] | None = None,
    duration_attribute: str | None = None,
) -> AsyncIterator[Span]:
    """Async variant of :func:`instrumented_span`."""

    async with span_context_async(
        tracer,
        span_name,
        attributes=attributes,
        baggage_entries=baggage_entries,
    ) as span:
        start_time = time.time()
        try:
            yield span
        finally:
            if duration_attribute is not None:
                span.set_attribute(
                    duration_attribute, (time.time() - start_time) * 1000
                )
