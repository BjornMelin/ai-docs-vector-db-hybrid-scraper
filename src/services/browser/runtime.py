"""Shared runtime utilities for browser providers.

This module centralizes retries, error taxonomy, and Prometheus metrics to
ensure consistent behaviour across providers while keeping code simple.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import httpx
from prometheus_client import Counter, Histogram
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.services.browser.models import ProviderKind


class ErrorClass(str, Enum):
    """Canonical error taxonomy for provider operations."""

    TRANSIENT_NETWORK = "transient_network"
    RATE_LIMITED = "rate_limited"
    ANTI_BOT_CHALLENGE = "anti_bot_challenge"
    INVALID_INPUT = "invalid_input"
    PROVIDER_BUG = "provider_bug"
    UNEXPECTED = "unexpected"


_OP_COUNTER = Counter(
    "browser_operation_total",
    "Count of provider operations",
    labelnames=("provider", "operation", "outcome", "error_class"),
)

_OP_LATENCY = Histogram(
    "browser_operation_latency_seconds",
    "Latency of provider operations",
    labelnames=("provider", "operation"),
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    ),
)


def classify_exception(exc: BaseException) -> ErrorClass:
    """Map exceptions to a stable error class.

    The taxonomy keeps labels low-cardinality for Prometheus while preserving
    enough signal for routing/triage.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 503):
            return ErrorClass.RATE_LIMITED
        if 400 <= status < 500:
            return ErrorClass.INVALID_INPUT
        return ErrorClass.TRANSIENT_NETWORK
    if isinstance(exc, httpx.RequestError | TimeoutError):
        return ErrorClass.TRANSIENT_NETWORK
    # Generic buckets for provider SDK exceptions. Keep broad and simple.
    name = exc.__class__.__name__.lower()
    if "captcha" in name or "challenge" in name:
        return ErrorClass.ANTI_BOT_CHALLENGE
    return ErrorClass.UNEXPECTED


@asynccontextmanager
async def measure(
    *, provider: ProviderKind, operation: str
) -> AsyncIterator[None]:  # pragma: no cover - thin wrapper
    """Async context manager recording metrics for an operation.

    On success, records latency and increments success counter. On exception,
    increments failure counter with a classified error and re-raises.
    """
    labels = {"provider": provider.value, "operation": operation}
    start = time.perf_counter()
    try:
        yield
        _OP_LATENCY.labels(**labels).observe(time.perf_counter() - start)
        _OP_COUNTER.labels(**labels, outcome="success", error_class="").inc()
    except BaseException as exc:  # pragma: no cover - passthrough
        _OP_LATENCY.labels(**labels).observe(time.perf_counter() - start)
        _OP_COUNTER.labels(
            **labels, outcome="failure", error_class=classify_exception(exc).value
        ).inc()
        raise


async def execute_with_retry(
    *,
    provider: ProviderKind,
    operation: str,
    func: Callable[[], Awaitable[Any]],
    attempts: int = 3,
    min_wait: float = 0.1,
    max_wait: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (
        httpx.RequestError,
        TimeoutError,
    ),
) -> Any:
    """Run ``func`` under metrics + retry policy.

    Args:
        provider: Provider identifier.
        operation: Short operation label (e.g. "scrape", "search").
        func: Awaitable callable to execute.
        attempts: Max attempts including the first.
        min_wait: Minimum backoff seconds.
        max_wait: Maximum backoff seconds.
        retry_on: Exception types considered transient.
    """
    async with measure(provider=provider, operation=operation):
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(attempts),
                wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
                retry=retry_if_exception_type(retry_on),
            ):
                with attempt:
                    return await func()
        except RetryError as retry_exc:  # expose final cause
            last_exception = retry_exc.last_attempt.exception()
            if last_exception is not None:
                raise last_exception from retry_exc
            raise retry_exc from None
