"""Lightweight tracking primitives used across the project."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import Meter

from .tracing import span


@dataclass
class _OperationStats:
    count: int = 0
    success_count: int = 0
    total_duration_s: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def record(
        self,
        *,
        duration_s: float,
        tokens: int | None,
        cost_usd: float | None,
        success: bool,
    ) -> None:
        self.count += 1
        if success:
            self.success_count += 1
        self.total_duration_s += duration_s
        if tokens is not None:
            self.total_tokens += tokens
        if cost_usd is not None:
            self.total_cost_usd += cost_usd


def _operation_payload(  # pylint: disable=too-many-arguments
    *,
    operation_type: str,
    provider: str,
    model: str,
    duration_s: float,
    tokens: int | None,
    cost_usd: float | None,
    success: bool,
) -> dict[str, Any]:
    """Assemble a standard payload describing an AI operation.

    Args:
        operation_type: Logical operation identifier such as ``llm_call``.
        provider: Name of the provider handling the request.
        model: Model identifier reported by the provider.
        duration_s: Wall-clock duration for the operation.
        tokens: Optional number of tokens processed.
        cost_usd: Optional cost incurred in USD.
        success: Whether the operation completed successfully.

    Returns:
        dict[str, Any]: Structured payload used for in-memory aggregation.
    """
    return {
        "operation": operation_type,
        "provider": provider,
        "model": model,
        "duration_s": duration_s,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "success": success,
    }


class AIOperationTracker:
    """Track AI activity in-memory while emitting optional metrics."""

    def __init__(self, meter: Meter | None = None) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, _OperationStats] = defaultdict(_OperationStats)
        self._meter = meter or metrics.get_meter(__name__)
        self._duration_histogram = self._meter.create_histogram(
            "ai.operation.duration",
            description="Wall clock duration of AI operations in seconds",
            unit="s",
        )
        self._token_counter = self._meter.create_counter(
            "ai.operation.tokens",
            description="Tokens processed by AI operations",
        )
        self._cost_counter = self._meter.create_counter(
            "ai.operation.cost",
            description="Cost of AI operations in USD",
            unit="USD",
        )

    def record_operation(  # pylint: disable=too-many-arguments
        self,
        *,
        operation: str,
        provider: str,
        model: str,
        duration_s: float,
        tokens: int | None = None,
        cost_usd: float | None = None,
        success: bool = True,
    ) -> None:
        """Record metrics for a traced AI operation.

        Args:
            operation: Logical operation identifier to aggregate under.
            provider: Provider responsible for executing the request.
            model: Model identifier reported by the provider.
            duration_s: Duration in seconds.
            tokens: Optional count of tokens consumed.
            cost_usd: Optional monetary cost of the operation.
            success: Flag indicating whether the call succeeded.
        """
        labels = {
            "operation": operation,
            "provider": provider,
            "model": model,
            "success": str(success),
        }

        with self._lock:
            self._stats[operation].record(
                duration_s=duration_s,
                tokens=tokens,
                cost_usd=cost_usd,
                success=success,
            )

        self._duration_histogram.record(duration_s, labels)
        if tokens is not None:
            self._token_counter.add(tokens, labels)
        if cost_usd is not None:
            self._cost_counter.add(cost_usd, labels)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return an immutable snapshot of aggregated operation statistics."""
        with self._lock:
            return {
                name: {
                    "count": stat.count,
                    "success_count": stat.success_count,
                    "total_duration_s": stat.total_duration_s,
                    "total_tokens": stat.total_tokens,
                    "total_cost_usd": stat.total_cost_usd,
                }
                for name, stat in self._stats.items()
            }

    def reset(self) -> None:
        """Clear all recorded statistics."""
        with self._lock:
            self._stats.clear()


class PerformanceTracker:
    """Minimal tracker used by agent components to measure durations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def track(self, label: str) -> Any:
        """Context manager capturing execution duration under ``label``.

        Args:
            label: Identifier used to group durations.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self._durations[label].append(duration)

    def summary(self) -> dict[str, float]:
        """Return average duration per tracked label."""
        with self._lock:
            return {
                label: (sum(values) / len(values)) if values else 0.0
                for label, values in self._durations.items()
            }


class TraceCorrelationManager:
    """Tiny helper for carrying request correlation metadata."""

    def __init__(self) -> None:
        self._local = threading.local()

    def set_context(self, **context: Any) -> None:
        """Update correlation metadata for the current execution context.

        Args:
            **context: Key-value pairs to append to the correlation state.
        """
        existing = getattr(self._local, "context", {})
        merged = {**existing, **context}
        self._local.context = merged
        active_span = trace.get_current_span()
        if active_span and active_span.is_recording():
            for key, value in context.items():
                active_span.set_attribute(f"correlation.{key}", value)

    def get_context(self) -> dict[str, Any]:
        """Return a shallow copy of the stored correlation context."""
        return getattr(self._local, "context", {}).copy()

    @contextmanager
    def correlated_operation(self, **context: Any) -> Any:
        """Temporarily extend the correlation context within a block."""
        previous = self.get_context()
        self.set_context(**context)
        try:
            yield self.get_context()
        finally:
            self._local.context = previous

    def clear(self) -> None:
        """Remove any stored correlation metadata."""
        self._local.context = {}


@lru_cache(maxsize=1)
def get_ai_tracker() -> AIOperationTracker:
    return AIOperationTracker()


@lru_cache(maxsize=1)
def get_correlation_manager() -> TraceCorrelationManager:
    return TraceCorrelationManager()


def record_ai_operation(  # pylint: disable=too-many-arguments
    *,
    operation_type: str,
    provider: str,
    model: str,
    duration_s: float,
    tokens: int | None = None,
    cost_usd: float | None = None,
    success: bool = True,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a completed AI operation using the global tracker.

    Args:
        operation_type: Logical operation identifier to aggregate under.
        provider: Provider executing the request.
        model: Provider model identifier.
        duration_s: Wall-clock duration in seconds.
        tokens: Optional token count for the request.
        cost_usd: Optional USD cost.
        success: Whether the operation succeeded.
        prompt_tokens: Optional prompt token count.
        completion_tokens: Optional completion token count.
        attributes: Optional additional span attributes to set.
    """
    current_span = trace.get_current_span()
    if current_span and current_span.get_span_context().is_valid:
        span_attrs: dict[str, Any] = {
            "gen_ai.operation.name": operation_type,
            "gen_ai.provider": provider,
            "gen_ai.request.model": model,
        }
        if tokens is not None:
            span_attrs["gen_ai.usage.total_tokens"] = tokens
        if prompt_tokens is not None:
            span_attrs["gen_ai.usage.prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            span_attrs["gen_ai.usage.completion_tokens"] = completion_tokens
        if cost_usd is not None:
            span_attrs["gen_ai.cost.usd"] = cost_usd
        if attributes:
            span_attrs.update(attributes)
        for key, value in span_attrs.items():
            if value is not None:
                current_span.set_attribute(key, value)

    payload = _operation_payload(
        operation_type=operation_type,
        provider=provider,
        model=model,
        duration_s=duration_s,
        tokens=tokens,
        cost_usd=cost_usd,
        success=success,
    )
    get_ai_tracker().record_operation(**payload)


def track_cost(
    *,
    operation_type: str,
    provider: str,
    model: str,
    cost_usd: float,
) -> None:
    """Record a cost-only event in the global tracker."""
    record_ai_operation(
        operation_type=operation_type,
        provider=provider,
        model=model,
        duration_s=0.0,
        cost_usd=cost_usd,
    )


def track_llm_call(
    *,
    provider: str,
    model: str,
    tokens: int | None = None,
    cost_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Trace and record a large language model call.

    Args:
        provider: Provider executing the call.
        model: Model identifier for logging and metrics.
        tokens: Optional number of tokens processed.
        cost_usd: Optional cost for the call.
        metadata: Optional tracing metadata to append to the span.
    """
    start = time.perf_counter()
    with span(
        "ai.llm.call",
        attributes={"ai.provider": provider, "ai.model": model, **(metadata or {})},
    ):
        duration = time.perf_counter() - start
    record_ai_operation(
        operation_type="llm_call",
        provider=provider,
        model=model,
        duration_s=duration,
        tokens=tokens,
        cost_usd=cost_usd,
    )


def track_embedding_generation(
    *,
    provider: str,
    model: str,
    batch_size: int | None = None,
    cost_usd: float | None = None,
) -> None:
    """Trace and record an embedding generation pass."""
    start = time.perf_counter()
    attributes: dict[str, Any] = {"ai.provider": provider, "ai.model": model}
    if batch_size is not None:
        attributes["ai.batch_size"] = batch_size
    with span("ai.embedding", attributes=attributes):
        duration = time.perf_counter() - start
    record_ai_operation(
        operation_type="embedding",
        provider=provider,
        model=model,
        duration_s=duration,
        cost_usd=cost_usd,
    )


def track_vector_search(
    *,
    collection: str,
    provider: str,
    model: str,
    result_count: int | None = None,
) -> None:
    """Trace and record vector search latency."""
    start = time.perf_counter()
    attributes: dict[str, Any] = {
        "vector.collection": collection,
        "ai.provider": provider,
        "ai.model": model,
    }
    if result_count is not None:
        attributes["vector.result_count"] = result_count
    with span("ai.vector.search", attributes=attributes):
        duration = time.perf_counter() - start
    record_ai_operation(
        operation_type="vector_search",
        provider=provider,
        model=model,
        duration_s=duration,
    )


def track_rag_pipeline(
    *,
    provider: str,
    model: str,
    stages: dict[str, float] | None = None,
) -> None:
    """Trace and record an entire RAG pipeline execution."""
    attributes: dict[str, Any] = {
        "ai.provider": provider,
        "ai.model": model,
        "rag.stage_count": len(stages or {}),
    }
    start = time.perf_counter()
    with span("ai.rag.pipeline", attributes=attributes):
        duration = time.perf_counter() - start
    record_ai_operation(
        operation_type="rag_pipeline",
        provider=provider,
        model=model,
        duration_s=duration,
    )
