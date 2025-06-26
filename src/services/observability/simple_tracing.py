"""Simplified OpenTelemetry tracing for AI documentation scraper.

This module provides minimal but effective observability focused on AI/ML operations,
keeping only the most valuable tracking for a portfolio project.
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


logger = logging.getLogger(__name__)

# Cost estimates for AI operations (in USD)
AI_COSTS = {
    "openai": {
        "text-embedding-3-small": {"per_1k_tokens": 0.00002},
        "text-embedding-3-large": {"per_1k_tokens": 0.00013},
        "gpt-3.5-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
        "gpt-4": {"input_per_1k": 0.03, "output_per_1k": 0.06},
    },
    "cohere": {
        "embed-english-v3.0": {"per_1k_tokens": 0.0001},
        "command": {"input_per_1k": 0.0015, "output_per_1k": 0.002},
    },
}


def setup_tracing(
    service_name: str = "ai-doc-scraper", otlp_endpoint: str | None = None
) -> None:
    """Initialize OpenTelemetry with OTLP exporter.

    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP endpoint (defaults to localhost:4317)
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "development",
        }
    )

    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    return trace.get_tracer("ai_doc_scraper")


def calculate_token_cost(
    provider: str, model: str, tokens: int, operation_type: str = "embedding"
) -> float:
    """Calculate estimated cost for AI operations.

    Args:
        provider: AI provider (openai, cohere, etc.)
        model: Model name
        tokens: Number of tokens
        operation_type: Type of operation (embedding, completion)

    Returns:
        Estimated cost in USD
    """
    if provider not in AI_COSTS or model not in AI_COSTS[provider]:
        return 0.0

    model_costs = AI_COSTS[provider][model]

    if operation_type == "embedding":
        cost_per_1k = model_costs.get("per_1k_tokens", 0)
        return (tokens / 1000) * cost_per_1k
    else:
        # For completions, assume 50/50 input/output split
        input_cost = model_costs.get("input_per_1k", 0)
        output_cost = model_costs.get("output_per_1k", 0)
        return (tokens / 2000) * (input_cost + output_cost)


@contextmanager
def trace_operation(
    operation_name: str, operation_type: str = "general", **attributes: Any
):
    """Simple context manager for tracing operations.

    Args:
        operation_name: Name of the operation
        operation_type: Type of operation (embedding, search, etc.)
        **attributes: Additional span attributes

    Yields:
        OpenTelemetry span
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(operation_name) as span:
        span.set_attribute("operation.type", operation_type)

        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value))

        start_time = time.time()

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("duration_ms", duration_ms)


def track_ai_cost(
    provider: str,
    model: str,
    operation: str = "embedding",
) -> Callable:
    """Decorator to track AI operation costs.

    Args:
        provider: AI provider (openai, cohere, etc.)
        model: Model name
        operation: Operation type (embedding, completion)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_operation(
                f"ai.{operation}",
                operation_type="ai",
                provider=provider,
                model=model,
            ) as span:
                start_time = time.time()

                # Store operation context for cost calculation
                operation_context = {
                    "tokens": 0,
                    "embeddings": 0,
                    "cost": 0.0,
                }

                # Execute the function
                result = await func(*args, **kwargs)

                # Extract metrics from result
                if operation == "embedding" and result:
                    if isinstance(result, list):
                        operation_context["embeddings"] = len(result)
                        # Estimate tokens from input
                        if args:
                            input_text = args[0]
                            if isinstance(input_text, str):
                                operation_context["tokens"] = len(input_text) // 4
                            elif isinstance(input_text, list):
                                operation_context["tokens"] = sum(
                                    len(t) // 4 for t in input_text
                                )

                    # Calculate cost
                    operation_context["cost"] = calculate_token_cost(
                        provider, model, operation_context["tokens"], operation
                    )

                # Add metrics to span
                span.set_attribute("ai.tokens", operation_context["tokens"])
                span.set_attribute(
                    "ai.embeddings_generated", operation_context["embeddings"]
                )
                span.set_attribute("ai.cost_usd", operation_context["cost"])
                span.set_attribute("ai.latency_ms", (time.time() - start_time) * 1000)

                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_operation(
                f"ai.{operation}",
                operation_type="ai",
                provider=provider,
                model=model,
            ) as span:
                start_time = time.time()
                result = func(*args, **kwargs)

                # Simple cost tracking for sync operations
                span.set_attribute("ai.latency_ms", (time.time() - start_time) * 1000)

                return result

        return async_wrapper if hasattr(func, "__apatches__") else sync_wrapper

    return decorator


def track_vector_search(collection: str, top_k: int = 10) -> Callable:
    """Decorator to track vector search operations.

    Args:
        collection: Vector collection name
        top_k: Number of results to retrieve

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_operation(
                "vector.search",
                operation_type="search",
                collection=collection,
                top_k=top_k,
            ) as span:
                start_time = time.time()

                # Execute search
                result = await func(*args, **kwargs)

                # Extract search metrics
                if result and hasattr(result, "points") and result.points:
                    span.set_attribute("search.results_count", len(result.points))
                    if result.points:
                        scores = [p.score for p in result.points if p.score is not None]
                        if scores:
                            span.set_attribute("search.top_score", max(scores))
                            span.set_attribute(
                                "search.avg_score", sum(scores) / len(scores)
                            )

                span.set_attribute(
                    "search.latency_ms", (time.time() - start_time) * 1000
                )

                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_operation(
                "vector.search",
                operation_type="search",
                collection=collection,
                top_k=top_k,
            ) as span:
                start_time = time.time()
                result = func(*args, **kwargs)
                span.set_attribute(
                    "search.latency_ms", (time.time() - start_time) * 1000
                )
                return result

        return async_wrapper if hasattr(func, "__apatches__") else sync_wrapper

    return decorator


# AI Cost Summary Tracker
class AICostTracker:
    """Simple in-memory tracker for AI operation costs."""

    def __init__(self):
        self.operations: dict[str, dict[str, Any]] = {}
        self.total_cost = 0.0
        self.operation_count = 0

    def record_operation(
        self,
        provider: str,
        model: str,
        operation_type: str,
        tokens: int,
        cost: float,
        latency_ms: float,
    ):
        """Record an AI operation."""
        key = f"{provider}:{model}:{operation_type}"

        if key not in self.operations:
            self.operations[key] = {
                "count": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
            }

        op = self.operations[key]
        op["count"] += 1
        op["total_tokens"] += tokens
        op["total_cost"] += cost
        op["avg_latency_ms"] = (
            op["avg_latency_ms"] * (op["count"] - 1) + latency_ms
        ) / op["count"]

        self.total_cost += cost
        self.operation_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_operations": self.operation_count,
            "operations_by_type": {
                key: {
                    "count": op["count"],
                    "total_tokens": op["total_tokens"],
                    "total_cost_usd": round(op["total_cost"], 4),
                    "avg_latency_ms": round(op["avg_latency_ms"], 2),
                }
                for key, op in self.operations.items()
            },
        }


# Global cost tracker instance
cost_tracker = AICostTracker()
