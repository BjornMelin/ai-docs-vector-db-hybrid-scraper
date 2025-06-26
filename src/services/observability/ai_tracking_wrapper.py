"""Wrapper to integrate existing AI tracking with simplified tracing.

This module bridges the sophisticated AI tracking with our simplified
observability approach, preserving the impressive features while reducing complexity.
"""

import functools
from contextlib import contextmanager

from .ai_tracking import get_ai_tracker
from .simple_tracing import cost_tracker, trace_operation


@contextmanager
def track_ai_operation(
    operation_type: str,
    provider: str,
    model: str,
    **kwargs,
):
    """Unified AI operation tracking that combines both systems.

    Args:
        operation_type: Type of AI operation (embedding, llm_call, vector_search)
        provider: AI provider (openai, cohere, etc.)
        model: Model name
        **kwargs: Additional parameters

    Yields:
        Operation context dict
    """
    # Use the sophisticated AI tracker
    ai_tracker = get_ai_tracker()

    # Create operation result dict
    operation_result = {
        "embeddings": None,
        "response": None,
        "usage": None,
        "cost": None,
        "cache_hit": False,
    }

    # Use appropriate tracker based on operation type
    if operation_type == "embedding_generation":
        tracker_cm = ai_tracker.track_embedding_generation(
            provider=provider,
            model=model,
            input_texts=kwargs.get("input_texts", ""),
            expected_dimensions=kwargs.get("expected_dimensions"),
        )
    elif operation_type == "llm_call":
        tracker_cm = ai_tracker.track_llm_call(
            provider=provider,
            model=model,
            operation=kwargs.get("operation", "completion"),
            expected_max_tokens=kwargs.get("max_tokens"),
        )
    elif operation_type == "vector_search":
        tracker_cm = ai_tracker.track_vector_search(
            collection_name=kwargs.get("collection", "default"),
            query_type=kwargs.get("query_type", "semantic"),
            top_k=kwargs.get("top_k"),
        )
    else:
        # Fallback to simple tracing
        tracker_cm = trace_operation(
            f"ai.{operation_type}",
            operation_type="ai",
            provider=provider,
            model=model,
            **kwargs,
        )

    with tracker_cm as context:
        # If using AI tracker, we get a dict we can populate
        if isinstance(context, dict):
            yield context
        else:
            # Otherwise, yield our operation result
            yield operation_result

        # Record in our simple cost tracker
        if operation_result.get("cost"):
            tokens = operation_result.get("tokens", 0)
            if operation_type == "llm_call" and operation_result.get("usage"):
                usage = operation_result["usage"]
                if hasattr(usage, "total_tokens"):
                    tokens = usage.total_tokens

            cost_tracker.record_operation(
                provider=provider,
                model=model,
                operation_type=operation_type,
                tokens=tokens,
                cost=operation_result["cost"],
                latency_ms=kwargs.get("duration_ms", 0),
            )


def simple_ai_decorator(
    operation_type: str,
    provider: str = "unknown",
    model: str = "unknown",
):
    """Simplified decorator for AI operations.

    Args:
        operation_type: Type of operation
        provider: AI provider
        model: Model name

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with track_ai_operation(
                operation_type=operation_type,
                provider=provider,
                model=model,
                input_args=args,
            ) as context:
                result = await func(*args, **kwargs)

                # Update context based on result
                if operation_type == "embedding_generation" and result:
                    context["embeddings"] = result
                    context["cost"] = 0.02 * (
                        len(result) if isinstance(result, list) else 1
                    )
                elif operation_type == "llm_call" and result:
                    context["response"] = result
                    if hasattr(result, "usage"):
                        context["usage"] = result.usage
                        context["cost"] = 0.001 * getattr(
                            result.usage, "total_tokens", 0
                        )

                return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with track_ai_operation(
                operation_type=operation_type,
                provider=provider,
                model=model,
            ):
                return func(*args, **kwargs)

        return async_wrapper if hasattr(func, "__await__") else sync_wrapper

    return decorator
