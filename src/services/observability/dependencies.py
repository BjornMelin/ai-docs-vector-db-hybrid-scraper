"""FastAPI dependencies for observability."""

from __future__ import annotations

# pylint: disable=duplicate-code
import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends  # type: ignore[import]

from .config import ObservabilityConfig, get_observability_config
from .init import initialize_observability
from .tracing import get_tracer
from .tracking import get_ai_tracker, record_ai_operation, track_cost


logger = logging.getLogger(__name__)

ObservabilityConfigDep = Annotated[
    ObservabilityConfig, Depends(get_observability_config)
]


@lru_cache
def get_observability_service() -> dict[str, Any]:
    """Initialize observability once and expose tracers/meters to routes."""
    config = get_observability_config()
    enabled = initialize_observability(config)

    tracer = get_tracer() if enabled else trace_noop()

    return {
        "config": config,
        "enabled": enabled,
        "tracer": tracer,
        "ai_tracker": get_ai_tracker(),
    }


def trace_noop() -> Any:  # pragma: no cover - trivial fallback
    """Return a no-op tracer for fallback when observability is disabled."""

    class _Tracer:
        def start_as_current_span(self, *_args: Any, **_kwargs: Any):
            """Return null context for no-op tracing."""
            return nullcontext()

    return _Tracer()


ObservabilityServiceDep = Annotated[dict[str, Any], Depends(get_observability_service)]


def ai_tracer_dep(service: ObservabilityServiceDep) -> Any:
    """Extract AI tracer from observability service dependency."""
    return service["tracer"]


AITracerDep = Annotated[Any, Depends(ai_tracer_dep)]


async def record_ai_operation_metrics(  # pylint: disable=too-many-arguments
    operation_type: str,
    provider: str,
    model: str,
    *,
    duration_s: float,
    tokens: int | None = None,
    cost_usd: float | None = None,
    service: ObservabilityServiceDep,
) -> None:
    """Record AI operation metrics if observability is enabled."""
    if not service["enabled"]:
        return
    record_ai_operation(
        operation_type=operation_type,
        provider=provider,
        model=model,
        duration_s=duration_s,
        tokens=tokens,
        cost_usd=cost_usd,
    )


async def track_ai_cost_metrics(  # pylint: disable=too-many-arguments
    operation_type: str,
    provider: str,
    model: str,
    cost_usd: float,
    service: ObservabilityServiceDep,
) -> None:
    """Track AI cost metrics if observability is enabled."""
    if not service["enabled"]:
        return
    track_cost(
        operation_type=operation_type,
        provider=provider,
        model=model,
        cost_usd=cost_usd,
    )


__all__ = [
    "AITracerDep",
    "ObservabilityConfigDep",
    "ObservabilityServiceDep",
    "get_observability_service",
    "record_ai_operation_metrics",
    "track_ai_cost_metrics",
]
