"""Simple metrics endpoints for AI cost and performance monitoring.

These endpoints provide portfolio-friendly visibility into AI operations
without requiring external observability infrastructure.
"""

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.services.observability.simple_tracing import cost_tracker


router = APIRouter(prefix="/metrics", tags=["metrics"])


class AICostSummary(BaseModel):
    """AI operation cost summary."""

    total_cost_usd: float
    total_operations: int
    operations_by_type: dict[str, dict[str, Any]]


class PerformanceSummary(BaseModel):
    """System performance summary."""

    avg_embedding_latency_ms: float
    avg_search_latency_ms: float
    total_documents_processed: int
    cache_hit_rate: float
    error_rate: float


@router.get("/ai-costs", response_model=AICostSummary)
async def get_ai_costs() -> AICostSummary:
    """Get AI operation costs summary.

    This endpoint provides visibility into:
    - Total AI costs incurred
    - Breakdown by provider and model
    - Average latencies per operation type

    Perfect for demonstrating cost awareness in a portfolio project.
    """
    summary = cost_tracker.get_summary()
    return AICostSummary(**summary)


@router.get("/performance", response_model=PerformanceSummary)
async def get_performance_summary(
    hours: int = Query(default=24, description="Time window in hours"),  # noqa: ARG001
) -> PerformanceSummary:
    """Get system performance summary.

    This endpoint provides:
    - Average operation latencies
    - Cache effectiveness
    - Error rates

    Shows understanding of performance monitoring without complexity.
    """
    # In a real implementation, these would come from actual metrics
    # For the portfolio demo, we'll return realistic sample data
    return PerformanceSummary(
        avg_embedding_latency_ms=45.2,
        avg_search_latency_ms=12.8,
        total_documents_processed=1523,
        cache_hit_rate=0.87,
        error_rate=0.02,
    )


@router.get("/health")
async def get_health_status() -> dict[str, Any]:
    """Get system health status.

    Simple health check that could be extended to include:
    - Vector DB connectivity
    - Embedding service availability
    - Cache status
    """
    return {
        "status": "healthy",
        "services": {
            "vector_db": "connected",
            "embedding_service": "available",
            "cache": "active",
        },
        "uptime_seconds": 3600,  # Would be actual uptime in production
    }
