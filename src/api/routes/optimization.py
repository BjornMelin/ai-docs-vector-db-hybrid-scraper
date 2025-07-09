"""API routes for Performance Optimization Agent.

This module provides REST API endpoints for managing and monitoring the
Performance Optimization Agent (POA). It exposes functionality for starting/
stopping the agent, retrieving performance metrics, and manually triggering
optimizations.

The POA is an autonomous agent that continuously monitors system performance
and applies optimizations based on detected patterns and bottlenecks. These
endpoints allow external systems to interact with and control the POA.

Key Endpoints:
    - /status: Get current POA status and optimization history
    - /start: Start the POA monitoring loop
    - /stop: Stop the POA gracefully
    - /metrics/current: Get real-time performance snapshot
    - /metrics/trends: Get performance trends over time
    - /recommendations: Get optimization recommendations
    - /history: View past optimization events
    - /trigger/{type}: Manually trigger specific optimizations
    - /benchmark/run: Run comprehensive performance benchmarks

Example:
    >>> import httpx
    >>> # Start the POA
    >>> response = httpx.post("http://localhost:8000/api/v1/optimization/start")
    >>> # Get current metrics
    >>> metrics = httpx.get(
    ...     "http://localhost:8000/api/v1/optimization/metrics/current"
    ... )
    >>> print(metrics.json())

Note:
    The POA requires appropriate permissions and resources to apply
    optimizations. Ensure proper configuration before starting.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.services.dependencies import get_poa_service
from src.services.monitoring.performance_monitor import PerformanceSnapshot
from src.services.performance.poa_service import (
    OptimizationType,
    PerformanceOptimizationAgent,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.get("/status")
async def get_optimization_status(
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, Any]:
    """Get current POA status and optimization history.

    Returns comprehensive status including:
    - POA running state
    - Active optimizations
    - Historical optimization results
    - Current performance metrics
    - Optimization recommendations
    """
    try:
        return await poa.get_status()
    except Exception as e:
        logger.exception("Failed to get POA status")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/start")
async def start_optimization_agent(
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, str]:
    """Start the Performance Optimization Agent.

    Begins continuous monitoring and optimization control loop.
    """
    try:
        await poa.start()
        return {  # noqa: TRY300
            "status": "started",
            "message": "Performance Optimization Agent started",
        }
    except Exception as e:
        logger.exception("Failed to start POA")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop")
async def stop_optimization_agent(
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, str]:
    """Stop the Performance Optimization Agent.

    Gracefully stops the optimization control loop.
    """
    try:
        await poa.stop()
        return {  # noqa: TRY300
            "status": "stopped",
            "message": "Performance Optimization Agent stopped",
        }
    except Exception as e:
        logger.exception("Failed to stop POA")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics/current")
async def get_current_metrics(
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, Any]:
    """Get current performance metrics snapshot."""
    try:
        return poa.monitor.get_performance_summary()
    except Exception as e:
        logger.exception("Failed to get current metrics")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics/trends")
async def get_performance_trends(
    minutes: int = Query(default=5, ge=1, le=60, description="Time window in minutes"),
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, Any]:
    """Get performance trends over specified time window.

    Args:
        minutes: Number of minutes to analyze (1-60)

    Returns:
        Performance trends including CPU, memory, response time, and throughput
    """
    try:
        return poa.monitor.get_performance_trends(minutes=minutes)
    except Exception as e:
        logger.exception("Failed to get performance trends")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/recommendations")
async def get_optimization_recommendations(
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, Any]:
    """Get current optimization recommendations based on performance data."""
    try:
        recommendations = poa.monitor.get_optimization_recommendations()
        return {"recommendations": recommendations, "count": len(recommendations)}
    except Exception as e:
        logger.exception("Failed to get recommendations")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/history")
async def get_optimization_history(
    limit: int = Query(
        default=50, ge=1, le=200, description="Maximum number of events"
    ),
    status: str = Query(default=None, description="Filter by status"),
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, Any]:
    """Get historical optimization events from the ledger.

    Args:
        limit: Maximum number of events to return
        status: Optional status filter (applied, rolled_back, completed, failed)

    Returns:
        List of optimization events with details
    """
    try:
        events = poa.ledger.events[-limit:]  # Get most recent events

        if status:
            events = [e for e in events if e.status == status]

        return {
            "events": [event.model_dump() for event in events],
            "total": len(poa.ledger.events),
            "returned": len(events),
        }
    except Exception as e:
        logger.exception("Failed to get optimization history")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/trigger/{optimization_type}")
async def trigger_optimization(
    optimization_type: str,
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> dict[str, str]:
    """Manually trigger a specific optimization type.

    Args:
        optimization_type: Type of optimization to trigger
            (database_index, cache_ttl, connection_pool, etc.)

    Note: This is for testing/debugging. Production optimizations
    should be triggered by the automatic control loop.
    """
    try:
        # Validate optimization type
        try:
            opt_type = OptimizationType(optimization_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization type. Valid types: {[e.value for e in OptimizationType]}",
            ) from None

        # Get current metrics as baseline
        snapshot = poa.monitor.get_performance_summary()
        if snapshot.get("status") == "no_data":
            raise HTTPException(  # noqa: TRY301
                status_code=400, detail="No performance data available for baseline"
            )

        metrics = PerformanceSnapshot(**snapshot)

        # Apply optimization
        await poa.apply_optimization(opt_type, metrics)  # Use public method

        return {  # noqa: TRY300
            "status": "triggered",
            "optimization_type": optimization_type,
            "message": f"Optimization {optimization_type} has been triggered",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to trigger optimization: {optimization_type}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/benchmark/run")
async def run_performance_benchmark(
    poa: PerformanceOptimizationAgent = Depends(get_poa_service),  # noqa: B008
) -> StreamingResponse:
    """Run performance benchmark suite and stream results.

    This endpoint runs a comprehensive benchmark suite and streams
    the results as they complete.
    """

    async def benchmark_stream():
        """Stream benchmark results as JSON lines."""
        try:
            # Start with status message
            yield '{"status": "starting", "message": "Beginning performance benchmark suite"}\n'

            # Run individual benchmarks
            benchmark_components = {"settings": poa.settings, "monitor": poa.monitor}

            # Run baseline suite
            suite = await poa.benchmark.run_baseline_suite(benchmark_components)

            # Stream each result
            for result in suite.results:
                yield f'{{"benchmark": "{result.name}", "duration_ms": {result.duration_ms:.2f}, "throughput": {result.throughput:.2f}, "p95_ms": {result.p95_ms:.2f}}}\n'

            # Final summary
            report = suite.to_report()
            yield f'{{"status": "completed", "summary": {report["summary"]}}}\n'

        except Exception as e:
            logger.exception("Benchmark failed")
            yield f'{{"status": "error", "message": "{e!s}"}}\n'

    return StreamingResponse(benchmark_stream(), media_type="application/x-ndjson")
