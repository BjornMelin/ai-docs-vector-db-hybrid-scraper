"""Monitoring utilities for database query and connection activity."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryMetrics:
    """Structured query performance metrics."""

    query_id: str
    start_time: float
    duration_ms: float | None = None
    success: bool = True
    error_message: str | None = None


class QueryMonitor:
    """Track database query durations and outcomes."""

    def __init__(self) -> None:
        """Initialise the query monitor."""

        self._active_queries: dict[str, QueryMetrics] = {}
        self._completed_queries: list[QueryMetrics] = []

    async def initialize(self) -> None:
        """Initialise monitoring resources."""

        logger.info("Query monitor initialised")

    async def cleanup(self) -> None:
        """Clean up query monitoring resources."""

        self._active_queries.clear()
        logger.info("Query monitor cleaned up")

    def start_query(self) -> str:
        """Start tracking a query and return its identifier."""

        query_id = f"query_{time.time_ns()}"
        self._active_queries[query_id] = QueryMetrics(
            query_id=query_id,
            start_time=time.time(),
        )
        return query_id

    def record_success(self, query_id: str) -> None:
        """Record successful query completion."""

        query = self._active_queries.pop(query_id, None)
        if query is None:
            return
        query.duration_ms = (time.time() - query.start_time) * 1000
        query.success = True
        self._completed_queries.append(query)

    def record_failure(self, query_id: str, error: str) -> None:
        """Record failed query execution."""

        query = self._active_queries.pop(query_id, None)
        if query is None:
            return
        query.duration_ms = (time.time() - query.start_time) * 1000
        query.success = False
        query.error_message = error
        self._completed_queries.append(query)

    async def get_performance_summary(self) -> dict[str, Any]:
        """Return aggregated query performance metrics."""

        if not self._completed_queries:
            return {"total_queries": 0, "successful_queries": 0, "avg_duration_ms": 0.0}

        successful_queries = [q for q in self._completed_queries if q.success]
        avg_duration = (
            sum(q.duration_ms or 0.0 for q in successful_queries)
            / len(successful_queries)
            if successful_queries
            else 0.0
        )

        return {
            "total_queries": len(self._completed_queries),
            "successful_queries": len(successful_queries),
            "avg_duration_ms": avg_duration,
        }


class ConnectionMonitor:
    """Capture lightweight connection pool telemetry."""

    def __init__(self) -> None:
        """Initialise connection monitoring state."""

        self._pool_metrics: dict[str, Any] = {}

    def record_connection_event(
        self, event_type: str, metadata: dict[str, Any]
    ) -> None:
        """Record a connection pool event."""

        self._pool_metrics[event_type] = {
            "timestamp": time.time(),
            "metadata": metadata,
        }

    def get_pool_status(self) -> dict[str, Any]:
        """Return a snapshot of connection pool metrics."""

        return {
            "events": self._pool_metrics.copy(),
        }


__all__ = ["ConnectionMonitor", "QueryMetrics", "QueryMonitor"]
