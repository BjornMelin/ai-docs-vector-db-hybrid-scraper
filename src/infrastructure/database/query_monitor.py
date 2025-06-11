"""Query performance monitoring for database operations.

This module provides comprehensive monitoring of database query performance,
including slow query detection, latency histograms, and query pattern analysis.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Statistics for a database query."""

    query_hash: str
    query_pattern: str
    execution_count: int
    total_time_ms: float
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    slow_query_count: int
    last_execution: float


class QueryMonitorConfig(BaseModel):
    """Configuration for query performance monitoring."""

    enabled: bool = Field(default=True, description="Enable query monitoring")
    slow_query_threshold_ms: float = Field(
        default=100.0, gt=0, description="Threshold for slow query detection (ms)"
    )
    max_tracked_queries: int = Field(
        default=1000, gt=0, description="Maximum number of queries to track"
    )
    stats_window_hours: int = Field(
        default=24, gt=0, description="Statistics window in hours"
    )
    histogram_buckets: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        description="Histogram buckets for query latency (seconds)",
    )
    log_slow_queries: bool = Field(default=True, description="Log slow queries")
    track_query_patterns: bool = Field(
        default=True, description="Track normalized query patterns"
    )


class QueryMonitor:
    """Monitors database query performance and provides analytics.

    This class tracks query execution times, identifies slow queries,
    and provides performance metrics for optimization.
    """

    def __init__(self, config: QueryMonitorConfig | None = None):
        """Initialize the query monitor.

        Args:
            config: Query monitoring configuration
        """
        self.config = config or QueryMonitorConfig()
        self._query_stats: dict[str, QueryStats] = {}
        self._latency_histogram: dict[float, int] = defaultdict(int)
        self._active_queries: dict[str, float] = {}  # query_id -> start_time
        self._lock = asyncio.Lock()
        self._total_queries = 0
        self._slow_queries = 0

    async def start_query(self, query: str, query_id: str | None = None) -> str:
        """Start monitoring a query execution.

        Args:
            query: SQL query string
            query_id: Optional custom query ID

        Returns:
            Query ID for tracking
        """
        if not self.config.enabled:
            return query_id or ""

        if query_id is None:
            query_id = f"query_{int(time.time() * 1000000)}"

        start_time = time.time()

        async with self._lock:
            self._active_queries[query_id] = start_time
            self._total_queries += 1

        return query_id

    async def end_query(self, query_id: str, query: str, success: bool = True) -> float:
        """End monitoring a query execution.

        Args:
            query_id: Query ID from start_query
            query: SQL query string
            success: Whether the query executed successfully

        Returns:
            Query execution time in milliseconds
        """
        if not self.config.enabled or query_id not in self._active_queries:
            return 0.0

        end_time = time.time()
        execution_time_ms = (end_time - self._active_queries[query_id]) * 1000

        async with self._lock:
            # Remove from active queries
            del self._active_queries[query_id]

            if success:
                await self._record_query_stats(query, execution_time_ms)

        return execution_time_ms

    async def record_query(self, query: str, execution_time_ms: float) -> None:
        """Record a completed query execution.

        Args:
            query: SQL query string
            execution_time_ms: Execution time in milliseconds
        """
        if not self.config.enabled:
            return

        async with self._lock:
            await self._record_query_stats(query, execution_time_ms)

    async def get_query_stats(self, limit: int = 20) -> list[QueryStats]:
        """Get query statistics sorted by total execution time.

        Args:
            limit: Maximum number of stats to return

        Returns:
            List of query statistics
        """
        async with self._lock:
            stats = list(self._query_stats.values())
            stats.sort(key=lambda x: x.total_time_ms, reverse=True)
            return stats[:limit]

    async def get_slow_queries(self, limit: int = 10) -> list[QueryStats]:
        """Get slowest queries by average execution time.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of slow query statistics
        """
        async with self._lock:
            slow_queries = [
                stats
                for stats in self._query_stats.values()
                if stats.slow_query_count > 0
            ]
            slow_queries.sort(key=lambda x: x.avg_time_ms, reverse=True)
            return slow_queries[:limit]

    async def get_latency_histogram(self) -> dict[str, int]:
        """Get query latency histogram.

        Returns:
            Dictionary mapping latency buckets to counts
        """
        async with self._lock:
            return {
                f"≤{bucket}s": self._latency_histogram[bucket]
                for bucket in self.config.histogram_buckets
            }

    async def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        async with self._lock:
            if not self._query_stats:
                return {
                    "total_queries": 0,
                    "slow_queries": 0,
                    "slow_query_percentage": 0.0,
                    "unique_queries": 0,
                    "avg_execution_time_ms": 0.0,
                    "active_queries": len(self._active_queries),
                }

            all_avg_times = [stats.avg_time_ms for stats in self._query_stats.values()]
            overall_avg = (
                sum(all_avg_times) / len(all_avg_times) if all_avg_times else 0.0
            )

            slow_query_percentage = (
                (self._slow_queries / self._total_queries * 100)
                if self._total_queries > 0
                else 0.0
            )

            # Calculate totals from actual stats
            total_executions = sum(
                stats.execution_count for stats in self._query_stats.values()
            )
            total_slow_queries = sum(
                stats.slow_query_count for stats in self._query_stats.values()
            )

            slow_query_percentage = (
                (total_slow_queries / total_executions * 100)
                if total_executions > 0
                else 0.0
            )

            return {
                "total_queries": total_executions,
                "slow_queries": total_slow_queries,
                "slow_query_percentage": round(slow_query_percentage, 2),
                "unique_queries": len(self._query_stats),
                "avg_execution_time_ms": round(overall_avg, 2),
                "active_queries": len(self._active_queries),
            }

    async def cleanup_old_stats(self) -> int:
        """Clean up old query statistics.

        Returns:
            Number of entries cleaned up
        """
        if not self.config.enabled:
            return 0

        cutoff_time = time.time() - (self.config.stats_window_hours * 3600)

        async with self._lock:
            old_queries = [
                query_hash
                for query_hash, stats in self._query_stats.items()
                if stats.last_execution < cutoff_time
            ]

            for query_hash in old_queries:
                del self._query_stats[query_hash]

            return len(old_queries)

    def _normalize_query(self, query: str) -> tuple[str, str]:
        """Normalize a query to identify patterns.

        Args:
            query: Raw SQL query

        Returns:
            Tuple of (query_hash, normalized_pattern)
        """
        if not self.config.track_query_patterns:
            return query, query

        # Simple normalization - replace literals with placeholders
        import re

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", query.strip())

        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)

        # Replace numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)

        # Replace IN clauses with multiple values
        normalized = re.sub(
            r"IN\s*\([^)]+\)", "IN (?)", normalized, flags=re.IGNORECASE
        )

        return normalized, normalized

    async def _record_query_stats(self, query: str, execution_time_ms: float) -> None:
        """Record statistics for a query execution.

        Args:
            query: SQL query string
            execution_time_ms: Execution time in milliseconds
        """
        query_hash, query_pattern = self._normalize_query(query)

        # Update latency histogram
        execution_time_s = execution_time_ms / 1000.0
        for bucket in self.config.histogram_buckets:
            if execution_time_s <= bucket:
                self._latency_histogram[bucket] += 1
                break
        else:
            # Execution time exceeds all buckets
            if self.config.histogram_buckets:
                self._latency_histogram[self.config.histogram_buckets[-1]] += 1

        # Check if it's a slow query
        is_slow = execution_time_ms >= self.config.slow_query_threshold_ms
        if is_slow:
            self._slow_queries += 1
            if self.config.log_slow_queries:
                logger.warning(
                    f"Slow query detected: {execution_time_ms:.2f}ms - {query[:200]}..."
                )

        # Update query statistics
        if query_hash in self._query_stats:
            stats = self._query_stats[query_hash]
            stats.execution_count += 1
            stats.total_time_ms += execution_time_ms
            stats.min_time_ms = min(stats.min_time_ms, execution_time_ms)
            stats.max_time_ms = max(stats.max_time_ms, execution_time_ms)
            stats.avg_time_ms = stats.total_time_ms / stats.execution_count
            stats.last_execution = time.time()

            if is_slow:
                stats.slow_query_count += 1
        else:
            # Check if we need to make room for new queries
            if len(self._query_stats) >= self.config.max_tracked_queries:
                # Remove the oldest query
                oldest_query = min(
                    self._query_stats.keys(),
                    key=lambda k: self._query_stats[k].last_execution,
                )
                del self._query_stats[oldest_query]

            # Create new stats entry
            self._query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_pattern=query_pattern,
                execution_count=1,
                total_time_ms=execution_time_ms,
                min_time_ms=execution_time_ms,
                max_time_ms=execution_time_ms,
                avg_time_ms=execution_time_ms,
                slow_query_count=1 if is_slow else 0,
                last_execution=time.time(),
            )
