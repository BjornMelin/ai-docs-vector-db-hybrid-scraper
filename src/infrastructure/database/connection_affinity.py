"""Connection affinity manager for query pattern optimization.

This module provides intelligent connection routing based on query patterns,
connection specialization, and performance optimization strategies.
"""

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries for connection routing."""

    READ = "read"
    WRITE = "write"
    ANALYTICS = "analytics"
    TRANSACTION = "transaction"
    MAINTENANCE = "maintenance"


class ConnectionSpecialization(Enum):
    """Connection specialization types."""

    GENERAL = "general"
    READ_OPTIMIZED = "read_optimized"
    WRITE_OPTIMIZED = "write_optimized"
    ANALYTICS_OPTIMIZED = "analytics_optimized"
    TRANSACTION_OPTIMIZED = "transaction_optimized"


@dataclass
class QueryPattern:
    """Represents a normalized query pattern with performance metrics."""

    pattern_id: str
    normalized_query: str
    sample_query: str
    query_type: QueryType

    # Performance metrics
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float("inf")
    max_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0

    # Connection performance tracking
    connection_performance: dict[str, float] = field(default_factory=dict)
    preferred_connection_id: str | None = None

    # Pattern characteristics
    complexity_score: float = 0.0
    resource_intensity: float = 0.0
    last_seen: float = field(default_factory=time.time)

    def update_performance(self, execution_time_ms: float) -> None:
        """Update performance metrics with new execution."""
        self.execution_count += 1
        self.total_execution_time_ms += execution_time_ms
        self.min_execution_time_ms = min(self.min_execution_time_ms, execution_time_ms)
        self.max_execution_time_ms = max(self.max_execution_time_ms, execution_time_ms)
        self.avg_execution_time_ms = self.total_execution_time_ms / self.execution_count
        self.last_seen = time.time()

    def add_connection_performance(
        self, connection_id: str, execution_time_ms: float
    ) -> None:
        """Track performance for specific connection."""
        if connection_id not in self.connection_performance:
            self.connection_performance[connection_id] = execution_time_ms
        else:
            # Exponential moving average
            alpha = 0.3
            current_avg = self.connection_performance[connection_id]
            self.connection_performance[connection_id] = (
                alpha * execution_time_ms + (1 - alpha) * current_avg
            )

        # Update preferred connection if this one performs better
        if not self.preferred_connection_id or self.connection_performance[
            connection_id
        ] < self.connection_performance.get(self.preferred_connection_id, float("inf")):
            self.preferred_connection_id = connection_id


@dataclass
class ConnectionStats:
    """Statistics for a database connection."""

    connection_id: str
    specialization: ConnectionSpecialization

    # Usage metrics
    total_queries: int = 0
    active_queries: int = 0
    last_used: float = field(default_factory=time.time)

    # Performance metrics
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0

    # Query type distribution
    query_type_counts: dict[QueryType, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

    def update_usage(
        self, query_type: QueryType, execution_time_ms: float, success: bool = True
    ) -> None:
        """Update connection usage statistics."""
        self.total_queries += 1
        self.query_type_counts[query_type] += 1
        self.last_used = time.time()

        # Update response time (exponential moving average)
        alpha = 0.2
        self.avg_response_time_ms = (
            alpha * execution_time_ms + (1 - alpha) * self.avg_response_time_ms
        )

        # Update success rate
        if success:
            self.success_rate = alpha + (1 - alpha) * self.success_rate
        else:
            self.error_count += 1
            self.success_rate = (1 - alpha) * self.success_rate

    def get_load_score(self) -> float:
        """Calculate current load score (0.0 = idle, 1.0 = fully loaded)."""
        # Consider active queries, recent usage, and response time
        base_load = min(
            1.0, self.active_queries / 10.0
        )  # Assume 10 concurrent queries = full load

        # Factor in recent usage
        time_since_last_use = time.time() - self.last_used
        recency_factor = max(
            0.0, 1.0 - (time_since_last_use / 60.0)
        )  # Decay over 1 minute

        # Factor in response time (higher response time = higher load)
        response_time_factor = min(
            1.0, self.avg_response_time_ms / 1000.0
        )  # 1 second = full load

        return min(1.0, (base_load + recency_factor * 0.3 + response_time_factor * 0.2))


class ConnectionAffinityManager:
    """Intelligent connection routing based on query patterns and connection specialization.

    This manager provides:
    - Query pattern recognition and normalization
    - Connection specialization based on usage patterns
    - Intelligent routing for optimal performance
    - Load balancing across specialized connections
    - Performance tracking and optimization
    """

    def __init__(self, max_patterns: int = 1000, max_connections: int = 50):
        """Initialize connection affinity manager.

        Args:
            max_patterns: Maximum number of query patterns to track
            max_connections: Maximum number of connections to manage
        """
        self.max_patterns = max_patterns
        self.max_connections = max_connections

        # Pattern and connection tracking
        self.query_patterns: dict[str, QueryPattern] = {}
        self.connection_stats: dict[str, ConnectionStats] = {}

        # Performance cache
        self.performance_cache: dict[
            tuple[str, str], float
        ] = {}  # (pattern_id, connection_id) -> avg_time
        self.cache_ttl = 300.0  # 5 minutes

        # Connection pools by specialization
        self.specialized_pools: dict[ConnectionSpecialization, list[str]] = {
            spec: [] for spec in ConnectionSpecialization
        }

        # Load balancing
        self._connection_locks: dict[str, asyncio.Lock] = {}
        self._round_robin_counters: dict[QueryType, int] = defaultdict(int)

        # Cleanup tracking
        self._last_cleanup = time.time()
        self._cleanup_interval = 600.0  # 10 minutes

    async def get_optimal_connection(
        self,
        query: str,
        query_type: QueryType = QueryType.READ,
        exclude_connections: list[str] | None = None,
    ) -> str | None:
        """Get optimal connection for the given query and type.

        Args:
            query: SQL query string
            query_type: Type of query operation
            exclude_connections: Connections to exclude from selection

        Returns:
            Optimal connection ID or None if no suitable connection available
        """
        exclude_connections = exclude_connections or []

        # Normalize query and get pattern
        pattern_id = self._normalize_query(query)
        pattern = await self._get_or_create_pattern(pattern_id, query, query_type)

        # Try to get connection based on pattern affinity
        if (
            pattern.preferred_connection_id
            and pattern.preferred_connection_id not in exclude_connections
        ):
            connection_id = pattern.preferred_connection_id
            if await self._is_connection_available(connection_id):
                await self._track_connection_selection(
                    connection_id, pattern_id, "affinity"
                )
                return connection_id

        # Fall back to specialized pool routing
        connection_id = await self._route_to_specialized_pool(
            query_type, exclude_connections
        )
        if connection_id:
            await self._track_connection_selection(
                connection_id, pattern_id, "specialized"
            )
            return connection_id

        # Last resort: round-robin load balancing
        connection_id = await self._round_robin_selection(
            query_type, exclude_connections
        )
        if connection_id:
            await self._track_connection_selection(
                connection_id, pattern_id, "round_robin"
            )
            return connection_id

        logger.warning(f"No available connections for query type {query_type.value}")
        return None

    async def register_connection(
        self,
        connection_id: str,
        specialization: ConnectionSpecialization = ConnectionSpecialization.GENERAL,
    ) -> None:
        """Register a new connection with the affinity manager.

        Args:
            connection_id: Unique connection identifier
            specialization: Connection specialization type
        """
        self.connection_stats[connection_id] = ConnectionStats(
            connection_id=connection_id, specialization=specialization
        )

        # Add to specialized pool
        if connection_id not in self.specialized_pools[specialization]:
            self.specialized_pools[specialization].append(connection_id)

        # Create connection lock
        self._connection_locks[connection_id] = asyncio.Lock()

        logger.info(
            f"Registered connection {connection_id} with specialization {specialization.value}"
        )

    async def unregister_connection(self, connection_id: str) -> None:
        """Unregister a connection from the affinity manager.

        Args:
            connection_id: Connection identifier to remove
        """
        if connection_id in self.connection_stats:
            specialization = self.connection_stats[connection_id].specialization

            # Remove from specialized pool
            if connection_id in self.specialized_pools[specialization]:
                self.specialized_pools[specialization].remove(connection_id)

            # Clean up
            del self.connection_stats[connection_id]
            if connection_id in self._connection_locks:
                del self._connection_locks[connection_id]

            # Update patterns that preferred this connection
            for pattern in self.query_patterns.values():
                if pattern.preferred_connection_id == connection_id:
                    pattern.preferred_connection_id = None
                if connection_id in pattern.connection_performance:
                    del pattern.connection_performance[connection_id]

            logger.info(f"Unregistered connection {connection_id}")

    async def track_query_performance(
        self,
        connection_id: str,
        query: str,
        execution_time_ms: float,
        query_type: QueryType,
        success: bool = True,
    ) -> None:
        """Track query performance for connection affinity optimization.

        Args:
            connection_id: Connection that executed the query
            query: SQL query string
            execution_time_ms: Query execution time in milliseconds
            query_type: Type of query operation
            success: Whether the query succeeded
        """
        # Normalize query and update pattern
        pattern_id = self._normalize_query(query)
        pattern = await self._get_or_create_pattern(pattern_id, query, query_type)

        pattern.update_performance(execution_time_ms)
        pattern.add_connection_performance(connection_id, execution_time_ms)

        # Update connection stats
        if connection_id in self.connection_stats:
            self.connection_stats[connection_id].update_usage(
                query_type, execution_time_ms, success
            )

        # Update performance cache
        cache_key = (pattern_id, connection_id)
        self.performance_cache[cache_key] = execution_time_ms

        # Trigger specialization analysis if needed
        if pattern.execution_count % 10 == 0:  # Every 10 executions
            await self._analyze_connection_specialization(connection_id)

        # Periodic cleanup
        await self._maybe_cleanup()

    async def get_connection_recommendations(
        self, query_type: QueryType
    ) -> dict[str, Any]:
        """Get connection recommendations for optimization.

        Args:
            query_type: Type of queries to analyze

        Returns:
            Dictionary with recommendations and analysis
        """
        available_connections = [
            conn_id
            for conn_id, stats in self.connection_stats.items()
            if await self._is_connection_available(conn_id)
        ]

        if not available_connections:
            return {"error": "No available connections"}

        # Analyze current load distribution
        load_analysis = {}
        specialization_analysis = {}

        for conn_id in available_connections:
            stats = self.connection_stats[conn_id]
            load_score = stats.get_load_score()

            load_analysis[conn_id] = {
                "load_score": load_score,
                "active_queries": stats.active_queries,
                "avg_response_time_ms": stats.avg_response_time_ms,
                "success_rate": stats.success_rate,
                "specialization": stats.specialization.value,
            }

            # Group by specialization
            spec = stats.specialization.value
            if spec not in specialization_analysis:
                specialization_analysis[spec] = []
            specialization_analysis[spec].append(conn_id)

        # Generate recommendations
        recommendations = []

        # Check for load balancing issues
        load_scores = [stats["load_score"] for stats in load_analysis.values()]
        if load_scores and max(load_scores) - min(load_scores) > 0.5:
            recommendations.append(
                "Consider redistributing load - significant imbalance detected"
            )

        # Check for performance issues
        slow_connections = [
            conn_id
            for conn_id, stats in load_analysis.items()
            if stats["avg_response_time_ms"] > 500  # 500ms threshold
        ]
        if slow_connections:
            recommendations.append(
                f"Performance issue detected in connections: {slow_connections}"
            )

        # Check specialization coverage
        optimal_query_type_mapping = {
            QueryType.READ: ConnectionSpecialization.READ_OPTIMIZED,
            QueryType.WRITE: ConnectionSpecialization.WRITE_OPTIMIZED,
            QueryType.ANALYTICS: ConnectionSpecialization.ANALYTICS_OPTIMIZED,
            QueryType.TRANSACTION: ConnectionSpecialization.TRANSACTION_OPTIMIZED,
        }

        optimal_spec = optimal_query_type_mapping.get(query_type)
        if optimal_spec and optimal_spec.value not in specialization_analysis:
            recommendations.append(
                f"Consider adding {optimal_spec.value} connections for {query_type.value} workloads"
            )

        return {
            "query_type": query_type.value,
            "available_connections": len(available_connections),
            "load_analysis": load_analysis,
            "specialization_analysis": specialization_analysis,
            "recommendations": recommendations,
            "pattern_cache_size": len(self.query_patterns),
            "performance_cache_size": len(self.performance_cache),
        }

    def _normalize_query(self, query: str) -> str:
        """Normalize SQL query to create a pattern identifier."""
        if not query:
            return "empty_query"

        # Convert to lowercase and normalize whitespace
        normalized = re.sub(r"\s+", " ", query.lower().strip())

        # Replace literals with placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # String literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)  # Numeric literals
        normalized = re.sub(r"\b\d+\.\d+\b", "?", normalized)  # Decimal literals

        # Normalize IN clauses
        normalized = re.sub(r"in\s*\([^)]+\)", "in (?)", normalized)

        # Normalize BETWEEN clauses
        normalized = re.sub(r"between\s+\S+\s+and\s+\S+", "between ? and ?", normalized)

        # Create hash for very long queries
        if len(normalized) > 200:
            hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:8]
            normalized = normalized[:150] + f"...#{hash_suffix}"

        return normalized

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type based on SQL keywords."""
        query_lower = query.lower().strip()

        # Check for transaction keywords
        if any(
            keyword in query_lower
            for keyword in ["begin", "commit", "rollback", "savepoint"]
        ):
            return QueryType.TRANSACTION

        # Check for write operations
        if any(
            keyword in query_lower
            for keyword in ["insert", "update", "delete", "create", "drop", "alter"]
        ):
            return QueryType.WRITE

        # Check for analytics patterns
        analytics_keywords = [
            "group by",
            "having",
            "window",
            "partition by",
            "over(",
            "sum(",
            "count(",
            "avg(",
            "max(",
            "min(",
        ]
        if any(keyword in query_lower for keyword in analytics_keywords):
            return QueryType.ANALYTICS

        # Check for maintenance operations
        if any(
            keyword in query_lower
            for keyword in ["analyze", "vacuum", "reindex", "cluster"]
        ):
            return QueryType.MAINTENANCE

        # Default to read for SELECT and other operations
        return QueryType.READ

    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.0 - 1.0)."""
        query_lower = query.lower()
        complexity = 0.0

        # Base complexity factors
        complexity += len(query) / 1000.0  # Length factor
        complexity += query_lower.count("join") * 0.2
        complexity += query_lower.count("subquery") * 0.3
        complexity += query_lower.count("union") * 0.2
        complexity += query_lower.count("group by") * 0.2
        complexity += query_lower.count("order by") * 0.1
        complexity += query_lower.count("having") * 0.2
        complexity += query_lower.count("exists") * 0.3
        complexity += query_lower.count("in (select") * 0.4

        return min(1.0, complexity)

    async def _get_or_create_pattern(
        self, pattern_id: str, sample_query: str, query_type: QueryType
    ) -> QueryPattern:
        """Get existing pattern or create new one."""
        if pattern_id not in self.query_patterns:
            # Check if we need to evict old patterns
            if len(self.query_patterns) >= self.max_patterns:
                await self._evict_old_patterns()

            # Create new pattern
            complexity = self._calculate_query_complexity(sample_query)

            self.query_patterns[pattern_id] = QueryPattern(
                pattern_id=pattern_id,
                normalized_query=pattern_id,
                sample_query=sample_query,
                query_type=query_type,
                complexity_score=complexity,
                resource_intensity=complexity * 0.8,  # Approximate resource intensity
            )

        return self.query_patterns[pattern_id]

    async def _route_to_specialized_pool(
        self, query_type: QueryType, exclude_connections: list[str]
    ) -> str | None:
        """Route query to specialized connection pool."""
        # Map query types to preferred specializations
        specialization_preference = {
            QueryType.READ: [
                ConnectionSpecialization.READ_OPTIMIZED,
                ConnectionSpecialization.GENERAL,
            ],
            QueryType.WRITE: [
                ConnectionSpecialization.WRITE_OPTIMIZED,
                ConnectionSpecialization.GENERAL,
            ],
            QueryType.ANALYTICS: [
                ConnectionSpecialization.ANALYTICS_OPTIMIZED,
                ConnectionSpecialization.READ_OPTIMIZED,
                ConnectionSpecialization.GENERAL,
            ],
            QueryType.TRANSACTION: [
                ConnectionSpecialization.TRANSACTION_OPTIMIZED,
                ConnectionSpecialization.WRITE_OPTIMIZED,
                ConnectionSpecialization.GENERAL,
            ],
            QueryType.MAINTENANCE: [ConnectionSpecialization.GENERAL],
        }

        preferred_specs = specialization_preference.get(
            query_type, [ConnectionSpecialization.GENERAL]
        )

        for specialization in preferred_specs:
            available_connections = [
                conn_id
                for conn_id in self.specialized_pools[specialization]
                if conn_id not in exclude_connections
                and await self._is_connection_available(conn_id)
            ]

            if available_connections:
                # Select least loaded connection
                best_connection = min(
                    available_connections,
                    key=lambda conn_id: self.connection_stats[conn_id].get_load_score(),
                )
                return best_connection

        return None

    async def _round_robin_selection(
        self, query_type: QueryType, exclude_connections: list[str]
    ) -> str | None:
        """Select connection using round-robin load balancing."""
        available_connections = [
            conn_id
            for conn_id in self.connection_stats.keys()
            if conn_id not in exclude_connections
            and await self._is_connection_available(conn_id)
        ]

        if not available_connections:
            return None

        # Simple round-robin
        counter = self._round_robin_counters[query_type]
        selected_connection = available_connections[
            counter % len(available_connections)
        ]
        self._round_robin_counters[query_type] = counter + 1

        return selected_connection

    async def _is_connection_available(self, connection_id: str) -> bool:
        """Check if connection is available for use."""
        if connection_id not in self.connection_stats:
            return False

        stats = self.connection_stats[connection_id]

        # Check basic availability criteria
        if stats.success_rate < 0.5:  # Less than 50% success rate
            return False

        if stats.get_load_score() > 0.9:  # Over 90% loaded
            return False

        # Check if connection is responsive (last used within reasonable time)
        time_since_last_use = time.time() - stats.last_used
        if time_since_last_use > 1800:  # 30 minutes
            return False

        return True

    async def _track_connection_selection(
        self, connection_id: str, pattern_id: str, selection_method: str
    ) -> None:
        """Track connection selection for analytics."""
        logger.debug(
            f"Selected connection {connection_id} for pattern {pattern_id[:50]} "
            f"using method {selection_method}"
        )

        # Increment active queries for the connection
        if connection_id in self.connection_stats:
            self.connection_stats[connection_id].active_queries += 1

    async def _analyze_connection_specialization(self, connection_id: str) -> None:
        """Analyze and potentially update connection specialization."""
        if connection_id not in self.connection_stats:
            return

        stats = self.connection_stats[connection_id]

        # Analyze query type distribution
        total_queries = sum(stats.query_type_counts.values())
        if total_queries < 10:  # Not enough data
            return

        query_type_ratios = {
            query_type: count / total_queries
            for query_type, count in stats.query_type_counts.items()
        }

        # Determine if specialization should be updated
        dominant_query_type = max(query_type_ratios, key=query_type_ratios.get)
        dominant_ratio = query_type_ratios[dominant_query_type]

        if dominant_ratio > 0.7:  # 70% of queries are of one type
            optimal_specialization = {
                QueryType.READ: ConnectionSpecialization.READ_OPTIMIZED,
                QueryType.WRITE: ConnectionSpecialization.WRITE_OPTIMIZED,
                QueryType.ANALYTICS: ConnectionSpecialization.ANALYTICS_OPTIMIZED,
                QueryType.TRANSACTION: ConnectionSpecialization.TRANSACTION_OPTIMIZED,
            }.get(dominant_query_type, ConnectionSpecialization.GENERAL)

            if stats.specialization != optimal_specialization:
                logger.info(
                    f"Suggesting specialization change for connection {connection_id}: "
                    f"{stats.specialization.value} -> {optimal_specialization.value} "
                    f"(dominant query type: {dominant_query_type.value} at {dominant_ratio:.1%})"
                )

    async def _evict_old_patterns(self) -> None:
        """Evict old, unused query patterns to free memory."""
        current_time = time.time()

        # Sort patterns by last seen time and execution count
        patterns_by_age = sorted(
            self.query_patterns.items(),
            key=lambda x: (x[1].last_seen, x[1].execution_count),
        )

        # Remove oldest 10% of patterns
        evict_count = max(1, len(patterns_by_age) // 10)

        for pattern_id, pattern in patterns_by_age[:evict_count]:
            # Only evict if pattern hasn't been used recently
            if current_time - pattern.last_seen > 3600:  # 1 hour
                del self.query_patterns[pattern_id]
                logger.debug(f"Evicted old query pattern: {pattern_id[:50]}")

    async def _maybe_cleanup(self) -> None:
        """Perform periodic cleanup of caches and metrics."""
        current_time = time.time()

        if current_time - self._last_cleanup > self._cleanup_interval:
            # Clean performance cache
            expired_keys = [
                key
                for key, timestamp in self.performance_cache.items()
                if current_time - timestamp > self.cache_ttl
            ]

            for key in expired_keys:
                del self.performance_cache[key]

            # Clean old query patterns
            await self._evict_old_patterns()

            self._last_cleanup = current_time

            logger.debug(
                f"Cleanup completed: {len(expired_keys)} cache entries removed, "
                f"{len(self.query_patterns)} patterns retained"
            )

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "summary": {
                "total_patterns": len(self.query_patterns),
                "total_connections": len(self.connection_stats),
                "cache_size": len(self.performance_cache),
            },
            "top_patterns": [
                {
                    "pattern_id": pattern.pattern_id[:100],
                    "execution_count": pattern.execution_count,
                    "avg_time_ms": pattern.avg_execution_time_ms,
                    "query_type": pattern.query_type.value,
                    "preferred_connection": pattern.preferred_connection_id,
                }
                for pattern in sorted(
                    self.query_patterns.values(),
                    key=lambda p: p.execution_count,
                    reverse=True,
                )[:10]
            ],
            "connection_performance": {
                conn_id: {
                    "specialization": stats.specialization.value,
                    "total_queries": stats.total_queries,
                    "active_queries": stats.active_queries,
                    "avg_response_time_ms": stats.avg_response_time_ms,
                    "success_rate": stats.success_rate,
                    "load_score": stats.get_load_score(),
                    "query_type_distribution": {
                        qtype.value: count
                        for qtype, count in stats.query_type_counts.items()
                    },
                }
                for conn_id, stats in self.connection_stats.items()
            },
            "specialization_distribution": {
                spec.value: len(connections)
                for spec, connections in self.specialized_pools.items()
            },
        }
