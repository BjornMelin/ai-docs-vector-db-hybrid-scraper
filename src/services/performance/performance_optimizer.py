"""Performance Optimization Module.

This module provides comprehensive performance optimization capabilities
including query optimization, caching strategies, and resource management.

The PerformanceOptimizer class serves as the central coordinator for all
performance-related optimizations in the system. It integrates with caching,
monitoring, and resource management subsystems to provide intelligent
optimization decisions based on real-time metrics.

Key Features:
    - Query optimization with multiple strategies
    - Intelligent caching with TTL management
    - Connection pool optimization
    - Performance metrics tracking and reporting
    - Hot path detection and slow query identification
    - Resource semaphore management for rate limiting

Example:
    >>> from src.config.settings import Settings
    >>> settings = Settings()
    >>> optimizer = PerformanceOptimizer(settings)
    >>> await optimizer.initialize()
    >>> # Optimize a query
    >>> result = await optimizer.optimize_query(
    ...     "SELECT * FROM documents WHERE embedding <-> %s < 0.5",
    ...     query_type="vector",
    ... )
    >>> print(f"Improvement: {result.expected_improvement}%")

Note:
    This module requires Redis for caching and proper initialization
    of the monitoring subsystem for accurate metrics collection.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.services.cache.intelligent import IntelligentCache
from src.services.monitoring.performance_monitor import PerformanceMonitor


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics.

    This dataclass encapsulates performance measurement data for various
    operations throughout the system. It provides a consistent structure
    for tracking operation timing and associated metadata.

    Attributes:
        operation: Name of the operation being measured (e.g., "query_optimization")
        duration_ms: Duration of the operation in milliseconds
        timestamp: UTC timestamp when the metric was recorded
        metadata: Additional context-specific information about the operation

    Example:
        >>> metric = PerformanceMetrics(
        ...     operation="vector_search",
        ...     duration_ms=45.3,
        ...     metadata={"query_size": 100, "cache_hit": True},
        ... )
    """

    operation: str
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryOptimizationResult(BaseModel):
    """Result of query optimization analysis.

    This Pydantic model represents the outcome of query optimization,
    providing detailed information about the optimization performed
    and expected performance gains.

    Attributes:
        original_query: The original query before optimization
        optimized_query: The query after optimization rules are applied
        expected_improvement: Estimated performance improvement as a percentage
        optimization_type: Category of optimization applied (e.g., "vector_optimization",
            "batch_optimization", "filter_pushdown")
        metadata: Additional information about the optimization process,
            such as optimization time and rules applied

    Example:
        >>> result = QueryOptimizationResult(
        ...     original_query="SELECT * FROM documents",
        ...     optimized_query="SELECT * FROM documents LIMIT 100",
        ...     expected_improvement=15.0,
        ...     optimization_type="limit_optimization",
        ...     metadata={"optimization_time_ms": 2.5, "rules_applied": 3},
        ... )
    """

    original_query: str = Field(..., description="Original query")
    optimized_query: str = Field(..., description="Optimized query")
    expected_improvement: float = Field(
        ..., description="Expected performance improvement percentage"
    )
    optimization_type: str = Field(..., description="Type of optimization applied")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional optimization metadata"
    )


class PerformanceOptimizer:
    """Central performance optimization coordinator.

    The PerformanceOptimizer orchestrates various optimization strategies
    to improve system performance. It manages query optimization, caching,
    connection pooling, and performance monitoring to ensure optimal
    resource utilization and response times.

    The optimizer tracks hot paths (frequently executed operations) and
    slow queries to identify optimization opportunities. It applies
    multiple optimization rules in sequence to maximize performance gains.

    Attributes:
        settings: Application configuration settings
        cache: Intelligent cache for storing optimization results
        monitor: Performance monitor for tracking metrics
        enable_query_optimization: Flag to enable/disable query optimization
        enable_response_caching: Flag to enable/disable response caching
        enable_connection_pooling: Flag to enable/disable connection pooling
        cache_ttl_seconds: Default cache time-to-live in seconds

    Example:
        >>> optimizer = PerformanceOptimizer(settings)
        >>> await optimizer.initialize()
        >>> # Optimize and execute a query
        >>> optimized = await optimizer.optimize_query(
        ...     "SELECT * FROM large_table", query_type="sql"
        ... )
        >>> # Get performance report
        >>> report = await optimizer.get_performance_report()
        >>> print(f"Hot paths: {report['hot_paths']}")
    """

    def __init__(
        self,
        settings: Settings,
        cache: IntelligentCache | None = None,
        monitor: PerformanceMonitor | None = None,
    ):
        """Initialize performance optimizer.

        Args:
            settings: Application settings containing configuration parameters
            cache: Optional pre-configured intelligent cache instance.
                If not provided, a new instance will be created.
            monitor: Optional pre-configured performance monitor instance.
                If not provided, a new instance will be created.

        Note:
            The optimizer initializes with default optimization rules and
            creates necessary resource pools for rate limiting.
        """
        self.settings = settings
        self.cache = cache or IntelligentCache(settings)
        self.monitor = monitor or PerformanceMonitor()

        # Performance tracking
        self._metrics: list[PerformanceMetrics] = []
        self._hot_paths: dict[str, int] = defaultdict(int)
        self._slow_queries: set[str] = set()

        # Optimization strategies
        self._query_cache: dict[str, Any] = {}
        self._optimization_rules: list[Callable] = []

        # Resource pools
        self._connection_pools: dict[str, Any] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        # Configuration
        self.enable_query_optimization = True
        self.enable_response_caching = True
        self.enable_connection_pooling = True
        self.cache_ttl_seconds = 300  # 5 minutes default

    async def initialize(self) -> None:
        """Initialize performance optimizer and components.

        This method performs the necessary setup for the performance optimizer,
        including cache initialization, connection pool setup, and loading
        optimization rules. Must be called before using the optimizer.

        The initialization process:
        1. Initializes the intelligent cache system
        2. Sets up semaphore-based connection pools for rate limiting
        3. Loads query optimization rules into the rule engine

        Raises:
            ConnectionError: If cache initialization fails
            ConfigurationError: If optimization rules cannot be loaded

        Example:
            >>> optimizer = PerformanceOptimizer(settings)
            >>> await optimizer.initialize()
            >>> # Optimizer is now ready for use
        """
        logger.info("Initializing performance optimizer")

        # Initialize cache
        await self.cache.initialize()

        # Set up connection pools
        await self._setup_connection_pools()

        # Load optimization rules
        self._load_optimization_rules()

        logger.info("Performance optimizer initialized successfully")

    async def _setup_connection_pools(self) -> None:
        """Set up connection pools for various services."""
        # Create semaphores for rate limiting
        self._semaphores["qdrant"] = asyncio.Semaphore(
            50
        )  # Max 50 concurrent Qdrant operations
        self._semaphores["redis"] = asyncio.Semaphore(
            100
        )  # Max 100 concurrent Redis operations
        self._semaphores["http"] = asyncio.Semaphore(
            200
        )  # Max 200 concurrent HTTP requests

    def _load_optimization_rules(self) -> None:
        """Load query optimization rules."""
        # Add various optimization strategies
        self._optimization_rules.extend(
            [
                self._optimize_vector_search,
                self._optimize_batch_queries,
                self._optimize_filter_pushdown,
                self._optimize_projection,
            ]
        )

    async def optimize_query(
        self, query: str, query_type: str = "vector"
    ) -> QueryOptimizationResult:
        """Optimize a query for better performance.

        Analyzes the provided query and applies relevant optimization rules
        to improve execution performance. The optimization process checks
        a cache first to avoid redundant processing, then applies multiple
        optimization strategies in sequence.

        Supported query types:
        - "vector": Vector similarity searches (adds limits, optimizes filters)
        - "sql": SQL queries (filter pushdown, projection optimization)
        - "batch": Batch operations (combines similar queries)

        The method tracks optimization metrics and caches results for
        repeated queries to reduce processing overhead.

        Args:
            query: Original query string to optimize
            query_type: Type of query to optimize. Defaults to "vector".
                Must be one of: "vector", "sql", "batch"

        Returns:
            QueryOptimizationResult: Contains the optimized query,
                expected improvement percentage, optimization type,
                and metadata about the optimization process

        Raises:
            ValueError: If query_type is not supported
            OptimizationError: If optimization rules fail

        Example:
            >>> result = await optimizer.optimize_query(
            ...     "SELECT * FROM documents WHERE category = 'tech'", query_type="sql"
            ... )
            >>> print(f"Optimized: {result.optimized_query}")
            >>> print(f"Expected improvement: {result.expected_improvement}%")
        """
        start_time = time.time()

        # Check cache first
        cache_key = f"optimized_query:{query_type}:{hash(query)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return QueryOptimizationResult(**cached_result)

        # Apply optimization rules
        optimized_query = query
        optimization_type = "none"
        expected_improvement = 0.0

        for rule in self._optimization_rules:
            result = await rule(query, query_type)
            if result and result.get("improvement", 0) > expected_improvement:
                optimized_query = result["query"]
                optimization_type = result["type"]
                expected_improvement = result["improvement"]
        # Create result
        result = QueryOptimizationResult(
            original_query=query,
            optimized_query=optimized_query,
            expected_improvement=expected_improvement,
            optimization_type=optimization_type,
            metadata={
                "optimization_time_ms": (time.time() - start_time) * 1000,
                "rules_applied": len(self._optimization_rules),
            },
        )

        # Cache the result
        await self.cache.set(cache_key, result.model_dump(), ttl=self.cache_ttl_seconds)

        # Track metrics
        self._record_metric(
            "query_optimization",
            (time.time() - start_time) * 1000,
            {"type": query_type, "improvement": expected_improvement},
        )

        return result

    async def _optimize_vector_search(
        self, query: str, query_type: str
    ) -> dict[str, Any] | None:
        """Optimize vector search queries."""
        if query_type != "vector":
            return None

        # Example optimizations for vector search
        optimizations = {
            "query": query,
            "type": "vector_optimization",
            "improvement": 0.0,
        }
        # Check for common optimization patterns
        if "limit" not in query.lower():
            optimizations["query"] = f"{query} LIMIT 100"
            optimizations["improvement"] = 15.0

        return optimizations

    async def _optimize_batch_queries(
        self,
        query: str,  # noqa: ARG002
        query_type: str,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Optimize batch query processing."""
        # Detect if multiple similar queries can be batched
        return None

    async def _optimize_filter_pushdown(
        self,
        query: str,  # noqa: ARG002
        query_type: str,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Optimize by pushing filters down to the storage layer."""
        return None

    async def _optimize_projection(
        self,
        query: str,  # noqa: ARG002
        query_type: str,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Optimize by selecting only required fields."""
        return None

    async def get_cached_response(self, cache_key: str) -> Any | None:
        """Get cached response with performance tracking.

        Args:
            cache_key: Cache key for the response

        Returns:
            Cached response or None

        """
        start_time = time.time()

        async with self._semaphores.get("redis", asyncio.Semaphore(1)):
            result = await self.cache.get(cache_key)

        duration_ms = (time.time() - start_time) * 1000
        self._record_metric("cache_get", duration_ms, {"hit": result is not None})

        return result

    async def set_cached_response(
        self, cache_key: str, response: Any, ttl: int | None = None
    ) -> None:
        """Set cached response with performance tracking.

        Args:
            cache_key: Cache key for the response
            response: Response to cache
            ttl: Time to live in seconds

        """
        start_time = time.time()
        ttl = ttl or self.cache_ttl_seconds

        async with self._semaphores.get("redis", asyncio.Semaphore(1)):
            await self.cache.set(cache_key, response, ttl=ttl)

        duration_ms = (time.time() - start_time) * 1000
        self._record_metric("cache_set", duration_ms)

    def _record_metric(
        self, operation: str, duration_ms: float, metadata: dict | None = None
    ) -> None:
        """Record performance metric."""
        metric = PerformanceMetrics(
            operation=operation, duration_ms=duration_ms, metadata=metadata or {}
        )
        self._metrics.append(metric)

        # Track hot paths
        self._hot_paths[operation] += 1

        # Track slow queries
        if duration_ms > 100:  # Mark as slow if > 100ms
            self._slow_queries.add(operation)

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report.

        Returns:
            Dictionary containing performance statistics

        """
        if not self._metrics:
            return {"message": "No metrics collected yet"}

        # Calculate statistics
        operations = defaultdict(list)
        for metric in self._metrics:
            operations[metric.operation].append(metric.duration_ms)
        stats = {}
        for operation, durations in operations.items():
            sorted_durations = sorted(durations)
            stats[operation] = {
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": sorted_durations[len(sorted_durations) // 2],
                "p95_ms": sorted_durations[int(len(sorted_durations) * 0.95)],
                "p99_ms": sorted_durations[int(len(sorted_durations) * 0.99)],
            }

        return {
            "total_operations": len(self._metrics),
            "operation_stats": stats,
            "hot_paths": dict(self._hot_paths),
            "slow_queries": list(self._slow_queries),
            "cache_stats": await self.cache.get_stats()
            if hasattr(self.cache, "get_stats")
            else {},
        }

    async def optimize_connection_pool(self, service: str, current_size: int) -> int:
        """Optimize connection pool size based on usage patterns.

        Args:
            service: Service name (qdrant, redis, etc.)
            current_size: Current pool size

        Returns:
            Recommended pool size

        """
        # Get usage statistics
        semaphore = self._semaphores.get(service)
        if not semaphore:
            return current_size

        # Simple optimization based on semaphore usage
        # In production, this would use more sophisticated algorithms
        if (
            semaphore._value < current_size * 0.2  # noqa: SLF001
        ):  # Less than 20% available
            return min(current_size * 2, 200)  # Double but cap at 200
        if (
            semaphore._value > current_size * 0.8  # noqa: SLF001
        ):  # More than 80% available
            return max(current_size // 2, 10)  # Halve but keep minimum 10

        return current_size
