"""Central metrics registry and monitoring decorators for application observability.

This module provides a comprehensive metrics collection system using Prometheus
with support for vector search, embeddings, cache, and ML model performance monitoring.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.registry import REGISTRY, CollectorRegistry
from pydantic import BaseModel, Field


F = TypeVar("F", bound=Callable[..., Any])


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""

    enabled: bool = Field(default=True, description="Enable metrics collection")
    export_port: int = Field(default=8000, description="Prometheus metrics export port")
    namespace: str = Field(default="ml_app", description="Metrics namespace prefix")
    include_system_metrics: bool = Field(
        default=True, description="Include system metrics"
    )
    collection_interval: float = Field(
        default=1.0, description="Metrics collection interval in seconds"
    )


class MetricsRegistry:
    """Central registry for all application metrics.

    Manages Prometheus metrics for vector search, embeddings, cache performance,
    and system health monitoring. Provides decorators for automatic metric collection.
    """

    def __init__(
        self, config: MetricsConfig, registry: CollectorRegistry | None = None
    ):
        """Initialize metrics registry.

        Args:
            config: Metrics configuration
            registry: Prometheus registry (uses default if not provided)

        """
        self.config = config
        self.registry = registry or REGISTRY
        self._metrics: dict[str, Any] = {}
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Set up all application metrics."""
        namespace = self.config.namespace

        # === Vector Search Metrics ===
        self._metrics["search_duration"] = Histogram(
            f"{namespace}_vector_search_duration_seconds",
            "Time spent on vector search operations",
            ["collection", "query_type"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        self._metrics["search_requests"] = Counter(
            f"{namespace}_vector_search_requests_total",
            "Total number of vector search requests",
            ["collection", "status"],
            registry=self.registry,
        )

        self._metrics["search_concurrent"] = Gauge(
            f"{namespace}_vector_search_concurrent_requests",
            "Current number of concurrent search requests",
            registry=self.registry,
        )

        self._metrics["search_result_quality"] = Histogram(
            f"{namespace}_vector_search_result_quality_score",
            "Quality score of search results",
            ["collection"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

        # === Embedding Generation Metrics ===
        self._metrics["embedding_duration"] = Histogram(
            f"{namespace}_embedding_generation_duration_seconds",
            "Time spent generating embeddings",
            ["provider", "model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        self._metrics["embedding_requests"] = Counter(
            f"{namespace}_embedding_requests_total",
            "Total number of embedding generation requests",
            ["provider", "model", "status"],
            registry=self.registry,
        )

        self._metrics["embedding_queue_depth"] = Gauge(
            f"{namespace}_embedding_queue_depth",
            "Current embedding queue depth",
            ["provider"],
            registry=self.registry,
        )

        self._metrics["embedding_cost"] = Counter(
            f"{namespace}_embedding_cost_total",
            "Total cost of embedding generation",
            ["provider", "model"],
            registry=self.registry,
        )

        self._metrics["embedding_batch_size"] = Histogram(
            f"{namespace}_embedding_batch_size",
            "Size of embedding generation batches",
            ["provider"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry,
        )

        # === Cache Performance Metrics ===
        self._metrics["cache_hits"] = Counter(
            f"{namespace}_cache_hits_total",
            "Total number of cache hits",
            ["cache_layer", "cache_type"],
            registry=self.registry,
        )

        self._metrics["cache_misses"] = Counter(
            f"{namespace}_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        self._metrics["cache_operations"] = Counter(
            f"{namespace}_cache_operations_total",
            "Total cache operations by type and result",
            ["cache_type", "operation", "result"],
            registry=self.registry,
        )

        self._metrics["cache_duration"] = Histogram(
            f"{namespace}_cache_operation_duration_seconds",
            "Cache operation latency",
            ["cache_type", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )

        self._metrics["cache_memory_usage"] = Gauge(
            f"{namespace}_cache_memory_usage_bytes",
            "Memory usage of cache",
            ["cache_type", "cache_name"],
            registry=self.registry,
        )

        self._metrics["cache_evictions"] = Counter(
            f"{namespace}_cache_evictions_total",
            "Total number of cache evictions",
            ["cache_type", "cache_name"],
            registry=self.registry,
        )

        # === Qdrant Metrics ===
        self._metrics["qdrant_collection_size"] = Gauge(
            f"{namespace}_qdrant_collection_size",
            "Number of vectors in Qdrant collection",
            ["collection"],
            registry=self.registry,
        )

        self._metrics["qdrant_memory_usage"] = Gauge(
            f"{namespace}_qdrant_memory_usage_bytes",
            "Memory usage of Qdrant collections",
            ["collection"],
            registry=self.registry,
        )

        self._metrics["qdrant_operations"] = Counter(
            f"{namespace}_qdrant_operations_total",
            "Total Qdrant operations",
            ["operation", "collection", "status"],
            registry=self.registry,
        )

        # === Task Queue Metrics ===
        self._metrics["task_queue_size"] = Gauge(
            f"{namespace}_task_queue_size",
            "Number of tasks in queue",
            ["queue", "status"],
            registry=self.registry,
        )

        self._metrics["task_execution_duration"] = Histogram(
            f"{namespace}_task_execution_duration_seconds",
            "Time spent executing tasks",
            ["task_name", "status"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
            registry=self.registry,
        )

        self._metrics["task_requests"] = Counter(
            f"{namespace}_task_requests_total",
            "Total number of task requests",
            ["task_name", "status"],
            registry=self.registry,
        )

        self._metrics["worker_active"] = Gauge(
            f"{namespace}_workers_active",
            "Number of active workers",
            ["queue"],
            registry=self.registry,
        )

        # === Browser Automation Metrics ===
        self._metrics["browser_tier_health"] = Gauge(
            f"{namespace}_browser_tier_health_status",
            "Health status of browser automation tiers (1=healthy, 0=unhealthy)",
            ["tier"],
            registry=self.registry,
        )

        self._metrics["browser_requests"] = Counter(
            f"{namespace}_browser_requests_total",
            "Total browser automation requests",
            ["tier", "status"],
            registry=self.registry,
        )

        self._metrics["browser_response_time"] = Histogram(
            f"{namespace}_browser_response_time_seconds",
            "Browser automation response times",
            ["tier"],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        # === Health & System Metrics ===
        self._metrics["service_health"] = Gauge(
            f"{namespace}_service_health_status",
            "Health status of services (1=healthy, 0=unhealthy)",
            ["service"],
            registry=self.registry,
        )

        self._metrics["dependency_health"] = Gauge(
            f"{namespace}_dependency_health_status",
            "Health status of dependencies (1=healthy, 0=unhealthy)",
            ["dependency"],
            registry=self.registry,
        )

        if self.config.include_system_metrics:
            self._setup_system_metrics()

    def _setup_system_metrics(self) -> None:
        """Set up system-level metrics."""
        namespace = self.config.namespace

        self._metrics["system_cpu_usage"] = Gauge(
            f"{namespace}_system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry,
        )

        self._metrics["system_memory_usage"] = Gauge(
            f"{namespace}_system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry,
        )

        self._metrics["system_disk_usage"] = Gauge(
            f"{namespace}_system_disk_usage_bytes",
            "System disk usage in bytes",
            ["path"],
            registry=self.registry,
        )

    def update_system_metrics(self) -> None:
        """Update system metrics with current values."""
        if not self.config.include_system_metrics:
            return

        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self._metrics["system_cpu_usage"].set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self._metrics["system_memory_usage"].set(memory.used)

        # Disk usage for root
        disk = psutil.disk_usage("/")
        self._metrics["system_disk_usage"].labels(path="/").set(disk.used)

    def monitor_search_performance(
        self, collection: str = "default", query_type: str = "semantic"
    ) -> Callable[[F], F]:
        """Decorator to monitor vector search performance.

        Args:
            collection: Name of the vector collection
            query_type: Type of query (semantic, hybrid, etc.)

        Returns:
            Decorated function with search performance monitoring

        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                self._metrics["search_concurrent"].inc()

                try:
                    result = await func(*args, **kwargs)
                except Exception:
                    self._metrics["search_requests"].labels(
                        collection=collection, status="error"
                    ).inc()
                    raise
                else:
                    self._metrics["search_requests"].labels(
                        collection=collection, status="success"
                    ).inc()

                    # Extract quality score if available
                    if hasattr(result, "scores") and result.scores:
                        max_score = max(result.scores)
                        self._metrics["search_result_quality"].labels(
                            collection=collection
                        ).observe(max_score)

                    return result

                finally:
                    duration = time.time() - start_time
                    self._metrics["search_duration"].labels(
                        collection=collection, query_type=query_type
                    ).observe(duration)
                    self._metrics["search_concurrent"].dec()

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                self._metrics["search_concurrent"].inc()

                try:
                    result = func(*args, **kwargs)
                except Exception:
                    self._metrics["search_requests"].labels(
                        collection=collection, status="error"
                    ).inc()
                    raise
                else:
                    self._metrics["search_requests"].labels(
                        collection=collection, status="success"
                    ).inc()
                    return result

                finally:
                    duration = time.time() - start_time
                    self._metrics["search_duration"].labels(
                        collection=collection, query_type=query_type
                    ).observe(duration)
                    self._metrics["search_concurrent"].dec()

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def monitor_embedding_generation(
        self, provider: str, model: str = "default"
    ) -> Callable[[F], F]:
        """Decorator to monitor embedding generation performance.

        Args:
            provider: Embedding provider (openai, fastembed, etc.)
            model: Model name

        Returns:
            Decorated function with embedding monitoring

        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    self._metrics["embedding_requests"].labels(
                        provider=provider, model=model, status="success"
                    ).inc()

                    # Track batch size if available
                    if hasattr(result, "__len__"):
                        self._metrics["embedding_batch_size"].labels(
                            provider=provider
                        ).observe(len(result))

                    return result

                except Exception:
                    self._metrics["embedding_requests"].labels(
                        provider=provider, model=model, status="error"
                    ).inc()
                    raise

                finally:
                    duration = time.time() - start_time
                    self._metrics["embedding_duration"].labels(
                        provider=provider, model=model
                    ).observe(duration)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    self._metrics["embedding_requests"].labels(
                        provider=provider, model=model, status="success"
                    ).inc()
                    return result

                except Exception:
                    self._metrics["embedding_requests"].labels(
                        provider=provider, model=model, status="error"
                    ).inc()
                    raise

                finally:
                    duration = time.time() - start_time
                    self._metrics["embedding_duration"].labels(
                        provider=provider, model=model
                    ).observe(duration)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def monitor_cache_operations(
        self, cache_type: str, cache_name: str
    ) -> Callable[[F], F]:
        """Decorator to monitor cache operations.

        Args:
            cache_type: Type of cache (redis, local, embedding, etc.)
            cache_name: Name of the cache instance

        Returns:
            Decorated function with cache monitoring

        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    if result is not None:
                        self._metrics["cache_hits"].labels(
                            cache_type=cache_type, cache_name=cache_name
                        ).inc()
                    else:
                        self._metrics["cache_misses"].labels(
                            cache_type=cache_type, cache_name=cache_name
                        ).inc()
                    return result
                except Exception:
                    self._metrics["cache_misses"].labels(
                        cache_type=cache_type, cache_name=cache_name
                    ).inc()
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        self._metrics["cache_hits"].labels(
                            cache_type=cache_type, cache_name=cache_name
                        ).inc()
                    else:
                        self._metrics["cache_misses"].labels(
                            cache_type=cache_type, cache_name=cache_name
                        ).inc()
                    return result
                except Exception:
                    self._metrics["cache_misses"].labels(
                        cache_type=cache_type, cache_name=cache_name
                    ).inc()
                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def monitor_cache_performance(
        self, cache_type: str, operation: str = "get"
    ) -> Callable[[F], F]:
        """Decorator to monitor cache performance.

        Args:
            cache_type: Type of cache (local, distributed, embeddings, etc.)
            operation: Cache operation (get, set, delete, etc.)

        Returns:
            Decorated function with cache performance monitoring

        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    self._metrics["cache_operations"].labels(
                        cache_type=cache_type, operation=operation, result="success"
                    ).inc()
                    return result

                except Exception:
                    self._metrics["cache_operations"].labels(
                        cache_type=cache_type, operation=operation, result="error"
                    ).inc()
                    raise

                finally:
                    duration = time.time() - start_time
                    self._metrics["cache_duration"].labels(
                        cache_type=cache_type, operation=operation
                    ).observe(duration)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    self._metrics["cache_operations"].labels(
                        cache_type=cache_type, operation=operation, result="success"
                    ).inc()
                    return result

                except Exception:
                    self._metrics["cache_operations"].labels(
                        cache_type=cache_type, operation=operation, result="error"
                    ).inc()
                    raise

                finally:
                    duration = time.time() - start_time
                    self._metrics["cache_duration"].labels(
                        cache_type=cache_type, operation=operation
                    ).observe(duration)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def record_cache_hit(self, cache_layer: str, cache_type: str) -> None:
        """Record a cache hit.

        Args:
            cache_layer: Cache layer (local, distributed)
            cache_type: Type of cache data (embeddings, crawl, search, etc.)

        """
        self._metrics["cache_hits"].labels(
            cache_layer=cache_layer, cache_type=cache_type
        ).inc()

    def record_cache_miss(self, cache_type: str) -> None:
        """Record a cache miss.

        Args:
            cache_type: Type of cache data (embeddings, crawl, search, etc.)

        """
        self._metrics["cache_misses"].labels(cache_type=cache_type).inc()

    def update_cache_memory_usage(
        self, cache_type: str, cache_name: str, memory_bytes: int
    ) -> None:
        """Update cache memory usage.

        Args:
            cache_type: Type of cache (local, distributed)
            cache_name: Name of the cache instance
            memory_bytes: Memory usage in bytes

        """
        self._metrics["cache_memory_usage"].labels(
            cache_type=cache_type, cache_name=cache_name
        ).set(memory_bytes)

    def update_cache_stats(self, cache_manager) -> None:
        """Update cache statistics from cache manager.

        Args:
            cache_manager: CacheManager instance to collect stats from

        """
        try:
            if hasattr(cache_manager, "local_cache") and cache_manager.local_cache:
                local_stats = cache_manager.local_cache.get_stats()
                self.update_cache_memory_usage(
                    "local", "main", int(local_stats.get("memory_bytes", 0))
                )

            if (
                hasattr(cache_manager, "distributed_cache")
                and cache_manager.distributed_cache
            ):
                # Memory stats for distributed cache would need to be collected differently
                # This is a placeholder for future implementation
                pass

        except Exception as e:
            # Log error but don't raise to avoid breaking monitoring

            logging.getLogger(__name__).warning(f"Failed to update cache stats: {e}")  # TODO: Convert f-string to logging format

    def record_embedding_cost(self, provider: str, model: str, cost: float) -> None:
        """Record embedding generation cost.

        Args:
            provider: Embedding provider
            model: Model name
            cost: Cost in USD

        """
        self._metrics["embedding_cost"].labels(provider=provider, model=model).inc(cost)

    def update_queue_depth(self, provider: str, depth: int) -> None:
        """Update embedding queue depth.

        Args:
            provider: Embedding provider
            depth: Current queue depth

        """
        self._metrics["embedding_queue_depth"].labels(provider=provider).set(depth)

    def update_service_health(self, service: str, healthy: bool) -> None:
        """Update service health status.

        Args:
            service: Service name
            healthy: Whether service is healthy

        """
        self._metrics["service_health"].labels(service=service).set(1 if healthy else 0)

    def update_dependency_health(self, dependency: str, healthy: bool) -> None:
        """Update dependency health status.

        Args:
            dependency: Dependency name (qdrant, redis, etc.)
            healthy: Whether dependency is healthy

        """
        self._metrics["dependency_health"].labels(dependency=dependency).set(
            1 if healthy else 0
        )

    def update_qdrant_metrics(
        self, collection: str, size: int, memory_usage: int
    ) -> None:
        """Update Qdrant collection metrics.

        Args:
            collection: Collection name
            size: Number of vectors
            memory_usage: Memory usage in bytes

        """
        self._metrics["qdrant_collection_size"].labels(collection=collection).set(size)
        self._metrics["qdrant_memory_usage"].labels(collection=collection).set(
            memory_usage
        )

    def record_qdrant_operation(
        self, operation: str, collection: str, success: bool
    ) -> None:
        """Record Qdrant operation.

        Args:
            operation: Operation type (insert, search, delete, etc.)
            collection: Collection name
            success: Whether operation succeeded

        """
        status = "success" if success else "error"
        self._metrics["qdrant_operations"].labels(
            operation=operation, collection=collection, status=status
        ).inc()

    def record_task_queue_size(self, queue: str, status: str, size: int) -> None:
        """Record task queue size.

        Args:
            queue: Queue name
            status: Task status (pending, running, complete, failed)
            size: Number of tasks

        """
        self._metrics["task_queue_size"].labels(queue=queue, status=status).set(size)

    def record_task_execution(
        self, task_name: str, duration_seconds: float, success: bool
    ) -> None:
        """Record task execution metrics.

        Args:
            task_name: Name of the task
            duration_seconds: Execution duration
            success: Whether task succeeded

        """
        status = "success" if success else "error"
        self._metrics["task_execution_duration"].labels(
            task_name=task_name, status=status
        ).observe(duration_seconds)
        self._metrics["task_requests"].labels(task_name=task_name, status=status).inc()

    def update_worker_count(self, queue: str, count: int) -> None:
        """Update active worker count.

        Args:
            queue: Queue name
            count: Number of active workers

        """
        self._metrics["worker_active"].labels(queue=queue).set(count)

    def record_browser_request(
        self, tier: str, duration_seconds: float, success: bool
    ) -> None:
        """Record browser automation request metrics.

        Args:
            tier: Browser tier name
            duration_seconds: Request duration
            success: Whether request succeeded

        """
        status = "success" if success else "error"
        self._metrics["browser_requests"].labels(tier=tier, status=status).inc()
        self._metrics["browser_response_time"].labels(tier=tier).observe(
            duration_seconds
        )

    def update_browser_tier_health(self, tier: str, healthy: bool) -> None:
        """Update browser tier health status.

        Args:
            tier: Browser tier name
            healthy: Whether tier is healthy

        """
        self._metrics["browser_tier_health"].labels(tier=tier).set(1 if healthy else 0)

    def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if self.config.enabled:
            start_http_server(self.config.export_port, registry=self.registry)

    def get_metric(self, name: str) -> Any | None:
        """Get metric by name.

        Args:
            name: Metric name

        Returns:
            Prometheus metric object or None if not found

        """
        return self._metrics.get(name)


# Global metrics registry instance
_global_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get global metrics registry instance.

    Returns:
        Global MetricsRegistry instance

    Raises:
        RuntimeError: If registry not initialized

    """
    if _global_registry is None:
        msg = "Metrics registry not initialized. Call initialize_metrics() first."
        raise RuntimeError(msg)
    return _global_registry


def initialize_metrics(config: MetricsConfig) -> MetricsRegistry:
    """Initialize global metrics registry.

    Args:
        config: Metrics configuration

    Returns:
        Initialized MetricsRegistry instance

    """
    global _global_registry
    _global_registry = MetricsRegistry(config)
    return _global_registry