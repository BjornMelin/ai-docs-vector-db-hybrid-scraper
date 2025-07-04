import typing


"""Advanced caching patterns for high-performance data access."""

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from .dragonfly_cache import DragonflyCache


logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service is recovered


class CircuitBreakerPattern:
    """Circuit breaker pattern for fault tolerance in agentic systems.

    Prevents cascading failures by monitoring service health and temporarily
    stopping requests to failing services. Designed for agent coordination
    where service failures can impact multi-agent workflows.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
        success_threshold: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to monitor for failures
            success_threshold: Consecutive successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold

        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0

        # Monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0

    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info(
                    "Circuit breaker entering half-open state for recovery test"
                )
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return False
            return True
        return False

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute (can be async)
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RuntimeError: If circuit is open
            Exception: If function fails (when circuit allows it)
        """
        self.total_requests += 1

        # Check if circuit is open
        if self.is_open():
            self.total_failures += 1
            raise RuntimeError(
                f"Circuit breaker is OPEN. Service unavailable. "
                f"Failure count: {self.failure_count}/{self.failure_threshold}, "
                f"Next retry in: {self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
            )

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - update circuit state
            await self._on_success()
            return result

        except self.expected_exception as e:
            # Expected failure - update circuit state
            await self._on_failure()
            raise e

    async def _on_success(self) -> None:
        """Handle successful execution."""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"Circuit breaker half-open success: {self.success_count}/{self.success_threshold}"
            )

            if self.success_count >= self.success_threshold:
                logger.info("Circuit breaker recovered, returning to CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on successful execution
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
                logger.debug(
                    f"Circuit breaker partial recovery: {self.failure_count}/{self.failure_threshold} failures"
                )

    async def _on_failure(self) -> None:
        """Handle failed execution."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(
            f"Circuit breaker failure: {self.failure_count}/{self.failure_threshold}"
        )

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker test failed, returning to OPEN state")
            self.state = CircuitState.OPEN
            self.success_count = 0

        elif self.failure_count >= self.failure_threshold:
            logger.error(f"Circuit breaker OPENED due to {self.failure_count} failures")
            self.state = CircuitState.OPEN

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker metrics
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": self.total_failures / max(self.total_requests, 1),
            "success_rate": self.total_successes / max(self.total_requests, 1),
            "last_failure_time": self.last_failure_time,
            "next_retry_in": max(
                0, self.recovery_timeout - (time.time() - self.last_failure_time)
            )
            if self.state == CircuitState.OPEN
            else 0,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        logger.info("Circuit breaker manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0


class CachePatterns:
    """Advanced caching patterns optimized for DragonflyDB performance."""

    def __init__(self, cache: DragonflyCache, task_queue_manager):
        """Initialize with DragonflyDB cache instance.

        Args:
            cache: DragonflyDB cache instance
            task_queue_manager: Required task queue manager for background tasks
        """
        self.cache = cache
        self._task_queue_manager = task_queue_manager
        self._background_tasks: set = set()

    def _create_background_task(self, coro):
        """Create a background task and manage its lifecycle."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def cache_aside(
        self,
        key: str,
        fetch_func: Callable,
        ttl: int = 3600,
        stale_while_revalidate: int = 60,
    ) -> Any:
        """Cache-aside pattern with stale-while-revalidate for optimal performance.

        This pattern provides excellent performance by:
        1. Serving stale data immediately when available
        2. Refreshing cache in background to minimize latency
        3. Using distributed locking to prevent cache stampedes

        Args:
            key: Cache key
            fetch_func: Function to fetch fresh data (can be async)
            ttl: Time to live in seconds
            stale_while_revalidate: Seconds before TTL to refresh in background

        Returns:
            Cached or fresh data
        """
        # Try to get from cache first
        cached = await self.cache.get(key)

        if cached is not None:
            # Check remaining TTL to determine if stale
            ttl_remaining = await self.cache.ttl(key)

            if ttl_remaining < stale_while_revalidate:
                # Return stale data immediately and refresh in background
                logger.debug(f"Serving stale data for {key}, refreshing in background")
                # Fire and forget - refresh happens in background
                self._create_background_task(self._refresh_cache(key, fetch_func, ttl))

            return cached

        # Cache miss - fetch and cache
        logger.debug(f"Cache miss for {key}, fetching fresh data")
        return await self._refresh_cache(key, fetch_func, ttl)

    async def _refresh_cache(self, key: str, fetch_func: Callable, ttl: int) -> Any:
        """Refresh cache with distributed lock to prevent stampedes."""
        lock_key = f"lock:{key}"

        # Try to acquire lock with short TTL
        lock_acquired = await self.cache.set(lock_key, "1", ttl=10, nx=True)

        if lock_acquired:
            try:
                # We have the lock - fetch fresh data
                logger.debug(f"Acquired lock for {key}, fetching data")

                if asyncio.iscoroutinefunction(fetch_func):
                    data = await fetch_func()
                else:
                    data = fetch_func()

                # Cache the fresh data
                await self.cache.set(key, data, ttl=ttl)
                logger.debug(f"Cached fresh data for {key}")

                return data

            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.error(f"Error fetching data for {key}: {e}")
                # If we fail, try to return stale data if available
                cached = await self.cache.get(key)
                if cached is not None:
                    return cached
                raise

            finally:
                # Always release the lock
                await self.cache.delete(lock_key)

        else:
            # Lock not acquired - another process is refreshing
            # Wait briefly for them to finish
            logger.debug(f"Lock not acquired for {key}, waiting for refresh")

            for _ in range(10):  # Wait up to 5 seconds
                await asyncio.sleep(0.5)
                cached = await self.cache.get(key)
                if cached is not None:
                    return cached

            # Fallback - fetch data ourselves if still not available
            logger.warning(f"Timeout waiting for refresh of {key}, fetching directly")
            if asyncio.iscoroutinefunction(fetch_func):
                return await fetch_func()
            return fetch_func()

    async def batch_cache(
        self,
        keys: list[str],
        fetch_func: Callable[[list[str]], dict[str, Any]],
        ttl: int = 3600,
    ) -> dict[str, Any]:
        """Efficient batch caching pattern.

        Optimizes for DragonflyDB's superior batch performance by:
        1. Using MGET for efficient batch retrieval
        2. Only fetching missing data
        3. Using MSET for efficient batch storage

        Args:
            keys: List of cache keys
            fetch_func: Function that takes missing keys and returns dict of data
            ttl: Time to live in seconds

        Returns:
            Dictionary mapping keys to values
        """
        logger.debug(f"Batch cache request for {len(keys)} keys")

        # Get existing values using DragonflyDB's optimized MGET
        cached_values = await self.cache.mget(keys)

        # Separate cached from missing
        results = {}
        missing_keys = []

        for key, value in zip(keys, cached_values, strict=False):
            if value is not None:
                results[key] = value
            else:
                missing_keys.append(key)

        logger.debug(f"Found {len(results)} cached, {len(missing_keys)} missing")

        # Fetch missing values if any
        if missing_keys:
            try:
                if asyncio.iscoroutinefunction(fetch_func):
                    fresh_data = await fetch_func(missing_keys)
                else:
                    fresh_data = fetch_func(missing_keys)

                # Cache fresh data using DragonflyDB's optimized MSET
                if fresh_data:
                    await self.cache.mset(fresh_data, ttl=ttl)
                    results.update(fresh_data)
                    logger.debug(f"Cached {len(fresh_data)} fresh items")

            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.error(f"Error fetching batch data: {e}")
                # Continue with partial results

        return results

    async def cached_computation(
        self,
        func: Callable,
        *args,
        cache_key: str | None = None,
        ttl: int = 3600,
        **kwargs,
    ) -> Any:
        """Cache expensive computation results with automatic key generation.

        Args:
            func: Function to cache (can be async)
            *args: Function arguments
            cache_key: Optional custom cache key
            ttl: Time to live in seconds
            **kwargs: Function keyword arguments

        Returns:
            Computation result (cached or fresh)
        """
        # Generate cache key from function and arguments
        if not cache_key:
            key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = f"compute:{hashlib.sha256(key_data.encode()).hexdigest()}"

        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for computation {func.__name__}")
            return cached

        # Compute and cache
        logger.debug(f"Cache miss for computation {func.__name__}, computing")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache the result
            await self.cache.set(cache_key, result, ttl=ttl)
            return result

        except (ConnectionError, OSError, PermissionError) as e:
            logger.error(f"Error in cached computation {func.__name__}: {e}")
            raise

    async def write_through(
        self,
        key: str,
        value: Any,
        persist_func: Callable[[str, Any], Any],
        ttl: int = 3600,
    ) -> bool:
        """Write-through caching pattern.

        Writes to both cache and persistent storage atomically.

        Args:
            key: Cache key
            value: Value to store
            persist_func: Function to persist to database (can be async)
            ttl: Cache TTL in seconds

        Returns:
            Success status
        """
        try:
            # Write to persistent storage first
            if asyncio.iscoroutinefunction(persist_func):
                await persist_func(key, value)
            else:
                persist_func(key, value)

            # Then write to cache
            success = await self.cache.set(key, value, ttl=ttl)

            if success:
                logger.debug(f"Write-through success for {key}")
            else:
                logger.warning(f"Write-through cache failed for {key}")

            return success

        except (ConnectionError, OSError, PermissionError) as e:
            logger.error(f"Write-through error for {key}: {e}")
            return False

    async def write_behind(
        self,
        key: str,
        value: Any,
        persist_func: Callable[[str, Any], Any],
        ttl: int = 3600,
        delay: float = 1.0,
    ) -> bool:
        """Write-behind (write-back) caching pattern.

        Writes to cache immediately and persists asynchronously using task queue.

        Args:
            key: Cache key
            value: Value to store
            persist_func: Function to persist to database (can be async)
            ttl: Cache TTL in seconds
            delay: Delay before persisting in seconds

        Returns:
            Cache write success status
        """
        # Write to cache immediately
        success = await self.cache.set(key, value, ttl=ttl)

        if success:
            # Task queue is required for write-behind pattern
            if not self._task_queue_manager:
                raise RuntimeError(
                    "TaskQueueManager is required for write-behind caching. "
                    "Initialize CachePatterns with a TaskQueueManager instance."
                )

            # Get module and function name for the persist function
            persist_module = persist_func.__module__
            persist_name = persist_func.__name__

            job_id = await self._task_queue_manager.enqueue(
                "persist_cache",
                key=key,
                value=value,
                persist_func_module=persist_module,
                persist_func_name=persist_name,
                delay=delay,
                _delay=int(delay),  # ARQ expects integer seconds
            )

            if job_id:
                logger.debug(f"Write-behind queued for {key} with job ID: {job_id}")
            else:
                raise RuntimeError(
                    f"Failed to queue write-behind persistence for {key}"
                )

        return success

    async def cache_warming(
        self,
        keys_and_data: dict[str, Any],
        ttl: int = 3600,
        batch_size: int = 100,
    ) -> int:
        """Warm cache with precomputed data in batches.

        Uses DragonflyDB's superior batch performance for efficient warming.

        Args:
            keys_and_data: Dictionary of keys to data
            ttl: Time to live in seconds
            batch_size: Number of items per batch

        Returns:
            Number of items successfully cached
        """
        total_cached = 0
        items = list(keys_and_data.items())

        # Process in batches for optimal performance
        for i in range(0, len(items), batch_size):
            batch = dict(items[i : i + batch_size])

            try:
                success = await self.cache.mset(batch, ttl=ttl)
                if success:
                    total_cached += len(batch)
                    logger.debug(
                        f"Cache warming batch {i // batch_size + 1}: {len(batch)} items"
                    )
                else:
                    logger.warning(f"Cache warming batch {i // batch_size + 1} failed")

            except (ConnectionError, RuntimeError, TimeoutError) as e:
                logger.error(f"Cache warming batch error: {e}")

        logger.info(
            f"Cache warming completed: {total_cached}/{len(items)} items cached"
        )
        return total_cached

    async def refresh_ahead(
        self,
        key: str,
        fetch_func: Callable,
        ttl: int = 3600,
        refresh_threshold: float = 0.8,
    ) -> Any:
        """Refresh-ahead caching pattern.

        Proactively refreshes cache before expiration.

        Args:
            key: Cache key
            fetch_func: Function to fetch fresh data
            ttl: Time to live in seconds
            refresh_threshold: Fraction of TTL after which to refresh (0.8 = 80%)

        Returns:
            Cached or fresh data
        """
        # Get data and remaining TTL
        cached = await self.cache.get(key)
        ttl_remaining = await self.cache.ttl(key)

        if cached is not None:
            # Check if we should refresh proactively
            refresh_point = ttl * refresh_threshold

            if ttl_remaining < (ttl - refresh_point):
                # Start background refresh
                logger.debug(f"Refresh-ahead triggered for {key}")
                # Fire and forget - refresh happens in background
                self._create_background_task(self._refresh_cache(key, fetch_func, ttl))

            return cached

        # Cache miss - fetch immediately
        return await self._refresh_cache(key, fetch_func, ttl)
