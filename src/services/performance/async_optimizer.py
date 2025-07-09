"""Asynchronous Processing Optimization Module.

This module provides optimizations for async operations including
concurrency control, task batching, and background processing.

The AsyncOptimizer class manages asynchronous operations efficiently by:
- Controlling concurrency with semaphores
- Batching operations for better throughput
- Managing background tasks with proper lifecycle
- Creating task queues with worker pools
- Providing parallel mapping capabilities

Key Features:
    - Rate-limited operations with named semaphores
    - Batch processing with configurable concurrency
    - Background task management and tracking
    - Task queue system with multiple workers
    - Parallel map operations with concurrency control

Example:
    >>> optimizer = AsyncOptimizer(max_concurrency=50)
    >>> # Rate-limited operation
    >>> result = await optimizer.rate_limited_operation(
    ...     fetch_data, "api_calls", url="https://api.example.com"
    ... )
    >>> # Batch processing
    >>> items = list(range(1000))
    >>> results = await optimizer.batch_process(
    ...     items, process_batch, batch_size=100
    ... )

Note:
    This module uses Python 3.11+ TaskGroup for efficient task management
    via the gather_with_taskgroup utility function.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from src.utils.async_utils import gather_with_taskgroup


logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncOptimizer:
    """Optimize asynchronous operations for better performance.

    The AsyncOptimizer provides a centralized system for managing
    asynchronous operations with controlled concurrency, efficient
    batching, and proper resource management. It helps prevent
    resource exhaustion and improves overall application throughput.

    The optimizer maintains internal state for:
    - Named semaphores for different operation types
    - Task queues with worker pools
    - Background task tracking

    Attributes:
        max_concurrency: Default maximum number of concurrent operations
        _semaphores: Dictionary of named semaphores for rate limiting
        _task_queues: Dictionary of task queues with workers
        _background_tasks: Set of tracked background tasks

    Example:
        >>> async def main():
        ...     optimizer = AsyncOptimizer(max_concurrency=50)
        ...
        ...     # Create a task queue
        ...     queue = await optimizer.create_task_queue(
        ...         "processor", max_workers=5
        ...     )
        ...
        ...     # Add tasks to queue
        ...     for item in items:
        ...         await queue.put({"func": process_item, "args": (item,)})
        ...
        ...     # Shutdown when done
        ...     await optimizer.shutdown()
    """

    def __init__(self, max_concurrency: int = 100):
        """Initialize async optimizer.

        Args:
            max_concurrency: Maximum concurrent operations allowed by default.
                Can be overridden per operation. Recommended values:
                - Light operations (cache reads): 100-200
                - API calls: 20-50
                - Heavy processing: 5-20

        Note:
            The optimizer starts with empty semaphore and queue dictionaries.
            Resources are created on-demand as operations are performed.
        """
        self.max_concurrency = max_concurrency
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._task_queues: dict[str, asyncio.Queue] = {}
        self._background_tasks: set[asyncio.Task] = set()

    def get_semaphore(self, name: str, limit: int | None = None) -> asyncio.Semaphore:
        """Get or create a named semaphore for rate limiting.

        Semaphores are created on-demand and cached for reuse. This allows
        different parts of the application to share rate limiting for the
        same resource type (e.g., "api_calls", "db_queries").

        Common semaphore names:
        - "api_calls": External API rate limiting
        - "db_queries": Database connection limiting
        - "file_io": File system operation limiting
        - "cpu_intensive": Heavy computation limiting

        Args:
            name: Semaphore name for identification and reuse
            limit: Concurrency limit for this semaphore. If None,
                uses the instance's max_concurrency value

        Returns:
            asyncio.Semaphore: Semaphore instance for the given name

        Example:
            >>> sem = optimizer.get_semaphore("api_calls", limit=10)
            >>> async with sem:
            ...     response = await make_api_call()
        """
        if name not in self._semaphores:
            limit = limit or self.max_concurrency
            self._semaphores[name] = asyncio.Semaphore(limit)

        return self._semaphores[name]

    async def rate_limited_operation(
        self, func: Callable, semaphore_name: str, *args, **kwargs
    ) -> Any:
        """Execute an operation with rate limiting.

        Args:
            func: Async function to execute
            semaphore_name: Name of semaphore to use
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        """
        semaphore = self.get_semaphore(semaphore_name)

        async with semaphore:
            return await func(*args, **kwargs)

    async def batch_process(
        self,
        items: list[T],
        process_func: Callable[[list[T]], Any],
        batch_size: int = 100,
        max_concurrent_batches: int = 10,
    ) -> list[Any]:
        """Process items in batches with concurrency control.

        Divides a large list of items into smaller batches and processes
        them concurrently. This is ideal for operations that benefit from
        batching (e.g., bulk database inserts, API calls with batch endpoints).

        The method ensures that no more than max_concurrent_batches are
        processed simultaneously, preventing resource exhaustion while
        maximizing throughput.

        Processing flow:
        1. Items are divided into batches of batch_size
        2. Batches are processed concurrently up to max_concurrent_batches
        3. Results are collected and flattened if they are lists
        4. Original order is preserved in the results

        Args:
            items: List of items to process in batches
            process_func: Async function that accepts a list of items
                and returns results. Should handle a batch of items.
            batch_size: Number of items per batch. Optimal size depends
                on the operation - DB inserts: 100-1000, API calls: 10-100
            max_concurrent_batches: Maximum number of batches to process
                concurrently. Adjust based on resource constraints.

        Returns:
            list[Any]: Flattened list of results from all batches.
                If process_func returns lists, they are flattened into
                a single list. Otherwise, results are collected as-is.

        Example:
            >>> async def bulk_insert(batch):
            ...     return await db.insert_many(batch)
            >>> items = [{"name": f"doc_{i}"} for i in range(1000)]
            >>> results = await optimizer.batch_process(
            ...     items, bulk_insert, batch_size=100, max_concurrent_batches=5
            ... )
        """
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batches.append(batch)

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def process_with_limit(batch):
            async with semaphore:
                return await process_func(batch)

        # Use TaskGroup for efficient processing
        results = await gather_with_taskgroup(
            *[process_with_limit(batch) for batch in batches]
        )

        # Flatten results if they are lists
        flattened = []
        for result in results:
            if isinstance(result, list):
                flattened.extend(result)
            else:
                flattened.append(result)

        return flattened

    def create_background_task(
        self, coro: Any, name: str | None = None
    ) -> asyncio.Task:
        """Create a tracked background task.

        Args:
            coro: Coroutine to run in background
            name: Optional task name

        Returns:
            Created task

        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)

        # Remove from set when complete
        task.add_done_callback(self._background_tasks.discard)

        return task

    async def create_task_queue(
        self, name: str, max_workers: int = 10, max_queue_size: int = 1000
    ) -> asyncio.Queue:
        """Create a task queue with workers.

        Args:
            name: Queue name
            max_workers: Number of worker tasks
            max_queue_size: Maximum queue size

        Returns:
            Created queue

        """
        if name in self._task_queues:
            return self._task_queues[name]

        queue = asyncio.Queue(maxsize=max_queue_size)
        self._task_queues[name] = queue

        # Start workers
        for i in range(max_workers):
            worker_name = f"{name}_worker_{i}"
            self.create_background_task(
                self._queue_worker(queue, worker_name), name=worker_name
            )

        return queue

    async def _queue_worker(self, queue: asyncio.Queue, name: str) -> None:
        """Worker for processing queue items."""
        logger.info(f"Starting queue worker: {name}")

        while True:
            try:
                # Get item from queue
                item = await queue.get()

                if item is None:  # Shutdown signal
                    break

                # Process item
                if callable(item):
                    await item()
                elif isinstance(item, dict) and "func" in item:
                    func = item["func"]
                    args = item.get("args", ())
                    kwargs = item.get("kwargs", {})
                    await func(*args, **kwargs)

            except Exception:
                logger.exception(f"Queue worker {name} error")

            finally:
                queue.task_done()

    async def parallel_map(
        self,
        func: Callable[[T], Any],
        items: list[T],
        max_concurrency: int | None = None,
    ) -> list[Any]:
        """Apply async function to items in parallel.

        Args:
            func: Async function to apply
            items: Items to process
            max_concurrency: Maximum concurrent operations

        Returns:
            List of results in same order as items

        """
        max_concurrency = max_concurrency or self.max_concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_with_limit(item):
            async with semaphore:
                return await func(item)

        return await gather_with_taskgroup(
            *[process_with_limit(item) for item in items]
        )

    async def shutdown(self) -> None:
        """Shutdown all background tasks and queues."""
        logger.info("Shutting down async optimizer")

        # Stop all queue workers
        for queue in self._task_queues.values():
            # Send shutdown signal to workers
            for _ in range(queue.maxsize):
                await queue.put(None)

        # Cancel remaining background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()
        self._task_queues.clear()
        self._semaphores.clear()

        logger.info("Async optimizer shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get async operation statistics.

        Returns:
            Statistics dictionary

        """
        return {
            "active_background_tasks": len(self._background_tasks),
            "task_queues": list(self._task_queues.keys()),
            "semaphores": {
                name: {
                    "limit": sem._value + sem._initial_value - sem._value,  # noqa: SLF001
                    "available": sem._value,  # noqa: SLF001
                }
                for name, sem in self._semaphores.items()
            },
        }
