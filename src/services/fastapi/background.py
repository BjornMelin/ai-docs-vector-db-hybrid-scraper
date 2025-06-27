"""Background task management for FastAPI production deployment.

This module provides comprehensive background task management including
task scheduling, monitoring, and lifecycle management for production environments.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import BackgroundTasks


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Background task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of a background task execution."""

    task_id: str
    status: TaskStatus
    start_time: datetime | None = None
    end_time: datetime | None = None
    result: Any = None
    error: str | None = None

    @property
    def duration(self) -> timedelta | None:
        """Get task execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )


@dataclass
class BackgroundTask:
    """Background task definition."""

    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Post-initialization validation."""
        if not callable(self.func):
            raise ValueError("Task function must be callable")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")


class BackgroundTaskManager:
    """Production-grade background task manager.

    Features:
    - Task prioritization and queuing
    - Retry logic with exponential backoff
    - Task monitoring and health checks
    - Resource management and limits
    - Graceful shutdown handling
    """

    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        """Initialize background task manager.

        Args:
            max_workers: Maximum number of concurrent workers
            max_queue_size: Maximum size of task queue
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size

        # Task storage
        self._tasks: dict[str, BackgroundTask] = {}
        self._results: dict[str, TaskResult] = {}
        self._task_lock = threading.Lock()

        # Queue management
        self._task_queue: asyncio.Queue = None
        self._workers: list[asyncio.Task] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # State management
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

    async def start(self) -> None:
        """Start the background task manager."""
        if self._running:
            return

        self._task_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._shutdown_event.clear()
        self._running = True

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

        logger.info(f"Background task manager started with {self.max_workers} workers")

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the background task manager gracefully.

        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        if not self._running:
            return

        logger.info("Stopping background task manager...")
        self._running = False
        self._shutdown_event.set()

        # Cancel all pending tasks in queue
        while not self._task_queue.empty():
            try:
                task_id = await asyncio.wait_for(self._task_queue.get(), timeout=0.1)
                await self._cancel_task(task_id)
            except TimeoutError:
                break

        # Wait for workers to finish
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning("Timeout waiting for workers to finish, cancelling...")
                for worker in self._workers:
                    if not worker.done():
                        worker.cancel()

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        self._workers.clear()
        logger.info("Background task manager stopped")

    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs,
    ) -> str:
        """Submit a background task for execution.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            task_id: Optional task ID (generated if None)
            priority: Task priority
            max_retries: Maximum number of retry attempts
            timeout: Task execution timeout
            **kwargs: Keyword arguments for function

        Returns:
            Task ID string

        Raises:
            RuntimeError: If manager is not running
            ValueError: If queue is full
        """
        if not self._running:
            raise RuntimeError("Task manager is not running")

        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task-{int(time.time() * 1000)}-{id(func)}"

        # Create task
        task = BackgroundTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
        )

        # Store task
        with self._task_lock:
            self._tasks[task_id] = task
            self._results[task_id] = TaskResult(
                task_id=task_id, status=TaskStatus.PENDING
            )
            self._total_tasks += 1

        # Add to queue
        try:
            await self._task_queue.put(task_id)
            logger.debug(f"Task {task_id} submitted successfully")
            return task_id
        except asyncio.QueueFull as e:
            # Remove from storage if queue is full
            with self._task_lock:
                del self._tasks[task_id]
                del self._results[task_id]
                self._total_tasks -= 1
            raise ValueError("Task queue is full") from e

    async def get_task_result(self, task_id: str) -> TaskResult | None:
        """Get result of a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Task result or None if not found
        """
        with self._task_lock:
            return self._results.get(task_id)

    async def wait_for_task(
        self, task_id: str, timeout: float | None = None
    ) -> TaskResult:
        """Wait for a task to complete.

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait

        Returns:
            Task result

        Raises:
            asyncio.TimeoutError: If timeout is reached
            ValueError: If task not found
        """
        start_time = time.time()

        while True:
            result = await self.get_task_result(task_id)
            if result is None:
                raise ValueError(f"Task {task_id} not found")

            if result.is_complete:
                return result

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")

            await asyncio.sleep(0.1)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False if not found or already complete
        """
        return await self._cancel_task(task_id)

    async def _cancel_task(self, task_id: str) -> bool:
        """Internal task cancellation logic."""
        with self._task_lock:
            result = self._results.get(task_id)
            if result and not result.is_complete:
                result.status = TaskStatus.CANCELLED
                result.end_time = datetime.now(tz=UTC)
                logger.debug(f"Task {task_id} cancelled")
                return True
        return False

    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop for processing tasks.

        Args:
            worker_name: Name of the worker for logging
        """
        logger.debug(f"Worker {worker_name} started")

        while self._running:
            try:
                # Wait for task or shutdown
                task_id = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)

                await self._execute_task(task_id, worker_name)

            except TimeoutError:
                # Check if we should shutdown
                if self._shutdown_event.is_set():
                    break
                continue
            except Exception as e:
                logger.exception(f"Worker {worker_name} error: {e}")

        logger.debug(f"Worker {worker_name} stopped")

    async def _execute_task(self, task_id: str, worker_name: str) -> None:
        """Execute a single task.

        Args:
            task_id: Task identifier
            worker_name: Name of executing worker
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            result = self._results.get(task_id)

        if not task or not result:
            logger.warning(f"Task {task_id} not found for execution")
            return

        # Check if task was cancelled
        if result.status == TaskStatus.CANCELLED:
            return

        # Update task status
        result.status = TaskStatus.RUNNING
        result.start_time = datetime.now(tz=UTC)

        logger.debug(f"Worker {worker_name} executing task {task_id}")

        try:
            # Execute task with timeout
            if asyncio.iscoroutinefunction(task.func):
                # Async function
                if task.timeout:
                    task_result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs), timeout=task.timeout
                    )
                else:
                    task_result = await task.func(*task.args, **task.kwargs)
            # Sync function - run in thread pool
            elif task.timeout:
                task_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, lambda: task.func(*task.args, **task.kwargs)
                    ),
                    timeout=task.timeout,
                )
            else:
                task_result = await asyncio.get_event_loop().run_in_executor(
                    self._executor, lambda: task.func(*task.args, **task.kwargs)
                )

            # Task completed successfully
            result.status = TaskStatus.COMPLETED
            result.result = task_result
            result.end_time = datetime.now(tz=UTC)

            with self._task_lock:
                self._completed_tasks += 1

            logger.debug(f"Task {task_id} completed successfully")

        except TimeoutError:
            # Task timeout
            result.status = TaskStatus.FAILED
            result.error = f"Task timeout after {task.timeout} seconds"
            result.end_time = datetime.now(tz=UTC)

            await self._handle_task_retry(task, result)

        except Exception as e:
            # Task failed
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now(tz=UTC)

            logger.exception(f"Task {task_id} failed: {e}")
            await self._handle_task_retry(task, result)

    async def _handle_task_retry(
        self, task: BackgroundTask, result: TaskResult
    ) -> None:
        """Handle task retry logic.

        Args:
            task: Failed task
            result: Task result
        """
        if task.retry_count < task.max_retries:
            task.retry_count += 1

            # Reset result for retry
            result.status = TaskStatus.PENDING
            result.start_time = None
            result.end_time = None
            result.error = None

            # Calculate retry delay (exponential backoff)
            delay = task.retry_delay * (2 ** (task.retry_count - 1))

            logger.info(
                f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries}) after {delay}s"
            )

            # Schedule retry
            retry_task = asyncio.create_task(self._schedule_retry(task.task_id, delay))
            # Store reference to prevent task from being garbage collected
            self._retry_tasks = getattr(self, "_retry_tasks", set())
            self._retry_tasks.add(retry_task)
            retry_task.add_done_callback(self._retry_tasks.discard)
        else:
            # Max retries reached
            with self._task_lock:
                self._failed_tasks += 1
            logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")

    async def _schedule_retry(self, task_id: str, delay: float) -> None:
        """Schedule a task retry after delay.

        Args:
            task_id: Task identifier
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)

        if self._running:
            try:
                await self._task_queue.put(task_id)
            except asyncio.QueueFull:
                logger.exception(f"Failed to retry task {task_id}: queue is full")

    def get_statistics(self) -> dict[str, Any]:
        """Get task manager statistics.

        Returns:
            Statistics dictionary
        """
        with self._task_lock:
            queue_size = self._task_queue.qsize() if self._task_queue else 0

            return {
                "total_tasks": self._total_tasks,
                "completed_tasks": self._completed_tasks,
                "failed_tasks": self._failed_tasks,
                "pending_tasks": queue_size,
                "active_workers": len([w for w in self._workers if not w.done()]),
                "max_workers": self.max_workers,
                "queue_size": queue_size,
                "max_queue_size": self.max_queue_size,
                "running": self._running,
            }

    async def get_task_list(self, status: TaskStatus | None = None) -> list[dict]:
        """Get list of tasks with optional status filter.

        Args:
            status: Optional status filter

        Returns:
            List of task information
        """
        task_list = []

        with self._task_lock:
            for task_id, result in self._results.items():
                if status is None or result.status == status:
                    task = self._tasks.get(task_id)
                    task_info = {
                        "task_id": task_id,
                        "status": result.status.value,
                        "created_at": task.created_at.isoformat() if task else None,
                        "start_time": result.start_time.isoformat()
                        if result.start_time
                        else None,
                        "end_time": result.end_time.isoformat()
                        if result.end_time
                        else None,
                        "duration": result.duration.total_seconds()
                        if result.duration
                        else None,
                        "retry_count": task.retry_count if task else 0,
                        "error": result.error,
                    }
                    task_list.append(task_info)

        return task_list


# Global task manager instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get the global background task manager.

    Returns:
        Background task manager instance
    """
    global _task_manager  # noqa: PLW0603
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


async def initialize_task_manager(
    max_workers: int = 4, max_queue_size: int = 1000
) -> None:
    """Initialize the global background task manager.

    Args:
        max_workers: Maximum number of concurrent workers
        max_queue_size: Maximum size of task queue
    """
    global _task_manager  # noqa: PLW0603
    if _task_manager is None:
        _task_manager = BackgroundTaskManager(max_workers, max_queue_size)
    await _task_manager.start()


async def cleanup_task_manager() -> None:
    """Clean up the global background task manager."""
    global _task_manager  # noqa: PLW0603
    if _task_manager:
        await _task_manager.stop()
        _task_manager = None


# Convenience functions for FastAPI integration


def submit_background_task(
    background_tasks: BackgroundTasks, func: Callable, *args, **kwargs
) -> None:
    """Submit a task using FastAPI's BackgroundTasks.

    Args:
        background_tasks: FastAPI BackgroundTasks instance
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    background_tasks.add_task(func, *args, **kwargs)


async def submit_managed_task(
    func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs
) -> str:
    """Submit a task to the managed task queue.

    Args:
        func: Function to execute
        *args: Positional arguments
        priority: Task priority
        **kwargs: Keyword arguments

    Returns:
        Task ID
    """
    manager = get_task_manager()
    return await manager.submit_task(func, *args, priority=priority, **kwargs)


# Export all classes and functions
__all__ = [
    "BackgroundTask",
    "BackgroundTaskManager",
    "TaskPriority",
    "TaskResult",
    "TaskStatus",
    "cleanup_task_manager",
    "get_task_manager",
    "initialize_task_manager",
    "submit_background_task",
    "submit_managed_task",
]
