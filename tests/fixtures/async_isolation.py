"""Async test isolation and resource management fixtures.

This module provides fixtures for proper isolation of async tests,
including event loop management, async resource cleanup, and
concurrent execution patterns.
"""

import asyncio
import time
import weakref
from contextlib import asynccontextmanager, suppress
from typing import Any
from unittest.mock import AsyncMock

import pytest_asyncio


try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]


class AsyncResourceManager:
    """Manages async resources with automatic cleanup."""

    def __init__(self):
        self._resources: set[Any] = set()
        self._cleanup_callbacks: list[asyncio.Task] = []

    async def register_resource(self, resource: Any, cleanup_callback=None):
        """Register a resource for cleanup."""
        self._resources.add(resource)

        if cleanup_callback:
            # Use weak reference to avoid circular dependencies
            weak_resource = weakref.ref(resource)

            async def cleanup():
                if weak_resource() is not None:
                    await cleanup_callback(weak_resource())

            self._cleanup_callbacks.append(cleanup)

    async def cleanup_all(self):
        """Clean up all registered resources."""
        # Run cleanup callbacks
        if self._cleanup_callbacks:
            await asyncio.gather(
                *[cb() for cb in self._cleanup_callbacks], return_exceptions=True
            )

        # Close any resources with close methods
        for resource in self._resources:
            if hasattr(resource, "aclose"):
                with suppress(Exception):
                    await resource.aclose()
            elif hasattr(resource, "close"):
                close_method = resource.close
                if asyncio.iscoroutinefunction(close_method):
                    with suppress(Exception):
                        await close_method()
                else:
                    with suppress(Exception):
                        close_method()

        self._resources.clear()
        self._cleanup_callbacks.clear()


@pytest_asyncio.fixture
async def async_resource_manager():
    """Provide async resource manager with automatic cleanup."""
    manager = AsyncResourceManager()

    try:
        yield manager
    finally:
        await manager.cleanup_all()


@pytest_asyncio.fixture
async def isolated_event_loop():
    """Isolated event loop for test isolation."""
    # Create a  event loop for this test
    loop = asyncio._event_loop()
    asyncio.set_event_loop(loop)

    try:
        yield loop
    finally:
        # Cancel all pending tasks
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]

        if pending_tasks:
            for task in pending_tasks:
                task.cancel()

            # Wait for cancellation to complete
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Close the loop
        loop.close()


@pytest_asyncio.fixture
async def async_test_context():
    """Comprehensive async test context with resource management."""
    context = {"tasks": [], "queues": [], "locks": [], "events": [], "semaphores": []}

    # Helper functions
    async def create_task(coro):
        task = asyncio.create_task(coro)
        context["tasks"].append(task)
        return task

    def create_queue(maxsize=0):
        queue = asyncio.Queue(maxsize=maxsize)
        context["queues"].append(queue)
        return queue

    def create_lock():
        lock = asyncio.Lock()
        context["locks"].append(lock)
        return lock

    def create_event():
        event = asyncio.Event()
        context["events"].append(event)
        return event

    def create_semaphore(value=1):
        semaphore = asyncio.Semaphore(value)
        context["semaphores"].append(semaphore)
        return semaphore

    # Add helper methods to context
    context.update(
        {
            "create_task": create_task,
            "create_queue": create_queue,
            "create_lock": create_lock,
            "create_event": create_event,
            "create_semaphore": create_semaphore,
        }
    )

    try:
        yield context
    finally:
        # Cleanup tasks
        for task in context["tasks"]:
            if not task.done():
                task.cancel()

        if context["tasks"]:
            await asyncio.gather(*context["tasks"], return_exceptions=True)

        # Clear queues
        for queue in context["queues"]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break


@pytest_asyncio.fixture
async def async_timeout_manager():
    """Manage timeouts for async operations in tests."""

    class TimeoutManager:
        def __init__(self):
            self.default_timeout = 30.0

        @asynccontextmanager
        async def timeout(self, seconds: float | None = None):
            """Context manager for timeout operations."""
            timeout_value = seconds or self.default_timeout

            try:
                async with asyncio.timeout(timeout_value):
                    yield
            except TimeoutError:
                timeout_message = f"Operation timed out after {timeout_value}s"
                raise AssertionError(timeout_message)

        async def wait_for(self, coro, timeout_seconds: float | None = None) -> Any:
            """Wait for a coroutine with timeout."""
            timeout_value = timeout_seconds or self.default_timeout

            try:
                return await asyncio.wait_for(coro, timeout=timeout_value)
            except TimeoutError:
                timeout_message = f"Operation timed out after {timeout_value}s"
                raise AssertionError(timeout_message)

    return TimeoutManager()


@pytest_asyncio.fixture
async def async_mock_manager():
    """Manage async mocks with proper cleanup."""
    mocks = []

    def create_async_mock(*args, **kwargs):
        mock = AsyncMock(*args, **kwargs)
        mocks.append(mock)
        return mock

    def create_mock_coroutine(return_value=None, side_effect=None):
        async def mock_coro(*args, **kwargs):
            if side_effect:
                if callable(side_effect):
                    result = side_effect(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                raise side_effect
            return return_value

        return create_async_mock(side_effect=mock_coro)

    try:
        yield {
            "create_async_mock": create_async_mock,
            "create_mock_coroutine": create_mock_coroutine,
            "mocks": mocks,
        }
    finally:
        # Reset all mocks
        for mock in mocks:
            mock.reset_mock()


@pytest_asyncio.fixture
async def async_performance_profiler():
    """Async performance profiling for tests."""

    class AsyncProfiler:
        def __init__(self):
            self.profiles = {}

        @asynccontextmanager
        async def profile(self, name: str):
            """Profile an async operation."""
            start_time = time.perf_counter()
            start_memory = 0

            if psutil is not None:
                # Try to get memory info if psutil available
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024

            try:
                yield
            finally:
                end_time = time.perf_counter()
                end_memory = start_memory

                if psutil is not None:
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024 / 1024

                self.profiles[name] = {
                    "duration_seconds": end_time - start_time,
                    "memory_start_mb": start_memory,
                    "memory_end_mb": end_memory,
                    "memory_delta_mb": end_memory - start_memory,
                }

        def get_profile(self, name: str):
            """Get profile results."""
            return self.profiles.get(name)

        def assert_performance(
            self,
            name: str,
            max_duration: float | None = None,
            max_memory_mb: float | None = None,
        ) -> None:
            """Assert performance constraints."""
            profile = self.profiles.get(name)
            if not profile:
                error_message = f"No profile found for '{name}'"
                raise AssertionError(error_message)

            if max_duration and profile["duration_seconds"] > max_duration:
                duration_message = (
                    f"Profile '{name}' exceeded max duration: "
                    f"{profile['duration_seconds']:.3f}s > {max_duration}s"
                )
                raise AssertionError(duration_message)

            if max_memory_mb and profile["memory_delta_mb"] > max_memory_mb:
                memory_message = (
                    f"Profile '{name}' exceeded max memory: "
                    f"{profile['memory_delta_mb']:.2f}MB > {max_memory_mb}MB"
                )
                raise AssertionError(memory_message)

    return AsyncProfiler()
