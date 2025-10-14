"""Async test isolation and resource management fixtures.

This module provides fixtures for proper isolation of async tests,
including event loop management, async resource cleanup, and
concurrent execution patterns.
"""

import asyncio
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import Any, TypeVar
from unittest.mock import AsyncMock

import pytest_asyncio


T = TypeVar("T")


try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]


class AsyncResourceManager:
    """Manages async resources with automatic cleanup."""

    def __init__(self):
        """Initialize the async resource manager."""
        self._resources: set[Any] = set()
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []

    async def register_resource(
        self,
        resource: Any,
        cleanup_callback: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        """Register a resource for cleanup."""
        self._resources.add(resource)

        if cleanup_callback:
            # Use weak reference to avoid circular dependencies
            weak_resource = weakref.ref(resource)

            async def cleanup() -> None:
                resource_ref = weak_resource()
                if resource_ref is not None:
                    await cleanup_callback(resource_ref)

            self._cleanup_callbacks.append(cleanup)

    async def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        # Run cleanup callbacks
        if self._cleanup_callbacks:
            await asyncio.gather(
                *(callback() for callback in self._cleanup_callbacks),
                return_exceptions=True,
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
async def async_resource_manager() -> AsyncIterator[AsyncResourceManager]:
    """Provide async resource manager with automatic cleanup."""
    manager = AsyncResourceManager()

    try:
        yield manager
    finally:
        await manager.cleanup_all()


@pytest_asyncio.fixture
async def isolated_event_loop() -> AsyncIterator[asyncio.AbstractEventLoop]:
    """Isolated event loop for test isolation."""
    loop = asyncio.new_event_loop()
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
        asyncio.set_event_loop(None)


@pytest_asyncio.fixture
async def async_test_context() -> AsyncIterator[dict[str, Any]]:
    """Async test context with resource management."""
    tasks: list[asyncio.Task[Any]] = []
    queues: list[asyncio.Queue[Any]] = []
    locks: list[asyncio.Lock] = []
    events: list[asyncio.Event] = []
    semaphores: list[asyncio.Semaphore] = []

    context: dict[str, Any] = {
        "tasks": tasks,
        "queues": queues,
        "locks": locks,
        "events": events,
        "semaphores": semaphores,
    }

    # Helper functions
    async def create_task(coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    def create_queue(maxsize: int = 0) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        queues.append(queue)
        return queue

    def create_lock() -> asyncio.Lock:
        lock = asyncio.Lock()
        locks.append(lock)
        return lock

    def create_event() -> asyncio.Event:
        event = asyncio.Event()
        events.append(event)
        return event

    def create_semaphore(value: int = 1) -> asyncio.Semaphore:
        semaphore = asyncio.Semaphore(value)
        semaphores.append(semaphore)
        return semaphore

    # Add helper methods to context
    context["create_task"] = create_task
    context["create_queue"] = create_queue
    context["create_lock"] = create_lock
    context["create_event"] = create_event
    context["create_semaphore"] = create_semaphore

    try:
        yield context
    finally:
        # Cleanup tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Clear queues
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break


class TimeoutManager:
    """Manage timeouts for async operations in tests."""

    def __init__(self) -> None:
        """Initialize the timeout manager."""
        self.default_timeout: float = 30.0

    @asynccontextmanager
    async def timeout(self, seconds: float | None = None) -> AsyncIterator[None]:
        """Context manager for timeout operations."""
        timeout_value = seconds or self.default_timeout

        try:
            async with asyncio.timeout(timeout_value):
                yield
        except TimeoutError as exc:
            timeout_message = f"Operation timed out after {timeout_value}s"
            raise AssertionError(timeout_message) from exc

    async def wait_for(
        self, coro: Awaitable[T], timeout_seconds: float | None = None
    ) -> T:
        """Wait for a coroutine with timeout."""
        timeout_value = timeout_seconds or self.default_timeout

        try:
            return await asyncio.wait_for(coro, timeout=timeout_value)
        except TimeoutError as exc:
            timeout_message = f"Operation timed out after {timeout_value}s"
            raise AssertionError(timeout_message) from exc


class AsyncProfiler:
    """Async performance profiling helper for tests."""

    def __init__(self) -> None:
        """Initialize the async profiler."""
        self.profiles: dict[str, dict[str, float]] = {}

    @asynccontextmanager
    async def profile(self, name: str) -> AsyncIterator[None]:
        """Profile an async operation."""
        start_time = time.perf_counter()
        start_memory = 0.0

        if psutil is not None:
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

    def get_profile(self, name: str) -> dict[str, float] | None:
        """Get profile results if available."""
        return self.profiles.get(name)

    def assert_performance(
        self,
        name: str,
        max_duration: float | None = None,
        max_memory_mb: float | None = None,
    ) -> None:
        """Assert recorded metrics stay within provided thresholds."""
        profile = self.profiles.get(name)
        if not profile:
            error_message = f"No profile found for '{name}'"
            raise AssertionError(error_message)

        if max_duration is not None and profile["duration_seconds"] > max_duration:
            duration_message = (
                f"Profile '{name}' exceeded max duration: "
                f"{profile['duration_seconds']:.3f}s > {max_duration}s"
            )
            raise AssertionError(duration_message)

        if max_memory_mb is not None and profile["memory_delta_mb"] > max_memory_mb:
            memory_message = (
                f"Profile '{name}' exceeded max memory: "
                f"{profile['memory_delta_mb']:.2f}MB > {max_memory_mb}MB"
            )
            raise AssertionError(memory_message)


@pytest_asyncio.fixture
async def async_timeout_manager() -> TimeoutManager:
    """Provide timeout manager fixture."""
    return TimeoutManager()


@pytest_asyncio.fixture
async def async_mock_manager() -> AsyncIterator[dict[str, Any]]:
    """Manage async mocks with proper cleanup."""
    mocks: list[AsyncMock] = []

    def create_async_mock(*args: Any, **kwargs: Any) -> AsyncMock:
        mock = AsyncMock(*args, **kwargs)
        mocks.append(mock)
        return mock

    def create_mock_coroutine(
        return_value: Any = None,
        side_effect: Callable[..., Any] | BaseException | None = None,
    ) -> AsyncMock:
        async def mock_coro(*args: Any, **kwargs: Any) -> Any:
            if side_effect is not None:
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
async def async_performance_profiler() -> AsyncIterator[AsyncProfiler]:
    """Async performance profiling fixture."""
    profiler = AsyncProfiler()
    try:
        yield profiler
    finally:
        profiler.profiles.clear()
