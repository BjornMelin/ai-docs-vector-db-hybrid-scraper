"""Async-specific fixtures for testing asynchronous code.

This module provides fixtures specifically designed for testing
async functions and managing async resources properly.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest_asyncio


@pytest_asyncio.fixture
async def async_test_client():
    """Async HTTP test client with session management."""
    import httpx

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    ) as client:
        # Add test headers
        client.headers.update({"User-Agent": "TestClient/1.0", "X-Test-Run": "true"})
        yield client


@pytest_asyncio.fixture
async def async_connection_pool():
    """Async connection pool for database testing."""
    pool = MagicMock()

    # Connection methods
    async def acquire_connection():
        conn = MagicMock()
        conn.execute = AsyncMock(return_value=MagicMock(rows=[]))
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchone = AsyncMock(return_value=None)
        conn.close = AsyncMock()
        return conn

    pool.acquire = asynccontextmanager(acquire_connection)
    pool.close = AsyncMock()

    yield pool

    await pool.close()


@pytest_asyncio.fixture
async def async_rate_limiter():
    """Async rate limiter for testing rate-limited operations."""

    class AsyncRateLimiter:
        def __init__(self, rate: int = 10, period: float = 1.0):
            self.rate = rate
            self.period = period
            self.calls = []
            self._lock = asyncio.Lock()

        async def acquire(self):
            async with self._lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                # Remove old calls
                self.calls = [t for t in self.calls if now - t < self.period]

                if len(self.calls) >= self.rate:
                    sleep_time = self.period - (now - self.calls[0])
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()

                self.calls.append(now)

        async def __aenter__(self):
            await self.acquire()
            return self

        async def __aexit__(self, *args):
            pass

    return AsyncRateLimiter()


@pytest_asyncio.fixture
async def async_task_manager():
    """Manage async tasks with proper cleanup."""
    tasks = []

    class AsyncTaskManager:
        async def create_task(self, coro):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        async def gather(self, *coros):
            results = await asyncio.gather(*coros, return_exceptions=True)
            return results

        async def cancel_all(self):
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

    manager = AsyncTaskManager()
    yield manager

    # Cleanup
    await manager.cancel_all()


@pytest_asyncio.fixture
async def async_event_emitter():
    """Async event emitter for testing event-driven code."""

    class AsyncEventEmitter:
        def __init__(self):
            self.listeners = {}
            self.events = []

        def on(self, event: str, handler):
            if event not in self.listeners:
                self.listeners[event] = []
            self.listeners[event].append(handler)

        async def emit(self, event: str, data: Any = None):
            self.events.append({"event": event, "data": data})

            if event in self.listeners:
                for handler in self.listeners[event]:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)

        def clear(self):
            self.listeners.clear()
            self.events.clear()

    emitter = AsyncEventEmitter()
    yield emitter
    emitter.clear()


@pytest_asyncio.fixture
async def async_cache():
    """Async cache implementation for testing."""

    class AsyncCache:
        def __init__(self):
            self.data = {}
            self.hits = 0
            self.misses = 0

        async def get(self, key: str) -> Any:
            await asyncio.sleep(0)  # Simulate async operation
            if key in self.data:
                self.hits += 1
                return self.data[key]
            self.misses += 1
            return None

        async def set(self, key: str, value: Any, ttl: int = 3600):
            await asyncio.sleep(0)  # Simulate async operation
            self.data[key] = value

        async def delete(self, key: str):
            await asyncio.sleep(0)  # Simulate async operation
            self.data.pop(key, None)

        async def clear(self):
            self.data.clear()
            self.hits = 0
            self.misses = 0

        def stats(self):
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "size": len(self.data),
            }

    cache = AsyncCache()
    yield cache
    await cache.clear()


@pytest_asyncio.fixture
async def async_queue():
    """Async queue for testing producer-consumer patterns."""
    queue = asyncio.Queue(maxsize=100)

    # Add helper methods
    queue.drain = async_drain_queue
    queue.fill = async_fill_queue

    yield queue

    # Ensure queue is empty
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def async_drain_queue(queue):
    """Drain all items from queue."""
    items = []
    while not queue.empty():
        try:
            item = queue.get_nowait()
            items.append(item)
        except asyncio.QueueEmpty:
            break
    return items


async def async_fill_queue(queue, items):
    """Fill queue with items."""
    for item in items:
        await queue.put(item)
