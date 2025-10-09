"""Async-specific fixtures for testing asynchronous code.

This module provides fixtures specifically designed for testing
async functions and managing async resources properly.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport


@pytest.fixture
def async_test_client():
    """Factory returning an async HTTP client with lifespan management."""

    @asynccontextmanager
    async def _create_client(
        app: FastAPI,
        base_url: str = "http://test",
        **client_kwargs: Any,
    ):
        client_options = {
            "timeout": httpx.Timeout(30.0),
            "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10),
        }
        client_options.update(client_kwargs)

        @asynccontextmanager
        async def _lifespan_context() -> AsyncIterator[None]:
            async with LifespanManager(app):
                yield

        async with _lifespan_context():
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=base_url,
                **client_options,
            ) as client:
                client.headers.update(
                    {"User-Agent": "TestClient/1.0", "X-Test-Run": "true"}
                )
                yield client

    return _create_client


@pytest_asyncio.fixture
async def async_connection_pool():
    """Async connection pool for database testing."""
    pool = MagicMock()

    @asynccontextmanager
    async def acquire_connection() -> AsyncGenerator[MagicMock, None]:
        conn = MagicMock()
        conn.execute = AsyncMock(return_value=MagicMock(rows=[]))
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchone = AsyncMock(return_value=None)
        conn.close = AsyncMock()
        try:
            yield conn
        finally:
            await conn.close()

    pool.acquire = acquire_connection
    pool.close = AsyncMock()

    yield pool

    await pool.close()


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
            return await asyncio.gather(*coros, return_exceptions=True)

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

    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=100)

    class AsyncQueueTools:
        def __init__(self, queue: asyncio.Queue[Any]):
            self.queue = queue

        async def drain(self) -> list[Any]:
            return await async_drain_queue(self.queue)

        async def fill(self, values: list[Any]) -> None:
            await async_fill_queue(self.queue, values)

    tools = AsyncQueueTools(queue)

    yield tools

    # Ensure queue is empty
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def async_drain_queue(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain all items from queue."""

    items = []
    while not queue.empty():
        try:
            item = queue.get_nowait()
            items.append(item)
        except asyncio.QueueEmpty:
            break
    return items


async def async_fill_queue(queue: asyncio.Queue[Any], items: list[Any]) -> None:
    """Fill queue with items."""

    for item in items:
        await queue.put(item)
