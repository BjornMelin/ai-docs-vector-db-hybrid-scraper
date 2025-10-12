"""Fixtures for Redis-compatible testing utilities."""

from collections.abc import AsyncIterator

import pytest_asyncio
from fakeredis import FakeServer, aioredis as fakeredis_aioredis

from src.services.cache.dragonfly_cache import DragonflyCache


@pytest_asyncio.fixture
async def fakeredis_server() -> AsyncIterator[FakeServer]:
    """Provide a shared FakeRedis server for deterministic state."""

    server = FakeServer()
    try:
        yield server
    finally:
        # FakeServer does not expose close semantics; rely on GC
        pass


@pytest_asyncio.fixture
async def fakeredis_cache(
    fakeredis_server: FakeServer,
) -> AsyncIterator[DragonflyCache]:
    """Yield a DragonflyCache backed by fakeredis for integration testing."""

    client = fakeredis_aioredis.FakeRedis(
        server=fakeredis_server,
        decode_responses=False,
    )
    cache = DragonflyCache(
        redis_url="redis://fakeredis",
        key_prefix="aidocs:",
        client=client,
    )
    try:
        yield cache
    finally:
        await cache.close()
        await client.aclose(close_connection_pool=True)
