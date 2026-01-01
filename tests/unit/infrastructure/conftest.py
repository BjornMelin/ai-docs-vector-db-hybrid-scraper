"""Fixtures for infrastructure unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def minimal_qdrant_config() -> SimpleNamespace:
    """Minimal Qdrant configuration for container testing."""
    return SimpleNamespace(
        url="http://localhost:6333",
        api_key="test-key",
        timeout=30,
        prefer_grpc=False,
        use_grpc=False,
        collection_name="test-collection",
        enable_grouping=False,
    )


@pytest.fixture
def minimal_cache_config() -> SimpleNamespace:
    """Minimal cache configuration for container testing."""
    return SimpleNamespace(
        enable_dragonfly_cache=True,
        host="localhost",
        port=6379,
        db=0,
        password=None,
        ssl=False,
        connection_pool_size=10,
        default_ttl=3600,
    )


@pytest.fixture
def minimal_firecrawl_config() -> SimpleNamespace:
    """Minimal Firecrawl configuration for container testing."""
    return SimpleNamespace(
        api_key="fc-test-key",
        api_url="https://api.firecrawl.dev",
        timeout_seconds=30,
        default_formats=["markdown"],
    )


@pytest.fixture
def minimal_config_namespace(
    minimal_qdrant_config: SimpleNamespace,
    minimal_cache_config: SimpleNamespace,
    minimal_firecrawl_config: SimpleNamespace,
) -> SimpleNamespace:
    """Complete minimal configuration namespace for container testing."""
    return SimpleNamespace(
        qdrant=minimal_qdrant_config,
        cache=minimal_cache_config,
        firecrawl=minimal_firecrawl_config,
        data_dir="/tmp/test-data",
    )


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """AsyncQdrantClient mock for container tests."""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
    client.create_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    return client


@pytest.fixture
def mock_dragonfly_client() -> MagicMock:
    """Redis/Dragonfly client mock for container tests."""
    client = MagicMock()
    client.ping = MagicMock(return_value=True)
    client.get = MagicMock(return_value=None)
    client.set = MagicMock(return_value=True)
    client.delete = MagicMock(return_value=1)
    return client


@pytest.fixture
def mock_firecrawl_client() -> MagicMock:
    """Firecrawl client mock for container tests."""
    client = MagicMock()
    client.scrape = AsyncMock(return_value={"data": {"markdown": "test"}})
    client.crawl = AsyncMock(return_value={"status": "completed"})
    return client


def create_mock_settings(**overrides: Any) -> SimpleNamespace:
    """Factory function to create mock settings with overrides."""
    defaults = {
        "qdrant": SimpleNamespace(
            url="http://localhost:6333",
            api_key="test-key",
            timeout=30,
            prefer_grpc=False,
            collection_name="documents",
            enable_grouping=False,
        ),
        "cache": SimpleNamespace(
            enable_dragonfly_cache=False,
            host="localhost",
            port=6379,
        ),
        "firecrawl": SimpleNamespace(
            api_key="fc-test",
            api_url=None,
        ),
        "data_dir": "/tmp/test",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
