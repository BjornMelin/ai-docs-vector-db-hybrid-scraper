# Optional Service Adapters

This guide records recommended patterns for wiring optional search or caching
services after the core application trimmed its service registry to the
embedding and vector components. The goal is to keep adapters outside the
library surface while providing ready-to-use recipes.

## In-Memory Dev Adapter

The snippet below shows how to register a trivial search implementation for
local demos without changing the application factory.

```python
from collections.abc import Iterable, Sequence
from typing import Any

from fastapi import FastAPI

from src.api.app_factory import create_app
from src.architecture.service_factory import ModeAwareServiceFactory

class InMemorySearchService:
    """Naive keyword search for development."""

    def __init__(self) -> None:
        self._documents: list[str] = []

    async def initialize(self) -> None:
        self._documents.append("welcome to the demo site")

    async def cleanup(self) -> None:
        self._documents.clear()

    def get_service_name(self) -> str:
        return "in_memory_search"

    async def add_documents(self, docs: Iterable[str]) -> None:
        self._documents.extend(docs)

    async def search(self, query: str, limit: int = 5) -> Sequence[str]:
        query_lower = query.lower()
        return [doc for doc in self._documents if query_lower in doc.lower()][:limit]


def build_app_with_dev_search() -> FastAPI:
    app = create_app()
    factory: ModeAwareServiceFactory = app.state.service_factory  # type: ignore[attr-defined]
    factory.register_universal_service("search_service", InMemorySearchService)
    return app
```

## Production Adapter Pattern

For production, wire concrete clients through the same service factory. The
example below registers an OpenSearch-backed hybrid search service.

```python
from fastapi import FastAPI
from opensearchpy import OpenSearch

from src.api.app_factory import create_app
from src.architecture.service_factory import ModeAwareServiceFactory

class OpenSearchHybridService:
    """Delegates search operations to OpenSearch."""

    def __init__(self, client: OpenSearch) -> None:
        self._client = client

    async def initialize(self) -> None:
        return None

    async def cleanup(self) -> None:
        self._client.transport.close()

    def get_service_name(self) -> str:
        return "opensearch_hybrid"

    async def search(self, query: str, *, index: str, size: int = 10) -> dict[str, Any]:
        return self._client.search(index=index, body={"query": {"match": {"text": query}}}, size=size)


def build_app_with_opensearch() -> FastAPI:
    app = create_app()
    factory: ModeAwareServiceFactory = app.state.service_factory  # type: ignore[attr-defined]
    client = OpenSearch(hosts=[{"host": "search.internal", "port": 9200}])
    factory.register_universal_service(
        "search_service",
        lambda: OpenSearchHybridService(client),
    )
    return app
```

## Cache Dependency Override

When a service only needs cache helpers, prefer FastAPI dependency overrides
instead of the service factory.

```python
from fastapi import Depends, FastAPI
from redis.asyncio import Redis

async def get_cache(redis: Redis = Depends(Redis.from_url)) -> Redis:
    return redis

app = FastAPI()
app.dependency_overrides[Redis.from_url] = lambda: Redis.from_url("redis://cache.internal:6379/0")
```

These recipes keep the core codebase minimal while providing clear extension
points for teams that require additional platform features.
