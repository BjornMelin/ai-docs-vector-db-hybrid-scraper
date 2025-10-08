"""Shared pytest fixtures and fakes."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import pytest

from src.contracts.retrieval import SearchRecord


class FakeMCP:
    """Minimal FastMCP stand-in that records registered tools."""

    def __init__(self) -> None:
        """Init registry."""

        self.tools: dict[str, Any] = {}

    def tool(self, *_, name: str | None = None, **__):  # pragma: no cover
        """Return decorator that stores a tool function."""

        def _decorator(fn):
            tool_name = name or fn.__name__
            self.tools[tool_name] = fn
            return fn

        return _decorator


class FakeVectorStoreService:
    """In-memory vector service fake."""

    def __init__(self) -> None:
        """Init store."""

        self._inited = False
        self.embedding_dimension = 1536
        self._docs: dict[str, list[dict[str, Any]]] = {"documentation": []}
        self._indexes: dict[str, list[str]] = {}
        for i in range(5):
            self._docs["documentation"].append(
                {"id": f"d{i}", "content": f"Doc {i}", "score": 0.9 - i * 0.1}
            )

    def is_initialized(self) -> bool:
        """Return init state."""

        return self._inited

    async def initialize(self) -> None:
        """Mark initialized."""

        self._inited = True

    async def cleanup(self) -> None:
        """Noop cleanup."""

        self._inited = False

    async def list_collections(self) -> list[str]:
        """List collections."""

        return list(self._docs.keys())

    async def collection_stats(self, name: str) -> dict:
        """Return simple stats."""

        count = len(self._docs.get(name, []))
        return {"points_count": count, "vectors": {"size": count}}

    async def ensure_collection(self, schema) -> None:  # noqa: ANN001
        """Ensure collection exists."""

        self._docs.setdefault(schema.name, [])

    async def drop_collection(self, name: str) -> None:
        """Drop collection."""

        self._docs.pop(name, None)

    async def list_documents(
        self, name: str, *, limit: int, offset: str | None
    ) -> tuple[list[dict], str | None]:
        """Return paginated docs."""

        items = self._docs.get(name, [])
        start = int(offset or 0)
        end = min(start + limit, len(items))
        return items[start:end], str(end) if end < len(items) else None

    async def upsert_documents(self, name: str, docs: Iterable[Any]) -> None:
        """Insert docs."""

        for d in docs:
            self._docs.setdefault(name, []).append(
                {"id": d.id, "content": d.content, "metadata": d.metadata}
            )

    async def get_document(self, name: str, point_id: str) -> dict | None:
        """Get document payload."""

        for d in self._docs.get(name, []):
            if d.get("id") == point_id:
                return d
        return None

    async def search_documents(
        self,
        name: str,
        query: str,
        *,
        limit: int,
        filters: dict | None,
    ) -> list[SearchRecord]:
        """Return synthetic matches."""

        base: list[SearchRecord] = []
        for i in range(1, limit + 1):
            payload: dict[str, Any] = {"q": query}
            if filters:
                payload["filters"] = filters
            base.append(
                SearchRecord(
                    id=str(i),
                    score=1.0 - i * 0.01,
                    content="",
                    metadata=payload,
                )
            )
        return base

    async def recommend(
        self,
        name: str,
        *,
        positive_ids: list[str],
        limit: int,
        filters: dict | None,
    ) -> list[SearchRecord]:
        """Recommend based on seed."""

        seed = positive_ids[0]
        recs: list[SearchRecord] = []
        for i in range(limit + 1):
            payload: dict[str, Any] = {"seed": seed}
            if filters:
                payload["filters"] = filters
            recs.append(
                SearchRecord(
                    id=f"rec{i}",
                    score=0.95 - i * 0.05,
                    content="",
                    metadata=payload,
                )
            )
        # include the seed itself to test exclusion
        recs.append(
            SearchRecord(id=seed, score=1.0, content="", metadata={"seed": seed})
        )
        return recs

    async def ensure_payload_indexes(self, name: str, defs: dict[str, Any]) -> dict:
        """Create indexes meta."""

        fields = list(defs.keys())
        self._indexes[name] = fields
        return {
            "indexed_fields": fields,
            "indexed_fields_count": len(fields),
            "points_count": len(self._docs.get(name, [])),
            "payload_schema": defs,
        }

    async def drop_payload_indexes(self, name: str, fields: Iterable[str]) -> None:
        """Drop indexes."""

        existing = set(self._indexes.get(name, []))
        remain = [f for f in existing if f not in set(fields)]
        self._indexes[name] = remain

    async def get_payload_index_summary(self, name: str) -> dict:
        """Return index summary."""

        fields = self._indexes.get(name, [])
        return {
            "indexed_fields": fields,
            "indexed_fields_count": len(fields),
            "points_count": len(self._docs.get(name, [])),
        }


class FakeCache:
    """Tiny cache fake with counters."""

    def __init__(self) -> None:
        """Init cache."""

        self._store: dict[str, Any] = {}
        self._total_requests = 0

    async def get(self, key: str) -> Any | None:
        """Get item."""

        self._total_requests += 1
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set item."""

        del ttl
        self._store[key] = value

    async def clear_pattern(self, pattern: str) -> int:
        """Clear by pattern."""

        del pattern
        n = len(self._store)
        self._store.clear()
        return n

    async def clear_all(self) -> int:
        """Clear all."""

        n = len(self._store)
        self._store.clear()
        return n

    async def get_stats(self) -> dict:
        """Return stats."""

        return {
            "hit_rate": 0.5,
            "size": len(self._store),
            "total_requests": self._total_requests,
        }


class FakeCrawlManager:
    """Simple crawler fake."""

    async def scrape_url(self, url: str, **kwargs) -> dict:
        """Return canned scrape result."""

        del kwargs
        return {
            "success": True,
            "url": url,
            "content": f"Content for {url}",
            "title": "T",
            "metadata": {},
        }


class FakeEmbeddingManager:
    """Embedding manager fake."""

    async def generate_embeddings(
        self, texts: list[str], provider_name: str | None, generate_sparse: bool
    ) -> dict:
        """Return dense embeddings."""

        del provider_name, generate_sparse
        return {
            "embeddings": [[0.1, 0.2]] * len(texts),
            "model": "test-model",
            "provider": "fake",
            "tokens": 42,
        }

    def estimate_cost(self, texts: list[str], provider_name: str | None) -> dict:
        """Return cost estimate."""

        del provider_name
        return {"fake": {"total_cost": 0.0, "count": len(texts)}}

    def get_provider_info(self) -> dict:
        """Return provider info."""

        return {"fake": {"model": "test-model", "dimensions": 2, "max_tokens": 8191}}


class FakeProjectStorage:
    """In-memory project store."""

    def __init__(self) -> None:
        """Init store."""

        self._store: dict[str, dict] = {}

    async def save_project(self, pid: str, data: dict) -> None:
        """Save record."""

        self._store[pid] = data

    async def list_projects(self) -> list[dict]:
        """List records."""

        return list(self._store.values())

    async def get_project(self, pid: str) -> dict | None:
        """Get record."""

        return self._store.get(pid)

    async def update_project(self, pid: str, updates: dict) -> None:
        """Update record."""

        self._store[pid].update(updates)

    async def delete_project(self, pid: str) -> None:
        """Delete record."""

        self._store.pop(pid, None)


class FakeClientManager:
    """Client manager fake."""

    def __init__(self) -> None:
        """Init fakes."""

        self.unified_config = type(
            "Cfg",
            (),
            {
                "qdrant": type("Q", (), {"url": "http://qdrant"})(),
                "cache": type("C", (), {"max_items": 100, "redis_url": None})(),
                "openai": type("O", (), {"api_key": os.getenv("OPENAI_API_KEY")})(),
                "firecrawl": type(
                    "F", (), {"api_key": os.getenv("AI_DOCS__FIRECRAWL__API_KEY")}
                )(),
            },
        )()
        self._vector = FakeVectorStoreService()
        self._cache = FakeCache()
        self._crawl = FakeCrawlManager()
        self._embed = FakeEmbeddingManager()
        self._projects = FakeProjectStorage()

    async def get_vector_store_service(self) -> FakeVectorStoreService:
        """Return vector service."""

        return self._vector

    async def get_cache_manager(self) -> FakeCache:
        """Return cache manager."""

        return self._cache

    async def get_crawl_manager(self) -> FakeCrawlManager:
        """Return crawl manager."""

        return self._crawl

    async def get_embedding_manager(self) -> FakeEmbeddingManager:
        """Return embedding manager."""

        return self._embed

    async def get_project_storage(self) -> FakeProjectStorage:
        """Return project storage."""

        return self._projects


@pytest.fixture()
def fake_client_manager() -> FakeClientManager:
    """Provide a fake ClientManager."""

    return FakeClientManager()


@pytest.fixture()
def fake_mcp() -> FakeMCP:
    """Provide a fake MCP app."""

    return FakeMCP()


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Select asyncio backend."""

    return "asyncio"
