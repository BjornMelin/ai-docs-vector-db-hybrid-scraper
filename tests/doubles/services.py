"""Centralized test doubles for core services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock


@dataclass
class FakeCache:
    """In-memory cache double for testing."""

    _store: dict[str, Any] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    async def get(self, key: str) -> Any | None:
        """Retrieve a value from the fake cache."""
        value = self._store.get(key)
        if value is not None:
            self._hits += 1
        else:
            self._misses += 1
        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a value in the fake cache."""
        self._store[key] = value

    async def delete(self, key: str) -> bool:
        """Remove a value from the fake cache."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def clear(self) -> None:
        """Clear all values from the fake cache."""
        self._store.clear()

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._store

    @property
    def hit_count(self) -> int:
        """Return the number of cache hits."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Return the number of cache misses."""
        return self._misses


@dataclass
class FakeVectorStoreService:
    """Vector store service double for testing."""

    _documents: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _initialized: bool = False
    embedding_dimension: int = 3
    collection_name: str = "test-collection"

    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the fake service."""
        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up the fake service."""
        self._initialized = False
        self._documents.clear()

    async def ensure_collection(self, schema: Any) -> None:
        """Ensure collection exists."""
        name = getattr(schema, "name", "default")
        if name not in self._documents:
            self._documents[name] = []

    async def upsert_documents(
        self, collection: str, documents: list[Any]
    ) -> list[str]:
        """Upsert documents to the collection."""
        if collection not in self._documents:
            self._documents[collection] = []
        ids = []
        for doc in documents:
            doc_id = getattr(doc, "id", str(len(self._documents[collection])))
            ids.append(doc_id)
            self._documents[collection].append(
                {"id": doc_id, "content": getattr(doc, "content", "")}
            )
        return ids

    async def search_documents(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Search documents in the collection."""
        docs = self._documents.get(collection, [])
        return docs[:limit]


@dataclass
class FakeCrawlManager:
    """Crawl manager double for testing."""

    _crawled_urls: list[str] = field(default_factory=list)

    async def crawl(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate crawling a URL."""
        self._crawled_urls.append(url)
        return {"url": url, "content": f"Content from {url}", "success": True}

    async def batch_crawl(
        self,
        urls: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Simulate batch crawling."""
        return [await self.crawl(url) for url in urls]


@dataclass
class FakeEmbeddingManager:
    """Embedding manager double for testing."""

    _dimension: int = 3
    _call_count: int = 0

    async def embed_query(self, text: str) -> list[float]:
        """Generate fake embedding for a query."""
        self._call_count += 1
        return [0.1, 0.2, 0.3]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings for documents."""
        return [await self.embed_query(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


@dataclass
class FakeProjectStorage:
    """Project storage double for testing.

    Mirrors the real ProjectStorage API from src/infrastructure/project_storage.py.
    """

    _projects: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def load_projects(self) -> dict[str, dict[str, Any]]:
        """Load all projects into cache and return a copy."""
        return {key: dict(value) for key, value in self._projects.items()}

    async def save_project(self, project_id: str, project_data: dict[str, Any]) -> None:
        """Persist a single project definition."""
        self._projects[project_id] = dict(project_data)

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Return a project by identifier."""
        project = self._projects.get(project_id)
        return dict(project) if project else None

    async def list_projects(self) -> list[dict[str, Any]]:
        """Return the list of all projects."""
        return [dict(project) for project in self._projects.values()]

    async def update_project(self, project_id: str, updates: dict[str, Any]) -> None:
        """Apply partial updates to an existing project."""
        if project_id not in self._projects:
            msg = f"Project {project_id} not found"
            raise KeyError(msg)
        self._projects[project_id].update(updates)

    async def delete_project(self, project_id: str) -> None:
        """Remove a project from storage (idempotent)."""
        self._projects.pop(project_id, None)


@dataclass
class FakeMCPServer:
    """MCP server double for testing."""

    _tools: dict[str, Any] = field(default_factory=dict)
    _running: bool = False

    def register_tool(self, name: str, handler: Any) -> None:
        """Register a tool with the server."""
        self._tools[name] = handler

    async def start(self) -> None:
        """Start the fake server."""
        self._running = True

    async def stop(self) -> None:
        """Stop the fake server."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


@dataclass
class FakeServiceContainer:
    """Service container double that provides fake services."""

    cache: FakeCache = field(default_factory=FakeCache)
    vector_store: FakeVectorStoreService = field(default_factory=FakeVectorStoreService)
    crawl_manager: FakeCrawlManager = field(default_factory=FakeCrawlManager)
    embedding_manager: FakeEmbeddingManager = field(
        default_factory=FakeEmbeddingManager
    )
    project_storage: FakeProjectStorage = field(default_factory=FakeProjectStorage)
    mcp_server: FakeMCPServer = field(default_factory=FakeMCPServer)

    # Mock clients for compatibility
    qdrant_client: AsyncMock = field(default_factory=AsyncMock)
    dragonfly_client: MagicMock = field(default_factory=MagicMock)

    def cache_manager(self) -> FakeCache:
        """Return the fake cache manager."""
        return self.cache

    def vector_store_service(self) -> FakeVectorStoreService:
        """Return the fake vector store service."""
        return self.vector_store

    def crawl_manager_service(self) -> FakeCrawlManager:
        """Return the fake crawl manager."""
        return self.crawl_manager

    def embedding_manager_service(self) -> FakeEmbeddingManager:
        """Return the fake embedding manager."""
        return self.embedding_manager

    def project_storage_service(self) -> FakeProjectStorage:
        """Return the fake project storage."""
        return self.project_storage
