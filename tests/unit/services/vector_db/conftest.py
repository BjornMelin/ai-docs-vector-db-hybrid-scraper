"""Fixtures and stubs for vector store service unit tests."""

# pylint: disable=c-extension-no-member

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from dependency_injector import providers
from qdrant_client import AsyncQdrantClient

from src.config.models import QdrantConfig, ScoreNormalizationStrategy, SearchStrategy
from src.infrastructure.container import ApplicationContainer
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import CollectionSchema, TextDocument


class _DenseEmbeddingStub:
    """Deterministic FastEmbed replacement for unit tests."""

    def __init__(self, model_name: str, **_: object) -> None:
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class _SparseEmbeddingStub:
    """Sparse FastEmbed replacement returning fixed coordinates."""

    def __init__(self, model_name: str, **_: object) -> None:
        self.model_name = model_name

    def embed_query(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(indices=[0, 1], values=[1.0, 0.5])


@pytest.fixture(autouse=True)
def fastembed_stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Patch FastEmbed embedding classes with lightweight stubs."""
    monkeypatch.setattr(
        "src.services.vector_db.service.FastEmbedEmbeddings",
        _DenseEmbeddingStub,
    )
    monkeypatch.setattr(
        "src.services.vector_db.service.FastEmbedSparseRuntime",
        _SparseEmbeddingStub,
    )
    monkeypatch.setattr(
        "src.services.vector_db.service.FastEmbedSparseType",
        _SparseEmbeddingStub,
    )
    yield


def build_vector_store_service(
    container: ApplicationContainer,
) -> VectorStoreService:
    """Create a VectorStoreService instance from the DI container."""
    return container.vector_store_service()


async def initialize_vector_store_service(
    container: ApplicationContainer,
) -> VectorStoreService:
    """Construct and initialize a VectorStoreService for reuse."""
    service = build_vector_store_service(container)
    await service.initialize()
    return service


@pytest.fixture
def qdrant_client_mock() -> AsyncMock:
    """Create a mocked Qdrant async client."""
    client = AsyncMock(spec=AsyncQdrantClient)
    client.collection_exists.return_value = True
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.get_collection.return_value = SimpleNamespace(
        points_count=0,
        indexed_vectors_count=0,
        config=SimpleNamespace(payload_schema={}),
    )
    client.scroll.return_value = ([], None)
    client.recommend.return_value = []
    client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    return client


@pytest.fixture
def vector_container(
    config_stub: Any,
    qdrant_client_mock: AsyncMock,
) -> Iterator[ApplicationContainer]:
    """Provide a container with overrides for vector service dependencies."""
    container = ApplicationContainer()
    container.qdrant_client.override(providers.Object(qdrant_client_mock))
    container.vector_store_service.override(
        providers.Singleton(
            VectorStoreService,
            config=config_stub,
            async_qdrant_client=qdrant_client_mock,
        )
    )
    try:
        yield container
    finally:
        container.vector_store_service.reset_override()
        container.qdrant_client.reset_override()


@pytest.fixture
def collection_schema() -> CollectionSchema:
    """Provide a reusable collection schema for adapter tests."""
    return CollectionSchema(name="docs", vector_size=3, distance="cosine")


@pytest.fixture
def sample_documents() -> list[TextDocument]:
    """Provide deterministic text documents for upsert tests."""
    return [
        TextDocument(
            id="doc-1",
            content="alpha",
            metadata={
                "doc_id": "doc-1",
                "chunk_id": 0,
                "chunk_index": 0,
                "total_chunks": 2,
                "tenant": "default",
                "source": "https://example.com/alpha",
                "uri_or_path": "https://example.com/alpha",
                "content_hash": "hash-alpha",
            },
        ),
        TextDocument(
            id="doc-2",
            content="beta",
            metadata={
                "doc_id": "doc-2",
                "chunk_id": 1,
                "chunk_index": 1,
                "total_chunks": 2,
                "tenant": "default",
                "source": "https://example.com/beta",
                "uri_or_path": "https://example.com/beta",
                "content_hash": "hash-beta",
            },
        ),
    ]


@pytest.fixture
def config_stub() -> Any:
    """Create a minimal config stub expected by VectorStoreService."""

    class _FastEmbedConfig:
        dense_model = "stub-model"
        sparse_model = "stub-sparse"

    class _EmbeddingConfig:
        dense_model = "stub-model"
        sparse_model = "stub-sparse"
        retrieval_mode = SearchStrategy.DENSE

    class _QueryProcessingConfig:
        enable_score_normalization = False
        score_normalization_strategy = ScoreNormalizationStrategy.MIN_MAX
        score_normalization_epsilon = 1e-6

    class _Config:
        fastembed = _FastEmbedConfig()
        embedding = _EmbeddingConfig()
        qdrant = QdrantConfig(enable_grouping=False)
        query_processing = _QueryProcessingConfig()

    return _Config()


@pytest.fixture
async def initialized_vector_store_service(
    vector_container: ApplicationContainer,
) -> AsyncIterator[VectorStoreService]:
    """Initialize VectorStoreService for tests and ensure cleanup."""
    service = await initialize_vector_store_service(vector_container)
    if service.config and hasattr(service.config, "qdrant"):
        service.config.qdrant.enable_grouping = False
    try:
        yield service
    finally:
        await service.cleanup()


__all__ = [
    "build_vector_store_service",
    "collection_schema",
    "config_stub",
    "initialize_vector_store_service",
    "initialized_vector_store_service",
    "qdrant_client_mock",
    "sample_documents",
    "vector_container",
]
