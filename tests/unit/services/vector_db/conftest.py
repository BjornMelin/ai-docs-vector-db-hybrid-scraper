"""Shared fixtures for vector database unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from dependency_injector import providers
from qdrant_client import AsyncQdrantClient

from src.config.models import QdrantConfig, ScoreNormalizationStrategy
from src.infrastructure.container import ApplicationContainer
from src.services.embeddings.base import EmbeddingProvider
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import CollectionSchema, TextDocument


class _LangChainEmbeddingStub:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        base = [float(index + 1) for index in range(self._dimension)]
        return [base for _ in texts]

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [float(index + 1) for index in range(self._dimension)]


class EmbeddingProviderStub:
    """Minimal async embedding provider for tests."""

    def __init__(self, dimension: int = 3) -> None:
        self.embedding_dimension = dimension
        self.initialized = False
        self.cleaned_up = False
        self._embedding = _LangChainEmbeddingStub(dimension)

    async def initialize(self) -> None:
        self.initialized = True
        self.cleaned_up = False

    async def cleanup(self) -> None:
        self.cleaned_up = True
        self.initialized = False

    async def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        base_vector = [float(index + 1) for index in range(self.embedding_dimension)]
        return [base_vector for _ in texts]

    @property
    def langchain_embeddings(self) -> _LangChainEmbeddingStub:
        return self._embedding


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
def embeddings_provider_stub() -> EmbeddingProviderStub:
    """Provide an embedding provider stub with deterministic vectors."""

    return EmbeddingProviderStub(dimension=3)


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
    return client


@pytest.fixture
def vector_container(
    config_stub: Any,
    qdrant_client_mock: AsyncMock,
    embeddings_provider_stub: EmbeddingProviderStub,
) -> Iterator[ApplicationContainer]:
    """Provide a container with overrides for vector service dependencies."""

    container = ApplicationContainer()
    container.qdrant_client.override(providers.Object(qdrant_client_mock))
    container.vector_store_service.override(
        providers.Singleton(
            VectorStoreService,
            config=config_stub,
            async_qdrant_client=qdrant_client_mock,
            embeddings_provider=cast(EmbeddingProvider, embeddings_provider_stub),
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
        TextDocument(id="doc-1", content="alpha", metadata={"lang": "py"}),
        TextDocument(id="doc-2", content="beta", metadata={"lang": "js"}),
    ]


@pytest.fixture
def config_stub() -> Any:
    """Create a minimal config stub expected by VectorStoreService."""

    class _FastEmbedConfig:
        dense_model = "stub-model"

    class _QueryProcessingConfig:
        enable_score_normalization = False
        score_normalization_strategy = ScoreNormalizationStrategy.MIN_MAX
        score_normalization_epsilon = 1e-6

    class _Config:
        fastembed = _FastEmbedConfig()
        qdrant = QdrantConfig(enable_grouping=False)
        query_processing = _QueryProcessingConfig()

    return _Config()


@pytest.fixture
async def initialized_vector_store_service(
    embeddings_provider_stub: EmbeddingProviderStub,
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
    "EmbeddingProviderStub",
    "build_vector_store_service",
    "initialize_vector_store_service",
    "embeddings_provider_stub",
    "qdrant_client_mock",
    "vector_container",
    "collection_schema",
    "sample_documents",
    "config_stub",
    "initialized_vector_store_service",
]
