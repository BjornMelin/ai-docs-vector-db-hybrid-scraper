"""Shared fixtures for vector database unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient

from src.infrastructure.client_manager import ClientManager
from src.services.embeddings.base import EmbeddingProvider
from src.services.vector_db.adapter_base import CollectionSchema, TextDocument
from src.services.vector_db.service import VectorStoreService


class EmbeddingProviderStub:
    """Minimal async embedding provider for tests."""

    def __init__(self, dimension: int = 3) -> None:
        self.embedding_dimension = dimension
        self.initialized = False
        self.cleaned_up = False

    async def initialize(self) -> None:
        self.initialized = True
        self.cleaned_up = False

    async def cleanup(self) -> None:
        self.cleaned_up = True
        self.initialized = False

    async def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        base_vector = [float(index + 1) for index in range(self.embedding_dimension)]
        return [base_vector for _ in texts]


class ClientManagerStub:
    """Client manager stub that records lifecycle interactions."""

    def __init__(self, qdrant_client: AsyncQdrantClient | AsyncMock) -> None:
        self._qdrant_client = qdrant_client
        self.initialize_calls = 0
        self.cleanup_calls = 0
        self.is_initialized = False

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.is_initialized = True

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        self.is_initialized = False

    async def get_qdrant_client(self) -> AsyncQdrantClient | AsyncMock:
        return self._qdrant_client


def build_vector_store_service(
    config: Any,
    client_manager: ClientManagerStub,
    embeddings_provider: EmbeddingProviderStub,
) -> VectorStoreService:
    """Create a VectorStoreService instance wired with provided stubs."""

    return VectorStoreService(
        config=config,
        client_manager=cast(ClientManager, client_manager),
        embeddings_provider=cast(EmbeddingProvider, embeddings_provider),
    )


async def initialize_vector_store_service(
    config: Any,
    client_manager: ClientManagerStub,
    embeddings_provider: EmbeddingProviderStub,
) -> VectorStoreService:
    """Construct and initialize a VectorStoreService for reuse."""

    service = build_vector_store_service(config, client_manager, embeddings_provider)
    await service.initialize()
    return service


@pytest.fixture
def embeddings_provider_stub() -> EmbeddingProviderStub:
    """Provide an embedding provider stub with deterministic vectors."""

    return EmbeddingProviderStub(dimension=3)


@pytest.fixture
def qdrant_client_mock() -> AsyncMock:
    """Create a mocked Qdrant async client."""

    return AsyncMock(spec=AsyncQdrantClient)


@pytest.fixture
def client_manager_stub(qdrant_client_mock: AsyncMock) -> ClientManagerStub:
    """Provide a client manager stub wrapping the mocked Qdrant client."""

    return ClientManagerStub(qdrant_client=qdrant_client_mock)


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
        model = "stub-model"

    class _Config:
        fastembed = _FastEmbedConfig()

    return _Config()


@pytest.fixture
async def initialized_vector_store_service(
    embeddings_provider_stub: EmbeddingProviderStub,
    client_manager_stub: ClientManagerStub,
    config_stub: Any,
) -> AsyncIterator[VectorStoreService]:
    """Initialize VectorStoreService for tests and ensure cleanup."""

    service = await initialize_vector_store_service(
        config=config_stub,
        client_manager=client_manager_stub,
        embeddings_provider=embeddings_provider_stub,
    )
    try:
        yield service
    finally:
        await service.cleanup()


__all__ = [
    "EmbeddingProviderStub",
    "ClientManagerStub",
    "build_vector_store_service",
    "initialize_vector_store_service",
    "embeddings_provider_stub",
    "qdrant_client_mock",
    "client_manager_stub",
    "collection_schema",
    "sample_documents",
    "config_stub",
    "initialized_vector_store_service",
]
