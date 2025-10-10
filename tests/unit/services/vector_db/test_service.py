"""Unit tests for the Qdrant-backed VectorStoreService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from src.services.vector_db.service import VectorStoreService, _filter_from_mapping
from src.services.vector_db.types import CollectionSchema, TextDocument, VectorRecord
from tests.unit.services.vector_db.conftest import (
    ClientManagerStub,
    EmbeddingProviderStub,
    build_vector_store_service,
)


@pytest.fixture
async def initialized_service() -> AsyncIterator[VectorStoreService]:
    """Provide an initialized VectorStoreService with stubbed dependencies."""

    embeddings = EmbeddingProviderStub(dimension=3)

    async_client = AsyncMock(spec=AsyncQdrantClient)
    async_client.collection_exists.return_value = True
    async_client.upsert.return_value = None
    async_client.search.return_value = []
    client_manager = ClientManagerStub(async_client)

    config = SimpleNamespace(
        fastembed=SimpleNamespace(model="stub"),
        qdrant=SimpleNamespace(
            url="http://localhost:6333",
            api_key=None,
            timeout=1.0,
            prefer_grpc=False,
            use_grpc=False,
            grpc_port=6334,
            collection_name="documents",
            enable_grouping=False,
        ),
        query_processing=SimpleNamespace(
            enable_score_normalization=False,
            score_normalization_strategy=None,
            score_normalization_epsilon=1e-6,
        ),
    )

    service = build_vector_store_service(config, client_manager, embeddings)
    await service.initialize()
    try:
        yield service
    finally:
        await service.cleanup()


@pytest.mark.asyncio
async def test_initialize_sets_vector_store(
    initialized_service: VectorStoreService,
) -> None:
    """Vector store should be initialized once with configured collection name."""

    service = initialized_service
    assert service.is_initialized()
    assert service.embedding_dimension == 3
    assert isinstance(service._async_client, AsyncMock)


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(
    initialized_service: VectorStoreService,
) -> None:
    """ensure_collection should call the async client when collection missing."""

    client = initialized_service._async_client
    assert isinstance(client, AsyncMock)
    client.collection_exists.return_value = False

    schema = CollectionSchema(name="new", vector_size=3)
    await initialized_service.ensure_collection(schema)

    client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_documents_invokes_async_client(
    initialized_service: VectorStoreService,
) -> None:
    """Upserting documents should call the async Qdrant client with points."""

    docs = [
        TextDocument(id="doc-1", content="alpha", metadata={"tenant": "acme"}),
        TextDocument(id="doc-2", content="beta", metadata={}),
    ]

    await initialized_service.upsert_documents("documents", docs)

    async_client = initialized_service._async_client
    assert isinstance(async_client, AsyncMock)
    async_client.upsert.assert_awaited_once()
    _, kwargs = async_client.upsert.call_args
    assert kwargs["collection_name"] == "documents"
    points = list(kwargs["points"])
    assert len(points) == 2
    payloads = [p.payload for p in points]
    assert payloads[0]["tenant"] == "acme"
    assert payloads[1]["tenant"] == "default"
    assert "created_at" in payloads[0]


@pytest.mark.asyncio
async def test_upsert_vectors_calls_qdrant_async_client(
    initialized_service: VectorStoreService,
) -> None:
    """upsert_vectors should prepare PointStruct payloads for Async client."""

    service = initialized_service
    async_client = service._async_client
    assert isinstance(async_client, AsyncMock)

    await service.upsert_vectors(
        "documents",
        [
            VectorRecord(
                id="doc-1",
                vector=[0.1, 0.2, 0.3],
                payload={"foo": "bar"},
            )
        ],
    )

    async_client.upsert.assert_awaited_once()
    _, kwargs = async_client.upsert.call_args
    assert kwargs["collection_name"] == "documents"
    points = list(kwargs["points"])
    assert len(points) == 1
    point = points[0]
    assert point.payload["foo"] == "bar"


def test_filter_from_mapping_handles_sequences_and_scalars() -> None:
    """Sequence values should map to MatchAny, scalars to MatchValue."""

    filters = _filter_from_mapping({"tags": ["doc", "faq"], "lang": "en"})
    assert isinstance(filters, models.Filter)
    raw_must = filters.must
    if raw_must is None:
        must_conditions: list[models.Condition] = []
    elif isinstance(raw_must, list):
        must_conditions = list(raw_must)
    else:
        must_conditions = [raw_must]
    match_any = None
    for condition in must_conditions:
        if getattr(condition, "key", None) != "tags":
            continue
        candidate = getattr(condition, "match", None)
        if isinstance(candidate, models.MatchAny):
            match_any = candidate
            break
    assert match_any is not None
    assert isinstance(match_any, models.MatchAny)
    assert match_any.any == ["doc", "faq"]

    match_value = None
    for condition in must_conditions:
        if getattr(condition, "key", None) != "lang":
            continue
        candidate = getattr(condition, "match", None)
        if isinstance(candidate, models.MatchValue):
            match_value = candidate
            break
    assert match_value is not None
    assert match_value.value == "en"


def test_filter_from_mapping_treats_strings_as_scalars() -> None:
    """String sequences should not be treated as iterables for MatchAny."""

    filters = _filter_from_mapping({"category": "docs"})
    assert isinstance(filters, models.Filter)
    raw_must = filters.must
    assert raw_must is not None
    condition = raw_must[0] if isinstance(raw_must, list) else raw_must
    condition_match = getattr(condition, "match", None)
    assert isinstance(condition_match, models.MatchValue)
    assert condition_match.value == "docs"


@pytest.mark.asyncio
async def test_search_documents_returns_vector_matches(
    initialized_service: VectorStoreService,
) -> None:
    """search_documents should normalise results returned by Qdrant."""

    async_client = initialized_service._async_client
    assert isinstance(async_client, AsyncMock)
    async_client.search.return_value = [
        SimpleNamespace(
            id="doc-123",
            score=0.87,
            payload={"doc_id": "doc-123", "topic": "testing"},
        )
    ]

    matches = await initialized_service.search_documents(
        "documents",
        query="alpha",
        limit=5,
    )

    assert len(matches) == 1
    match = matches[0]
    assert match.id == "doc-123"
    assert match.metadata is not None
    assert match.metadata["topic"] == "testing"
    assert pytest.approx(match.score) == 0.87
