"""Unit tests for the LangChain-backed VectorStoreService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from src.infrastructure.container import ApplicationContainer
from src.services.vector_db.service import VectorStoreService, _filter_from_mapping
from src.services.vector_db.types import CollectionSchema, TextDocument, VectorRecord


class StubVectorStore:
    """Minimal stand-in for LangChain's QdrantVectorStore."""

    def __init__(self, *, collection_name: str, **_: object) -> None:  # noqa: D401
        self.collection_name = collection_name
        self.add_calls: list[tuple[list[Document], list[str]]] = []
        self.search_return: list[tuple[Document, float]] = []

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **_: object,
    ) -> None:
        self.add_calls.append((list(documents), list(ids or [])))

    def similarity_search_with_score_by_vector(
        self,
        *,
        vector: list[float],
        k: int,
        vector_filter: object | None = None,
    ) -> list[tuple[Document, float]]:
        _ = (vector, k, vector_filter)
        return self.search_return or [
            (
                Document(page_content="stub", metadata={"doc_id": "doc-1"}),
                0.42,
            )
        ]


@pytest.fixture
async def initialized_service(
    monkeypatch: pytest.MonkeyPatch,
    vector_container: ApplicationContainer,
) -> AsyncIterator[VectorStoreService]:
    """Provide an initialized VectorStoreService with stubbed dependencies."""

    stub_store = StubVectorStore(collection_name="documents")
    monkeypatch.setattr(
        "src.services.vector_db.service.QdrantVectorStore",
        lambda **kwargs: stub_store,
    )
    monkeypatch.setattr(
        "src.services.vector_db.service.VectorStoreService._build_sync_client",
        lambda self, cfg: MagicMock(),
    )

    service = vector_container.vector_store_service()
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


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(
    initialized_service: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_collection should call the async client when collection missing."""

    client = initialized_service._async_client
    assert isinstance(client, AsyncMock)
    client.collection_exists.return_value = False

    schema = CollectionSchema(name="new", vector_size=3)
    await initialized_service.ensure_collection(schema)

    client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_documents_invokes_vector_store(
    initialized_service: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upserting documents should delegate to the LangChain vector store."""

    store = initialized_service._vector_store
    assert isinstance(store, StubVectorStore)

    async def _immediate(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", _immediate)

    docs = [
        TextDocument(id="doc-1", content="alpha", metadata={"tenant": "acme"}),
        TextDocument(id="doc-2", content="beta", metadata={}),
    ]

    await initialized_service.upsert_documents("documents", docs)

    assert len(store.add_calls) == 1
    documents, ids = store.add_calls[0]
    assert [doc.page_content for doc in documents] == ["alpha", "beta"]
    assert len(ids) == 2 and ids[0] != ids[1]
    first_metadata = documents[0].metadata or {}
    second_metadata = documents[1].metadata or {}
    assert first_metadata["tenant"] == "acme"
    assert "created_at" in first_metadata
    assert second_metadata["tenant"] == "default"


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_documents should normalise results from the vector store."""

    store = initialized_service._vector_store
    assert isinstance(store, StubVectorStore)

    result_doc = Document(
        page_content="alpha",
        metadata={"doc_id": "doc-123", "topic": "testing"},
    )
    store.search_return = [(result_doc, 0.87)]

    async def _immediate(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", _immediate)

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
