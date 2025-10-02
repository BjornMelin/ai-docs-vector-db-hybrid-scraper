"""Unit tests for :class:`VectorStoreService`."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.services.errors import EmbeddingServiceError
from src.services.vector_db.adapter import QdrantVectorAdapter
from src.services.vector_db.adapter_base import (
    VectorMatch,
    VectorRecord,
)
from src.services.vector_db.service import VectorStoreService
from tests.unit.services.vector_db.conftest import initialize_vector_store_service


@pytest.mark.asyncio
async def test_initialize_creates_adapter(
    embeddings_provider_stub,
    client_manager_stub,
    qdrant_client_mock,
    config_stub,
) -> None:
    """Service initialize should construct the adapter once and mark state."""

    client_manager_stub._qdrant_client = qdrant_client_mock
    service = await initialize_vector_store_service(
        config_stub,
        client_manager_stub,
        embeddings_provider_stub,
    )

    assert isinstance(service._require_adapter(), QdrantVectorAdapter)
    assert client_manager_stub.initialize_calls == 1
    assert embeddings_provider_stub.initialized is True


@pytest.mark.asyncio
async def test_initialize_is_idempotent(
    embeddings_provider_stub,
    client_manager_stub,
    qdrant_client_mock,
    config_stub,
) -> None:
    """Double initialization should not re-invoke dependencies."""

    client_manager_stub._qdrant_client = qdrant_client_mock
    service = await initialize_vector_store_service(
        config_stub,
        client_manager_stub,
        embeddings_provider_stub,
    )
    await service.initialize()

    assert client_manager_stub.initialize_calls == 1


@pytest.mark.asyncio
async def test_cleanup_resets_state(
    embeddings_provider_stub,
    client_manager_stub,
    qdrant_client_mock,
    config_stub,
) -> None:
    """Cleanup should release adapter and embeddings provider."""

    client_manager_stub._qdrant_client = qdrant_client_mock
    service = await initialize_vector_store_service(
        config_stub,
        client_manager_stub,
        embeddings_provider_stub,
    )
    await service.cleanup()

    assert embeddings_provider_stub.cleaned_up is True
    with pytest.raises(EmbeddingServiceError, match="not initialized"):
        service._require_adapter()


@pytest.mark.asyncio
async def test_add_document_upserts_vector(
    initialized_vector_store_service,
    embeddings_provider_stub,
) -> None:
    """Adding a document should upsert a VectorRecord via the adapter."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    service._adapter = adapter_mock

    document_id = await service.add_document(
        "docs",
        "example text",
        metadata={"topic": "testing"},
    )

    adapter_mock.upsert.assert_awaited_once()
    args, kwargs = adapter_mock.upsert.call_args
    assert args[0] == "docs"
    records: Sequence[VectorRecord] = args[1]
    assert len(records) == 1
    record = records[0]
    assert len(record.id) == 32
    payload = record.payload or {}
    assert payload["doc_id"] == document_id
    assert payload["topic"] == "testing"
    assert payload["chunk_id"] == 0
    assert payload["tenant"] == "default"
    assert payload["source"] == "inline"
    assert "content_hash" in payload
    assert "created_at" in payload
    assert kwargs["batch_size"] is None


@pytest.mark.asyncio
async def test_upsert_documents_handles_batch_size(
    initialized_vector_store_service,
    embeddings_provider_stub,
    sample_documents,
) -> None:
    """Upsert should forward batch size and reuse embedding provider output."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    service._adapter = adapter_mock

    await service.upsert_documents(
        "docs",
        sample_documents,
        batch_size=64,
    )

    adapter_mock.upsert.assert_awaited_once()
    _, kwargs = adapter_mock.upsert.call_args
    assert kwargs["batch_size"] == 64


@pytest.mark.asyncio
async def test_upsert_documents_raises_on_embedding_failure(
    embeddings_provider_stub,
    client_manager_stub,
    qdrant_client_mock,
    sample_documents,
    config_stub,
) -> None:
    """Embedding failures should surface as EmbeddingServiceError."""

    async def _raise(_: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("boom")

    embeddings_provider_stub.generate_embeddings = _raise
    client_manager_stub._qdrant_client = qdrant_client_mock
    service = await initialize_vector_store_service(
        config_stub,
        client_manager_stub,
        embeddings_provider_stub,
    )
    service._adapter = AsyncMock(spec=QdrantVectorAdapter)

    with pytest.raises(EmbeddingServiceError, match="Failed to generate embeddings"):
        await service.upsert_documents("docs", sample_documents)


@pytest.mark.asyncio
async def test_get_document_returns_payload(
    initialized_vector_store_service,
) -> None:
    """Fetching a document should unwrap payload and preserve id."""

    service: VectorStoreService = initialized_vector_store_service
    match = VectorMatch(id="doc-1", score=0.8, payload={"foo": "bar"})
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.retrieve.return_value = [match]
    service._adapter = adapter_mock

    payload = await service.get_document("docs", "doc-1")

    assert payload == {"foo": "bar", "id": "doc-1"}
    adapter_mock.retrieve.assert_awaited_once_with(
        "docs",
        ["doc-1"],
        with_payload=True,
        with_vectors=False,
    )


@pytest.mark.asyncio
async def test_list_documents_includes_ids(
    initialized_vector_store_service,
) -> None:
    """list_documents should inject IDs into returned payloads."""

    service: VectorStoreService = initialized_vector_store_service
    matches = [
        VectorMatch(id="doc-1", score=0.9, payload={"topic": "async"}),
        VectorMatch(id="doc-2", score=0.7, payload={"topic": "sync"}),
    ]
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.scroll.return_value = (matches, "next")
    service._adapter = adapter_mock

    documents, cursor = await service.list_documents("docs", limit=2)

    assert cursor == "next"
    assert documents == [
        {"topic": "async", "id": "doc-1"},
        {"topic": "sync", "id": "doc-2"},
    ]


@pytest.mark.asyncio
async def test_hybrid_search_merges_embeddings(
    initialized_vector_store_service,
    embeddings_provider_stub,
) -> None:
    """Hybrid search should request embeddings and call adapter.hybrid_query."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.hybrid_query.return_value = [
        VectorMatch(id="doc-1", score=0.5, payload={}),
    ]
    service._adapter = adapter_mock

    results = await service.hybrid_search(
        "docs",
        "test query",
        sparse_vector={1: 0.3},
        limit=5,
    )

    assert len(results) == 1
    adapter_mock.hybrid_query.assert_awaited_once()
    _, kwargs = adapter_mock.hybrid_query.call_args
    assert kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_search_documents_returns_adapter_matches(
    initialized_vector_store_service,
) -> None:
    """search_documents should embed the query and return adapter results."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.query.return_value = [
        VectorMatch(id="x", score=0.42, payload={"title": "Doc"}),
    ]
    service._adapter = adapter_mock

    matches = await service.search_documents("docs", "query", limit=3)

    assert matches[0].id == "x"
    adapter_mock.query.assert_awaited_once()
    adapter_mock.query_groups.assert_not_called()


@pytest.mark.asyncio
async def test_search_vector_passthrough(
    initialized_vector_store_service,
) -> None:
    """search_vector should forward dense vectors directly to adapter.query."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.query.return_value = []
    service._adapter = adapter_mock

    await service.search_vector("docs", [0.1, 0.2, 0.3], limit=2)

    adapter_mock.query.assert_awaited_once_with(
        "docs",
        [0.1, 0.2, 0.3],
        limit=2,
        filters=None,
    )
    adapter_mock.query_groups.assert_not_called()


@pytest.mark.asyncio
async def test_search_documents_client_side_grouping_fallback(
    client_manager_stub,
    qdrant_client_mock,
    embeddings_provider_stub,
    config_stub,
) -> None:
    """Grouping disabled, service should group client-side and annotate metadata."""

    client_manager_stub._qdrant_client = qdrant_client_mock
    service = await initialize_vector_store_service(
        config_stub,
        client_manager_stub,
        embeddings_provider_stub,
    )
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.query.return_value = [
        VectorMatch(
            id="chunk-1",
            score=0.9,
            raw_score=0.9,
            payload={"doc_id": "doc-1", "content": "alpha"},
        ),
        VectorMatch(
            id="chunk-2",
            score=0.8,
            raw_score=0.8,
            payload={"doc_id": "doc-1", "content": "beta"},
        ),
        VectorMatch(
            id="chunk-3",
            score=0.5,
            raw_score=0.5,
            payload={"doc_id": "doc-2", "content": "gamma"},
        ),
    ]
    service._adapter = adapter_mock

    results = await service.search_documents(
        collection="docs",
        query="test",
        limit=2,
        group_by="doc_id",
        group_size=1,
        normalize_scores=False,
    )

    adapter_mock.query.assert_awaited_once()
    assert len(results) == 2
    first, second = results
    assert first.payload["_grouping"]["applied"] is False
    assert first.payload["_grouping"]["group_id"] == "doc-1"
    assert first.payload["_grouping"]["rank"] == 1
    assert second.payload["_grouping"]["group_id"] == "doc-2"
    assert {match.id for match in results} == {"chunk-1", "chunk-3"}


@pytest.mark.asyncio
async def test_search_documents_normalizes_scores_when_enabled(
    client_manager_stub,
    qdrant_client_mock,
    embeddings_provider_stub,
    config_stub,
) -> None:
    """Score normalisation should populate normalized_score and update score."""

    client_manager_stub._qdrant_client = qdrant_client_mock
    config_stub.query_processing.enable_score_normalization = True  # type: ignore[attr-defined]
    service = await initialize_vector_store_service(
        config_stub,
        client_manager_stub,
        embeddings_provider_stub,
    )
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.query.return_value = [
        VectorMatch(
            id="chunk-1",
            score=0.9,
            raw_score=0.9,
            payload={"doc_id": "doc-1", "content": "alpha"},
        ),
        VectorMatch(
            id="chunk-2",
            score=0.8,
            raw_score=0.8,
            payload={"doc_id": "doc-2", "content": "beta"},
        ),
    ]
    service._adapter = adapter_mock

    results = await service.search_documents(
        collection="docs",
        query="test",
        limit=2,
        normalize_scores=True,
    )

    assert [match.raw_score for match in results] == [0.9, 0.8]
    assert [match.normalized_score for match in results] == [1.0, 0.0]
    assert [match.score for match in results] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_search_documents_uses_query_groups_when_supported(
    initialized_vector_store_service,
) -> None:
    """Grouping should be used when enabled and supported by the adapter."""

    service: VectorStoreService = initialized_vector_store_service
    config = service.config
    assert config is not None
    config.qdrant.enable_grouping = True
    config.qdrant.group_by_field = "doc_id"
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.supports_query_groups.return_value = True
    grouped_match = VectorMatch(
        id="doc-1",
        score=0.9,
        payload={"_grouping": {"applied": True, "group_id": "doc-1"}},
    )
    adapter_mock.query_groups.return_value = ([grouped_match], True)
    adapter_mock.get_collection_info.return_value = SimpleNamespace(
        payload_schema={}, status="green", vectors_count=0, points_count=0
    )
    service._adapter = adapter_mock

    matches = await service.search_documents("docs", "query", limit=1)

    adapter_mock.query_groups.assert_awaited_once()
    adapter_mock.query.assert_not_called()
    assert matches == [grouped_match]


@pytest.mark.asyncio
async def test_delete_document_forwards_to_adapter(
    initialized_vector_store_service,
) -> None:
    """Deleting a document should call adapter.delete with ids."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    service._adapter = adapter_mock

    deleted = await service.delete_document("docs", "doc-9")

    assert deleted is True
    adapter_mock.delete.assert_awaited_once_with("docs", ids=["doc-9"])


@pytest.mark.asyncio
async def test_ensure_collection_uses_adapter(
    initialized_vector_store_service,
    collection_schema,
) -> None:
    """ensure_collection delegates to the adapter."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    service._adapter = adapter_mock

    adapter_mock.collection_exists.return_value = False

    await service.ensure_collection(collection_schema)

    adapter_mock.collection_exists.assert_awaited_once_with(collection_schema.name)
    adapter_mock.create_collection.assert_awaited_once_with(collection_schema)


@pytest.mark.asyncio
async def test_ensure_collection_noop_when_existing(
    initialized_vector_store_service,
    collection_schema,
) -> None:
    """ensure_collection returns early if the collection is present."""

    service: VectorStoreService = initialized_vector_store_service
    adapter_mock = AsyncMock(spec=QdrantVectorAdapter)
    adapter_mock.collection_exists.return_value = True
    service._adapter = adapter_mock

    await service.ensure_collection(collection_schema)

    adapter_mock.collection_exists.assert_awaited_once_with(collection_schema.name)
    adapter_mock.create_collection.assert_not_awaited()
