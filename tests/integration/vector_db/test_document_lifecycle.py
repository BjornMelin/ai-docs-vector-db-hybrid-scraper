"""Cross-layer document lifecycle tests against an in-memory Qdrant store."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.config import Settings
from src.config.models import Environment
from src.services.vector_db.service import VectorStoreService, dense_vector_config


pytestmark = pytest.mark.service


class _SyncQdrantAdapter:
    """Expose the sync in-memory client through the async methods under test."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    async def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    async def create_collection(self, **kwargs: Any) -> bool:
        return self.client.create_collection(**kwargs)

    async def retrieve(self, **kwargs: Any) -> list[Any]:
        return self.client.retrieve(**kwargs)

    async def get_collection(self, **kwargs: Any) -> Any:
        return self.client.get_collection(**kwargs)

    async def scroll(self, **kwargs: Any) -> tuple[list[Any], Any]:
        return self.client.scroll(**kwargs)

    async def delete(self, **kwargs: Any) -> Any:
        return self.client.delete(**kwargs)

    async def delete_collection(self, collection_name: str) -> bool:
        return self.client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_add_search_list_get_delete_uses_one_native_contract() -> None:
    """The returned ID must address the exact dependency-native stored point."""
    collection = "lifecycle"
    dimension = 8
    client = QdrantClient(location=":memory:")
    embeddings = DeterministicFakeEmbedding(size=dimension)
    adapter = _SyncQdrantAdapter(client)
    service = VectorStoreService(
        config=Settings(environment=Environment.TESTING),
        async_qdrant_client=cast(AsyncQdrantClient, adapter),
    )
    service._sync_client = client  # pylint: disable=protected-access
    service._dense_embeddings = embeddings  # type: ignore[assignment]  # pylint: disable=protected-access
    service._embedding_dimension = dimension  # pylint: disable=protected-access

    try:
        await service.ensure_collection(collection)
        stats = await service.collection_stats(collection)
        assert dense_vector_config(stats)["size"] == dimension
        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )
        service._vector_store = store  # pylint: disable=protected-access
        service._vector_stores[collection] = store  # pylint: disable=protected-access

        point_id = await service.add_document(
            collection,
            "Qdrant stores one canonical payload.",
            metadata={
                "id": "caller-controlled-id",
                "content": "caller-controlled-content",
                "topic": "contracts",
                "source": "integration-test",
            },
        )

        populated_stats = await service.collection_stats(collection)
        assert populated_stats["points_count"] == 1
        assert populated_stats["indexed_vectors"] == 0

        stored = client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        assert len(stored) == 1
        assert str(stored[0].id) == point_id
        assert stored[0].payload is not None
        assert set(stored[0].payload) == {"page_content", "metadata"}
        assert stored[0].payload["page_content"] == (
            "Qdrant stores one canonical payload."
        )
        assert stored[0].payload["metadata"]["topic"] == "contracts"
        assert "id" not in stored[0].payload["metadata"]
        assert "content" not in stored[0].payload["metadata"]

        fetched = await service.get_document(collection, point_id)
        assert fetched is not None
        assert fetched["id"] == point_id
        assert fetched["content"] == "Qdrant stores one canonical payload."
        assert fetched["topic"] == "contracts"

        listed, next_offset = await service.list_documents(collection, limit=10)
        assert listed == [fetched]
        assert next_offset is None

        matches = await service.search_documents(
            collection,
            query="canonical Qdrant payload",
            limit=5,
            filters={"topic": "contracts"},
        )
        assert [match.id for match in matches] == [point_id]
        assert matches[0].content == "Qdrant stores one canonical payload."
        assert matches[0].metadata is not None
        assert matches[0].metadata["topic"] == "contracts"

        assert await service.delete_document(collection, point_id) is True
        assert await service.get_document(collection, point_id) is None
        assert await service.delete_document(collection, point_id) is False
        await service.drop_collection(collection)
        assert client.collection_exists(collection) is False
    finally:
        client.close()


@pytest.mark.asyncio
async def test_complete_reingest_removes_obsolete_trailing_chunks() -> None:
    """Shrinking a document must not leave stale chunks searchable."""
    collection = "replacement"
    dimension = 8
    client = QdrantClient(location=":memory:")
    embeddings = DeterministicFakeEmbedding(size=dimension)
    adapter = _SyncQdrantAdapter(client)
    service = VectorStoreService(
        config=Settings(environment=Environment.TESTING),
        async_qdrant_client=cast(AsyncQdrantClient, adapter),
    )
    service._sync_client = client  # pylint: disable=protected-access
    service._dense_embeddings = embeddings  # type: ignore[assignment]  # pylint: disable=protected-access
    service._embedding_dimension = dimension  # pylint: disable=protected-access

    try:
        await service.ensure_collection(collection)
        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )
        service._vector_store = store  # pylint: disable=protected-access
        service._vector_stores[collection] = store  # pylint: disable=protected-access

        def chunks(total: int) -> list[Document]:
            return [
                Document(
                    page_content=f"chunk {index}",
                    metadata={
                        "doc_id": "stable-doc",
                        "tenant": "tenant-a",
                        "source": "integration-test",
                        "chunk_index": index,
                        "total_chunks": total,
                    },
                )
                for index in range(total)
            ]

        await service.replace_document_chunks(collection, chunks(4))
        await service.replace_document_chunks(collection, chunks(2))

        points, _ = client.scroll(
            collection_name=collection,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        payloads = [point.payload for point in points]
        assert all(payload is not None for payload in payloads)
        assert sorted(
            cast(dict[str, Any], payload)["metadata"]["chunk_index"]
            for payload in payloads
        ) == [0, 1]
    finally:
        client.close()
