"""Vector store service backed by the Qdrant adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.config import get_config
from src.services.base import BaseService
from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import EmbeddingServiceError

from .adapter import QdrantVectorAdapter
from .adapter_base import CollectionSchema, TextDocument, VectorMatch, VectorRecord


if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager


class VectorStoreService(BaseService):
    """High-level vector store operations built on top of Qdrant."""

    def __init__(
        self,
        config=None,
        client_manager: ClientManager | None = None,
        embeddings_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the VectorStoreService."""

        config = config or get_config()
        if client_manager is None:
            from src.infrastructure.client_manager import (
                ClientManager as _ClientManager,
            )

            client_manager = _ClientManager()
        if embeddings_provider is None:
            model_name = getattr(config.fastembed, "model", "BAAI/bge-small-en-v1.5")
            embeddings_provider = FastEmbedProvider(model_name=model_name)

        super().__init__(config)
        self._client_manager = client_manager
        self._embeddings = embeddings_provider
        self._adapter: QdrantVectorAdapter | None = None

    async def initialize(self) -> None:
        """Initialize the service and its dependencies."""

        if self.is_initialized():
            return
        await self._client_manager.initialize()
        client = await self._client_manager.get_qdrant_client()
        self._adapter = QdrantVectorAdapter(client)
        await self._embeddings.initialize()
        self._mark_initialized()

    async def cleanup(self) -> None:
        """Clean up resources."""

        if self._adapter:
            self._adapter = None
        await self._embeddings.cleanup()
        self._mark_uninitialized()

    async def ensure_collection(self, schema: CollectionSchema) -> None:
        """Ensure a collection exists with the given schema."""

        adapter = self._require_adapter()
        await adapter.create_collection(schema)

    async def drop_collection(self, name: str) -> None:
        """Drop a collection by name."""

        adapter = self._require_adapter()
        await adapter.drop_collection(name)

    async def list_collections(self) -> list[str]:
        """List all collections."""

        adapter = self._require_adapter()
        return await adapter.list_collections()

    async def collection_stats(self, name: str) -> Mapping[str, Any]:
        """Get statistics for a collection."""

        adapter = self._require_adapter()
        return await adapter.get_collection_stats(name)

    async def add_document(
        self,
        collection: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add a single document to the store."""

        document_id = str(uuid4())
        await self.upsert_documents(
            collection,
            [
                TextDocument(
                    id=document_id,
                    content=content,
                    metadata=dict(metadata or {}),
                )
            ],
        )
        return document_id

    async def upsert_documents(
        self,
        collection: str,
        documents: Sequence[TextDocument],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Upsert documents into a collection.

        Args:
            collection: Name of the collection.
            documents: Sequence of text documents to upsert.
            batch_size: Optional batch size for upsert operation.
        """

        if not documents:
            return
        adapter = self._require_adapter()
        try:
            embeddings = await self._embeddings.generate_embeddings(
                [document.content for document in documents]
            )
        except Exception as exc:  # pragma: no cover - provider-specific failures
            msg = f"Failed to generate embeddings: {exc}"
            raise EmbeddingServiceError(msg) from exc
        records = [
            VectorRecord(
                id=document.id,
                vector=vector,
                payload=document.metadata,
            )
            for document, vector in zip(documents, embeddings, strict=True)
        ]
        await adapter.upsert(collection, records, batch_size=batch_size)

    async def upsert_vectors(
        self,
        collection: str,
        records: Sequence[VectorRecord],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Upsert vector records into a collection.

        Args:
            collection: Name of the collection.
            records: Sequence of vector records to upsert.
            batch_size: Optional batch size for upsert operation.
        """

        adapter = self._require_adapter()
        await adapter.upsert(collection, records, batch_size=batch_size)

    async def get_document(
        self,
        collection: str,
        document_id: str,
    ) -> Mapping[str, Any] | None:
        adapter = self._require_adapter()
        records = await adapter.retrieve(
            collection,
            [document_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = dict(records[0].payload or {})
        payload.setdefault("id", records[0].id)
        return payload

    async def delete_document(self, collection: str, document_id: str) -> bool:
        adapter = self._require_adapter()
        await adapter.delete(collection, ids=[document_id])
        return True

    async def list_documents(
        self,
        collection: str,
        *,
        limit: int = 50,
        offset: str | None = None,
    ) -> tuple[list[Mapping[str, Any]], str | None]:
        adapter = self._require_adapter()
        records, next_offset = await adapter.scroll(
            collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        documents: list[Mapping[str, Any]] = []
        for record in records:
            payload = dict(record.payload or {})
            payload.setdefault("id", record.id)
            documents.append(payload)
        return documents, next_offset

    async def clear_collection(self, collection: str) -> None:
        schema = CollectionSchema(
            name=collection,
            vector_size=self._embeddings.embedding_dimension,
            distance="cosine",
        )
        adapter = self._require_adapter()
        await adapter.drop_collection(collection)
        await adapter.create_collection(schema)

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        """Delete records from a collection.

        Args:
            collection: Name of the collection.
            ids: Optional sequence of record IDs to delete.
            filters: Optional filters for deletion.
        """

        adapter = self._require_adapter()
        await adapter.delete(collection, ids=ids, filters=filters)

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Search documents using a text query.

        Args:
            collection: Name of the collection.
            query: Text query string.
            limit: Maximum number of results to return.
            filters: Optional filters for the search.

        Returns:
            List of vector matches.
        """

        adapter = self._require_adapter()
        try:
            [embedding] = await self._embeddings.generate_embeddings([query])
        except Exception as exc:  # pragma: no cover
            msg = f"Failed to embed query: {exc}"
            raise EmbeddingServiceError(msg) from exc
        return await adapter.query(
            collection,
            embedding,
            limit=limit,
            filters=filters,
        )

    async def search_vector(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Search using a vector.

        Args:
            collection: Name of the collection.
            vector: Vector to search with.
            limit: Maximum number of results to return.
            filters: Optional filters for the search.

        Returns:
            List of vector matches.
        """

        adapter = self._require_adapter()
        return await adapter.query(
            collection,
            vector,
            limit=limit,
            filters=filters,
        )

    async def hybrid_search(
        self,
        collection: str,
        query: str,
        sparse_vector: Mapping[int, float] | None = None,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Perform hybrid search using dense and sparse vectors.

        Args:
            collection: Name of the collection.
            query: Text query string.
            sparse_vector: Optional sparse vector for hybrid search.
            limit: Maximum number of results to return.
            filters: Optional filters for the search.

        Returns:
            List of vector matches.
        """

        adapter = self._require_adapter()
        dense_embedding, *_ = await self._embeddings.generate_embeddings([query])
        return await adapter.hybrid_query(
            collection,
            dense_embedding,
            sparse_vector,
            limit=limit,
            filters=filters,
        )

    def _require_adapter(self) -> QdrantVectorAdapter:
        """Get the adapter, raising error if not initialized.

        Returns:
            The Qdrant vector adapter.
        """

        if not self._adapter:
            msg = "VectorStoreService is not initialized"
            raise EmbeddingServiceError(msg)
        return self._adapter
