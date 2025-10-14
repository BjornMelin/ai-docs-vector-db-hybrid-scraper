"""Vector store service implemented on top of LangChain's Qdrant integration."""
# pylint: disable=too-many-arguments,too-many-return-statements,too-many-branches,too-many-locals,too-many-lines

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from src.config.loader import Settings
from src.config.models import (
    QueryProcessingConfig,
    ScoreNormalizationStrategy,
    SearchStrategy,
)
from src.contracts.retrieval import SearchRecord
from src.services.errors import EmbeddingServiceError
from src.services.observability.tracing import set_span_attributes

from .payload_schema import (
    CanonicalPayload,
    PayloadValidationError,
    ensure_canonical_payload,
)
from .types import CollectionSchema, TextDocument, VectorRecord


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing aid
    from langchain_qdrant import FastEmbedSparse as FastEmbedSparseType
else:  # pragma: no cover - runtime fallback
    FastEmbedSparseType = Any

try:  # pragma: no cover - optional sparse dependency
    from langchain_qdrant import FastEmbedSparse as FastEmbedSparseRuntime
except ModuleNotFoundError:  # pragma: no cover - defer sparse usage checks
    FastEmbedSparseRuntime = None  # type: ignore[assignment]


_RETRIEVAL_MODE_MAP: dict[SearchStrategy, RetrievalMode] = {
    SearchStrategy.DENSE: RetrievalMode.DENSE,
    SearchStrategy.SPARSE: RetrievalMode.SPARSE,
    SearchStrategy.HYBRID: RetrievalMode.HYBRID,
}


class VectorStoreService:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """High-level vector store wrapper using LangChain's QdrantVectorStore."""

    def __init__(
        self,
        *,
        config: Settings,
        async_qdrant_client: AsyncQdrantClient,
    ) -> None:
        """Initialize the VectorStoreService."""

        self.config = config
        self.collection_name: str | None = None
        self._async_client: AsyncQdrantClient | None = async_qdrant_client
        self._sync_client: QdrantClient | None = None
        self._vector_store: QdrantVectorStore | None = None
        self._dense_embeddings: FastEmbedEmbeddings | None = None
        self._sparse_embeddings: FastEmbedSparseType | None = None
        self._embedding_dimension: int | None = None
        self._dense_model_name = self._resolve_dense_model()
        self._sparse_model_name = self._resolve_sparse_model()
        self._retrieval_mode: SearchStrategy = self._resolve_retrieval_mode()

    def is_initialized(self) -> bool:
        """Return True when a vector store has been constructed."""

        return self._vector_store is not None

    async def initialize(self) -> None:
        """Initialize Qdrant clients and embeddings."""

        if self.is_initialized():
            return

        cfg = self._require_qdrant_config()
        dense_embedding = FastEmbedEmbeddings(model_name=self._dense_model_name)
        probe_vector = await asyncio.to_thread(dense_embedding.embed_query, "__probe__")
        self._embedding_dimension = len(probe_vector)
        sparse_embedding: FastEmbedSparseType | None = None
        if self._retrieval_mode in {SearchStrategy.SPARSE, SearchStrategy.HYBRID}:
            if not self._sparse_model_name:
                msg = "Sparse or hybrid retrieval requires a sparse embedding model"
                raise EmbeddingServiceError(msg)
            if FastEmbedSparseRuntime is None:
                msg = (
                    "langchain-qdrant extras are required for sparse retrieval; "
                    "install with `uv add langchain-qdrant[fastembed]`"
                )
                raise EmbeddingServiceError(msg)
            sparse_embedding = FastEmbedSparseRuntime(
                model_name=self._sparse_model_name
            )
        self._dense_embeddings = dense_embedding
        self._sparse_embeddings = sparse_embedding
        retrieval_mode = _RETRIEVAL_MODE_MAP.get(
            self._retrieval_mode, RetrievalMode.DENSE
        )
        self._sync_client = self._build_sync_client(cfg)
        self.collection_name = getattr(cfg, "collection_name", None)
        self._vector_store = QdrantVectorStore(
            client=self._sync_client,
            collection_name=cfg.collection_name,
            embedding=self._dense_embeddings,
            retrieval_mode=retrieval_mode,
            sparse_embedding=sparse_embedding,
        )
        logger.info("VectorStoreService initialized via LangChain QdrantVectorStore")

    async def cleanup(self) -> None:
        """Release Qdrant clients and embeddings."""

        self._vector_store = None
        self._sync_client = None
        self._async_client = None
        self._dense_embeddings = None
        self._sparse_embeddings = None
        self._embedding_dimension = None

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the dense embeddings."""

        if self._embedding_dimension is None:
            msg = "FastEmbed embeddings have not been initialized"
            raise EmbeddingServiceError(msg)
        return self._embedding_dimension

    async def ensure_collection(self, schema: CollectionSchema) -> None:
        """Ensure a collection with the supplied schema exists."""

        client = self._require_async_client()
        if await client.collection_exists(schema.name):
            return
        dense_name = getattr(self._vector_store, "vector_name", "") or ""
        dense_params = models.VectorParams(
            size=self.embedding_dimension,
            distance=_distance_from_string(schema.distance),
        )
        if dense_name:
            vectors_config: models.VectorParams | dict[str, models.VectorParams] = {
                dense_name: dense_params
            }
        else:
            vectors_config = dense_params
        sparse_config = None
        if schema.requires_sparse:
            sparse_name = (
                getattr(self._vector_store, "sparse_vector_name", "langchain-sparse")
                or "langchain-sparse"
            )
            sparse_config = {
                sparse_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(),
                )
            }
        await client.create_collection(
            collection_name=schema.name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

    async def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""

        client = self._require_async_client()
        await client.delete_collection(name)

    async def list_collections(self) -> list[str]:
        """Return the identifiers for all collections."""

        client = self._require_async_client()
        response = await client.get_collections()
        return [collection.name for collection in response.collections]

    async def get_collection_info(self, name: str) -> Mapping[str, Any]:
        """Fetch raw collection metadata."""

        client = self._require_async_client()
        info = await client.get_collection(collection_name=name)
        return _serialize_collection_info(info)

    async def get_payload_index_summary(self, name: str) -> Mapping[str, Any]:
        """Return a payload index summary for the supplied collection."""

        info = await self.get_collection_info(name)
        payload_schema = info.get("payload_schema", {})
        indexed_fields = sorted(payload_schema.keys())
        return {
            "indexed_fields_count": len(indexed_fields),
            "indexed_fields": indexed_fields,
            "payload_schema": payload_schema,
            "points_count": info.get("points_count", 0),
        }

    async def ensure_payload_indexes(
        self,
        name: str,
        definitions: Mapping[str, models.PayloadSchemaType],
    ) -> Mapping[str, Any]:
        """Ensure payload indexes with the requested schemas exist."""

        client = self._require_async_client()
        summary = await self.get_payload_index_summary(name)
        existing_schema: Mapping[str, Mapping[str, Any]] = summary.get(
            "payload_schema", {}
        )
        for field, schema in definitions.items():
            if not _schema_matches(existing_schema.get(field), schema):
                await client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
        return await self.get_payload_index_summary(name)

    async def drop_payload_indexes(self, name: str, fields: Iterable[str]) -> None:
        """Drop payload indexes for the given fields if present."""

        client = self._require_async_client()
        summary = await self.get_payload_index_summary(name)
        existing_fields = set(summary.get("indexed_fields", []))
        for field in fields:
            if field in existing_fields:
                await client.delete_payload_index(
                    collection_name=name,
                    field_name=field,
                    wait=True,
                )

    async def collection_stats(self, name: str) -> Mapping[str, Any]:
        """Return statistics for a collection."""

        client = self._require_async_client()
        info = await client.get_collection(collection_name=name)
        return _serialize_collection_info(info)

    async def add_document(
        self,
        collection: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add a single document and return its identifier."""

        document_id = str(uuid4())
        normalized_metadata = dict(metadata or {})
        normalized_metadata.setdefault("doc_id", document_id)
        normalized_metadata.setdefault(
            "chunk_id", normalized_metadata.get("chunk_index", 0)
        )
        normalized_metadata.setdefault(
            "tenant", normalized_metadata.get("tenant") or "default"
        )
        normalized_metadata.setdefault(
            "source",
            normalized_metadata.get("source")
            or normalized_metadata.get("url")
            or "inline",
        )
        normalized_metadata.setdefault("created_at", datetime.now(UTC).isoformat())

        await self.upsert_documents(
            collection,
            [
                TextDocument(
                    id=document_id,
                    content=content,
                    metadata=normalized_metadata,
                )
            ],
        )
        return document_id

    async def upsert_documents(
        self,
        collection: str,
        documents: Sequence[TextDocument | Document],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Upsert a batch of documents via LangChain vector store."""

        if not documents:
            return

        await self.ensure_collection(
            CollectionSchema(
                name=collection,
                vector_size=self.embedding_dimension,
                requires_sparse=(
                    self._sparse_embeddings is not None
                    and self._retrieval_mode
                    in {SearchStrategy.SPARSE, SearchStrategy.HYBRID}
                ),
            )
        )

        store = self._require_vector_store(collection)
        normalized_documents: list[TextDocument] = []
        for document in documents:
            if isinstance(document, Document):
                metadata = dict(document.metadata or {})
                identifier = str(
                    metadata.get("doc_id")
                    or metadata.get("id")
                    or getattr(document, "id", uuid4().hex)
                )
                normalized_documents.append(
                    TextDocument(
                        id=identifier,
                        content=document.page_content,
                        metadata=metadata,
                    )
                )
            else:
                normalized_documents.append(
                    TextDocument(
                        id=document.id,
                        content=document.content,
                        metadata=dict(document.metadata or {}),
                    )
                )

        langchain_documents: list[Document] = []
        canonical_payloads: list[CanonicalPayload] = []
        for document in normalized_documents:
            try:
                payload = ensure_canonical_payload(
                    document.metadata,
                    content=document.content,
                    id_hint=document.id,
                )
            except PayloadValidationError as exc:  # pragma: no cover - defensive
                msg = f"Invalid payload for document '{document.id}': {exc}"
                raise EmbeddingServiceError(msg) from exc
            canonical_payloads.append(payload)
            metadata = dict(payload.payload)
            content = metadata.get("content", document.content)
            langchain_documents.append(
                Document(page_content=content, metadata=metadata)
            )

        ids = [payload.point_id for payload in canonical_payloads]

        await asyncio.to_thread(
            store.add_documents,
            documents=langchain_documents,
            ids=ids,
        )

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        """Delete points by identifiers or filter."""

        client = self._require_async_client()
        if ids:
            await client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=list(ids)),
            )
            return
        if filters:
            filter_obj = _filter_from_mapping(filters)
            if filter_obj is not None:
                await client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=filter_obj),
                )

    async def upsert_vectors(
        self,
        collection: str,
        records: Sequence[VectorRecord],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Insert or update pre-embedded vectors."""

        if not records:
            return

        await self.ensure_collection(
            CollectionSchema(
                name=collection,
                vector_size=self.embedding_dimension,
                requires_sparse=any(record.sparse_vector for record in records),
            )
        )

        client = self._require_async_client()
        points: list[models.PointStruct] = []
        for record in records:
            dense_vector = list(record.vector)
            vector_payload: Any = dense_vector
            if record.sparse_vector:
                vector_payload = {
                    "default": dense_vector,
                    "sparse": models.SparseVector(
                        indices=list(record.sparse_vector.keys()),
                        values=list(record.sparse_vector.values()),
                    ),
                }
            points.append(
                models.PointStruct(
                    id=record.id,
                    vector=vector_payload,
                    payload=dict(record.payload or {}),
                )
            )

        await client.upsert(
            collection_name=collection,
            points=points,
        )

    async def get_document(
        self,
        collection: str,
        document_id: str,
    ) -> Mapping[str, Any] | None:
        """Fetch a document payload by identifier."""

        client = self._require_async_client()
        records = await client.retrieve(
            collection_name=collection,
            ids=[document_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = dict(records[0].payload or {})
        payload.setdefault("id", document_id)
        return payload

    async def delete_document(self, collection: str, document_id: str) -> bool:
        """Delete a document by identifier."""

        before = await self.get_document(collection, document_id)
        if before is None:
            return False
        await self.delete(collection, ids=[document_id])
        return True

    async def list_documents(
        self,
        collection: str,
        *,
        limit: int,
        offset: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """List documents with pagination support."""

        client = self._require_async_client()
        points, next_offset = await client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        documents = [dict(point.payload or {}) for point in points]
        for point, payload in zip(points, documents, strict=False):
            payload.setdefault("id", str(point.id))
        next_token = str(next_offset) if next_offset is not None else None
        return documents, next_token

    async def embed_query(self, query: str) -> Sequence[float]:
        """Generate an embedding for the supplied query."""

        try:
            if self._dense_embeddings is None:
                msg = "FastEmbed embeddings have not been initialized"
                raise EmbeddingServiceError(msg)
            embedding = await asyncio.to_thread(
                self._dense_embeddings.embed_query,
                query,
            )
        except Exception as exc:  # pragma: no cover - provider-specific failures
            msg = f"Failed to embed query: {exc}"
            raise EmbeddingServiceError(msg) from exc
        return embedding

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for documents."""

        try:
            if self._dense_embeddings is None:
                msg = "FastEmbed embeddings have not been initialized"
                raise EmbeddingServiceError(msg)
            return await asyncio.to_thread(
                self._dense_embeddings.embed_documents,
                list(texts),
            )
        except Exception as exc:  # pragma: no cover - provider-specific failures
            msg = f"Failed to embed documents: {exc}"
            raise EmbeddingServiceError(msg) from exc

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
        group_by: str | None = None,
        group_size: int | None = None,
        overfetch_multiplier: float | None = None,
        normalize_scores: bool | None = None,
    ) -> list[SearchRecord]:  # pylint: disable=too-many-arguments
        """Execute a dense similarity search with optional grouping."""

        vector = await self.embed_query(query)
        records, grouping_applied = await self._query_with_optional_grouping(
            collection,
            vector,
            limit=limit,
            group_by=group_by,
            group_size=group_size or 1,
            filters=filters,
            overfetch_multiplier=overfetch_multiplier,
        )
        records = self._annotate_grouping_metadata(
            records,
            group_by=group_by,
            grouping_applied=grouping_applied,
        )
        return self._normalize_scores(records, enabled=normalize_scores)

    async def search_vector(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchRecord]:
        """Perform a similarity search using a precomputed vector."""

        records, _ = await self._query_with_optional_grouping(
            collection,
            vector,
            limit=limit,
            group_by=None,
            group_size=1,
            filters=filters,
            overfetch_multiplier=None,
        )
        return records

    async def hybrid_search(
        self,
        collection: str,
        query: str | None = None,
        *,
        dense_vector: Sequence[float] | None = None,
        sparse_vector: Mapping[int, float] | None = None,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchRecord]:  # pylint: disable=too-many-arguments
        """Perform a hybrid search over dense and sparse representations."""

        dense_payload = dense_vector
        sparse_payload_mapping = sparse_vector
        mode = self._retrieval_mode

        if query is not None:
            if mode in {SearchStrategy.DENSE, SearchStrategy.HYBRID}:
                dense_payload = await self.embed_query(query)
            if mode in {SearchStrategy.SPARSE, SearchStrategy.HYBRID} and (
                self._sparse_embeddings is not None
            ):
                sparse_payload = await asyncio.to_thread(
                    self._sparse_embeddings.embed_query,
                    query,
                )
                sparse_payload_mapping = dict(
                    zip(sparse_payload.indices, sparse_payload.values, strict=False)
                )

        if mode is SearchStrategy.DENSE:
            if dense_payload is None:
                msg = "Dense retrieval requires a query or dense vector"
                raise EmbeddingServiceError(msg)
            return await self.search_vector(
                collection,
                dense_payload,
                limit=limit,
                filters=filters,
            )

        client = self._require_async_client()
        store = self._require_vector_store(collection)
        query_filter = _filter_from_mapping(filters)
        sparse_name = getattr(store, "sparse_vector_name", "langchain-sparse")

        if mode is SearchStrategy.SPARSE:
            if not sparse_payload_mapping:
                msg = "Sparse retrieval requires a sparse vector"
                raise EmbeddingServiceError(msg)
            sparse_query = models.SparseVector(
                indices=list(sparse_payload_mapping.keys()),
                values=list(sparse_payload_mapping.values()),
            )
            result = await client.query_points(
                collection_name=collection,
                query=sparse_query,
                using=sparse_name or "langchain-sparse",
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [
                _scored_point_to_record(collection, point)
                for point in getattr(result, "points", []) or []
            ]

        if dense_payload is None:
            msg = "Hybrid retrieval requires a dense vector"
            raise EmbeddingServiceError(msg)
        if not sparse_payload_mapping:
            return await self.search_vector(
                collection,
                dense_payload,
                limit=limit,
                filters=filters,
            )

        sparse_query = models.SparseVector(
            indices=list(sparse_payload_mapping.keys()),
            values=list(sparse_payload_mapping.values()),
        )
        dense_name = getattr(store, "vector_name", "") or None
        prefetch = [
            models.Prefetch(
                query=list(dense_payload),
                using=dense_name,
                filter=query_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=sparse_query,
                using=sparse_name or "langchain-sparse",
                filter=query_filter,
                limit=limit,
            ),
        ]
        result = await client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            _scored_point_to_record(collection, point)
            for point in getattr(result, "points", []) or []
        ]

    async def recommend(
        self,
        collection: str,
        *,
        positive_ids: Sequence[str] | None = None,
        vector: Sequence[float] | None = None,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchRecord]:
        """Return records related to supplied positive examples."""

        if not positive_ids and vector is None:
            msg = "`positive_ids` or `vector` must be provided for recommend"
            raise ValueError(msg)

        client = self._require_async_client()
        recommend_kwargs: dict[str, Any] = {
            "collection_name": collection,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
            "query_filter": _filter_from_mapping(filters),
        }
        if positive_ids:
            recommend_kwargs["positive"] = list(positive_ids)
        if vector is not None:
            recommend_kwargs["query_vector"] = list(vector)
        if recommend_kwargs["query_filter"] is None:
            recommend_kwargs.pop("query_filter")

        results = await client.recommend(**recommend_kwargs)
        return [_scored_point_to_record(collection, point) for point in results]

    # ------------------------------------------------------------------
    # Internal helpers

    def _require_async_client(self) -> AsyncQdrantClient:
        """Return the async Qdrant client."""

        if self._async_client is None:
            msg = "VectorStoreService not initialized"
            raise RuntimeError(msg)
        return self._async_client

    def _require_vector_store(self, collection: str) -> QdrantVectorStore:
        """Return the vector store for the collection."""

        if self._vector_store is None:
            msg = "VectorStoreService not initialized"
            raise RuntimeError(msg)
        # LangChain's vector store keeps the collection name; override if needed.
        self._vector_store.collection_name = collection
        return self._vector_store

    def _require_qdrant_config(self) -> Any:
        """Return the Qdrant configuration."""

        cfg = getattr(self.config, "qdrant", None)
        if cfg is None:
            msg = "Qdrant configuration missing"
            raise EmbeddingServiceError(msg)
        return cfg

    def _resolve_retrieval_mode(self) -> SearchStrategy:
        """Determine the retrieval mode based on settings and provider."""

        embedding_cfg = getattr(self.config, "embedding", None)
        mode = getattr(embedding_cfg, "retrieval_mode", None)
        if isinstance(mode, SearchStrategy):
            return mode
        if hasattr(self.config, "get_effective_search_strategy"):
            return cast(SearchStrategy, self.config.get_effective_search_strategy())
        return SearchStrategy.DENSE

    def _resolve_dense_model(self) -> str:
        """Return the configured dense embedding model identifier."""

        embedding_cfg = getattr(self.config, "embedding", None)
        candidate = getattr(embedding_cfg, "dense_model", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        fastembed_cfg = getattr(self.config, "fastembed", None)
        return str(getattr(fastembed_cfg, "dense_model", "BAAI/bge-small-en-v1.5"))

    def _resolve_sparse_model(self) -> str | None:
        """Return the configured sparse embedding model identifier, if any."""

        embedding_cfg = getattr(self.config, "embedding", None)
        candidate = getattr(embedding_cfg, "sparse_model", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        fastembed_cfg = getattr(self.config, "fastembed", None)
        fallback = getattr(fastembed_cfg, "sparse_model", None)
        return str(fallback) if isinstance(fallback, str) and fallback else None

    async def _query_with_optional_grouping(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        group_by: str | None,
        group_size: int,
        filters: Mapping[str, Any] | None,
        overfetch_multiplier: float | None,
    ) -> tuple[list[SearchRecord], bool]:  # pylint: disable=too-many-arguments,too-many-locals
        """Query with optional grouping support."""

        cfg = self._require_qdrant_config()
        grouping_enabled = bool(group_by) and bool(
            getattr(cfg, "enable_grouping", False)
        )

        if grouping_enabled and group_by:
            records, applied = await self._query_with_server_grouping(
                collection,
                vector,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
                filters=filters,
            )
            if applied:
                set_span_attributes(
                    {
                        "qdrant.grouping.status": "applied",
                        "qdrant.grouping.collection": collection,
                    }
                )
                return records, True
            set_span_attributes(
                {
                    "qdrant.grouping.status": "fallback",
                    "qdrant.grouping.collection": collection,
                }
            )

        fetch_limit = int(limit * (overfetch_multiplier or 2.0))
        store = self._require_vector_store(collection)
        vector_filter = _filter_from_mapping(filters)
        to_thread_kwargs: dict[str, Any] = {
            "vector": list(vector),
            "k": fetch_limit,
        }
        if vector_filter is not None:
            to_thread_kwargs["filter"] = vector_filter
        documents_with_scores = await asyncio.to_thread(
            store.similarity_search_with_score_by_vector,
            **to_thread_kwargs,
        )
        records = [
            _document_to_record(collection, document, score)
            for document, score in documents_with_scores
        ]

        if grouping_enabled and group_by:
            records = self._group_client_side(
                records,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
            )
            return records, False

        set_span_attributes(
            {
                "qdrant.grouping.status": "disabled",
                "qdrant.grouping.collection": collection,
            }
        )
        return records[:limit], False

    async def _query_with_server_grouping(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        group_by: str,
        group_size: int,
        limit: int,
        filters: Mapping[str, Any] | None,
    ) -> tuple[list[SearchRecord], bool]:  # pylint: disable=too-many-arguments
        """Query with server-side grouping."""

        client = self._require_async_client()
        cfg = self._require_qdrant_config()
        if not getattr(cfg, "enable_grouping", False):
            return [], False

        query_filter = _filter_from_mapping(filters)

        try:
            response = await client.query_points_groups(
                collection_name=collection,
                group_by=group_by,
                query=list(vector),
                limit=limit,
                group_size=group_size,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # pragma: no cover - driver-specific fallbacks
            return [], False

        records: list[SearchRecord] = []
        for group in getattr(response, "groups", []) or []:
            hits = getattr(group, "hits", [])
            if not hits:
                continue
            hit = hits[0]
            payload: dict[str, Any] = dict(hit.payload or {})
            payload["_grouping"] = {
                "applied": True,
                "group_id": getattr(group, "id", None),
            }
            records.append(
                SearchRecord.from_payload(
                    {
                        "id": str(hit.id),
                        "content": (
                            payload.get("content")
                            or payload.get("page_content")
                            or payload.get("text")
                            or ""
                        ),
                        "score": float(hit.score),
                        "raw_score": float(hit.score),
                        "metadata": payload,
                        "collection": collection,
                    }
                )
            )
        return records, bool(records)

    def _group_client_side(
        self,
        records: list[SearchRecord],
        *,
        group_by: str,
        group_size: int,
        limit: int,
    ) -> list[SearchRecord]:
        """Group matches client-side."""

        groups: dict[str, list[SearchRecord]] = {}
        for record in records:
            metadata: dict[str, Any] = dict(record.metadata or {})
            group_id = metadata.get(group_by) or metadata.get("doc_id")
            if group_id is None:
                group_id = record.id
            metadata["doc_id"] = group_id
            record.metadata = metadata
            groups.setdefault(str(group_id), []).append(record)

        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (item[1][0].raw_score or item[1][0].score),
            reverse=True,
        )

        limited_records: list[SearchRecord] = []
        for _, group_matches in ordered_groups:
            for group_rank, record in enumerate(group_matches[:group_size], start=1):
                metadata = dict(record.metadata or {})
                group_id = metadata.get(group_by) or metadata.get("doc_id") or record.id
                metadata["_grouping"] = {
                    "applied": False,
                    "group_id": group_id,
                    "rank": group_rank,
                }
                metadata["doc_id"] = group_id
                record.metadata = metadata
                record.group_id = group_id
                record.group_rank = group_rank
                record.grouping_applied = False
                limited_records.append(record)
            if len(limited_records) >= limit:
                break
        return limited_records

    def _annotate_grouping_metadata(
        self,
        records: list[SearchRecord],
        *,
        group_by: str | None,
        grouping_applied: bool,
    ) -> list[SearchRecord]:
        """Annotate matches with grouping metadata."""

        if not group_by:
            return records
        for rank, record in enumerate(records, start=1):
            metadata: dict[str, Any] = dict(record.metadata or {})
            group_info: dict[str, Any] = dict(metadata.get("_grouping") or {})
            group_info["group_id"] = (
                metadata.get(group_by) or metadata.get("doc_id") or record.id
            )
            group_info["rank"] = rank
            group_info["applied"] = grouping_applied
            metadata["_grouping"] = group_info
            record.metadata = metadata
            record.group_id = group_info["group_id"]
            record.group_rank = rank
            record.grouping_applied = grouping_applied
        return records

    def _normalize_scores(
        self, records: list[SearchRecord], *, enabled: bool | None
    ) -> list[SearchRecord]:  # pylint: disable=too-many-branches,too-many-return-statements
        """Normalize match scores."""

        if not records:
            return records

        for record in records:
            if record.raw_score is None:
                record.raw_score = float(record.score)

        if not enabled:
            return records

        query_cfg: QueryProcessingConfig | None = getattr(
            self.config, "query_processing", None
        )
        strategy = (
            query_cfg.score_normalization_strategy
            if query_cfg is not None
            else ScoreNormalizationStrategy.MIN_MAX
        )
        if strategy == ScoreNormalizationStrategy.NONE:
            return records

        scores = [float(record.raw_score or record.score) for record in records]
        epsilon = max(
            float(
                (query_cfg.score_normalization_epsilon if query_cfg else 1e-6) or 1e-6
            ),
            1e-9,
        )

        if strategy == ScoreNormalizationStrategy.MIN_MAX:
            minimum = min(scores)
            maximum = max(scores)
            span = maximum - minimum
            if span < epsilon:
                for record in records:
                    record.score = 1.0
                    record.normalized_score = 1.0
                return records
            for record in records:
                normalized = ((record.raw_score or record.score) - minimum) / span
                record.score = normalized
                record.normalized_score = normalized
            return records

        if strategy == ScoreNormalizationStrategy.Z_SCORE:
            mean = statistics.fmean(scores)
            std_dev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            if std_dev < epsilon:
                for record in records:
                    record.score = 0.0
                    record.normalized_score = 0.0
                return records
            for record in records:
                normalized = ((record.raw_score or record.score) - mean) / std_dev
                record.score = normalized
                record.normalized_score = normalized
            return records

        return records

    def _build_sync_client(self, cfg: Any) -> QdrantClient:
        """Build the synchronous Qdrant client."""

        timeout = getattr(cfg, "timeout", 30)
        return QdrantClient(
            url=str(getattr(cfg, "url", "http://localhost:6333")),
            api_key=getattr(cfg, "api_key", None),
            timeout=int(timeout) if timeout is not None else None,
            prefer_grpc=bool(
                getattr(cfg, "prefer_grpc", False) or getattr(cfg, "use_grpc", False)
            ),
            grpc_port=int(getattr(cfg, "grpc_port", 6334)),
        )


# ----------------------------------------------------------------------
# Helper utilities


def _document_to_record(
    collection: str,
    document: Document,
    score: float,
) -> SearchRecord:
    """Convert a LangChain document into a canonical search record."""

    metadata: dict[str, Any] = dict(document.metadata or {})
    metadata.setdefault("page_content", document.page_content)
    identifier = (
        metadata.get("point_id")
        or metadata.get("doc_id")
        or getattr(document, "id", None)
        or uuid4().hex
    )
    record_payload = {
        "id": str(identifier),
        "content": (
            metadata.get("content")
            or metadata.get("page_content")
            or document.page_content
            or ""
        ),
        "score": float(score),
        "raw_score": float(score),
        "metadata": metadata,
        "collection": collection,
    }
    return SearchRecord.from_payload(record_payload)


def _scored_point_to_record(collection: str, point: Any) -> SearchRecord:
    """Convert a Qdrant scored point into a canonical search record."""

    payload: dict[str, Any] = dict(getattr(point, "payload", {}) or {})
    score = float(getattr(point, "score", 0.0) or 0.0)
    record_payload = {
        "id": str(getattr(point, "id", uuid4())),
        "content": (
            payload.get("content")
            or payload.get("page_content")
            or payload.get("text")
            or ""
        ),
        "score": score,
        "raw_score": score,
        "metadata": payload or None,
        "collection": collection,
    }
    return SearchRecord.from_payload(record_payload)


def _serialize_collection_info(info: Any) -> Mapping[str, Any]:
    """Serialize collection info."""

    config = getattr(info, "config", None)
    payload_schema = (
        cast(dict[str, Any], getattr(config, "payload_schema", {})) if config else {}
    )
    config_payload = (
        config.dict() if (config is not None and hasattr(config, "dict")) else {}
    )
    return {
        "points_count": getattr(info, "points_count", 0),
        "indexed_vectors": getattr(info, "indexed_vectors_count", 0),
        "payload_schema": payload_schema,
        "config": config_payload,
    }


def _schema_matches(
    existing: Mapping[str, Any] | None, schema: models.PayloadSchemaType
) -> bool:
    """Check if schema matches existing."""

    if not existing:
        return False
    return existing.get("type") == schema.value if isinstance(schema, Enum) else False


def _distance_from_string(name: str) -> models.Distance:
    """Convert distance name to enum."""

    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    return mapping.get(name.lower(), models.Distance.COSINE)


def _filter_from_mapping(filters: Mapping[str, Any] | None) -> models.Filter | None:
    """Convert mapping to Qdrant filter."""

    if not filters:
        return None
    must_conditions = []
    for key, value in filters.items():
        if isinstance(value, Mapping):
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    range=models.Range(**value),
                )
            )
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=list(value)),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(
                        value=cast("models.ValueVariants", value),
                    ),
                )
            )
    return models.Filter(must=must_conditions)
