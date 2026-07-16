"""Vector store service implemented on top of LangChain's Qdrant integration."""
# pylint: disable=too-many-arguments,too-many-return-statements,too-many-branches,too-many-locals,too-many-lines

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, cast
from uuid import uuid4

import grpc
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from src.config.loader import Settings
from src.config.models import (
    QueryProcessingConfig,
    ScoreNormalizationStrategy,
    SearchStrategy,
)
from src.contracts.retrieval import SearchRecord
from src.services.errors import EmbeddingServiceError
from src.services.observability.tracing import set_span_attributes

from .payload_schema import PayloadValidationError, normalize_document


logger = logging.getLogger(__name__)


_RETRIEVAL_MODE_MAP: dict[SearchStrategy, RetrievalMode] = {
    SearchStrategy.DENSE: RetrievalMode.DENSE,
    SearchStrategy.SPARSE: RetrievalMode.SPARSE,
    SearchStrategy.HYBRID: RetrievalMode.HYBRID,
}
_DENSE_VECTOR_NAME = QdrantVectorStore.VECTOR_NAME
_SPARSE_VECTOR_NAME = QdrantVectorStore.SPARSE_VECTOR_NAME


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
        self._async_client: AsyncQdrantClient | None = async_qdrant_client
        self._sync_client: QdrantClient | None = None
        self._vector_store: QdrantVectorStore | None = None
        self._vector_stores: dict[str, QdrantVectorStore] = {}
        self._dense_embeddings: FastEmbedEmbeddings | None = None
        self._sparse_embeddings: FastEmbedSparse | None = None
        self._embedding_dimension: int | None = None
        self._dense_model_name = config.fastembed.dense_model
        self._sparse_model_name = config.fastembed.sparse_model
        self._retrieval_mode = config.get_effective_search_strategy()
        # ponytail: production uses one writer process; add a distributed lock
        # before supporting replacement ingestion from multiple processes.
        self._replacement_lock = asyncio.Lock()

    def is_initialized(self) -> bool:
        """Return True when a vector store has been constructed."""
        return self._vector_store is not None

    @property
    def default_collection_name(self) -> str:
        """Return the configured collection used when callers omit one."""
        return str(self._require_qdrant_config().collection_name)

    async def initialize(self) -> None:
        """Initialize Qdrant clients and embeddings."""
        if self.is_initialized():
            return

        cfg = self._require_qdrant_config()
        dense_embedding = FastEmbedEmbeddings(
            model_name=self._dense_model_name,
            cache_dir=self.config.fastembed.cache_dir,
            max_length=self.config.fastembed.max_length,
            batch_size=self.config.fastembed.batch_size,
        )
        probe_vector = await asyncio.to_thread(dense_embedding.embed_query, "__probe__")
        self._embedding_dimension = len(probe_vector)
        sparse_embedding: FastEmbedSparse | None = None
        if self._retrieval_mode in {SearchStrategy.SPARSE, SearchStrategy.HYBRID}:
            if not self._sparse_model_name:
                msg = "Sparse or hybrid retrieval requires a sparse embedding model"
                raise EmbeddingServiceError(msg)
            sparse_embedding = FastEmbedSparse(
                model_name=self._sparse_model_name,
                cache_dir=self.config.fastembed.cache_dir,
                batch_size=self.config.fastembed.batch_size,
            )
        self._dense_embeddings = dense_embedding
        self._sparse_embeddings = sparse_embedding
        retrieval_mode = _RETRIEVAL_MODE_MAP.get(
            self._retrieval_mode, RetrievalMode.DENSE
        )
        self._sync_client = self._build_sync_client(cfg)
        await self.ensure_collection(cfg.collection_name)
        self._vector_store = QdrantVectorStore(
            client=self._sync_client,
            collection_name=cfg.collection_name,
            embedding=self._dense_embeddings,
            retrieval_mode=retrieval_mode,
            sparse_embedding=sparse_embedding,
        )
        self._vector_stores[cfg.collection_name] = self._vector_store
        logger.info("VectorStoreService initialized via LangChain QdrantVectorStore")

    async def cleanup(self) -> None:
        """Release Qdrant clients and embeddings."""
        sync_client = self._sync_client
        self._vector_store = None
        self._vector_stores.clear()
        self._sync_client = None
        self._async_client = None
        self._dense_embeddings = None
        self._sparse_embeddings = None
        self._embedding_dimension = None
        if sync_client is not None:
            await asyncio.to_thread(sync_client.close)

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the dense embeddings."""
        if self._embedding_dimension is None:
            msg = "FastEmbed embeddings have not been initialized"
            raise EmbeddingServiceError(msg)
        return self._embedding_dimension

    async def ensure_collection(self, name: str) -> None:
        """Ensure a canonical collection with the supplied name exists."""
        client = self._require_async_client()
        if await client.collection_exists(name):
            await self._validate_collection_contract(client, name)
            return
        dense_name = _DENSE_VECTOR_NAME
        dense_params = models.VectorParams(
            size=self.embedding_dimension,
            distance=models.Distance.COSINE,
        )
        if dense_name:
            vectors_config: models.VectorParams | dict[str, models.VectorParams] = {
                dense_name: dense_params
            }
        else:
            vectors_config = dense_params
        sparse_config = None
        if self._retrieval_mode in {SearchStrategy.SPARSE, SearchStrategy.HYBRID}:
            sparse_name = _SPARSE_VECTOR_NAME
            sparse_config = {
                sparse_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(),
                )
            }
        try:
            await client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config,
            )
            return
        except grpc.aio.AioRpcError as create_error:
            if create_error.code() is not grpc.StatusCode.ALREADY_EXISTS:
                raise
            await self._verify_concurrent_collection_creation(
                client,
                name,
                create_error,
            )
        except UnexpectedResponse as create_error:
            if create_error.status_code != 409:
                raise
            await self._verify_concurrent_collection_creation(
                client,
                name,
                create_error,
            )
        except ValueError as create_error:
            if str(create_error) != f"Collection {name} already exists":
                raise
            await self._verify_concurrent_collection_creation(
                client,
                name,
                create_error,
            )
        await self._validate_collection_contract(client, name)

    async def _validate_collection_contract(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
    ) -> None:
        """Reject collections that cannot store the configured retrieval vectors."""
        info = await client.get_collection(collection_name=collection_name)
        params = info.config.params
        vectors = params.vectors
        if _DENSE_VECTOR_NAME:
            dense_params = (
                vectors.get(_DENSE_VECTOR_NAME)
                if isinstance(vectors, Mapping)
                else None
            )
            dense_names = set(vectors) if isinstance(vectors, Mapping) else set()
            dense_names_match = dense_names == {_DENSE_VECTOR_NAME}
        else:
            dense_params = None if isinstance(vectors, Mapping) else vectors
            dense_names_match = not isinstance(vectors, Mapping)

        distance = getattr(dense_params, "distance", None)
        distance_value = getattr(distance, "value", distance)
        dense_matches = (
            dense_names_match
            and getattr(dense_params, "size", None) == self.embedding_dimension
            and distance_value == models.Distance.COSINE.value
        )

        sparse_vectors = params.sparse_vectors or {}
        sparse_names = (
            set(sparse_vectors) if isinstance(sparse_vectors, Mapping) else set()
        )
        expected_sparse_names = (
            {_SPARSE_VECTOR_NAME}
            if self._retrieval_mode in {SearchStrategy.SPARSE, SearchStrategy.HYBRID}
            else set()
        )
        if dense_matches and sparse_names == expected_sparse_names:
            return

        msg = (
            f"Collection '{collection_name}' does not match the canonical vector "
            f"contract (dense dimension {self.embedding_dimension}, cosine distance, "
            f"sparse vectors {sorted(expected_sparse_names)}). Clear the collection "
            "and fully re-ingest it before serving traffic."
        )
        raise EmbeddingServiceError(msg)

    @staticmethod
    async def _verify_concurrent_collection_creation(
        client: AsyncQdrantClient,
        collection_name: str,
        create_error: Exception,
    ) -> None:
        """Accept a create conflict only after proving the collection exists."""
        try:
            created_by_peer = await client.collection_exists(collection_name)
        # Verification is best-effort across local, HTTP, and gRPC clients; any
        # failure must preserve the original create error for callers.
        except Exception as verification_error:  # pylint: disable=broad-exception-caught
            raise create_error from verification_error
        if not created_by_peer:
            raise create_error
        set_span_attributes({"qdrant.collection.concurrent_creation": True})
        logger.debug(
            "Collection '%s' was created by a concurrent initializer",
            collection_name,
        )

    async def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        client = self._require_async_client()
        await client.delete_collection(name)
        self._vector_stores.pop(name, None)

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
            stored_field = _metadata_field(field)
            if not _schema_matches(existing_schema.get(stored_field), schema):
                await client.create_payload_index(
                    collection_name=name,
                    field_name=stored_field,
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
            stored_field = _metadata_field(field)
            if stored_field in existing_fields:
                await client.delete_payload_index(
                    collection_name=name,
                    field_name=stored_field,
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
        point_ids = await self.upsert_documents(
            collection,
            [
                Document(
                    id=document_id,
                    page_content=content,
                    metadata=dict(metadata or {}),
                )
            ],
        )
        return point_ids[0]

    async def upsert_documents(
        self,
        collection: str,
        documents: Sequence[Document],
    ) -> list[str]:
        """Upsert a batch of documents via LangChain vector store."""
        ids, _ = await self._persist_documents(collection, documents)
        return ids

    async def replace_document_chunks(
        self,
        collection: str,
        documents: Sequence[Document],
    ) -> list[str]:
        """Persist a complete trusted chunk set and prune its obsolete tail."""
        if not documents:
            msg = "A complete document replacement requires at least one chunk"
            raise EmbeddingServiceError(msg)
        async with self._replacement_lock:
            ids, canonical_documents = await self._persist_documents(
                collection,
                documents,
            )
            await self._prune_replaced_document_tails(collection, canonical_documents)
            return ids

    async def _persist_documents(
        self,
        collection: str,
        documents: Sequence[Document],
    ) -> tuple[list[str], list[Document]]:
        """Normalize and persist documents without inferring replacement intent."""
        if not documents:
            return [], []

        await self.ensure_collection(collection)

        store = self._require_vector_store(collection)
        langchain_documents: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata or {})
            id_hint = str(metadata.get("doc_id") or document.id or uuid4())
            try:
                canonical_document = normalize_document(document, id_hint=id_hint)
            except PayloadValidationError as exc:  # pragma: no cover - defensive
                msg = f"Invalid payload for document '{id_hint}': {exc}"
                raise EmbeddingServiceError(msg) from exc
            langchain_documents.append(canonical_document)

        ids = [str(document.id) for document in langchain_documents]
        if len(ids) != len(set(ids)):
            msg = (
                "Document batch contains duplicate tenant, doc_id, and chunk_index keys"
            )
            raise EmbeddingServiceError(msg)

        await asyncio.to_thread(
            store.add_documents,
            documents=langchain_documents,
            ids=ids,
            wait=True,
        )
        return ids, langchain_documents

    async def _prune_replaced_document_tails(
        self,
        collection: str,
        documents: Sequence[Document],
    ) -> None:
        """Remove obsolete trailing chunks after a complete document replacement."""
        chunk_sets: dict[tuple[str, str, int], set[int]] = {}
        for document in documents:
            metadata = document.metadata
            total_chunks = metadata.get("total_chunks")
            chunk_index = metadata.get("chunk_index")
            if (
                not isinstance(total_chunks, int)
                or isinstance(total_chunks, bool)
                or total_chunks < 1
                or not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
            ):
                continue
            key = (
                str(metadata["tenant"]),
                str(metadata["doc_id"]),
                total_chunks,
            )
            chunk_sets.setdefault(key, set()).add(chunk_index)

        client = self._require_async_client()
        for (tenant, doc_id, total_chunks), indexes in chunk_sets.items():
            if indexes != set(range(total_chunks)):
                continue
            await client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.tenant",
                                match=models.MatchValue(value=tenant),
                            ),
                            models.FieldCondition(
                                key="metadata.doc_id",
                                match=models.MatchValue(value=doc_id),
                            ),
                            models.FieldCondition(
                                key="metadata.chunk_index",
                                range=models.Range(gte=total_chunks),
                            ),
                        ]
                    )
                ),
                wait=True,
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
        return _point_payload_to_document(
            dict(records[0].payload or {}),
            point_id=str(records[0].id),
        )

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
        documents = [
            _point_payload_to_document(
                dict(point.payload or {}),
                point_id=str(point.id),
            )
            for point in points
        ]
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
        sparse_name = getattr(store, "sparse_vector_name", _SPARSE_VECTOR_NAME)

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
                using=sparse_name or _SPARSE_VECTOR_NAME,
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
        dense_name = getattr(store, "vector_name", _DENSE_VECTOR_NAME) or None
        prefetch = [
            models.Prefetch(
                query=list(dense_payload),
                using=dense_name,
                filter=query_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=sparse_query,
                using=sparse_name or _SPARSE_VECTOR_NAME,
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
        positive: list[Any] = []
        if positive_ids:
            positive.extend(list(positive_ids))
        if vector is not None:
            positive.append(list(vector))

        query_filter = _filter_from_mapping(filters)
        response = await client.query_points(
            collection_name=collection,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=positive,
                )
            ),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            _scored_point_to_record(collection, point)
            for point in getattr(response, "points", []) or []
        ]

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
        if (
            self._vector_store is None
            or self._sync_client is None
            or self._dense_embeddings is None
        ):
            msg = "VectorStoreService not initialized"
            raise RuntimeError(msg)
        store = self._vector_stores.get(collection)
        if store is None:
            store = QdrantVectorStore(
                client=self._sync_client,
                collection_name=collection,
                embedding=self._dense_embeddings,
                retrieval_mode=_RETRIEVAL_MODE_MAP[self._retrieval_mode],
                sparse_embedding=self._sparse_embeddings,
                validate_collection_config=False,
            )
            self._vector_stores[collection] = store
        return store

    def _require_qdrant_config(self) -> Any:
        """Return the Qdrant configuration."""
        cfg = getattr(self.config, "qdrant", None)
        if cfg is None:
            msg = "Qdrant configuration missing"
            raise EmbeddingServiceError(msg)
        return cfg

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
            "embedding": list(vector),
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
                group_by=_metadata_field(group_by),
                query=list(vector),
                limit=limit,
                group_size=group_size,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        except (
            ApiException,
            UnexpectedResponse,
            ResponseHandlingException,
        ):  # pragma: no cover - qdrant client exceptions
            return [], False

        records: list[SearchRecord] = []
        for group in getattr(response, "groups", []) or []:
            for rank, hit in enumerate(
                (getattr(group, "hits", []) or [])[:group_size],
                start=1,
            ):
                payload: dict[str, Any] = dict(hit.payload or {})
                content, metadata = _unpack_native_payload(payload)
                metadata["_grouping"] = {
                    "applied": True,
                    "group_id": getattr(group, "id", None),
                    "rank": rank,
                }
                records.append(
                    SearchRecord.from_payload(
                        {
                            "id": str(hit.id),
                            "content": content,
                            "score": float(hit.score),
                            "raw_score": float(hit.score),
                            "metadata": metadata,
                            "collection": collection,
                        }
                    )
                )
                if len(records) == limit:
                    return records, True
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
            group_id = metadata.get(group_by)
            if group_id is None:
                group_id = metadata.get("doc_id")
            if group_id is None:
                group_id = record.id
            group_id = str(group_id)
            record.metadata = metadata
            groups.setdefault(group_id, []).append(record)

        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (item[1][0].raw_score or item[1][0].score),
            reverse=True,
        )

        limited_records: list[SearchRecord] = []
        for _, group_matches in ordered_groups:
            for group_rank, record in enumerate(group_matches[:group_size], start=1):
                metadata = dict(record.metadata or {})
                group_id = metadata.get(group_by)
                if group_id is None:
                    group_id = metadata.get("doc_id")
                if group_id is None:
                    group_id = record.id
                group_id = str(group_id)
                metadata["_grouping"] = {
                    "applied": False,
                    "group_id": group_id,
                    "rank": group_rank,
                }
                record.metadata = metadata
                record.group_id = group_id
                record.group_rank = group_rank
                record.grouping_applied = False
                limited_records.append(record)
                if len(limited_records) == limit:
                    return limited_records
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
        for fallback_rank, record in enumerate(records, start=1):
            metadata: dict[str, Any] = dict(record.metadata or {})
            group_info: dict[str, Any] = dict(metadata.get("_grouping") or {})
            group_id = group_info.get("group_id")
            if group_id is None:
                group_id = record.group_id
            if group_id is None:
                group_id = metadata.get(group_by)
            if group_id is None:
                group_id = metadata.get("doc_id")
            if group_id is None:
                group_id = record.id
            group_id = str(group_id)
            group_rank = group_info.get("rank") or record.group_rank or fallback_rank
            group_info["group_id"] = group_id
            group_info["rank"] = group_rank
            group_info["applied"] = grouping_applied
            metadata["_grouping"] = group_info
            record.metadata = metadata
            record.group_id = group_id
            record.group_rank = group_rank
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
    identifier = metadata.pop("_id")
    metadata.pop("_collection_name")
    record_payload = {
        "id": str(identifier),
        "content": document.page_content,
        "score": float(score),
        "raw_score": float(score),
        "metadata": metadata,
        "collection": collection,
    }
    return SearchRecord.from_payload(record_payload)


def _scored_point_to_record(collection: str, point: Any) -> SearchRecord:
    """Convert a Qdrant scored point into a canonical search record."""
    payload: dict[str, Any] = dict(getattr(point, "payload", {}) or {})
    content, metadata = _unpack_native_payload(payload)
    score = float(getattr(point, "score", 0.0) or 0.0)
    record_payload = {
        "id": str(point.id),
        "content": content,
        "score": score,
        "raw_score": score,
        "metadata": metadata,
        "collection": collection,
    }
    return SearchRecord.from_payload(record_payload)


def _unpack_native_payload(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Decode QdrantVectorStore's dependency-native payload shape."""
    raw_metadata = payload["metadata"]
    if not isinstance(raw_metadata, Mapping):
        msg = "QdrantVectorStore metadata payload must be a mapping"
        raise TypeError(msg)
    content = payload["page_content"]
    if not isinstance(content, str):
        msg = "QdrantVectorStore page_content payload must be a string"
        raise TypeError(msg)
    return content, dict(raw_metadata)


def _point_payload_to_document(
    payload: Mapping[str, Any],
    *,
    point_id: str,
) -> dict[str, Any]:
    """Flatten one native Qdrant payload for the public document contract."""
    content, metadata = _unpack_native_payload(payload)
    return {**metadata, "id": point_id, "content": content}


def dense_vector_config(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dense vector configuration from serialized Qdrant stats."""
    config = stats.get("config")
    if not isinstance(config, Mapping):
        return {}
    params = config.get("params")
    if not isinstance(params, Mapping):
        return {}
    vectors = params.get("vectors")
    if not isinstance(vectors, Mapping):
        return {}
    if "size" in vectors:
        return dict(vectors)
    named_vector = vectors.get(_DENSE_VECTOR_NAME)
    if isinstance(named_vector, Mapping):
        return dict(named_vector)
    if len(vectors) == 1:
        only_vector = next(iter(vectors.values()))
        if isinstance(only_vector, Mapping):
            return dict(only_vector)
    return {}


def _serialize_collection_info(info: Any) -> Mapping[str, Any]:
    """Serialize collection info."""
    config = getattr(info, "config", None)
    raw_payload_schema = getattr(info, "payload_schema", {}) or {}
    payload_schema = {
        str(field): (
            details.model_dump(mode="json")
            if hasattr(details, "model_dump")
            else details
        )
        for field, details in raw_payload_schema.items()
    }
    config_payload = (
        config.model_dump(mode="json")
        if (config is not None and hasattr(config, "model_dump"))
        else {}
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
    return (
        existing.get("data_type") == schema.value if isinstance(schema, Enum) else False
    )


def _filter_from_mapping(filters: Mapping[str, Any] | None) -> models.Filter | None:
    """Convert mapping to Qdrant filter."""
    if not filters:
        return None
    must_conditions = []
    for key, value in filters.items():
        stored_key = _metadata_field(key)
        if isinstance(value, Mapping):
            must_conditions.append(
                models.FieldCondition(
                    key=stored_key,
                    range=models.Range(**value),
                )
            )
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            must_conditions.append(
                models.FieldCondition(
                    key=stored_key,
                    match=models.MatchAny(any=list(value)),
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key=stored_key,
                    match=models.MatchValue(
                        value=cast("models.ValueVariants", value),
                    ),
                )
            )
    return models.Filter(must=must_conditions)


def _metadata_field(field: str) -> str:
    """Map a logical metadata field to its dependency-native Qdrant path."""
    return field if field.startswith("metadata.") else f"metadata.{field}"
