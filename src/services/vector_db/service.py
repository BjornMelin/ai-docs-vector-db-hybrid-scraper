"""Vector store service implemented on top of LangChain's Qdrant integration."""
# pylint: disable=too-many-arguments,too-many-return-statements,too-many-branches,too-many-locals

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from src.config import get_config
from src.config.models import QueryProcessingConfig, ScoreNormalizationStrategy
from src.services.base import BaseService
from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import EmbeddingServiceError
from src.services.monitoring.metrics import get_metrics_registry

from .payload_schema import (
    CanonicalPayload,
    PayloadValidationError,
    ensure_canonical_payload,
)
from .types import CollectionSchema, TextDocument, VectorMatch, VectorRecord


if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


class VectorStoreService(BaseService):  # pylint: disable=too-many-public-methods
    """High-level vector store wrapper using LangChain's QdrantVectorStore."""

    def __init__(
        self,
        config=None,
        client_manager: ClientManager | None = None,
        embeddings_provider: EmbeddingProvider | None = None,
    ) -> None:
        config = config or get_config()
        if client_manager is None:
            from src.infrastructure.client_manager import (  # pylint: disable=import-outside-toplevel
                ClientManager as _ClientManager,
            )

            client_manager = _ClientManager()
        if embeddings_provider is None:
            model_name = getattr(config.fastembed, "model", "BAAI/bge-small-en-v1.5")
            embeddings_provider = FastEmbedProvider(model_name=model_name)

        super().__init__(config)
        self._client_manager = client_manager
        self._embeddings = embeddings_provider
        self._async_client: AsyncQdrantClient | None = None
        self._sync_client: QdrantClient | None = None
        self._vector_store: QdrantVectorStore | None = None

    async def initialize(self) -> None:
        """Initialize Qdrant clients and embeddings."""

        if self.is_initialized():
            return

        await self._client_manager.initialize()
        self._async_client = await self._client_manager.get_qdrant_client()
        await self._embeddings.initialize()
        cfg = self._require_qdrant_config()
        adapter = self._require_embedding_adapter()
        self._sync_client = self._build_sync_client(cfg)
        self._vector_store = QdrantVectorStore(
            client=self._sync_client,
            collection_name=cfg.collection_name,
            embedding=adapter,
            retrieval_mode=RetrievalMode.DENSE,
        )
        self._mark_initialized()
        logger.info("VectorStoreService initialised via LangChain QdrantVectorStore")

    async def cleanup(self) -> None:
        """Release Qdrant clients and embeddings."""

        self._vector_store = None
        self._sync_client = None
        self._async_client = None
        await self._embeddings.cleanup()
        self._mark_uninitialized()

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the dense embeddings."""

        return self._embeddings.embedding_dimension

    async def ensure_collection(self, schema: CollectionSchema) -> None:
        """Ensure a collection with the supplied schema exists."""

        client = self._require_async_client()
        if await client.collection_exists(schema.name):
            return
        vectors_config = models.VectorParams(
            size=schema.vector_size,
            distance=_distance_from_string(schema.distance),
        )
        sparse_config = None
        if schema.requires_sparse:
            sparse_config = {
                "default": models.SparseVectorParams(
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

    async def delete_collection(self, name: str) -> None:
        """Backward-compatible alias for drop_collection."""

        await self.drop_collection(name)

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
        documents: Sequence[TextDocument],
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
            )
        )

        store = self._require_vector_store(collection)
        texts = [document.content for document in documents]
        canonical_payloads: list[CanonicalPayload] = []
        for document in documents:
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

        ids = [payload.point_id for payload in canonical_payloads]
        metadatas = [payload.payload for payload in canonical_payloads]

        await asyncio.to_thread(
            store.add_texts,
            texts=texts,
            metadatas=metadatas,
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
            [embedding] = await self._embeddings.generate_embeddings([query])
        except Exception as exc:  # pragma: no cover - provider-specific failures
            msg = f"Failed to embed query: {exc}"
            raise EmbeddingServiceError(msg) from exc
        return embedding

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for documents."""

        try:
            return await self._embeddings.generate_embeddings(list(texts))
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
    ) -> list[VectorMatch]:  # pylint: disable=too-many-arguments
        """Execute a dense similarity search with optional grouping."""

        vector = await self.embed_query(query)
        matches, grouping_applied = await self._query_with_optional_grouping(
            collection,
            vector,
            limit=limit,
            group_by=group_by,
            group_size=group_size or 1,
            filters=filters,
            overfetch_multiplier=overfetch_multiplier,
        )
        matches = self._annotate_grouping_metadata(
            matches,
            group_by=group_by,
            grouping_applied=grouping_applied,
        )
        matches = self._normalize_scores(matches, enabled=normalize_scores)
        return matches

    async def search_vector(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Perform a similarity search using a precomputed vector."""

        matches, _ = await self._query_with_optional_grouping(
            collection,
            vector,
            limit=limit,
            group_by=None,
            group_size=1,
            filters=filters,
            overfetch_multiplier=None,
        )
        return matches

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float] | None = None,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:  # pylint: disable=too-many-arguments
        """Perform a hybrid search when sparse vectors are supplied."""

        if not sparse_vector:
            return await self.search_vector(
                collection,
                dense_vector,
                limit=limit,
                filters=filters,
            )

        client = self._require_async_client()
        query_filter = _filter_from_mapping(filters)
        sparse_query = models.SparseVector(
            indices=list(sparse_vector.keys()),
            values=list(sparse_vector.values()),
        )
        dense_prefetch: dict[str, Any] = {
            "query": list(dense_vector),
            "using": "default",
            "limit": max(limit, 20),
        }
        sparse_prefetch: dict[str, Any] = {
            "query": sparse_query,
            "using": "sparse",
            "limit": max(limit, 20),
        }
        if query_filter is not None:
            dense_prefetch["filter"] = query_filter
            sparse_prefetch["filter"] = query_filter
        result = await client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(**dense_prefetch),
                models.Prefetch(**sparse_prefetch),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        matches: list[VectorMatch] = []
        for point in result.points:
            matches.append(
                VectorMatch(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload,
                    raw_score=point.score,
                    collection=collection,
                )
            )
        return matches

    async def recommend(
        self,
        collection: str,
        *,
        positive_ids: Sequence[str] | None = None,
        vector: Sequence[float] | None = None,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
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
        matches: list[VectorMatch] = []
        for point in results:
            matches.append(
                VectorMatch(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload,
                    raw_score=point.score,
                    collection=collection,
                )
            )
        return matches

    # ------------------------------------------------------------------
    # Internal helpers

    def _require_async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            msg = "VectorStoreService not initialised"
            raise RuntimeError(msg)
        return self._async_client

    def _require_vector_store(self, collection: str) -> QdrantVectorStore:
        if self._vector_store is None:
            msg = "VectorStoreService not initialised"
            raise RuntimeError(msg)
        # LangChain's vector store keeps the collection name; override if needed.
        self._vector_store.collection_name = collection
        return self._vector_store

    def _require_qdrant_config(self) -> Any:
        cfg = getattr(self.config, "qdrant", None)
        if cfg is None:
            msg = "Qdrant configuration missing"
            raise EmbeddingServiceError(msg)
        return cfg

    def _require_embedding_adapter(self) -> Any:
        adapter = getattr(self._embeddings, "langchain_embeddings", None)
        if adapter is None:
            msg = "Embedding provider does not expose LangChain embeddings"
            raise EmbeddingServiceError(msg)
        return adapter

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
    ) -> tuple[list[VectorMatch], bool]:  # pylint: disable=too-many-arguments,too-many-locals
        registry = None
        try:
            registry = get_metrics_registry()
        except RuntimeError:  # pragma: no cover - monitoring optional
            registry = None

        cfg = self._require_qdrant_config()
        grouping_enabled = bool(group_by) and bool(
            getattr(cfg, "enable_grouping", False)
        )

        if grouping_enabled and group_by:
            matches, applied = await self._query_with_server_grouping(
                collection,
                vector,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
                filters=filters,
            )
            if applied:
                if registry is not None:
                    registry.record_grouping_attempt(collection, "applied")
                return matches, True
            if registry is not None:
                registry.record_grouping_attempt(collection, "fallback")

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
        matches = [
            _document_to_match(collection, document, score)
            for document, score in documents_with_scores
        ]

        if grouping_enabled and group_by:
            matches = self._group_client_side(
                matches,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
            )
            return matches, False

        if registry is not None:
            registry.record_grouping_attempt(collection, "disabled")
        return matches[:limit], False

    async def _query_with_server_grouping(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        group_by: str,
        group_size: int,
        limit: int,
        filters: Mapping[str, Any] | None,
    ) -> tuple[list[VectorMatch], bool]:  # pylint: disable=too-many-arguments
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

        matches: list[VectorMatch] = []
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
            matches.append(
                VectorMatch(
                    id=str(hit.id),
                    score=hit.score,
                    payload=payload,
                    raw_score=hit.score,
                    collection=collection,
                )
            )
        return matches, bool(matches)

    def _group_client_side(
        self,
        matches: list[VectorMatch],
        *,
        group_by: str,
        group_size: int,
        limit: int,
    ) -> list[VectorMatch]:
        groups: dict[str, list[VectorMatch]] = {}
        for match in matches:
            payload: dict[str, Any] = dict(match.payload or {})
            group_id = payload.get(group_by) or payload.get("doc_id")
            if group_id is None:
                group_id = match.id
            payload["doc_id"] = group_id
            match.payload = payload
            groups.setdefault(str(group_id), []).append(match)

        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (item[1][0].raw_score or item[1][0].score),
            reverse=True,
        )

        limited_matches: list[VectorMatch] = []
        for _, group_matches in ordered_groups:
            for group_rank, match in enumerate(group_matches[:group_size], start=1):
                payload = dict(match.payload or {})
                group_id = payload.get(group_by) or payload.get("doc_id") or match.id
                payload["_grouping"] = {
                    "applied": False,
                    "group_id": group_id,
                    "rank": group_rank,
                }
                payload["doc_id"] = group_id
                match.payload = payload
                limited_matches.append(match)
            if len(limited_matches) >= limit:
                break
        return limited_matches

    def _annotate_grouping_metadata(
        self,
        matches: list[VectorMatch],
        *,
        group_by: str | None,
        grouping_applied: bool,
    ) -> list[VectorMatch]:
        if not group_by:
            return matches
        for rank, match in enumerate(matches, start=1):
            payload: dict[str, Any] = dict(match.payload or {})
            group_info: dict[str, Any] = dict(payload.get("_grouping") or {})
            group_info["group_id"] = (
                payload.get(group_by) or payload.get("doc_id") or match.id
            )
            group_info["rank"] = rank
            group_info["applied"] = grouping_applied
            payload["_grouping"] = group_info
            match.payload = payload
        return matches

    def _normalize_scores(
        self, matches: list[VectorMatch], *, enabled: bool | None
    ) -> list[VectorMatch]:  # pylint: disable=too-many-branches,too-many-return-statements
        if not matches:
            return matches

        for match in matches:
            if match.raw_score is None:
                match.raw_score = float(match.score)

        if not enabled:
            return matches

        query_cfg: QueryProcessingConfig | None = getattr(
            self.config, "query_processing", None
        )
        strategy = (
            query_cfg.score_normalization_strategy
            if query_cfg is not None
            else ScoreNormalizationStrategy.MIN_MAX
        )
        if strategy == ScoreNormalizationStrategy.NONE:
            return matches

        scores = [float(match.raw_score or match.score) for match in matches]
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
                for match in matches:
                    match.score = 1.0
                    match.normalized_score = 1.0
                return matches
            for match in matches:
                normalized = ((match.raw_score or match.score) - minimum) / span
                match.score = normalized
                match.normalized_score = normalized
            return matches

        if strategy == ScoreNormalizationStrategy.Z_SCORE:
            mean = statistics.fmean(scores)
            std_dev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            if std_dev < epsilon:
                for match in matches:
                    match.score = 0.0
                    match.normalized_score = 0.0
                return matches
            for match in matches:
                normalized = ((match.raw_score or match.score) - mean) / std_dev
                match.score = normalized
                match.normalized_score = normalized
            return matches

        return matches

    def _build_sync_client(self, cfg: Any) -> QdrantClient:
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


def _document_to_match(
    collection: str,
    document: Document,
    score: float,
) -> VectorMatch:
    payload: dict[str, Any] = dict(document.metadata or {})
    payload.setdefault("page_content", document.page_content)
    identifier = (
        payload.get("point_id")
        or payload.get("doc_id")
        or getattr(document, "id", None)
        or uuid4().hex
    )
    return VectorMatch(
        id=str(identifier),
        score=score,
        payload=payload,
        raw_score=score,
        collection=collection,
    )


def _serialize_collection_info(info: Any) -> Mapping[str, Any]:
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
    if not existing:
        return False
    return existing.get("type") == schema.value if isinstance(schema, Enum) else False


def _distance_from_string(name: str) -> models.Distance:
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    return mapping.get(name.lower(), models.Distance.COSINE)


def _filter_from_mapping(filters: Mapping[str, Any] | None) -> models.Filter | None:
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
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
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
