"""Vector store service built on Qdrant's async Python client.

Provides lifecycle, payload indexes, CRUD, embeddings, dense and hybrid search,
recommendations, and optional grouping while leaning on native client
capabilities. Error handling is minimal and linter-safe.

References: Qdrant Query API hybrid+RRF, grouping; LangChain Qdrant APIs.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models

from src.config.loader import Settings
from src.config.models import QueryProcessingConfig, ScoreNormalizationStrategy
from src.contracts.retrieval import SearchRecord
from src.services.base import BaseService
from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import EmbeddingServiceError
from src.services.observability.tracing import set_span_attributes

from .payload_schema import (
    CanonicalPayload,
    PayloadValidationError,
    ensure_canonical_payload,
)
from .types import CollectionSchema, TextDocument, VectorMatch, VectorRecord


if TYPE_CHECKING:  # pragma: no cover
    from src.infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)

DEFAULT_TENANT: Final[str] = "default"
DEFAULT_SOURCE: Final[str] = "inline"
SPARSE_NAME: Final[str] = "sparse"
DEFAULT_OVERFETCH: Final[float] = 2.0

FilterMap = Mapping[str, Any]


class VectorStoreService(BaseService):  # pylint: disable=too-many-public-methods
    """High-level wrapper around Qdrant via LangChain."""

    def __init__(
        self,
        *,
        config: Settings,
        client_manager: ClientManager,
        embeddings_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the service."""
        super().__init__(config)
        self._client_manager = client_manager
        self._embeddings = embeddings_provider or FastEmbedProvider(
            model_name=getattr(config.fastembed, "model", "BAAI/bge-small-en-v1.5")
        )
        self._async_client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Initialize clients, embeddings, and vector store."""
        if self.is_initialized():
            return
        await self._client_manager.initialize()
        self._async_client = await self._client_manager.get_qdrant_client()
        await self._embeddings.initialize()

        if getattr(self.config, "qdrant", None) is None:
            raise EmbeddingServiceError("Qdrant configuration missing")
        self._mark_initialized()
        logger.info("VectorStoreService initialized")

    async def cleanup(self) -> None:
        """Release clients and embeddings."""
        self._async_client = None
        await self._embeddings.cleanup()
        self._mark_uninitialized()

    @property
    def embedding_dimension(self) -> int:
        """Return dense embedding dimension."""
        return self._embeddings.embedding_dimension

    # ---- collections ---------------------------------------------------------

    async def ensure_collection(self, schema: CollectionSchema) -> None:
        """Ensure a collection exists with the requested schema."""
        client = self._require_async_client()
        if await client.collection_exists(schema.name):
            return

        distance_aliases = {
            "cos": "COSINE",
            "cosine": "COSINE",
            "dot": "DOT",
            "ip": "DOT",
            "l2": "EUCLID",
            "euclid": "EUCLID",
            "l1": "MANHATTAN",
            "manhattan": "MANHATTAN",
        }
        requested = str(schema.distance or "COSINE").lower()
        distance_name = distance_aliases.get(requested, requested).upper()
        distance = models.Distance.__members__.get(
            distance_name, models.Distance.COSINE
        )
        if distance_name not in models.Distance.__members__:
            logger.warning(
                "Unknown distance metric '%s'. Defaulting to COSINE.", schema.distance
            )

        vectors_config = models.VectorParams(size=schema.vector_size, distance=distance)
        sparse_config = (
            {SPARSE_NAME: models.SparseVectorParams()}
            if schema.requires_sparse
            else None
        )

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
        """Return all collection names."""
        client = self._require_async_client()
        return [c.name for c in (await client.get_collections()).collections]

    async def get_collection_info(self, name: str) -> Mapping[str, Any]:
        """Return collection metadata and payload schema."""
        client = self._require_async_client()
        info = await client.get_collection(collection_name=name)
        config = getattr(info, "config", None)
        payload_schema = (
            cast(dict[str, Any], getattr(config, "payload_schema", {}))
            if config
            else {}
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

    async def get_payload_index_summary(self, name: str) -> Mapping[str, Any]:
        """Return payload index summary for a collection."""
        info = await self.get_collection_info(name)
        schema = cast(dict[str, Any], info.get("payload_schema", {}))
        fields = sorted(schema.keys())
        return {
            "indexed_fields_count": len(fields),
            "indexed_fields": fields,
            "payload_schema": schema,
            "points_count": info.get("points_count", 0),
        }

    async def ensure_payload_indexes(
        self, name: str, definitions: Mapping[str, models.PayloadSchemaType]
    ) -> Mapping[str, Any]:
        """Ensure payload indexes match the requested definitions."""
        client = self._require_async_client()
        summary = await self.get_payload_index_summary(name)
        existing: Mapping[str, Mapping[str, Any]] = cast(
            Mapping[str, Mapping[str, Any]], summary.get("payload_schema", {})
        )
        for field, schema in definitions.items():
            present = existing.get(field)
            expected = schema.value if isinstance(schema, Enum) else str(schema)
            if not present or present.get("type") != expected:
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
        existing = set(
            (await self.get_payload_index_summary(name)).get("indexed_fields", [])
        )
        for field in fields:
            if field in existing:
                await client.delete_payload_index(
                    collection_name=name, field_name=field, wait=True
                )

    async def collection_stats(self, name: str) -> Mapping[str, Any]:
        """Alias of get_collection_info for compatibility."""
        return await self.get_collection_info(name)

    # ---- document CRUD -------------------------------------------------------

    async def add_document(
        self,
        collection: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add a single document and return its id."""
        doc_id = str(uuid4())
        md = dict(metadata or {})
        md.setdefault("doc_id", doc_id)
        md.setdefault("chunk_id", md.get("chunk_index", 0))
        md.setdefault("tenant", md.get("tenant") or DEFAULT_TENANT)
        md.setdefault("source", md.get("source") or md.get("url") or DEFAULT_SOURCE)
        md.setdefault("created_at", datetime.now(UTC).isoformat())
        await self.upsert_documents(
            collection, [TextDocument(id=doc_id, content=content, metadata=md)]
        )
        return doc_id

    async def upsert_documents(
        self,
        collection: str,
        documents: Sequence[TextDocument],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Upsert a batch of text documents with embeddings."""
        if not documents:
            return
        await self.ensure_collection(
            CollectionSchema(name=collection, vector_size=self.embedding_dimension)
        )
        texts = [d.content for d in documents]
        can: list[CanonicalPayload] = []
        for d in documents:
            try:
                can.append(
                    ensure_canonical_payload(
                        d.metadata, content=d.content, id_hint=d.id
                    )
                )
            except PayloadValidationError as exc:  # pragma: no cover
                raise EmbeddingServiceError(
                    f"Invalid payload for '{d.id}': {exc}"
                ) from exc

        embeddings = await self._embeddings.generate_embeddings(texts)
        client = self._require_async_client()
        ids = [p.point_id for p in can]
        metadatas = [p.payload for p in can]
        logger.debug("Upserting %d documents into %s", len(ids), collection)

        points = [
            models.PointStruct(id=pid, vector=list(vec), payload=metadata)
            for pid, vec, metadata in zip(ids, embeddings, metadatas, strict=True)
        ]

        if batch_size and batch_size > 0:
            for start in range(0, len(points), batch_size):
                chunk = points[start : start + batch_size]
                await client.upsert(collection_name=collection, points=chunk)
            return

        await client.upsert(collection_name=collection, points=points)

    async def delete(
        self,
        collection: str,
        *,
        ids: Sequence[str] | None = None,
        filters: FilterMap | None = None,
    ) -> None:
        """Delete points by ids or filter."""
        client = self._require_async_client()
        if ids:
            await client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=list(ids)),
            )
            return
        if filters:
            f = _filter_from_mapping(filters)
            if f is not None:
                await client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=f),
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
                requires_sparse=any(r.sparse_vector for r in records),
            )
        )

        client = self._require_async_client()

        def _point(r: VectorRecord) -> models.PointStruct:
            dense = list(r.vector)
            payload = dict(r.payload or {})
            if not r.sparse_vector:
                return models.PointStruct(id=r.id, vector=dense, payload=payload)
            sparse = models.SparseVector(
                indices=list(r.sparse_vector.keys()),
                values=list(r.sparse_vector.values()),
            )
            return models.PointStruct(
                id=r.id,
                vector=dense,
                sparse_vector={SPARSE_NAME: sparse},  # type: ignore[call-arg]
                payload=payload,
            )

        points = [_point(r) for r in records]
        logger.debug("Upserting %d vectors into %s", len(points), collection)
        await client.upsert(collection_name=collection, points=points)

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
        """Delete a document by id. Returns True if removed."""
        if await self.get_document(collection, document_id) is None:
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
        docs = [dict(p.payload or {}) for p in points]
        for p, payload in zip(points, docs, strict=False):
            payload.setdefault("id", str(p.id))
        return docs, (str(next_offset) if next_offset is not None else None)

    # ---- embeddings ----------------------------------------------------------

    async def embed_query(self, query: str) -> Sequence[float]:
        """Generate an embedding for a query string."""
        try:
            return (await self._embeddings.generate_embeddings([query]))[0]
        except Exception as exc:  # pragma: no cover - provider specific
            raise EmbeddingServiceError(f"Failed to embed query: {exc}") from exc

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        try:
            return await self._embeddings.generate_embeddings(list(texts))
        except Exception as exc:  # pragma: no cover - provider specific
            raise EmbeddingServiceError(f"Failed to embed documents: {exc}") from exc

    # ---- search --------------------------------------------------------------

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        filters: FilterMap | None = None,
        group_by: str | None = None,
        group_size: int | None = None,
        overfetch_multiplier: float | None = None,
        normalize_scores: bool | None = None,
    ) -> list[SearchRecord]:
        """Dense similarity search with optional grouping."""
        # pylint: disable=too-many-arguments,too-many-locals
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
        if group_by:
            for rank, m in enumerate(matches, start=1):
                p = dict(m.payload or {})
                gid = p.get(group_by) or p.get("doc_id") or m.id
                grouping_meta = cast(Mapping[str, Any], p.get("_grouping") or {})
                info = dict(grouping_meta)
                info.update(
                    {"group_id": gid, "rank": rank, "applied": grouping_applied}
                )
                p["_grouping"] = info
                m.payload = p

        matches = self._normalize_scores(matches, enabled=normalize_scores)
        return [
            SearchRecord.from_vector_match(m, collection_name=collection)
            for m in matches
        ]

    async def search_vector(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: FilterMap | None = None,
    ) -> list[SearchRecord]:
        """Similarity search using a precomputed vector."""
        # pylint: disable=too-many-arguments
        matches, _ = await self._query_with_optional_grouping(
            collection,
            vector,
            limit=limit,
            group_by=None,
            group_size=1,
            filters=filters,
            overfetch_multiplier=None,
        )
        return [
            SearchRecord.from_vector_match(m, collection_name=collection)
            for m in matches
        ]

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float] | None = None,
        *,
        limit: int = 10,
        filters: FilterMap | None = None,
    ) -> list[SearchRecord]:
        """Hybrid search using Query API prefetch + RRF; dense-only fallback."""
        # pylint: disable=too-many-arguments
        if not sparse_vector:
            return await self.search_vector(collection, dense_vector, limit=limit)

        client = self._require_async_client()
        qf = _filter_from_mapping(filters)

        dense_prefetch: dict[str, Any] = {
            "query": list(dense_vector),
            "limit": max(limit, 20),
        }
        sparse_prefetch: dict[str, Any] = {
            "query": models.SparseVector(
                indices=list(sparse_vector.keys()), values=list(sparse_vector.values())
            ),
            "using": SPARSE_NAME,
            "limit": max(limit, 20),
        }
        if qf is not None:
            dense_prefetch["filter"] = qf
            sparse_prefetch["filter"] = qf

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
        matches = [
            VectorMatch(
                id=str(p.id),
                score=p.score,
                payload=p.payload,
                raw_score=p.score,
                collection=collection,
            )
            for p in result.points
        ]
        return [
            SearchRecord.from_vector_match(m, collection_name=collection)
            for m in matches
        ]

    async def recommend(
        self,
        collection: str,
        *,
        positive_ids: Sequence[str] | None = None,
        vector: Sequence[float] | None = None,
        limit: int = 10,
        filters: FilterMap | None = None,
    ) -> list[SearchRecord]:
        """Return records related to supplied positive examples."""
        # pylint: disable=too-many-arguments
        if not positive_ids and vector is None:
            raise ValueError("`positive_ids` or `vector` must be provided")

        client = self._require_async_client()
        kwargs: dict[str, Any] = {
            "collection_name": collection,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
            "query_filter": _filter_from_mapping(filters),
        }
        if positive_ids:
            kwargs["positive"] = list(positive_ids)
        if vector is not None:
            kwargs["query_vector"] = list(vector)
        if kwargs["query_filter"] is None:
            kwargs.pop("query_filter")

        results = await client.recommend(**kwargs)
        matches = [
            VectorMatch(
                id=str(p.id),
                score=p.score,
                payload=p.payload,
                raw_score=p.score,
                collection=collection,
            )
            for p in results
        ]
        return [
            SearchRecord.from_vector_match(m, collection_name=collection)
            for m in matches
        ]

    # ---- internals -----------------------------------------------------------

    def _require_async_client(self) -> AsyncQdrantClient:
        """Return the async Qdrant client."""
        if self._async_client is None:
            raise RuntimeError("VectorStoreService not initialized")
        return self._async_client

    async def _query_with_optional_grouping(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        group_by: str | None,
        group_size: int,
        filters: FilterMap | None,
        overfetch_multiplier: float | None,
    ) -> tuple[list[VectorMatch], bool]:
        """Query with optional server-side grouping and client-side fallback."""
        # pylint: disable=too-many-arguments,too-many-locals
        cfg = getattr(self.config, "qdrant", None)
        server_grouping = bool(group_by) and bool(
            getattr(cfg, "enable_grouping", False)
        )
        client = self._require_async_client()

        if server_grouping and group_by:
            matches, applied = await self._query_with_server_grouping(
                collection,
                vector,
                group_by=group_by,
                group_size=group_size,
                limit=limit,
                filters=filters,
            )
            if applied:
                set_span_attributes(
                    {"qdrant.grouping": "applied", "qdrant.collection": collection}
                )
                return matches, True
            set_span_attributes(
                {"qdrant.grouping": "fallback", "qdrant.collection": collection}
            )

        fetch_multiplier = overfetch_multiplier or DEFAULT_OVERFETCH
        fetch_limit = max(limit, int(limit * fetch_multiplier))
        vector_filter = _filter_from_mapping(filters)

        search_kwargs: dict[str, Any] = {
            "collection_name": collection,
            "query_vector": list(vector),
            "limit": fetch_limit,
            "with_payload": True,
            "with_vectors": False,
        }
        if vector_filter is not None:
            search_kwargs["query_filter"] = vector_filter

        points = await client.search(**search_kwargs)
        matches: list[VectorMatch] = []
        for point in points:
            payload = dict(point.payload or {})
            if "page_content" not in payload and "content" in payload:
                payload["page_content"] = payload["content"]
            matches.append(
                VectorMatch(
                    id=str(point.id),
                    score=point.score,
                    payload=payload,
                    raw_score=point.score,
                    collection=collection,
                )
            )

        grouping_state = "disabled" if not group_by else "not_applied"
        set_span_attributes(
            {"qdrant.grouping": grouping_state, "qdrant.collection": collection}
        )
        return matches[:limit], False

    async def _query_with_server_grouping(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        group_by: str,
        group_size: int,
        limit: int,
        filters: FilterMap | None,
    ) -> tuple[list[VectorMatch], bool]:
        """Server-side grouping; on error, return empty to trigger fallback."""
        # pylint: disable=too-many-arguments,too-many-locals
        client = self._require_async_client()
        qf = _filter_from_mapping(filters)
        try:
            response = await client.query_points_groups(
                collection_name=collection,
                group_by=group_by,
                query=list(vector),
                limit=limit,
                group_size=group_size,
                query_filter=qf,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # pragma: no cover - surfaced via fallback
            logger.warning(
                "Server-side grouping failed for collection '%s': %s",
                collection,
                exc,
            )
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

    def _normalize_scores(
        self, matches: list[VectorMatch], *, enabled: bool | None
    ) -> list[VectorMatch]:
        """Normalize match scores with MIN_MAX or Z_SCORE as configured."""
        # pylint: disable=too-many-return-statements
        if not matches or not enabled:
            return matches

        for m in matches:
            if m.raw_score is None:
                m.raw_score = float(m.score)

        qcfg: QueryProcessingConfig | None = getattr(
            self.config, "query_processing", None
        )
        strategy = (
            qcfg.score_normalization_strategy
            if qcfg is not None
            else ScoreNormalizationStrategy.MIN_MAX
        )
        if strategy == ScoreNormalizationStrategy.NONE:
            return matches

        scores = [float(m.raw_score or m.score) for m in matches]
        eps = max(
            float((qcfg.score_normalization_epsilon if qcfg else 1e-6) or 1e-6), 1e-9
        )

        if strategy == ScoreNormalizationStrategy.MIN_MAX:
            lo, hi = min(scores), max(scores)
            span = hi - lo
            if span < eps:
                for m in matches:
                    m.score = 1.0
                    m.normalized_score = 1.0
                return matches
            for m in matches:
                val = float(m.raw_score or m.score)
                norm = (val - lo) / span
                m.score = norm
                m.normalized_score = norm
            return matches

        if strategy == ScoreNormalizationStrategy.Z_SCORE:
            mean = statistics.fmean(scores)
            std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            if std < eps:
                for m in matches:
                    m.score = 0.0
                    m.normalized_score = 0.0
                return matches
            for m in matches:
                val = float(m.raw_score or m.score)
                norm = (val - mean) / std
                m.score = norm
                m.normalized_score = norm
            return matches

        return matches


# ---- module helpers ---------------------------------------------------------


def _filter_from_mapping(filters: Mapping[str, Any] | None) -> models.Filter | None:
    """Convert a mapping into a Qdrant Filter."""
    if not filters:
        return None
    must: list[models.FieldCondition] = []
    for key, value in filters.items():
        if isinstance(value, Mapping):
            must.append(models.FieldCondition(key=key, range=models.Range(**value)))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            must.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
            )
        else:
            must.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=cast("models.ValueVariants", value)),
                )
            )
    return models.Filter(must=list(must))  # type: ignore[arg-type]
