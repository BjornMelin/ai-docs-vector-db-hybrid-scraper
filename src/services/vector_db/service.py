"""Vector store service backed by the Qdrant adapter."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from qdrant_client import models

from src.config import get_config
from src.config.models import QueryProcessingConfig, ScoreNormalizationStrategy
from src.services.base import BaseService
from src.services.embeddings.base import EmbeddingProvider
from src.services.embeddings.fastembed_provider import FastEmbedProvider
from src.services.errors import EmbeddingServiceError
from src.services.monitoring.metrics import get_metrics_registry

from .adapter import QdrantVectorAdapter
from .adapter_base import CollectionSchema, TextDocument, VectorMatch, VectorRecord
from .payload_schema import (
    CanonicalPayload,
    PayloadValidationError,
    ensure_canonical_payload,
)


if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager


class VectorStoreService(BaseService):  # pylint: disable=too-many-public-methods
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

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the dense embedding vectors."""

        return self._embeddings.embedding_dimension

    async def ensure_collection(self, schema: CollectionSchema) -> None:
        """Ensure a collection exists with the given schema."""

        adapter = self._require_adapter()
        if await adapter.collection_exists(schema.name):
            return
        await adapter.create_collection(schema)

    async def drop_collection(self, name: str) -> None:
        """Drop a collection by name."""

        adapter = self._require_adapter()
        await adapter.drop_collection(name)

    async def list_collections(self) -> list[str]:
        """List all collections."""

        adapter = self._require_adapter()
        return await adapter.list_collections()

    async def get_collection_info(self, name: str) -> Mapping[str, Any]:
        """Return raw collection metadata."""

        adapter = self._require_adapter()
        info = await adapter.get_collection_info(name)
        return _serialize_collection_info(info)

    async def get_payload_index_summary(self, name: str) -> Mapping[str, Any]:
        """Return a summary of payload indexes for the collection."""

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
        self, name: str, definitions: Mapping[str, models.PayloadSchemaType]
    ) -> Mapping[str, Any]:
        """Ensure the expected payload indexes exist for the collection."""

        adapter = self._require_adapter()
        summary = await self.get_payload_index_summary(name)
        existing_schema: Mapping[str, Mapping[str, Any]] = summary.get(
            "payload_schema", {}
        )
        for field, schema in definitions.items():
            if not _schema_matches(existing_schema.get(field), schema):
                await adapter.create_payload_index(name, field, schema)
        return await self.get_payload_index_summary(name)

    async def drop_payload_indexes(self, name: str, fields: Iterable[str]) -> None:
        """Drop the specified payload indexes if they exist."""

        adapter = self._require_adapter()
        summary = await self.get_payload_index_summary(name)
        existing_fields = set(summary.get("indexed_fields", []))
        for field in fields:
            if field in existing_fields:
                await adapter.delete_payload_index(name, field)

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

        records: list[VectorRecord] = []
        for document, vector in zip(documents, embeddings, strict=True):
            try:
                canonical: CanonicalPayload = ensure_canonical_payload(
                    document.metadata,
                    content=document.content,
                    id_hint=document.id,
                )
            except PayloadValidationError as exc:  # pragma: no cover - defensive
                msg = f"Invalid payload for document '{document.id}': {exc}"
                raise EmbeddingServiceError(msg) from exc

            records.append(
                VectorRecord(
                    id=canonical.point_id,
                    vector=vector,
                    payload=canonical.payload,
                )
            )

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

    async def embed_query(self, query: str) -> Sequence[float]:
        """Return the embedding vector for the supplied query string."""

        await self.initialize()
        try:
            [embedding] = await self._embeddings.generate_embeddings([query])
        except Exception as exc:  # pragma: no cover - surfaces consistent errors
            msg = f"Failed to embed query: {exc}"
            raise EmbeddingServiceError(msg) from exc
        return embedding

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for the supplied document texts."""

        await self.initialize()
        try:
            embeddings = await self._embeddings.generate_embeddings(list(texts))
        except Exception as exc:  # pragma: no cover - surfaces consistent errors
            msg = f"Failed to embed documents: {exc}"
            raise EmbeddingServiceError(msg) from exc
        return embeddings

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
    ) -> list[VectorMatch]:
        """Search documents with optional grouping and score normalization."""

        embedding = await self.embed_query(query)

        qdrant_cfg = getattr(self.config, "qdrant", None)
        query_cfg: QueryProcessingConfig | None = getattr(
            self.config, "query_processing", None
        )

        resolved_group_by = group_by or getattr(qdrant_cfg, "group_by_field", "doc_id")
        resolved_group_size = max(
            1, int(group_size or getattr(qdrant_cfg, "group_size", 1))
        )
        multiplier_default = float(getattr(qdrant_cfg, "groups_limit_multiplier", 2.0))
        resolved_multiplier = overfetch_multiplier or multiplier_default
        if resolved_multiplier <= 0:
            resolved_multiplier = 1.0
        effective_limit = max(
            limit * resolved_group_size,
            int(limit * resolved_multiplier),
            limit,
        )

        matches, grouping_applied = await self._query_with_optional_grouping(
            collection=collection,
            vector=embedding,
            limit=effective_limit,
            group_by=resolved_group_by,
            group_size=resolved_group_size,
            filters=filters,
        )

        matches = self._attach_collection_metadata(matches, collection)

        if resolved_group_by and not grouping_applied:
            matches = self._group_client_side(
                matches,
                group_by=resolved_group_by,
                group_size=resolved_group_size,
                limit=effective_limit,
            )

        matches = self._annotate_grouping_metadata(
            matches,
            group_by=resolved_group_by,
            grouping_applied=grouping_applied,
        )

        should_normalize = normalize_scores
        if should_normalize is None and query_cfg is not None:
            should_normalize = query_cfg.enable_score_normalization
        matches = self._normalize_scores(matches, enabled=should_normalize)

        return matches[:limit]

    async def _query_with_optional_grouping(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        group_by: str | None,
        group_size: int,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[list[VectorMatch], bool]:
        adapter = self._require_adapter()
        qdrant_cfg = getattr(self.config, "qdrant", None)
        grouping_enabled = bool(getattr(qdrant_cfg, "enable_grouping", False)) and bool(
            group_by
        )
        grouping_applied = False
        registry = None
        try:
            registry = get_metrics_registry()
        except RuntimeError:  # pragma: no cover - monitoring disabled
            registry = None

        if grouping_enabled and group_by is not None:
            await self.ensure_payload_indexes(
                collection,
                {group_by: models.PayloadSchemaType.KEYWORD},
            )

            start_time = time.perf_counter()
            matches, grouping_applied = await adapter.query_groups(
                collection,
                vector,
                group_by=group_by,
                limit=limit,
                group_size=group_size,
                filters=filters,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            if registry is not None:
                status = "applied" if grouping_applied else "fallback"
                registry.record_grouping_attempt(
                    collection, status, duration_ms=duration_ms
                )
            if grouping_applied:
                return matches[:limit], True
            return matches, False

        matches = await adapter.query(
            collection,
            vector,
            limit=limit,
            filters=filters,
        )
        if registry is not None:
            registry.record_grouping_attempt(collection, "disabled")
        return matches, False

    @staticmethod
    def _attach_collection_metadata(
        matches: list[VectorMatch], collection: str
    ) -> list[VectorMatch]:
        for match in matches:
            match.collection = match.collection or collection
            payload = dict(match.payload or {})
            match.payload = payload
        return matches

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
            payload = dict(match.payload or {})
            group_id = payload.get(group_by) or payload.get("doc_id")
            if group_id is None:
                group_id = match.id
            payload.setdefault("doc_id", group_id)
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
                payload.setdefault("doc_id", group_id)
                match.payload = payload
                limited_matches.append(match)
            if len(limited_matches) >= limit:
                break
        return limited_matches

    @staticmethod
    def _annotate_grouping_metadata(
        matches: list[VectorMatch],
        *,
        group_by: str | None,
        grouping_applied: bool,
    ) -> list[VectorMatch]:
        if not group_by:
            return matches
        for rank, match in enumerate(matches, start=1):
            payload = dict(match.payload or {})
            group_info = dict(payload.get("_grouping") or {})
            group_info.setdefault(
                "group_id",
                payload.get(group_by) or payload.get("doc_id") or match.id,
            )
            group_info.setdefault("rank", rank)
            group_info.setdefault("applied", grouping_applied)
            payload["_grouping"] = group_info
            match.payload = payload
        return matches

    def _normalize_scores(
        self, matches: list[VectorMatch], *, enabled: bool | None
    ) -> list[VectorMatch]:
        if not matches:
            return matches

        # Always record raw_score before normalization
        for match in matches:
            if match.raw_score is None:
                match.raw_score = float(match.score)

        if not enabled:
            return matches

        config = getattr(self, "config", None)
        query_cfg: QueryProcessingConfig | None = getattr(
            config, "query_processing", None
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

        matches, _ = await self._query_with_optional_grouping(
            collection,
            vector,
            limit=limit,
            group_by=None,
            group_size=1,
            filters=filters,
        )
        return matches

    async def hybrid_search(  # pylint: disable=too-many-arguments
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

    async def retrieve_documents(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[VectorMatch]:
        """Retrieve specific documents by ID.

        Args:
            collection: Name of the collection.
            ids: Sequence of record IDs to fetch.
            with_payload: Include payload data.
            with_vectors: Include stored vectors.

        Returns:
            List of vector matches with payloads and optional vectors.
        """

        adapter = self._require_adapter()
        return await adapter.retrieve(
            collection,
            ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    async def recommend_similar(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        *,
        positive_ids: Sequence[str] | None = None,
        vector: Sequence[float] | None = None,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Return records similar to the supplied identifiers or vector."""

        adapter = self._require_adapter()
        return await adapter.recommend(
            collection,
            positive_ids=positive_ids,
            vector=vector,
            limit=limit,
            filters=filters,
        )

    async def scroll(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        *,
        limit: int = 64,
        offset: str | None = None,
        filters: Mapping[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[VectorMatch], str | None]:
        """Scroll through a collection with pagination."""

        adapter = self._require_adapter()
        return await adapter.scroll(
            collection,
            limit=limit,
            offset=offset,
            filters=filters,
            with_payload=with_payload,
            with_vectors=with_vectors,
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


def _serialize_collection_info(info: models.CollectionInfo) -> Mapping[str, Any]:
    """Convert a Qdrant CollectionInfo into a plain mapping."""

    payload_schema_raw = getattr(info, "payload_schema", {}) or {}
    payload_schema = {
        name: _serialize_payload_schema_entry(entry)
        for name, entry in payload_schema_raw.items()
    }
    return {
        "status": getattr(info, "status", None),
        "vectors_count": getattr(info, "vectors_count", None),
        "points_count": getattr(info, "points_count", None),
        "payload_schema": payload_schema,
    }


def _serialize_payload_schema_entry(entry: Any) -> Mapping[str, Any]:
    """Normalize payload schema entries to serializable dictionaries."""

    if entry is None:
        return {}
    if hasattr(entry, "to_dict"):
        raw = entry.to_dict()
    elif isinstance(entry, dict):
        raw = dict(entry)
    else:
        raw = {
            key: getattr(entry, key)
            for key in dir(entry)
            if not key.startswith("_") and not callable(getattr(entry, key))
        }
    data_type = raw.get("data_type") or raw.get("type")
    if isinstance(data_type, (models.PayloadSchemaType, Enum)):
        raw["data_type"] = data_type.value
    return raw


def _schema_matches(
    existing: Mapping[str, Any] | None,
    expected: models.PayloadSchemaType,
) -> bool:
    """Return True if an existing index matches the desired definition."""

    if existing is None:
        return False
    current_type = existing.get("data_type") or existing.get("type")
    target = expected.value
    if isinstance(target, Enum):  # pragma: no cover - defensive
        target = target.value
    return bool(target == current_type)
