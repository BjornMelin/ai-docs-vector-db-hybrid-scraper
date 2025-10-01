"""Qdrant adapter implementation backed by the official client."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from .adapter_base import CollectionSchema, VectorAdapter, VectorMatch, VectorRecord


class QdrantVectorAdapter(VectorAdapter):
    """Implementation of :class:`VectorAdapter` using Qdrant."""

    def __init__(self, client: AsyncQdrantClient) -> None:
        """Initialize the QdrantVectorAdapter."""

        self._client = client

    async def create_collection(self, schema: CollectionSchema) -> None:
        """Create the collection if it does not exist, otherwise ensure schema."""
        vectors_config = {
            "default": models.VectorParams(
                size=schema.vector_size,
                distance=_distance_from_string(schema.distance),
            )
        }
        sparse_config = None
        if schema.requires_sparse:
            sparse_config = {
                "default": models.SparseVectorParams(
                    index=models.SparseIndexParams(),
                )
            }

        await self._client.recreate_collection(
            collection_name=schema.name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

    async def drop_collection(self, name: str, *, missing_ok: bool = True) -> None:
        """Drop a collection.

        Args:
            name: Name of the collection to drop.
            missing_ok: If True, do not raise error if collection does not exist.
        """

        try:
            await self._client.delete_collection(name)
        except UnexpectedResponse as exc:
            if exc.status_code == 404 and missing_ok:
                return
            raise

    async def list_collections(self) -> list[str]:
        """List all collections."""

        collections = await self._client.get_collections()
        return [collection.name for collection in collections.collections]

    async def upsert(
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

        if not records:
            return
        points = [
            models.PointStruct(
                id=record.id,
                vector=list(record.vector),
                payload=dict(record.payload or {}),
            )
            for record in records
        ]
        await self._client.upsert(
            collection_name=collection,
            points=points,
            batch_size=batch_size,
        )

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

        if ids:
            await self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=list(ids)),
            )
            return
        if filters:
            await self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=cast(models.Filter, _filter_from_mapping(filters))
                ),
            )

    async def query(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Query the collection with a vector.

        Args:
            collection: Name of the collection.
            vector: Query vector.
            limit: Maximum number of results.
            filters: Optional filters for the query.

        Returns:
            List of vector matches.
        """

        results = await self._client.search(
            collection_name=collection,
            query_vector=list(vector),
            limit=limit,
            query_filter=_filter_from_mapping(filters),
        )
        return [
            VectorMatch(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results
        ]

    # pylint: disable=too-many-arguments  # Exposes full hybrid search knobs.
    async def hybrid_query(
        self,
        collection: str,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float] | None = None,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Perform hybrid search using dense and sparse vectors.

        Args:
            collection: Name of the collection.
            dense_vector: Dense vector for search.
            sparse_vector: Optional sparse vector for hybrid search.
            limit: Maximum number of results.
            filters: Optional filters for the query.

        Returns:
            List of vector matches.
        """

        if not sparse_vector:
            return await self.query(
                collection,
                dense_vector,
                limit=limit,
                filters=filters,
            )

        prefetch_queries = [
            models.Prefetch(
                query=list(dense_vector),
                using="default",
                limit=max(limit, 20),
                filter=_filter_from_mapping(filters),
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values()),
                ),
                using="default",
                limit=max(limit, 20),
                filter=_filter_from_mapping(filters),
            ),
        ]
        result = await self._client.query_points(
            collection_name=collection,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorMatch(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in result.points
        ]

    async def get_collection_stats(self, name: str) -> Mapping[str, Any]:
        """Get statistics for a collection."""

        info = await self._client.get_collection(name)
        return {
            "points_count": info.points_count,
            "indexed_vectors": info.indexed_vectors_count,
            "vectors": info.config.params.dict() if info.config else {},
        }

    async def retrieve(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[VectorMatch]:
        if not ids:
            return []
        records = await self._client.retrieve(
            collection_name=collection,
            ids=list(ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        matches: list[VectorMatch] = []
        for record in records:
            matches.append(
                VectorMatch(
                    id=str(record.id),
                    score=getattr(record, "score", 0.0),
                    payload=record.payload if with_payload else None,
                    vector=_coerce_vector_output(record.vector)
                    if with_vectors
                    else None,
                )
            )
        return matches

    async def scroll(
        self,
        collection: str,
        *,
        limit: int = 64,
        offset: str | None = None,
        filters: Mapping[str, Any] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[VectorMatch], str | None]:
        points, next_offset = await self._client.scroll(
            collection_name=collection,
            scroll_filter=_filter_from_mapping(filters),
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        matches = [
            VectorMatch(
                id=str(point.id),
                score=getattr(point, "score", 0.0),
                payload=point.payload if with_payload else None,
                vector=_coerce_vector_output(point.vector) if with_vectors else None,
            )
            for point in points
        ]
        return matches, cast(str | None, next_offset)


def _distance_from_string(metric: str) -> models.Distance:
    """Convert a distance metric string to Qdrant Distance enum."""

    normalized = metric.lower()
    if normalized in {"dot", "dotproduct", "dot_product"}:
        return models.Distance.DOT
    if normalized in {"l2", "euclidean"}:
        return models.Distance.EUCLID
    return models.Distance.COSINE


def _filter_from_mapping(
    filters: Mapping[str, Any] | None,
) -> models.Filter | None:
    """Convert a mapping of filters to a Qdrant Filter object."""

    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, Mapping):
            if "in" in value:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=_normalize_match_any(value["in"])),
                    )
                )
                continue
            msg = "Unsupported mapping-based filter for key %s"
            raise TypeError(msg % key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=_normalize_match_any(value)),
                )
            )
            continue
        conditions.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=_normalize_match_value(value)),
            )
        )
    return models.Filter(must=conditions)


def _normalize_match_value(value: Any) -> models.ValueVariants:
    """Coerce simple values into the strict type accepted by Qdrant filters."""

    if isinstance(value, bool):
        return cast(models.ValueVariants, value)
    if isinstance(value, int) and not isinstance(value, bool):
        return cast(models.ValueVariants, value)
    if isinstance(value, str):
        return cast(models.ValueVariants, value)
    msg = "Unsupported match value type for Qdrant filter: %s"
    raise TypeError(msg % type(value).__name__)


def _normalize_match_any(values: Any) -> models.AnyVariants:
    """Validate and normalize collection filters expressed with IN semantics."""

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        iterable: Iterable[Any] = values
    elif isinstance(values, Iterable):
        iterable = values
    else:
        raise TypeError("MatchAny expects an iterable value collection")
    items = list(iterable)
    if not items:
        raise ValueError("MatchAny requires a non-empty sequence of values")
    if all(isinstance(item, str) for item in items):
        return cast(models.AnyVariants, items)
    if all(isinstance(item, (bool, int)) for item in items):
        coerced = [int(item) for item in items]
        return cast(models.AnyVariants, coerced)
    msg = "MatchAny only accepts sequences of strings or integers"
    raise TypeError(msg)


def _coerce_vector_output(
    vector: models.VectorStructOutput | None,
) -> Sequence[float] | None:
    """Normalize Qdrant vector outputs to the adapter's dense sequence shape."""

    if vector is None:
        return None
    if isinstance(vector, Mapping):
        return _coerce_vector_mapping(vector)
    if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
        return _coerce_vector_sequence(vector)
    return None


def _coerce_vector_sequence(vector: Sequence[Any]) -> Sequence[float] | None:
    """Convert nested dense vector sequences into a flat tuple of floats."""

    if not vector:
        return ()
    first = vector[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        return _coerce_vector_sequence(cast(Sequence[Any], first))
    if all(isinstance(component, (int, float)) for component in vector):
        numeric_vector = cast(Sequence[float], vector)
        return tuple(float(component) for component in numeric_vector)
    return None


def _coerce_vector_mapping(vector: Mapping[str, Any]) -> Sequence[float] | None:
    """Resolve named vector structs, preferring the default dense entry."""

    preferred = vector.get("default")
    candidates: list[Any] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(value for key, value in vector.items() if key != "default")
    for candidate in candidates:
        if isinstance(candidate, models.SparseVector):
            continue
        normalized = _coerce_vector_output(
            cast(models.VectorStructOutput, candidate)
        )
        if normalized is not None:
            return normalized
    return None
