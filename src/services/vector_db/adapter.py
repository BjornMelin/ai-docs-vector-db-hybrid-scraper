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
        self._supports_query_groups: bool | None = None

    async def _ensure_grouping_capability(self) -> None:
        if self._supports_query_groups is not None:
            return
        self._supports_query_groups = hasattr(self._client, "query_points_groups")

    def supports_query_groups(self) -> bool:
        """Return cached capability flag for QueryPointGroups support."""

        return bool(self._supports_query_groups)

    async def create_collection(self, schema: CollectionSchema) -> None:
        """Create the collection if it does not exist, otherwise ensure schema."""

        if await self._client.collection_exists(schema.name):
            try:
                await self._client.delete_collection(schema.name)
            except UnexpectedResponse as exc:  # pragma: no cover - defensive
                if exc.status_code != 404:
                    raise
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

        await self._client.create_collection(
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

    async def get_collection_info(self, name: str) -> models.CollectionInfo:
        """Return detailed metadata for a collection."""

        return await self._client.get_collection(collection_name=name)

    async def create_payload_index(
        self,
        collection: str,
        field_name: str,
        field_schema: models.PayloadSchemaType,
    ) -> None:
        """Create or update a payload index for the specified field."""

        await self._client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )

    async def delete_payload_index(self, collection: str, field_name: str) -> None:
        """Drop a payload index if it exists."""

        await self._client.delete_payload_index(
            collection_name=collection,
            field_name=field_name,
            wait=True,
        )

    async def upsert(
        self,
        collection: str,
        records: Sequence[VectorRecord],
        *,
        batch_size: int | None = None,
    ) -> None:
        """Upsert vector records into a collection."""

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
        upsert_kwargs: dict[str, Any] = {}
        if batch_size is not None:
            upsert_kwargs["batch_size"] = batch_size
        await self._client.upsert(
            collection_name=collection,
            points=points,
            **upsert_kwargs,
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
        """Query the collection with a vector."""

        query_response = await self._client.query_points(
            collection_name=collection,
            query=list(vector),
            limit=limit,
            query_filter=_filter_from_mapping(filters),
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorMatch(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in query_response.points
        ]

    async def query_groups(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        vector: Sequence[float],
        *,
        group_by: str,
        limit: int = 10,
        group_size: int = 1,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[list[VectorMatch], bool]:
        """Attempt a grouped query with graceful fallback."""

        await self._ensure_grouping_capability()
        if not self.supports_query_groups():
            matches = await self.query(collection, vector, limit=limit, filters=filters)
            return matches, False

        try:
            response = await self._client.query_points_groups(
                collection_name=collection,
                group_by=group_by,
                query=list(vector),
                limit=limit,
                group_size=group_size,
                query_filter=_filter_from_mapping(filters),
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            if exc.status_code in {400, 404, 405, 501}:
                self._supports_query_groups = False
                matches = await self.query(
                    collection, vector, limit=limit, filters=filters
                )
                return matches, False
            raise
        except AttributeError:
            self._supports_query_groups = False
            matches = await self.query(collection, vector, limit=limit, filters=filters)
            return matches, False

        matches: list[VectorMatch] = []
        for group in getattr(response, "groups", []) or []:
            hits = getattr(group, "hits", [])
            if not hits:
                continue
            hit = hits[0]
            payload = dict(hit.payload or {})
            payload["_grouping"] = {
                "applied": True,
                "group_id": getattr(group, "id", None),
            }
            matches.append(
                VectorMatch(
                    id=str(hit.id),
                    score=hit.score,
                    payload=payload,
                )
            )
        return matches, True

    async def hybrid_query(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float] | None = None,
        *,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Perform hybrid search using dense and sparse vectors."""

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

    async def recommend(  # pylint: disable=too-many-arguments
        self,
        collection: str,
        *,
        positive_ids: Sequence[str] | None = None,
        vector: Sequence[float] | None = None,
        limit: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        if not positive_ids and vector is None:
            msg = "`positive_ids` or `vector` must be provided for recommend"
            raise ValueError(msg)

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

        results = await self._client.recommend(**recommend_kwargs)
        return [
            VectorMatch(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results
        ]

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
    clause_keys = {"must", "should", "must_not"} & set(filters)
    if clause_keys:
        filter_kwargs: dict[str, list[models.FieldCondition]] = {}
        for clause in clause_keys:
            entries = filters.get(clause)
            if not entries:
                continue
            if not isinstance(entries, Sequence):
                msg = "Filter clause %s must be a sequence"
                raise TypeError(msg % clause)
            filter_kwargs[clause] = [
                _field_condition_from_entry(entry) for entry in entries
            ]
        must_conditions = cast(list[models.Condition] | None, filter_kwargs.get("must"))
        should_conditions = cast(
            list[models.Condition] | None, filter_kwargs.get("should")
        )
        must_not_conditions = cast(
            list[models.Condition] | None, filter_kwargs.get("must_not")
        )
        return models.Filter(
            must=must_conditions,
            should=should_conditions,
            must_not=must_not_conditions,
        )
    conditions = []
    for key, value in filters.items():
        if isinstance(value, Mapping):
            range_keys = {"gt", "gte", "lt", "lte"} & set(value)
            if range_keys:
                range_kwargs = {
                    range_key: value[range_key]
                    for range_key in ("gt", "gte", "lt", "lte")
                    if range_key in value
                }
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.Range(**range_kwargs),
                    )
                )
                continue
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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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
        iterable = list(values)
    else:
        msg = "MatchAny expects an iterable value collection"
        raise TypeError(msg)

    normalized = [_normalize_match_value(value) for value in iterable]
    return cast(models.AnyVariants, normalized)


def _field_condition_from_entry(entry: Mapping[str, Any]) -> models.FieldCondition:
    """Build a FieldCondition from a structured filter clause entry."""

    key = entry.get("key")
    if not isinstance(key, str) or not key:
        msg = "Filter entry must define a non-empty 'key'"
        raise ValueError(msg)

    if "range" in entry:
        range_payload = entry["range"]
        if not isinstance(range_payload, Mapping):
            msg = "Range filter must supply a mapping"
            raise TypeError(msg)
        range_kwargs = {
            part: range_payload[part]
            for part in ("gt", "gte", "lt", "lte")
            if part in range_payload
        }
        return models.FieldCondition(key=key, range=models.Range(**range_kwargs))

    if "in" in entry:
        return models.FieldCondition(
            key=key,
            match=models.MatchAny(any=_normalize_match_any(entry["in"])),
        )

    if "values" in entry:
        return models.FieldCondition(
            key=key,
            match=models.MatchAny(any=_normalize_match_any(entry["values"])),
        )

    if "value" in entry:
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=_normalize_match_value(entry["value"])),
        )

    msg = "Unsupported filter entry format for key %s"
    raise ValueError(msg % key)


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
        normalized = _coerce_vector_output(cast(models.VectorStructOutput, candidate))
        if normalized is not None:
            return normalized
    return None
