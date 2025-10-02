"""Unit tests for :class:`QdrantVectorAdapter` and helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from httpx import Headers
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.services.vector_db.adapter import (
    QdrantVectorAdapter,
    _coerce_vector_mapping,
    _coerce_vector_output,
    _coerce_vector_sequence,
    _filter_from_mapping,
    _normalize_match_any,
    _normalize_match_value,
)
from src.services.vector_db.adapter_base import (
    CollectionSchema,
    VectorMatch,
    VectorRecord,
)


@pytest.fixture
def adapter(qdrant_client_mock: AsyncMock) -> QdrantVectorAdapter:
    """Instantiate the adapter with the shared AsyncQdrantClient mock."""

    return QdrantVectorAdapter(qdrant_client_mock)


@pytest.mark.asyncio
async def test_create_collection_noop_when_existing(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
    collection_schema: CollectionSchema,
) -> None:
    """Existing collections should not be dropped or recreated."""

    qdrant_client_mock.collection_exists.return_value = True

    await adapter.create_collection(collection_schema)

    qdrant_client_mock.delete_collection.assert_not_awaited()
    qdrant_client_mock.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_collection_skips_drop_when_absent(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
    collection_schema: CollectionSchema,
) -> None:
    """create_collection should avoid dropping when collection is absent."""

    qdrant_client_mock.collection_exists.return_value = False

    await adapter.create_collection(collection_schema)

    qdrant_client_mock.delete_collection.assert_not_awaited()
    qdrant_client_mock.create_collection.assert_awaited_once()
    kwargs = qdrant_client_mock.create_collection.call_args.kwargs
    vector_params: models.VectorParams = kwargs["vectors_config"]
    assert vector_params.size == collection_schema.vector_size
    assert vector_params.distance == models.Distance.COSINE


@pytest.mark.asyncio
async def test_collection_exists_proxy(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """collection_exists should delegate to the client."""

    qdrant_client_mock.collection_exists.return_value = True

    assert await adapter.collection_exists("docs") is True
    qdrant_client_mock.collection_exists.assert_awaited_once_with("docs")


@pytest.mark.asyncio
async def test_drop_collection_handles_missing(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """drop_collection should ignore 404 responses when missing_ok is True."""

    exc = UnexpectedResponse(404, "Not Found", b"", Headers())
    qdrant_client_mock.delete_collection.side_effect = exc

    await adapter.drop_collection("docs", missing_ok=True)


@pytest.mark.asyncio
async def test_list_collections_extracts_names(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """list_collections should return names of discovered collections."""

    qdrant_client_mock.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="docs"), SimpleNamespace(name="api")]
    )

    names = await adapter.list_collections()

    assert names == ["docs", "api"]


@pytest.mark.asyncio
async def test_upsert_builds_points(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """upsert should map VectorRecord objects to PointStruct payloads."""

    records = [
        VectorRecord(id="1", vector=(0.1, 0.2), payload={"tag": "a"}),
        VectorRecord(id="2", vector=(0.3, 0.4), payload={"tag": "b"}),
    ]

    await adapter.upsert("docs", records, batch_size=10)

    qdrant_client_mock.upsert.assert_awaited_once()
    kwargs = qdrant_client_mock.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["batch_size"] == 10
    points = kwargs["points"]
    assert len(points) == 2
    assert points[0].id == "1"
    assert points[0].payload == {"tag": "a"}


@pytest.mark.asyncio
async def test_upsert_ignores_empty_batch(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """empty upsert input should no-op."""

    await adapter.upsert("docs", [])

    qdrant_client_mock.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_delete_by_ids(
    adapter: QdrantVectorAdapter, qdrant_client_mock: AsyncMock
) -> None:
    """delete should pass explicit IDs when provided."""

    await adapter.delete("docs", ids=["1", "2"])

    qdrant_client_mock.delete.assert_awaited_once()
    kwargs = qdrant_client_mock.delete.call_args.kwargs
    selector = kwargs["points_selector"]
    assert selector.points == ["1", "2"]


@pytest.mark.asyncio
async def test_delete_with_filters(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """delete should translate filters into FilterSelector."""

    await adapter.delete("docs", filters={"lang": ["py", "js"]})

    qdrant_client_mock.delete.assert_awaited_once()
    selector = qdrant_client_mock.delete.call_args.kwargs["points_selector"]
    assert isinstance(selector.filter, models.Filter)


@pytest.mark.asyncio
async def test_query_returns_vector_matches(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """query should normalize scored points into VectorMatch objects."""

    qdrant_client_mock.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="1", score=0.9, payload={"title": "A"}),
            SimpleNamespace(id="2", score=0.8, payload={"title": "B"}),
        ]
    )

    matches = await adapter.query("docs", [0.1, 0.2, 0.3], limit=2)

    assert matches == [
        VectorMatch(id="1", score=0.9, payload={"title": "A"}, vector=None),
        VectorMatch(id="2", score=0.8, payload={"title": "B"}, vector=None),
    ]


@pytest.mark.asyncio
async def test_hybrid_query_without_sparse_delegates(
    adapter: QdrantVectorAdapter,
    monkeypatch,
) -> None:
    """Hybrid query should fall back to dense query when sparse_vector absent."""

    async def _query(*args: Any, **kwargs: Any) -> list[VectorMatch]:
        return [VectorMatch(id="x", score=0.1, payload=None)]

    monkeypatch.setattr(adapter, "query", _query)

    matches = await adapter.hybrid_query("docs", [0.1, 0.2], None)

    assert matches[0].id == "x"


@pytest.mark.asyncio
async def test_hybrid_query_with_sparse_builds_prefetch(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """Hybrid query with sparse vector should emit two prefetch entries."""

    qdrant_client_mock.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id="1", score=0.5, payload={})]
    )

    await adapter.hybrid_query(
        "docs",
        dense_vector=[0.1, 0.2],
        sparse_vector={1: 0.3},
        limit=4,
        filters={"lang": "py"},
    )

    qdrant_client_mock.query_points.assert_awaited_once()
    kwargs = qdrant_client_mock.query_points.call_args.kwargs
    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2
    assert kwargs["limit"] == 4


@pytest.mark.asyncio
async def test_get_collection_stats(
    adapter: QdrantVectorAdapter, qdrant_client_mock: AsyncMock
) -> None:
    """Adapter should reshape stats response into a mapping."""

    qdrant_client_mock.get_collection.return_value = SimpleNamespace(
        points_count=3,
        indexed_vectors_count=3,
        config=SimpleNamespace(
            params=SimpleNamespace(dict=lambda: {"default": {"size": 3}})
        ),
    )

    stats = await adapter.get_collection_stats("docs")

    assert stats["points_count"] == 3
    assert stats["vectors"]["default"]["size"] == 3


@pytest.mark.asyncio
async def test_retrieve_with_vectors(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """retrieve should map return objects respecting payload/vector flags."""

    qdrant_client_mock.retrieve.return_value = [
        SimpleNamespace(
            id="1",
            score=0.6,
            payload={"title": "Doc"},
            vector=[0.1, 0.2, 0.3],
        )
    ]

    matches = await adapter.retrieve(
        "docs",
        ["1"],
        with_payload=True,
        with_vectors=True,
    )

    assert matches[0].vector == (0.1, 0.2, 0.3)


@pytest.mark.asyncio
async def test_scroll_returns_matches_and_offset(
    adapter: QdrantVectorAdapter,
    qdrant_client_mock: AsyncMock,
) -> None:
    """scroll should surface points and next offset from client."""

    qdrant_client_mock.scroll.return_value = (
        [
            SimpleNamespace(id="1", payload={"k": "v"}, vector=[0.2]),
        ],
        "cursor",
    )

    matches, cursor = await adapter.scroll("docs", limit=1, with_vectors=True)

    assert cursor == "cursor"
    assert matches[0].vector == (0.2,)


def test_normalize_match_value_accepts_simple_types() -> None:
    """_normalize_match_value should admit bool, int, and string inputs."""

    assert _normalize_match_value(True) is True
    assert _normalize_match_value(3) == 3
    assert _normalize_match_value("tag") == "tag"


def test_normalize_match_value_accepts_float() -> None:
    """Floating point values should be accepted for numeric filters."""

    assert _normalize_match_value(0.5) == 0.5


def test_normalize_match_any_validates_sequences() -> None:
    """Sequences should be coerced to AnyVariants with consistent types."""

    result = _normalize_match_any(["a", "b"])
    assert list(result) == ["a", "b"]


def test_normalize_match_any_accepts_mixed_types() -> None:
    """Mixed sequences should normalize to the native Qdrant value variants."""

    result = _normalize_match_any(["a", 1])
    assert list(result) == ["a", 1]


def test_filter_from_mapping_none() -> None:
    """None input should return None."""

    assert _filter_from_mapping(None) is None


def test_filter_from_mapping_creates_filter() -> None:
    """Mappings should become Qdrant Filter objects."""

    flt = _filter_from_mapping({"lang": ["py", "js"]})
    assert isinstance(flt, models.Filter)


def test_coerce_vector_sequence_normalizes_nested() -> None:
    """Nested sequences should flatten to tuples of floats."""

    assert _coerce_vector_sequence([[0.1, 0.2]]) == (0.1, 0.2)


def test_coerce_vector_output_handles_mapping() -> None:
    """Mappings containing dense vectors should be normalized."""

    mapping = cast(models.VectorStructOutput, {"default": [0.1, 0.2, 0.3]})
    assert _coerce_vector_output(mapping) == (0.1, 0.2, 0.3)


def test_coerce_vector_mapping_skips_sparse() -> None:
    """Sparse vectors should be ignored when coercing mappings."""

    mapping = {
        "default": models.SparseVector(indices=[0], values=[1.0]),
        "alt": [0.1, 0.2],
    }
    assert _coerce_vector_mapping(mapping) == (0.1, 0.2)
