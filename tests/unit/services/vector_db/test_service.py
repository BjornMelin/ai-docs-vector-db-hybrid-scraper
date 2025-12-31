"""Unit tests for the LangChain-backed VectorStoreService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from src.config.models import SearchStrategy
from src.infrastructure.container import ApplicationContainer
from src.services.vector_db.service import VectorStoreService, _filter_from_mapping
from src.services.vector_db.types import CollectionSchema, TextDocument, VectorRecord


class StubVectorStore:
    """Minimal stand-in for LangChain's QdrantVectorStore."""

    def __init__(self, *, collection_name: str, **_: object) -> None:
        """Initialize the stub vector store with a collection name."""
        self.collection_name = collection_name
        self.add_calls: list[tuple[list[Document], list[str]]] = []
        self.search_return: list[tuple[Document, float]] = []
        self.vector_name = "dense"
        self.sparse_vector_name = "langchain-sparse"

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **_: object,
    ) -> None:
        """Record call parameters for later inspection."""
        self.add_calls.append((documents, list(ids or [])))

    def similarity_search_with_score_by_vector(
        self,
        *,
        vector: list[float],
        k: int,
        search_filter: object | None = None,
    ) -> list[tuple[Document, float]]:
        """Record call parameters and return preset results."""
        _ = (vector, k, search_filter)
        return self.search_return or [
            (
                Document(page_content="stub", metadata={"doc_id": "doc-1"}),
                0.42,
            )
        ]


@pytest.fixture
async def initialized_service(
    monkeypatch: pytest.MonkeyPatch,
    vector_container: ApplicationContainer,
) -> AsyncIterator[VectorStoreService]:
    """Provide an initialized VectorStoreService with stubbed dependencies."""
    stub_store = StubVectorStore(collection_name="documents")
    monkeypatch.setattr(
        "src.services.vector_db.service.QdrantVectorStore",
        lambda **kwargs: stub_store,
    )
    monkeypatch.setattr(
        "src.services.vector_db.service.VectorStoreService._build_sync_client",
        lambda self, cfg: MagicMock(),
    )

    async def _to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", _to_thread)

    service = vector_container.vector_store_service()
    await service.initialize()
    try:
        yield service
    finally:
        await service.cleanup()


@pytest.mark.asyncio
async def test_initialize_sets_vector_store(
    initialized_service: VectorStoreService,
) -> None:
    """Vector store should be initialized once with configured collection name."""
    service = initialized_service
    assert service.is_initialized()
    assert service.embedding_dimension == 3


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(
    initialized_service: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_collection should call the async client when collection missing."""
    client = initialized_service._async_client
    assert isinstance(client, AsyncMock)
    client.collection_exists.return_value = False

    schema = CollectionSchema(name="new", vector_size=3)
    await initialized_service.ensure_collection(schema)

    client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_documents_invokes_vector_store(
    initialized_service: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upserting documents should delegate to the LangChain vector store."""
    store = initialized_service._vector_store
    assert isinstance(store, StubVectorStore)

    async def _immediate(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", _immediate)

    docs = [
        TextDocument(id="doc-1", content="alpha", metadata={"tenant": "acme"}),
        TextDocument(id="doc-2", content="beta", metadata={}),
    ]

    await initialized_service.upsert_documents("documents", docs)

    assert len(store.add_calls) == 1
    documents, ids = store.add_calls[0]
    assert [doc.page_content for doc in documents] == ["alpha", "beta"]
    assert all(len(identifier) == 32 for identifier in ids)
    first_metadata = documents[0].metadata or {}
    second_metadata = documents[1].metadata or {}
    assert first_metadata["tenant"] == "acme"
    assert first_metadata["doc_id"] == "doc-1"
    assert first_metadata["chunk_id"] == 0
    assert "content_hash_previous" not in first_metadata
    assert len(first_metadata["content_hash"]) == 32
    assert first_metadata["content"] == "alpha"
    assert "created_at" in first_metadata
    assert second_metadata["tenant"] == "default"
    assert second_metadata["doc_id"] == "doc-2"
    assert second_metadata["chunk_id"] == 0
    assert "content_hash_previous" not in second_metadata
    assert "created_at" in second_metadata


@pytest.mark.asyncio
async def test_upsert_vectors_calls_qdrant_async_client(
    initialized_service: VectorStoreService,
) -> None:
    """upsert_vectors should prepare PointStruct payloads for Async client."""
    service = initialized_service
    async_client = service._async_client
    assert isinstance(async_client, AsyncMock)

    await service.upsert_vectors(
        "documents",
        [
            VectorRecord(
                id="doc-1",
                vector=[0.1, 0.2, 0.3],
                payload={"foo": "bar"},
            )
        ],
    )

    async_client.upsert.assert_awaited_once()
    _, kwargs = async_client.upsert.call_args
    assert kwargs["collection_name"] == "documents"
    points = list(kwargs["points"])
    assert len(points) == 1
    point = points[0]
    assert point.payload["foo"] == "bar"


def test_filter_from_mapping_handles_sequences_and_scalars() -> None:
    """Sequence values should map to MatchAny, scalars to MatchValue."""
    filters = _filter_from_mapping({"tags": ["doc", "faq"], "lang": "en"})
    assert isinstance(filters, models.Filter)
    raw_must = filters.must
    if raw_must is None:
        must_conditions: list[models.Condition] = []
    elif isinstance(raw_must, list):
        must_conditions = list(raw_must)
    else:
        must_conditions = [raw_must]
    match_any = None
    for condition in must_conditions:
        if getattr(condition, "key", None) != "tags":
            continue
        candidate = getattr(condition, "match", None)
        if isinstance(candidate, models.MatchAny):
            match_any = candidate
            break
    assert match_any is not None
    assert isinstance(match_any, models.MatchAny)
    assert match_any.any == ["doc", "faq"]

    match_value = None
    for condition in must_conditions:
        if getattr(condition, "key", None) != "lang":
            continue
        candidate = getattr(condition, "match", None)
        if isinstance(candidate, models.MatchValue):
            match_value = candidate
            break
    assert match_value is not None
    assert match_value.value == "en"


def test_filter_from_mapping_treats_strings_as_scalars() -> None:
    """String sequences should not be treated as iterables for MatchAny."""
    filters = _filter_from_mapping({"category": "docs"})
    assert isinstance(filters, models.Filter)
    raw_must = filters.must
    assert raw_must is not None
    condition = raw_must[0] if isinstance(raw_must, list) else raw_must
    condition_match = getattr(condition, "match", None)
    assert isinstance(condition_match, models.MatchValue)
    assert condition_match.value == "docs"


@pytest.mark.asyncio
async def test_search_documents_returns_vector_matches(
    initialized_service: VectorStoreService,
) -> None:
    """search_documents should normalise results from the vector store."""
    store = initialized_service._vector_store
    assert isinstance(store, StubVectorStore)

    result_doc = Document(
        page_content="alpha",
        metadata={"doc_id": "doc-123", "topic": "testing"},
    )
    store.search_return = [(result_doc, 0.87)]

    matches = await initialized_service.search_documents(
        "documents",
        query="alpha",
        limit=5,
    )

    assert len(matches) == 1
    match = matches[0]
    assert match.id == "doc-123"
    assert match.metadata is not None
    assert match.metadata["topic"] == "testing"
    assert pytest.approx(match.score) == 0.87


@pytest.mark.asyncio
async def test_hybrid_search_uses_dense_path_when_sparse_missing(
    initialized_service: VectorStoreService,
) -> None:
    """Hybrid search should degrade to dense search when sparse payload absent."""
    service = initialized_service
    service._retrieval_mode = SearchStrategy.DENSE
    store = service._vector_store
    assert isinstance(store, StubVectorStore)
    store.search_return = [
        (Document(page_content="dense", metadata={"doc_id": "dense-1"}), 0.51)
    ]

    results = await service.hybrid_search("documents", query="dense query", limit=1)

    assert len(results) == 1
    assert results[0].id == "dense-1"
    assert pytest.approx(results[0].score) == 0.51


@pytest.mark.asyncio
async def test_hybrid_search_executes_hybrid_prefetch(
    initialized_service: VectorStoreService,
) -> None:
    """Hybrid search should request dense and sparse prefetch queries."""
    service = initialized_service
    service._retrieval_mode = SearchStrategy.HYBRID
    service._sparse_embeddings = SimpleNamespace(  # type: ignore[assignment]
        embed_query=lambda _: SimpleNamespace(indices=[0], values=[1.0])
    )
    client = service._async_client
    assert isinstance(client, AsyncMock)
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id="hybrid", score=0.73, payload={"doc_id": "hybrid"})]
    )

    results = await service.hybrid_search("documents", query="hybrid query", limit=1)

    client.query_points.assert_awaited()
    assert len(results) == 1
    assert results[0].id == "hybrid"
    assert pytest.approx(results[0].score) == 0.73


class TestSparseInitialization:
    """Tests for sparse embedding initialization edge cases."""

    @pytest.mark.asyncio
    async def test_initialize_raises_without_sparse_model_for_hybrid(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Hybrid retrieval should fail when sparse_model is missing."""
        from src.config.models import SearchStrategy
        from src.services.errors import EmbeddingServiceError
        from src.services.vector_db.service import VectorStoreService

        monkeypatch.setattr(
            "src.services.vector_db.service.VectorStoreService._build_sync_client",
            lambda self, cfg: MagicMock(),
        )

        config_stub.embedding.retrieval_mode = SearchStrategy.HYBRID
        config_stub.embedding.sparse_model = None
        config_stub.fastembed.sparse_model = None

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        with pytest.raises(EmbeddingServiceError, match="sparse embedding model"):
            await service.initialize()

    @pytest.mark.asyncio
    async def test_initialize_raises_without_sparse_model_for_sparse_mode(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pure sparse retrieval should also fail when sparse_model is absent."""
        from src.config.models import SearchStrategy
        from src.services.errors import EmbeddingServiceError
        from src.services.vector_db.service import VectorStoreService

        monkeypatch.setattr(
            "src.services.vector_db.service.VectorStoreService._build_sync_client",
            lambda self, cfg: MagicMock(),
        )

        config_stub.embedding.retrieval_mode = SearchStrategy.SPARSE
        config_stub.embedding.sparse_model = None
        config_stub.fastembed.sparse_model = None  # Use None instead of empty string

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        with pytest.raises(EmbeddingServiceError, match="sparse embedding model"):
            await service.initialize()

    @pytest.mark.asyncio
    async def test_initialize_raises_without_fastembed_sparse_runtime(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing FastEmbedSparse runtime should produce informative error."""
        from src.config.models import SearchStrategy
        from src.services.errors import EmbeddingServiceError
        from src.services.vector_db.service import VectorStoreService

        monkeypatch.setattr(
            "src.services.vector_db.service.VectorStoreService._build_sync_client",
            lambda self, cfg: MagicMock(),
        )

        monkeypatch.setattr(
            "src.services.vector_db.service.FastEmbedSparseRuntime", None
        )
        config_stub.embedding.retrieval_mode = SearchStrategy.HYBRID
        config_stub.fastembed.sparse_model = "stub-sparse"

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        with pytest.raises(EmbeddingServiceError, match="langchain-qdrant extras"):
            await service.initialize()


class TestEnsureCollectionSparseConfig:
    """Tests for ensure_collection sparse vector configuration."""

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_sparse_vectors_config(
        self,
        initialized_service: VectorStoreService,
    ) -> None:
        """When requires_sparse=True, should include sparse_vectors_config."""
        client = initialized_service._async_client
        assert isinstance(client, AsyncMock)
        client.collection_exists.return_value = False

        schema = CollectionSchema(
            name="hybrid-collection", vector_size=3, requires_sparse=True
        )
        await initialized_service.ensure_collection(schema)

        client.create_collection.assert_awaited_once()
        call_kwargs = client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "hybrid-collection"
        assert call_kwargs.get("sparse_vectors_config") is not None

    @pytest.mark.asyncio
    async def test_ensure_collection_omits_sparse_config_when_not_required(
        self,
        initialized_service: VectorStoreService,
    ) -> None:
        """When requires_sparse=False, sparse_vectors_config should be None."""
        client = initialized_service._async_client
        assert isinstance(client, AsyncMock)
        client.collection_exists.return_value = False

        schema = CollectionSchema(
            name="dense-only", vector_size=3, requires_sparse=False
        )
        await initialized_service.ensure_collection(schema)

        client.create_collection.assert_awaited_once()
        call_kwargs = client.create_collection.call_args.kwargs
        assert call_kwargs.get("sparse_vectors_config") is None


class TestQueryWithServerGrouping:
    """Tests for _query_with_server_grouping and grouping fallback."""

    @pytest.mark.asyncio
    async def test_query_with_server_grouping_success(
        self,
        initialized_service: VectorStoreService,
    ) -> None:
        """Successful server-side grouping should return grouped records."""
        service = initialized_service
        service.config.qdrant.enable_grouping = True

        client = service._async_client
        assert isinstance(client, AsyncMock)

        mock_hit = SimpleNamespace(
            id="doc-grouped",
            score=0.88,
            payload={"doc_id": "doc-grouped", "content": "grouped result"},
        )
        mock_group = SimpleNamespace(id="group-1", hits=[mock_hit])
        client.query_points_groups = AsyncMock(
            return_value=SimpleNamespace(groups=[mock_group])
        )

        records, applied = await service._query_with_server_grouping(
            "documents",
            [1.0, 2.0, 3.0],
            group_by="doc_id",
            group_size=1,
            limit=10,
            filters=None,
        )

        assert applied is True
        assert len(records) == 1
        assert records[0].id == "doc-grouped"
        assert pytest.approx(records[0].score) == 0.88

    @pytest.mark.asyncio
    async def test_query_with_server_grouping_fallback_on_exception(
        self,
        initialized_service: VectorStoreService,
    ) -> None:
        """API exceptions should trigger fallback to empty result."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        service = initialized_service
        service.config.qdrant.enable_grouping = True

        client = service._async_client
        assert isinstance(client, AsyncMock)
        client.query_points_groups = AsyncMock(
            side_effect=UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content=b"error",
                headers={},
            )
        )

        records, applied = await service._query_with_server_grouping(
            "documents",
            [1.0, 2.0, 3.0],
            group_by="doc_id",
            group_size=1,
            limit=10,
            filters=None,
        )

        assert applied is False
        assert records == []

    @pytest.mark.asyncio
    async def test_query_with_server_grouping_disabled_returns_empty(
        self,
        initialized_service: VectorStoreService,
    ) -> None:
        """When grouping is disabled, should return empty without querying."""
        service = initialized_service
        service.config.qdrant.enable_grouping = False

        records, applied = await service._query_with_server_grouping(
            "documents",
            [1.0, 2.0, 3.0],
            group_by="doc_id",
            group_size=1,
            limit=10,
            filters=None,
        )

        assert applied is False
        assert records == []


class TestNormalizeScores:
    """Tests for _normalize_scores method."""

    def test_normalize_scores_returns_empty_for_empty_input(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """Empty record list should return empty list unchanged."""
        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        result = service._normalize_scores([], enabled=True)

        assert result == []

    def test_normalize_scores_disabled_returns_unchanged(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """When disabled, scores should remain unchanged."""
        from src.contracts.retrieval import SearchRecord

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        records = [
            SearchRecord.from_payload(
                {"id": "r1", "content": "a", "score": 0.8, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r2", "content": "b", "score": 0.5, "collection": "docs"}
            ),
        ]

        result = service._normalize_scores(records, enabled=False)

        assert result[0].score == 0.8
        assert result[1].score == 0.5

    def test_normalize_scores_min_max_strategy(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """MIN_MAX strategy should scale scores to 0-1 range."""
        from src.config.models import ScoreNormalizationStrategy
        from src.contracts.retrieval import SearchRecord

        config_stub.query_processing.score_normalization_strategy = (
            ScoreNormalizationStrategy.MIN_MAX
        )

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        records = [
            SearchRecord.from_payload(
                {"id": "r1", "content": "a", "score": 0.9, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r2", "content": "b", "score": 0.5, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r3", "content": "c", "score": 0.3, "collection": "docs"}
            ),
        ]

        result = service._normalize_scores(records, enabled=True)

        # min=0.3, max=0.9, span=0.6
        # r1: (0.9-0.3)/0.6 = 1.0
        # r2: (0.5-0.3)/0.6 = 0.333...
        # r3: (0.3-0.3)/0.6 = 0.0
        assert pytest.approx(result[0].score, rel=1e-2) == 1.0
        assert pytest.approx(result[1].score, rel=1e-2) == 0.333
        assert pytest.approx(result[2].score, rel=1e-2) == 0.0

    def test_normalize_scores_min_max_all_same(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """When all scores are identical, MIN_MAX should return 1.0."""
        from src.config.models import ScoreNormalizationStrategy
        from src.contracts.retrieval import SearchRecord

        config_stub.query_processing.score_normalization_strategy = (
            ScoreNormalizationStrategy.MIN_MAX
        )

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        records = [
            SearchRecord.from_payload(
                {"id": "r1", "content": "a", "score": 0.7, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r2", "content": "b", "score": 0.7, "collection": "docs"}
            ),
        ]

        result = service._normalize_scores(records, enabled=True)

        assert result[0].score == 1.0
        assert result[1].score == 1.0

    def test_normalize_scores_z_score_strategy(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """Z_SCORE strategy should standardize scores around mean."""
        from src.config.models import ScoreNormalizationStrategy
        from src.contracts.retrieval import SearchRecord

        config_stub.query_processing.score_normalization_strategy = (
            ScoreNormalizationStrategy.Z_SCORE
        )

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        records = [
            SearchRecord.from_payload(
                {"id": "r1", "content": "a", "score": 1.0, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r2", "content": "b", "score": 0.5, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r3", "content": "c", "score": 0.0, "collection": "docs"}
            ),
        ]

        result = service._normalize_scores(records, enabled=True)

        # mean=0.5, std_dev≈0.408
        # Normalized scores should sum to ~0 with symmetric distribution
        scores = [r.score for r in result]
        assert pytest.approx(sum(scores), abs=1e-6) == 0.0

    def test_normalize_scores_z_score_all_same(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """When all scores are identical, Z_SCORE should return 0.0."""
        from src.config.models import ScoreNormalizationStrategy
        from src.contracts.retrieval import SearchRecord

        config_stub.query_processing.score_normalization_strategy = (
            ScoreNormalizationStrategy.Z_SCORE
        )

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        records = [
            SearchRecord.from_payload(
                {"id": "r1", "content": "a", "score": 0.6, "collection": "docs"}
            ),
            SearchRecord.from_payload(
                {"id": "r2", "content": "b", "score": 0.6, "collection": "docs"}
            ),
        ]

        result = service._normalize_scores(records, enabled=True)

        assert result[0].score == 0.0
        assert result[1].score == 0.0

    def test_normalize_scores_none_strategy_returns_unchanged(
        self,
        config_stub: object,
        qdrant_client_mock: AsyncMock,
    ) -> None:
        """NONE strategy should leave scores unchanged."""
        from src.config.models import ScoreNormalizationStrategy
        from src.contracts.retrieval import SearchRecord

        config_stub.query_processing.score_normalization_strategy = (
            ScoreNormalizationStrategy.NONE
        )

        service = VectorStoreService(
            config=config_stub, async_qdrant_client=qdrant_client_mock
        )

        records = [
            SearchRecord.from_payload(
                {"id": "r1", "content": "a", "score": 0.85, "collection": "docs"}
            ),
        ]

        result = service._normalize_scores(records, enabled=True)

        assert result[0].score == 0.85
