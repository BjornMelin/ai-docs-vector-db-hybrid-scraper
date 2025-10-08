"""Unit tests for the LangChain VectorServiceRetriever wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.contracts.retrieval import SearchRecord
from src.services.rag import RAGConfig, VectorServiceRetriever


def _make_vector_store_mock(initialized: bool = True) -> MagicMock:
    """Create a mock VectorStoreService with sensible defaults."""

    store = MagicMock()
    store.is_initialized.return_value = initialized
    store.initialize = AsyncMock()
    store.search_documents = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_aget_relevant_documents_initializes_service() -> None:
    """The retriever should initialize the vector service lazily."""

    vector_store = _make_vector_store_mock(initialized=False)
    vector_store.search_documents.return_value = []
    retriever = VectorServiceRetriever(vector_store, "docs")

    await retriever._aget_relevant_documents(
        "query",
        run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
    )

    vector_store.initialize.assert_awaited_once()
    vector_store.search_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_aget_relevant_documents_returns_documents() -> None:
    """Documents returned by the vector store should be normalized."""

    match = SearchRecord(
        id="doc-1",
        score=0.87,
        content="lorem",
        title="Note",
        metadata={"content": "lorem", "title": "Note"},
    )

    vector_store = _make_vector_store_mock()
    vector_store.search_documents.return_value = [match]

    retriever = VectorServiceRetriever(vector_store, "docs")
    documents = await retriever._aget_relevant_documents(
        "query",
        run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
    )

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "lorem"
    assert documents[0].metadata["source_id"] == "doc-1"
    assert documents[0].metadata["score"] == pytest.approx(0.87)


def test_with_search_kwargs_creates_new_instance() -> None:
    """with_search_kwargs must return a new retriever with overrides applied."""

    vector_store = _make_vector_store_mock()
    rag_config = RAGConfig(compression_enabled=False)
    retriever = VectorServiceRetriever(
        vector_store,
        "docs",
        k=3,
        rag_config=rag_config,
    )

    overridden = retriever.with_search_kwargs(k=7, filters={"category": "api"})

    assert overridden is not retriever
    assert isinstance(overridden, VectorServiceRetriever)
    assert overridden._k == 7
    assert overridden._filters == {"category": "api"}
    # Ensure original instance unchanged
    assert retriever._k == 3
    assert retriever._filters is None


@pytest.mark.asyncio
async def test_compression_pipeline_filters_documents(monkeypatch) -> None:
    """Compression should remove low-similarity documents via LangChain pipeline."""

    class _StubEmbeddings(Embeddings):
        def __init__(self, *_, **__):
            return

        def embed_documents(self, texts):
            return [[1.0, 0.0] if "keep" in text else [0.0, 1.0] for text in texts]

        def embed_query(self, text):
            _ = text
            return [1.0, 0.0]

    monkeypatch.setattr(
        "src.services.rag.retriever.FastEmbedEmbeddings", _StubEmbeddings
    )

    match_keep = SearchRecord(
        id="doc-keep",
        score=0.9,
        content="please keep this",
        title="Keep",
        metadata={"content": "please keep this", "title": "Keep"},
    )
    match_drop = SearchRecord(
        id="doc-drop",
        score=0.6,
        content="discard me",
        title="Drop",
        metadata={"content": "discard me", "title": "Drop"},
    )

    vector_store = _make_vector_store_mock()
    vector_store.config = SimpleNamespace(fastembed=SimpleNamespace(model="stub-model"))
    vector_store.search_documents.return_value = [match_keep, match_drop]

    rag_config = RAGConfig(
        compression_enabled=True,
        compression_similarity_threshold=0.7,
    )
    retriever = VectorServiceRetriever(
        vector_store,
        "docs",
        rag_config=rag_config,
    )

    documents = await retriever._aget_relevant_documents(
        "retain important",
        run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
    )

    assert len(documents) == 1
    assert documents[0].metadata["source_id"] == "doc-keep"
    stats = retriever.compression_stats
    assert stats is not None
    assert stats.documents_compressed == 1
    assert stats.tokens_after <= stats.tokens_before


def test_get_relevant_documents_sync_path_runs_event_loop(monkeypatch) -> None:
    """When no loop is running, the sync method should call asyncio.run."""

    vector_store = _make_vector_store_mock()
    retriever = VectorServiceRetriever(vector_store, "docs")

    def _raise_runtime_error() -> asyncio.AbstractEventLoop:
        raise RuntimeError()

    monkeypatch.setattr(asyncio, "get_running_loop", _raise_runtime_error)

    captured_query: list[str] = []

    async def fake_aget(query: str, *, run_manager) -> list[Document]:
        captured_query.append(query)
        return []

    monkeypatch.setattr(retriever, "_aget_relevant_documents", fake_aget)

    def fake_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch("asyncio.run", side_effect=fake_run) as run_mock:
        retriever._get_relevant_documents("hello")

    run_mock.assert_called_once()
    assert captured_query == ["hello"]


def test_get_relevant_documents_sync_raises_when_loop_active(monkeypatch) -> None:
    """If a loop is running, the sync helper should abort with RuntimeError."""

    vector_store = _make_vector_store_mock()
    retriever = VectorServiceRetriever(vector_store, "docs")

    def _get_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    monkeypatch.setattr(asyncio, "get_running_loop", _get_loop)

    with pytest.raises(RuntimeError):
        retriever._get_relevant_documents("hello")
