"""Unit tests for the LangChain VectorServiceRetriever wrapper."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.services.rag.retriever import VectorServiceRetriever


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

    match = SimpleNamespace(
        id="doc-1",
        score=0.87,
        payload={"content": "lorem", "title": "Note"},
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
    retriever = VectorServiceRetriever(vector_store, "docs", k=3)

    overridden = retriever.with_search_kwargs(k=7, filters={"category": "api"})

    assert overridden is not retriever
    assert isinstance(overridden, VectorServiceRetriever)
    assert overridden._k == 7
    assert overridden._filters == {"category": "api"}
    # Ensure original instance unchanged
    assert retriever._k == 3
    assert retriever._filters is None


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
