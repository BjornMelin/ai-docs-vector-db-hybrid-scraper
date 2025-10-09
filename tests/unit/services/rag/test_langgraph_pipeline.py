"""Tests for the LangGraph-backed RAG pipeline."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.contracts.retrieval import SearchRecord
from src.services.query_processing.models import SearchRequest
from src.services.rag import (
    LangGraphRAGPipeline,
    RAGConfig,
    RAGGenerator,
    RAGRequest,
    RAGResult,
    RagTracingCallback,
    SourceAttribution,
)
from src.services.rag.langgraph_pipeline import _StaticDocumentRetriever
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import VectorMatch


pytest.importorskip("langgraph")


class VectorServiceFake(VectorStoreService):
    """Vector service stub returning deterministic matches."""

    def __init__(self, matches: list[VectorMatch]) -> None:
        super().__init__()
        self._matches = matches
        self.config = SimpleNamespace(
            fastembed=SimpleNamespace(model="BAAI/bge-small-en-v1.5")
        )

    async def initialize(self) -> None:  # pragma: no cover - simple stub
        self._mark_initialized()

    async def cleanup(self) -> None:  # pragma: no cover - simple stub
        self._mark_uninitialized()

    async def search_documents(
        self,
        collection: str,
        query: str,
        **_: Any,
    ) -> list[VectorMatch]:
        return list(self._matches)

    async def list_collections(self) -> list[str]:  # pragma: no cover
        return ["docs"]


def dummy_generator_factory(
    config: RAGConfig, retriever: BaseRetriever
) -> DummyGenerator:
    """Return a dummy generator for pipeline tests."""

    return DummyGenerator(config, retriever)


class DummyGenerator(RAGGenerator):
    """Generator stub producing static answers without LLM calls."""

    last_callbacks: Sequence[BaseCallbackHandler] | None = None

    def __init__(self, config: RAGConfig, retriever: BaseRetriever) -> None:
        super().__init__(config=config, retriever=retriever)

    async def initialize(self) -> None:  # pragma: no cover - trivial
        self._mark_initialized()

    async def cleanup(self) -> None:  # pragma: no cover - trivial
        self._mark_uninitialized()

    def register_callbacks(
        self, callbacks: Sequence[BaseCallbackHandler]
    ) -> None:  # pragma: no cover - trivial
        super().register_callbacks(callbacks)
        DummyGenerator.last_callbacks = list(callbacks)

    async def generate_answer(self, request: RAGRequest) -> RAGResult:
        document = None
        if isinstance(self._retriever, _StaticDocumentRetriever):  # noqa: SLF001
            document = next(iter(self._retriever._documents), None)  # noqa: SLF001
        source = SourceAttribution(
            source_id=str(document.metadata.get("source_id", "missing"))
            if document
            else "missing",
            title=str(document.metadata.get("title", "")) if document else "",
            url=document.metadata.get("url") if document else None,
            excerpt=document.page_content if document else "",
            score=float(document.metadata.get("score", 0.0)) if document else 0.0,
        )
        return RAGResult(
            answer=f"Answer to {request.query}",
            confidence_score=0.9,
            sources=[source],
            generation_time_ms=1.0,
            metrics=None,
        )


@pytest.mark.asyncio
async def test_pipeline_returns_answer_with_retriever_results() -> None:
    match = VectorMatch(
        id="doc-1",
        score=0.9,
        raw_score=0.9,
        payload={
            "content": "LangGraph overview",
            "title": "LangGraph",
            "url": "https://example.test/langgraph",
            "_grouping": {"applied": True, "group_id": "doc-1", "rank": 1},
        },
        collection="docs",
    )
    service = VectorServiceFake([match])
    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, service),
        generator_factory=dummy_generator_factory,
    )

    request = SearchRequest(query="langgraph", limit=5)
    rag_config = RAGConfig(compression_enabled=False)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
    result = await pipeline.run(
        query="langgraph",
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    assert not caught_warnings
    assert result is not None
    result_dict = cast(dict[str, Any], result)
    assert result_dict["answer"] == "Answer to langgraph"
    assert result_dict["confidence"] == pytest.approx(0.9)
    assert result_dict["sources"] == [
        {
            "source_id": "doc-1",
            "title": "LangGraph",
            "url": "https://example.test/langgraph",
            "excerpt": "LangGraph overview",
            "score": 0.9,
        }
    ]
    assert DummyGenerator.last_callbacks is not None


@pytest.mark.asyncio
async def test_pipeline_uses_prefetched_records_when_retrieval_empty() -> None:
    service = VectorServiceFake([])
    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, service),
        generator_factory=dummy_generator_factory,
    )

    search_record = SearchRecord(
        id="seed-1",
        content="Vector DB primer",
        title="Primer",
        url=None,
        metadata={"score": 0.6},
        score=0.6,
        raw_score=0.6,
        normalized_score=0.6,
        collection="docs",
        group_id=None,
        group_rank=None,
        grouping_applied=False,
    )

    request = SearchRequest(query="primer", limit=3)
    rag_config = RAGConfig(compression_enabled=False)

    result = await pipeline.run(
        query="primer",
        request=request,
        rag_config=rag_config,
        collection="docs",
        prefetched_records=[search_record],
    )

    assert result is not None
    result_dict = cast(dict[str, Any], result)
    assert result_dict["answer"].startswith("Answer to primer")
    sources = cast(list[dict[str, Any]], result_dict["sources"])
    assert sources[0]["source_id"] == "seed-1"


def test_rag_tracing_callback_records_token_usage() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    callback = RagTracingCallback(tracer)

    callback.on_llm_start({"name": "gpt-4o-mini"}, ["prompt"], run_id="run-1")

    class _Response:
        llm_output = {
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            }
        }

    callback.on_llm_end(_Response(), run_id="run-1")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "rag.llm"
    attributes = span.attributes
    assert attributes is not None
    assert attributes["rag.llm.prompt_count"] == 1
    assert attributes["rag.llm.model"] == "gpt-4o-mini"
    assert attributes["rag.llm.prompt_tokens"] == 5
    assert attributes["rag.llm.completion_tokens"] == 7
    assert attributes["rag.llm.total_tokens"] == 12


@pytest.mark.asyncio
async def test_pipeline_returns_generation_payload(monkeypatch) -> None:
    match = VectorMatch(
        id="doc-1",
        score=0.9,
        raw_score=0.9,
        payload={
            "content": "LangGraph overview",
            "title": "LangGraph",
            "url": "https://example.test/langgraph",
            "_grouping": {"applied": True, "group_id": "doc-1", "rank": 1},
        },
        collection="docs",
    )
    service = VectorServiceFake([match])
    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, service),
        generator_factory=dummy_generator_factory,
    )

    request = SearchRequest(query="langgraph", limit=5)
    rag_config = RAGConfig(compression_enabled=False)

    result = await pipeline.run(
        query="langgraph",
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    assert result is not None
    assert result["answer"].startswith("Answer for")
    assert result["confidence"] == pytest.approx(0.87)
    assert len(result["sources"]) == 2


@pytest.mark.asyncio
async def test_pipeline_populates_retriever_compression_stats(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.rag.langgraph_pipeline.LangGraphRAGPipeline._build_compressor",
        lambda self, config: DocumentCompressorPipeline(transformers=[]),
    )

    captured: dict[str, SimpleNamespace] = {}

    async def fake_maybe_compress(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        stats = SimpleNamespace(
            tokens_before=10,
            tokens_after=5,
            documents_processed=len(documents),
            documents_compressed=max(0, len(documents) - 1),
        )
        self._compression_stats = stats
        captured["stats"] = stats
        return documents

    monkeypatch.setattr(
        "src.services.rag.langgraph_pipeline.VectorServiceRetriever._maybe_compress_documents",
        fake_maybe_compress,
    )

    match = VectorMatch(
        id="doc-1",
        score=0.9,
        raw_score=0.9,
        payload={
            "content": "LangGraph overview",
            "title": "LangGraph",
            "url": "https://example.test/langgraph",
            "_grouping": {"applied": True, "group_id": "doc-1", "rank": 1},
        },
        collection="docs",
    )
    service = VectorServiceFake([match])
    pipeline = LangGraphRAGPipeline(
        cast(VectorStoreService, service),
        generator_factory=dummy_generator_factory,
    )

    request = SearchRequest(query="compress", limit=3)
    rag_config = RAGConfig(compression_enabled=True)

    await pipeline.run(
        query="compress",
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    stats = captured.get("stats")
    assert stats is not None
    assert stats.tokens_before == 10
    assert stats.tokens_after == 5
