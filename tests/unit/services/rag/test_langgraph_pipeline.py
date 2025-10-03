"""Tests for the LangGraph-backed RAG pipeline."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.documents import Document
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


pytest.importorskip("langgraph")

from src.contracts.retrieval import SearchRecord
from src.services.query_processing.models import SearchRequest
from src.services.rag import RAGConfig, RAGResult, SourceAttribution
from src.services.rag.langgraph_pipeline import (
    LangGraphRAGPipeline,
    RagTracingCallback,
)
from src.services.vector_db.types import VectorMatch


class VectorServiceFake:
    """Vector service stub returning deterministic matches."""

    def __init__(self, matches: list[VectorMatch]) -> None:
        self._matches = matches
        self._initialised = False
        self.config = SimpleNamespace(
            fastembed=SimpleNamespace(model="BAAI/bge-small-en-v1.5")
        )

    async def initialize(self) -> None:  # pragma: no cover - simple stub
        self._initialised = True

    async def cleanup(self) -> None:  # pragma: no cover - simple stub
        self._initialised = False

    def is_initialized(self) -> bool:
        return self._initialised

    async def search_documents(
        self,
        collection: str,
        query: str,
        **_: Any,
    ) -> list[VectorMatch]:
        return list(self._matches)

    async def list_collections(self) -> list[str]:  # pragma: no cover
        return ["docs"]


class DummyGenerator:
    """Generator stub producing static answers without LLM calls."""

    last_callbacks: list[Any] | None = None

    def __init__(self, config: RAGConfig, retriever: Any) -> None:  # noqa: D401
        self._config = config
        self._documents = list(getattr(retriever, "_documents", []))
        self.callbacks = None

    def register_callbacks(self, callbacks) -> None:  # pragma: no cover - trivial
        self.callbacks = list(callbacks)
        DummyGenerator.last_callbacks = self.callbacks

    async def initialize(self) -> None:  # pragma: no cover - trivial
        return None

    async def cleanup(self) -> None:  # pragma: no cover - trivial
        return None

    async def generate_answer(self, request) -> RAGResult:
        doc = self._documents[0] if self._documents else None
        source = SourceAttribution(
            source_id=str(doc.metadata.get("source_id", "missing"))
            if doc
            else "missing",
            title=str(doc.metadata.get("title", "")) if doc else "",
            url=doc.metadata.get("url") if doc else None,
            excerpt=doc.page_content if doc else "",
            score=float(doc.metadata.get("score", 0.0)) if doc else 0.0,
        )
        answer_text = f"Answer to {request.query}"
        return RAGResult(
            answer=answer_text,
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
    pipeline = LangGraphRAGPipeline(service, generator_factory=DummyGenerator)

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
    assert result["answer"] == "Answer to langgraph"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["sources"] == [
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
    pipeline = LangGraphRAGPipeline(service, generator_factory=DummyGenerator)

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
    assert result["answer"].startswith("Answer to primer")
    assert result["sources"][0]["source_id"] == "seed-1"


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
    assert span.attributes["rag.llm.prompt_count"] == 1
    assert span.attributes["rag.llm.model"] == "gpt-4o-mini"
    assert span.attributes["rag.llm.prompt_tokens"] == 5
    assert span.attributes["rag.llm.completion_tokens"] == 7
    assert span.attributes["rag.llm.total_tokens"] == 12


@pytest.mark.asyncio
async def test_pipeline_emits_metrics_when_registry_available(monkeypatch) -> None:
    class MetricsStub:
        def __init__(self) -> None:
            self.generation_calls: list[dict[str, Any]] = []
            self.compression_calls: list[tuple[str, Any]] = []

        def record_rag_generation_stats(self, **kwargs: Any) -> None:
            self.generation_calls.append(kwargs)

        def record_compression_stats(self, collection: str, stats: Any) -> None:
            self.compression_calls.append((collection, stats))

    metrics_stub = MetricsStub()
    monkeypatch.setattr(
        "src.services.rag.langgraph_pipeline._safe_get_metrics_registry",
        lambda: metrics_stub,
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
    pipeline = LangGraphRAGPipeline(service, generator_factory=DummyGenerator)

    request = SearchRequest(query="langgraph", limit=5)
    rag_config = RAGConfig(compression_enabled=False)

    await pipeline.run(
        query="langgraph",
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    assert len(metrics_stub.generation_calls) == 1
    call = metrics_stub.generation_calls[0]
    assert call["collection"] == "docs"
    assert call["model"] == rag_config.model
    assert metrics_stub.compression_calls == []


@pytest.mark.asyncio
async def test_pipeline_records_compression_metrics(monkeypatch) -> None:
    class MetricsStub:
        def __init__(self) -> None:
            self.generation_calls: list[dict[str, Any]] = []
            self.compression_calls: list[tuple[str, Any]] = []

        def record_rag_generation_stats(self, **kwargs: Any) -> None:
            self.generation_calls.append(kwargs)

        def record_compression_stats(self, collection: str, stats: Any) -> None:
            self.compression_calls.append((collection, stats))

    metrics_stub = MetricsStub()
    monkeypatch.setattr(
        "src.services.rag.langgraph_pipeline._safe_get_metrics_registry",
        lambda: metrics_stub,
    )
    monkeypatch.setattr(
        "src.services.rag.langgraph_pipeline.LangGraphRAGPipeline._build_compressor",
        lambda self, config: DocumentCompressorPipeline(transformers=[]),
    )

    async def fake_maybe_compress(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        self._compression_stats = SimpleNamespace(
            tokens_before=10,
            tokens_after=5,
            documents_processed=len(documents),
            documents_compressed=max(0, len(documents) - 1),
        )
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
    pipeline = LangGraphRAGPipeline(service, generator_factory=DummyGenerator)

    request = SearchRequest(query="compress", limit=3)
    rag_config = RAGConfig(compression_enabled=True)

    await pipeline.run(
        query="compress",
        request=request,
        rag_config=rag_config,
        collection="docs",
    )

    assert len(metrics_stub.compression_calls) == 1
    collection, stats = metrics_stub.compression_calls[0]
    assert collection == "docs"
    assert stats.tokens_before == 10
    assert stats.tokens_after == 5
    assert stats.documents_processed == 1
    assert stats.documents_compressed == 0
