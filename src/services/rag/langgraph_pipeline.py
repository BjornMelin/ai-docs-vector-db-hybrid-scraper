"""LangGraph-based RAG pipeline orchestrating retrieval, grading, and generation."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NotRequired, Required, TypedDict

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry import trace

from src.contracts.retrieval import SearchRecord
from src.services.query_processing.models import SearchRequest
from src.services.vector_db.service import VectorStoreService

from .generator import RAGGenerator
from .models import RAGConfig, RAGRequest, RAGResult
from .retriever import VectorServiceRetriever


try:  # pragma: no cover - optional dependency guard
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
except ModuleNotFoundError:  # pragma: no cover
    FastEmbedEmbeddings = None  # type: ignore[assignment]


try:  # pragma: no cover - optional dependency guard
    from langgraph.graph import StateGraph

    LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _build_callback_handlers(tracer: trace.Tracer) -> list[BaseCallbackHandler]:
    return [RagTracingCallback(tracer)]


GeneratorFactory = Callable[[RAGConfig, BaseRetriever], RAGGenerator]


class RagTracingCallback(BaseCallbackHandler):
    """LangChain callback that emits OpenTelemetry spans for LLM runs."""

    def __init__(self, tracer: trace.Tracer) -> None:
        self._tracer = tracer
        self._spans: dict[str, trace.Span] = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **_: Any
    ) -> None:
        span = self._tracer.start_span("rag.llm")
        span.set_attribute("rag.llm.prompt_count", len(prompts))
        span.set_attribute("rag.llm.model", serialized.get("name", "unknown"))
        self._spans[str(run_id)] = span

    def on_llm_end(self, response: Any, *, run_id: Any, **_: Any) -> None:
        span = self._spans.pop(str(run_id), None)
        if span is None:
            return
        llm_output = getattr(response, "llm_output", {}) or {}
        token_usage = llm_output.get("token_usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = token_usage.get(key)
            if value is not None:
                span.set_attribute(f"rag.llm.{key}", value)
        span.end()


class _RAGGraphState(TypedDict):
    """Internal LangGraph state for the RAG pipeline."""

    query: Required[str]
    documents: NotRequired[list[Document]]
    graded_documents: NotRequired[list[Document]]
    prefetched_documents: NotRequired[list[Document]]
    confidence: NotRequired[float | None]
    answer: NotRequired[str]
    answer_sources: NotRequired[list[dict[str, Any]]]


class _StaticDocumentRetriever(BaseRetriever):
    """Retriever that serves a static list of documents for generation."""

    def __init__(self, documents: Sequence[Document]) -> None:
        super().__init__()
        self._documents = list(documents)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:  # type: ignore[override]
        return list(self._documents)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:  # type: ignore[override]
        return list(self._documents)


class LangGraphRAGPipeline:
    """Construct and execute a LangGraph RAG pipeline for query responses."""

    def __init__(
        self,
        vector_service: VectorStoreService,
        *,
        generator_factory: GeneratorFactory | None = None,
    ) -> None:
        self._vector_service = vector_service
        self._generator_factory = generator_factory or self._default_generator
        self._tracer = trace.get_tracer(__name__)
        self._callback_handlers = _build_callback_handlers(self._tracer)

    @staticmethod
    def _default_generator(config: RAGConfig, retriever: BaseRetriever) -> RAGGenerator:
        """Factory that instantiates the default RAG generator."""

        return RAGGenerator(config, retriever)

    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    async def run(
        self,
        *,
        query: str,
        request: SearchRequest,
        rag_config: RAGConfig,
        collection: str,
        prefetched_records: Sequence[SearchRecord] | None = None,
    ) -> dict[str, Any] | None:
        """Execute the LangGraph pipeline and return structured outputs."""

        if not LANGGRAPH_AVAILABLE:
            msg = "LangGraph must be installed to run the LangGraphRAGPipeline"
            raise RuntimeError(msg)

        assert StateGraph is not None  # for type-checkers; guarded above

        with self._tracer.start_as_current_span("rag.pipeline") as pipeline_span:
            pipeline_span.set_attribute("rag.collection", collection)
            pipeline_span.set_attribute("rag.query.length", len(query))
            pipeline_span.set_attribute(
                "rag.prefetched.count", len(prefetched_records or [])
            )

            effective_config = rag_config.model_copy(
                update={
                    "retriever_top_k": request.rag_top_k or rag_config.retriever_top_k,
                    "max_tokens": request.rag_max_tokens or rag_config.max_tokens,
                }
            )
            base_config = effective_config.model_copy(
                update={"compression_enabled": False}
            )

            retriever = VectorServiceRetriever(
                vector_service=self._vector_service,
                collection=collection,
                k=effective_config.retriever_top_k,
                filters=request.filters,
                rag_config=base_config,
            )

            compressor = self._build_compressor(effective_config)
            contextual_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever,
            )
            pipeline_span.set_attribute(
                "rag.retriever_tool", "vector_context_retriever"
            )

            prefetched_documents = self._records_to_documents(prefetched_records or [])
            tracer = self._tracer

            async def retrieve_node(state: _RAGGraphState) -> dict[str, Any]:
                with tracer.start_as_current_span("rag.retrieve") as span:
                    documents = await contextual_retriever.ainvoke(state["query"])
                    used_prefetched = False
                    if not documents:
                        documents = list(state.get("prefetched_documents") or [])
                        used_prefetched = bool(documents)
                    span.set_attribute("rag.retrieve.documents", len(documents))
                    span.set_attribute(
                        "rag.retrieve.prefetched_used", int(used_prefetched)
                    )
                    stats = getattr(retriever, "compression_stats", None)
                    if documents:
                        for key, value in self._iter_items(stats):
                            span.set_attribute(f"rag.compress.{key}", value)
                return {"documents": documents}

            async def grade_node(state: _RAGGraphState) -> dict[str, Any]:
                with tracer.start_as_current_span("rag.grade") as span:
                    documents = state.get("documents") or []
                    if not documents:
                        span.set_attribute("rag.grade.input_count", 0)
                        return {"graded_documents": []}
                    scored_docs = [
                        (doc, self._document_score(doc)) for doc in documents
                    ]
                    scored_docs.sort(key=lambda item: item[1], reverse=True)
                    span.set_attribute("rag.grade.input_count", len(scored_docs))
                    threshold = 0.10
                    filtered_docs = [
                        doc
                        for doc, score_value in scored_docs
                        if score_value >= threshold
                    ]
                    if not filtered_docs and scored_docs:
                        filtered_docs = [scored_docs[0][0]]
                    confidence = self._compute_confidence(filtered_docs)
                    span.set_attribute("rag.grade.filtered_count", len(filtered_docs))
                    if confidence is not None:
                        span.set_attribute("rag.grade.confidence.heuristic", confidence)
                return {"graded_documents": filtered_docs, "confidence": confidence}

            async def generate_node(state: _RAGGraphState) -> dict[str, Any]:
                with tracer.start_as_current_span("rag.generate") as span:
                    documents = state.get("graded_documents") or []
                    span.set_attribute("rag.generate.context_count", len(documents))
                    if not documents:
                        logger.debug(
                            "LangGraph RAG pipeline skipped generation due to empty "
                            "context"
                        )
                        return {}
                    static_retriever = _StaticDocumentRetriever(documents)
                    generator = self._generator_factory(
                        effective_config, static_retriever
                    )
                    if self._callback_handlers and hasattr(
                        generator, "register_callbacks"
                    ):
                        generator.register_callbacks(self._callback_handlers)
                    await generator.initialize()
                    try:
                        rag_request = RAGRequest(
                            query=query,
                            top_k=len(documents),
                            filters=request.filters,
                            max_tokens=effective_config.max_tokens,
                            temperature=effective_config.temperature,
                            include_sources=effective_config.include_sources,
                        )
                        rag_result: RAGResult = await generator.generate_answer(
                            rag_request
                        )
                    except Exception:  # pragma: no cover - defensive guard
                        logger.exception("LangGraph RAG generation failed")
                        span.set_attribute("rag.generate.error", True)
                        return {}
                    finally:
                        await generator.cleanup()

                    sources = [source.model_dump() for source in rag_result.sources]
                    confidence = rag_result.confidence_score
                    if confidence is None:
                        confidence = state.get("confidence")
                    if confidence is not None:
                        span.set_attribute("rag.generate.confidence", confidence)
                    span.set_attribute(
                        "rag.generate.answer_length", len(rag_result.answer)
                    )
                    span.set_attribute(
                        "rag.generate.latency_ms", rag_result.generation_time_ms
                    )
                    for key, value in self._iter_items(rag_result.metrics):
                        span.set_attribute(f"rag.generate.{key}", value)
                    return {
                        "answer": rag_result.answer.strip(),
                        "answer_sources": sources,
                        "confidence": confidence,
                    }

            workflow = StateGraph(_RAGGraphState)
            workflow.add_node("retrieve", retrieve_node)
            workflow.add_node("grade", grade_node)
            workflow.add_node("generate", generate_node)
            workflow.add_edge("retrieve", "grade")
            workflow.add_edge("grade", "generate")
            workflow.set_entry_point("retrieve")

            app = workflow.compile(checkpointer=None)
            final_state = await app.ainvoke(
                {
                    "query": query,
                    "prefetched_documents": prefetched_documents,
                }
            )

            answer = final_state.get("answer")
            if not answer:
                pipeline_span.set_attribute("rag.answer.generated", False)
                return None
            pipeline_span.set_attribute("rag.answer.generated", True)
            pipeline_span.set_attribute("rag.answer.length", len(answer))
            return {
                "answer": answer,
                "confidence": final_state.get("confidence"),
                "sources": final_state.get("answer_sources") or [],
            }

    def _build_compressor(self, config: RAGConfig) -> DocumentCompressorPipeline:
        """Create a document compressor pipeline based on configuration."""

        if not config.compression_enabled:
            return DocumentCompressorPipeline(transformers=[])

        if FastEmbedEmbeddings is None:  # pragma: no cover - optional dependency guard
            warnings.warn(
                "FastEmbedEmbeddings not available; disabling compression.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return DocumentCompressorPipeline(transformers=[])

        fastembed_config = getattr(self._vector_service.config, "fastembed", None)
        model_name = getattr(fastembed_config, "model", "BAAI/bge-small-en-v1.5")
        embeddings = FastEmbedEmbeddings(model_name=model_name)
        transformer = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=config.compression_similarity_threshold,
        )
        return DocumentCompressorPipeline(transformers=[transformer])

    @staticmethod
    def _iter_items(obj: Any) -> Iterable[tuple[str, Any]]:
        """Yield key/value pairs from mapping-like instrumentation payloads."""

        empty: tuple[tuple[str, Any], ...] = ()
        if obj is None:
            return empty
        if isinstance(obj, Mapping):
            return tuple((str(key), value) for key, value in obj.items())
        items_getter = getattr(obj, "items", None)
        if callable(items_getter):
            try:
                candidate = items_getter()
            except TypeError:
                return empty
            if isinstance(candidate, Iterable):
                pairs: list[tuple[str, Any]] = []
                for item in candidate:
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], str)
                    ):
                        pairs.append((item[0], item[1]))
                if pairs:
                    return tuple(pairs)
        return empty

    @staticmethod
    def _records_to_documents(records: Sequence[SearchRecord]) -> list[Document]:
        documents: list[Document] = []
        for record in records:
            metadata = dict(record.metadata or {})
            metadata.setdefault("source_id", record.id)
            metadata.setdefault("title", record.title)
            metadata.setdefault("url", record.url)
            score_value = (
                record.normalized_score
                if record.normalized_score is not None
                else record.score
            )
            if score_value is None:
                score_value = 0.0
            metadata.setdefault("score", float(score_value))
            documents.append(
                Document(page_content=record.content or "", metadata=metadata)
            )
        return documents

    @staticmethod
    def _document_score(document: Document) -> float:
        try:
            return float(document.metadata.get("score", 0.0))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0

    def _compute_confidence(self, documents: Sequence[Document]) -> float | None:
        scores = [
            self._document_score(doc)
            for doc in documents
            if self._document_score(doc) > 0
        ]
        if not scores:
            return None
        normalized = [max(0.0, min(1.0, score)) for score in scores]
        return sum(normalized) / len(normalized)


__all__ = ["LangGraphRAGPipeline"]
