"""LangChain powered Retrieval-Augmented Generation service."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from src.services.base import BaseService
from src.services.errors import EmbeddingServiceError

from .models import (
    AnswerMetrics,
    RAGConfig,
    RAGRequest,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)


logger = logging.getLogger(__name__)


class RAGGenerator(BaseService):
    """Generate answers from retrieved context using LangChain primitives."""

    _DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant that answers questions using "
                    "the supplied context. Cite sources using [source_id] markers. "
                    "If the context is insufficient reply that the answer is unknown."
                ),
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
            ),
        ]
    )

    def __init__(
        self,
        config: RAGConfig,
        retriever: BaseRetriever,
        chat_model: ChatOpenAI | None = None,
    ) -> None:
        super().__init__(None)
        self._config = config
        self._retriever = retriever
        self._chat_model = chat_model
        self._generation_count = 0
        self._total_latency_ms = 0.0

    @property
    def config(self) -> RAGConfig:
        """Return the generator configuration."""

        return self._config

    @property
    def retriever(self) -> BaseRetriever:
        """Expose the underlying retriever instance."""

        return self._retriever

    async def initialize(self) -> None:
        """Prepare the chat model and executable chain."""

        if self.is_initialized():
            return

        if self._chat_model is None:
            params: dict[str, Any] = {
                "model": self._config.model,
                "temperature": self._config.temperature,
            }
            if self._config.max_tokens is not None:
                params["max_tokens"] = self._config.max_tokens
            self._chat_model = cast(ChatOpenAI, ChatOpenAI(**params))  # type: ignore[call-arg]

        self._mark_initialized()
        logger.info("RAG generator initialised using LangChain components")

    async def cleanup(self) -> None:
        """Release cached model instances."""

        self._chat_model = None
        self._mark_uninitialized()

    @property
    def llm_client_available(self) -> bool:
        """Indicate whether the underlying chat model is initialised."""

        return self._chat_model is not None

    def clear_cache(self) -> None:
        """No-op retained for compatibility with management endpoints."""

        logger.debug("RAGGenerator.clear_cache invoked; no cache to purge")

    async def generate_answer(  # pylint: disable=too-many-locals
        self, request: RAGRequest
    ) -> RAGResult:
        """Generate an answer based on retrieval results."""

        self._validate_initialized()

        llm = self._chat_model
        if llm is None:
            raise EmbeddingServiceError("Chat model is not initialised")

        start_time = time.perf_counter()
        max_tokens = request.max_tokens or self._config.max_tokens
        temperature = request.temperature or self._config.temperature
        include_sources = (
            request.include_sources
            if request.include_sources is not None
            else self._config.include_sources
        )

        context_documents = await self._collect_documents(request, include_sources)
        if not context_documents:
            msg = "Retriever returned no documents for the supplied query"
            raise EmbeddingServiceError(msg)

        context_block = self._render_context(context_documents, include_sources)
        if not context_block.strip():
            msg = "No usable context available for RAG generation"
            raise EmbeddingServiceError(msg)

        if request.max_tokens is not None or request.temperature is not None:
            override_params: dict[str, Any] = {
                "model": self._config.model,
                "temperature": temperature,
            }
            if max_tokens is not None:
                override_params["max_tokens"] = max_tokens
            llm = cast(ChatOpenAI, ChatOpenAI(**override_params))  # type: ignore[call-arg]

        chain = self._DEFAULT_PROMPT | llm
        try:
            message = await chain.ainvoke(
                {"context": context_block, "question": request.query}
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("RAG generation failed")
            msg = f"RAG generation failed: {exc}"
            raise EmbeddingServiceError(msg) from exc

        raw_content = getattr(message, "content", "")
        if isinstance(raw_content, list):
            response_text = "".join(str(part) for part in raw_content)
        else:
            response_text = str(raw_content)

        latency_ms = (time.perf_counter() - start_time) * 1000
        sources = self._build_sources(context_documents) if include_sources else []
        confidence = self._derive_confidence(context_documents) if sources else None
        metrics = self._build_metrics(
            latency_ms, getattr(message, "response_metadata", {})
        )

        self._generation_count += 1
        self._total_latency_ms += latency_ms

        return RAGResult(
            answer=response_text.strip(),
            confidence_score=confidence,
            sources=sources,
            generation_time_ms=latency_ms,
            metrics=metrics,
        )

    def get_metrics(self) -> RAGServiceMetrics:
        """Return aggregated generation metrics for observability endpoints."""

        avg_time = (
            self._total_latency_ms / self._generation_count
            if self._generation_count
            else None
        )
        return RAGServiceMetrics(
            generation_count=self._generation_count,
            avg_generation_time_ms=avg_time,
            total_generation_time_ms=self._total_latency_ms,
        )

    async def _collect_documents(
        self, request: RAGRequest, include_sources: bool
    ) -> list[Document]:
        """Retrieve context documents using the configured LangChain retriever."""

        search_kwargs: dict[str, Any] = {
            "k": request.top_k or self._config.retriever_top_k
        }
        if request.filters:
            search_kwargs["filters"] = request.filters

        retriever = self._retriever
        with_kwargs = getattr(retriever, "with_search_kwargs", None)
        if callable(with_kwargs):
            retriever = cast(BaseRetriever, with_kwargs(**search_kwargs))  # type: ignore[call-arg]

        documents_raw = await retriever.ainvoke(request.query)
        documents: list[Document] = []
        for item in cast(
            Sequence[Any],
            documents_raw if isinstance(documents_raw, Sequence) else [documents_raw],
        ):
            if isinstance(item, Document):
                documents.append(item)
            else:
                documents.append(Document(page_content=str(item), metadata={}))

        if include_sources:
            for document in documents:
                document.metadata.setdefault("source_id", document.metadata.get("id"))

        return documents

    @staticmethod
    def _build_metrics(latency_ms: float, metadata: dict[str, Any]) -> AnswerMetrics:
        """Construct token usage metrics from LLM response metadata."""

        usage = metadata.get("token_usage") or metadata.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

        return AnswerMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generation_time_ms=latency_ms,
        )

    @staticmethod
    def _render_context(documents: Iterable[Document], include_sources: bool) -> str:
        """Render the context block supplied to the chat prompt."""

        lines: list[str] = []
        for idx, document in enumerate(documents, start=1):
            meta = document.metadata
            title = str(meta.get("title") or f"Document {idx}")
            header = f"[source_{idx}] {title}"
            lines.append(header)
            lines.append(document.page_content.strip())
            if include_sources and meta.get("url"):
                lines.append(f"URL: {meta['url']}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_sources(documents: Iterable[Document]) -> list[SourceAttribution]:
        """Create structured source attributions returned to the caller."""

        sources: list[SourceAttribution] = []
        for document in documents:
            meta = document.metadata
            sources.append(
                SourceAttribution(
                    source_id=str(meta.get("source_id") or meta.get("id") or "unknown"),
                    title=meta.get("title") or "Untitled source",
                    url=meta.get("url"),
                    excerpt=meta.get("excerpt"),
                    score=_normalize_score(meta.get("score")),
                )
            )
        return sources

    def _derive_confidence(self, documents: Iterable[Document]) -> float | None:
        """Derive a confidence score from supplied retrieval metadata."""

        if not self._config.confidence_from_scores:
            return None

        scores = []
        for doc in documents:
            score = doc.metadata.get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        if not scores:
            return None
        normalised = [(score + 1) / 2 for score in scores]
        return max(0.0, min(1.0, sum(normalised) / len(normalised)))


def _normalize_score(value: Any) -> float | None:
    """Normalise scores to the [0, 1] range when possible."""

    if isinstance(value, (int, float)):
        if -1.0 <= value <= 1.0:
            return (float(value) + 1) / 2
        if 0.0 <= value <= 1.0:
            return float(value)
    return None
