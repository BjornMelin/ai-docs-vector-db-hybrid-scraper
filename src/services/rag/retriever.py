"""LangChain retriever backed by the VectorStoreService adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, cast

from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    from langchain_community.document_transformers import EmbeddingsRedundantFilter
    from langchain_community.embeddings import FastEmbedEmbeddings
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FastEmbedEmbeddings requires the 'langchain-community' package."
    ) from exc

try:  # pragma: no cover - optional dependency
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal installs
    tiktoken = None  # type: ignore[assignment]

from src.services.monitoring.metrics import get_metrics_registry
from src.services.rag.models import RAGConfig
from src.services.vector_db.service import VectorStoreService


@dataclass(slots=True)
class CompressionStats:
    """Lightweight telemetry emitted after document compression."""

    documents_processed: int = 0
    documents_compressed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def reduction_ratio(self) -> float:
        """Return the token reduction ratio achieved by compression."""

        if self.tokens_before == 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)

    def to_dict(self) -> dict[str, float | int]:
        """Serialize statistics for logging or metrics exports."""

        return {
            "documents_processed": self.documents_processed,
            "documents_compressed": self.documents_compressed,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_ratio": self.reduction_ratio,
        }


class VectorServiceRetriever(BaseRetriever):
    """Expose VectorStoreService searches through the LangChain retriever API."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        vector_service: VectorStoreService,
        collection: str,
        *,
        k: int = 5,
        filters: Mapping[str, Any] | None = None,
        rag_config: RAGConfig | None = None,
    ) -> None:
        super().__init__()
        self._vector_service = vector_service
        self._collection = collection
        self._k = k
        self._filters: Mapping[str, Any] | None = filters
        self._rag_config_value = rag_config
        if rag_config and rag_config.compression_enabled:
            self._compression_pipeline = self._build_compression_pipeline(rag_config)
        else:
            self._compression_pipeline = None
        self._compression_stats: CompressionStats | None = None

    def with_search_kwargs(
        self,
        *,
        k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Self:
        """Return a new retriever with updated search configuration."""

        return self.__class__(
            self._vector_service,
            self._collection,
            k=k or self._k,
            filters=filters if filters is not None else self._filters,
            rag_config=self._rag_config_value,
        )

    @property
    def compression_stats(self) -> CompressionStats | None:
        """Return compression statistics from the most recent retrieval."""

        return self._compression_stats

    @property
    def _rag_config(self) -> RAGConfig | None:
        return self._rag_config_value

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            async_manager = AsyncCallbackManagerForRetrieverRun.get_noop_manager()
            return asyncio.run(
                self._aget_relevant_documents(query, run_manager=async_manager)
            )

        msg = (
            "VectorServiceRetriever does not support synchronous retrieval while "
            "an event loop is active; call 'ainvoke' instead."
        )
        raise RuntimeError(msg)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if not self._vector_service.is_initialized():
            await self._vector_service.initialize()

        matches = await self._vector_service.search_documents(
            self._collection,
            query,
            limit=self._k,
            filters=self._filters,
        )

        documents: list[Document] = []
        for match in matches:
            payload = dict(match.payload or {})
            content = str(payload.get("content") or payload.get("text") or "")
            metadata = payload.copy()
            metadata.setdefault("source_id", match.id)
            metadata.setdefault("score", match.score)
            metadata.setdefault("title", payload.get("title"))
            documents.append(Document(page_content=content, metadata=metadata))

        documents = await self._maybe_compress_documents(query, documents)

        await run_manager.on_retriever_end(documents)

        return documents

    def _build_compression_pipeline(
        self, rag_config: RAGConfig
    ) -> DocumentCompressorPipeline | None:
        if not rag_config.compression_enabled:
            self._rag_config_value = None
            return None

        fastembed_config = getattr(self._vector_service.config, "fastembed", None)
        model_name = getattr(fastembed_config, "model", None)
        embeddings = FastEmbedEmbeddings(
            model_name=cast(str, model_name or "BAAI/bge-small-en-v1.5")
        )

        if tiktoken is not None:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=max(50, rag_config.compression_absolute_max_tokens),
                chunk_overlap=0,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(50, rag_config.compression_absolute_max_tokens),
                chunk_overlap=0,
            )

        transformers = [
            splitter,
            EmbeddingsRedundantFilter(embeddings=embeddings),
            EmbeddingsFilter(
                embeddings=embeddings,
                similarity_threshold=rag_config.compression_similarity_threshold,
            ),
        ]

        pipeline = DocumentCompressorPipeline(transformers=transformers)
        self._rag_config_value = rag_config
        return pipeline

    async def _maybe_compress_documents(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        if self._compression_pipeline is None or not documents:
            self._compression_stats = None
            return documents

        stats = CompressionStats(documents_processed=len(documents))
        stats.tokens_before = sum(
            _estimate_tokens(doc.page_content) for doc in documents
        )

        try:
            compressed_docs = list(
                await self._compression_pipeline.acompress_documents(  # type: ignore[attr-defined]
                    documents,
                    query=query,
                )
            )
        except AttributeError:
            compressed_docs = list(
                self._compression_pipeline.compress_documents(  # type: ignore[attr-defined]
                    documents,
                    query=query,
                )
            )

        stats.tokens_after = sum(
            _estimate_tokens(doc.page_content) for doc in compressed_docs
        )
        stats.documents_compressed = max(
            0, stats.documents_processed - len(compressed_docs)
        )
        self._compression_stats = stats

        try:
            registry = get_metrics_registry()
        except RuntimeError:  # pragma: no cover - monitoring disabled
            registry = None
        if registry is not None:
            registry.record_compression_stats(self._collection, stats)

        return compressed_docs


def _estimate_tokens(content: str) -> int:
    """Estimate token count using tiktoken when available."""

    if not content:
        return 0
    if tiktoken is not None:  # pragma: no branch - quick exit when installed
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(content))
    return max(1, len(content.split()))


__all__ = ["VectorServiceRetriever", "CompressionStats"]
