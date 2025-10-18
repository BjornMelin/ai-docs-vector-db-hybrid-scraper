"""LangChain-backed FastEmbed provider implementation."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from src.services.errors import EmbeddingServiceError
from src.services.observability.tracing import trace_function
from src.services.observability.tracking import record_ai_operation

from .base import EmbeddingProvider


if TYPE_CHECKING:  # pragma: no cover - typing aid
    from langchain_community.embeddings.fastembed import (
        FastEmbedEmbeddings as _FastEmbedEmbeddings,
    )
    from langchain_qdrant import FastEmbedSparse as _FastEmbedSparse
else:  # pragma: no cover - runtime fallback
    _FastEmbedEmbeddings = Any
    _FastEmbedSparse = Any

# Runtime handles cached for reuse and monkeypatching in tests.
FastEmbedEmbeddings = cast(type[_FastEmbedEmbeddings] | None, None)
_SPARSE_RUNTIME_UNSET = object()
FastEmbedSparseRuntime = cast(
    type[_FastEmbedSparse] | None | object, _SPARSE_RUNTIME_UNSET
)

logger = logging.getLogger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    """Thin asynchronous wrapper around LangChain's FastEmbed embeddings."""

    _DEFAULT_SPARSE_MODEL = "qdrant/bm25"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        sparse_model: str | None = None,
        doc_embed_type: str = "default",
    ) -> None:
        """Configure the FastEmbed provider.

        Args:
            model_name: Dense embedding model identifier supported by FastEmbed.
            sparse_model: Optional sparse model identifier for hybrid retrieval.
            doc_embed_type: Embedding mode passed to LangChain (``default`` or
                ``passage``).
        """
        super().__init__(model_name)
        self._doc_embed_type = doc_embed_type
        self._dense: _FastEmbedEmbeddings | None = None
        self._sparse: _FastEmbedSparse | None = None
        self._sparse_model_name = sparse_model or self._DEFAULT_SPARSE_MODEL
        self._initialized = False

    @trace_function()
    async def initialize(self) -> None:
        """Initialize the underlying LangChain embeddings."""
        if self._initialized:
            return

        dense_cls = _load_fastembed_embeddings()
        self._dense = dense_cls(
            model_name=self.model_name,
            doc_embed_type=self._doc_embed_type,  # type: ignore[arg-type]
        )
        # Probe the dimension lazily on a background thread to avoid blocking the loop.
        probe = await asyncio.to_thread(self._dense.embed_query, "probe")
        self.dimensions = len(probe)
        self._initialized = True
        logger.info("FastEmbed provider initialized using LangChain components")

    async def cleanup(self) -> None:
        """Release cached embedding instances."""
        self._dense = None
        self._sparse = None
        self._initialized = False
        logger.debug("FastEmbed provider cleaned up")

    @property
    def langchain_embeddings(self) -> _FastEmbedEmbeddings:
        """Expose the underlying LangChain embedding instance."""
        if self._dense is None:
            msg = "FastEmbedProvider has not been initialized"
            raise EmbeddingServiceError(msg)
        return self._dense

    @trace_function()
    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate dense embeddings for the given texts."""
        if not self._initialized or self._dense is None:
            msg = "FastEmbedProvider has not been initialized"
            raise EmbeddingServiceError(msg)

        if not texts:
            return []

        dense_model = self._dense

        start = time.perf_counter()
        success = True

        async def _encode() -> list[list[float]]:
            return await asyncio.to_thread(dense_model.embed_documents, texts)

        try:
            return await _encode()
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            record_ai_operation(
                operation_type="embedding",
                provider="fastembed",
                model=self.model_name,
                duration_s=duration,
                tokens=None,
                cost_usd=0.0,
                success=success,
            )

    @trace_function()
    async def generate_sparse_embeddings(
        self, texts: Sequence[str]
    ) -> list[dict[str, Sequence[float]]]:
        """Generate sparse embeddings when supported."""
        if not texts:
            return []
        if not self._initialized:
            msg = "FastEmbedProvider has not been initialized"
            raise EmbeddingServiceError(msg)

        sparse_runtime = _load_fastembed_sparse_runtime()
        if sparse_runtime is None:
            msg = "langchain-qdrant is required for sparse embeddings"
            raise EmbeddingServiceError(msg)
        if self._sparse is None:
            self._sparse = sparse_runtime(self._sparse_model_name)

        start = time.perf_counter()
        success = True
        try:
            results = await asyncio.to_thread(self._sparse.embed_documents, list(texts))
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            record_ai_operation(
                operation_type="sparse_embedding",
                provider="fastembed",
                model=self._sparse_model_name,
                duration_s=duration,
                tokens=None,
                cost_usd=0.0,
                success=success,
            )

        serialized: list[dict[str, Sequence[float]]] = []
        for value in results:
            indices = getattr(value, "indices", None)
            scores = getattr(value, "values", None)
            if indices is None or scores is None:
                msg = "Sparse embedding payload missing indices/values"
                raise EmbeddingServiceError(msg)
            serialized.append({"indices": list(indices), "values": list(scores)})
        return serialized

    @property
    def cost_per_token(self) -> float:
        """Return the per-token cost for FastEmbed operations."""
        return 0.0

    @property
    def max_tokens_per_request(self) -> int:
        """Expose the FastEmbed max sequence length."""
        if self._dense is None:
            return 512
        return getattr(self._dense, "max_length", 512)


def _load_fastembed_embeddings() -> type[_FastEmbedEmbeddings]:
    """Import and cache the FastEmbed dense embedding implementation."""
    global FastEmbedEmbeddings  # pylint: disable=global-statement

    if FastEmbedEmbeddings is not None:
        return FastEmbedEmbeddings

    try:
        module = import_module("langchain_community.embeddings.fastembed")
    except ModuleNotFoundError as exc:
        msg = (
            "langchain-community with FastEmbed support is required. "
            "Install via `pip install langchain-community[fastembed]`."
        )
        raise EmbeddingServiceError(msg) from exc

    candidate = getattr(module, "FastEmbedEmbeddings", None)
    if candidate is None:
        msg = "FastEmbedEmbeddings not available in langchain_community."
        raise EmbeddingServiceError(msg)

    FastEmbedEmbeddings = cast(type[_FastEmbedEmbeddings], candidate)
    return FastEmbedEmbeddings


def _load_fastembed_sparse_runtime() -> type[_FastEmbedSparse] | None:
    """Import and cache the optional sparse embedding implementation."""
    global FastEmbedSparseRuntime  # pylint: disable=global-statement

    if FastEmbedSparseRuntime is None:
        return None
    if FastEmbedSparseRuntime is not _SPARSE_RUNTIME_UNSET:
        return cast(type[_FastEmbedSparse], FastEmbedSparseRuntime)

    try:
        module = import_module("langchain_qdrant")
    except ModuleNotFoundError:
        FastEmbedSparseRuntime = None
        return None

    candidate = getattr(module, "FastEmbedSparse", None)
    if candidate is None:
        FastEmbedSparseRuntime = None
        return None

    FastEmbedSparseRuntime = cast(type[_FastEmbedSparse], candidate)
    return FastEmbedSparseRuntime
