"""LangChain-backed FastEmbed provider implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from src.services.errors import EmbeddingServiceError
from src.services.monitoring.metrics import get_metrics_registry

from .base import EmbeddingProvider


if TYPE_CHECKING:  # pragma: no cover - typing aid
    from langchain_qdrant import FastEmbedSparse as FastEmbedSparseType
else:  # pragma: no cover - runtime fallback
    FastEmbedSparseType = Any

try:  # pragma: no cover - optional sparse dependency
    from langchain_qdrant import FastEmbedSparse as FastEmbedSparseRuntime
except ModuleNotFoundError:  # pragma: no cover - defer sparse usage checks
    FastEmbedSparseRuntime = None  # type: ignore[assignment]


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
        self._dense: FastEmbedEmbeddings | None = None
        self._sparse: FastEmbedSparseType | None = None
        self._sparse_model_name = sparse_model or self._DEFAULT_SPARSE_MODEL
        self._initialized = False
        self._metrics = None
        try:
            self._metrics = get_metrics_registry()
        except RuntimeError:  # pragma: no cover - monitoring optional
            self._metrics = None

    async def initialize(self) -> None:
        """Initialise the underlying LangChain embeddings."""

        if self._initialized:
            return

        self._dense = FastEmbedEmbeddings(
            model_name=self.model_name,
            doc_embed_type=self._doc_embed_type,  # type: ignore[arg-type]
        )
        # Probe the dimension lazily on a background thread to avoid blocking the loop.
        probe = await asyncio.to_thread(self._dense.embed_query, "probe")
        self.dimensions = len(probe)
        self._initialized = True
        logger.info("FastEmbed provider initialised using LangChain components")

    async def cleanup(self) -> None:
        """Release cached embedding instances."""

        self._dense = None
        self._sparse = None
        self._initialized = False
        logger.debug("FastEmbed provider cleaned up")

    @property
    def langchain_embeddings(self) -> FastEmbedEmbeddings:
        """Expose the underlying LangChain embedding instance."""

        if self._dense is None:
            msg = "FastEmbedProvider has not been initialised"
            raise EmbeddingServiceError(msg)
        return self._dense

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate dense embeddings for the given texts."""

        if not self._initialized or self._dense is None:
            msg = "FastEmbedProvider has not been initialised"
            raise EmbeddingServiceError(msg)

        if not texts:
            return []

        dense_model = self._dense

        async def _encode() -> list[list[float]]:
            return await asyncio.to_thread(dense_model.embed_documents, texts)

        if self._metrics is None:
            return await _encode()

        decorator = self._metrics.monitor_embedding_generation(
            provider="fastembed", model=self.model_name
        )
        return await decorator(_encode)()

    async def generate_sparse_embeddings(
        self, texts: Sequence[str]
    ) -> list[dict[str, Sequence[float]]]:
        """Generate sparse embeddings when supported."""

        if not texts:
            return []
        if not self._initialized:
            msg = "FastEmbedProvider has not been initialised"
            raise EmbeddingServiceError(msg)
        if FastEmbedSparseRuntime is None:
            msg = "langchain-qdrant is required for sparse embeddings"
            raise EmbeddingServiceError(msg)
        if self._sparse is None:
            self._sparse = FastEmbedSparseRuntime(model_name=self._sparse_model_name)

        results = await asyncio.to_thread(self._sparse.embed_documents, list(texts))
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
        """Local embeddings are cost-free."""

        return 0.0

    @property
    def max_tokens_per_request(self) -> int:
        """Expose the FastEmbed max sequence length."""

        if self._dense is None:
            return 512
        return getattr(self._dense, "max_length", 512)
