"""LangChain retriever backed by the VectorStoreService adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Self

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.services.vector_db.service import VectorStoreService


class VectorServiceRetriever(BaseRetriever):
    """Expose VectorStoreService searches through the LangChain retriever API."""

    def __init__(
        self,
        vector_service: VectorStoreService,
        collection: str,
        *,
        k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._vector_service = vector_service
        self._collection = collection
        self._k = k
        self._filters: Mapping[str, Any] | None = filters

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
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Synchronously fetch relevant documents."""

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
        """Return documents retrieved for ``query``."""

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

        await run_manager.on_retriever_end(documents)

        return documents
