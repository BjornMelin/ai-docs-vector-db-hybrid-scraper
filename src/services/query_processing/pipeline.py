"""Final query processing pipeline wrapper around the search orchestrator.

This module provides a lightweight pipeline wrapper that delegates to the
SearchOrchestrator, offering an interface for query processing
operations with flexible request handling.
"""

from __future__ import annotations

from typing import Any

from src.services.base import BaseService

from .models import SearchRequest, SearchResponse
from .orchestrator import SearchOrchestrator


class QueryProcessingPipeline(BaseService):
    """Lightweight pipeline that delegates to :class:`SearchOrchestrator`.

    This pipeline provides an interface for search operations,
    handling request coercion and delegation to the underlying orchestrator.
    """

    def __init__(
        self, orchestrator: SearchOrchestrator, config: Any | None = None
    ) -> None:
        """Initialize the query processing pipeline.

        Args:
            orchestrator: The search orchestrator to delegate operations to.
            config: Optional configuration for the pipeline.

        Raises:
            ValueError: If orchestrator is None.
        """
        if orchestrator is None:
            raise ValueError("Orchestrator cannot be None")
        super().__init__(config)
        self.orchestrator = orchestrator

    async def initialize(self) -> None:
        """Initialize the pipeline and its dependencies.

        This method initializes the underlying orchestrator if not already
        initialized and marks the pipeline as ready for processing.
        """
        if self._initialized:
            return
        await self.orchestrator.initialize()
        self._initialized = True

    async def cleanup(self) -> None:  # pragma: no cover - symmetrical teardown
        """Clean up pipeline resources.

        This method cleans up the underlying orchestrator and marks the
        pipeline as uninitialized.
        """
        if not self._initialized:
            return
        await self.orchestrator.cleanup()
        self._initialized = False

    async def process(
        self,
        request: SearchRequest | str | dict[str, Any],
        *,
        collection: str | None = None,
        limit: int | None = None,
        **override_kwargs: Any,
    ) -> SearchResponse:
        """Process a search request through the pipeline.

        Args:
            request: Search request as SearchRequest, string query, or dict.
            collection: Optional collection override.
            limit: Optional result limit override.
            **override_kwargs: Additional keyword arguments to merge.

        Returns:
            Search response containing results and metadata.

        Raises:
            RuntimeError: If pipeline is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("QueryProcessingPipeline not initialized")

        search_request = self._coerce_request(
            request,
            collection_override=collection,
            limit_override=limit,
            extra_kwargs=override_kwargs,
        )
        return await self.orchestrator.search(search_request)

    @staticmethod
    def _coerce_request(
        request: SearchRequest | str | dict[str, Any],
        *,
        collection_override: str | None,
        limit_override: int | None,
        extra_kwargs: dict[str, Any],
    ) -> SearchRequest:
        """Coerce various request formats into a SearchRequest.

        Args:
            request: Request in various formats (SearchRequest, str, or dict).
            collection_override: Optional collection name to override.
            limit_override: Optional result limit to override.
            extra_kwargs: Additional keyword arguments to merge.

        Returns:
            A standardized SearchRequest object.

        Raises:
            TypeError: If request type is not supported.
        """
        if isinstance(request, SearchRequest):
            return request
        if isinstance(request, str):
            payload: dict[str, Any] = {
                "query": request,
                "collection": collection_override,
                "limit": limit_override or 10,
            }
            payload.update(extra_kwargs)
            return SearchRequest(**payload)
        if isinstance(request, dict):
            payload = {**request}
            if collection_override is not None:
                payload.setdefault("collection", collection_override)
            if limit_override is not None:
                payload.setdefault("limit", limit_override)
            payload.update(extra_kwargs)
            return SearchRequest(**payload)
        msg = f"Unsupported request type: {type(request)!r}"
        raise TypeError(msg)


__all__ = ["QueryProcessingPipeline"]
