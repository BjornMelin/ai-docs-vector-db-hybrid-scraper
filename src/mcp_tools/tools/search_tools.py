"""MCP search tools built on top of VectorStoreService.

This module exposes thin wrappers around the vector search service.
Each tool delegates directly to :class:`VectorStoreService`, ensuring the MCP
surface stays aligned with the unified adapter and avoids bespoke caching or
embedding logic.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable

from fastmcp import Context

from src.config import SearchStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import (
    FilteredSearchRequest,
    HyDESearchRequest,
    MultiStageSearchRequest,
    SearchRequest,
)
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools._shared import ensure_vector_service, match_to_result


logger = logging.getLogger(__name__)


async def _execute_search(
    client_manager: ClientManager,
    request: SearchRequest,
    *,
    ctx: Context | None = None,
) -> list[SearchResult]:
    """Execute a single-stage search using the configured strategy."""

    service = await ensure_vector_service(client_manager)

    try:
        if ctx:
            await ctx.info(
                f"Running {request.strategy.value} search in "
                f"collection '{request.collection}'"
            )
    except Exception:  # pragma: no cover - defensive logging guard
        logger.debug("Failed to emit MCP info event for search request")

    try:
        matches = await service.search_documents(
            request.collection,
            request.query,
            limit=request.limit,
            filters=request.filters,
        )

        return [
            match_to_result(match, include_metadata=request.include_metadata)
            for match in matches[: request.limit]
        ]
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Vector search failed")
        if ctx:
            await ctx.error(f"Search failed: {exc}")
        raise


async def _run_stage_search(
    client_manager: ClientManager,
    base_request: MultiStageSearchRequest,
    stage_definition: dict,
    *,
    ctx: Context | None = None,
) -> list[SearchResult]:
    """Execute a single stage within a multi-stage request."""

    stage_limit = int(stage_definition.get("limit", base_request.limit))
    filters = stage_definition.get("filters") or stage_definition.get("filter")

    stage_request = SearchRequest(
        query=base_request.query,
        collection=base_request.collection,
        limit=stage_limit,
        filters=filters,
        include_metadata=base_request.include_metadata,
        strategy=SearchStrategy.HYBRID,
    )
    return await _execute_search(client_manager, stage_request, ctx=ctx)


def _deduplicate_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Deduplicate search results by identifier while keeping best scores."""

    ranked: OrderedDict[str, SearchResult] = OrderedDict()
    for result in results:
        existing = ranked.get(result.id)
        if existing is None or result.score > existing.score:
            ranked[result.id] = result
    # Order by score descending for determinism after dedupe
    return sorted(ranked.values(), key=lambda item: item.score, reverse=True)


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register consolidated search tools with the MCP server."""

    async def _search_documents_direct(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        return await _execute_search(client_manager, request, ctx=ctx)

    @mcp.tool()
    async def search_documents(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search documents using the configured vector strategy."""

        return await _search_documents_direct(request, ctx)

    @mcp.tool()
    async def multi_stage_search(
        request: MultiStageSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Execute a simplified multi-stage search pipeline.

        Each stage reuses the primary query with optional per-stage filters. Results
        from all stages are merged, deduplicated, and ranked by score before the top
        ``request.limit`` items are returned.
        """

        try:
            if ctx:
                await ctx.info(
                    f"Running multi-stage search with {len(request.stages)} stages"
                )
        except Exception:  # pragma: no cover - defensive logging guard
            logger.debug("Failed to emit MCP info event for multi-stage search")

        stage_results: list[SearchResult] = []
        for stage in request.stages:
            try:
                stage_results.extend(
                    await _run_stage_search(client_manager, request, stage, ctx=ctx)
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Multi-stage search stage failed")
                if ctx:
                    await ctx.warning(f"Stage execution failed: {exc}")

        deduped = _deduplicate_results(stage_results)
        return deduped[: request.limit]

    @mcp.tool()
    async def hyde_search(
        request: HyDESearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Compatibility wrapper: performs a hybrid vector search.

        HyDE-specific generation has been retired; this tool now delegates to the
        unified vector search pipeline while preserving the existing MCP contract.
        """

        base_request = SearchRequest(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            filters=request.filters,
            include_metadata=request.include_metadata,
            strategy=SearchStrategy.HYBRID,
        )
        return await _execute_search(client_manager, base_request, ctx=ctx)

    @mcp.tool()
    async def filtered_search(
        request: FilteredSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Execute a search constrained by structured filters."""

        base_request = SearchRequest(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            filters=request.filters,
            include_metadata=request.include_metadata,
            strategy=SearchStrategy.HYBRID,
        )
        return await _execute_search(client_manager, base_request, ctx=ctx)
