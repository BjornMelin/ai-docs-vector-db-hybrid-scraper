"""
Unified retrieval tools for the MCP server.

Exposes a single, typed surface for all search operations:
- Dense/hybrid search
- Boolean-filtered search
- Multi-stage search
- Context-extended search
- Recommendations (similar items)
- RRF reranking
- Collection scrolling

Relies on the project's VectorStoreService and existing Pydantic schemas.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from fastmcp import Context

from src.config import SearchStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import (  # type: ignore[import]
    FilteredSearchRequest,
    MultiStageSearchRequest,
    SearchRequest,
)
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools._shared import ensure_vector_service, match_to_result
from src.services.vector_db.types import VectorMatch


logger = logging.getLogger(__name__)


def _dedupe_best(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Deduplicate by id keeping the best score."""

    ranked: OrderedDict[str, SearchResult] = OrderedDict()
    for res in results:
        existing = ranked.get(res.id)
        if existing is None or res.score > existing.score:
            ranked[res.id] = res
    return sorted(ranked.values(), key=lambda r: r.score, reverse=True)


async def _run_search(
    client_manager: ClientManager,
    request: SearchRequest,
    *,
    ctx: Context | None = None,
) -> list[SearchResult]:
    """Execute a single search pass via VectorStoreService."""

    service = await ensure_vector_service(client_manager)
    if ctx:
        await ctx.info(
            f"search: strategy={request.strategy.value} "
            f"collection={request.collection} limit={request.limit}"
        )

    matches = await service.search_documents(
        request.collection,
        request.query,
        limit=request.limit,
        filters=request.filters,
    )
    return [
        match_to_result(m, include_metadata=request.include_metadata)
        for m in matches[: request.limit]
    ]


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register the retrieval tools with FastMCP.

    All tool names are unique across the server, per MCP requirements.
    """

    @mcp.tool()
    async def search_documents(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search documents in a collection using the configured strategy.

        Args:
            request: SearchRequest containing query, collection, limit, filters, etc.
            ctx: MCP context for logging.

        Returns:
            A ranked list of SearchResult items.
        """
        return await _run_search(client_manager, request, ctx=ctx)

    @mcp.tool()
    async def filtered_search(
        request: FilteredSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search constrained by boolean filters.

        Args:
            request: FilteredSearchRequest with must/should/must_not clauses.
            ctx: MCP context.

        Returns:
            Ranked SearchResult list after applying filters.
        """
        base = SearchRequest(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            filters=request.filters,
            include_metadata=request.include_metadata,
            strategy=SearchStrategy.HYBRID,
        )
        return await _run_search(client_manager, base, ctx=ctx)

    @mcp.tool()
    async def multi_stage_search(
        request: MultiStageSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Run a simplified multi-stage pipeline with per-stage filters.

        Merges and deduplicates across stages, then returns the top-N results.
        """
        if ctx:
            await ctx.info(f"multi_stage: stages={len(request.stages)}")

        collected: list[SearchResult] = []
        for stage in request.stages:
            stage_limit = int(stage.get("limit", request.limit))
            filters = stage.get("filters") or stage.get("filter")
            stage_req = SearchRequest(
                query=request.query,
                collection=request.collection,
                limit=stage_limit,
                filters=filters,
                include_metadata=request.include_metadata,
                strategy=SearchStrategy.HYBRID,
            )
            collected.extend(await _run_search(client_manager, stage_req, ctx=ctx))

        return _dedupe_best(collected)[: request.limit]

    @mcp.tool()
    async def search_with_context(  # simple "wider net" retrieval
        query: str,
        collection: str,
        limit: int = 10,
        context_size: int = 3,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchResult]:
        """Return primary results plus extra context hits."""
        base = SearchRequest(
            query=query,
            collection=collection,
            limit=max(limit + max(context_size, 0), limit),
            include_metadata=include_metadata,
            strategy=SearchStrategy.HYBRID,
        )
        results = await _run_search(client_manager, base, ctx=ctx)
        return results[: base.limit]

    @mcp.tool()
    async def recommend_similar(
        collection: str,
        point_id: str,
        limit: int = 10,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchResult]:
        """Recommend items similar to a given point using Qdrant's API."""
        service = await ensure_vector_service(client_manager)
        if ctx:
            await ctx.info(f"recommend: collection={collection} id={point_id}")

        # Validate existence, then recommend
        doc = await service.get_document(collection, point_id)
        if doc is None:
            raise ValueError(f"document '{point_id}' not found in '{collection}'")

        matches: list[VectorMatch] = await service.recommend(
            collection, positive_ids=[point_id], limit=limit + 1
        )
        filtered = [m for m in matches if m.id != point_id][:limit]
        return [match_to_result(m, include_metadata=include_metadata) for m in filtered]

    @mcp.tool()
    async def reranked_search(
        query: str,
        collection: str,
        limit: int = 10,
        rerank_limit: int = 50,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchResult]:
        """Apply Reciprocal Rank Fusion (RRF) over a larger candidate set."""
        base = SearchRequest(
            query=query,
            collection=collection,
            limit=max(rerank_limit, limit),
            include_metadata=include_metadata,
            strategy=SearchStrategy.HYBRID,
        )
        candidates = await _run_search(client_manager, base, ctx=ctx)

        scored = [(1.0 / (60 + i), r) for i, r in enumerate(candidates, start=1)]
        scored.sort(key=lambda t: t[0], reverse=True)
        reranked = [r for _, r in scored[:limit]]
        if ctx:
            await ctx.info(f"rrf: candidates={len(candidates)} -> top={len(reranked)}")
        return reranked

    @mcp.tool()
    async def scroll_collection(
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Paginate through all documents in a collection."""
        service = await ensure_vector_service(client_manager)
        docs, next_offset = await service.list_documents(
            collection, limit=limit, offset=offset
        )
        if ctx:
            await ctx.info(
                f"scroll: collection={collection} fetched={len(docs)} "
                f"next={next_offset}"
            )
        return {"documents": docs, "next_offset": next_offset}
