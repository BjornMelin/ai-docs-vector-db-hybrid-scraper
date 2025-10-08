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
from typing import TYPE_CHECKING, Any

from fastmcp import Context

from src.config.models import SearchStrategy
from src.mcp_tools.models.requests import (  # type: ignore[import]
    FilteredSearchRequest,
    MultiStageSearchRequest,
    SearchRequest,
)
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools._shared import ensure_vector_service, match_to_result
from src.services.vector_db.types import VectorMatch


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.infrastructure.client_manager import ClientManager
else:  # pragma: no cover - runtime alias for tooling
    ClientManager = Any


def _dedupe_by_id(matches: Iterable[VectorMatch]) -> list[VectorMatch]:
    """Return matches deduped by id while keeping highest score."""

    ranked: OrderedDict[str, VectorMatch] = OrderedDict()
    for m in matches:
        prev = ranked.get(m.id)
        if prev is None or m.score > prev.score:
            ranked[m.id] = m
    return sorted(ranked.values(), key=lambda match: match.score, reverse=True)


def _to_results(
    matches: Iterable[VectorMatch], *, include_metadata: bool
) -> list[SearchResult]:
    """Convert vector matches to SearchResult objects."""

    return [match_to_result(m, include_metadata=include_metadata) for m in matches]


def _rrf_rank(matches: list[VectorMatch], *, k: int = 60) -> list[VectorMatch]:
    """Return matches ranked by Reciprocal Rank Fusion."""

    scored: list[tuple[float, VectorMatch]] = []
    for idx, m in enumerate(matches, start=1):
        scored.append((1.0 / (k + idx), m))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [m for _, m in scored]


async def _search_matches(
    client_manager: ClientManager,
    request: SearchRequest,
    *,
    ctx: Context | None = None,
) -> list[VectorMatch]:
    """Run a vector search and return the raw matches."""

    service = await ensure_vector_service(client_manager)
    if ctx:
        await ctx.info(
            f"search: strategy={request.strategy.value} "
            f"collection={request.collection} limit={request.limit}"
        )
    try:
        matches = await service.search_documents(
            request.collection,
            request.query,
            limit=request.limit,
            filters=request.filters,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.exception("search_documents failed")
        if ctx:
            await ctx.error(f"search failed: {exc}")
        raise
    if ctx:
        await ctx.info(f"search -> {len(matches)} raw matches")
    return matches


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register unified retrieval tools."""

    @mcp.tool()
    async def search_documents(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Execute a single-stage search."""

        matches = await _search_matches(client_manager, request, ctx=ctx)
        return _to_results(
            matches[: request.limit], include_metadata=request.include_metadata
        )

    @mcp.tool()
    async def filtered_search(
        request: FilteredSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Execute a search with structured filters."""

        base = SearchRequest(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            filters=request.filters,
            include_metadata=request.include_metadata,
            strategy=SearchStrategy.HYBRID,
        )
        return await search_documents(base, ctx)  # noqa: B023

    @mcp.tool()
    async def multi_stage_search(
        request: MultiStageSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Execute a simplified multi‑stage search."""

        service = await ensure_vector_service(client_manager)
        all_matches: list[VectorMatch] = []

        for stage in request.stages:
            stage_limit = int(stage.get("limit", request.limit))
            stage_filters = stage.get("filters") or stage.get("filter")
            try:
                stage_matches = await service.search_documents(
                    request.collection,
                    request.query,
                    limit=stage_limit,
                    filters=stage_filters,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                logger.warning("multi_stage stage failed: %s", exc)
                continue
            all_matches.extend(stage_matches)

        deduped = _dedupe_by_id(all_matches)
        await ctx.info(f"multi_stage_search -> {len(deduped)} unique after merge")
        return _to_results(
            deduped[: request.limit], include_metadata=request.include_metadata
        )

    @mcp.tool()
    async def search_with_context(
        query: str,
        collection: str,
        limit: int = 10,
        context_size: int = 3,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchResult]:
        """Return primary results plus an extended context set."""

        extended_limit = max(limit + max(context_size, 0), limit)
        base_request = SearchRequest(
            query=query,
            collection=collection,
            limit=extended_limit,
            include_metadata=include_metadata,
            filters=None,
            strategy=SearchStrategy.HYBRID,
        )
        matches = await _search_matches(client_manager, base_request, ctx=ctx)
        return _to_results(matches[:extended_limit], include_metadata=include_metadata)

    @mcp.tool()
    async def recommend_similar(  # pylint: disable=too-many-arguments
        point_id: str,
        collection: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchResult]:
        """Recommend documents similar to a given point."""

        service = await ensure_vector_service(client_manager)
        payload = await service.get_document(collection, point_id)
        if payload is None:
            msg = f"document {point_id} not found in {collection}"
            if ctx:
                await ctx.error(msg)
            raise ValueError(msg)
        matches = await service.recommend(
            collection,
            positive_ids=[point_id],
            limit=limit + 1,
            filters=filters,
        )
        filtered = [
            m for m in matches if m.id != point_id and m.score >= score_threshold
        ]
        return _to_results(filtered[:limit], include_metadata=include_metadata)

    @mcp.tool()
    async def reranked_search(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search then apply RRF scoring for shallow reranking."""

        service = await ensure_vector_service(client_manager)
        baseline = await service.search_documents(
            request.collection,
            request.query,
            limit=max(50, request.limit),
            filters=request.filters,
        )
        ranked = _rrf_rank(baseline)
        return _to_results(
            ranked[: request.limit], include_metadata=request.include_metadata
        )

    @mcp.tool()
    async def scroll_collection(
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Scroll a collection with pagination."""

        service = await ensure_vector_service(client_manager)
        docs, next_off = await service.list_documents(
            collection, limit=limit, offset=offset
        )
        if ctx:
            await ctx.info(f"scroll {collection} -> {len(docs)} docs")
        return {"documents": docs, "next_offset": next_off}
