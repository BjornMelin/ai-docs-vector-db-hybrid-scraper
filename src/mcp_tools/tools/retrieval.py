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
from pydantic import BaseModel, ConfigDict, Field

from src.config.models import SearchStrategy
from src.contracts.retrieval import SearchRecord
from src.mcp_tools.tools._shared import ensure_vector_service
from src.models.search import SearchRequest as CoreSearchRequest


logger = logging.getLogger(__name__)


class MultiStageSearchPayload(BaseModel):
    """Payload for multi-stage retrieval combining several filter passes."""

    query: str = Field(..., min_length=1, description="User query text.")
    collection: str = Field(
        default="documentation", min_length=1, description="Target collection name."
    )
    stages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered stage definitions with optional filters and limits.",
    )
    limit: int = Field(
        default=10, ge=1, le=1000, description="Maximum number of results to return."
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in results."
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Base filters applied to each stage."
    )

    model_config = ConfigDict(extra="forbid")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.infrastructure.client_manager import ClientManager
else:  # pragma: no cover - runtime alias for tooling
    ClientManager = Any


def _dedupe_by_id(matches: Iterable[SearchRecord]) -> list[SearchRecord]:
    """Return matches deduped by id while keeping highest score."""

    ranked: OrderedDict[str, SearchRecord] = OrderedDict()
    for record in matches:
        previous = ranked.get(record.id)
        if previous is None or record.score > previous.score:
            ranked[record.id] = record
    return list(ranked.values())


def _maybe_strip_metadata(
    records: Iterable[SearchRecord], *, include_metadata: bool
) -> list[SearchRecord]:
    """Return records with optional metadata stripping."""

    if include_metadata:
        return list(records)
    return [
        record.model_copy(update={"metadata": None}, deep=True) for record in records
    ]


def _rrf_rank(matches: list[SearchRecord], *, k: int = 60) -> list[SearchRecord]:
    """Return matches ranked by Reciprocal Rank Fusion."""

    scored: list[tuple[float, SearchRecord]] = []
    for idx, record in enumerate(matches, start=1):
        scored.append((1.0 / (k + idx), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored]


async def _search_matches(
    client_manager: ClientManager,
    request: CoreSearchRequest,
    *,
    ctx: Context | None = None,
) -> list[SearchRecord]:
    """Run a vector search and return the raw matches."""

    service = await ensure_vector_service(client_manager)
    collection_value = request.collection or "documentation"
    if ctx:
        strategy_value = (
            request.search_strategy.value
            if hasattr(request.search_strategy, "value")
            else str(request.search_strategy)
        )
        await ctx.info(
            f"search: strategy={strategy_value} "
            f"collection={collection_value} limit={request.limit}"
        )
    try:
        matches = await service.search_documents(
            collection_value,
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
        request: CoreSearchRequest, ctx: Context
    ) -> list[SearchRecord]:
        """Execute a single-stage search."""

        matches = await _search_matches(client_manager, request, ctx=ctx)
        return _maybe_strip_metadata(
            matches[: request.limit], include_metadata=request.include_metadata
        )

    @mcp.tool()
    async def filtered_search(
        request: CoreSearchRequest, ctx: Context
    ) -> list[SearchRecord]:
        """Execute a search with structured filters."""

        normalized_request = request.model_copy(
            update={"search_strategy": SearchStrategy.HYBRID}
        )
        matches = await _search_matches(client_manager, normalized_request, ctx=ctx)
        return _maybe_strip_metadata(
            matches[: normalized_request.limit],
            include_metadata=normalized_request.include_metadata,
        )

    @mcp.tool()
    async def multi_stage_search(
        payload: MultiStageSearchPayload, ctx: Context
    ) -> list[SearchRecord]:
        """Execute a simplified multi‑stage search."""

        service = await ensure_vector_service(client_manager)
        all_matches: list[SearchRecord] = []

        for stage in payload.stages:
            stage_limit = int(stage.get("limit", payload.limit))
            stage_filters = stage.get("filters") or stage.get("filter")
            try:
                stage_matches = await service.search_documents(
                    payload.collection,
                    payload.query,
                    limit=stage_limit,
                    filters=stage_filters,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                logger.warning("multi_stage stage failed: %s", exc)
                continue
            all_matches.extend(stage_matches)

        deduped = _dedupe_by_id(all_matches)
        await ctx.info(f"multi_stage_search -> {len(deduped)} unique after merge")
        return _maybe_strip_metadata(
            deduped[: payload.limit],
            include_metadata=payload.include_metadata,
        )

    @mcp.tool()
    async def search_with_context(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        query: str,
        collection: str,
        limit: int = 10,
        context_size: int = 3,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchRecord]:
        """Return primary results plus an extended context set."""

        extended_limit = max(limit + max(context_size, 0), limit)
        base_request = CoreSearchRequest.model_validate(
            {
                "query": query,
                "collection": collection,
                "limit": extended_limit,
                "include_metadata": include_metadata,
                "filters": None,
                "search_strategy": SearchStrategy.HYBRID,
            }
        )
        matches = await _search_matches(client_manager, base_request, ctx=ctx)
        return _maybe_strip_metadata(
            matches[:extended_limit], include_metadata=include_metadata
        )

    @mcp.tool()
    async def recommend_similar(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        point_id: str,
        collection: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
        ctx: Context | None = None,
    ) -> list[SearchRecord]:
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
        return _maybe_strip_metadata(
            filtered[:limit], include_metadata=include_metadata
        )

    @mcp.tool()
    async def reranked_search(
        request: CoreSearchRequest, ctx: Context
    ) -> list[SearchRecord]:
        """Search then apply RRF scoring for shallow reranking."""

        service = await ensure_vector_service(client_manager)
        collection_value = request.collection or "documentation"
        baseline = await service.search_documents(
            collection_value,
            request.query,
            limit=max(50, request.limit),
            filters=request.filters,
        )
        ranked = _rrf_rank(baseline)
        return _maybe_strip_metadata(
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
