"""Model Context Protocol tools for query processing.

The MCP surface mirrors the simplified `SearchOrchestrator` by exposing a
single, library-first search entry point. The tool keeps the request schema
intentionally small so additional pipeline features can be layered on without
reintroducing the previous federated abstractions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastmcp import Context
from pydantic import BaseModel, Field

from src.contracts.retrieval import SearchRecord
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import SearchResult
from src.services.query_processing import SearchOrchestrator, SearchRequest


logger = logging.getLogger(__name__)


_ORCHESTRATOR: SearchOrchestrator | None = None


class SearchToolRequest(BaseModel):
    """Parameters accepted by the MCP search tool."""

    query: str = Field(..., description="Search query to execute")
    collection: str | None = Field(
        default=None, description="Optional collection override"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    enable_expansion: bool = Field(
        default=False, description="Apply synonym-based query expansion"
    )
    enable_personalization: bool = Field(
        default=False, description="Apply lightweight personalization"
    )
    user_id: str | None = Field(
        default=None, description="User identifier used for personalization"
    )
    enable_rag: bool = Field(
        default=False, description="Generate a lightweight RAG answer"
    )


async def _get_orchestrator(client_manager: ClientManager) -> SearchOrchestrator:
    """Initialise (once) and return the shared orchestrator instance."""

    # noqa: PLW0603 - intentional module-level cache  # pylint: disable=global-statement  # pyright: ignore[reportGlobalUsage]
    global _ORCHESTRATOR

    if _ORCHESTRATOR is not None:
        return _ORCHESTRATOR

    vector_service = await client_manager.get_vector_store_service()
    orchestrator = SearchOrchestrator(vector_store_service=vector_service)
    await orchestrator.initialize()
    _ORCHESTRATOR = orchestrator
    return orchestrator


def _convert_results(records: list[SearchRecord], *, limit: int) -> list[SearchResult]:
    """Convert orchestrator records into MCP response objects."""

    results: list[SearchResult] = []
    for record in records[:limit]:
        metadata: dict[str, Any] = {}
        if isinstance(record.metadata, Mapping):
            metadata.update(record.metadata)
        if record.collection:
            metadata.setdefault("collection", record.collection)
        if record.raw_score is not None:
            metadata.setdefault("raw_score", record.raw_score)
        if record.normalized_score is not None:
            metadata.setdefault("normalized_score", record.normalized_score)
        if record.grouping_applied is not None:
            metadata.setdefault("grouping_applied", record.grouping_applied)

        results.append(
            SearchResult(
                id=record.id,
                content=record.content,
                score=record.score,
                title=record.title,
                url=record.url,
                metadata=metadata or None,
            )
        )

    return results


def register_query_processing_tools(mcp, client_manager: ClientManager) -> None:
    """Register the search tool with the MCP server."""

    @mcp.tool()
    async def search_documents(
        request: SearchToolRequest, ctx: Context
    ) -> list[SearchResult]:
        """Execute a search through the shared orchestrator."""

        orchestrator = await _get_orchestrator(client_manager)
        await ctx.debug(
            "Executing search",
        )
        search_request = SearchRequest(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            enable_expansion=request.enable_expansion,
            enable_personalization=request.enable_personalization,
            user_id=request.user_id,
        )

        try:
            orchestrator_result = await orchestrator.search(search_request)
        except Exception:  # pragma: no cover - surfaced via MCP logs
            await ctx.error("Search execution failed")
            logger.exception("Search execution failed")
            raise

        converted = _convert_results(orchestrator_result.records, limit=request.limit)

        await ctx.info(
            "Search completed",
        )
        return converted

    logger.info("Query processing MCP tools registered")


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Match the public registration surface used by the tool registry."""

    register_query_processing_tools(mcp, client_manager)


__all__ = [
    "SearchToolRequest",
    "register_query_processing_tools",
    "register_tools",
]
