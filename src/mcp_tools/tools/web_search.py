"""Web search MCP tools powered by Tavily."""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from fastmcp import Context

from src.security.ml_security import MLSecurityValidator


if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from tavily import TavilyClient as _TavilyClient  # type: ignore
else:  # pragma: no cover - runtime uses dynamic import instead
    _TavilyClient = Any

logger = logging.getLogger(__name__)


def _resolve_tavily_client() -> type[_TavilyClient]:
    """Return the Tavily client class, raising when the integration is missing."""
    try:
        module = importlib.import_module("tavily")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        msg = "tavily-python not installed. Run: pip install tavily-python"
        raise ImportError(msg) from exc

    client = getattr(module, "TavilyClient", None)
    if client is None or not isinstance(client, type):
        msg = "tavily-python missing TavilyClient export"
        raise ImportError(msg)
    return client


def _build_error_response(query: str, error: Exception) -> dict[str, Any]:
    """Format a uniform error payload for MCP clients."""
    return {
        "query": query,
        "results": [],
        "error": str(error),
    }


def register_tools(mcp) -> None:
    """Register Tavily-powered web search tools."""

    @mcp.tool()
    async def web_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_answer: bool = False,
        include_images: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Perform web search using Tavily AI.

        Args:
            query: Search query.
            max_results: Maximum results (1-20, default 5).
            search_depth: Either "basic" or "advanced" search depth.
            include_answer: Include AI-generated answer content when True.
            include_images: Include relevant images in the response when True.
            include_domains: Optional whitelist of allowed domains.
            exclude_domains: Optional blacklist of domains to filter out.
            ctx: MCP context for structured logging and messaging.

        Returns:
            Search results with URLs, extracted content, and metadata.
        """
        try:
            if ctx:
                await ctx.info(f"Web search: '{query}' (depth={search_depth})")

            security_validator = MLSecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            tavily_client_cls = _resolve_tavily_client()

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                msg = "TAVILY_API_KEY environment variable not set"
                if ctx:
                    await ctx.error(msg)
                raise ValueError(msg)

            tavily = tavily_client_cls(api_key=api_key)
            response = tavily.search(
                query=validated_query,
                max_results=min(max_results, 20),
                search_depth=search_depth,
                include_answer=include_answer,
                include_images=include_images,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            results = {
                "query": validated_query,
                "results": response.get("results", []),
                "answer": response.get("answer") if include_answer else None,
                "images": response.get("images", []) if include_images else [],
                "response_time": response.get("response_time", 0),
            }

            if ctx:
                await ctx.info(f"Found {len(results['results'])} results")

            return results

        except Exception as exc:
            logger.exception("Web search failed")
            if ctx:
                await ctx.error(f"Search error: {exc}")
            return _build_error_response(query, exc)

    @mcp.tool()
    async def advanced_web_search(
        query: str,
        max_results: int = 10,
        include_raw_content: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Perform advanced web search with full content extraction.

        Args:
            query: Search query string.
            max_results: Maximum results (1-20).
            include_raw_content: Include full page HTML when True.
            ctx: MCP context for structured logging and messaging.

        Returns:
            Detailed search results with expanded metadata.
        """
        try:
            if ctx:
                await ctx.info(f"Advanced search: '{query}'")

            security_validator = MLSecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            tavily_client_cls = _resolve_tavily_client()

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                msg = "TAVILY_API_KEY not set"
                raise ValueError(msg)

            tavily = tavily_client_cls(api_key=api_key)
            response = tavily.search(
                query=validated_query,
                max_results=min(max_results, 20),
                search_depth="advanced",
                include_answer=True,
                include_raw_content=include_raw_content,
            )

            results = {
                "query": validated_query,
                "results": response.get("results", []),
                "answer": response.get("answer"),
                "response_time": response.get("response_time", 0),
            }

            if ctx:
                await ctx.info(f"Found {len(results['results'])} advanced results")

            return results

        except Exception as exc:
            logger.exception("Advanced search failed")
            if ctx:
                await ctx.error(f"Advanced search error: {exc}")
            return _build_error_response(query, exc)
