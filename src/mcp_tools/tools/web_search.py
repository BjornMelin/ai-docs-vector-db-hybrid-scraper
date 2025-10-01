"""Web search tools leveraging Tavily API for production-grade results."""

import logging
import os
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.security import MLSecurityValidator


# Lazy import to avoid hard dependency
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register Tavily-powered web search tools."""

    @mcp.tool()
    async def web_search(
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_images: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Perform web search using Tavily AI.

        Args:
            query: Search query
            max_results: Maximum results (1-20, default 5)
            search_depth: 'basic' or 'advanced'
            include_answer: Include AI-generated answer
            include_images: Include relevant images
            include_domains: Whitelist specific domains
            exclude_domains: Blacklist specific domains
            ctx: MCP context

        Returns:
            Search results with URLs, content, scores
        """
        try:
            if ctx:
                await ctx.info(f"Web search: '{query}' (depth={search_depth})")

            # Validate query
            security_validator = MLSecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Check for Tavily
            if not TavilyClient:
                msg = "tavily-python not installed. Run: pip install tavily-python"
                if ctx:
                    await ctx.error(msg)
                raise ImportError(msg)

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                msg = "TAVILY_API_KEY environment variable not set"
                if ctx:
                    await ctx.error(msg)
                raise ValueError(msg)

            # Execute Tavily search
            tavily = TavilyClient(api_key=api_key)
            response = tavily.search(
                query=validated_query,
                max_results=min(max_results, 20),
                search_depth=search_depth,
                include_answer=include_answer,
                include_images=include_images,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            # Format response
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

        except Exception as e:
            logger.exception("Web search failed")
            if ctx:
                await ctx.error(f"Search error: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e),
            }

    @mcp.tool()
    async def advanced_web_search(
        query: str,
        max_results: int = 10,
        include_raw_content: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Perform advanced web search with full content extraction.

        Args:
            query: Search query
            max_results: Maximum results (1-20)
            include_raw_content: Include full page HTML
            ctx: MCP context

        Returns:
            Detailed search results with full content
        """
        try:
            if ctx:
                await ctx.info(f"Advanced search: '{query}'")

            security_validator = MLSecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            if not TavilyClient:
                raise ImportError("tavily-python not installed")

            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY not set")

            tavily = TavilyClient(api_key=api_key)
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

        except Exception as e:
            logger.exception("Advanced search failed")
            if ctx:
                await ctx.error(f"Advanced search error: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e),
            }
