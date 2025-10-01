"""MCP Tools Package.

This package contains all modular tool implementations for the MCP server.
Each module exports a register_tools function that registers its tools with
the FastMCP instance.
"""

from . import (
    agentic_rag,
    analytics,
    cache,
    collection_management,  # renamed from collections to avoid stdlib conflict
    configuration,
    content_intelligence,
    cost_estimation,
    crawling,
    document_management,
    documents,
    embeddings,
    filtering_tools,
    lightweight_scrape,
    payload_indexing,
    projects,
    query_processing,
    query_processing_tools,
    rag,
    search,
    search_tools,
    system_health,
    utilities,
    web_search,
)


__all__ = [
    "agentic_rag",
    "analytics",
    "cache",
    "collection_management",
    "configuration",
    "content_intelligence",
    "cost_estimation",
    "crawling",
    "document_management",
    "documents",
    "embeddings",
    "filtering_tools",
    "lightweight_scrape",
    "payload_indexing",
    "projects",
    "query_processing",
    "query_processing_tools",
    "rag",
    "search",
    "search_tools",
    "system_health",
    "utilities",
    "web_search",
]
