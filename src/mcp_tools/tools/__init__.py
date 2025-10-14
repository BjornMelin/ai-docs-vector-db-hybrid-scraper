"""MCP Tools package with explicit, final exports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.mcp_tools.tools import (
        analytics,
        cache,
        collection_management,
        configuration,
        content_intelligence,
        cost_estimation,
        crawling,
        documents,
        embeddings,
        lightweight_scrape,
        payload_indexing,
        projects,
        rag,
        retrieval,
        system_health,
        web_search,
    )


__all__ = [
    "analytics",
    "cache",
    "collection_management",
    "configuration",
    "content_intelligence",
    "cost_estimation",
    "crawling",
    "documents",
    "embeddings",
    "lightweight_scrape",
    "payload_indexing",
    "projects",
    "rag",
    # final surfaces
    "retrieval",
    "system_health",
    "web_search",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name not in __all__:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
