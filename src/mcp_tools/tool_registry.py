"""Register MCP tool modules with FastMCP."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

from fastmcp import FastMCP

from src.infrastructure.client_manager import ClientManager

from . import tools


logger = logging.getLogger(__name__)

_REGISTRATION_PIPELINE: Final[
    list[tuple[str, Callable[[FastMCP, ClientManager], None]]]
] = [
    ("retrieval", tools.retrieval.register_tools),
    ("documents", tools.documents.register_tools),
    ("embeddings", tools.embeddings.register_tools),
    ("lightweight_scrape", tools.lightweight_scrape.register_tools),
    ("collection_management", tools.collection_management.register_tools),
    ("projects", tools.projects.register_tools),
    ("payload_indexing", tools.payload_indexing.register_tools),
    ("analytics", tools.analytics.register_tools),
    ("cache", tools.cache.register_tools),
    ("content_intelligence", tools.content_intelligence.register_tools),
    ("system_health", tools.system_health.register_tools),
    ("web_search", tools.web_search.register_tools),
    ("cost_estimation", tools.cost_estimation.register_tools),
]


async def register_all_tools(mcp: FastMCP, client_manager: ClientManager) -> None:
    """Register all final tool modules with the FastMCP instance."""
    for name, registrar in _REGISTRATION_PIPELINE:
        logger.debug("Registering tool module '%s'", name)
        registrar(mcp, client_manager)
    logger.info("Registered %d MCP tool modules", len(_REGISTRATION_PIPELINE))
