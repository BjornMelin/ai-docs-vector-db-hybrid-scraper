"""Utilities for registering all MCP tool modules with FastMCP."""

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
    ("search", tools.search.register_tools),
    ("documents", tools.documents.register_tools),
    ("embeddings", tools.embeddings.register_tools),
    ("lightweight_scrape", tools.lightweight_scrape.register_tools),
    ("collection_management", tools.collection_management.register_tools),
    ("projects", tools.projects.register_tools),
    ("search_tools", tools.search_tools.register_tools),
    ("query_processing_tools", tools.query_processing_tools.register_tools),
    ("payload_indexing", tools.payload_indexing.register_tools),
    ("analytics", tools.analytics.register_tools),
    ("cache", tools.cache.register_tools),
    ("utilities", tools.utilities.register_tools),
    ("content_intelligence", tools.content_intelligence.register_tools),
]


async def register_all_tools(mcp: FastMCP, client_manager: ClientManager) -> None:
    """Register all tool modules with the supplied FastMCP instance.

    Args:
        mcp: FastMCP server that should receive tool registrations.
        client_manager: Client manager used by individual tool modules.
    """

    registered: list[str] = []
    for tool_name, registrar in _REGISTRATION_PIPELINE:
        logger.debug("Registering tool module '%s'", tool_name)
        registrar(mcp, client_manager)
        registered.append(tool_name)

    try:
        tools.agentic_rag.register_tools(mcp, client_manager)
        registered.append("agentic_rag")
    except ImportError as exc:
        logger.info("Skipping agentic_rag tool module: %s", exc)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to register agentic_rag tool module")

    logger.info("Registered %d MCP tool modules", len(registered))
