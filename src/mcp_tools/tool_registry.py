"""Register MCP tool modules with FastMCP."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from src.services.cache.manager import CacheManager
from src.services.core.project_storage import ProjectStorage
from src.services.embeddings.manager import EmbeddingManager
from src.services.observability.health_manager import HealthCheckManager
from src.services.vector_db.service import VectorStoreService

from . import tools


logger = logging.getLogger(__name__)


async def register_all_tools(
    mcp: FastMCP,
    *,
    vector_service: VectorStoreService | None = None,
    cache_manager: CacheManager | None = None,
    crawl_manager: Any | None = None,
    content_intelligence_service: Any | None = None,
    project_storage: ProjectStorage | None = None,
    embedding_manager: EmbeddingManager | None = None,
    health_manager: HealthCheckManager | None = None,
) -> None:
    """Register the full MCP tool suite with dependency overrides."""

    logger.debug("Registering MCP tool modules")
    tools.retrieval.register_tools(mcp, vector_service=vector_service)
    tools.documents.register_tools(
        mcp,
        vector_service=vector_service,
        cache_manager=cache_manager,
        crawl_manager=crawl_manager,
        content_intelligence_service=content_intelligence_service,
    )
    tools.embeddings.register_tools(mcp, embedding_manager=embedding_manager)
    tools.lightweight_scrape.register_tools(mcp, crawl_manager=crawl_manager)
    tools.collection_management.register_tools(
        mcp,
        vector_service=vector_service,
        cache_manager=cache_manager,
    )
    tools.projects.register_tools(
        mcp,
        vector_service=vector_service,
        project_storage=project_storage,
    )
    tools.payload_indexing.register_tools(mcp, vector_service=vector_service)
    tools.analytics.register_tools(mcp, vector_service=vector_service)
    tools.cache.register_tools(mcp, cache_manager=cache_manager)
    tools.content_intelligence.register_tools(
        mcp,
        content_service=content_intelligence_service,
    )
    tools.system_health.register_tools(mcp, health_manager=health_manager)
    tools.web_search.register_tools(mcp)
    tools.cost_estimation.register_tools(mcp)
    logger.info("Registered MCP tools")
