"""Main MCP Server module - initializes FastMCP and registers all tools."""

import logging

from fastmcp import FastMCP

from .service_manager import UnifiedServiceManager
from .tools import (
    analytics,
    cache,
    collections,
    documents,
    embeddings,
    projects,
    search,
    utils,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ai-docs-vector-db-unified")

# Initialize service manager (shared across all tools)
service_manager = UnifiedServiceManager()

# Register all tools with the MCP server
# This maintains backward compatibility while organizing code better

# Search & Retrieval Tools
search.register_tools(mcp, service_manager)

# Embedding Tools
embeddings.register_tools(mcp, service_manager)

# Document Management Tools
documents.register_tools(mcp, service_manager)

# Project Management Tools
projects.register_tools(mcp, service_manager)

# Collection Management Tools
collections.register_tools(mcp, service_manager)

# Analytics Tools
analytics.register_tools(mcp, service_manager)

# Cache Management Tools
cache.register_tools(mcp, service_manager)

# Utility Tools
utils.register_tools(mcp, service_manager)

logger.info("MCP Server initialized with all tools registered")