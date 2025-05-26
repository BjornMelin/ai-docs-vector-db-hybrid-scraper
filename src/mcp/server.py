"""Main MCP Server module - initializes FastMCP and registers all tools."""

import logging

from fastmcp import FastMCP

from .service_manager import UnifiedServiceManager
from .tools import collections
from .tools import search

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ai-docs-vector-db-unified")

# Initialize service manager (shared across all tools)
service_manager = UnifiedServiceManager()

# Register all tools with the MCP server
# Only registering existing tools

# Search & Retrieval Tools
search.register_tools(mcp, service_manager)

# Collection Management Tools
collections.register_tools(mcp, service_manager)

logger.info("MCP Server initialized with all tools registered")
