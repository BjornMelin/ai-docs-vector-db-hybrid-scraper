"""MCP Server Package - Modularized implementation of the unified MCP server."""

from .server import mcp
from .service_manager import UnifiedServiceManager

__all__ = ["UnifiedServiceManager", "mcp"]
