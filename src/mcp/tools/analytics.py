"""Analytics and monitoring tools for MCP server."""

import logging

from ..service_manager import UnifiedServiceManager

logger = logging.getLogger(__name__)


def register_tools(mcp, service_manager: UnifiedServiceManager):
    """Register analytics tools with the MCP server."""
    # TODO: Move analytics tools from unified_mcp_server.py
    pass
