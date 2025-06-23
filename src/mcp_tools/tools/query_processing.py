import typing
"""Advanced Query Processing tools for MCP server."""

import logging

from ...infrastructure.client_manager import ClientManager
from .helpers import QueryProcessingPipelineFactory
from .helpers import QueryValidationHelper
from .helpers import ResponseConverter
from .helpers.tool_registrars import register_advanced_query_processing_tool
from .helpers.tool_registrars import register_query_analysis_tool
from .helpers.tool_registrars_additional import register_pipeline_health_tool
from .helpers.tool_registrars_additional import register_pipeline_metrics_tool
from .helpers.tool_registrars_additional import register_pipeline_warmup_tool

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register advanced query processing tools with the MCP server."""
    # Create helper instances
    factory = QueryProcessingPipelineFactory(client_manager)
    converter = ResponseConverter()
    validator = QueryValidationHelper()

    # Register all tools using focused helper functions
    register_advanced_query_processing_tool(mcp, factory, converter, validator)
    register_query_analysis_tool(mcp, factory, converter, validator)
    register_pipeline_health_tool(mcp, factory)
    register_pipeline_metrics_tool(mcp, factory)
    register_pipeline_warmup_tool(mcp, factory)
