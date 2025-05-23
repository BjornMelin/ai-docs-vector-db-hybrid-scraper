#!/usr/bin/env python3
"""Integration tests for MCP servers.

Tests actual functionality without complex mocking.
"""

import pytest
from src.enhanced_mcp_server import enhanced_mcp
from src.mcp_server import mcp


class TestMCPServerRegistration:
    """Test that tools are properly registered with the MCP server."""

    def test_basic_server_has_all_tools(self):
        """Test that basic MCP server has all expected tools."""
        expected_tools = {
            "scrape_url",
            "search",
            "list_collections",
            "create_collection",
            "delete_collection",
            "get_collection_info",
            "clear_cache",
        }

        # Get registered tool names
        registered_tools = set()
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            for tool_name in mcp._tool_manager._tools:
                registered_tools.add(tool_name)

        assert expected_tools.issubset(registered_tools), (
            f"Missing tools: {expected_tools - registered_tools}"
        )

    def test_enhanced_server_has_additional_tools(self):
        """Test that enhanced MCP server has all expected tools."""
        expected_tools = {
            # Enhanced tools
            "create_project",
            "list_projects",
            "plan_scraping",
            "execute_scraping_plan",
            "smart_search",
            "index_documentation",
            "update_project",
            "export_project",
        }

        # Get registered tool names
        registered_tools = set()
        if hasattr(enhanced_mcp, "_tool_manager") and hasattr(
            enhanced_mcp._tool_manager, "_tools"
        ):
            for tool_name in enhanced_mcp._tool_manager._tools:
                registered_tools.add(tool_name)

        assert expected_tools.issubset(registered_tools), (
            f"Missing tools: {expected_tools - registered_tools}"
        )


class TestMCPServerResources:
    """Test MCP server resources."""

    def test_basic_server_has_resources(self):
        """Test that basic MCP server has expected resources."""
        expected_resources = {
            "config://environment",
            "stats://database",
        }

        # Get registered resource names
        registered_resources = set()
        if hasattr(mcp, "_resource_manager") and hasattr(
            mcp._resource_manager, "_resources"
        ):
            for resource_name in mcp._resource_manager._resources:
                registered_resources.add(resource_name)

        assert expected_resources.issubset(registered_resources), (
            f"Missing resources: {expected_resources - registered_resources}"
        )

    def test_enhanced_server_has_resources(self):
        """Test that enhanced MCP server has expected resources."""
        expected_resources = {
            "projects://list",
            "stats://system",
        }

        # Get registered resource names
        registered_resources = set()
        if hasattr(enhanced_mcp, "_resource_manager") and hasattr(
            enhanced_mcp._resource_manager, "_resources"
        ):
            for resource_name in enhanced_mcp._resource_manager._resources:
                registered_resources.add(resource_name)

        assert expected_resources.issubset(registered_resources), (
            f"Missing resources: {expected_resources - registered_resources}"
        )


class TestMCPServerConfiguration:
    """Test MCP server configuration."""

    def test_basic_server_name_and_instructions(self):
        """Test basic server metadata."""
        assert mcp.name == "AI Docs Vector DB"
        assert "scraping and vector search capabilities" in mcp.instructions

    def test_enhanced_server_name_and_instructions(self):
        """Test enhanced server metadata."""
        assert enhanced_mcp.name == "Enhanced AI Docs Vector DB"
        assert "Project-based documentation management" in enhanced_mcp.instructions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
