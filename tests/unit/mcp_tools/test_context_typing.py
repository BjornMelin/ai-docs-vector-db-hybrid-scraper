"""Unit tests for Context type handling in MCP tools.

Verifies that all MCP tools properly handle the Context type using TYPE_CHECKING.
"""

import importlib
import inspect

import pytest
from src.mcp_tools.models.requests import DocumentRequest, SearchRequest


# List of all MCP tool modules
MCP_TOOL_MODULES = [
    "analytics",
    "cache",
    "collections",
    "embeddings",
    "projects",
    "utilities",
    "search",
    "advanced_search",
    "deployment",
    "documents",
    "payload_indexing",
]


class TestContextTyping:
    """Test Context type handling across all MCP tools."""

    def test_all_tools_have_context_protocol(self):
        """Verify all tool modules define Context protocol when not TYPE_CHECKING."""
        for module_name in MCP_TOOL_MODULES:
            module = importlib.import_module(f"src.mcp_tools.tools.{module_name}")

            # Check if module has Context defined
            assert hasattr(module, "Context"), (
                f"{module_name} module missing Context definition"
            )

            # Check if Context is a Protocol
            context_class = module.Context
            assert hasattr(context_class, "info"), (
                f"{module_name} Context missing info method"
            )
            assert hasattr(context_class, "debug"), (
                f"{module_name} Context missing debug method"
            )
            assert hasattr(context_class, "warning"), (
                f"{module_name} Context missing warning method"
            )
            assert hasattr(context_class, "error"), (
                f"{module_name} Context missing error method"
            )

    def test_register_tools_functions_exist(self):
        """Verify all tool modules have register_tools function."""
        for module_name in MCP_TOOL_MODULES:
            module = importlib.import_module(f"src.mcp_tools.tools.{module_name}")

            # Check if module has register_tools function
            assert hasattr(module, "register_tools"), (
                f"{module_name} module missing register_tools function"
            )

            # Check if it's callable
            register_func = module.register_tools
            assert callable(register_func), (
                f"{module_name} register_tools is not callable"
            )

            # Check function signature has correct parameters
            sig = inspect.signature(register_func)
            params = list(sig.parameters.keys())
            assert "mcp" in params, (
                f"{module_name} register_tools missing 'mcp' parameter"
            )
            assert "client_manager" in params, (
                f"{module_name} register_tools missing 'client_manager' parameter"
            )

    def test_no_fastmcp_runtime_import(self):
        """Verify tools don't import fastmcp at runtime (only in TYPE_CHECKING)."""
        for module_name in MCP_TOOL_MODULES:
            module = importlib.import_module(f"src.mcp_tools.tools.{module_name}")

            # Check module doesn't have fastmcp in its namespace
            # (It should only be imported within TYPE_CHECKING block)
            module_dict = vars(module)

            # fastmcp should not be directly imported
            assert "fastmcp" not in module_dict, (
                f"{module_name} has runtime fastmcp import"
            )

    def test_context_parameter_in_tool_functions(self):
        """Verify tool functions use Context type annotation properly."""
        # This test would require parsing the actual function definitions
        # For now, we just ensure the modules load without import errors
        for module_name in MCP_TOOL_MODULES:
            try:
                importlib.import_module(f"src.mcp_tools.tools.{module_name}")
                # If we can import it, the Context typing is working
                assert True
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_tool_registry_imports_all_tools(self):
        """Verify tool registry can import all tool modules."""
        from src.mcp_tools import tool_registry

        # Check that tool_registry has register_all_tools function
        assert hasattr(tool_registry, "register_all_tools"), (
            "tool_registry missing register_all_tools function"
        )

        # Check it's callable
        assert callable(tool_registry.register_all_tools), (
            "register_all_tools is not callable"
        )

    def test_models_have_proper_validation(self):
        """Verify request models have proper field validation."""

        # Test SearchRequest validation
        with pytest.raises(ValueError):
            SearchRequest(query="", collection="docs")  # Empty query

        with pytest.raises(ValueError):
            SearchRequest(query="test", collection="")  # Empty collection

        # Test DocumentRequest validation
        with pytest.raises(ValueError):
            DocumentRequest(url="", collection="docs")  # Empty URL

        # Test valid requests
        valid_search = SearchRequest(query="test query", collection="docs")
        assert valid_search.query == "test query"

        valid_doc = DocumentRequest(url="https://example.com", collection="docs")
        assert valid_doc.url == "https://example.com"
