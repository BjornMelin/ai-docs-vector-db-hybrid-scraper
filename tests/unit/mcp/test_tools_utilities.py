"""Unit tests for MCP utilities tools module."""

from unittest.mock import MagicMock

import pytest

from src.mcp.tools.utilities import register_tools


class TestUtilitiesTools:
    """Test cases for utilities tools functionality."""

    def test_register_tools_function_exists(self):
        """Test that register_tools function exists and is callable."""
        assert callable(register_tools)

    def test_register_tools_basic_registration(self):
        """Test basic tool registration without errors."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()
        
        # Should not raise any exceptions
        register_tools(mock_mcp, mock_client_manager)
        
        # Verify that the MCP tool decorator was called
        # The exact number depends on how many tools are registered
        assert mock_mcp.tool.call_count > 0

    def test_register_tools_with_none_client_manager(self):
        """Test tool registration handles None client manager."""
        mock_mcp = MagicMock()
        
        # Should not raise exceptions even with None client manager
        register_tools(mock_mcp, None)
        
        assert mock_mcp.tool.call_count > 0

    def test_register_tools_decorates_functions(self):
        """Test that tools are properly decorated."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()
        
        # Capture decorated functions
        decorated_functions = []
        
        def mock_decorator():
            def decorator(func):
                decorated_functions.append(func)
                return func
            return decorator
        
        mock_mcp.tool = mock_decorator
        
        register_tools(mock_mcp, mock_client_manager)
        
        # Should have decorated at least one function
        assert len(decorated_functions) > 0
        
        # All decorated items should be callable
        for func in decorated_functions:
            assert callable(func)

    @pytest.mark.asyncio
    async def test_estimate_costs_tool_basic_functionality(self):
        """Test the estimate_costs tool basic functionality."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()
        
        # Capture the estimate_costs function
        estimate_costs_func = None
        
        def capture_decorator():
            def decorator(func):
                nonlocal estimate_costs_func
                if func.__name__ == "estimate_costs":
                    estimate_costs_func = func
                return func
            return decorator
        
        mock_mcp.tool = capture_decorator
        
        register_tools(mock_mcp, mock_client_manager)
        
        # Test the estimate_costs function if captured
        if estimate_costs_func:
            # Test basic cost estimation
            result = await estimate_costs_func(text_count=100, average_length=1000)
            
            assert isinstance(result, dict)
            assert "text_count" in result
            assert "estimated_tokens" in result
            assert "embedding_cost" in result
            assert result["text_count"] == 100
            assert result["estimated_tokens"] > 0
            assert result["embedding_cost"] >= 0

    @pytest.mark.asyncio
    async def test_estimate_costs_with_storage(self):
        """Test estimate_costs with storage cost calculation."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()
        
        estimate_costs_func = None
        
        def capture_decorator():
            def decorator(func):
                nonlocal estimate_costs_func
                if func.__name__ == "estimate_costs":
                    estimate_costs_func = func
                return func
            return decorator
        
        mock_mcp.tool = capture_decorator
        
        register_tools(mock_mcp, mock_client_manager)
        
        if estimate_costs_func:
            # Test with storage costs included
            result = await estimate_costs_func(
                text_count=100, average_length=1000, include_storage=True
            )
            
            assert "storage_gb" in result
            assert "storage_cost_monthly" in result
            assert "total_cost" in result
            assert result["storage_gb"] >= 0
            assert result["storage_cost_monthly"] >= 0
            assert result["total_cost"] >= result["embedding_cost"]

    @pytest.mark.asyncio
    async def test_estimate_costs_without_storage(self):
        """Test estimate_costs without storage cost calculation."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()
        
        estimate_costs_func = None
        
        def capture_decorator():
            def decorator(func):
                nonlocal estimate_costs_func
                if func.__name__ == "estimate_costs":
                    estimate_costs_func = func
                return func
            return decorator
        
        mock_mcp.tool = capture_decorator
        
        register_tools(mock_mcp, mock_client_manager)
        
        if estimate_costs_func:
            # Test without storage costs
            result = await estimate_costs_func(
                text_count=50, average_length=500, include_storage=False
            )
            
            # Should not include storage-related fields
            assert "storage_gb" not in result
            assert "storage_cost_monthly" not in result
            assert "total_cost" not in result
            
            # Should still include basic fields
            assert "text_count" in result
            assert "estimated_tokens" in result
            assert "embedding_cost" in result

    def test_module_imports(self):
        """Test that the module can be imported without errors."""
        from src.mcp.tools import utilities
        
        assert hasattr(utilities, 'register_tools')
        assert callable(utilities.register_tools)

    def test_logger_configuration(self):
        """Test that the logger is properly configured."""
        from src.mcp.tools.utilities import logger
        
        import logging
        assert isinstance(logger, logging.Logger)
        assert logger.name == "src.mcp.tools.utilities"