"""Comprehensive tests for utilities tools module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from fastmcp import Context
from src.infrastructure.client_manager import ClientManager
from src.mcp.tools import utilities


class TestUtilitiesTools:
    """Test utilities tool functions."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager."""
        cm = Mock(spec=ClientManager)

        # Mock config with all required attributes
        config = Mock()
        config.qdrant_url = "http://localhost:6333"
        config.openai_api_key = "test-key"
        config.firecrawl_api_key = None
        config.redis_url = None
        config.cache_config = Mock(max_items=1000)
        cm.config = config

        return cm

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        ctx = Mock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance that captures registered tools."""
        mcp = Mock()
        mcp._tools = {}

        def tool_decorator(func=None, **kwargs):
            def wrapper(f):
                mcp._tools[f.__name__] = f
                return f

            return wrapper if func is None else wrapper(func)

        mcp.tool = tool_decorator
        return mcp

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that utilities tools are registered correctly."""
        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Check that tools were registered
        assert "estimate_costs" in mock_mcp._tools
        assert "validate_configuration" in mock_mcp._tools

    @pytest.mark.asyncio
    async def test_estimate_costs_basic(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test estimate_costs functionality."""
        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        estimate_func = mock_mcp._tools["estimate_costs"]

        # Call the function
        result = await estimate_func(text_count=100, average_length=1000)

        # Verify results
        assert result["text_count"] == 100
        assert result["estimated_tokens"] == 25000  # 100 * 1000 / 4
        assert result["embedding_cost"] == 0.0005  # 25000 * 0.00002 / 1000
        assert result["provider"] == "openai/text-embedding-3-small"
        assert "storage_gb" in result
        assert "storage_cost_monthly" in result
        assert "total_cost" in result

    @pytest.mark.asyncio
    async def test_estimate_costs_without_storage(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test estimate_costs without storage costs."""
        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        estimate_func = mock_mcp._tools["estimate_costs"]

        # Call the function without storage
        result = await estimate_func(
            text_count=50, average_length=500, include_storage=False
        )

        # Verify results
        assert result["text_count"] == 50
        assert result["estimated_tokens"] == 6250  # 50 * 500 / 4
        assert result["embedding_cost"] == 0.0001  # 6250 * 0.00002 / 1000
        assert "storage_gb" not in result
        assert "storage_cost_monthly" not in result
        assert "total_cost" not in result

    @pytest.mark.asyncio
    async def test_validate_configuration(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test validate_configuration functionality."""
        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        validate_func = mock_mcp._tools["validate_configuration"]

        # Call the function
        result = await validate_func()

        # Verify results
        assert result["valid"] is True
        assert result["config"]["qdrant_url"] == "http://localhost:6333"
        assert result["config"]["openai"] == "configured"
        assert len(result["warnings"]) == 1
        assert "Firecrawl API key not configured" in result["warnings"]
        assert result["config"]["cache"]["l1_enabled"] is True
        assert result["config"]["cache"]["l1_max_items"] == 1000
        assert result["config"]["cache"]["l2_enabled"] is False

    @pytest.mark.asyncio
    async def test_validate_configuration_with_firecrawl(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test validate_configuration with Firecrawl configured."""
        # Configure Firecrawl API key
        mock_client_manager.config.firecrawl_api_key = "fc-test-key"

        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        validate_func = mock_mcp._tools["validate_configuration"]

        # Call the function
        result = await validate_func()

        # Verify results
        assert result["valid"] is True
        assert result["config"]["firecrawl"] == "configured"
        assert len(result["warnings"]) == 0

    @pytest.mark.asyncio
    async def test_validate_configuration_missing_keys(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test validate_configuration with missing API keys."""
        # Configure mock with no API keys
        mock_client_manager.config.openai_api_key = None
        mock_client_manager.config.firecrawl_api_key = None

        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        validate_func = mock_mcp._tools["validate_configuration"]

        # Call the function
        result = await validate_func()

        # Verify results
        assert result["valid"] is True  # Still valid, just warnings
        assert len(result["warnings"]) == 2
        assert "OpenAI API key not configured" in result["warnings"]
        assert "Firecrawl API key not configured" in result["warnings"]
        assert "openai" not in result["config"]
        assert "firecrawl" not in result["config"]

    @pytest.mark.asyncio
    async def test_validate_configuration_with_redis(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test validate_configuration with Redis L2 cache enabled."""
        # Configure mock with Redis URL
        mock_client_manager.config.redis_url = "redis://localhost:6379"

        # Register tools
        utilities.register_tools(mock_mcp, mock_client_manager)

        # Get the registered function
        validate_func = mock_mcp._tools["validate_configuration"]

        # Call the function
        result = await validate_func()

        # Verify results
        assert result["config"]["cache"]["l2_enabled"] is True
