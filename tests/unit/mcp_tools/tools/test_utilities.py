"""Tests for MCP utility tools."""

from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock

import pytest

from src.mcp_tools.models.responses import ConfigValidationResponse, GenericDictResponse


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    context = Mock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


@pytest.fixture
def mock_client_manager():
    """Create a mock client manager with unified config."""
    manager = Mock()

    # Mock unified config structure
    mock_config = Mock()

    # Mock Qdrant config
    mock_qdrant = Mock()
    mock_qdrant.url = "http://localhost:6333"
    mock_config.qdrant = mock_qdrant

    # Mock OpenAI config
    mock_openai = Mock()
    mock_openai.api_key = "sk-test-key"
    mock_config.openai = mock_openai

    # Mock Firecrawl config
    mock_firecrawl = Mock()
    mock_firecrawl.api_key = "fc-test-key"
    mock_config.firecrawl = mock_firecrawl

    # Mock Cache config
    mock_cache = Mock()
    mock_cache.max_items = 1000
    mock_cache.redis_url = "redis://localhost:6379"
    mock_config.cache = mock_cache

    manager.unified_config = mock_config

    return manager


@pytest.mark.asyncio
async def test_utility_tools_registration(mock_client_manager, _mock_context):
    """Test that utility tools are properly registered."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    assert "estimate_costs" in registered_tools
    assert "validate_configuration" in registered_tools


@pytest.mark.asyncio
async def test_estimate_costs_basic(mock_client_manager, mock_context):
    """Test basic cost estimation without storage."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test basic cost estimation
    result = await registered_tools["estimate_costs"](
        text_count=100, average_length=500, include_storage=False, ctx=mock_context
    )

    assert isinstance(result, GenericDictResponse)
    assert result.text_count == 100
    assert result.estimated_tokens == 12500  # 100 * 500 / 4
    assert result.embedding_cost == 0.0003  # 12500 * 0.00002 / 1000
    assert result.provider == "openai/text-embedding-3-small"
    assert not hasattr(result, "storage_gb")
    assert not hasattr(result, "storage_cost_monthly")
    assert not hasattr(result, "total_cost")

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_estimate_costs_with_storage(mock_client_manager, mock_context):
    """Test cost estimation including storage costs."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test cost estimation with storage
    result = await registered_tools["estimate_costs"](
        text_count=1000, average_length=1000, include_storage=True, ctx=mock_context
    )

    assert isinstance(result, GenericDictResponse)
    assert result.text_count == 1000
    assert result.estimated_tokens == 250000  # 1000 * 1000 / 4
    assert result.embedding_cost == 0.005  # 250000 * 0.00002 / 1000

    # Storage calculations: 1000 * 1536 * 4 bytes = 6,144,000 bytes = 0.006144 GB
    expected_storage_gb = 0.0061  # Rounded to 4 decimal places
    expected_storage_cost = expected_storage_gb * 0.20  # $0.20 per GB/month
    expected_total = 0.005 + expected_storage_cost

    assert result.storage_gb == expected_storage_gb
    assert result.storage_cost_monthly == round(expected_storage_cost, 4)
    assert result.total_cost == round(expected_total, 4)

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_estimate_costs_default_parameters(mock_client_manager, mock_context):
    """Test cost estimation with default parameters."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test with default parameters
    result = await registered_tools["estimate_costs"](text_count=50, ctx=mock_context)

    assert isinstance(result, GenericDictResponse)
    assert result.text_count == 50
    # Default average_length=1000, so: 50 * 1000 / 4 = 12500 tokens
    assert result.estimated_tokens == 12500
    # Default include_storage=True, so storage costs should be included
    assert hasattr(result, "storage_gb")
    assert hasattr(result, "storage_cost_monthly")
    assert hasattr(result, "total_cost")


@pytest.mark.asyncio
async def test_estimate_costs_without_context(mock_client_manager):
    """Test cost estimation without context parameter."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test without ctx parameter (None)
    result = await registered_tools["estimate_costs"](
        text_count=10, average_length=200, include_storage=False, ctx=None
    )

    assert isinstance(result, GenericDictResponse)
    assert result.text_count == 10
    assert result.estimated_tokens == 500  # 10 * 200 / 4


@pytest.mark.asyncio
async def test_estimate_costs_error_handling(mock_client_manager, mock_context):
    """Test error handling in cost estimation."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test with invalid input that causes error
    with pytest.raises(TypeError):
        await registered_tools["estimate_costs"](
            text_count="invalid",  # Should be int
            ctx=mock_context,
        )

    # Verify error logging would be called
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_validate_configuration_success(mock_client_manager, mock_context):
    """Test successful configuration validation."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test configuration validation
    result = await registered_tools["validate_configuration"](ctx=mock_context)

    assert isinstance(result, ConfigValidationResponse)
    assert result.status == "success"
    assert result.errors is None
    assert result.details["valid"] is True
    assert len(result.details["errors"]) == 0
    assert len(result.details["warnings"]) == 0

    # Verify configuration details
    config = result.details["config"]
    assert config["qdrant_url"] == "http://localhost:6333"
    assert config["openai"] == "configured"
    assert config["firecrawl"] == "configured"
    assert config["cache"]["l1_enabled"] is True
    assert config["cache"]["l1_max_items"] == 1000
    assert config["cache"]["l2_enabled"] is True

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_validate_configuration_missing_api_keys(
    mock_client_manager, mock_context
):
    """Test configuration validation with missing API keys."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Configure missing API keys
    mock_client_manager.unified_config.openai.api_key = None
    mock_client_manager.unified_config.firecrawl.api_key = ""

    register_tools(mock_mcp, mock_client_manager)

    # Test configuration validation
    result = await registered_tools["validate_configuration"](ctx=mock_context)

    assert isinstance(result, ConfigValidationResponse)
    assert result.status == "success"  # Still success, just with warnings
    assert result.details["valid"] is True  # No errors, just warnings
    assert len(result.details["errors"]) == 0
    assert len(result.details["warnings"]) == 2

    # Verify warnings
    warnings = result.details["warnings"]
    assert "OpenAI API key not configured" in warnings
    assert "Firecrawl API key not configured" in warnings

    # Verify configuration details
    config = result.details["config"]
    assert "openai" not in config
    assert "firecrawl" not in config

    # Verify warning logging
    mock_context.warning.assert_called()


@pytest.mark.asyncio
async def test_validate_configuration_no_redis(mock_client_manager, mock_context):
    """Test configuration validation without Redis cache."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Configure without Redis
    mock_client_manager.unified_config.cache.redis_url = None

    register_tools(mock_mcp, mock_client_manager)

    # Test configuration validation
    result = await registered_tools["validate_configuration"](ctx=mock_context)

    assert isinstance(result, ConfigValidationResponse)
    assert result.status == "success"

    # Verify cache configuration
    cache_config = result.details["config"]["cache"]
    assert cache_config["l1_enabled"] is True
    assert cache_config["l2_enabled"] is False  # No Redis URL


@pytest.mark.asyncio
async def test_validate_configuration_without_context(mock_client_manager):
    """Test configuration validation without context parameter."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test without ctx parameter (None)
    result = await registered_tools["validate_configuration"](ctx=None)

    assert isinstance(result, ConfigValidationResponse)
    assert result.status == "success"
    assert result.details["valid"] is True


@pytest.mark.asyncio
async def test_validate_configuration_error_handling(mock_client_manager, mock_context):
    """Test error handling in configuration validation."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock client_manager to raise exception when accessing config
    type(mock_client_manager.unified_config.qdrant).url = PropertyMock(
        side_effect=Exception("Config access error")
    )

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is properly handled and re-raised
    with pytest.raises(Exception, match="Config access error"):
        await registered_tools["validate_configuration"](ctx=mock_context)

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_estimate_costs_large_numbers(mock_client_manager, mock_context):
    """Test cost estimation with large numbers."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test with large numbers
    result = await registered_tools["estimate_costs"](
        text_count=1000000,  # 1 million texts
        average_length=2000,
        include_storage=True,
        ctx=mock_context,
    )

    assert isinstance(result, GenericDictResponse)
    assert result.text_count == 1000000
    assert result.estimated_tokens == 500000000  # 1M * 2000 / 4
    assert result.embedding_cost == 10.0  # 500M * 0.00002 / 1000

    # Storage: 1M * 1536 * 4 bytes = 6.144 GB
    assert result.storage_gb == 6.144
    assert result.storage_cost_monthly == 1.2288  # 6.144 * 0.20
    assert result.total_cost == 11.2288  # 10.0 + 1.2288


@pytest.mark.asyncio
async def test_validate_configuration_partial_config(mock_client_manager, mock_context):
    """Test configuration validation with partially configured system."""
    from src.mcp_tools.tools.utilities import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Configure only some services
    mock_client_manager.unified_config.openai.api_key = "sk-test-key"
    mock_client_manager.unified_config.firecrawl.api_key = None
    mock_client_manager.unified_config.cache.redis_url = None

    register_tools(mock_mcp, mock_client_manager)

    # Test configuration validation
    result = await registered_tools["validate_configuration"](ctx=mock_context)

    assert isinstance(result, ConfigValidationResponse)
    assert result.status == "success"
    assert len(result.details["warnings"]) == 1
    assert "Firecrawl API key not configured" in result.details["warnings"]

    # Verify partial configuration
    config = result.details["config"]
    assert config["openai"] == "configured"
    assert "firecrawl" not in config
    assert config["cache"]["l2_enabled"] is False
