"""Tests for MCP deployment and alias management tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.enums import SearchStrategy
from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.models.responses import ABTestAnalysisResponse
from src.mcp_tools.models.responses import AliasesResponse
from src.mcp_tools.models.responses import CanaryStatusResponse
from src.mcp_tools.models.responses import OperationStatus
from src.mcp_tools.models.responses import SearchResult


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
    """Create a mock client manager."""
    manager = Mock()

    # Mock alias manager
    mock_alias_manager = Mock()
    mock_alias_manager.get_collection_for_alias = AsyncMock()
    mock_alias_manager.list_aliases = AsyncMock()
    mock_alias_manager.create_alias = AsyncMock()
    manager.get_alias_manager = AsyncMock(return_value=mock_alias_manager)

    # Mock blue-green deployment
    mock_blue_green = Mock()
    mock_blue_green.deploy_new_version = AsyncMock()
    manager.get_blue_green_deployment = AsyncMock(return_value=mock_blue_green)

    # Mock A/B testing
    mock_ab_testing = Mock()
    mock_ab_testing.create_experiment = AsyncMock()
    mock_ab_testing.analyze_experiment = AsyncMock()
    manager.get_ab_testing = AsyncMock(return_value=mock_ab_testing)

    # Mock canary deployment
    mock_canary = Mock()
    mock_canary.start_canary = AsyncMock()
    mock_canary.get_deployment_status = AsyncMock()
    mock_canary.pause_deployment = AsyncMock()
    mock_canary.resume_deployment = AsyncMock()
    manager.get_canary_deployment = AsyncMock(return_value=mock_canary)

    return manager


@pytest.mark.asyncio
async def test_deployment_tools_registration(mock_client_manager, mock_context):
    """Test that deployment tools are properly registered."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    expected_tools = [
        "search_with_alias", "list_aliases", "create_alias", "deploy_new_index",
        "start_ab_test", "analyze_ab_test", "start_canary_deployment",
        "get_canary_status", "pause_canary", "resume_canary"
    ]

    for tool in expected_tools:
        assert tool in registered_tools


@pytest.mark.asyncio
async def test_search_with_alias_success(mock_client_manager, mock_context):
    """Test successful search with alias."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.get_collection_for_alias.return_value = "docs_v2"

    mock_search_results = [
        SearchResult(
            id="doc1",
            content="Test content",
            score=0.9,
            url="https://example.com/doc1",
            title="Test Document",
            metadata={"type": "documentation"}
        )
    ]

    with patch('src.mcp_tools.tools._search_utils.search_documents_core', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_search_results

        register_tools(mock_mcp, mock_client_manager)

        # Test search_with_alias function
        result = await registered_tools["search_with_alias"](
            query="test query",
            alias="documentation",
            limit=5,
            strategy=SearchStrategy.HYBRID,
            enable_reranking=True,
            ctx=mock_context
        )

        assert len(result) == 1
        assert result[0].id == "doc1"
        assert result[0].content == "Test content"

        # Verify alias resolution
        alias_manager.get_collection_for_alias.assert_called_once_with("documentation")

        # Verify search was called with correct parameters
        mock_search.assert_called_once()
        search_request = mock_search.call_args[0][0]
        assert isinstance(search_request, SearchRequest)
        assert search_request.query == "test query"
        assert search_request.collection == "docs_v2"
        assert search_request.limit == 5
        assert search_request.strategy == SearchStrategy.HYBRID
        assert search_request.enable_reranking is True


@pytest.mark.asyncio
async def test_search_with_alias_not_found(mock_client_manager, mock_context):
    """Test search with alias when alias doesn't exist."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - alias not found
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.get_collection_for_alias.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test with non-existent alias
    with pytest.raises(ValueError, match="Alias missing_alias not found"):
        await registered_tools["search_with_alias"](
            query="test query",
            alias="missing_alias",
            ctx=mock_context
        )


@pytest.mark.asyncio
async def test_list_aliases_success(mock_client_manager, mock_context):
    """Test successful alias listing."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_aliases = {
        "documentation": "docs_v2",
        "api": "api_v1",
        "guides": "guides_latest"
    }

    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.list_aliases.return_value = mock_aliases

    register_tools(mock_mcp, mock_client_manager)

    # Test list_aliases function
    result = await registered_tools["list_aliases"](ctx=mock_context)

    assert isinstance(result, AliasesResponse)
    assert result.aliases == mock_aliases

    # Verify service calls
    alias_manager.list_aliases.assert_called_once()

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_list_aliases_without_context(mock_client_manager):
    """Test alias listing without context."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.list_aliases.return_value = {"docs": "docs_v1"}

    register_tools(mock_mcp, mock_client_manager)

    # Test without context
    result = await registered_tools["list_aliases"](ctx=None)

    assert isinstance(result, AliasesResponse)
    assert result.aliases == {"docs": "docs_v1"}


@pytest.mark.asyncio
async def test_create_alias_success(mock_client_manager, mock_context):
    """Test successful alias creation."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.create_alias.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test create_alias function
    result = await registered_tools["create_alias"](
        alias_name="new_docs",
        collection_name="docs_v3",
        force=True,
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "success"
    assert result.details["alias"] == "new_docs"
    assert result.details["collection"] == "docs_v3"

    # Verify service calls
    alias_manager.create_alias.assert_called_once_with(
        alias_name="new_docs",
        collection_name="docs_v3",
        force=True
    )

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_create_alias_failure(mock_client_manager, mock_context):
    """Test alias creation failure."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - creation fails
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.create_alias.return_value = False

    register_tools(mock_mcp, mock_client_manager)

    # Test create_alias function
    result = await registered_tools["create_alias"](
        alias_name="failed_alias",
        collection_name="docs_v1",
        force=False,
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "error"
    assert result.details["alias"] == "failed_alias"
    assert result.details["collection"] == "docs_v1"

    # Verify warning was logged
    mock_context.warning.assert_called()


@pytest.mark.asyncio
async def test_deploy_new_index_success(mock_client_manager, mock_context):
    """Test successful blue-green deployment."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_deployment_result = {
        "new_collection": "docs_v4",
        "old_collection": "docs_v3",
        "validation_passed": True,
        "deployment_time": "2023-12-01T10:00:00Z"
    }

    blue_green = await mock_client_manager.get_blue_green_deployment()
    blue_green.deploy_new_version.return_value = mock_deployment_result

    register_tools(mock_mcp, mock_client_manager)

    # Test deploy_new_index function
    result = await registered_tools["deploy_new_index"](
        alias="documentation",
        source="collection:docs_v4",
        validation_queries=["python", "javascript"],
        rollback_on_failure=True,
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "success"
    assert result.details == mock_deployment_result

    # Verify service calls
    blue_green.deploy_new_version.assert_called_once_with(
        alias_name="documentation",
        data_source="collection:docs_v4",
        validation_queries=["python", "javascript"],
        rollback_on_failure=True
    )

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_deploy_new_index_default_validation(mock_client_manager, mock_context):
    """Test deployment with default validation queries."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    blue_green = await mock_client_manager.get_blue_green_deployment()
    blue_green.deploy_new_version.return_value = {"new_collection": "docs_v5"}

    register_tools(mock_mcp, mock_client_manager)

    # Test without validation queries (should use defaults)
    result = await registered_tools["deploy_new_index"](
        alias="documentation",
        source="crawl:new",
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "success"

    # Verify default validation queries were used
    blue_green.deploy_new_version.assert_called_once()
    call_args = blue_green.deploy_new_version.call_args[1]
    assert call_args["validation_queries"] == [
        "python asyncio",
        "react hooks",
        "fastapi authentication"
    ]


@pytest.mark.asyncio
async def test_start_ab_test_success(mock_client_manager, mock_context):
    """Test successful A/B test creation."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    ab_testing = await mock_client_manager.get_ab_testing()
    ab_testing.create_experiment.return_value = "exp_123"

    register_tools(mock_mcp, mock_client_manager)

    # Test start_ab_test function
    result = await registered_tools["start_ab_test"](
        experiment_name="embedding_comparison",
        control_collection="docs_v1",
        treatment_collection="docs_v2",
        traffic_split=0.3,
        metrics=["relevance", "latency"],
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "started"
    assert result.details["experiment_id"] == "exp_123"
    assert result.details["control"] == "docs_v1"
    assert result.details["treatment"] == "docs_v2"
    assert result.details["traffic_split"] == 0.3

    # Verify service calls
    ab_testing.create_experiment.assert_called_once_with(
        experiment_name="embedding_comparison",
        control_collection="docs_v1",
        treatment_collection="docs_v2",
        traffic_split=0.3,
        metrics_to_track=["relevance", "latency"]
    )

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_analyze_ab_test_success(mock_client_manager, mock_context):
    """Test successful A/B test analysis."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_analysis = {
        "experiment_id": "exp_123",
        "concluded": True,
        "metrics": {
            "variant_a_conversion": 0.75,
            "variant_b_conversion": 0.82,
            "p_value": 0.003
        },
        "recommendation": "Deploy variant B"
    }

    ab_testing = await mock_client_manager.get_ab_testing()
    ab_testing.analyze_experiment.return_value = mock_analysis

    register_tools(mock_mcp, mock_client_manager)

    # Test analyze_ab_test function
    result = await registered_tools["analyze_ab_test"](
        experiment_id="exp_123",
        ctx=mock_context
    )

    assert isinstance(result, ABTestAnalysisResponse)
    # Verify the response contains the analysis data
    # (Note: We need to check the actual model structure)

    # Verify service calls
    ab_testing.analyze_experiment.assert_called_once_with("exp_123")

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_start_canary_deployment_success(mock_client_manager, mock_context):
    """Test successful canary deployment start."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    canary = await mock_client_manager.get_canary_deployment()
    canary.start_canary.return_value = "canary_456"

    register_tools(mock_mcp, mock_client_manager)

    # Test start_canary_deployment function
    custom_stages = [
        {"percentage": 10, "duration_minutes": 5},
        {"percentage": 50, "duration_minutes": 10},
        {"percentage": 100, "duration_minutes": 1}
    ]

    result = await registered_tools["start_canary_deployment"](
        alias="documentation",
        new_collection="docs_v5",
        stages=custom_stages,
        auto_rollback=True,
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "started"
    assert result.details["deployment_id"] == "canary_456"
    assert result.details["alias"] == "documentation"
    assert result.details["new_collection"] == "docs_v5"

    # Verify service calls
    canary.start_canary.assert_called_once_with(
        alias_name="documentation",
        new_collection="docs_v5",
        stages=custom_stages,
        auto_rollback=True
    )

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_start_canary_deployment_invalid_stages(mock_client_manager, mock_context):
    """Test canary deployment with invalid stage configuration."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test with invalid stages - missing percentage
    invalid_stages = [
        {"duration_minutes": 5}  # Missing percentage
    ]

    with pytest.raises(ValueError, match="Stage 0 missing required 'percentage' field"):
        await registered_tools["start_canary_deployment"](
            alias="documentation",
            new_collection="docs_v5",
            stages=invalid_stages,
            ctx=mock_context
        )


@pytest.mark.asyncio
async def test_start_canary_deployment_invalid_percentage(mock_client_manager, mock_context):
    """Test canary deployment with invalid percentage values."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test with invalid percentage > 100
    invalid_stages = [
        {"percentage": 150, "duration_minutes": 5}  # Invalid percentage
    ]

    with pytest.raises(ValueError, match="Stage 0 percentage must be between 0 and 100"):
        await registered_tools["start_canary_deployment"](
            alias="documentation",
            new_collection="docs_v5",
            stages=invalid_stages,
            ctx=mock_context
        )


@pytest.mark.asyncio
async def test_get_canary_status_success(mock_client_manager, mock_context):
    """Test successful canary status retrieval."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_status = {
        "deployment_id": "canary_456",
        "status": "running",
        "current_stage": 1,
        "current_percentage": 25,
        "health_metrics": {"error_rate": 0.02, "latency_p95": 120},
        "start_time": "2023-12-01T10:00:00Z"
    }

    canary = await mock_client_manager.get_canary_deployment()
    canary.get_deployment_status.return_value = mock_status

    register_tools(mock_mcp, mock_client_manager)

    # Test get_canary_status function
    result = await registered_tools["get_canary_status"](
        deployment_id="canary_456",
        ctx=mock_context
    )

    assert isinstance(result, CanaryStatusResponse)

    # Verify service calls
    canary.get_deployment_status.assert_called_once_with("canary_456")

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_pause_canary_success(mock_client_manager, mock_context):
    """Test successful canary deployment pause."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    canary = await mock_client_manager.get_canary_deployment()
    canary.pause_deployment.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test pause_canary function
    result = await registered_tools["pause_canary"](
        deployment_id="canary_456",
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "paused"
    assert result.details["deployment_id"] == "canary_456"

    # Verify service calls
    canary.pause_deployment.assert_called_once_with("canary_456")

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_pause_canary_failure(mock_client_manager, mock_context):
    """Test canary deployment pause failure."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - pause fails
    canary = await mock_client_manager.get_canary_deployment()
    canary.pause_deployment.return_value = False

    register_tools(mock_mcp, mock_client_manager)

    # Test pause_canary function
    result = await registered_tools["pause_canary"](
        deployment_id="canary_456",
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "failed"
    assert result.details["deployment_id"] == "canary_456"

    # Verify warning was logged
    mock_context.warning.assert_called()


@pytest.mark.asyncio
async def test_resume_canary_success(mock_client_manager, mock_context):
    """Test successful canary deployment resume."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    canary = await mock_client_manager.get_canary_deployment()
    canary.resume_deployment.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test resume_canary function
    result = await registered_tools["resume_canary"](
        deployment_id="canary_456",
        ctx=mock_context
    )

    assert isinstance(result, OperationStatus)
    assert result.status == "resumed"
    assert result.details["deployment_id"] == "canary_456"

    # Verify service calls
    canary.resume_deployment.assert_called_once_with("canary_456")

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_error_handling_list_aliases(mock_client_manager, mock_context):
    """Test error handling in list_aliases."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - service raises exception
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.list_aliases.side_effect = Exception("Service error")

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is properly handled and re-raised
    with pytest.raises(Exception, match="Service error"):
        await registered_tools["list_aliases"](ctx=mock_context)

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_error_handling_create_alias(mock_client_manager, mock_context):
    """Test error handling in create_alias."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - service raises exception
    alias_manager = await mock_client_manager.get_alias_manager()
    alias_manager.create_alias.side_effect = Exception("Alias creation failed")

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is properly handled and re-raised
    with pytest.raises(Exception, match="Alias creation failed"):
        await registered_tools["create_alias"](
            alias_name="test_alias",
            collection_name="test_collection",
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_error_handling_ab_test(mock_client_manager, mock_context):
    """Test error handling in A/B test operations."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - service raises exception
    ab_testing = await mock_client_manager.get_ab_testing()
    ab_testing.create_experiment.side_effect = Exception("AB test creation failed")

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is properly handled and re-raised
    with pytest.raises(Exception, match="AB test creation failed"):
        await registered_tools["start_ab_test"](
            experiment_name="test_experiment",
            control_collection="control",
            treatment_collection="treatment",
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_error_handling_canary(mock_client_manager, mock_context):
    """Test error handling in canary deployment operations."""
    from src.mcp_tools.tools.deployment import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - service raises exception
    canary = await mock_client_manager.get_canary_deployment()
    canary.get_deployment_status.side_effect = Exception("Status retrieval failed")

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is properly handled and re-raised
    with pytest.raises(Exception, match="Status retrieval failed"):
        await registered_tools["get_canary_status"](
            deployment_id="canary_123",
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()
