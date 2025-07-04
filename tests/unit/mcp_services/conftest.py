"""Shared fixtures for MCP services testing.

Provides common test fixtures for testing FastMCP 2.0+ modular services
with proper mocking and integration test patterns.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_client_manager():
    """Create a mock ClientManager for testing services."""
    mock_manager = Mock()
    mock_manager.initialize = AsyncMock()

    # Mock client providers
    mock_manager.get_openai_client = AsyncMock()
    mock_manager.get_qdrant_client = AsyncMock()
    mock_manager.get_redis_client = AsyncMock()
    mock_manager.get_firecrawl_client = AsyncMock()
    mock_manager.get_http_client = AsyncMock()

    # Mock parallel processing system
    mock_manager.parallel_processing_system = Mock()

    return mock_manager


@pytest.fixture
def mock_fastmcp_instance():
    """Create a mock FastMCP instance for testing."""
    mock_mcp = Mock()
    mock_mcp.tool = Mock()
    mock_mcp.name = "test-service"
    mock_mcp.instructions = "Test service instructions"

    # Mock tool decorator
    def mock_tool_decorator(func=None, name=None, description=None):
        def decorator(f):
            # Store tool registration for verification
            if not hasattr(mock_mcp, "_registered_tools"):
                mock_mcp._registered_tools = []
            mock_mcp._registered_tools.append(
                {"function": f, "name": name or f.__name__, "description": description}
            )
            return f

        return decorator(func) if func else decorator

    mock_mcp.tool.side_effect = mock_tool_decorator

    return mock_mcp


@pytest.fixture
def mock_mcp_tools():
    """Create mock MCP tools modules."""
    tools = {}

    # Mock tool modules with register_tools function
    tool_modules = [
        "hybrid_search",
        "hyde_search",
        "multi_stage_search",
        "search_with_reranking",
        "web_search",
        "collections",
        "content_intelligence",
        "crawling",
        "document_management",
        "projects",
        "agentic_rag",
        "analytics",
        "query_processing",
        "configuration",
        "cost_estimation",
        "embeddings",
        "filtering",
        "system_health",
    ]

    for tool_name in tool_modules:
        mock_tool = Mock()
        mock_tool.register_tools = Mock()
        tools[tool_name] = mock_tool

    return tools


@pytest.fixture
def mock_observability_components():
    """Create mock observability components for analytics service testing."""
    components = {
        "ai_tracker": Mock(),
        "correlation_manager": Mock(),
        "performance_monitor": Mock(),
    }

    # Configure AI tracker
    components["ai_tracker"].track_operation = AsyncMock()
    components["ai_tracker"].get_metrics = AsyncMock(
        return_value={
            "total_operations": 150,
            "avg_latency_ms": 245.0,
            "success_rate": 0.95,
            "cost_per_operation": 0.003,
        }
    )

    # Configure correlation manager
    components["correlation_manager"].create_correlation_id = Mock(
        return_value="test-correlation-123"
    )
    components["correlation_manager"].get_trace_data = AsyncMock(
        return_value={
            "traces": [
                {"span_id": "span1", "operation": "search", "duration_ms": 120},
                {"span_id": "span2", "operation": "embed", "duration_ms": 80},
            ]
        }
    )

    # Configure performance monitor
    components["performance_monitor"].get_metrics = AsyncMock(
        return_value={
            "cpu_usage": 0.65,
            "memory_usage": 0.78,
            "request_rate": 45.2,
            "error_rate": 0.02,
        }
    )

    return components


@pytest.fixture
def mock_agentic_orchestrator():
    """Create mock AgenticOrchestrator for testing."""
    mock_orchestrator = Mock()
    mock_orchestrator.initialize = AsyncMock()
    mock_orchestrator.orchestrate = AsyncMock()

    # Configure mock orchestration result
    mock_result = Mock()
    mock_result.success = True
    mock_result.results = {"status": "completed", "data": {"items": 3}}
    mock_result.tools_used = ["search_tool", "analytics_tool"]
    mock_result.reasoning = "Used search and analytics tools to complete workflow"
    mock_result.latency_ms = 450.0
    mock_result.confidence = 0.89

    mock_orchestrator.orchestrate.return_value = mock_result

    return mock_orchestrator


@pytest.fixture
def mock_discovery_engine():
    """Create mock discovery engine for testing."""
    mock_engine = Mock()
    mock_engine.initialize_discovery = AsyncMock()
    mock_engine.get_tool_recommendations = AsyncMock(
        return_value=[
            {
                "tool": "hybrid_search",
                "confidence": 0.92,
                "reason": "Best for complex queries",
            },
            {
                "tool": "analytics_tool",
                "confidence": 0.85,
                "reason": "Good for metrics collection",
            },
        ]
    )

    return mock_engine


@pytest.fixture
def sample_service_capabilities():
    """Sample service capabilities data for testing."""
    return {
        "search": {
            "service": "search",
            "version": "2.0",
            "capabilities": [
                "hybrid_search",
                "hyde_search",
                "multi_stage_search",
                "search_reranking",
                "autonomous_web_search",
            ],
            "autonomous_features": [
                "provider_optimization",
                "strategy_adaptation",
                "quality_assessment",
            ],
            "status": "active",
        },
        "document": {
            "service": "document",
            "version": "2.0",
            "capabilities": [
                "document_management",
                "intelligent_crawling",
                "5_tier_crawling",
                "collection_management",
            ],
            "autonomous_features": [
                "tier_selection_optimization",
                "content_quality_assessment",
            ],
            "status": "active",
        },
    }


@pytest.fixture
def sample_workflow_description():
    """Sample workflow description for testing orchestration."""
    return """
    Process a complex research query:
    1. Search for relevant documents using hybrid search
    2. Analyze document quality and relevance
    3. Extract key insights and metrics
    4. Generate comprehensive analysis report
    """


@pytest.fixture
def sample_performance_constraints():
    """Sample performance constraints for testing."""
    return {
        "max_latency_ms": 2000,
        "min_confidence": 0.8,
        "max_cost": 0.05,
        "preferred_providers": ["openai", "qdrant"],
    }


@pytest.fixture
async def initialized_mock_services(
    mock_client_manager,
    mock_mcp_tools,
    mock_observability_components,
    mock_agentic_orchestrator,
    mock_discovery_engine,
):
    """Create initialized mock services for complex integration testing."""
    services = {}

    # Mock service initialization
    for service_name in ["search", "document", "analytics", "system"]:
        service_mock = Mock()
        service_mock.initialize = AsyncMock()
        service_mock.get_service_info = AsyncMock(
            return_value={
                "service": service_name,
                "status": "active",
                "capabilities": [f"{service_name}_capability"],
            }
        )
        service_mock.get_mcp_server = Mock()
        services[service_name] = service_mock

    return services


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
