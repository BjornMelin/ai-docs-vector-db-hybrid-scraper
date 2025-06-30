"""Conftest for multi-agent integration tests.

Provides fixtures for testing multi-agent coordination, result fusion,
and performance optimization scenarios without making real API calls.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents.agentic_orchestrator import AgenticOrchestrator, ToolResponse
from src.services.agents.core import AgentState, BaseAgentDependencies
from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    ToolCapabilityType,
)
from src.services.agents.query_orchestrator import QueryOrchestrator


@pytest.fixture
def mock_client_manager() -> Mock:
    """Enhanced mock client manager for multi-agent testing."""
    client_manager = Mock(spec=ClientManager)

    # Mock all the client manager methods that agents use
    client_manager.get_openai_client = AsyncMock()
    client_manager.get_qdrant_client = AsyncMock()
    client_manager.get_search_orchestrator = AsyncMock()
    client_manager.get_embedding_manager = AsyncMock()
    client_manager.get_vector_db_service = AsyncMock()

    # Mock health check
    client_manager.health_check = AsyncMock(return_value=True)

    return client_manager


@pytest.fixture
def agent_dependencies(mock_client_manager) -> BaseAgentDependencies:
    """Create agent dependencies with mocked client manager."""
    config = get_config()
    session_state = AgentState(session_id=str(uuid4()))

    return BaseAgentDependencies(
        client_manager=mock_client_manager, config=config, session_state=session_state
    )


@pytest.fixture
def mock_pydantic_ai_agent():
    """Mock Pydantic-AI agent to prevent real API calls."""
    agent = Mock()

    # Mock the run method to return realistic responses
    async def mock_run(prompt, **kwargs):
        """Mock agent run method."""
        response = Mock()
        response.data = {
            "success": True,
            "analysis": {
                "complexity": "moderate",
                "strategy": "balanced",
                "recommended_tools": ["hybrid_search"],
                "confidence": 0.85,
            },
            "result": f"Processed: {prompt[:50]}...",
            "performance_metrics": {
                "latency_ms": 150.0,
                "tokens_used": 50,
                "cost_estimate": 0.001,
            },
        }
        response.usage = Mock()
        response.usage.total_tokens = 50
        return response

    agent.run = AsyncMock(side_effect=mock_run)

    # Mock tool registration
    def mock_tool(func):
        """Mock tool decorator."""
        return func

    agent.tool = mock_tool

    return agent


@pytest.fixture
def mock_agentic_orchestrator(agent_dependencies, mock_pydantic_ai_agent):
    """Create a mock AgenticOrchestrator with proper initialization."""
    with patch("src.services.agents.core.Agent") as mock_agent_class:
        mock_agent_class.return_value = mock_pydantic_ai_agent

        orchestrator = AgenticOrchestrator()
        orchestrator._initialized = True
        orchestrator.agent = mock_pydantic_ai_agent

        # Mock the orchestrate method
        async def mock_orchestrate(request, context, dependencies):
            """Mock orchestrate method."""
            return ToolResponse(
                success=True,
                result={
                    "orchestration_id": str(uuid4()),
                    "results": {
                        "query_analysis": {
                            "complexity": "moderate",
                            "strategy": "balanced",
                        },
                        "tool_results": [
                            {
                                "tool": "hybrid_search",
                                "success": True,
                                "result": "Mock search results",
                            }
                        ],
                    },
                    "performance": {
                        "total_time_ms": 200.0,
                        "api_calls": 2,
                        "tokens_used": 75,
                    },
                },
                tools_used=["hybrid_search", "content_classification"],
                confidence=0.85,
                latency_ms=200.0,
                reasoning="Mock orchestration completed successfully",
            )

        orchestrator.orchestrate = AsyncMock(side_effect=mock_orchestrate)

        return orchestrator


@pytest.fixture
def mock_dynamic_tool_discovery(agent_dependencies):
    """Create a mock DynamicToolDiscovery with proper initialization."""
    discovery = DynamicToolDiscovery()
    discovery._initialized = True

    # Mock the discover_tools_for_task method
    async def mock_discover_tools(task_description, context=None):
        """Mock tool discovery method."""
        current_time = datetime.now(tz=timezone.utc).isoformat()

        return [
            ToolCapability(
                name="hybrid_search",
                capability_type=ToolCapabilityType.SEARCH,
                description="Hybrid vector and keyword search",
                input_types=["text", "query"],
                output_types=["documents", "results"],
                confidence_score=0.9,
                last_updated=current_time,
                requirements={},
                constraints={},
                compatible_tools=[],
                dependencies=[],
            ),
            ToolCapability(
                name="content_classification",
                capability_type=ToolCapabilityType.ANALYSIS,
                description="Content classification and categorization",
                input_types=["text", "content"],
                output_types=["categories", "classification"],
                confidence_score=0.7,
                last_updated=current_time,
                requirements={},
                constraints={},
                compatible_tools=[],
                dependencies=[],
            ),
        ]

    discovery.discover_tools_for_task = AsyncMock(side_effect=mock_discover_tools)

    # Mock performance tracking
    async def mock_update_performance(tool_name, metrics):
        """Mock performance update method."""
        return True

    discovery.update_tool_performance = AsyncMock(side_effect=mock_update_performance)

    return discovery


@pytest.fixture
def mock_query_orchestrator(agent_dependencies, mock_pydantic_ai_agent):
    """Create a mock QueryOrchestrator with proper initialization."""
    with patch("src.services.agents.core.Agent") as mock_agent_class:
        mock_agent_class.return_value = mock_pydantic_ai_agent

        orchestrator = QueryOrchestrator()
        orchestrator._initialized = True
        orchestrator.agent = mock_pydantic_ai_agent

        # Mock the orchestrate_query method
        async def mock_orchestrate_query(query, collection="documentation", **kwargs):
            """Mock query orchestration method."""
            return {
                "success": True,
                "orchestration_id": str(uuid4()),
                "result": {
                    "query": query,
                    "collection": collection,
                    "analysis": {
                        "complexity": "moderate",
                        "domain": "technical",
                        "strategy": "balanced",
                        "confidence": 0.8,
                    },
                    "processing_plan": {
                        "steps": [
                            "analyze_query_intent",
                            "delegate_to_specialist",
                            "coordinate_multi_stage_search",
                        ],
                        "estimated_time_ms": 500.0,
                    },
                    "results": {
                        "documents_found": 5,
                        "relevance_scores": [0.95, 0.88, 0.82, 0.78, 0.75],
                        "response_time_ms": 450.0,
                    },
                },
            }

        orchestrator.orchestrate_query = AsyncMock(side_effect=mock_orchestrate_query)

        return orchestrator


@pytest.fixture
def multi_agent_system(
    mock_agentic_orchestrator,
    mock_dynamic_tool_discovery,
    mock_query_orchestrator,
    agent_dependencies,
):
    """Create a complete multi-agent system for testing."""
    # Create additional orchestrator agents for parallel testing
    orchestrator_2 = AgenticOrchestrator()
    orchestrator_2._initialized = True
    orchestrator_2.agent = Mock()

    # Mock the orchestrate method for orchestrator_2
    async def mock_orchestrate_2(request, context, dependencies):
        """Mock orchestrate method for orchestrator_2."""
        return ToolResponse(
            success=True,
            result={"orchestration_id": str(uuid4()), "agent": "orchestrator_2"},
            tools_used=["hybrid_search"],
            confidence=0.8,
            latency_ms=180.0,
            reasoning="Mock orchestration from agent 2",
        )

    orchestrator_2.orchestrate = AsyncMock(side_effect=mock_orchestrate_2)

    # Create orchestrator_3
    orchestrator_3 = AgenticOrchestrator()
    orchestrator_3._initialized = True
    orchestrator_3.agent = Mock()

    # Mock the orchestrate method for orchestrator_3
    async def mock_orchestrate_3(request, context, dependencies):
        """Mock orchestrate method for orchestrator_3."""
        return ToolResponse(
            success=True,
            result={"orchestration_id": str(uuid4()), "agent": "orchestrator_3"},
            tools_used=["content_analysis"],
            confidence=0.75,
            latency_ms=220.0,
            reasoning="Mock orchestration from agent 3",
        )

    orchestrator_3.orchestrate = AsyncMock(side_effect=mock_orchestrate_3)

    return {
        "orchestrator": mock_agentic_orchestrator,
        "tool_discovery": mock_dynamic_tool_discovery,
        "query_orchestrator": mock_query_orchestrator,
        "dependencies": agent_dependencies,
        "agent_pool": {
            "orchestrator_1": mock_agentic_orchestrator,
            "orchestrator_2": orchestrator_2,
            "orchestrator_3": orchestrator_3,
            "tool_discovery": mock_dynamic_tool_discovery,
            "query_orchestrator": mock_query_orchestrator,
        },
    }


@pytest.fixture
def performance_test_data():
    """Generate test data for performance optimization scenarios."""
    return {
        "baseline_queries": [
            "What is the purpose of this documentation?",
            "How do I configure the API settings?",
            "What are the authentication requirements?",
            "How do I troubleshoot connection issues?",
            "What are the rate limiting policies?",
        ],
        "complex_queries": [
            "Analyze the performance characteristics of the hybrid search implementation and recommend optimizations",
            "Compare the effectiveness of different embedding models for technical documentation retrieval",
            "Evaluate the security implications of the authentication system and suggest improvements",
            "Design a comprehensive monitoring strategy for the vector database operations",
            "Create a migration plan for upgrading the search infrastructure while maintaining availability",
        ],
        "expected_improvements": {
            "latency_reduction": 0.6,  # 60% reduction
            "throughput_increase": 3.0,  # 3x improvement
            "accuracy_improvement": 0.25,  # 25% improvement
            "cost_reduction": 0.4,  # 40% reduction
        },
    }


@pytest.fixture
def coordination_test_scenarios():
    """Test scenarios for coordination pattern testing."""
    return [
        {
            "scenario_id": "hierarchical_simple",
            "description": "Simple hierarchical coordination with 3 agents",
            "agent_count": 3,
            "coordination_pattern": "hierarchical",
            "expected_handoffs": 2,
            "expected_latency_ms": 300.0,
        },
        {
            "scenario_id": "parallel_balanced",
            "description": "Balanced parallel processing with 4 agents",
            "agent_count": 4,
            "coordination_pattern": "parallel",
            "expected_handoffs": 0,
            "expected_latency_ms": 200.0,
        },
        {
            "scenario_id": "hybrid_complex",
            "description": "Complex hybrid coordination with 6 agents",
            "agent_count": 6,
            "coordination_pattern": "hybrid",
            "expected_handoffs": 3,
            "expected_latency_ms": 400.0,
        },
    ]


@pytest.fixture
def fusion_test_data():
    """Test data for result fusion algorithm testing."""
    return {
        "sample_results": [
            {
                "agent_id": "agent_1",
                "confidence": 0.9,
                "results": {
                    "relevance_score": 0.85,
                    "category": "technical",
                    "sentiment": "neutral",
                },
                "performance": {"latency_ms": 150.0, "accuracy": 0.88},
            },
            {
                "agent_id": "agent_2",
                "confidence": 0.7,
                "results": {
                    "relevance_score": 0.78,
                    "category": "technical",
                    "sentiment": "positive",
                },
                "performance": {"latency_ms": 200.0, "accuracy": 0.82},
            },
            {
                "agent_id": "agent_3",
                "confidence": 0.8,
                "results": {
                    "relevance_score": 0.92,
                    "category": "business",
                    "sentiment": "neutral",
                },
                "performance": {"latency_ms": 120.0, "accuracy": 0.90},
            },
        ],
        "expected_fusion_results": {
            "confidence_weighted": {
                "relevance_score": 0.86,  # Weighted average
                "category": "technical",  # Majority
                "sentiment": "neutral",  # Highest confidence
            },
            "performance_weighted": {
                "relevance_score": 0.89,  # Performance-weighted
                "category": "business",  # Best performing agent
                "sentiment": "neutral",
            },
        },
    }
